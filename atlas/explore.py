r"""Export the corpus as a browsable snapshot for Embedding Atlas and Datasette.

Datasette reads SQLite only, and the corpus lives in Postgres, so both browser
tools read files that this module writes from the atlas cache. Nothing goes back
to the database: the snapshot is a cache, rebuilt on demand.

    poetry run python -m atlas.refresh --rebuild   # when the cache is stale
    poetry run python -m atlas.explore             # writes data/atlas/explore.*

    uvx embedding-atlas@0.26.0 data/atlas/explore.parquet \
        --text chunk_text --x x --y y --neighbors neighbors \
        --labels data/atlas/explore_labels.parquet
    uvx --with datasette-plot==0.1.0 datasette@1.0a43 data/atlas/explore.db \
        -m atlas/explore_metadata.yaml -c atlas/explore_datasette.yaml

Two geometric choices, both recorded in explore.json: the map is a UMAP of the
centred vectors, for looking only, and the neighbours are exact top-k by raw
cosine, the similarity retrieval ranks by.

Like the rest of atlas/, this builds dense n-by-n matrices. That fits to roughly
20k chunks; past that, neighbours need a blockwise or approximate search.

Design: docs/plans/2026-11-19-corpus-explorer-design.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from atlas.geometry import centre, hubness
from atlas.loader import CACHE_DIR, check_drift, fingerprint, load
from atlas.pericopes import families
from atlas.structure import exemplars, project

logger = logging.getLogger(__name__)

META_COLUMNS = [
    "uuid",
    "sutta_uid",
    "nikaya",
    "doc_title",
    "chunk_index",
    "word_count",
    "chunk_text",
]
FAMILY_THRESHOLD = 0.90
UMAP_NEIGHBOURS = 15
SEED = 42

LAUNCH = """\
Open the map:
    uvx embedding-atlas@0.26.0 {out}/explore.parquet --text chunk_text --x x --y y \\
        --neighbors neighbors --labels {out}/explore_labels.parquet
Open the tables:
    uvx --with datasette-plot==0.1.0 datasette@1.0a43 {out}/explore.db \\
        -m atlas/explore_metadata.yaml -c atlas/explore_datasette.yaml"""

GOLD_PATH = Path("data/labels/genre_gold_v2.jsonl")
GOLD_RENAME = {"mode": "gold_mode", "is_verse": "gold_is_verse", "has_simile": "gold_has_simile"}
GOLD_COLUMNS = list(GOLD_RENAME.values())
PRED_COLUMNS = ["pred_mode", "pred_confidence"]

K = 10
LABEL_TOP = 40
LABEL_MIN_SIZE = 3
LABEL_WORDS = 8

# A markdown heading ("# With Ghaṭīkāra") or breadcrumb ("* Middle Discourses 81").
MARKUP_LINE = re.compile(r"^\s*(#+|\*)\s")

# Datasette shows the keys as links between tables, and finds chunks_fts through
# its content=[chunks] option (the sqlite-utils convention) to offer search.
SCHEMA = """
CREATE TABLE suttas (
    sutta_uid TEXT PRIMARY KEY,
    doc_title TEXT,
    nikaya TEXT,
    n_chunks INTEGER
);
CREATE TABLE chunks (
    uuid TEXT PRIMARY KEY,
    sutta_uid TEXT REFERENCES suttas(sutta_uid),
    nikaya TEXT,
    chunk_index INTEGER,
    word_count INTEGER,
    chunk_text TEXT,
    x REAL,
    y REAL,
    hub_count INTEGER,
    neighbour_mix REAL,
    family_id INTEGER,
    family_size INTEGER,
    gold_mode TEXT,
    gold_is_verse INTEGER,
    gold_has_simile INTEGER,
    pred_mode TEXT,
    pred_confidence REAL
);
CREATE TABLE neighbours (
    uuid TEXT REFERENCES chunks(uuid),
    rank INTEGER,
    neighbour_uuid TEXT REFERENCES chunks(uuid),
    cosine REAL,
    PRIMARY KEY (uuid, rank)
);
CREATE INDEX chunks_by_sutta ON chunks(sutta_uid);
CREATE INDEX chunks_by_family ON chunks(family_id);
CREATE INDEX neighbours_by_neighbour ON neighbours(neighbour_uuid);
CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_text, content=[chunks]);
"""
CHUNK_COLUMNS = [
    "uuid",
    "sutta_uid",
    "nikaya",
    "chunk_index",
    "word_count",
    "chunk_text",
    "x",
    "y",
    "hub_count",
    "neighbour_mix",
    "family_id",
    "family_size",
    *GOLD_COLUMNS,
    *PRED_COLUMNS,
]


def neighbours(vectors: np.ndarray, k: int = K) -> tuple[np.ndarray, np.ndarray]:
    """Each row's k nearest rows by cosine, nearest first, excluding itself.

    Raw cosine is the similarity retrieval ranks by, so these are the chunks
    retrieval would return for a query sitting on this one. On unit-norm
    vectors cosine is a plain dot product.
    """
    k = min(k, len(vectors) - 1)
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, -np.inf)
    top = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(similarity, top, axis=1), axis=1, kind="stable")
    ids = np.take_along_axis(top, order, axis=1)
    return ids, np.take_along_axis(similarity, ids, axis=1)


def neighbour_mix(ids: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Share of each row's neighbours that come from its own sutta.

    Near 1, a chunk's closest company is its own discourse. Near 0, it shares
    its wording with other suttas, the way stock formulas do.
    """
    sutta = df["sutta_uid"].to_numpy()
    return np.asarray((sutta[ids] == sutta[:, None]).mean(axis=1))


def number_families(components: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Renumber near-duplicate components so that family 1 is the largest.

    Returns a family id per row, 0 for a chunk with no near-duplicate, and the
    size of each row's component, counting the row itself.
    """
    sizes = np.bincount(components)
    real = np.flatnonzero(sizes > 1)
    ranked = real[np.argsort(-sizes[real], kind="stable")]
    renumbered = np.zeros(len(sizes), dtype=np.int64)
    renumbered[ranked] = np.arange(1, len(ranked) + 1)
    return renumbered[components], sizes[components]


def opening(text: str, words: int = LABEL_WORDS) -> str:
    """The first words of a passage, skipping markdown headings and breadcrumbs."""
    body = [line for line in text.splitlines() if not MARKUP_LINE.match(line)]
    tokens = " ".join(body).split() or text.split()
    return " ".join(tokens[:words]) + ("…" if len(tokens) > words else "")


def family_labels(
    table: pd.DataFrame,
    vectors: np.ndarray,
    top: int = LABEL_TOP,
    min_size: int = LABEL_MIN_SIZE,
) -> pd.DataFrame:
    """Map labels for the largest pericope families, in Embedding Atlas's shape.

    Each label sits at the median position of its members and names the family
    by its size and the opening words of its exemplar, the member nearest the
    family centroid. Embedding Atlas shows higher priority labels first.

    The name goes in both `text` and `content`: Embedding Atlas 0.26.0 documents
    `text` for --labels, but its frontend renders `content` and crashes on a
    label that lacks it.
    """
    ids = table["family_id"].fillna(0).to_numpy(dtype=np.int64)
    ids = np.where(table["family_size"].to_numpy() >= min_size, ids, 0)
    chosen = np.unique(ids[ids > 0])[:top]  # family ids rank by size, 1 = largest
    members_of = np.where(np.isin(ids, chosen), ids, -1)
    exemplar = exemplars(vectors, members_of)

    x, y = table["x"].to_numpy(), table["y"].to_numpy()
    text = table["chunk_text"].to_numpy()
    rows = []
    for family in chosen.tolist():
        members = members_of == family
        size = int(members.sum())
        name = f"{size}× {opening(text[exemplar[family]])}"
        rows.append(
            {
                "x": float(np.median(x[members])),
                "y": float(np.median(y[members])),
                "text": name,
                "content": name,
                "priority": float(size),
            }
        )
    return pd.DataFrame(rows, columns=["x", "y", "text", "content", "priority"])


def load_gold(path: Path = GOLD_PATH) -> Optional[pd.DataFrame]:
    """Gold genre labels, or None when the file is missing."""
    if not path.exists():
        logger.warning("No gold labels at %s, so genre prediction is skipped.", path)
        return None
    return pd.read_json(path, lines=True)


def attach_gold(df: pd.DataFrame, gold: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Gold columns aligned to df's rows, and how many labels matched no chunk.

    Labels are keyed by (sutta_uid, chunk_index), not by chunk uuid, so
    re-chunking a sutta orphans its labels. Orphans are counted and reported.
    """
    keys = ["sutta_uid", "chunk_index"]
    labels = gold.rename(columns=GOLD_RENAME)[keys + GOLD_COLUMNS]
    aligned = df[keys].merge(labels, on=keys, how="left", validate="many_to_one")
    found = labels.merge(df[keys].drop_duplicates(), on=keys, how="left", indicator=True)
    aligned = aligned[GOLD_COLUMNS].astype(
        {"gold_is_verse": "boolean", "gold_has_simile": "boolean"}
    )
    return aligned.reset_index(drop=True), int((found["_merge"] == "left_only").sum())


def predict_genre(
    vectors: np.ndarray, df: pd.DataFrame, gold_mode: pd.Series
) -> tuple[pd.DataFrame, Optional[float]]:
    """A predicted rhetorical mode for every chunk, and the probe's honest score.

    The probe is the scale-up notebook's: standardised Voyage vectors into a
    balanced logistic regression. Gold rows get out-of-fold predictions, grouped
    by sutta, so any disagreement with gold comes from a sutta the model never
    trained on. A fit on all the gold rows predicts every other row.
    """
    labelled = np.flatnonzero(gold_mode.notna().to_numpy())
    if labelled.size == 0:
        logger.warning("No gold label matched a chunk, so genre prediction is skipped.")
        empty = pd.DataFrame({"pred_mode": [None] * len(df), "pred_confidence": np.nan})
        return empty, None

    sutta = df["sutta_uid"].to_numpy()
    # The notebook's row order, so that its folds, and therefore its score, carry over.
    labelled = labelled[np.lexsort((df["chunk_index"].to_numpy()[labelled], sutta[labelled]))]

    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    features, modes = unit[labelled], gold_mode.to_numpy()[labelled].astype(str)
    probe = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")
    )
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    held_out = cross_val_predict(
        probe, features, modes, groups=sutta[labelled], cv=folds, method="predict_proba"
    )

    proba = probe.fit(features, modes).predict_proba(unit)
    proba[labelled] = held_out
    classes = probe.classes_
    predictions = pd.DataFrame(
        {"pred_mode": classes[proba.argmax(axis=1)], "pred_confidence": proba.max(axis=1)}
    )
    return predictions, float(f1_score(modes, classes[held_out.argmax(axis=1)], average="macro"))


def build(
    vectors: np.ndarray, df: pd.DataFrame, gold: Optional[pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The snapshot: one row per chunk, the map labels, and the numbers behind them."""
    table = df[META_COLUMNS].reset_index(drop=True).copy()

    layout = project(centre(vectors), n_neighbors=UMAP_NEIGHBOURS, seed=SEED)
    table["x"] = layout[:, 0].astype(float)
    table["y"] = layout[:, 1].astype(float)

    ids, cosines = neighbours(vectors)
    table["neighbors"] = [
        {"ids": row.tolist(), "distances": (1.0 - cos.astype(float)).tolist()}
        for row, cos in zip(ids, cosines)
    ]
    table["hub_count"] = hubness(vectors, K)
    table["neighbour_mix"] = neighbour_mix(ids, table)

    components, family_stats = families(vectors, table, FAMILY_THRESHOLD)
    family_id, family_size = number_families(components)
    table["family_id"] = pd.Series(family_id, dtype="Int64").mask(family_id == 0)
    table["family_size"] = family_size

    stats: dict = {
        "chunks": len(table),
        "map": {
            "method": "umap",
            "vectors": "centred",
            "metric": "cosine",
            "n_neighbors": UMAP_NEIGHBOURS,
            "min_dist": 0.1,
            "seed": SEED,
        },
        "neighbours": {"k": K, "vectors": "raw", "metric": "cosine"},
        "families": {"threshold": FAMILY_THRESHOLD, **family_stats},
        "genre": None,
    }

    if gold is None:
        for column in GOLD_COLUMNS + PRED_COLUMNS:
            table[column] = None
    else:
        gold_columns, unmatched = attach_gold(table, gold)
        predictions, score = predict_genre(vectors, table, gold_columns["gold_mode"])
        table = pd.concat([table, gold_columns, predictions], axis=1)
        stats["genre"] = {
            "gold_labels": len(gold),
            "matched": int(gold_columns["gold_mode"].notna().sum()),
            "unmatched": unmatched,
            "probe_oof_macro_f1": score,
        }
        if unmatched:
            logger.warning("%d of %d gold labels matched no chunk.", unmatched, len(gold))

    return table, family_labels(table, vectors), stats


def _replace(path: Path, write: Callable[[Path], object]) -> None:
    """Write beside the target, then swap it in, so no reader sees half a file."""
    partial = path.with_name(path.name + ".partial")
    partial.unlink(missing_ok=True)
    write(partial)
    os.replace(partial, path)


def write_parquet(table: pd.DataFrame, labels: pd.DataFrame, out_dir: Path) -> None:
    """Embedding Atlas's inputs: one row per chunk in cache order, and the family labels.

    Row order is part of the format: Embedding Atlas reads neighbour ids as
    zero-based row indices.
    """
    _replace(out_dir / "explore.parquet", lambda path: table.to_parquet(path, index=False))
    _replace(out_dir / "explore_labels.parquet", lambda path: labels.to_parquet(path, index=False))


def write_sqlite(table: pd.DataFrame, path: Path) -> None:
    """Datasette's input: suttas, chunks and neighbours, linked by keys and searchable.

    Chunks go in reading order, sutta by sutta. Neighbour links name chunks by
    uuid, so this order is free to differ from the parquet's.
    """
    suttas = (
        table.groupby("sutta_uid", sort=True)
        .agg(
            doc_title=("doc_title", "first"), nikaya=("nikaya", "first"), n_chunks=("uuid", "size")
        )
        .reset_index()
    )
    chunks = table.sort_values(["sutta_uid", "chunk_index"])[CHUNK_COLUMNS]

    uuids = table["uuid"].to_numpy()
    ids = np.array([list(row["ids"]) for row in table["neighbors"]])
    distances = np.array([list(row["distances"]) for row in table["neighbors"]], dtype=float)
    links = pd.DataFrame(
        {
            "uuid": np.repeat(uuids, ids.shape[1]),
            "rank": np.tile(np.arange(1, ids.shape[1] + 1), len(ids)),
            "neighbour_uuid": uuids[ids.ravel()],
            "cosine": 1.0 - distances.ravel(),
        }
    )

    def write(partial: Path) -> None:
        with closing(sqlite3.connect(partial)) as db:
            db.executescript(SCHEMA)
            suttas.to_sql("suttas", db, if_exists="append", index=False)
            chunks.to_sql("chunks", db, if_exists="append", index=False)
            links.to_sql("neighbours", db, if_exists="append", index=False)
            db.execute(
                "INSERT INTO chunks_fts(rowid, chunk_text) SELECT rowid, chunk_text FROM chunks"
            )
            db.commit()

    _replace(path, write)


def write_provenance(stats: dict, out_dir: Path) -> None:
    """explore.json: which corpus the snapshot describes, and how it was built."""
    _replace(
        out_dir / "explore.json", lambda path: path.write_text(json.dumps(stats, indent=2) + "\n")
    )


def summary(stats: dict) -> str:
    """What the snapshot holds, in a few lines for the terminal."""
    found = stats["families"]
    lines = [
        f"Snapshot of {stats['chunks']:,} chunks (corpus {stats['corpus']}).",
        f"Pericope families: {found['n_families']:,}, holding {found['duplicate_mass']:.0%} of chunks.",
    ]
    genre = stats["genre"]
    if genre is None:
        lines.append("Genre: no gold labels, so no predictions.")
    elif genre["probe_oof_macro_f1"] is None:
        lines.append("Genre: no gold label matched a chunk, so no predictions.")
    else:
        lines.append(
            f"Genre: {genre['matched']} of {genre['gold_labels']} gold labels matched a chunk; "
            f"probe out-of-fold macro F1 {genre['probe_oof_macro_f1']:.3f}."
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out-dir", type=Path, default=CACHE_DIR, help="where to write the snapshot"
    )
    parser.add_argument("--gold", type=Path, default=GOLD_PATH, help="gold genre labels (JSONL)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not check_drift():
        print(
            "The atlas cache no longer matches the database. Rebuild it first:\n"
            "    poetry run python -m atlas.refresh --rebuild"
        )
        return 1

    vectors, df = load()
    table, labels, stats = build(vectors, df, load_gold(args.gold))
    stats["corpus"] = fingerprint(table["uuid"])
    stats["built_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(table, labels, args.out_dir)
    write_sqlite(table, args.out_dir / "explore.db")
    write_provenance(stats, args.out_dir)

    print(summary(stats))
    print(LAUNCH.format(out=args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
