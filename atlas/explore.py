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

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from atlas.structure import exemplars

logger = logging.getLogger(__name__)

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
        rows.append(
            {
                "x": float(np.median(x[members])),
                "y": float(np.median(y[members])),
                "text": f"{size}× {opening(text[exemplar[family]])}",
                "priority": float(size),
            }
        )
    return pd.DataFrame(rows, columns=["x", "y", "text", "priority"])


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
    aligned = aligned[GOLD_COLUMNS].astype({"gold_is_verse": "boolean", "gold_has_simile": "boolean"})
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
    X, y = unit[labelled], gold_mode.to_numpy()[labelled].astype(str)
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    held_out = cross_val_predict(probe, X, y, groups=sutta[labelled], cv=folds, method="predict_proba")

    proba = probe.fit(X, y).predict_proba(unit)
    proba[labelled] = held_out
    classes = probe.classes_
    predictions = pd.DataFrame(
        {"pred_mode": classes[proba.argmax(axis=1)], "pred_confidence": proba.max(axis=1)}
    )
    return predictions, float(f1_score(y, classes[held_out.argmax(axis=1)], average="macro"))
