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

import numpy as np
import pandas as pd

K = 10


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
