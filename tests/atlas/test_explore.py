"""Explorer snapshot tests: synthetic vectors, no network, no UMAP."""

import numpy as np
import pandas as pd

from atlas import explore


def _unit(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def _corpus(seed=0):
    """Twelve chunks in four suttas, with one planted pericope family across three of them."""
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((12, 16)).astype(np.float32)
    vectors[[1, 5, 9]] = vectors[1] + 0.001 * rng.standard_normal((3, 16))
    texts = [f"* Linked Discourses\n\n# Title {i}\nPassage number {i} about calm" for i in range(12)]
    texts[4] += " and restlessness"
    df = pd.DataFrame(
        {
            "uuid": [f"u{i:02d}" for i in range(12)],
            "sutta_uid": [f"sn{i // 3}" for i in range(12)],
            "nikaya": "sn",
            "doc_title": [f"Sutta {i // 3}" for i in range(12)],
            "chunk_index": [i % 3 for i in range(12)],
            "word_count": 50,
            "chunk_text": texts,
        }
    )
    return _unit(vectors), df


def test_neighbours_exclude_self_and_come_nearest_first():
    vectors, _ = _corpus()

    ids, cosines = explore.neighbours(vectors, k=4)

    assert ids.shape == (12, 4)
    assert not (ids == np.arange(12)[:, None]).any()
    assert (np.diff(cosines, axis=1) <= 1e-6).all()
    np.testing.assert_allclose(cosines, np.einsum("ij,ikj->ik", vectors, vectors[ids]), rtol=1e-5)
    assert set(ids[1, :2].tolist()) == {5, 9}


def test_neighbour_mix_is_the_share_from_the_same_sutta():
    df = pd.DataFrame({"sutta_uid": ["a", "a", "a", "b", "c"]})
    ids = np.array([[1, 2], [0, 3], [3, 4], [0, 4], [3, 2]])

    np.testing.assert_allclose(explore.neighbour_mix(ids, df), [1.0, 0.5, 0.0, 0.0, 0.0])


def test_families_are_numbered_largest_first_and_singletons_get_zero():
    components = np.array([4, 2, 4, 3, 2, 4, 0])

    family_id, size = explore.number_families(components)

    assert family_id.tolist() == [1, 2, 1, 0, 2, 1, 0]
    assert size.tolist() == [3, 2, 3, 1, 2, 3, 1]
