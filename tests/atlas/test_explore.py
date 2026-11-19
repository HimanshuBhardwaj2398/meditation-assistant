"""Explorer snapshot tests: synthetic vectors, no network, no UMAP."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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


def _with_families(df, **columns):
    """The test corpus with family 1 (rows 1, 5, 9) and family 2 (rows 3, 7) assigned."""
    return df.assign(
        family_id=pd.array(
            [pd.NA, 1, pd.NA, 2, pd.NA, 1, pd.NA, 2, pd.NA, 1, pd.NA, pd.NA], dtype="Int64"
        ),
        family_size=[1, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 1],
        **columns,
    )


def test_opening_skips_markdown_headings_and_breadcrumbs():
    text = "* Middle Discourses 81\n\n# With Ghaṭīkāra  \nSo I have heard. At one time the Buddha"

    assert explore.opening(text, words=5) == "So I have heard. At…"


def test_family_labels_sit_at_member_medians_and_rank_by_size():
    vectors, df = _corpus()
    table = _with_families(df, x=np.arange(12, dtype=float), y=np.arange(12, dtype=float) * 2)

    labels = explore.family_labels(table, vectors, min_size=2)

    assert labels.columns.tolist() == ["x", "y", "text", "priority"]
    assert labels["priority"].tolist() == [3.0, 2.0]
    assert labels.loc[0, ["x", "y"]].tolist() == [5.0, 10.0]
    assert labels.loc[0, "text"].startswith("3× Passage number")


def test_family_labels_skip_small_families_and_cap_the_count():
    vectors, df = _corpus()
    table = _with_families(df, x=0.0, y=0.0)

    assert len(explore.family_labels(table, vectors, min_size=3)) == 1
    assert len(explore.family_labels(table, vectors, min_size=2, top=1)) == 1


def _genre_corpus(n_suttas=10, per_sutta=6, dims=8, seed=0):
    """Three separable modes spread over many suttas, so every fold sees every mode."""
    rng = np.random.default_rng(seed)
    n = n_suttas * per_sutta
    modes = np.array(["doctrine", "narrative", "practice"])[np.arange(n) % 3]
    vectors = 3 * np.eye(3, dims)[np.arange(n) % 3] + 0.3 * rng.standard_normal((n, dims))
    df = pd.DataFrame(
        {
            "sutta_uid": [f"s{i // per_sutta}" for i in range(n)],
            "chunk_index": [i % per_sutta for i in range(n)],
        }
    )
    return _unit(vectors), df, pd.Series(modes)


def test_attach_gold_aligns_on_sutta_and_index_and_counts_orphans():
    _, df = _corpus()
    gold = pd.DataFrame(
        [
            {"sutta_uid": "sn0", "chunk_index": 1, "mode": "practice", "is_verse": False, "has_simile": True},
            {"sutta_uid": "sn2", "chunk_index": 0, "mode": "doctrine", "is_verse": True, "has_simile": False},
            {"sutta_uid": "sn9", "chunk_index": 0, "mode": "narrative", "is_verse": False, "has_simile": False},
        ]
    )

    aligned, unmatched = explore.attach_gold(df, gold)

    assert unmatched == 1
    assert aligned["gold_mode"].iloc[1] == "practice"
    assert aligned["gold_mode"].iloc[6] == "doctrine"
    assert aligned["gold_mode"].notna().sum() == 2
    assert bool(aligned["gold_is_verse"].iloc[6]) is True
    assert pd.isna(aligned["gold_is_verse"].iloc[0])


def test_predict_genre_recovers_separable_modes():
    vectors, df, modes = _genre_corpus()
    gold = modes.where(np.arange(len(modes)) % 2 == 0)

    predictions, score = explore.predict_genre(vectors, df, gold)

    assert score > 0.95
    assert (predictions["pred_mode"] == modes).mean() > 0.95
    assert predictions["pred_confidence"].between(1 / 3, 1).all()


def test_gold_rows_get_out_of_fold_predictions():
    """A probe fitted on random labels memorises them; held-out predictions cannot."""
    rng = np.random.default_rng(1)
    n, dims = 40, 64
    vectors = _unit(rng.standard_normal((n, dims)))
    df = pd.DataFrame({"sutta_uid": [f"s{i // 4}" for i in range(n)], "chunk_index": [i % 4 for i in range(n)]})
    gold = pd.Series(np.array(["doctrine", "narrative", "practice"])[rng.permutation(np.arange(n) % 3)])
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    assert (probe.fit(vectors, gold).predict(vectors) == gold).mean() > 0.95

    predictions, _ = explore.predict_genre(vectors, df, gold)

    assert (predictions["pred_mode"] == gold).mean() < 0.7


@pytest.fixture
def flat_projection(monkeypatch):
    """UMAP is slow to import and nothing asserts on its layout, so use a fixed one."""

    def layout(vectors, **_):
        return np.column_stack([np.arange(len(vectors)), np.zeros(len(vectors))])

    monkeypatch.setattr(explore, "project", layout)


def _stub_probe(monkeypatch):
    """Stand in for the probe where only the wiring around it is under test."""

    def predict(vectors, df, gold_mode):
        return pd.DataFrame({"pred_mode": ["doctrine"] * len(df), "pred_confidence": 0.5}), 0.8

    monkeypatch.setattr(explore, "predict_genre", predict)


def test_build_assembles_one_row_per_chunk(flat_projection):
    vectors, df = _corpus()

    table, labels, stats = explore.build(vectors, df, gold=None)

    assert table["uuid"].tolist() == df["uuid"].tolist()
    assert len(table.loc[0, "neighbors"]["ids"]) == explore.K
    assert table["hub_count"].sum() == 12 * explore.K
    assert table.loc[1, "family_id"] == 1 and table.loc[1, "family_size"] == 3
    assert pd.isna(table.loc[0, "family_id"])
    assert table["pred_mode"].isna().all()
    assert stats["families"]["n_families"] == 1 and stats["genre"] is None
    assert labels["text"].tolist()[0].startswith("3×")


def test_build_attaches_gold_and_predictions(flat_projection, monkeypatch):
    _stub_probe(monkeypatch)
    vectors, df = _corpus()
    gold = pd.DataFrame(
        [
            {"sutta_uid": "sn0", "chunk_index": 1, "mode": "practice", "is_verse": False, "has_simile": False},
            {"sutta_uid": "zz", "chunk_index": 0, "mode": "doctrine", "is_verse": False, "has_simile": False},
        ]
    )

    table, _, stats = explore.build(vectors, df, gold)

    assert table.loc[1, "gold_mode"] == "practice"
    assert (table["pred_mode"] == "doctrine").all()
    assert stats["genre"] == {"gold_labels": 2, "matched": 1, "unmatched": 1, "probe_oof_macro_f1": 0.8}
