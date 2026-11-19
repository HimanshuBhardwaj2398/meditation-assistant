# Design: Corpus Explorer, the Corpus in the Browser

**Date**: 2026-11-19
**Status**: Approved
**Author**: Himanshu (with Claude)
**Branch**: `feature/corpus-explorer`
**Scope**: Browse the corpus and its embedding space in two ready-made browser tools that
read one exported snapshot. Insight only. This design changes nothing in retrieval, in the
database, or in the corpus workbench.

---

## 1. Context

This design answers a question from 2026-11-18: can a Datasette-style tool show the
database in the browser and plot the embeddings, to find patterns?

Every look at the embedding space so far lives in a notebook: the atlas
(`atlas/atlas.ipynb`) and the two genre studies (`retrieval/genre_separability.ipynb`,
`retrieval/genre_classifier_scaleup.ipynb`). Each one re-runs the same Neon join and draws
fixed Plotly figures. To recolour a plot or read the chunks behind a dot, you edit code.

Decisions made in brainstorming (2026-11-19):

| Question | Decision |
|---|---|
| How it fits | Ready-made tools now, as a side quest outside the corpus-workbench arc. No frontend code. |
| Tools | **Embedding Atlas** (Apple, MIT) for the map. **Datasette** with `datasette-plot` for tables, SQL and quick charts. |
| Signals | The basics, plus genre (gold and predicted), pericope families, and hubs with neighbour mix. The eval-question overlay waits for a later design. |

### Facts that shaped the design

- **Datasette reads SQLite only** (checked November 2026), and the corpus lives in Neon.
  Both tools therefore read a snapshot file. At 7,144 chunks a snapshot costs about 30 MB
  of vectors and a minute of compute.
- **The atlas already keeps a snapshot.** `atlas/loader.py` caches vectors and metadata in
  `data/atlas/`, and `atlas/refresh.py` refuses a cache whose fingerprint no longer
  matches the database. On 2026-11-19 the cache held 1,593 DN and MN chunks from
  2026-08-30, against 7,144 in the database.
- **The atlas findings constrain the map**
  ([embedding atlas design](2026-08-16-embedding-atlas-design.md), measured on DN and MN).
  HDBSCAN left 76–100% of chunks unassigned, so the corpus has no topical clusters to draw.
  The nikāya carried no semantic signal between DN and MN (section 11 revisits this on the
  full canon). One shared direction made up 64% of every vector.
  The real discrete structure sat in the pericope families. Automatic cluster labels would
  invent structure the data lacks.
- **Genre shows in the geometry.** The probe in `retrieval/genre_classifier_scaleup.ipynb`
  recovers rhetorical mode from the stored Voyage vectors at 0.792 macro F1 across 291
  suttas, from the 482 gold labels in `data/labels/genre_gold_v2.jsonl`.
- **The tools' current releases set two pins.** `datasette-plot` 0.1.0 requires Datasette
  1.0, still a pre-release (1.0a43). The stable line (0.65.6) cannot load it. Embedding
  Atlas stands at 0.26.0.

## 2. Isolation from the corpus workbench

The workbench website will get its own exploration page. This design stays out of its way:

- The explorer lives on its own branch, `feature/corpus-explorer`, in a separate worktree
  cut from `main`.
- It adds files under `atlas/` and `tests/atlas/` and one line each in `docs/README.md`
  and `CLAUDE.md`. The workbench changes touch none of these files.
- It reads the database and writes nothing back: no tables, no migrations. The
  `chunk_genre` table stays with workbench step 3.
- It adds no API endpoints and nothing under `api/` or a future `console/`.
- Its tools listen on ports 5055 and 8001, clear of the API (8000) and Vite (5173).

Whether the website's exploration page reuses any of this stays open. Embedding Atlas ships
a React component (`embedding-atlas/react`) that could read the same snapshot.

## 3. Layout

```
atlas/
  explore.py               snapshot builder: pure functions, writers, CLI
  explore_metadata.yaml    Datasette title, source and column descriptions
  explore_datasette.yaml   Datasette stored queries
data/atlas/                gitignored, rebuilt on demand
  explore.parquet          one row per chunk, for Embedding Atlas
  explore_labels.parquet   map labels: the largest pericope families
  explore.db               SQLite with full-text search, for Datasette
  explore.json             provenance: corpus fingerprint, parameters, probe score
tests/atlas/test_explore.py
```

## 4. Data flow

```
Neon --atlas.refresh --rebuild--> data/atlas/{vectors.npy, meta.parquet}
     --python -m atlas.explore--> explore.parquet + explore_labels.parquet --> uvx embedding-atlas
                                  explore.db                               --> uvx datasette
                                  explore.json
```

The snapshot is a cache. Nothing flows back to Postgres, which keeps derived data
rebuildable, as the [enrichment layer design](enrichment-layer-design.md) requires.

## 5. Components

### explore.py

One module of plain functions over `X: np.ndarray[n, 1024]` and `df: pd.DataFrame`, in the
style of the rest of `atlas/`.

| Function | Job | Reuses |
|---|---|---|
| `neighbours` | Exact top 10 by raw cosine, self excluded. Returns row ids and cosines. | |
| `neighbour_mix` | Share of each chunk's neighbours from its own sutta. | |
| `predict_genre` | Joins gold labels on `(sutta_uid, chunk_index)` and predicts a mode for every chunk. | the scale-up notebook's probe |
| `family_labels` | Map labels for the largest pericope families. | `structure.exemplars` |
| `build` | Assembles the chunk table and the label table. | `geometry.centre`, `geometry.hubness`, `structure.project`, `pericopes.families` |
| `write_parquet`, `write_sqlite`, `write_provenance` | Write each output to a temp path, then `os.replace`, so a failed run leaves no half-written file. | `loader.fingerprint` |
| `main` | Refuses a stale cache, builds, writes, and prints a summary with the launch commands. | `loader.load`, `loader.check_drift` |

### Two geometric choices

- **Map position: UMAP of the centred vectors** (`n_neighbors=15`, `min_dist=0.1`, cosine,
  `seed=42`). Centring strips the shared direction that squeezes every similarity into a
  narrow band; on DN and MN it widened the within-sutta versus cross-nikāya gap from 0.064
  to 0.186. The map serves looking. Distance on it is not similarity.
- **Neighbours: exact top 10 by raw cosine in the full 1024 dimensions.** Retrieval ranks
  by raw cosine, so clicking a chunk answers "what would retrieval return near this one?".
  UMAP distortion cannot touch these lists.

`explore.json` records both choices.

### Genre prediction

The probe copies the scale-up notebook: `StandardScaler`, then
`LogisticRegression(max_iter=3000, class_weight="balanced")`, on the unit-norm Voyage
vectors.

- Gold rows get **out-of-fold** predictions from `StratifiedGroupKFold(5, shuffle=True,
  random_state=0)`, grouped by sutta, with rows sorted by `(sutta_uid, chunk_index)` as the
  notebook sorts them. When gold and prediction disagree, the model missed; it never saw
  that sutta in training.
- Every other chunk gets a prediction from a fit on all 482 gold rows.
- `pred_confidence` holds the top class probability. Low values mark the chunks most worth
  labelling next.
- `explore.json` stores the out-of-fold macro F1 as a regression check. It should land near
  the notebook's 0.792.

### Map labels

`--labels` replaces Embedding Atlas's automatic density labels with the corpus's real
structure: the 40 largest pericope families of size 3 or more. Each label sits at the
median map position of its members and reads as the family size plus the opening words of
the family exemplar, the member nearest the family centroid. The family size sets its
priority.

Embedding Atlas 0.26.0 needs two workarounds, found in the first browser run. Its docs name
`text` for `--labels`, but its frontend renders `content` and throws on a label without it,
so each label carries both. A label without a `level` shows only inside a narrow zoom
window, so each label gets level 0; overlap placement by priority then decides which
labels show at each zoom.

### Snapshot schema

`explore.parquet` holds one row per chunk, in cache order. Row order matters: Embedding
Atlas reads neighbour ids as zero-based row indices.

| Columns | Source |
|---|---|
| `uuid`, `sutta_uid`, `nikaya`, `doc_title`, `chunk_index`, `word_count`, `chunk_text` | the atlas cache |
| `x`, `y` | UMAP of the centred vectors |
| `neighbors` (parquet only): `{"ids": [...], "distances": [...]}`, distance = 1 − cosine | `neighbours` |
| `hub_count`, `neighbour_mix` | `geometry.hubness` at k=10, `neighbour_mix` |
| `family_id` (null without a family), `family_size` | `pericopes.families` at cosine 0.90 |
| `gold_mode`, `gold_is_verse`, `gold_has_simile` (null where unlabelled) | `genre_gold_v2.jsonl` |
| `pred_mode`, `pred_confidence` | `predict_genre` |

`explore.db` holds the same chunk columns without `neighbors`, spread over three tables and
a search index:

- `suttas(sutta_uid PK, doc_title, nikaya, n_chunks)`
- `chunks(uuid PK, sutta_uid FK → suttas, ...)`
- `neighbours(uuid FK → chunks, rank, neighbour_uuid FK → chunks, cosine)`, primary key
  `(uuid, rank)`
- `chunks_fts`, an FTS5 index over `chunk_text` with `content=[chunks]`, the sqlite-utils
  convention that Datasette detects

Neither file stores vectors.

### The tools

Both run through `uvx` with pinned versions, so the project gains no dependencies and the
Poetry environment stays untouched.

```bash
uvx embedding-atlas@0.26.0 data/atlas/explore.parquet \
    --text chunk_text --x x --y y --neighbors neighbors \
    --labels data/atlas/explore_labels.parquet

uvx --with datasette-plot==0.1.0 datasette@1.0a43 data/atlas/explore.db \
    -m atlas/explore_metadata.yaml -c atlas/explore_datasette.yaml
```

`atlas/explore_metadata.yaml` carries the title, the source and the column descriptions.
The descriptions carry the caveats: x and y serve looking only, and `pred_*` holds probe
output at 0.79 macro F1, not ground truth. `atlas/explore_datasette.yaml` holds five stored
queries:

| Query | Answers |
|---|---|
| `largest_families` | Which formulas repeat most, and across how many suttas? |
| `suttas_sharing_families` (`:sutta_uid`) | Which suttas share pericope families with this one? |
| `top_hubs` | Which chunks sit near everything, and what do they say? |
| `gold_vs_predicted` | Where does the probe disagree with a gold label? |
| `label_candidates` | Which chunks does the probe find hardest? |

## 6. Error handling

- A stale cache stops `main` with exit code 1 and prints the refresh command
  (`python -m atlas.refresh --rebuild`), the atlas's existing guard.
- A gold label that matches no chunk (a re-chunked sutta) produces a warning with the
  count. The run continues, and `explore.json` records the number.
- A missing gold file skips prediction. The `pred_*` columns stay null, and a warning says
  why.
- An empty collection raises `CollectionError` from the loader, as today.
- Scale: like the rest of `atlas/`, the module builds dense n×n matrices. They fit up to
  roughly 20k chunks; past that, neighbours need a blockwise or approximate search. The
  module docstring says so.

## 7. Testing

`tests/atlas/test_explore.py` uses synthetic vectors and no network, like the other atlas
tests.

| Target | Test |
|---|---|
| `neighbours` | Self excluded, sorted by cosine, distance = 1 − cosine. |
| `neighbour_mix` | Toy suttas give known shares. |
| `predict_genre` | Separable toy classes grouped by fake suttas get valid labels and confidences; gold rows get out-of-fold predictions; an unmatched gold label is counted, not dropped. |
| `family_labels` | Singletons skipped, labels at member medians, priority = size, top-N cap. |
| `write_sqlite` | Tables, keys, n×k neighbour rows, and an FTS `MATCH` that finds a planted word. |
| `write_parquet` | The neighbours column round-trips in Embedding Atlas's shape with zero-based ids. |
| `main` | A stale cache exits 1. |

The tests make no assertions about the UMAP layout or about how the tools render. A person
checks those in the browser, against the script below.

## 8. Exploration script

The definition of working, written before the build. It doubles as a first lesson in
reading an embedding map.

1. Colour the map by `nikaya`. DN and MN should mix, as the atlas found. Where another
   collection gathers, check the full-dimension neighbour lists before suspecting a loading
   or projection bug: structure that the lists share is real (section 11).
2. Colour by `pred_mode`. Do `practice` chunks form regions or scatter? Lasso a
   practice-heavy area and read five chunks.
3. Click an MN 10 chunk. Its DN 22 twin should appear among the neighbours, the atlas's
   built-in correctness check.
4. In Datasette, run `largest_families`, open the biggest family, and note which suttas
   share it. Then run `suttas_sharing_families` for `mn10`.
5. Run `top_hubs`. Formulaic openers, or real content?
6. Run `gold_vs_predicted` and `label_candidates`. Classifier errors, or label errors? Note
   ten chunks to label once workbench step 3 exists.

Steps 1 and 3 gate completion.

## 9. Dependencies

None added. The module uses numpy, pandas, pyarrow, scikit-learn, umap-learn and the
standard library's `sqlite3`, all present already. The two tools run in throwaway `uvx`
environments.

Datasette runs a pre-release because `datasette-plot` needs 1.0. If the pre-release
misbehaves, stable 0.65.6 without the plugin works too, with the stored queries moved into
the metadata file. That costs only the quick charts, and the map covers plotting anyway.

## 10. Out of scope

- **Eval questions on the map.** First the dataset's gold chunk UUIDs need a check that
  they survived the full ingest.
- **Live Postgres browsing.** Neon's console covers it.
- **Sharing the map.** `--export-application` builds a static copy; publishing it is a
  separate decision.
- **An exploration page on the workbench website.** See section 2.
- **Writing predictions to Postgres or `chunk_genre`, and any retrieval change.** Those go
  through the eval gate.

## 11. First run (2026-11-19)

The full canon: 7,144 chunks, exported in 41 seconds.

- **The probe reproduces.** All 482 gold labels matched a chunk. Out-of-fold macro F1 came
  to 0.792 and agreement to 82.8%, the scale-up notebook's numbers exactly. Median
  confidence is 0.997 and the 10th percentile 0.743, so `label_candidates` draws from a
  thin tail. The probe disagrees with 83 gold labels.
- **Pericope families hold 45% of the canon** at cosine 0.90, in 703 families (19% on DN
  and MN alone). The largest, 191 chunks across 188 suttas, is the Khandha Saṃyutta's
  "At Sāvatthī. 'Mendicants, form is impermanent…'".
- **The nikāya shows on the full canon.** Coloured by collection, the map does not mix: SN,
  AN and the verse books each gather. The full-dimension neighbour lists agree, so the
  projection did not invent it:

  | Collection | Neighbours from the same collection | By chance |
  |---|---|---|
  | SN | 78% | 33% |
  | AN | 73% | 33% |
  | Theragāthā | 58% | 4% |
  | Itivuttaka | 47% | 2% |

  Leaving out a chunk's own sutta, 59% of its neighbours still share its collection. The
  collections differ in form (verse books, AN's numbered lists, SN's topical chains), and
  the vectors encode form. The atlas finding holds for DN against MN only.
- **MN 10 and DN 22 pass.** Every one of MN 10's 15 chunks has its DN 22 twin as nearest
  neighbour, mostly at cosine 1.000. The alignment runs down the diagonal and skips DN 22
  chunks 13 to 20, the expanded Four Noble Truths section, which is the shape the atlas
  predicted.
- **Hubs:** AN 5.87's opening chunk sits in 74 other chunks' top 10; the median chunk sits
  in 8.
- **Gold labels live outside git.** `data/labels/` is untracked, so a fresh checkout or
  worktree has no gold file and the build skips prediction. Pass `--gold` with the file's
  path, or commit the labels.
