# Data-analysis notebooks

Two notebooks, publication quality. They are **dataset characterization +
training/grounding validation** — not the leakage/backwash results (those are the
explicit next-step notebooks). Every cell loads an artifact produced by the
`train/` and `data/` scripts; before those run on adroit, cells print a
`[pending]` note instead of failing, so each notebook executes top-to-bottom at
any time. Figures are written to `notebooks/figures/` as PDF (LaTeX) + PNG.

Shared logic lives in `curated/analysis/` (io schema, plotting style, occlusion
metrics) so the notebooks stay short and both CBM and MCBM go through one code
path.

## `01_funnybirds_data_and_validation.ipynb` — 10 slots

| # | Slot |
|---|------|
| 1 | Dataset summary table (images, classes, concepts, prevalence range) |
| 2 | Class×concept matrix (confirms one-hot-per-part loading) |
| 3 | CBM training curves (concept & task loss) |
| 4 | CBM final metrics (per-concept + task acc) |
| 5 | MCBM training curves (task, concept, **z-regularizer**) |
| 6 | MCBM final metrics |
| 7 | CBM vs MCBM bar (task acc, mean concept acc) |
| 8 | z distribution both models (**fixed-point / collapse check**) |
| 9 | Example images: predicted vs GT concepts |
| 10 | Intervention sanity check (acc vs #concepts intervened) |

## `02_cub_data_and_validation.ipynb` — Part A (10) + Part B (8)

**Part A — full CUB-200:** A1 dataset summary · A2 attribute-prevalence histogram ·
A3 CBM curves · A4 CBM metrics · A5 MCBM curves · A6 MCBM metrics · A7 CBM-vs-MCBM
bar · A8 z distribution · A9 example real images · A10 intervention curve.

**Part B — CUB70 occlusion/relabeling axis** (cap raised; this is a separate axis):

| # | Slot | Prof note |
|---|------|-----------|
| B11 | CUB70 summary (11 parts, mask area per part) | — |
| B12 | CUB label vs CUB70 visibility cross-tab (occluded-share per part) | #1 |
| B13 | Relabeled concept table (present→absent flip counts) | #1 |
| B14 | z (full-200 model) vs CUB70 visibility, per part | #2 |
| B15 | CUB70-only training curves | #3 |
| B16 | z vs CUB70 visibility on the CUB70-only model | #4 |
| B17 | Comparison table: 3 conditions (full200-on-cub70, cub70-orig, cub70-relabeled) | — |
| B18 | **Verdict table**: grounding violation rate by condition | #5 |

Slot B18 is the one that closes the loop: it says whether relabeling alone fixes
grounding, whether restricting the training data alone fixes it, or the problem
persists regardless — i.e. whether the full pixel-segmentation investment is
justified.

## Expected artifacts (produced on adroit)

```
$CURATED_DATA/
  funnybirds_processed/{train,val,test}.pkl,  funnybirds_visibility.parquet
  CUB_processed/class_attr_data_10[ _relabeled ]/*.pkl
  cub70_visibility.parquet,  cub70_relabel_diagnostics.parquet
  runs/<run>/history.parquet            # parsed training log: epoch, *_loss
  runs/<run>/eval_test.parquet          # io.EVAL_SCHEMA (per image×concept)
  runs/<run>/eval_cub70.parquet         # same, restricted to CUB70 images
  runs/{cub,funnybirds}_tti.parquet     # model, n_intervened, task_acc
```

`history.parquet` and `eval_*.parquet` are emitted by small dump steps documented
in `curated/analysis/io.py` (`build_eval_table_cbm` / `build_eval_table_mcbm`);
fill those in on the cluster where the checkpoints live.
