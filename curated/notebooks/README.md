# Notebooks — one per (dataset × step)

Six notebooks, one analysis step each. Data-characterization notebooks stand
alone; model notebooks are **thin viewers** over the artifacts the `train/` and
`analysis/` cluster scripts produce (`grounding/*.parquet`, `species_probe/*.json`,
`backwash_vs_gamma.csv`, `eval/*.parquet`). Heavy compute (GPU, renderer) lives in
the `.py` scripts; the notebooks load, plot, and interpret. Every cell that needs
an artifact prints `[pending] … produce it: <command>` when it's missing, so each
notebook runs top-to-bottom at any stage. Figures → `notebooks/figures/` (PDF+PNG).

Run from `curated/notebooks/` with `CURATED_DATA` exported.

| # | notebook | step | reads |
|---|----------|------|-------|
| 01 | `01_funnybirds_analysis` | FunnyBirds **data** (class balance, concept=f(class), species-constancy, part features) | pkls |
| 02 | `02_funnybirds_cbm` | FB **+ CBM** — deletion backwash + species probe + line-up | `grounding/`, `species_probe/` |
| 03 | `03_funnybirds_mcbm` | FB **+ MCBM** — backwash vs γ, species-code vs γ | `backwash_vs_gamma.csv`, `species_probe/` |
| 04 | `04_cub_analysis` | CUB **data** — full-CUB & CUB70 (hard-partitioned; CUB70 §5 = masks) | pkls, `cub70_visibility.parquet` |
| 05 | `05_cub_cbm` | CUB **+ CBM** — Part A full-CUB recall gap · Part B CUB70 occlusion + relabel | `eval/`, `grounding/` |
| 06 | `06_cub_mcbm` | CUB **+ MCBM** — same two partitions + CUB70 backwash vs γ | `eval/`, `grounding/`, `cub70_backwash_vs_gamma.csv` |

**Why CUB is split inside 05/06 (not two notebooks).** Full CUB (200) supports
only the recall-gap axis — attributes vary within a species, but there are **no
part masks**, so no occlusion/grounding. CUB70 (first 70 classes) ships per-part
segmentation masks → occlusion + part-level grounding on **real birds** +
relabeling. Same model, deliberately unequal evidence; the notebook hard-partitions
Part A (full, weak/broad) from Part B (CUB70, strong/causal).

Legacy exploratory notebooks are archived under `_legacy/`.

## Which artifacts, and how they're produced (adroit)

See `../RUNBOOK.md` for the per-claim commands. In short:
```
$CURATED_DATA/
  data_analysis/                          # 01/04 : data_analysis.py
  grounding/funnybirds-<model>-s*.parquet # 02/03 : grounding_deletion.py (via grounding_sweep.sh)
  species_probe/funnybirds-<model>-s*.json# 02/03 : species_probe.py (via grounding_sweep.sh)
  backwash_vs_gamma.{csv,png,pdf}         # 03    : collect_backwash.py
  cub70_visibility.parquet                # 04/05/06 : data/cub70 (WIP)
  eval/cub-<model>-s*.parquet             # 05/06 : io.build_eval_table (WIP)
  grounding/cub70-<model>-<labels>-s*.parquet  # 05/06 : CUB70 occlusion probe (WIP)
```
CUB pieces (05/06) are WIP — the notebooks are built and will populate once the
CUB70 training + occlusion scripts land (RUNBOOK C5/C6).
