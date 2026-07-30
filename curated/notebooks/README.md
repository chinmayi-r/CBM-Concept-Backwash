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
| 05 | `05_cub_cbm` | CBM output versus CUB70 mask visibility; full-CUB vs CUB70-trained | `cub70_eval/`, masks |
| 06 | `06_cub_mcbm` | Same visibility test for MCBM, including γ and training-partition comparisons | `cub70_eval/`, masks |

**Why CUB is split inside 05/06 (not two notebooks).** Full CUB (200) supports
only the recall-gap axis — attributes vary within a species, but there are **no
part masks**, so no occlusion/grounding. CUB70 (first 70 classes) ships per-part
segmentation masks → visibility grounding on real photographs. This is
observational, not a rendered deletion counterfactual. The masks cover test images
only, so they support evaluation-label diagnostics but not relabeled retraining.

Legacy exploratory notebooks are archived under `_legacy/`.

## Which artifacts, and how they're produced (adroit)

See `../RUNBOOK.md` for the per-claim commands. In short:
```
$CURATED_DATA/
  data_analysis/                          # 01/04 : data_analysis.py
  grounding/funnybirds-<model>-s*.parquet # 02/03 : grounding_deletion.py (via grounding_sweep.sh)
  species_probe/funnybirds-<model>-s*.json# 02/03 : species_probe.py (via grounding_sweep.sh)
  backwash_vs_gamma.{csv,png,pdf}         # 03    : collect_backwash.py
  cub70_visibility.parquet                # 04/05/06 : mask areas
  cub70_eval/<config>-s*.parquet          # 05/06 : saved model predictions
  cub70_relabel_diagnostics.parquet       # mask-vs-original-label disagreement
```
CUB notebooks populate after the checkpoints are trained and
`analysis/cub70_prepare_analysis.sh` exports their saved predictions.

## Visibility-aware-label follow-up (03rl)

`03rl_funnybirds_mcbm_relabeled.ipynb` is the matched causal follow-up to
notebooks 02 and 03. It reads the six validated standard/RLv2 seed-1 CSVs,
repeats the necessary notebook-02 controls as direct before/after comparisons,
and investigates residual failures by part, visibility, variant pair, controlled
source species, and actual rendered examples.

Execute it separately:

```bash
export CURATED_DATA=/scratch/network/cr7998/cv_emergence_project/curated_data
bash notebooks/run_03rl_notebook.sh
```

The runner fails closed unless all six CSVs and the semantic renderer-preflight
figure exist under `$CURATED_DATA/swap_fixed_v2_attempt2`. RLv2 deletion and
species-probe artifacts may remain pending; the notebook labels those missing
tests explicitly.
