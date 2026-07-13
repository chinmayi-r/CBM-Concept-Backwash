# REPRODUCTION PLAN — every reference-notebook section → curated, by intention

Goal (per your ask): go through **every** reference notebook, split into small batches,
state each batch's **intention**, and reproduce it in curated — **on the official
minimal_cbm models** — keeping what proves its intention, replacing what's rambling /
wrong / doesn't fit with something better, readable start-to-finish by a newcomer.

Status legend: ✅ done in curated · 🔁 needs the renderer-swap SLURM job
(`train/renderer_swap.slurm` → `analysis/z_ordering_swap.py`) · ✂️ replaced with a
better/cleaner version · ⏳ pending a dependency (trained model / masks).

---

## `fb_mcbm_renderer_swap.ipynb` (92 cells) → curated `notebooks/03_funnybirds_mcbm.ipynb`

| ref § | intention (what it's trying to show) | curated |
|------|--------------------------------------|---------|
| 0–3 setup, data, params | load FunnyBirds, concept names, class×concept, render params | ✅ (SETUP + nb01) |
| 4 renderer setup | bring up the Node renderer, url-encode a bird | 🔁 (`renderer_swap.slurm` starts it) |
| 5–6 MCBM load + inference | load MCBM, get z / c_preds from an image | ✂️ replaced: curated `load_model` (minimal_cbm), not old ckpt format |
| 5a/6a sanity checks | verify stored γ matches file; live z == cached z | ✅ keep as asserts in the driver |
| 7–8 part-swap + z-ordering record | swap part j to a donor variant; record z_old/z_new margins | 🔁 core of `z_ordering_swap.py` |
| 9–11 species pairs + main sweep | pick species pairs, run the swap for every γ | 🔁 driver `--gammas --seeds` |
| 12–13 load results, per-γ summary | frac_correct per part per γ | 🔁 → nb03 §"swap results" |
| 13 "every concept 50%??" diagnostic | realize the combined metric locks at 0.5 | ✂️ replace: decompose into fwd/bwd up front (the 0.5 was a metric artifact) |
| 14 detailed per-γ | normalized margin, per-part z boxplots, tail good-vs-fail | 🔁 nb03 detail cells |
| 15 cross-γ | frac_correct vs γ per part; violation heatmap | 🔁 |
| 16 occlusion filter | drop swaps where the part isn't visible | ✅ **already have this** = visible-only `changed_frac` gate |
| 17 z distributions | z_new/z_old spread per γ | 🔁 |
| 18 final summary table | one table, all γ | 🔁 |
| **19 fwd/bwd grounding heatmap** | the headline: fwd & bwd accuracy per part per γ | 🔁 **the money figure** |
| **20 IB compression vs response** | does IB compressing z buy grounding? | ✅ deletion-adapted (nb03 §5) → upgrade to swap version when 🔁 lands |
| 21 pre/post-swap z dists | species-encoder signature | 🔁 |
| 22 CBM comparison | does sigmoid fix grounding? (CBM row on the heatmap) | 🔁 (needs CBM swap too) |
| 23 why γ=0 ≠ CBM | z_old dist MCBM γ0 vs CBM | ✅ documented (DECISIONS "γ=0≠CBM") + 🔁 figure |
| 24 training-distribution confound | how often would visibility-relabeling flip a label | ✅ FunnyBirds visible-only gate + **CUB majority-vote analog (nb04)** |

## `fb_cbm_renderer_swap_v2.ipynb` → `notebooks/02_funnybirds_cbm.ipynb`
CBM analogue of the above (z-ordering margin on the sigmoid scale, occlusion, tail
confusion/anchoring). Renderer-swap parts 🔁 (same driver, `--config-prefix funnybirds-cbm`);
deletion + species-probe + per-species leakage ✅ (nb02 §1/§2/§3c).

## `fb_cbm_counterfactual.ipynb` → `notebooks/02_funnybirds_cbm.ipynb`
| §6 species probe ✅ · §5d per-species leakage ✅ · §8 three-level evidence ✂️ (fold into nb02 takeaway as the layered summary) |

## CUB recall notebooks (`03_ana_cub*.ipynb`, `mcbm_recall_full.ipynb`) → `notebooks/04` + `05/06`
Recall-gap axis. Raw-annotation version ✅ (nb04). Model-side recall gap ⏳ (needs a
trained CUB model → raw-CUB→pkl generator).

---

## Build order to close this out
1. **`analysis/z_ordering_swap.py`** — port the swap driver: reuse the renderer/swap/
   z-ordering/CSV logic from `run_z_ordering_sweep.py`, replace old model-loading with
   curated `load_model`. (Next build; iterate against the renderer on adroit.)
2. Run `train/renderer_swap.slurm` → CSVs in `$CURATED_DATA/swap/`.
3. nb03 §13–24 + nb02 swap cells read those CSVs (fwd/bwd heatmap = the headline).
4. raw-CUB→pkl generator → CUB70 training → nb05/06.
