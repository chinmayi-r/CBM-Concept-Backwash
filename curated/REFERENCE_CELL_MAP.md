# Reference → curated, cell by cell

Maps every substantive cell of `funnybird_notebooks/fb_mcbm_renderer_swap.ipynb` (92 cells)
to `curated/notebooks/03_funnybirds_mcbm.ipynb`, with the reason for each modification.

Status: ✅ ported ~as-is · ✏️ modified (reason given) · 🔁 moved to the SLURM driver ·
✂️ replaced with a cleaner equivalent · ❌ dropped (reason given).

---

## Setup & compute — ref cells 0–35  →  🔁 `analysis/z_ordering_swap.py` + `train/renderer_swap.slurm`
| ref | what it does | curated |
|-----|--------------|---------|
| 1–3 | imports, `ROOT` paths | 🔁 driver header |
| 5 `load_species_maps` | species↔variant maps from `dataset_test.json` | 🔁 driver (same logic) |
| 7–8 concept names, class×concept, meta | `FunnyBirdsDataset`, `load_meta` | ✏️ driver builds concept/species maps from `dataset_test.json` + `funnybirds_concepts` |
| 10–11 render params, `parts.json` | per-part variant lists, colors | 🔁 driver |
| 13–15 `json_to_url`, `check_renderer_alive`, eval transform | url-encode a bird; ping renderer | 🔁 driver; renderer started by the `.slurm` |
| 17 `load_mcbm_weights` | load **old** ckpt format (W_c,b_c,W_y,b_y) | ✂️ **replaced** by curated `load_model` on the **official minimal_cbm** checkpoints — the whole point of the reproduction |
| 19 `5a` γ-vs-filename check · 23 `6a` live-z==cached-z | sanity asserts | ✅ kept as asserts inside the driver |
| 21 `make_mcbm_inference_fns` | returns `(z, y_pred)` per image | ✏️ curated `make_run_fn` returns `(out["c_logits"][0], out["y_preds"][0])` — old "z" = concept logits = minimal_cbm `c_logits` (see DECISIONS) |
| 25–28 `swap_part_in_ann`, `z_ordering_record_mcbm` | swap one part to a donor variant, record z_old/z_new margins | 🔁 core of the driver |
| 30 species pairs (seed 42) · 32 renderer visual check · 34 main sweep | pick pairs, run the swap for every γ | 🔁 driver `--gammas --seeds`; writes `swap/funnybirds-mcbm-g*-s1.csv` |

**Why all of setup is gone from the notebook:** it needs a live Node renderer + GPU and takes
~90 min/model. That's exactly the heavy-compute-in-SLURM pattern you already use (`run_z_ordering*.sh`).
The notebook's job is to read the CSVs and plot. Nothing analytic was lost — only relocated.

---

## Load results — ref 36–38  →  ✏️ `load_swaps()` (nb03 §6)
Ref reads a hand-set folder of `fb_mcbm_z_ordering_gamma*_fix_v2.csv`. Curated globs
`swap/funnybirds-mcbm-g*-s1.csv` + the CBM CSV as reference. Same dataframes downstream.

---

## Per-γ tables & the "0.50" diagnostic — ref 39–44
| ref | what it does | curated |
|-----|--------------|---------|
| 39/43 `make_part_summary` | per-part frac_correct / frac_violations / mean_margin | ✅ folded into the **part×γ heatmap** (§6) + the **violation heatmap** (§7b) — a table becomes a figure |
| 40–42 "*every concept is exactly 50%??*" + fwd/bwd split | realizes the **combined** metric locks at 0.5 because it averages fwd (A←B) with bwd (B←A) | ✂️ **replaced.** The 0.5 was a metric artifact of averaging directions. Curated keeps **fwd/bwd separate from the start** (nb02 §6 plots them as separate bars), so the reader never hits the confusion. The "aha, it's 0.50" narrative is dropped on purpose. |
| 44 per-concept breakdown | frac_correct per (part, donor concept) | ✅ shown as **§7g top-20 worst concept slots** |

---

## Detailed per-γ figures — ref 45–60  (the "detail cluster")
| ref | what it does | curated |
|-----|--------------|---------|
| 46 `DETAIL_GAMMA` | pick one γ for the detail plots | ✏️ curated uses **first available γ** (`sorted(gammas)[0]`) |
| 47 `margin_norm = σ(z_new) − σ(z_old)` | rescale MCBM margin to CBM's [0,1] | ❌ **dropped.** Curated compares CBM vs MCBM on `ordering_correct` = **sign** of the margin, which is scale-free — the sigmoid renormalization is unnecessary (and `DECISIONS "γ=0≠CBM"` says the scales aren't meant to be forced equal). |
| 48 **z_old / z_new boxplots per part** | distribution of source vs donor concept per part | ✅ **this is the box-plot idiom curated §E now uses** — kept, and extended to the margin per part |
| 54 **tail margin histogram, 3×3 grid per variant** | one histogram of the tail margin per tail variant | ✏️ **generalized, not per-variant** — see the "histogram vs dots" note below |
| 55 tail concept **confusion matrix** (violations only, `is_anchoring`) | which tail slot fires on a mis-ordered swap | ✏️ ported to **nb02 §8** (all tail swaps, diagonal = correctly attributed). nb03 not yet (tail-only until the all-part swap re-run) |
| 56 source-species violation, **bars colored red/blue by median** | per-source-species tail failure | ✏️ **§7f, single color.** The red/blue-by-median split is the *exact* "we drew a random line dividing the dots" thing you flagged — dropped; the ranking carries it |
| 58 **margin vs p_cf_donor** (left) + **p_gt_donor vs p_cf_donor** (right) | downstream effect + the GT-intervention "ceiling" | ✏️❌ **split.** Left panel → **§7e**, de-tautologized (colored by visibility + a binned-mean trend, not red/blue by outcome). Right panel (**GT ceiling**) → **dropped**: minimal_cbm's y-head reads the *bottleneck*, not the concept logits, so the GT-intervention ceiling can't be computed here (documented) |
| 51–53, 60 good-vs-failing tail variants (side-by-side) | anecdotal 2-panel of a "best" and "worst" variant | ✂️ **replaced by §7g** (ranked top-20 violation bar) — same signal, denser, no cherry-pick |

---

## Cross-γ plots — ref 61–69
| ref | what it does | curated |
|-----|--------------|---------|
| 62–63 frac_correct & mean_margin **vs γ per part** | the cross-γ line plots | ✅ **§6d** (frac_correct vs γ) + **§7a** (mean margin vs γ) |
| 64 "why every cell 0.50 — species encoder signature" | markdown narrative of the 0.5 artifact | ✂️ dropped (we never average fwd+bwd) |
| 65 **frac_violations heatmap** part×γ | | ✅ **§7b** |
| 66–67 fwd/bwd accuracy **table + heatmaps** | the 0.50 decomposed into directions | ✏️ **partial gap.** nb02 shows fwd/bwd as bars (CBM); **nb03 currently shows the combined heatmap only.** Adding the fwd/bwd split heatmap to nb03 is a candidate "and more" (flagged below) |
| 68 **fwd_margin vs bwd_margin scatter** (anti-diagonal = pure species encoder) | clever single-point-per-(γ,part) diagnostic | ❌ **not ported.** One point per (γ,part) → very sparse, and the "anti-diagonal ⇒ species encoder" reading is subtle. Candidate to add if you want it |
| 69 **top-20 worst concepts per γ** (grid) | | ✅ **§7g** (single-γ; ref does a per-γ grid) |

---

## Occlusion / segmentation — ref 70–74  →  ✅ occlusion control + §7h
Ref: scatter `pixel_count_cf` vs margin + **violin by correct/violation** + all-vs-filtered frac_correct.
Curated: **occlusion control per γ** (tail vs grounded foot) + **§7h** (visibility-vs-margin scatter +
**visibility histogram by outcome**). The ref's `px≥threshold` filter = curated's visible-only gate
(`pixel_count_cf≥50` / `changed_frac>1e-3`). Violin→histogram-by-outcome (clearer counts).

---

## Distributions, summary, headline — ref 75–89
| ref | what it does | curated |
|-----|--------------|---------|
| 76 cross-γ z_new_orig/z_old_orig dists · 84 **pre/post-swap z overlay** | species-encoder signature | ✏️ **partly.** §7d (grounding-before-swap scatter) captures "donor low before swap"; the explicit **before-vs-after histogram overlay is a gap** (candidate to add) |
| 78 final summary table | | ✅ the heatmaps + `RESULTS.md` |
| 79–80 **fwd/bwd grounding heatmap — the headline** | the money figure | ✅ **§6 heatmap** + **§6e CBM-vs-MCBM** (combined `ordering_correct`; see the fwd/bwd note above) |
| 82 **IB compression vs swap response** | does compressing z buy grounding? | ✅ **§5** (deletion-adapted, since no live renderer in-notebook) + **§7a**; reason documented in §5 markdown |
| 85–87 **CBM row on the heatmap** | does the sigmoid fix grounding? | ✅ **§6e** |
| 88–89 **why γ=0 ≠ CBM**, z_old_orig dist MCBM vs CBM | | ✅ **§7i** (z distribution MCBM γ0 vs CBM) |

---

## Training-distribution confound — ref 90–91  →  ✏️ split across notebooks
Ref cell 91 computes *how often image-level relabeling (part < 5% of median pixels) would flip a
concept label*. Curated splits this:
- the **argument** (majority-vote standardization overwrites labels) → **nb04** on real CUB;
- the **FunnyBirds mechanism** (visible-only gate; tail hidden ~32% of the time) → nb02/nb03 gates;
- the **causal test** (actually relabel-then-retrain and re-measure) → **nb03rl** (pending pipeline).
This is also the **training-occlusion confound** markdown I added to nb02 §E and nb03.

---

## The one you asked about: histogram (ref 54) vs dots (curated §E)
- **Ref 54** = a **3×3 grid of histograms** of the tail margin, one panel per tail variant.
- I first replaced the per-swap **scatter** with a single **box plot** (a distribution, not a
  sign-colored cloud), because a scatter colored by the margin's sign is tautological.
- You said the dots were gone, so I **restored the dots** — but colored by **swap direction
  (fwd/bwd)**, which is independent of the margin's sign, and drew the **box on top**. So §E now
  has dots *and* a distribution summary.
- **Why not the ref's 3×3 per-variant histogram grid?** Two reasons: (1) with only 2 γ collected,
  a per-variant split gives tiny-n panels that read as noise; (2) the per-variant "which slot
  fails" signal is already carried, denser, by **§7g** (ranked violation) and the **confusion
  matrix** (nb02 §8). The **distribution** the histogram was showing is honored by the box.
- It's a one-line add if you want the granular per-variant histogram back once the full γ sweep
  gives enough samples per panel — say the word.
