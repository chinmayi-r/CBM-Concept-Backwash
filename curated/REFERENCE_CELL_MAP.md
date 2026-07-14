# Reference ↔ curated — every cell

Part A: every cell of `funnybird_notebooks/fb_mcbm_renderer_swap.ipynb` (92) **after the
setup block (0–35)**, what it does, and its curated status.
Part B: every curated `03_funnybirds_mcbm.ipynb` cell that has **no reference counterpart**.

Status: ✅ ported · ✏️ modified · ✂️ replaced with a cleaner equivalent · ❌ dropped ·
🔁 moved to the SLURM driver · ⚠️ GAP (in ref, not yet in curated).

---

## Setup — ref 0–35  →  🔁 `analysis/z_ordering_swap.py` + `train/renderer_swap.slurm`
Imports, paths, data-loading helpers, concept/species maps, render params, renderer setup,
MCBM loading (`load_mcbm_weights` ✂️→ curated `load_model` on official minimal_cbm ckpts),
inference fns, part-swap helpers, `z_ordering_record`, species pairs, the main sweep. All
relocated: needs a live renderer + GPU, ~90 min/model. Notebook reads the CSVs it writes.

---

## Part A — reference cells 36–91

| ref | kind | what it does | curated |
|----|----|----|----|
| 36 | md | §12 header "Load all gamma results" | — |
| 37 | md | how to point at the SLURM CSVs on Adroit | ✂️ replaced by `load_swaps()` + `[pending]` hints |
| 38 | code | load `fb_mcbm_z_ordering_gamma{g}.csv` → `gamma_dfs` | ✏️ `load_swaps()` globs `swap/funnybirds-mcbm-g*-s1.csv` |
| 39 | md | §13 header "Per-gamma summary tables" | — |
| 40 | md | "*first look: every concept 50%??*" | ✂️ dropped — the 0.5 was an averaging artifact |
| 41 | code | prints combined vs fwd/bwd split to expose the 0.5 | ✂️ dropped — curated never averages fwd+bwd |
| 42 | md | explains fwd/bwd cancel to 0.5 | ✂️ dropped (same reason) |
| 43 | code | `make_part_summary`: frac_correct/violations/margin per part | ✅ folded into the §6 + §7b heatmaps (table→figure) |
| 44 | code | per-(part,concept) breakdown | ✅ → §7g top-20 worst concept slots |
| 45 | md | §14 header "Detailed per-gamma analysis" | — |
| 46 | code | `DETAIL_GAMMA=0.1`; print overall/worst | ✏️ curated uses first-available γ; summary via heatmap |
| 47 | code | `margin_norm = σ(z_new)−σ(z_old)` (rescale to CBM) | ❌ dropped — compare on the **sign** (scale-free) |
| 48 | code | z_old_orig & z_new_orig **boxplots per part** | ✅ the box idiom curated §E now uses; + §7d |
| 49 | code | per-part scatter z_old_orig vs margin, **colored by outcome** | ❌ dropped — outcome-color = margin sign = tautological |
| 50 | code | tail scatter z_new_orig vs margin, colored by outcome | ❌ dropped — same tautology (§7d covers z_new_orig) |
| 51 | md | "Good vs failing tail variants" | — |
| 52 | code | split tail variants good/bad by **median** frac_correct | ✂️ replaced by §7g ranked violation (no median split) |
| 53 | code | tail scatter z_old_orig vs z_new_orig, colored by outcome | ✏️ → §7d (same axes, colored by **part**, not outcome) |
| 54 | code | tail margin **3×3 histogram grid** (per variant) | ✏️ generalized to §E box; per-variant grid not ported (tiny-n; §7g carries "which variant") |
| 55 | code | tail concept **confusion matrix** (violations, is_anchoring) | ✏️ ported to **nb02 §8**; nb03 pending all-part swap re-run |
| 56 | code | source-species violation bars, **red/blue by median** | ✏️ → §7f (single color; median split dropped — the "random dividing line" you flagged) |
| 57 | code | beak/wing violation by source species | ✂️ mostly dropped — tail is the story; §7f is the tail version |
| 58 | code | margin vs p_cf_donor (left) + **p_gt_donor vs p_cf_donor** (right, GT ceiling) | ✏️❌ left→§7e (de-taut); right **dropped** — minimal_cbm y-head reads bottleneck, GT ceiling can't compute |
| 59 | md | z-scale note MCBM vs CBM | ✅ in the Terms cell + DECISIONS |
| 60 | code | good-vs-failing donor variant 2-panel (anecdotal) | ✂️ replaced by §7g (ranked, no cherry-pick) |
| 61 | md | §15 header "Cross-gamma comparison plots" | — |
| 62 | code | table frac_correct per part per γ + pivot | ✅ → the §6 heatmap (pivot as heatmap) |
| 63 | code | line plots frac_correct vs γ + mean_margin vs γ per part | ✅ → §6d (frac_correct) + §7a (mean margin) |
| 64 | md | "why every cell 0.50 — species encoder signature" | ✂️ dropped (never combine fwd/bwd) |
| 65 | code | frac_violations **heatmap** part×γ | ✅ → §7b |
| 66 | code | fwd/bwd accuracy **table** (print) | ⚠️ GAP — nb03 doesn't print the fwd/bwd table |
| 67 | code | fwd & bwd accuracy **heatmaps** (2-panel) | ⚠️ GAP — nb02 has fwd/bwd bars, nb03 shows combined only |
| 68 | code | **fwd_margin vs bwd_margin scatter** (anti-diag = species encoder) | ❌ not ported — 1 sparse point/(γ,part); slick but subtle |
| 69 | code | top-20 worst concepts **per γ grid** | ✅ → §7g (single-γ) |
| 70 | md | §16 header "Occlusion analysis" | — |
| 71 | code | pixel_count **histogram per part** | ✅ → nb02 §8e; nb03 §7h has visibility |
| 72 | code | tail visibility vs margin scatter + **violin by outcome** | ✏️ → §7h (scatter + **histogram** by outcome) |
| 73 | code | before/after occlusion-filter table, all γ | ✅ → occlusion control cell (§ tail vs foot per γ) |
| 74 | code | plot frac_correct all vs filtered (tail) vs γ | ✅ → occlusion control cell |
| 75 | md | §17 header "cross-gamma z distributions" | — |
| 76 | code | z_new_orig & z_old_orig **boxplots per γ** (tail) | ✏️ partial → §7d/§7i; per-γ tail z_new_orig box is a candidate |
| 77 | md | §18 header "Final summary table" | — |
| 78 | code | print full summary table | ✅ → the heatmaps + `RESULTS.md` |
| 79 | md | §19 header "Fwd/bwd grounding heatmap — headline" | — |
| 80 | code | **fwd/bwd grounding heatmap** (gold-border cells) | ⚠️ GAP — curated §6 headline is **combined**, not fwd/bwd |
| 81 | md | §20 header "IB compression vs swap response" | — |
| 82 | code | z_old_orig std & z_new std vs γ + ratio | ✏️ → §5 (deletion-adapted); true swap version now possible |
| 83 | md | §21 header "pre/post swap z distributions" | — |
| 84 | code | pre (z_new_orig) vs post (z_new) **histograms per γ** | ✏️ partial → §7d scatter captures "pre low"; explicit before/after overlay is a candidate |
| 85 | md | §22 header "CBM comparison" | — |
| 86 | code | load `fb_cbm_z_ordering_v2.csv` → `cbm_df` | ✅ → `load_swaps()` loads `CB` (`funnybirds-cbm-s1.csv`) |
| 87 | code | extended **fwd/bwd** heatmap: all γ + CBM row | ✏️ → §6e is **combined** CBM-vs-MCBM; fwd/bwd version = the §80 GAP |
| 88 | md | §23 "why γ=0 ≠ CBM" (sigmoid vs linear table) | ✅ documented in DECISIONS + §7i caption |
| 89 | code | z_old_orig std MCBM γ's vs CBM + distribution | ✅ → §7i (z distribution MCBM γ0 vs CBM) |
| 90 | md | §24 "training distribution confound" | ✅ → training-occlusion confound markdown (nb02/03) + nb04 |
| 91 | code | pixel-count **relabel-flip** analysis (part<5% median) | ✏️ → nb03rl (relabel + retrain) + nb04 (CUB majority-vote) |

**Reference figures NOT yet in curated (the honest list):** ref 66/67/80/87 fwd/bwd split
(table + heatmaps) · ref 68 fwd/bwd margin scatter · ref 54 per-variant histogram grid ·
ref 76/84 per-γ z before/after overlay. The rest is ported, replaced, or deliberately dropped.

---

## Part B — curated nb03 cells with NO reference counterpart (added for rigor/clarity)

| curated | what it does | why it exists (not in ref) |
|----|----|----|
| **Terms** cell | glossary: minimality vs γ, retained_frac, margin, ordering_correct, … | readers (you) kept hitting the vocabulary; ref never defines it |
| **§1** overall retained_frac vs γ | **deletion**-based overall backwash vs γ | renderer-free metric available for all 6 γ; ref's overall was swap-only |
| **§2** per-part retained_frac vs γ | **deletion** per-part vs γ (±seed band) | the renderer-free per-part signal — runs without the expensive swap |
| **§3** species-probe vs γ | can you recover **species** from the concept vector? (chance 1/50) | **curated-original mechanism test** — directly measures "the bottleneck is a class code"; not in ref |
| **§4** "did γ bite?" control | rep_loss / mean\|z\| / task+concept acc vs γ | **curated-original rigor guard** — flat retention only refutes minimality if γ actually moved `z` (DECISIONS §D.5) |
| **§5** IB-compression vs grounding | z-std vs tail retention (deletion) | adapts ref §20 to the renderer-free signal + states the caveat |
| occlusion-control (per γ) | tail vs **grounded foot** visible-only vs γ | ref filtered tail alone; curated adds the grounded reference so "low" means "low vs a part that works" |
| **training-occlusion confound** md | test-time occlusion ≠ training-time occlusion; γ can't fix it; 03rl disentangles | **curated-original** — the deeper confound + the causal test; ref §24 only names the label correlation |
| objection-first captions (§E, §7e, §7f…) | each 📊 states the alternative reading before concluding | **curated-original discipline** — the "be a critic" pass |
| "How retained_frac is read as backwash" | naming: metric is `retained_frac`, backwash is the interpretation | **curated-original naming rule** — never label an axis "backwash" |
| **Takeaway** | lock only with ≥3 seeds + bounded-CI (DECISIONS §D.5) | curated-original stopping rule |

Plus notebooks with **no ref-MCBM analogue at all**: **nb04** (raw-CUB majority-vote analysis)
and **nb03rl** (relabel→retrain→swap causal test) — both new axes, not in the reference.
