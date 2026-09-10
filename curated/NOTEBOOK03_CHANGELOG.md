# Notebook03 loss-engineering implementation

## 2026-09-09 caption and RLv2 parity revision

All Notebook 03 result tables now state their population, units, denominator,
and reading rule before display. The pathway section explicitly defines the
learned numerical scale of `q_j(h_j)` and reports the cases where calibrated
`h` movement rises while raw-`z` movement shrinks, preventing reader rescaling
from being mistaken for a weaker image response.

Notebook 03rl is now built from the exact current Notebook 03 cell template.
The RLv2 runner uses explicit `funnybirds-mcbm-rlv2matched` checkpoint/config
names, `swap_fixed_v3_matched`, independent replay/table roots, resumable
per-gamma diagnostics, and the same pathway equations. It embeds the executed
Standard-MCBM result before each RLv2 counterpart and prints one complete
matched-difference table. No scientific training or Slurm work was added.

## 2026-09-09 complete-render review correction

The first complete real-data render at commit `59c5b09` executed 46 cells and
produced 29 images without notebook errors. A review of every current image
found presentation and accounting gaps rather than failed model inference.
The current revision replaces placeholder result prose with the observed
part-by-part numbers; adds starting and final raw-logit margins to the pathway
figure; prints the complete all-fronts table separately for tail, wing, beak,
foot, and eye; and expands the final answer across all six MCBM gammas.

The accepted Standard swap CSV contains a historical `response_delta` column,
whereas this chapter's declared definition is `m_cf - m_orig`. Six of 1,000
tail rows change strict event category under those two accountings: the copied
Notebook 02 panel reports 50.2% controlled backwash and 8.1% no donorward
movement, while the formula-derived comparison reports 50.8% and 7.5%.
Predictions, margins, checkpoints, and swap images are unchanged. The original
Standard panel remains visible as provenance; a compact audit table discloses
the disagreement, and all new comparisons use the declared formula.

No training, checkpoint change, swap regeneration, or scientific job is part
of this revision. It still requires one real-data rerender and inspection of
the altered report before the presentation correction is accepted.

## 2026-09-09 mechanism-first rebuild

The current 46-cell builder supersedes the mechanical 223-cell parity report.
It separates Koh→MCBM gamma 0 from gamma 0→positive gamma, displays exact
Notebook 02 outputs only where they are the clearest reference, and summarizes
all MCBM gammas together for matched metrics. It adds calibrated `h`→`q(h)`
localization, exact donor/source/third outcomes, per-value contributors, matched
ordinary health, and the original-restored off-target saved-head intervention.
Every Standard tag remains accounted in a rendered ledger; Figure 9b is retired
and replaced by a complete all-fronts table.

`build_mcbm_parity.py` is now only the Notebook 02 inventory/source validator.
Its obsolete sixfold cell-adaptation engine was deleted. The synthetic test now
compiles and checks the current notebook rather than exercising that removed
report. The runner builds the v3 pathway tables before notebook execution.

The file/method inventory below describes the earlier implementation and is
retained as provenance; where it conflicts with this section, this section is
current.

## Scope

Non-RLv2 official FunnyBird MCBM, gammas 0/0.1/0.3/1/3/5, with accepted Koh
Standard as the discovery baseline. No scientific model training or job submission.
Notebook02 user edits are not changed or committed by this implementation.

## Files and methods changed

| File | Methods / sections | Purpose |
|---|---|---|
| `analysis/build_03_standard_mcbm_report.py` | `md`, `code`, `review`, generated health cell, main build; architecture Markdown | Valid cell IDs, explicit h/z/loss/auxiliary definitions, shape-safe task accuracy, compile every generated cell; remove obsolete duplicate analyses and prewritten results. |
| `analysis/build_mcbm_parity.py` | `find`, `adapt`, `assemble`; `SETUP`, information, erasure, gradient and decision cells; per-figure Markdown | Account for every Standard visual/table; copy its construction and detailed methods, display baseline then each gamma, adapt only model-specific computations, keep secondary correlations in appendix. |
| `analysis/mcbm_loss_report.py` | `checkpoint_tag`, `array`, `softmax`, `task_head`, `score_reference`, `replacement_use` | Explicit nonlinear saved-head replay and fivefold label-mean replacement; absent/present N, mean, SD for both h and z. |
| `analysis/mcbm_loss_report.py` | `sha256`, `replay_counterfactual_h`, `off_target_erasure`, `loss_gradient_audit`, `preflight` | Frozen CUDA replay with accepted RGB/scalar/probability checks; off-target h erasure; exact upstream loss gradients without parameter updates; input/config inventory. |
| `analysis/funnybird_followup_diagnostics.py` | `grouped_predictions`, `conditional_information` | Constant training-fold outcomes use their constant prediction; optional subset progress reports keep long fits visible. Fitting rules unchanged. |
| `analysis/test_mcbm_loss_report.py` | `fixture`, `main` | Notebook schema/compilation and synthetic runtime tests, actual upstream loss-gradient check, short-batch scaling, immutable parameters, zero-effect erasure and constant-event edge case. |
| `notebooks/03_funnybirds_mcbm.ipynb` | generated Markdown and Python | Full loss-engineering chapter; no copied old observations presented as new results. |
| `notebooks/run_03_standard_mcbm_report.sh` | top-level stages | Explain the work, check real inputs/CUDA/baseline before fitting, stream progress, execute and export only after successful preceding stages. |
| `NOTEBOOK_03_06_MCBM_ROADMAP.md` | loss-story/reporting directions and current implementation status | Preserve the research questions, reporting requirements and honest execution boundary. |

## Parity and scientific boundaries

- 24 Standard figure/example/table constructions are explicitly enumerated.
  Four renderer/data-only constructions are shared rather than repeated six times.
  Model-dependent constructions are repeated per gamma. New Standard output cells
  cannot silently go unaccounted for.
- Three Standard result-writing cells are replaced by current MCBM data tables and
  review rules. Standard setup/provenance are replaced by the MCBM audit/ledger.
- Ordinary Koh and MCBM exports have different image counts. Raw magnitudes and
  conditional information across these models are not matched-image loss effects.
- Frozen MCBM erasure changes h, not concept z directly. The old/donor slots stay
  fixed. Its nonlinear class-gap change is not Koh's additive W formula and does
  not show the species head causing an upstream concept-margin error.
- Current stored configurations are printed, not claimed to establish historical
  training-time source identity. Recipe differences and uncontrolled old model
  initialization remain limitations.
- Shared training-view conflict counts are descriptors, not proof of historical
  MCBM training population identity or causal contribution percentages.
- Final-checkpoint gradients are local, noise-draw-dependent diagnostics. They
  are not observed training trajectories. Proposed new losses remain proposals.

## Verification and remaining work

Local synthetic run executed all model-dependent Standard plot constructions
except real CUDA image loading, including conditional-information fits and
grouped predictive audits. Nonlinear erasure mathematics was tested separately.
The additional official-loss gradient test checked the analytic representation
gradient, an incomplete final batch, and unchanged trained parameters.
Shell syntax and generated notebook schema/compilation are checked locally.

Real checkpoints/images live on Adroit. Scientific execution, frozen-image replay
against those bytes, and review of every actual rendered output remain necessary.
Do not describe the results as accepted merely because implementation tests pass.
