# Notebook03 loss-engineering implementation

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
