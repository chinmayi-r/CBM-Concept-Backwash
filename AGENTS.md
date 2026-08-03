# Canonical research contract

This file is the single current operating contract for this repository. Chat
memory is not authoritative. The large files under `curated/` are historical
provenance; read a specific one only when its evidence or implementation is
needed. Do not load all old handoffs before starting work.

## Research question and order

We are testing whether a concept bottleneck predicts a named concept from its
own part pixels or from species/body context (concept backwash).

Keep this order:

`non-RL data -> standard CBM discovery -> standard MCBM minimality -> RLv2 cause test -> CUB/CUB70`

RL never replaces the standard-CBM discovery story. CUB follows FunnyBird's
questions but is allowed to give a different answer.

## Immediate task

Rewrite the standard-CBM comparison in:

- `curated/notebooks/02_funnybirds_cbm.ipynb`
- `curated/notebooks/05_cub_cbm.ipynb`

Do not submit a Slurm job or start MCBM/RL work for this task. First make the two
CBM notebooks understandable and directly comparable. Move failed deletion,
patch, paste, and forecasting material to a clearly labelled methods appendix;
it must not interrupt the discovery chain.

## Non-negotiable variable rules

For image `i`, exact concept `j`, and available region mask `g`:

- `c_ij`: processed binary concept label.
- `v_ig`: available mask is visible.
- `a_ig`: visible mask area.
- `z_ij`: raw concept logit. This is the primary model quantity.
- `p_ij = sigmoid(z_ij)`: use only for explicitly thresholded performance.
- `c_hat_ij = 1[z_ij > 0]`.

FunnyBird controlled swap:

- `m_i = z_donor,cf - z_source,cf` is the final margin.
- `response_delta_i = m_cf - m_orig` measures donorward movement.
- Backwash candidate: `response_delta_i > 0` but `m_i < 0`.

CUB natural visibility is not a swap. Its closest observational quantities are:

- `visibility_effect_j = E[z_ij | c_ij=1,v_ig=1] - E[z_ij | c_ij=1,v_ig=0]`.
- `context_gap_j = E[z_ij | c_ij=1,v_ig=0] - E[z_ij | c_ij=0,v_ig=0]`.

Never call either CUB quantity a donor/source margin.

Formal model-health guard for exact concept `j`:

- `spread_j = Q95(z_ij) - Q05(z_ij)`.
- `label_separation_j = median(z_ij | c_ij=1) - median(z_ij | c_ij=0)`.
- `positive_recall_j = P(z_ij>0 | c_ij=1)` is a health statistic, not grounding
  evidence.
- Call an output exactly collapsed only when its raw-`z` spread is zero within a
  declared numerical tolerance. Do not diagnose collapse from rounded sigmoid
  probabilities.

## Dataset names must never be mixed

FunnyBird parts: `tail, wing, beak, foot, eye`.

CUB masks: `head, left_eye, right_eye, beak, neck, body, left_wing,
right_wing, left_leg, right_leg, tail`.

CUB coarse groups: `head, eye, beak, neck, body, wing, leg, tail`.

- A CUB result is always labelled `leg`, never `foot`.
- `foot -> leg` is allowed only inside an explicitly titled cross-dataset mapping.
- `body` is CUB-only because FunnyBird has no matched body intervention.
- A plot titled FunnyBird calibration may use `foot`; it belongs in the methods
  appendix, not in the middle of the CUB discovery story.

## Standard-CBM story to implement

For both datasets, proceed in this order:

1. Define the model and every variable.
2. Inventory images, species, exact concepts, and masks.
3. Show species/concept structure before model behavior.
4. Count positive-label/mask conflict for every exact concept.
5. Verify model health using raw `z`, label separation, balanced accuracy, and
   positive recall.
6. Test whether concept scores use the named region.
7. Test visibility, area, and bilateral-mask alternatives.
8. Restore the original CUB species-matched recall-gap analysis and add a raw-`z`
   companion.
9. Account for proposed contributors without pretending association is causation:
   visibility/occlusion, conflicting labels, exact variant/concept difficulty,
   number of alternatives/species support, then source species.
10. Show what residual remains. Do not promise that identified mechanisms sum to
    zero.
11. End with a direct FunnyBird/CUB evidence table and the causal boundary.

FunnyBird uses the validated controlled swap as primary evidence. CUB uses the
strongest available observational approximation and must not be described as
renderer-quality causal proof.

## Plot and notebook rules

- Use raw `z` for grounding. Probability is allowed only when the question is
  classification, recall, or another explicitly thresholded rate.
- A data fraction such as label/mask conflict is a rate, not a model probability;
  name it in plain language.
- No meaningless horizontal jitter, unlabeled spaghetti plots, or unexplained
  summary columns.
- Every output cell produces one numbered figure or one clearly numbered example
  set. Put the number/title in markdown before the output and inside image grids.
- For every important figure, display it in chat before interpretation and use:
  question -> variables/prediction -> figure -> literal observation ->
  alternatives -> discriminating test -> limited conclusion -> next question.
- Review every important figure, not a sample.
- Use seed-level replication for uncertainty; do not use reused image/species rows
  as independent error bars.

## Current completion matrix

This matrix is the research-level state. Live Slurm state must be refreshed on
Adroit before any cluster decision.

| Work item | Standard training | Matched RLv2 training | Valid fixed-render evaluation | Notebook status / next step |
|---|---|---|---|---|
| FunnyBird CBM | seeds 1-3 available | seed 1 complete; seed 2 failed; seed 3 dependency failed | seed-1 standard/RLv2 complete | Rewrite notebook 02 first; later rerun missing RL seeds |
| FunnyBird MCBM gamma 0 | seeds 1-3 available | seeds 1-3 complete | seed-1 standard/RLv2 complete | Later stage; do not replace CBM story |
| FunnyBird MCBM gamma 0.1 | seeds 1-3 available | seeds 1-3 complete | seed-1 standard/RLv2 complete | Later stage |
| FunnyBird MCBM gamma 0.3 | seed 1 available | seed 1 complete | broad replay status must be reconciled | Later all-gamma context |
| FunnyBird MCBM gamma 1 | seed 1 available | seed 1 complete | broad replay status must be reconciled | Later all-gamma context |
| FunnyBird MCBM gamma 3 | seed 1 available | seed 1 complete | broad replay status must be reconciled | Later all-gamma context |
| FunnyBird MCBM gamma 5 | seed 1 available | retry completed | broad replay status must be reconciled | Later all-gamma context |
| CUB70 CBM | seed-1 epoch-100 export available | not yet an accepted intervention | no valid renderer-equivalent edit | Rewrite notebook 05 now using raw `z` and recall |
| Full-CUB CBM | seed-1 epoch-100 export available | not applicable | natural-image comparison only | Use only as a clearly labelled same-image guard |
| CUB MCBM | outputs/checkpoints require reconciliation | not applicable | not started | Notebook 06 only after standard CUB CBM story is fixed |

Known failed/quarantined methods are evidence about method limitations, not proof
that CUB has no backwash: fixed-cache v1 black renders, reciprocal mask deletion,
randomized patch V1/V2 calibration, and the CUB beak/tail paste pilot.

## Cluster safety

Before recommending any cluster action, obtain a fresh full `squeue -u "$USER"`
and relevant `sacct`. Never infer a job payload from its short name: inspect
`scontrol show job -dd` and, when needed, `scontrol write batch_script`. Do not
release old held jobs without proving they are current. Do not restart completed
work. Record accepted new job state in this file's completion matrix.

