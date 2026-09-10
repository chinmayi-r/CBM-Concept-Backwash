# Notebook 05 rebuild contract

This is the construction and review checklist for the official-Koh CUB70
chapter. It records the reasoning recovered immediately before the CUB70 work
so a later context change cannot silently replace the scientific question.

## What the preceding FunnyBird chapters taught us

1. **Standard CBM discovery:** controlled one-part swaps directly measure
   `response_delta = m_cf - m_orig` and the event
   `response_delta > 0 and m_cf < 0`. This is the strongest backwash evidence.
2. **Separate four questions:** species information being recoverable from raw
   scores, the saved species head using score magnitudes, signed source evidence
   after a swap, and grounding in named pixels are different measurements.
3. **Wing is the control:** a part can encode substantial species information
   and still respond well to its inserted pixels. Decodability alone is not
   backwash and not a model-health verdict.
4. **RLv2 is a cause test, not the discovery model:** removing one
   label/visibility conflict reduced some tail failures, but it did not remove
   all species leakage or make every measurement improve. Joint task loss still
   rewards useful context.
5. **MCBM is a loss experiment:** compression can remove nuisance variation and
   useful intervention response together. A smaller or tidier representation is
   not automatically better grounded.
6. **Good reasoning permits failed explanations:** a proposed contributor gets
   credit only when its predeclared held-out or controlled prediction works.
   Different causes need not add to 100%, and a residual is not automatically a
   newly discovered mechanism.

## What changes in CUB70

- The model is the accepted official Koh Joint ResNet-50 Standard seed-1 model:
  image -> 112 raw concept logits `z` -> one linear 112-to-70 species head.
- CUB70 has natural photographs and released masks, but no accepted same-scene
  donor-part swap. Therefore it has no `m_orig`, `m_cf`, `response_delta`, or
  controlled CUB backwash rate.
- The closest observations are
  `visibility_effect = mean(z|c=1,v=1)-mean(z|c=1,v=0)` and
  `context_gap = mean(z|c=1,v=0)-mean(z|c=0,v=0)`.
- An absent released mask can mean real occlusion, pose, a coarse mask, or a
  missing annotation. Every conclusion must preserve that ambiguity.
- RLv2 is not assumed. A later CUB70 relabel/retrain is justified only if the
  mask-label conflict is credible, aligns with the accepted model's behavior,
  and the mask identity/coverage is adequate for training labels.

## Matched recall: what it can and cannot do

- For each exact concept, compare species that each have at least three positive
  and three negative images. Match both counts and bootstrap both label classes.
- Reuse the saved concept decision `z>0`; do not train a new recall classifier.
- Report positive-recall gap, balanced-accuracy gap, and a raw-`z` companion.
- First calibrate the identical statistic on FunnyBird exact concepts against
  their known controlled-swap event rates.
- It earns only `PROVISIONAL ORDINAL WARNING PROXY` if the association is
  positive overall, after part centering, and in at least four of five
  leave-one-part-out checks. Otherwise report
  `METHOD NOT CALIBRATED AS A BACKWASH PROXY`.
- Even if it passes, it ranks warnings. It does not estimate a CUB backwash rate
  and does not replace a controlled swap.

## Reuse rules

- Reuse tested plotting, matching, and mask-join mechanics only after checking
  their model identity, columns, denominators, grouping, and held-out logic.
- Never load `curated_data/cub70_eval/cub70-cbm-s1.parquet`; it was produced by
  the wrong `minimal_cbm` CBM implementation.
- Never use MCBM notation `h` or `q(h)` in Notebook 05.
- Never reuse a numerical result paragraph from the old CUB70 render.
- Do not load the legacy full-CUB export while the official full-CUB Koh result
  remains incomplete.

## Figure contract

Before every figure:

- state the question and prediction;
- define every quantity and formula;
- name the exact input population and accepted model;
- say whether a new diagnostic is fitted or the saved model is reused;
- define panels, axes, colors, groups, denominators, exclusions, and direction;
- give a numerical example when a sign or formula can be misunderstood.

After every figure:

- state the literal observed result;
- say exactly what it supports;
- give a plausible alternative;
- name the test that distinguishes that alternative;
- state a limited conclusion and the next question.

Every graph must print its complete source table. Unsupported quantities stay
blank rather than becoming zero. `leg` is used for CUB; `foot` appears only in
an explicitly labelled cross-dataset mapping. Image rows remain grouped in all
held-out analyses. Reused rows do not create model-level uncertainty; that
requires independently trained seeds.

## Chapter order and definition of done

1. Model and symbols.
2. Population and mask coverage.
3. Species/concept structure before model behavior.
4. Positive-label/mask conflict.
5. Raw-`z` health guard.
6. Species information recoverable from labels and raw scores.
7. Natural visibility, hidden context, area, and bilateral alternatives.
8. Matched recall/accuracy/raw-score species gaps, with FunnyBird calibration.
9. Held-out contributor accounting and species residuals.
10. Exact-concept and anatomical-group synthesis.
11. Rule-selected photographs with all masks.
12. Direct FunnyBird/CUB capability table and causal boundary.

Source implementation is complete only when all generated code cells compile
and the runner checks every required artifact before execution. Scientific
interpretation is complete only after Adroit executes the official-Koh report
and every current figure and printed table has been reviewed. Until then,
result slots must say `INCOMPLETE`, not prewrite an expected observation.
