# Standard MCBM reports: locked roadmap for notebooks 03 and 06

This file is the implementation contract for the standard (non-RLv2) MCBM
stage. Notebook 03 is completed and reviewed before notebook 06 is interpreted.
RLv2 remains a later causal follow-up.

## Scientific question

Notebook 02 showed that a standard FunnyBird CBM can react to an inserted part
yet retain a stronger score for the old source concept. MCBM adds a minimality
penalty. We ask whether that penalty merely compresses the representation, or
whether it changes *which pixels determine each concept*.

For concept `j`, the implementation emits an internal scalar slot `h_j` (called
`z` in the model code). A learned concept head converts that slot to the raw
concept logit used here:

`z_j = concept_head_j(h_j)`.

The class head reads the vector of internal slots. MCBM adds

`L_rep = sum_j 0.2 mean((h_j - (6 c_j - 3))^2)`

and trains with

`L = L_species + beta L_concept + gamma L_rep`.

Increasing `gamma` therefore predicts less within-label variation in `h_j`. It
does **not** mathematically require `h_j` or `z_j` to use the pixels of part `j`.
A species/body shortcut can output the correct `+3` or `-3` target and satisfy
the penalty.

## Pre-registered predictions

1. If the penalty is active, distance from the `+3/-3` target and within-label
   spread of `h` should fall as `gamma` increases, without destroying ordinary
   task/concept performance.
2. If minimality repairs grounding, on the *same fixed FunnyBird renders*:
   final donor-minus-source margin `m_cf` should rise, the controlled backwash
   rate `P(response_delta>0 and m_cf<0)` should fall, and exact inserted-value
   recognition should improve.
3. Compression alone is not repair. If `h` compresses but tail remains
   source-negative, or its donorward response weakens, minimality has not solved
   the input-source problem.
4. A gamma trend is provisional when only seed 1 has fixed-render replay. Model
   health may use all available seeds; causal gamma comparisons must show their
   actual seed count.

## Notebook 03: FunnyBird standard MCBM

Every output is introduced using question -> variables and prediction -> how to
read -> numbered figure -> literal observation -> alternatives -> discriminating
test -> limited conclusion -> next question.

| Figure | Required question and output | Claim protected |
|---|---|---|
| 1 | Data/checkpoint/fixed-render inventory by gamma and seed | We know exactly what was compared |
| 2 | MCBM implementation and compression: representation target error, within-label `h` spread, task accuracy, concept balanced accuracy/recall | `gamma` changed the intended quantity without simply breaking the model |
| 3 | Standard CBM baseline beside MCBM `gamma=0`, using identical predicates and fixed renders | Architecture/minimality comparison starts from the established discovery |
| 4 | `response_delta = m_cf-m_orig` distribution and positive-response rate, every part x gamma | Did the inserted pixels move the scores donorward? |
| 5 | Final margin `m_cf` and controlled backwash rate `P(response_delta>0,m_cf<0)`, every part x gamma | Did the donor actually finish above the old source? |
| 6 | Forward and backward rates separately | Pooled directions are not hiding cancellation/bookkeeping error |
| 7 | Visible-pixel strata using the same thresholds for every gamma | Occlusion does not alone explain the gamma result |
| 8 | Exact inserted-value confusion for every part and gamma | A two-slot margin is not hiding collapse onto another value |
| 9 | Source/donor value difficulty, then source-species residual after matching those values | Variant difficulty is accounted for before species/body context |
| 10 | Raw-logit species decoding versus processed-label control by part and gamma | Learned concept scores contain within-bucket species information |
| 11 | Matched species recall-gap diagnostic with balanced accuracy and valid pairing | Species-dependent model health/representation dependence, not swap proof |
| 12 | Downstream donor-species probability versus `m_cf` | Whether the grounding failure materially changes the final class output |
| 13 | Seed coverage and available seed-level replication | Separates supported findings from seed-1 gamma trends |
| 14 | Aligned summary of compression, response, final failure, exact-value error, and health | Minimality cannot be credited merely because one metric shrank |

The primary success metric is the controlled backwash predicate from notebook
02, not `ordering_correct` alone:

`response_delta > 0 and m_cf < 0`.

Example: `m_orig=-20`, `m_cf=-5` gives `response_delta=+15`. The model saw and
responded to the donor pixels, but the old source still wins by five logit units.

## Notebook 06: CUB/CUB70 standard MCBM

Notebook 06 begins only after Figures 1--14 in notebook 03 have been inspected.
It freezes the questions, not necessarily the mechanical operations.

| FunnyBird question | CUB operation | Boundary |
|---|---|---|
| Did gamma preserve ordinary model health? | Same operation on exact raw concept logits | Directly comparable |
| Did an inserted part move donor vs source? | Not available | No donor/source margin is invented |
| Does seeing the named region raise its score? | `visibility_effect_j = E[z|c=1,v=1]-E[z|c=1,v=0]` | Observational; photographs differ |
| Does context retain the concept when the region is absent? | `context_gap_j = E[z|c=1,v=0]-E[z|c=0,v=0]` | Observational; mask absence mixes occlusion and annotation failure |
| Is visibility/area sufficient? | Released-mask visibility, area, and bilateral counts | Only 11 masks; coverage denominators retained |
| Are exact values/species support important? | Exact 112 concepts, alternatives, positive-image support, species support | Same question, different data generation |
| Does source species organize residual scores? | Match exact concept, label, visibility/area, then estimate species residual | Association, not independent species manipulation |
| Is recall species-dependent? | CUB-positive/negative-count matched species pairs and raw-z companion | Uses the recall-v4 matching refinement |

Notebook 06 must include: all 11 mask names, all eight coarse groups, exact-mask
coverage, model-health guards, raw-z gamma curves, matched recall, contributor
accounting, and a final FunnyBird/CUB evidence table. CUB results use `leg`, never
`foot`, except in an explicitly labelled cross-dataset mapping.

## Definition of done

Notebook 03 is done only when all important executed figures have been displayed
in chat and reviewed literally, seed limitations are visible in the figures, and
no old invalid renderer output is loaded. Notebook 06 is done only after the
same review plus explicit `same operation`, `weaker approximation`, or `not
available` labels for every FunnyBird question. Neither notebook may conclude
that identified contributors explain the residual completely unless the measured
residual actually approaches zero under a valid test.
