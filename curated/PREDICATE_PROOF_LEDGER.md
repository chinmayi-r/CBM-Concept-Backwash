# Predicate-first proof ledger

This file states what must be true before a final claim is allowed. It applies
the same logic to FunnyBird and CUB while keeping their different intervention
strengths explicit.

## The event we are trying to prove

For a source bird with source part value `s` and an inserted donor value `d`:

- `before_margin = z_donor(original) - z_source(original)`
- `after_margin = z_donor(swapped) - z_source(swapped)`
- `donor_response = after_margin - before_margin`

A **candidate backwash event** requires both:

1. `donor_response > 0`: the inserted donor pixels moved the model toward the
   donor concept;
2. `after_margin < 0`: even after that response, the old source concept still
   wins.

The second condition alone is insufficient. If the edit never produced a
positive donor response, a negative final margin may simply mean that the edit
was poor or the donor was not recognized.

## Predicates required before final claims

| Predicate | Plain-language question | FunnyBird | CUB70 |
|---|---|---|---|
| P0 Model health | Does the unedited model classify ordinary images and their present concepts sensibly? | Required and supported for the selected checkpoints | Required; current CBM health checks are available |
| P1 Intervention validity | Did only the intended part change, and did it visibly change? | Supported by the semantic renderer gate and shared image hashes | **Failed** for the present deletion, patch, and beak/tail insertion proxies |
| P2 Donor response | Did the edited image move toward the donor concept? | Positive on average for every part in the validated swap | **Failed** in the beak/tail pilot: only 40% moved positively and medians were about zero |
| P3 Old source still wins | After a positive donor response, does the old source concept remain higher? | Supported most strongly for tail and partially for other parts | Cannot be interpreted until P1 and P2 pass |
| P4 Direction controls | Do forward/backward directions agree rather than cancel? Do grounded parts behave better? | Supported by direction and foot/wing controls | Not established by a calibrated intervention |
| P5 Same-example comparison | Are all model comparisons made on identical images and edits? | Supported for the fixed-render seed-1 comparison | Natural visibility comparisons use different photographs and remain observational |
| P6 Replication | Does the result persist across seeds and the claimed gamma range? | Seed 1 is the causal discovery; broader seed/gamma claims remain conditional on completed matched replays | Not applicable until a CUB intervention passes P1 and P2 |

## What may be concluded now

| Final claim | Required predicates | Status |
|---|---|---|
| FunnyBird CBM shows causal retained-source backwash under a clean part swap | P0–P5 | **Supported for the validated seed-1 experiment**, with tail strongest |
| MCBM minimality repairs the phenomenon | P0–P6 plus a clear improvement with gamma | **Not supported** by the present seed-1 pattern; broad stability is still limited by seed/gamma coverage |
| Visibility-conflicting training labels cause part of the FunnyBird failure | P0–P5 plus matched standard/RLv2 training and fixed images | **Provisionally supported at seed 1**; RLv2 improves tail but does not remove all candidate events |
| Exact variant difficulty contributes | Stable differences after visibility/direction controls | **Observed**, not independently manipulated |
| Source species/body context causes the remainder | Residual association plus an independent body/species intervention | **Not proved**; only the residual association is observed |
| All FunnyBird causes have been found | Candidate-event rate becomes negligible after valid causal interventions | **Not proved**; a residual remains |
| CUB70 has causal backwash at FunnyBird strength | P0–P6 with a calibrated CUB intervention | **Neither proved nor disproved**; the current intervention predicates failed |
| CUB70 contains the same risk factors | label/mask conflict, visibility, species structure, and contextual residuals | **Supported observationally**, with explicit confounding limits |

## Subtracting influences without inventing causal percentages

Use exactly the same matched rows throughout.

### Step A — raw candidate events

Count `donor_response > 0 and after_margin < 0` on every valid FunnyBird swap.
This is the phenomenon to explain.

### Step B — visibility selection

Repeat the count on high-visibility swaps. The difference from Step A answers:
“How concentrated are candidate events among low-visibility examples?” It does
**not** answer “what percentage was caused by visibility,” because examples were
removed rather than repaired.

### Step C — matched label intervention

On those exact same high-visibility images, compare standard training with
RLv2 training. Report all transitions:

- `1 -> 0`: candidate event resolved by RLv2;
- `0 -> 1`: candidate event introduced by RLv2;
- `1 -> 1`: candidate event remaining after RLv2.

This is the cleanest available causal subtraction because the rendered image is
identical and the intended training-label rule is what changed.

### Step D — organize the remainder

Within the remaining `1 -> 1` events, measure how much prediction improves when
we add exact source/donor variant, direction, then source species. This is a
predictive residual decomposition, not a causal decomposition. It tells us where
the remainder clusters and which next intervention is worth running.

### Step E — show what is left

Report the number and rate of remaining `1 -> 1` events by part, variant pair,
and source species, and display representative images. Call it **unexplained
residual candidate backwash**, not “species-caused backwash.”

## Why the same waterfall stops early for CUB70

CUB70 currently fails P1 and P2. Therefore its negative final margins cannot
enter Steps A–E as proven candidate backwash events. The CUB notebook should
show the failed edits, retain the natural visibility/species results as
observational evidence, and stop before making a causal residual waterfall.

