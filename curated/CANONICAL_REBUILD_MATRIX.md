# Canonical clean-room rebuild matrix

This matrix is the complete requested rebuild. Nothing in the old result
directories satisfies a row. Every accepted output must live below
`$CURATED_DATA/canonical_20260806_v1` and have a verified `SUCCESS` manifest.

## Ordering

The queue submits every seed-1 row first. Seeds 2 and 3 become eligible only
after the seed-1 wave finishes, but they are peers: seed 3 never depends on
seed 2. A scheduler time-limit warning may requeue the identical payload at
most twice. A normal code/data/numerical error is never looped automatically.

## Training and final untouched-test evaluation

| Dataset | Labels | Framework | Regime | Seeds | Final-test export |
|---|---|---|---|---|---|
| FunnyBird | species-level standard | official Koh | independent, sequential, joint, joint-sigmoid | 1, 2, 3 | each regime/seed |
| FunnyBird | image-level RLv2 | official Koh | independent, sequential, joint, joint-sigmoid | 1, 2, 3 | each regime/seed |
| CUB | standard | official Koh | independent, sequential, joint, joint-sigmoid | 1, 2, 3 | each regime/seed |
| CUB70 | standard | official Koh | independent, sequential, joint, joint-sigmoid | 1, 2, 3 | each regime/seed |
| FunnyBird | standard | official minimal_cbm MCBM | gamma 0, 0.1, 0.3, 1, 3, 5 | 1, 2, 3 | each gamma/seed |
| FunnyBird | RLv2 | official minimal_cbm MCBM | gamma 0, 0.1, 0.3, 1, 3, 5 | 1, 2, 3 | each gamma/seed |
| CUB | standard | official minimal_cbm MCBM | gamma 0, 0.1, 0.3, 1, 3, 5 | 1, 2, 3 | each gamma/seed |
| CUB70 | standard | official minimal_cbm MCBM | gamma 0, 0.1, 0.3, 1, 3, 5 | 1, 2, 3 | each gamma/seed |

Each of those four dataset/label cells also trains the `minimal_cbm` package's
own CBM with identical ResNet/data/optimizer settings. This is notebook 03/06's
internal architecture control. It is kept distinct from the official Koh CBM,
which remains the standard-CBM result for notebooks 02/05.

Koh `Concept_XtoC` and the required class-head/extraction stages are separate
manifested jobs. `Independent` and `Sequential` therefore use the same exact
trained concept model within a dataset/label/seed cell. All model selection
uses validation, and every table above gets a separate untouched-test pass.

CUB/CUB70 RLv2 is not a row because no CUB RLv2 target dataset has been
defined. Inventing one during submission would create the contamination this
rebuild is meant to remove.

## Controlled interventions

| Dataset | Model cells replayed on one fixed render cache | Seeds |
|---|---|---|
| FunnyBird standard | four Koh regimes + minimal_cbm CBM control + all six MCBM gammas | 1, 2, 3 |
| FunnyBird RLv2 | four Koh regimes + minimal_cbm CBM control + all six MCBM gammas | 1, 2, 3 |

Each replay verifies model manifests first and validates RGB identity/hashes.
CUB and CUB70 have no renderer, so their registered outputs are natural-image
observational tests rather than fabricated swaps.

## Definition of complete

- `02`: official-Koh FunnyBird standard-CBM discovery, three seeds, controlled swaps.
- `03`: FunnyBird MCBM gamma 0 through 5, three seeds, controlled swaps.
- `05`: official-Koh CUB and CUB70 standard-CBM observational comparison, three seeds.
- `06`: CUB and CUB70 MCBM gamma 0 through 5, three seeds.
- RLv2 comparison: FunnyBird Koh and MCBM matched to their own standard cells,
  three seeds, with the identical fixed-render replay.

Training, final evaluation, and intervention replay are separate completion
conditions. A checkpoint alone is not a completed row.

The matrix contains **294 required scientific stage manifests**: 98 for each
seed. A final verifier runs after both replication waves and writes
`completion.json`; it succeeds only at 294/294.

Submission creates a detached source snapshot at `canonical_20260806_v1/code`.
All 271 Slurm jobs run that commit and its pinned submodules, so later edits or
pulls in the normal checkout cannot silently change queued experiments.
