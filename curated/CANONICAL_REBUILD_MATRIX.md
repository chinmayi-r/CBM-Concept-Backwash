# Staged rebuild and completion matrix

This is a small research checklist, not an instruction to rebuild every model.

Legend: `[x]` completed and retained, `[ ]` missing, `[~]` existing output must
be replaced, `[s]` correct MCBM architecture but model initialization was not
seeded, and `[?]` inspect the saved artifact/log before deciding.

## Stage 1: FunnyBird

### Official Koh Joint CBM

| Work | seed 1 | seed 2 | seed 3 |
|---|:---:|:---:|:---:|
| standard training | [~] | [~] | [~] |
| standard fixed-render evaluation | [~] | [ ] | [ ] |
| RLv2 training, using existing relabelled records | [~] | [~] | [~] |
| RLv2 fixed-render evaluation | [~] | [ ] | [ ] |

The `[~]` outputs are legacy `minimal_cbm` CBM outputs. Preserve them under
`legacy_not_for_notebooks`; do not load them into notebooks 02 or 02rl.

### Official minimal_cbm MCBM

| gamma | standard s1 | standard s2 | standard s3 | RLv2 s1 | RLv2 s2 | RLv2 s3 | fixed-render evidence |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 0 | [s] | [s] | [s] | [s] | [s] | [s] | rerun checkpoints; reuse render cache |
| 0.1 | [s] | [s] | [s] | [s] | [s] | [s] | rerun checkpoints; reuse render cache |
| 0.3 | [s] | [ ] | [ ] | [s] | [ ] | [ ] | rerun checkpoints; reuse render cache |
| 1 | [s] | [ ] | [ ] | [s] | [ ] | [ ] | rerun checkpoints; reuse render cache |
| 3 | [s] | [ ] | [ ] | [s] | [ ] | [ ] | rerun checkpoints; reuse render cache |
| 5 | [s] | [ ] | [ ] | [s] | [ ] | [ ] | rerun checkpoints; reuse render cache |

The old MCBM cells used the intended architecture and loss, but the wrapper did
not seed model initialization. Preserve them under `legacy_not_for_notebooks`.
The fixed RGB render cache is input data and remains reusable.

## Stage 2: CUB70

Keep all outputs under a CUB70-specific root. CUB70 has 70 species and must
never load a 200-way Full-CUB head.

### Official Koh Joint CBM

| Work | seed 1 | seed 2 | seed 3 |
|---|:---:|:---:|:---:|
| standard training | [~] | [ ] | [ ] |
| natural-image evaluation for notebook 05 | [~] | [ ] | [ ] |

### Official minimal_cbm MCBM

| gamma | seed 1 | seed 2 | seed 3 |
|---:|:---:|:---:|:---:|
| 0 | [s] job 3343609 completed | [ ] | [ ] |
| 0.1 | [s] job 3343610 completed | [ ] | [ ] |
| 0.3 | [s] job 3343611 completed | [ ] | [ ] |
| 1 | [s] job 3343612 completed | [ ] | [ ] |
| 3 | [ ] job 3343613 ended with ERROR | [ ] | [ ] |
| 5 | [ ] job 3343614 ended with ERROR | [ ] | [ ] |

## Stage 3: Full CUB

Keep this stage separate from CUB70. Full CUB uses the paper's 200 species and
112 concepts.

### Official Koh Joint CBM

| Work | seed 1 | seed 2 | seed 3 |
|---|:---:|:---:|:---:|
| standard training | [~] | [ ] | [ ] |
| natural-image evaluation | [~] | [ ] | [ ] |

### Official minimal_cbm MCBM

No Full-CUB MCBM cell is accepted yet. Build this only after Stage 2 is
validated; do not infer it from CUB70 outputs.

## Execution order

1. Validate the single Stage-1 Koh standard seed-1 template end to end.
2. Run Stage-1 RLv2 seed 1 from the same template, changing only the data path.
3. Run their fixed-render evaluations.
4. Submit FunnyBird seeds 2 and 3 independently.
5. Validate Stage-2 Koh seed 1, then expand to seeds 2 and 3.
6. Diagnose CUB70 MCBM jobs 3343613 and 3343614; do not blind-rerun them.
7. Start Stage 3 only after Stage 2 artifacts and notebook paths are clean.

No seed depends on another seed. Automatic retry is limited to two identical
retries for scheduler/infrastructure interruption. Code, data, or numerical
errors stop for diagnosis and template correction.
