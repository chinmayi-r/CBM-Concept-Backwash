# Repository operating rules

This repository contains a long-running research audit. Chat memory is not the
source of truth.

Before proposing work under `curated/`, read, in this order:

1. `curated/NEW_CHAT_HANDOFF_2026-07-24.md`
2. `curated/CURRENT_STATE.md`
3. `curated/EXPERIMENT_TRACKER.md`
4. `curated/CATCHUP.md`
5. `curated/CBM_BASELINE.md`
6. `curated/REFERENCE_CELL_MAP.md`

Preserve the research order:

`non-RL data -> CBM discovery -> MCBM minimality -> RL causal follow-up -> CUB`

RL never replaces the non-RL discovery story.

Operational rules:

- Reconcile `curated/CURRENT_STATE.md` with a full `squeue -u "$USER"` and
  relevant `sacct` output before recommending a cluster action.
- Never infer a job's payload from its short Slurm name. Inspect `scontrol show
  job -dd` and, when exact submitted contents matter, `scontrol write
  batch_script`.
- Record every submitted job, dependency, commit, expected output, and acceptance
  signature in `curated/CURRENT_STATE.md`.
- A job is not accepted until its log proves the required audits and output.
- Do not release old held jobs merely because they exist. First decide whether
  they are superseded; cancel superseded jobs explicitly after preserving their
  submitted scripts if needed.
- Do not restart completed work. Continue from the shortest missing proof step.
- Before interpreting an important notebook figure, display it in chat with a
  plain caption and use:
  question -> variables/prediction -> figure -> literal observation ->
  alternatives -> discriminating test -> limited conclusion -> next question.

