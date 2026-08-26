# Canonical research contract

This file is the single current operating contract for this repository. Chat
memory is not authoritative. The large files under `curated/` are historical
provenance; read a specific one only when its evidence or implementation is
needed. Do not load all old handoffs before starting work.

## Research question and order

We are testing whether a concept bottleneck predicts a named concept from its
own part pixels or from species/body context (concept backwash).

Keep this order:

`non-RL data -> standard CBM discovery -> standard MCBM minimality -> RLv2 cause test -> CUB70 -> Full CUB`

RL never replaces the standard-CBM discovery story. CUB follows FunnyBird's
questions but is allowed to give a different answer.

## Canonical model implementations

Do not substitute one repository's model for another because its output format
is convenient.

- **Standard CBM:** use the Koh `ConceptBottleneck` **Joint architecture** with
  the professor-approved substitution of a ResNet-50 image encoder for
  Inception-v3: one raw
  concept logit per concept, a single linear concept-to-species layer, joint
  task plus concept loss, and `attr_loss_weight=0.01`. The class head reads raw
  concept logits; do not add `-use_sigmoid`. FunnyBird changes only the data,
  number of species (50), and number of concepts (26). CUB70 changes only the
  filtered data and dimensions (70 species, 112 concepts). Full CUB uses the
  paper's 200 species and 112 concepts.
- **FunnyBird seed-1 training protocol:** the accepted final protocol
  is `accelerated_v1`, not Koh's historical 1,000-epoch optimizer schedule. It
  keeps the Koh Joint architecture and normalized loss exactly, but trains the
  ResNet-50 model for 100 epochs with batch 128, SGD momentum 0.9, weight decay
  `0.0004`, AMP, eight loader workers, a five-epoch linear warm-up from learning
  rate `0.001` to `0.02`, and cosine decay to `0.00002`. It saves atomic restart
  state including the AMP scaler every epoch and full milestone checkpoints at
  epochs 25, 50, 75, and 100. These are declared scientific training settings,
  not an exact Koh-schedule reproduction. The accepted description is
  `ResNet-50 Koh-architecture Joint CBM, accelerated_v1`. Standard and RLv2
  seed 1 use this identical protocol; only their declared label/data view
  differs.
  Acceptance also requires the predeclared epoch-75 to epoch-100 ordinary-health
  stability audit stored in `CONVERGENCE.json`; controlled-swap grounding is a
  separate subsequent requirement.
  The completed FunnyBird standard seed-1 job `3357208` is the single recorded
  post-hoc exception: training and evaluation reached epoch 100, all four
  concept-health stability predicates passed, and only the symmetric task-
  accuracy predicate missed because accuracy improved from `0.978` to `0.992`
  on 500 test images. It may be reconciled without retraining only by
  `curated/analysis/reconcile_koh_accelerated_seed1.py`, which must preserve the
  original `INCOMPLETE` `CONVERGENCE.json`, verify the final and epoch-100
  parameters are identical, and record the limited post-hoc decision in
  `CONVERGENCE_DECISION.json`. This exception does not alter the gate for any
  other run and does not itself establish grounding.
- Koh Independent and Sequential are paper baselines, but they are not the
  primary model for the backwash mechanism because the task loss does not
  update their image-to-concept model. Do not train them unless a later,
  explicit scientific question requires them.
- Koh's model named `Standard` has zero concept-loss weight and is an ordinary
  species model, not the standard supervised CBM in this project.
- **MCBM:** use the official `minimal_cbm` repository with the single declared
  compatibility patch. Gamma zero remains an MCBM architecture with its
  minimality weight set to zero; it is not expected to equal Koh Joint CBM.
  Existing MCBM runs are accepted as independent trained runs. Their old seed
  labels do not guarantee exact replay because initialization was uncontrolled;
  this is a reproducibility limitation, not invalidation of their results.
- **Freeze the recorded MCBM input recipes.** Completed FunnyBird MCBMs use the
  custom 224-pixel ImageNet-normalized FunnyBird loader. Completed CUB70 MCBMs
  use the upstream 299-pixel CUB loader, including its historical normalization,
  with a ResNet-50 encoder. The latter is an Inception-era preprocessing recipe,
  but the models trained end-to-end on finite, consistent inputs; it is a
  declared limitation and cross-dataset preprocessing confound, not grounds to
  invalidate or retrain them. Any genuinely missing cell added to an existing
  dataset/gamma comparison must reuse that dataset's recorded preprocessing,
  backbone, weights family, augmentation, optimizer, and epoch configuration.
  Do not silently "correct" preprocessing inside an existing comparison.
- Never use `minimal_cbm`'s CBM implementation in notebooks 02 or 05.

The authoritative primary sources for the architecture and loss other than the
approved encoder substitution are
`external/ConceptBottleneck/CUB/README.md` and Koh et al. (2020), Sections 3-4.
The historical optimizer command remains a reproduction reference, but is not
the accepted FunnyBird seed-1 training schedule. Any wrapper must audit both the
preserved architecture/loss boundary and the declared accelerated protocol
before submission.
The ResNet adapter may replace only Koh Joint's image encoder while preserving
its scalar raw-concept outputs, auxiliary-output contract, linear class head,
loss, and raw-logit path. The separately declared `accelerated_v1` adapter may
change only optimizer mechanics, batches, precision, loader workers, schedule,
and stopping at the values listed above. It is approved only for FunnyBird
standard/RLv2 seed 1. It must reject any `minimal_cbm` import and pass the
structural, loss-source, input-integrity, seed-gate, schedule, finite-loss,
milestone, and restart-state audits before acceptance. The only other model/data training
adapter allowed for FunnyBird/CUB70 changes Koh's hard-coded
`N_CLASSES=200` before delegating to the repository's own `experiments.py`
`__main__`. Concept count remains Koh's existing `-n_attributes` argument.
Full-CUB Inception invokes `experiments.py` directly. Full-CUB ResNet enters
through the same audited constructor adapter with the unchanged class count of
200, then delegates to `experiments.py` `__main__`. Do not duplicate Koh's
seeding, parser, optimizer, scheduler, or training loop.
The ResNet import-boundary audit must additionally prove that Koh `CUB.train`
copied the requested `N_CLASSES` value and `build_koh_resnet50_joint` into its
module globals before `experiments.py` runs. The constructed model must contain
no Inception or MCBM module types.
Two data-edge adapters are also required: FunnyBird pickle views rewrite only
`img_path` to include Koh's required `CUB_200_2011` marker; CUB70 assigns neutral
positive weight `1.0` only to all-zero targets where Koh's original ratio
divides by zero. Every target with at least one positive uses Koh's exact formula.

Historical `koh_original` jobs use the opt-in infrastructure-only patch recorded
at `curated/patches/koh_restartable_training.patch`. It writes an atomic
epoch-boundary restart state containing the model, optimizer momentum,
scheduler, early-stop state, and all Python/NumPy/PyTorch RNG states. The
`accelerated_v1` protocol does not apply that legacy patch because it replaces
Koh's original `train()` at import time; its replacement trainer owns the
atomic restart state and additionally saves the AMP scaler. Both paths leave
the pinned submodule untouched and must pass their matching restart-equivalence
audit.
For `accelerated_v1`, that audit must enter the production replacement trainer,
preserve staged protocol/model/input manifests, simulate an epoch-boundary
interruption, resume, and obtain the same final parameters as an uninterrupted
run. It must also compare the accelerated Joint loss with Koh's official Joint
loss on the same batch before full training starts.

## Dataset staging

Keep these as separate stages and separate output roots:

1. FunnyBird: easiest controlled demonstration; renderer swaps are available.
2. CUB70: harder observational replication with 70 species and released masks.
3. Full CUB: final 200-species stage; never silently mix it with CUB70.

Finish and validate the seed-1 path for a stage before expanding that stage to
seeds 2 and 3. Seeds 2 and 3 are independent peers and must never depend on one
another.
The explicitly requested seed-1 campaign may keep two independent seed-1 jobs
resident or pending concurrently on Adroit. This does not authorize seed 2/3.
Slurm dependencies must prevent swaps from running before both FunnyBird
standard/RLv2 manifests and prevent MCBM follow-ups from running before their
dataset's standard seed-1 model.
Use the explicit scripts under `curated/train/entries/`, one completion-matrix
entry per script. There is no bulk campaign submission interface. FunnyBird
RLv2, CUB70 standard, and full-CUB standard are independent jobs and must print
`dependency=none`; only the fixed FunnyBird swaps may wait for the live RLv2
job. Full-CUB MCBM remains a later separately requested stage.

Old `minimal_cbm` CBM checkpoints are preserved under an unmistakable
`legacy_not_for_notebooks` root. Accepted notebook builders must reject that
root and verify the model framework in a saved manifest before loading a
checkpoint.

## Authoritative original notebook sources

Do not choose a recall notebook by the shortest filename.

- FunnyBird recall authority: `funnybird_notebooks/fb_recallv2.ipynb` (the
  developed executed version). Reproduce its two-stage rule: first match two
  species that each have sufficient positive and negative rows; use its
  all-positive-species fallback only when that first rule yields no pairs. The
  current curated validation split has image-varying concepts (observed maximum
  within-species prevalence below 0.9), so it requires the first rule rather
  than the fallback. Always print which rule and label population were used.
- Latest recall-method refinement: `notebooks/mcbm_recallv4.ipynb`. Its matching,
  vectorized bootstrap, balanced-accuracy, and gamma-analysis code supersede
  earlier `recall.ipynb` helper implementations. Its numerical results are MCBM
  results and must not be presented as standard-CBM results.
- `recall.ipynb` and earlier `mcbm_recallv2/v3` files are provenance, not the
  method authority.
- Renderer-swap authority and inspiration remain
  `funnybird_notebooks/fb_cbm_renderer_swap_v2.ipynb` and
  `funnybird_notebooks/fb_mcbm_renderer_swap.ipynb`.

Recall is a representation/model-health and species-dependence diagnostic. It is
not a replacement for the FunnyBird controlled swap or for raw concept-logit
analysis.

## Immediate task

Replace the legacy standard-CBM artifacts with official Koh Joint artifacts.
Do not retrain an existing MCBM cell solely because its old initialization was
not exactly replayable; train only genuinely missing MCBM evidence when needed.

Notebook runners 02, 02rl, and 05 must remain fail-closed until their official
Koh manifests and evaluations exist. MCBM notebooks 03 and 03rl may use the
existing accepted MCBM runs with the reproducibility limitation stated. Their
report
structure, figures, variable introductions, and definition of done follow
`curated/NOTEBOOK_REPORT_ROADMAP.md`. Failed deletion, patch, paste, and
forecasting material remains in a clearly labelled methods appendix.

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

## What each dataset actually lets us test

Start with this capability map before designing a cross-dataset figure. Match the
scientific question, not necessarily the mechanical operation.

| Question / method | FunnyBird | CUB/CUB70 | Consequence |
|---|---|---|---|
| Exact species-concept structure | Easy and exact; concepts are generated from species/part variants | Easy from image attributes, but labels vary within species and include uncertainty | Report each dataset's real structure; do not force the same prevalence model |
| Exact part visibility | Easy and exact from renderer part maps | Available only through 11 released masks; some are bilateral, missing, or coarser than the named attribute | CUB visibility is an approximation and must retain mask-coverage counts |
| Clean one-part deletion | Easy and robust with the renderer | No clean removal mechanism; inpainting/patch deletion changed texture and controls | Use deletion as causal evidence only on FunnyBird; CUB failed edits go to appendix |
| Clean donor-part swap | Easy and robust: same body, pose, camera, background; one part changes | No renderer or native donor parameter; paste pilot did not create reliable donor response | FunnyBird can use donor/source margin; CUB cannot |
| Verify inserted-pixel response | Direct `response_delta` on identical images | No accepted donor insertion | Do not invent a CUB donor margin |
| Natural visible-vs-hidden comparison | Available, but controlled swap is stronger | Easy and central using mask state, area, and left/right counts | Use raw `z`; CUB result is observational because photographs differ |
| Label/mask conflict count | Exact renderer-derived visibility and labels | Easy where a released mask maps to the concept; coarse masks can undercount conflict | Valid data diagnostic in both, with different mask precision |
| Visibility-aware relabel/retrain | Implemented and matched as RLv2 | Possible only if training-mask identity/coverage passes first | It is a causal label test, not part of initial CBM discovery |
| Per-species recall | Easy | Easy | Health/species-dependence diagnostic, not grounding proof |
| Matched recall gap | Follow `fb_recallv2`: match positive/negative counts when the current labels vary within species; use the all-positive fallback only for a genuinely species-constant population | Pair species after matching positive/negative counts because CUB varies within species | Same question, label-population-dependent implementation |
| Raw-`z` species effect | Easy on concept outputs and swaps | Easy on exact concepts; can match concept, label, visibility, and species | Directly comparable observational question |
| Species decoding from concept vector | Easy; 50 species | Easy; 70/200 species with different chance levels | Report chance and population explicitly |
| Exact variant/confusion matrix | Exact part variants and donor identities | Exact attribute values exist, but no controlled donor replacement | CUB can study value confusion on natural images, not post-swap confusion |
| Number of alternatives / frequency | Exact variants per FunnyBird part and species per variant | Selected values per attribute type and species support per exact value | Similar candidate mechanism, not identical data generation |
| MCBM gamma test | Trained sweep and fixed-render replay available/pending reconciliation | Separate CUB MCBM stage after standard CBM | Never let MCBM replace the standard-CBM question |

### Cross-dataset substitution rule

For every FunnyBird test, notebook 05 must say one of:

1. `same operation available` and run it;
2. `same question, weaker CUB approximation` and name the confounds;
3. `not available on CUB` and stop, without inventing an artificial gate.

The absence of a clean CUB swap is why the approaches differ. It does not justify
dropping the FunnyBird swap, and it does not justify calling all CUB evidence
worthless.

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
- The top of every model notebook must explain the implemented model structure
  before any result: what the encoder emits, what the concept head computes,
  what the class head reads, the training loss, and the distinction between
  latent `h`, raw concept logit `z`, probability `p`, binary label `c`, and
  thresholded prediction `c_hat`. Later sections must refer back to this
  notation rather than silently changing names.
- Every table column and plotted quantity must be defined in Markdown before it
  appears. A technical symbol such as `m_cf`, `response_delta`, or
  `context_gap` is allowed only after its formula, population, unit, direction,
  and scientific meaning have been explained. A plain-language axis title is
  also allowed, but the Markdown must map it explicitly to the corresponding
  variable or formula.
- Before every figure, explain all axes, panels, rows, colors, markers,
  reference lines, aggregation, denominator, and exclusions. State what moving
  left/right or up/down means. Include a small numerical example whenever the
  formula or sign could be misunderstood. A reader must be able to interpret
  the figure without reading its code cell.
- “Simple language” must not remove technical precision. First define the exact
  variable, then restate it in ordinary language and give an example. Do not use
  vague substitute names such as “candidate event,” “helped,” “score spread,”
  or “effect” without immediately stating the underlying predicate or formula.
- No result paragraph may introduce a variable, model component, population,
  matching rule, threshold, or metric that was not defined earlier in the
  notebook or immediately before the figure.
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

Legend: `DONE` remains usable, `REDO` exists under the wrong CBM framework,
`DONE-INIT-LIMIT` is a completed official MCBM whose numeric seed did not
control model initialization, `MISSING` has not completed, and `CHECK` requires
artifact/log reconciliation. `DONE-INIT-LIMIT` is not a retraining instruction.

### Standard CBM (official Koh Joint required)

| Stage | Standard s1 | Standard s2 | Standard s3 | RLv2 s1 | RLv2 s2 | RLv2 s3 | Evaluation |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| FunnyBird | MISSING accelerated_v1 | REDO; gated | REDO; gated | REDO; gated | REDO; gated | REDO; gated | after accepted seed 1, rerun fixed swaps from its final checkpoint |
| CUB70 | REDO job 3344162; Koh Inception | REDO job 3344163; Koh Inception | REDO job 3344164; Koh Inception | -- | -- | -- | preserve as historical Koh evidence; final all-ResNet comparison requires gated rerun after FunnyBird seed 1 |
| Full CUB | REDO | MISSING | MISSING | -- | -- | -- | separate 200-species natural-image stage |

Existing minimal_cbm-CBM outputs are wrong-framework legacy artifacts. The
completed CUB70 Koh jobs are official Joint models but use the superseded
Inception backbone. Preserve both categories under unmistakable historical
roots, but neither may enter the final all-ResNet notebooks. Existing RLv2
label records are reused; labels are not regenerated.

### FunnyBird MCBM

| gamma | Standard s1 | Standard s2 | Standard s3 | RLv2 s1 | RLv2 s2 | RLv2 s3 | Fixed-render evaluation |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 0 | DONE-INIT-LIMIT | DONE-INIT-LIMIT | DONE-INIT-LIMIT | DONE-INIT-LIMIT | DONE-INIT-LIMIT | DONE-INIT-LIMIT | reuse accepted checkpoints and render cache |
| 0.1 | DONE-INIT-LIMIT | DONE-INIT-LIMIT | DONE-INIT-LIMIT | DONE-INIT-LIMIT | DONE-INIT-LIMIT | DONE-INIT-LIMIT | reuse accepted checkpoints and render cache |
| 0.3 | DONE-INIT-LIMIT | MISSING | MISSING | DONE-INIT-LIMIT | MISSING | MISSING | train only genuinely missing cells with the frozen recipe |
| 1 | DONE-INIT-LIMIT | MISSING | MISSING | DONE-INIT-LIMIT | MISSING | MISSING | train only genuinely missing cells with the frozen recipe |
| 3 | DONE-INIT-LIMIT | MISSING | MISSING | DONE-INIT-LIMIT | MISSING | MISSING | train only genuinely missing cells with the frozen recipe |
| 5 | DONE-INIT-LIMIT | MISSING | MISSING | DONE-INIT-LIMIT | MISSING | MISSING | train only genuinely missing cells with the frozen recipe |

The July MCBM runs used the intended architecture, loss, data, and compatibility
patch, but `run_mcbm.py` did not seed Python/NumPy/PyTorch before constructing
the model. The numeric seed was used by data loaders but not by model
initialization. Preserve and use those completed independent runs with this
limitation stated. Do not rerun them solely to make their directory seed an
exact initialization replay.

### CUB70 MCBM

| gamma | Standard s1 | Standard s2 | Standard s3 |
|---:|:---:|:---:|:---:|
| 0 | DONE-INIT-LIMIT job 3343609 | MISSING | MISSING |
| 0.1 | DONE-INIT-LIMIT job 3343610 | MISSING | MISSING |
| 0.3 | DONE-INIT-LIMIT job 3343611 | MISSING | MISSING |
| 1 | DONE-INIT-LIMIT job 3343612 | MISSING | MISSING |
| 3 | ERROR job 3343613 | MISSING | MISSING |
| 5 | ERROR job 3343614 | MISSING | MISSING |

Full-CUB MCBM is a later separate stage and is not implied by CUB70 completion.

Known unusable or uncalibrated methods belong under `legacy_not_for_notebooks`;
they are evidence about method limitations, not proof that CUB has no backwash:
fixed-cache v1 black renders, reciprocal mask deletion, randomized patch V1/V2
calibration, and the CUB beak/tail paste pilot.

## Status words: never print a bare `FAIL`

Use exactly one of these descriptions and state what remains usable:

- `ERROR`: code or a job did not finish. There is no scientific result yet.
- `INVALID OUTPUT`: computation finished but produced corrupted or wrong inputs
  (for example, the fixed-cache-v1 black renders). Do not interpret it.
- `METHOD NOT CALIBRATED`: computation finished, but the proposed measurement did
  not reproduce a known FunnyBird control. This rejects that measurement as a
  CUB proxy; it does not reject the model result or the backwash hypothesis.
- `VALID TEST, NO SUPPORT`: inputs and method were valid, but the predicted effect
  was absent. This is genuine negative evidence for that specific prediction.
- `INCOMPLETE`: required models, seeds, or outputs are missing.
- `ACCEPTED FOR <limited claim>`: name the exact claim supported.

Do not turn a multi-check method into all-or-nothing project status. Report each
check separately. A failed optional robustness check cannot erase earlier accepted
evidence. Scientific gates must be justified by the claim they protect, specified
before inspecting the result, and no stricter than needed for that claim.

## Cluster safety

Before recommending any cluster action, obtain a fresh full `squeue -u "$USER"`
and relevant `sacct`. Never infer a job payload from its short name: inspect
`scontrol show job -dd` and, when needed, `scontrol write batch_script`. Do not
release old held jobs without proving they are current. Do not restart completed
work. Record accepted new job state in this file's completion matrix.

### Error and retry policy

- Every job writes a completion manifest only after its checkpoint and expected
  outputs pass finite-value, shape, framework, dataset, seed, and gamma checks.
- Automatically resubmit the identical payload at most twice only for an
  infrastructure interruption such as `TIMEOUT`, node failure, or preemption.
- Never automatically loop a Python exception, missing input, non-finite loss,
  or failed artifact validation. Capture the traceback, diagnose the cause,
  patch the shared template once, dry-run it, then submit a new independent job.
- A failed seed never blocks a different seed. Seed 3 does not depend on seed 2.
- Jobs 3343613 and 3343614 require log diagnosis before retry; they must not
  simply rerun indefinitely.
