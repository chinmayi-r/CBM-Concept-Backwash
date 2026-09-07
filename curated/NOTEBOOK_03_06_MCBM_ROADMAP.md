# Standard MCBM reports: locked roadmap for notebooks 03 and 06

This file is the implementation contract for the standard (non-RLv2) MCBM
stage. Notebook 03 is completed and reviewed before notebook 06 is interpreted.
RLv2 remains a later causal follow-up.

## Scientific question

### September 7 implementation status

The main builder now assembles all 24 Standard figure/example/table constructions
explicitly, keeping four shared renderer/data outputs once and repeating model
computations per gamma. The new computations include full/equal-width conditional
information, nonlinear saved-head magnitude replacement, frozen-swap off-target
h erasure, contributor prediction, and official-loss gradients at frozen checkpoints.
Detailed file/method changes and test coverage are in `NOTEBOOK03_CHANGELOG.md`.
Local synthetic runtime checks are not scientific execution: real CUDA replay and
review of every Adroit-rendered output remain pending. No training is submitted.

### September 6 clarification: loss incentives are the chapter's subject

#### Conversation review and reporting contract

Read the paginated message text of `Explain Counterfactual Margin`
(`6a9a7f90-616c-83ea-b7db-06928e662389`) through its oldest returned turn.
Initial item truncation was resolved by rereading with a larger item limit;
some stored assistant replies themselves end mid-sentence. Do not invent their
missing endings. The recent main-thread instructions are also available as
message text; older assistant turns available only as summaries are not raw reads.

| Batch of directions | Lesson applied |
|---|---|
| Exact example and fixed coordinates | Walk one tail_2 -> tail_7 case through all calculations. All nine tail outputs remain fixed; seven enter only the off-target diagnostic. |
| Absent versus present statistics | Print N, mean and SD for each label state and every concept before using baseline centering. Label-conditioned replacement uses supplied labels, not necessarily the model's original signs. |
| Why correlate? | Encoding, saved-head sensitivity and upstream grounding are distinct questions. Weak correlation establishes neither zero evidence nor equal evidence across swaps. |
| Between-part differences | Prioritize distributions and actual head interventions; compare all five parts. Pair-centering would erase between-pair means, so do not use it to rank absolute evidence. |
| Standard/RLv2/MCBM comparison | Account for every Standard output and retain its visual construction. Report model wiring, population and recipe differences explicitly. |
| Loss incentives and remedies | Explain h as a number and g as a function; distinguish algebraic predictions from observed training dynamics. Select new loss proposals from measured failures. |
| Scope and quality | MCBM is current. Mask-based CUB swaps are deferred. Review every actual output; do not substitute expected conclusions or compilation for completed analysis. |

Every figure must have, before its code: question/prediction; exact formulas;
inputs/model; new diagnostic versus saved classifier; axes/colors/groups;
denominators/exclusions; and a numerical example wherever signs or operations
could be misunderstood. After it: literal result; supported claim; plausible
alternative; distinguishing test; and next question. Use the same terminology
as the image. Keep per-part tables and actual images in chat reviews.

Corrections to earlier side-chat interpretations:

- Probability mass moved is sensitivity to a specified replacement, not the
  fraction of encoded information used. It can be small under saturated class
  probabilities; inspect class-score changes too.
- An absent-baseline deviation is not automatically a species fingerprint.
  A signed source/donor contribution is relative to that baseline, and changing
  those values may remove other variation as well.
- For Koh's linear head, the class-gap change under baseline replacement equals
  e algebraically; recomputing it checks implementation, not an independent
  confirmation of the mechanism. Probability and decision effects add meaning.
- For MCBM, use the actual nonlinear head on full h before and after replacement.
  The Koh linear-weight sum does not transfer unchanged.
- Species losses shape the encoder during training. A downstream head edit at
  evaluation does not feed back into the already computed concept scores.
- Larger between-part evidence is worth testing but cannot establish why the
  upstream margin is wrong. Different coordinate counts and score scales remain.

The chapter ends by answering: what compression changed, whether grounding
improved, and which measured failure motivates a particular better loss.

The user wants MCBM to motivate an investigation of losses that might reduce
backwash. The chapter must explain both implemented losses, test their
predictions, and permit negative findings. The following narrative supersedes
older figure-number prescriptions when they conflict with exact Notebook 02
visual parity. It is a design contract, not a claim that all analyses are executed.

Primary reading: Koh et al., https://proceedings.mlr.press/v119/koh20a.html
(Joint objective, raw-logit path, model-level intervention boundary), and
Almudevar et al., https://arxiv.org/html/2506.04877v3 (Sections 2, 4, 5, 6).
The local pinned source remains the authority for our actual implementation.

| Story step | Investigation and possible inference | Required presentation |
|---|---|---|
| 1. What is rewarded? | Explain Koh task/concept loss, MCBM task/concept/prototype loss, training noise, and which head reads h versus z. Include auxiliary terms and weighting; nominal coefficients alone do not measure gradient influence. | Short architecture recap, formulas, concrete positive/negative tail example. |
| 2. What did Standard establish? | Repeat every current Standard figure, including follow-ups and saved-head tests, with the same definitions and populations where available. | Copy the actual Standard construction; add gamma bars/lines only when readable, otherwise separate copies below it. Inventory every output, not just old numbered sections. |
| 3. Did compression happen? | Measure distance to +/-3, within-label variation and absent/present N/mean/SD for every h coordinate. | Added compression figure; raw z health remains the existing Standard construction. |
| 4. Did unwanted information decrease? | Repeat conditional log-loss gain and equal-three-coordinate analysis for h and z; labels define the structural baseline. | Preserve Standard A/B panels, controls and split rules. Decoding accuracy is not a substitute. |
| 5. Does the saved classifier use it? | Replace ordinary h magnitudes with fold-specific label means and rerun the saved MCBM species head. | Preserve the question and C/D visual construction; explain the nonlinear head and ordinary-image population. |
| 6. Did pixels gain control? | Repeat starting margin, donor increase, source decrease, final margin, response, outcome partition, visibility, confusion, support and species residuals for every part/gamma. | Exact Standard plots with gamma copies. Smaller raw margins alone are not improvement; print no-response and collapsed-output counts. |
| 7. Where does compression turn into a different response? | Inspect h-to-z transfer curves and slopes alongside h/z movement when saved swap h or verified frozen replay is available. | Added mechanism figure. Mark unavailable swap h explicitly; never infer h movement from z alone. |
| 8. What persists downstream after swaps? | Erase off-target h deviations and recompute the actual saved nonlinear head. | Standard intervention question repeated; do not reuse linear W-based e as an exact MCBM formula. Show class-gap/probability changes separately from unchanged concept margin. |
| 9. Which contributors survive? | Repeat Standard conflict, visibility, value-support and held-out prediction analyses. Audit all five parts; investigate surprising results without fixing a tail-only story in advance. | Same graphs and full per-part tables; group related swaps in evaluation folds. |
| 10. What does RLv2 distinguish? | Compare matched Standard-CBM/RLv2 evidence for changing supervision with MCBM evidence for changing representation penalty. | Preserve chapter order and graph construction. Do not describe the available comparisons as a factorial loss experiment unless matched cells/configurations establish that. |
| 11. Which loss is worth testing next? | Use measured failure modes to motivate prototype, categorical, visibility-aware or paired grounding objectives. | Clearly label proposals and predicted benefits/failure cases. No claim of improvement without a trained comparison. |

Working hypotheses, to be tested rather than written as results:

- Concept classification can be correct while magnitudes retain species clues.
- Compression can remove those clues while preserving a context-based concept
  predictor; training labels do not identify the pixel source of prediction.
- The same penalty can reduce both the original margin deficit and the response
  to changed pixels. Their difference determines the final margin; it does not
  by itself explain the learned mechanism.
- A learned concept head may amplify or flatten a compressed h interval.
- RLv2 removes one supervision conflict but cannot force locality by itself.
- A fall in the backwash predicate caused by zero response or collapse is not
  grounding repair. Keep all outcome categories and ordinary health visible.
- Lower task accuracy is not automatically evidence of successful nuisance
  removal. Check label sufficiency, optimization and exact-concept health.

For loss dynamics, distinguish algebraic incentives, frozen-checkpoint gradient
diagnostics, and observed training trajectories. Only saved training histories
can establish the latter. A final-checkpoint gradient cannot reconstruct why a
model learned its representation. Within-gamma comparisons must record any
learning-rate/recipe difference and the limited seed coverage.

The paper's information and intervention tests concern representation content
and internal edits. Our controlled image swaps additionally test whether the
encoder follows changed part pixels. A successful internal intervention does
not settle that question. The ideal information objective does not imply that
every finite trained model achieves it or that it generalizes to unseen swaps.

Completion requires an explicit Standard-output -> MCBM-output checklist,
including exclusions and reasons, followed by real execution and visual review.
The previous partial builder update is not complete parity.

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

Every output is introduced using notebook-02 connection -> question -> variables
and prediction -> method/exclusions -> complete axis/panel/color/denominator
guide with a numerical example where needed -> numbered figure -> literal
observation -> strongest alternative -> discriminating test -> limited
conclusion -> next question.

| Figure | Required question and output | Claim protected |
|---|---|---|
| 1 | Data/checkpoint/fixed-render inventory by gamma and seed | We know exactly what was compared |
| 2, 2b | MCBM implementation and compression plus every exact concept's raw-`z` spread, separation, balanced accuracy, and recall | `gamma` changed the intended quantity without hiding a broken exact output behind an average |
| 2c | Per-part target RMSE, within-label `h` spread, mean learned-head `|dz/dh|`, and locally flat-row fraction, with positive/negative/flat slopes printed | Separates compression of `h` from amplification or flattening by the learned `h -> z` concept head |
| 2d | Repeat all tail gamma outcomes after excluding every source/donor value-7 swap | Proves the one exactly collapsed gamma-zero tail output does not silently create the tail conclusion |
| 3 | Standard CBM baseline beside MCBM `gamma=0`, using identical predicates and fixed renders | The gamma-zero difference is training-noise/optimization baseline, not minimality |
| 4 | `response_delta = m_cf-m_orig` distribution and positive-response rate, every part x gamma | Did the inserted pixels move the scores donorward? |
| 5 | Final margin `m_cf` and controlled backwash rate `P(response_delta>0,m_cf<0)`, every part x gamma | Did the donor actually finish above the old source? |
| 6 | Forward and backward rates separately | Pooled directions are not hiding cancellation/bookkeeping error |
| 7, 7b | Visible-pixel strata plus the exact shared training label/visibility-conflict inventory | Occlusion and conflicting supervision are separated rather than conflated |
| 8, 8b | Exact inserted-value confusion for every part/gamma plus each value's species support | A two-slot margin is not hiding collapse; rarity is tested rather than asserted |
| 9 | Every supported source-species residual after exact value matching | The summary heatmap does not hide unequal species |
| 9b | Five-fold sequential accounting: part -> visibility -> exact values -> source species | A contributor earns credit only when held-out margin error falls |
| 10 | Held-out species decoding from known labels `c`, raw logits `z`, and internal slots `h`, by every block and model | Structural species information is separated from extra learned information |
| 11 | Authoritative `fb_recallv2` two-stage pairing, matched positive/negative bootstrap, and standardized raw-`z` gap | The pairing rule follows the labels actually present and is printed rather than assumed |
| 11b | Standard CBM and all gammas aligned on total backwash, visible backwash, shared label conflict, and exact-value error | The four notebook-02 measurements are directly comparable but never added |
| 12 | Standard CBM and all gammas: donor-species probability versus `m_cf`, separated by part | Grounding and downstream class cost remain distinct claims |
| 13 | Seed coverage and available seed-level replication | Separates supported findings from seed-1 gamma trends |
| 14 | Aligned summary of compression, response, final failure, exact-value error, and health | Minimality cannot be credited merely because one metric shrank |

The primary success metric is the controlled backwash predicate from notebook
02, not `ordering_correct` alone:

`response_delta > 0 and m_cf < 0`.

Example: `m_orig=-20`, `m_cf=-5` gives `response_delta=+15`. The model saw and
responded to the donor pixels, but the old source still wins by five logit units.

## Notebook 06: CUB/CUB70 standard MCBM

Notebook 06 begins only after Figures 1--14, including Figures 2c and 2d, in notebook 03 have been inspected.
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

Notebook 03 is done only when all important executed figures, including every
panel introduced as `2b`, `2c`, `2d`, `7b`, `8b`, `9b`, or `11b`, have been displayed
in chat and reviewed literally, seed limitations are visible in the figures, and
no old invalid renderer output is loaded. Notebook 06 is done only after the
same review plus explicit `same operation`, `weaker approximation`, or `not
available` labels for every FunnyBird question. Neither notebook may conclude
that identified contributors explain the residual completely unless the measured
residual actually approaches zero under a valid test.
