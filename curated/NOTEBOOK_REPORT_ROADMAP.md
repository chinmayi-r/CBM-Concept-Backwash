# Notebook report roadmap

## Purpose

This document specifies a ground-up rewrite of the research notebooks. The
notebooks are technical reports, not execution logs. A reader should be able to
open any notebook, learn the required notation, understand why each analysis is
present, and distinguish a direct observation from an explanation or a causal
claim.

The fixed research order is:

1. non-RL FunnyBird data;
2. standard FunnyBird CBM discovery;
3. standard FunnyBird MCBM minimality test;
4. matched RLv2 causal follow-up;
5. CUB/CUB70 data and standard CBM;
6. CUB/CUB70 MCBM.

RL does not replace the standard-CBM discovery. CUB does not have to reproduce
the FunnyBird result, and a weak CUB approximation must not be described as the
same experiment as a clean FunnyBird intervention.

## 1. Claims that the report must distinguish

### Claim A: ordinary model health

The model learned the class task and the labelled concept task well enough that
a grounding analysis is meaningful.

This requires task accuracy, concept balanced accuracy, positive recall, raw
score spread, and positive-versus-negative label separation. It does not prove
that a concept score comes from the named pixels.

### Claim B: pixel response

Changing or observing the pixels of the named part changes the corresponding
concept score in the expected direction.

FunnyBird can test this causally with controlled deletion and replacement. CUB
can test a weaker observational version by comparing naturally visible and
hidden instances of the same labelled concept.

### Claim C: concept backwash

The inserted donor pixels affect the model, but the final score still prefers
the old source concept. This is stronger and more specific than poor recall,
species information, or a small deletion response.

FunnyBird supplies the accepted test:

```text
response_delta_i > 0 and final_margin_i < 0
```

CUB has no accepted donor insertion and therefore cannot currently establish
this exact predicate. CUB can supply converging evidence for context-dependent
concept prediction, not a renderer-quality causal backwash claim.

### Claim D: a proposed contributor

Visibility, label/mask conflict, exact visual value, value frequency, number of
alternatives, or species identity predicts some of the grounding failure.

An association supports a contributor hypothesis. It does not identify a cause
unless that factor was changed while the remaining inputs were held fixed.
Matched RLv2 is the causal test of the training-label contributor.

### Claim E: remaining unexplained behavior

After the measured contributors are included, a residual remains. The report
must show it rather than claiming that all causes have been found or that their
effects mechanically add to zero.

## 2. Model explanation required near the start

The first model notebook must introduce the implementation before using any
model variable. Later notebooks may link back to this section and give a short
recap.

### 2.1 Standard CBM computation

For image `i`:

```text
image x_i
   |
   v
image encoder f_theta
   |
   v
latent concept slots h_i = (h_i1, ..., h_iJ)
   |                         |
   |                         +--> learned concept heads --> raw logits z_i --> probabilities
   |
   +------------------------------> class head --> class logits --> species prediction
```

The implementation in `minimal_cbm` calls the latent vector `z`; this report
renames it `h` to avoid confusing it with the raw concept logit. The class head
reads `h` directly. Each concept head reads the corresponding slice `h_j`.
Training minimizes

```text
L_CBM = L_task + beta * L_concept.
```

The FunnyBird and CUB configs use learned `1 -> 3 -> 1` concept heads, not
identity heads. The report must keep latent `h_j` and raw concept logit `z_j`
distinct. Because prediction files omit `c_logits`, the report replays the saved
concept-head weights on saved `h` and asserts that `sigmoid(z)` reproduces the
saved `c_preds`. Legacy swap CSV columns beginning with `z_` actually contain
`c_logits`; the notebook must say this explicitly.

### 2.2 What the variables mean

| Symbol | Plain meaning | Permitted use |
|---|---|---|
| `x_i` | image `i` | model input |
| `y_i` | species/class label | task evaluation |
| `c_ij` | processed 0/1 label for concept `j` in image `i` | concept supervision |
| `h_ij` | encoder latent slot for concept `j`; class head input | architecture and species-code analysis |
| `z_ij = q_j(h_ij)` | raw concept logit after the learned concept head | primary grounding measurement |
| `p_ij = sigmoid(z_ij)` | bounded concept probability | thresholded performance only |
| `c_hat_ij = 1[z_ij > 0]` | predicted present/absent concept | recall and balanced accuracy |
| `v_ig` | whether released/renderer mask `g` is visible | visibility analysis |
| `a_ig` | visible area of mask `g` | visibility-strength analysis |

Simple example: if concept `j` is `yellow tail`, a large positive `z_ij` means
the model predicts yellow tail. The grounding question is not merely
whether that slot is correct on ordinary photographs. It is whether changing
the tail pixels changes this value appropriately.

### 2.3 FunnyBird replacement variables

For a source bird with source concept `s` and an inserted donor concept `d`:

```text
m_orig = z_donor,orig - z_source,orig
m_cf   = z_donor,cf   - z_source,cf
response_delta = m_cf - m_orig
```

- `response_delta > 0`: the replacement moved the model toward the donor.
- `m_cf > 0`: after replacement, the donor finishes above the source.
- `response_delta > 0` and `m_cf < 0`: the model saw some donor evidence but
  retained a stronger preference for the old source. This is the main
  FunnyBird backwash event.

Example: before replacement, `z_donor=-12` and `z_source=18`, so `m_orig=-30`.
After replacement, `z_donor=4` and `z_source=10`, so `m_cf=-6` and
`response_delta=24`. The donor pixels caused a large response, but the old
source still wins by 6 raw-score units.

### 2.4 CUB observational variables

CUB does not have a clean donor replacement. For exact concept `j` and its
available coarse mask `g`:

```text
visibility_effect_j = mean(z_ij | c_ij=1, v_ig=1)
                    - mean(z_ij | c_ij=1, v_ig=0)

context_gap_j = mean(z_ij | c_ij=1, v_ig=0)
              - mean(z_ij | c_ij=0, v_ig=0)
```

`visibility_effect_j` asks whether the score is higher when the named region is
visible. `context_gap_j` asks whether the model still distinguishes a positive
label from a negative label when that region is hidden. A positive context gap
shows that information outside the visible named region predicts the concept.
Neither quantity is a donor/source margin.

### 2.5 MCBM computation

MCBM adds a penalty that pulls each representation value toward a target derived
from its concept label. In the current implementation that target is `-3` for a
negative label and `+3` for a positive label. The loss is

```text
L_MCBM = L_task + beta * L_concept + gamma * L_representation.
```

Increasing `gamma` should compress `z_j` toward its labelled concept target. It
does not directly tell the model which pixels to use. The MCBM report must first
show that the penalty changed the representation, then test whether grounding
also changed. A smaller score scale is not itself better grounding.

## 3. Standard section template for every notebook

Every substantive section uses this order and these exact headings:

1. **Question.** One sentence.
2. **Variables and prediction.** Define the population, unit of analysis,
   outcome, comparison, and the result predicted by the hypothesis.
3. **Method.** State filters, matching, aggregation, seeds, and exclusions.
4. **Figure N.** A short literal title appears both in markdown and on the plot.
5. **How to read the figure.** Explain axes, colors, symbols, and reference lines.
6. **Literal observation.** Report what is visible without explaining it.
7. **Alternative explanations.** Give the strongest plausible alternatives.
8. **Discriminating test.** State which next analysis separates those
   alternatives.
9. **Limited conclusion.** State only the claim supported by this result.
10. **Next question.** Motivate the following section.

No section may introduce a result in its explanatory paragraph before the
figure has been executed and inspected.

## 4. Common presentation rules

### 4.1 Front matter

Each notebook begins with:

- research question;
- dataset and exact population;
- model, checkpoint epoch, gamma if applicable, and seeds;
- a compact variable glossary;
- a table of claims the notebook can and cannot establish;
- required input files with hashes or clear provenance;
- exclusions and missing-data counts.

### 4.2 One output, one purpose

Each output cell produces one numbered figure, one numbered example set, or one
numbered table. Diagnostic printing is hidden or moved to an audit appendix.

### 4.3 Raw score first

Use raw `z` for grounding, distributions, visibility effects, context gaps, and
species residuals. Use probability or thresholded predictions only for task
accuracy, concept balanced accuracy, positive recall, or a downstream class
probability question.

### 4.3b Explanation contract for every result

The notebook is written for a technical reader who has not seen the code or the
chat. Model notation is introduced once near the top and reused consistently.
Before every figure or result table, Markdown must provide:

1. the exact variable name and formula;
2. the population and unit of analysis (image, swap, exact concept, species, or
   seed);
3. the measurement unit and denominator;
4. the meaning of every axis, panel, row, color, marker, and reference line;
5. the expected direction under the stated hypothesis;
6. a small numerical example whenever signs, margins, residuals, rates, or
   differences could be confusing;
7. a plain-language restatement that still refers back to the defined variable.

For example, do not write only “higher response is better.” Write that the
x-axis is `response_delta = m_cf - m_orig` in raw-logit units; a value of `+8`
means the donor-minus-source comparison moved eight raw-logit units toward the
donor after the swap. Likewise, do not write only “species helps.” State that
adding species lowered held-out RMSE from one declared value to another on the
same rows, which supports predictive organization but is not a causal species
manipulation.

Axis labels may use the exact symbol or a short plain-language label. In either
case, the Markdown must connect the displayed label to the formal variable. No
table column, abbreviation, threshold, matching rule, or summary statistic may
appear unexplained. A reader must be able to interpret the output without
opening the code cell.

### 4.4 Avoid unreadable plots

- No horizontal jitter when the x-axis is a named category.
- No unlabeled spaghetti lines.
- No bar without its denominator.
- No row-level error bar when rows reuse images or species.
- Show seed values as points; summarize across seeds only when at least two
  seeds exist.
- Use aligned dot plots or heatmaps for many exact concepts.
- Use distributions for per-image margins, not only means.
- Label every CUB part `leg`; use `foot` only for FunnyBird.

### 4.5 Examples are selected by declared rules

Image grids are not decoration. Before selecting examples, define the requested
strata, such as high conflict/high residual, high conflict/low residual,
positive visibility response, and negative visibility response. Print image ID,
species, exact concept, label, mask state, mask area, raw score, and prediction
on every example.

### 4.6 Evidence status

Each section ends with one of:

- `ACCEPTED FOR <limited claim>`;
- `VALID TEST, NO SUPPORT FOR <prediction>`;
- `METHOD NOT CALIBRATED FOR <intended use>`;
- `INVALID OUTPUT`;
- `INCOMPLETE`.

Never print a bare `FAIL`.

## 5. Notebook 01: FunnyBird data and hypotheses

### Purpose

Establish the non-RL data structure that makes species shortcuts possible before
examining a model.

### Required sequence

1. **Dataset inventory.** Images, train/validation/test records, 50 species,
   concept slots, five parts, variants per part, and missing records.
2. **How a bird is generated.** One labelled diagram showing body/context and the
   five replaceable parts.
3. **Species-to-concept construction.** Heatmap of species by exact concept,
   ordered by part. State that many FunnyBird concepts are constant within a
   species.
4. **Number of alternatives.** Aligned dot plot of variants per part and species
   support per variant. Do not infer model behavior yet.
5. **Visibility and area.** Per-part visible fraction and area distributions,
   including zero/nearly-zero cases.
6. **Label/mask conflict.** For every exact concept, plot the fraction of positive
   labels whose corresponding part is not visibly present. Use an aligned dot
   plot with exact concept names and denominators.
7. **Pre-model hypotheses.** State the predicted directions:
   low visibility and high label conflict may encourage context use; fewer or
   more uneven alternatives may change difficulty; species-constant labels make
   species context predictive.
8. **Boundary.** These are available shortcuts, not proof that the trained model
   uses them.

### Main outputs

Approximately six figures and two tables. No model logits, no MCBM, and no RL.

## 6. Notebook 02: Standard FunnyBird CBM discovery

### Purpose

Test whether the standard CBM concept scores are grounded in their own part
pixels and then investigate the graded failures.

### Required sequence

#### 02.1 Model and population

Show the CBM diagram, loss, variables, exact checkpoint, seeds, and test
population. Print the saved concept-head replay assertion described in Section 2.

#### 02.2 Ordinary model health

Use a compact table plus an exact-concept raw-score panel:

- task accuracy;
- concept balanced accuracy and positive recall;
- `spread_j = Q95(z_j)-Q05(z_j)`;
- `label_separation_j = median(z_j|c_j=1)-median(z_j|c_j=0)`.

The figure should be a concept-by-metric heatmap or aligned dot plot. It answers
whether a slot is active and discriminative, not whether it is grounded.

#### 02.3 Intervention validity

Display the semantic renderer audit and a complete original/swap/delete/part-map
grid for tail, wing, beak, foot, and eye. Report changed pixels and hashes. This
must precede every intervention result.

#### 02.4 Does each part cause a score response?

Plot the distribution of `response_delta` for every part, with a zero reference
line and forward/backward directions shown separately. This answers whether the
inserted pixels moved the scores toward the donor.

#### 02.5 Does the donor finish above the source?

Plot the final margin `m_cf` for the same rows. Pair it with a two-dimensional
quadrant plot:

- x-axis: `response_delta`;
- y-axis: `m_cf`;
- upper-right: donor response and donor wins;
- lower-right: donor response but old source still wins.

This is the primary backwash figure. Report all five parts, not tail alone.

#### 02.6 Is the result an averaging-direction artifact?

Show forward and backward estimates independently with seed points. Do not pool
opposite directions into one unexplained rate.

#### 02.7 Could visibility explain the failure?

For the same swap rows, show final margin and candidate-event rate across
predeclared target-mask area bins. Then show a visible-only estimate using the
same definition for every part. Literal conclusion: whether visibility changes
the result and whether a residual remains.

#### 02.8 Could exact visual value explain the failure?

Show an all-part donor-value confusion matrix and per-value final-margin
distribution. Report counts. Avoid a selected top-20-only plot in the main text;
the complete matrix is primary and ranked extremes go in the appendix.

#### 02.9 Could frequency or number of alternatives explain it?

Relate exact-value support and part-level number of alternatives to the final
margin using labelled points and leave-one-part-out sensitivity. With only five
parts, do not present a correlation as stable evidence.

#### 02.10 Does source species explain additional variation?

Within exact source value and visibility strata, estimate source-species
residuals in final margin. Plot residual distributions with species counts. This
is observational because body/species context was not independently changed.

#### 02.11 Sequential accounting

Use the exact same swap rows throughout. Show two separate panels:

1. **Selection/description:** raw final margin, visible-only final margin, and
   exact-value-matched final margin.
2. **Modelled residual:** cross-validated error after adding visibility, exact
   source/donor value and support, then source species.

Do not mix RLv2 into this standard-model waterfall. Do not call the reduction
causal. Report the residual at the end.

#### 02.12 Does concept failure change species prediction?

Only here use downstream species probability. Plot donor-species probability
against final margin using independent bins and report the mean effect. The
expected conclusion may be that the explanation is unreliable while task impact
is comparatively small.

#### 02.13 Standard-CBM conclusion

Give a claim table:

- which parts show a donorward response;
- which parts often retain the source;
- which proposed contributors receive support;
- what remains observational or unexplained;
- why MCBM is the next question.

### Main outputs

Approximately 12 figures, two visual audit sets, and three compact tables.
Everything else moves to a methods appendix.

## 7. Notebook 03: Standard FunnyBird MCBM minimality

### Purpose

Test whether constraining each concept representation toward its label target
improves the grounding behavior discovered in notebook 02.

### Required sequence

1. Recap the exact standard-CBM finding; do not rediscover it from scratch.
2. Define `gamma`, `L_representation`, and the `-3/+3` targets.
3. State the mechanism prediction: higher effective minimality should reduce
   irrelevant contextual variation and improve final margins if that variation
   causes the failure.
4. **Did gamma change the representation?** Plot representation loss, `|z|`, raw
   spread, task accuracy, and concept balanced accuracy for every gamma. State
   explicitly if the effect saturates at gamma 0.1.
5. Repeat the notebook-02 response-delta and final-margin figures for every gamma
   and every part using the same fixed renders.
6. Show forward/backward directions.
7. Repeat visible-only and exact-value analyses.
8. Show all-gamma value-confusion matrices, not only gamma 0 and 0.1.
9. Show seed points for gamma values with seeds 1-3; label gamma values with only
   seed 1 as provisional breadth results.
10. Sequentially account for visibility, exact value/support, and source species
    at each gamma, then show the residual.
11. Conclude whether compression occurred and, separately, whether grounding
    improved. Do not infer grounding from scale compression.

### Main outputs

Reuse the visual language of notebook 02. A reader should compare panels without
learning new metrics.

## 8. Notebook 03rl: Matched RLv2 causal follow-up

### Purpose

Test one proposed cause: positive concept labels attached to images where the
part is not visibly present.

### Required sequence

1. Link the motivating standard-CBM visibility/label-conflict result.
2. Define `c_ij^RL = c_ij * visible_ig` and show concrete before/after label
   examples for all five parts.
3. Report changed-label counts and denominators by part.
4. Prove record identity and configuration parity. Only concept labels may
   differ; training/validation images and every training setting must match.
5. Prove checkpoint and fixed-render parity, including epoch, render IDs, and
   hashes.
6. State predictions before showing results:
   stronger donorward response, less negative final margin, and the largest
   improvement where label conflict was most common.
7. For the standard and RLv2 model on the same renders, plot paired changes in
   `response_delta` and paired changes in `m_cf` for every part.
8. Classify each standard candidate event as resolved, unchanged, or newly
   introduced. Use those explicit names, not “helped.”
9. Show forward/backward, exact value, source species, and downstream task
   consequences.
10. Show seed 1 as provisional as soon as accepted. Then add seeds 2-3 without
    rewriting the seed-1 history.
11. Show the full available gamma sweep. Never stop the displayed sweep at 0.1
    merely because those gammas have more seeds; distinguish replicated core
    points from one-seed breadth points.
12. State what RLv2 resolves and the residual it leaves. RLv2 tests one cause,
    not the existence of the original phenomenon.

## 9. Notebook 04: CUB/CUB70 data and measurement limits

### Purpose

Introduce the real-bird dataset before examining model behavior and explain why
its evidence cannot mechanically duplicate FunnyBird.

### Required sequence

1. Define full CUB, CUB70, species counts, exact attributes, groups, train/test
   populations, and why CUB70 exists.
2. List all 11 released masks and the eight coarse groups:
   head, eye, beak, neck, body, wing, leg, tail.
3. Account for every missing mask image and missing species.
4. Map every exact attribute type to its available mask or mark it untestable.
5. Show bilateral visibility and area separately before collapsing left/right.
6. Show label/mask conflict for every exact concept with denominators.
7. Show species/concept prevalence, value support, and number of alternatives.
8. Give the method-capability table:

   - clean deletion: FunnyBird yes, CUB no;
   - clean donor swap: FunnyBird yes, CUB no;
   - natural visibility: both, central in CUB;
   - species-matched recall and raw-score comparisons: both, with different
     matching rules.

9. End with preregistered CUB predictions, without using model outputs.

## 10. Notebook 05: Standard CUB70 CBM observational test

### Purpose

Ask whether a real-bird CBM shows the same observable ingredients as the
FunnyBird backwash mechanism while respecting that CUB lacks a controlled swap.

### Required sequence

#### 05.1 Model and population

Repeat the concise CBM diagram and variable definitions. State CUB70 checkpoint,
epoch, seed, 70-species task, 112 concept outputs, exact mask-matched population,
and all exclusions.

#### 05.2 Ordinary model health and exact collapse definition

For every exact concept, plot raw-score spread, label separation, balanced
accuracy, and positive recall. Define exact collapse as raw-score spread below a
declared numerical tolerance. Probability rounding is not collapse.

Use this section to exclude only genuinely unusable output slots. Positive
recall remains a health measure, not grounding evidence.

#### 05.3 Is a species shortcut available?

Show exact concept prevalence by species, species decoding from the complete
concept vector, and chance level. This shows availability, not use.

#### 05.4 How often does the label conflict with visible evidence?

For every exact concept, plot

```text
P(mask absent | processed concept label = 1)
```

as an aligned dot plot grouped by attribute type. The y-axis is a data fraction,
not model probability. Exact concepts occupy named rows; there is no horizontal
jitter between category labels.

#### 05.5 Does seeing the named region change raw z?

Plot `visibility_effect_j` for every testable exact concept, centered at zero,
with concept names and sample counts. Add group summaries only after the complete
exact-concept view. Test area dose response and left/right alternatives on the
same labelled-positive population.

#### 05.6 Does the model retain concept information when the region is hidden?

Plot `context_gap_j` for every exact concept, centered at zero. A positive value
means hidden positive examples still score above hidden negative examples. This
is contextual prediction, not a donor/source margin.

#### 05.7 Species-matched recall and raw-score gap

Restore the original CUB recall question using the refined valid matching rule:
join the original image-level CUB attribute annotations to the CBM predictions;
do not use the majority-voted processed training labels for this test. Candidate
species must provide sufficient raw positive and negative support, and positive
and negative support counts are equalized. Plot both:

- thresholded positive-recall difference, for continuity with the original
  notebook;
- matched raw-`z` difference, so saturation and thresholding cannot hide scale.

State the exact unit, matching variables, bootstrap unit, and number of eligible
species/concepts.

#### 05.8 Could conflict, difficulty, or support explain the effects?

Across exact concepts, relate `visibility_effect_j` and `context_gap_j` to:

- label/mask conflict rate;
- positive and negative sample support;
- number of values for the attribute type;
- number of species supporting the exact value.

Use labelled dot plots and cross-validated summaries. These are concept-level
associations and must not be mixed with row-level causal language.

#### 05.9 Does species explain variation within an exact concept?

Within the same exact concept and mask state, compare species using raw `z`.
Require adequate within-species support and report all exclusions. A persistent
species effect is evidence that context predicts the score; it does not prove
that the class label flowed backward through the network.

#### 05.10 Sequential observational accounting

Do not build one misleading waterfall from incompatible units. Use two panels:

1. **Row-level model:** within exact concept, add visibility/area, then species;
   report held-out prediction error and residual species variation.
2. **Concept-level model:** explain `visibility_effect_j` or `context_gap_j` using
   label conflict, support, and number of alternatives; report held-out error and
   residual exact-concept variation.

This is the CUB counterpart to “subtracting influences one by one.” It shows what
each measured block accounts for while preserving the observational boundary.

#### 05.11 Rule-selected photographs and masks

Display complete example sets for:

- high conflict and high residual;
- high conflict and low residual;
- strong positive visibility effect;
- negative visibility effect;
- high hidden context gap;
- genuinely collapsed slots, if any.

Each set displays the original photograph, all available masks, the mapped mask,
and the exact numerical record. Then decide whether pose, missing/coarse masks,
output collapse, or genuine contextual prediction is the most plausible reading.

#### 05.12 Direct FunnyBird/CUB comparison

Use a question-matched table:

| Scientific question | FunnyBird evidence | CUB evidence | Same operation? | Allowed conclusion |
|---|---|---|---|---|
| Model healthy? | raw z and accuracy | raw z and accuracy | yes | comparable health check |
| Named pixels matter? | controlled response delta | natural visibility effect | no | causal FB; observational CUB |
| Context remains? | negative final donor/source margin | hidden context gap/species effect | no | exact backwash FB; contextual prediction CUB |
| Visibility contributes? | same-render area stratification | natural mask visibility | weaker in CUB | contributor support |
| Labels contribute? | matched RLv2 later | not yet accepted | no | no CUB causal label claim |

#### 05.13 Causal boundary and next question

State whether CUB provides converging observational evidence, contrary evidence,
or an inconclusive mixture. Move reciprocal deletion, randomized masking, and
paste pilots to a methods appendix labelled with their precise status and the
limited information retained from each.

### Main outputs

Approximately 11 figures, three example sets, and four compact tables. Failed
edit proxies are appendix material and do not interrupt the report.

## 11. Notebook 06: CUB/CUB70 MCBM

### Purpose

Only after notebook 05 is fixed, ask whether minimality changes the accepted CUB
observational quantities.

### Required sequence

1. Recap the accepted notebook-05 observations.
2. Define the MCBM loss and gamma targets.
3. Verify model/checkpoint availability and population identity.
4. Show whether gamma changed representation scale and health.
5. Repeat the same exact-concept `visibility_effect`, `context_gap`, matched
   recall, raw-score species, and sequential-accounting figures.
6. Distinguish replicated gamma points from one-seed points.
7. Conclude whether minimality changes the observational behavior. Do not claim a
   CUB causal swap result.

## 12. Appendix structure

Every notebook uses the same appendices:

### Appendix A: provenance and audits

Input paths, hashes, checkpoint metadata, data joins, exclusions, and code
versions.

### Appendix B: complete tables

All exact concepts, all parts, all species passing declared support, and all
seeds. The main report may summarize but cannot hide omitted categories.

### Appendix C: sensitivity analyses

Alternative visibility thresholds, aggregation rules, and support thresholds.

### Appendix D: methods that were not accepted

For each failed proxy, state:

1. intended question;
2. method;
3. calibration criterion;
4. literal result;
5. precise status;
6. what remains usable;
7. why it is not used in the main conclusion.

This is where reciprocal deletion, randomized patch masking, and CUB paste
pilots belong.

## 13. Build and review order

### Phase 1: rebuild the standard-CBM pair

1. Freeze shared plotting vocabulary and CBM variable checks.
2. Rewrite notebook 01 only where required to supply the pre-model data facts.
3. Rebuild notebook 02 from the accepted fixed FunnyBird results.
4. Rebuild notebook 04 as the CUB data/measurement report.
5. Rebuild notebook 05 using raw `z`, restored recall, species controls, and the
   two-level sequential accounting.
6. Execute and export 02 and 05.
7. Export every numbered figure separately.
8. Display every important figure in chat with a plain caption before accepting
   its interpretation.
9. Record a figure-by-figure verdict: keep, revise, remove, or needs data.

### Phase 2: minimality

Rebuild notebook 03 with the same metrics and then notebook 06. No new metric is
introduced merely to make MCBM look different.

### Phase 3: causal label follow-up

Rebuild notebook 03rl after reconciling live jobs and accepted outputs. Keep
seed-1 provisional results visible; add replication and broad gamma results as
separate evidence layers.

### Phase 4: final report audit

For every main figure:

1. verify input population and denominator;
2. verify axes and variable definitions;
3. display the figure;
4. state the literal observation;
5. state the strongest alternative;
6. identify the discriminating test;
7. limit the conclusion;
8. confirm that the next section follows logically.

## 14. Definition of done

The notebook series is complete when a new technical reader can answer all of
the following without consulting chat history:

1. What is the implemented CBM and what does `z_j` mean in these runs?
2. Why are ordinary accuracy and recall not grounding tests?
3. What exactly changes in the FunnyBird swap?
4. What do `response_delta` and final margin separately measure?
5. Which parts exhibit the effect and how variable is it across seeds?
6. Which proposed contributors are supported, causally tested, or still
   observational?
7. What residual remains after measured contributors?
8. What does gamma change, and does grounding change with it?
9. What exactly did RLv2 change and what did it resolve?
10. Which CUB measurements answer the same scientific question, and which do
    not?
11. Why can FunnyBird support a causal backwash claim while CUB currently supports
    only converging or contrary observational evidence?
12. Which results are accepted, provisional, incomplete, invalid, or method-not-
    calibrated?

If any answer depends on an unexplained plot, a hidden exclusion, or a term that
was introduced after it was used, the notebook is not finished.
