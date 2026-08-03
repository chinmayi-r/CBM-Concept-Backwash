# CUB70 CBM exploration plan

This document freezes the reasoning order before seeing the new CUB70 plots.
The CUB notebook must be exploratory: FunnyBird results suggest useful questions,
not answers that CUB is required to repeat.

## What the non-RL FunnyBird CBM notebook actually did

1. **Check that training worked.** Measure ordinary species and concept
   accuracy. If the model cannot solve its normal task, a strange intervention
   result is not interesting yet.
2. **Check untouched images.** On an ordinary bird, the concept that is present
   should score above an absent alternative. This establishes normal behavior
   only on familiar images.
3. **Delete one part.** Ask whether that part's own concept falls. A remaining
   score means other pixels can support the answer, but deletion alone does not
   identify those pixels.
4. **Validate the renderer edit.** Display original, swap, deletion, and changed
   pixels. No score is interpreted until the edit is visibly correct.
5. **Swap exactly one part.** Keep body, camera, and background fixed. Replace
   one part and ask whether the new part concept beats the old one. Tail was not
   assumed to fail; it emerged as the weakest part, while beak and eye were also
   imperfect and wing/foot were stronger controls.
6. **Check both directions and coverage.** Rule out reversed bookkeeping,
   missing directions, tiny samples, or a few species producing the result.
7. **Separate response from final victory.** Ask whether the inserted pixels
   moved the donor-versus-source comparison in the right direction even when the
   donor did not finish on top.
8. **Test visible pixel amount.** If poor visibility fully explains a part, its
   failures should disappear for large, clearly visible insertions.
9. **Test exact variants.** Some shapes or colors may be intrinsically harder.
   Inspect the full confusion matrix and every source/donor variant pair before
   calling a species effect.
10. **Only then test body/species.** Species and canonical part variant are tied
    together. Compare species only after accounting for variant, direction,
    visibility, and seed. The old notebook found remaining species differences,
    but did not independently swap the body, so this remained observational.
11. **Keep stored information separate.** Decoding species from concepts does
    not say which pixels produced a concept. It is supporting description, not a
    substitute for the part intervention.
12. **Check the species head.** A concept change moved donor-species probability
    only modestly. The demonstrated problem was explanation grounding, not broad
    task failure.
13. **End with two unresolved causes.** A controlled body swap was needed to
    prove body influence. Visibility-aware training labels were proposed later
    to test whether label/pixel conflict helped create the problem.

## What the non-RL FunnyBird MCBM notebook then did

1. **Ask a new question, without relabeling:** does the minimality penalty make
   each named score use the right pixels?
2. **Verify that gamma changed the representation.** Compression changed sharply
   from gamma 0 to 0.1 while ordinary accuracy stayed similar. Values above 0.1
   added little extra compression, so the experiment was mainly “off versus on.”
3. **Repeat the same validated swaps.** Use the same parts and decision rules as
   CBM. Minimality was not allowed to receive a different or easier test.
4. **Normalize within each model.** Gamma changes raw score scale. Measure how
   much of each model's original donor deficit was closed, not only raw numbers.
5. **Check seeds.** A connected gamma line can turn one seed into a false trend.
6. **Repeat visibility, deletion, variant, direction, and species controls.** The
   aim was to find which explanation changed and which survived minimality.
7. **Interpret the loss literally.** Minimality can force a score near +3 or -3
   without forcing it to read its named part. A body-based rule can be perfectly
   minimal.
8. **Limited conclusion.** Minimality compressed the bottleneck but did not
   guarantee grounding. Only after this conclusion was visibility-aware
   relabeling proposed as a separate causal follow-up.

## CUB has different objects, so inventory comes first

CUB uses 28 attribute types:

- bill color, bill length, bill shape;
- wing color, wing pattern, wing shape, primary-feather color;
- tail pattern, tail shape, upper-tail color, under-tail color;
- eye color;
- crown color, forehead color, nape color, head pattern;
- throat color;
- breast color, breast pattern, belly color, belly pattern;
- back color, back pattern, upperparts color, underparts color;
- leg color;
- whole-bird size and whole-bird shape.

CUB70 provides 11 masks: head, left eye, right eye, beak, neck, body,
left wing, right wing, left leg, right leg, and tail. The masks are less
specific than many attributes. For example, belly, breast, back, upperparts,
and underparts all have to use the one body mask. Size and whole-bird shape have
no valid local mask and must not enter a local grounding claim.

## New CUB70 CBM order

1. List all 28 attribute types, every selected concept value, all 11 masks, and
   the exact attribute-to-mask mapping.
2. Count images and species before looking at model behavior.
3. Count visibility separately for all 11 masks. For eyes, wings, and legs also
   count whether zero, one, or both sides are visible.
4. Describe concept structure before the model: number of values per attribute
   type and number of species carrying each positive concept.
5. Only now measure how often a positive concept label is paired with an absent
   named mask. This is label/mask disagreement, not relabeling and not a model
   result.
6. Check ordinary CUB70 CBM task accuracy, concept accuracy, per-type recall,
   and score spread. A constant score near 0.5 is model collapse, not grounding.
7. Compare visible and absent cases for every exact attribute type and every
   exact concept. Do not begin with eight pooled body parts.
8. For non-lateralized eye, wing, primary-feather, and leg concepts, test zero,
   one, or two visible sides; total area; largest-side area; and left-only versus
   right-only cases. Use the last comparison only as a pose/view warning.
9. Test visible area, species-matched comparisons, and negative-label controls
   only when the preceding plot produces an odd result that needs them.
10. Repeat the exact same tables for the full-CUB-trained CBM and compare matched
   concept points. This is a CBM-to-CBM comparison, not MCBM and not relabeling.
11. Display and inspect every figure before writing its observation. Stop with a
    limited CUB conclusion. MCBM comes only afterward.

## Revised stopping rule after the failed deletion and patch calibrations

The failed edit-based tests remain in the notebook because they show exactly why
CUB cannot inherit FunnyBird's renderer-level causal claim. They do not stop the
observational comparison chain. For every FunnyBird question, show the closest
CUB approximation beside it and label the difference:

| FunnyBird question | Closest CUB approximation | Maximum CUB conclusion |
|---|---|---|
| Does deletion leave the named concept active? | Natural mask absence, part-only/context-only failed edit, and small-patch target response | Local/contextual sensitivity candidate; not a clean deletion effect |
| Does visibility explain failures? | Same-species, exact-concept visible-minus-hidden contrast; mask area and visible-side dose | Visibility is associated with the score after partial composition control |
| Are labels positive when pixels are absent? | Positive-label/verified-mask disagreement by exact concept | Visibility-conflicting supervision is available; causal contribution needs matched retraining |
| Is species information stored in concepts? | Held-out species probe from each CUB concept block | Species information is encoded; it does not locate the pixels used |
| Are some variants intrinsically difficult? | Exact attribute-value recall, collapse guards, and per-concept visibility/violation points | Difficulty is heterogeneous across exact concepts |
| Does source species/body remain after variant control? | Exact-concept, species-matched residual analysis | Additional species/context association; still observational |
| Does the concept-layer problem damage classification? | Compare concept violation with species correctness/probability | Bound downstream cost without equating it with grounding |

Do not claim that FunnyBird mechanisms were completely identified. Visibility,
visibility-conflicting labels, and exact variant difficulty explain portions of
the result; source-body/species residuals remain and are not yet causal. The CUB
goal is converging evidence for the same candidate mechanisms, not numerical
elimination of all residual backwash. A calibrated 2-D donor-part swap is the
final optional strengthening step after this approximation chain, not a condition
for displaying the preceding evidence.

## Final beak/tail pilot now implemented

The final optional strengthening step is now a deliberately small, fail-closed
pilot rather than a broad CUB claim. It uses only clearly visible beak and tail
masks. For each target it compares a same-value paste with a different-value
paste made by the same crop/resize/mask procedure. The primary number is the
change in `donor z - source z` between those two pastes. Every saved intervention
sheet must be inspected before the number is interpreted.

A positive response means the pasted donor value changed the model in the
predicted direction. If the response is positive but the final donor-minus-source
margin remains negative, that is only a candidate CUB analogue of retained
source/context. It is not renderer-quality proof because donor pose, species,
lighting, mask shape, and resizing remain possible causes. The implementation is
`analysis/cub70_beak_tail_swap_pilot.py`; run it through
`analysis/run_cub70_beak_tail_swap_pilot.sh` inside an existing GPU allocation.

### Executed decision

The pilot produced 40 beak and 40 tail pairs and all sixteen saved sheets were
inspected. It failed the scientific direction gate: median donor response was
`-0.0037` for beak and `0.0000` for tail, with only 40% positive responses for
both parts. Target deletion was similarly inconsistent. Because the new donor
pixels did not first move the model reliably toward the donor, the negative final
donor-minus-source margins cannot be interpreted as CUB backwash. Preserve this
negative result and do not tune the pilot. CUB currently supports observational
ingredients, not renderer-strength causal backwash.
