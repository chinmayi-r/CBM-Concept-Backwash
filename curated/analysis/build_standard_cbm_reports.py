#!/usr/bin/env python3
"""Build the standard-CBM FunnyBird and CUB70 reports from first principles.

The notebooks deliberately contain analysis code but no embedded conclusions.
After execution, every numbered figure must be inspected before its literal
observation and limited conclusion are written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from pathlib import Path


HERE = Path(__file__).resolve().parent
CURATED = HERE.parent
NOTEBOOKS = CURATED / "notebooks"


def lines(source: str) -> list[str]:
    source = textwrap.dedent(source).strip("\n") + "\n"
    return source.splitlines(keepends=True)


def cell_id(tag: str, source: str) -> str:
    return f"{tag}-{hashlib.sha1(source.encode('utf-8')).hexdigest()[:12]}"


def md(tag: str, source: str) -> dict:
    source = textwrap.dedent(source).strip("\n") + "\n"
    return {
        "cell_type": "markdown",
        "id": cell_id(tag, source),
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(tag: str, source: str, alt: str | None = None) -> dict:
    source = textwrap.dedent(source).strip("\n") + "\n"
    if alt:
        source = f"# ALT: {alt}\n" + source
    metadata = {"alt": alt} if alt else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id(tag, source),
        "metadata": metadata,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


FIGURE_GUIDES = {
    "fb-q1": """
    Each row is one exact concept, such as `yellow tail`. The four panels use the
    same rows. `spread = Q95(z)-Q05(z)` asks whether the output changes across
    test images; exactly zero means a constant output. `label separation =
    median(z|c=1)-median(z|c=0)` asks how far positive-labelled images sit above
    negative-labelled images; positive is the expected direction. `balanced
    accuracy = (positive recall + negative recall)/2` gives positive and negative
    labels equal weight; 0.5 is chance for a binary concept. `positive recall =
    P(z>0|c=1)` is the fraction of labelled-positive images called positive.
    Example: positive recall 0.90 means 90 of 100 positive-labelled images have
    `z>0`. Dot color identifies the FunnyBird part: purple tail, blue wing,
    orange beak, green foot, and pink eye. The solid zero line marks no label
    separation; the dashed 0.5 lines mark chance balanced accuracy and 50%
    positive recall. These are health checks, not evidence about which pixels
    produced `z`.
    """,
    "fb-q2": """
    Figure 2a is the semantic preflight: for each of the five parts it shows the
    original, replacement, deletion, original part map, and replacement part
    map. Figure 2b is the compact four-column audit used by the analysis:
    original, replacement, deletion, and replacement part map. In both grids the
    named part should change while body, pose, camera, and background remain
    fixed. This validates the intervention before model scores are interpreted.
    """,
    "fb-q3": """
    Panel A puts part on the x-axis and `response_delta` in raw-logit units on the
    y-axis. The box spans the 25th--75th percentiles, the orange line is the
    median, and whiskers are the 5th--95th percentiles; outliers are omitted only
    from drawing. Zero means no donorward change and values above zero mean the
    donor gained relative to the old source. Panel B reports the fraction above
    zero, with `n` printed over each bar. Colors identify parts using the shared
    FunnyBird palette. This measures response size, not whether the donor wins.
    Example: a margin change from -20 before replacement to -5 afterward gives
    `response_delta=+15`, although the final margin remains negative.
    """,
    "fb-q4": """
    In the margin panel, zero separates donor wins (`m_cf>0`) from old-source wins
    (`m_cf<0`). In the quadrant panel, x is donorward movement and y is the final
    donor-minus-source score. The lower-right quadrant is the controlled
    backwash predicate `response_delta>0 and m_cf<0`: the new pixels moved the
    answer toward the donor, but the old source still finished higher. Boxes and colors use the Figure 3 definitions;
    translucent points are individual swaps and the legend maps color to part.
    Example: `m_cf=-5` means the old source finishes five raw-logit units above
    the donor.
    """,
    "fb-q5": """
    Each part has separate forward (`fwd`) and backward (`bwd`) replacement
    estimates, shown as unconnected circles and squares. The rate is
    the fraction of rows in the lower-right quadrant from Figure 4; the printed
    denominator is the number of swaps. Similar values in both directions argue
    against a pooled average hiding opposite effects. A rate of 0.60 means 60%
    of swaps in that direction satisfy both `response_delta>0` and `m_cf<0`.
    """,
    "fb-q6": """
    The x-axis bins swaps by the number of visible pixels in the inserted target
    part. One panel shows median final raw-logit margin; the other shows the
    responded-but-source-wins fraction. If visibility were the whole explanation,
    sufficiently large visible parts should make margins positive and drive that
    fraction near zero for every part. Point color identifies part; the table
    gives the exact denominator for every nonempty bin. The companion visible-only
    summary uses the same rule for all parts: `pixel_count_cf > 0`. A median
    margin of +3 means the donor finishes three raw-logit units above the source.
    """,
    "fb-q6b": """
    Each row is one exact concept. The x-axis is
    `P(visibility-aware label=0 | original label=1)`: the number of original
    positive training labels removed by the visibility rule divided by all
    original positive labels for that concept. A value of 0.25 means 25 of 100
    positive labels conflict with visible part evidence. Color identifies part.
    This is a data rate, not a model probability or causal model effect.
    """,
    "fb-q7": """
    Each heatmap row is the value actually inserted and each column is the value
    with the largest post-swap raw logit. A bright diagonal means the model names
    the inserted value; bright off-diagonal cells show systematic confusion.
    Every FunnyBird part and every value is included. The lower row gives the
    final-margin distribution for the same inserted values, with the number of
    swaps printed above each box. Thus recognition and retained-source margin are
    visible together rather than inferred from a diagonal rate alone. A diagonal
    value of 0.80 means the inserted value is highest in 80% of that row's swaps.
    """,
    "fb-q7b": """
    Each labelled point is one donor value. The x-axis is how many species support
    that value; the y-axis is the fraction of its swaps that responded donorward
    but still ended source-negative. A consistent downward or upward pattern
    would support frequency as an organizer. The number of alternatives is
    reported but cannot be cleanly separated with only five parts.
    """,
    "fb-q8": """
    First remove the average margin for the same part, source value, and donor
    value. Each remaining point is a source-species mean residual. Zero means that
    species behaves like the matched-value average; positive or negative values
    mean it systematically shifts the final margin. This is association with the
    unchanged bird context, not an independent species manipulation.
    """,
    "fb-q8b": """
    The y-axis is held-out species-classification accuracy from raw concept logits.
    The dashed reference is 1/50 chance. `all` uses all concept outputs; each part
    bar uses only that part's outputs. Above-chance decoding shows species
    information is available in the representation, not that it caused backwash.
    Accuracy 0.50 means half of held-out species are decoded correctly; chance is 0.02.
    """,
    "fb-q9": """
    Panel A compares the median final raw-logit margin for all rows, rows with a
    nonzero inserted-part mask, and rows with at least 100 inserted-part pixels;
    markers are unconnected because these are nested descriptive selections, not
    a trajectory. Panel B is held-out RMSE when predicting final margin; lower is
    better. Starting from part alone, blocks are added in order: visibility,
    exact source/donor values, then source species. A decrease means the new block
    predicts unseen render IDs better. The final nonzero error is the residual;
    an increase is negative evidence for that proposed organizer. RMSE 10 to 8
    is improvement; RMSE 10 to 11 is not.
    """,
    "fb-q10": """
    Swaps are divided into independent bins by final donor-minus-source concept
    margin on the x-axis. The y-axis is the model's mean probability for the donor
    species, with the number of rows printed per bin. This asks whether concept
    grounding failure has a downstream class effect; it is intentionally the one
    place where class probability, rather than raw concept `z`, is the outcome.
    """,
    "cub-q1": """
    Panel A shows, for each of the 11 released CUB masks, the fraction of 1,888
    joined photographs where mask area is at least 0.001 of image area, the
    declared visibility threshold.
    Panel B shows the median visible mask area divided by image area. `leg` is the
    CUB name; `foot` is never used here. Low coverage can mean true occlusion,
    pose, or missing/coarse annotation, which later photographs must distinguish.
    Visibility 0.25 means the released mask passes the threshold in 25% of the
    1,888 joined photographs.
    """,
    "cub-q2": """
    Each horizontal row is one exact concept. The x-axis is the number of the 70
    species carrying that exact value. Dot color is the number of positive
    photographs (yellow means more; purple means fewer), and dot area is the number of alternative
    values in the same attribute type (larger means more alternatives). A gray
    outline means no released-mask mapping. Example: x=20 means 20 species carry
    that value. This is label structure, not model behavior.
    """,
    "cub-q2b": """
    The y-axis is held-out species accuracy. The reference line is 1/70 chance.
    `all` uses all 112 raw concept logits; group bars use only logits mapped to
    that CUB region. Above-chance values mean species identity is encoded, not
    that species caused an individual concept prediction. Accuracy 0.14 is ten
    times the 1/70 chance rate, but is still only 14% species accuracy.
    """,
    "cub-q3": """
    Each row is one of 112 exact concepts and the four aligned panels have the
    same definitions as FunnyBird Figure 1: `spread=Q95(z)-Q05(z)`; `label
    separation=median(z|c=1)-median(z|c=0)`; balanced accuracy averages positive
    and negative recall; positive recall is `P(z>0|c=1)`. Zero spread within
    `1e-8` is the declared collapse rule. Moving right is healthier for the last
    three panels; spread only asks whether the output varies at all. For example,
    label separation +2 means the positive-label median is two logit units above
    the negative-label median.
    """,
    "cub-q4": """
    Each named row is one exact concept. The x-value is a data fraction:
    among images labelled positive for that concept, what fraction has no visible
    mapped mask? It is not a predicted probability. A value of 0.8 means 80 of
    every 100 positive-labelled examples lack a visible released mask.
    """,
    "cub-q5": """
    Each named point is one exact concept. The x-axis is
    `mean positive-labelled z when visible - mean positive-labelled z when hidden`.
    Right of zero means visibility accompanies a higher raw score; left means the
    visible group scores lower. Unlike a FunnyBird swap, these are different
    photographs, so pose, species composition, and mask quality can also differ.
    """,
    "cub-q6": """
    Each named point is one exact concept. The x-axis is the hidden-positive mean
    raw score minus the hidden-negative mean raw score. A value of +4 means that,
    even when the mapped region is absent, positive-labelled photographs score
    four raw-logit units above negative-labelled photographs. That is contextual
    prediction; it is not a donor/source margin and does not identify the cue.
    """,
    "cub-q7": """
    The bilateral panel compares mean raw `z` when zero, one, or two eye/wing/leg
    masks are visible. The area panel asks, within the same exact concept, whether
    larger visible masks accompany higher `z`. A steady upward pattern would fit
    local pixel reliance; mixed directions leave pose, species, and annotation as
    alternatives. The area outcome is
    `area_effect_j = mean(z | largest visible-area quartile, c=1) -
    mean(z | smallest visible-area quartile, c=1)`; +1 means the largest-area
    positive images score one raw-logit unit higher. Colors identify CUB groups.
    """,
    "cub-q8": """
    A matched pair contains two species with enough raw positive and negative
    examples for the same exact concept. Positive/negative counts are equalized.
    Both outcomes are absolute gaps, following the original recall notebooks.
    One panel shows `|recall_A-recall_B|`; the companion shows
    `|mean(z_pos,A)-mean(z_pos,B)|`. Zero means the matched species behave alike.
    For each species pair, the bootstrap resamples positive images within each
    species with replacement; the pair, not an individual image, is the summary
    unit. A recall gap of 0.30 is a 30-percentage-point difference; a raw-z gap
    of 2 is a two-logit-unit difference. These are health/species-dependence
    diagnostics, not grounding proof.
    """,
    "cub-q9": """
    The y-axis is held-out RMSE for predicting either the exact-concept visibility
    effect or context gap; lower is better. Starting from an intercept, conflict,
    image support, species support, and number of alternatives are added. A drop
    means the added concept-level information generalizes; a rise supplies no
    explanatory credit. RMSE 1.2 to 1.0 is improvement; 1.2 to 1.3 is not.
    This does not subtract causal effects.
    """,
    "cub-q10": """
    First subtract the mean raw `z` for the same exact concept and visible/hidden
    state. Each point then summarizes one species. Zero means the species matches
    that controlled average; remaining spread means species still organizes the
    score. Because photographs were not experimentally changed, this remains an
    observational context effect. A residual of +3 means that species lies three
    raw-logit units above the same concept-and-mask-state mean.
    """,
    "cub-q11": """
    The y-axis is held-out raw-`z` prediction error; lower is better. The same image
    rows are used throughout. Start with exact concept identity, add mask
    visibility and area, then add species. Each decrease measures extra predictive
    organization on unseen images. The remaining nonzero error is the residual,
    not automatically a new causal mechanism. RMSE 3.3 to 3.1 means the added
    block improves unseen-image prediction by 0.2 logit units.
    """,
    "cub-q12": """
    Each case occupies two rows: a mapped-mask-absent positive image followed by
    a mapped-mask-visible positive image for the same exact concept. Columns are
    the photograph, the mapped-region overlay, and every available released-mask
    overlay. Titles give species, exact concept, `c`, `c_hat`, raw `z`, and mapped
    area; tables give the selection rule and denominators. The images decide
    whether an extreme is genuine occlusion, missing/coarse annotation, or
    plausible contextual prediction. Collapsed concepts are excluded.
    """,
    "cub-q12b": """
    Each point is the same exact concept measured on the same photograph population
    by the CUB70-trained and full-CUB-trained CBMs after each model's raw logits
    are standardized within exact concept. The diagonal means equal standardized
    effect size. Agreement supports robustness to the training population;
    scatter or sign changes mean the magnitude is not stable across models.
    `(full=+0.5, CUB70=-0.5)` is a sign disagreement in within-concept standard-
    deviation units.
    """,
}


def question(tag: str, number: str, title: str, variables: str,
             prediction: str, method: str) -> dict:
    guide = textwrap.dedent(FIGURE_GUIDES[tag]).strip()
    # Keep this f-string at column zero. Indenting it while interpolating a
    # dedented multi-line guide makes the complete cell render as a code block.
    return md(tag, f"""## {number} · {title}

**Question.** {title}

**Variables and prediction.** {variables} {prediction}

**Method.** {method}

### Figure {number} · {title}

**How to read the figure.** {guide}
""")


REVIEWS = {
    "fb-r1": (
        "All 26 exact outputs vary substantially: raw-z spread is 20.77-56.88, "
        "label separation is 20.71-37.68, balanced accuracy is 0.966-1.000, and "
        "positive recall is 0.950-1.000. Image classification accuracy is 0.7504 "
        "and concept accuracy is 0.9944.",
        "Excellent label prediction can still come from species context rather than "
        "the named part, so this figure establishes health but not grounding.",
        "Use the same-image controlled replacement in Figures 3-4.",
        "ACCEPTED FOR seed-1 standard-CBM model health; no exact output is collapsed.",
        "Is the fixed renderer intervention itself valid?",
    ),
    "fb-r2": (
        "For tail, wing, beak, foot, and eye, the displayed replacement and deletion "
        "alter the named part while the body, pose, camera, and background remain fixed; "
        "the target part map contains the changed region.",
        "A few photographs alone would not certify the full cache.",
        "Retain the semantic preflight plus the accepted fixed-render hash/diversity "
        "validation across all evaluated models and render IDs.",
        "ACCEPTED FOR the validated FunnyBird fixed-render intervention.",
        "Do those inserted pixels move the raw concept comparison toward the donor?",
    ),
    "fb-r3": (
        "Median donorward movement is positive for every part, and the positive-response "
        "rates are tail 0.913, wing 0.999, beak 0.997, foot 0.995, and eye 0.993.",
        "A positive movement alone does not say that the inserted donor finishes above "
        "the old source.",
        "Inspect the final donor-minus-source margin jointly with response_delta.",
        "ACCEPTED FOR a causal within-image response to the inserted part pixels.",
        "After that response, which concept finishes stronger?",
    ),
    "fb-r4": (
        "The median final margin is negative for tail (-7.927) and beak (-1.368), near "
        "zero for eye (0.011), and positive for wing (13.665) and foot (13.559). The "
        "responded-but-source-wins rates are 0.608, 0.537, 0.491, 0.133, and 0.084, "
        "respectively.",
        "Starting preference, swap direction, target visibility, exact value difficulty, "
        "and source species could organize the unequal rates.",
        "Test those alternatives separately in Figures 5-9 without changing the event "
        "definition.",
        "ACCEPTED FOR the seed-1 controlled FunnyBird backwash predicate, strongest for "
        "tail, beak, and eye; wing and foot also contain minority events.",
        "Can swap direction create the pooled pattern?",
    ),
    "fb-r5": (
        "Forward and backward results have the same part ordering. Tail remains near "
        "0.60 in both directions, while wing and foot remain near 0.07-0.15; beak and "
        "eye remain intermediate.",
        "Individual source/donor value pairs can still be asymmetric even when pooled "
        "directions agree.",
        "Inspect every exact inserted value and both direction-specific denominators.",
        "ACCEPTED FOR excluding opposite-direction cancellation as the main explanation.",
        "How does exact target visibility change the event?",
    ),
    "fb-r6": (
        "Larger swapped-part masks generally move beak, eye, tail, and wing margins "
        "rightward and reduce their source-wins rate. Tail nevertheless remains negative "
        "through most bins and is non-monotone in the middle bins; wing and foot are "
        "already mostly donor-positive.",
        "Pixel count is associated with pose, source/donor value, and species, so bins "
        "do not isolate visibility causally by themselves.",
        "Hold exact values and species fixed, and test the visibility-aware label change "
        "later with matched RLv2 training.",
        "ACCEPTED FOR visibility as a contributor, not a sufficient explanation.",
        "Did ordinary training assign positive labels when the part was not visible?",
    ),
    "fb-r6b": (
        "Visibility-aware preprocessing removes 6,711 positive tail labels, compared "
        "with 334 beak, 247 eye, 42 foot, and 12 wing labels; every change occurs on a "
        "distinct training image.",
        "These are training-signal counts, not measured causal effects on the trained "
        "standard model.",
        "Compare otherwise matched standard and RLv2 checkpoints on the same fixed renders.",
        "ACCEPTED FOR a measured label/visibility conflict, especially for tail; causal "
        "credit remains deferred to notebook 03rl.",
        "Are some exact visual variants much harder than others?",
    ),
    "fb-r7": (
        "Post-swap donor-value recognition is graded: diagonal rates are tail 0.272, "
        "wing 0.807, beak 0.448, foot 0.905, and eye 0.501. Tail is weakest and foot "
        "strongest, rather than all parts failing equally.",
        "Different parts have different numbers and frequencies of variants, so raw "
        "diagonal rates are not directly interchangeable.",
        "Relate each donor value to species support and its part's alternative count.",
        "ACCEPTED FOR exact-value difficulty as an additional graded contributor.",
        "Do rarity or a larger choice set organize those value-level failures?",
    ),
    "fb-r7b": (
        "Within each part, source-species support does not show a consistent monotone "
        "relationship with the event rate. Tail variants remain high and wing/foot "
        "variants low across overlapping support values.",
        "The number of alternatives is constant within a part and therefore remains "
        "confounded with all other part-level differences.",
        "Use more independent part families or a design that changes choice-set size "
        "while holding pixels and species fixed.",
        "VALID TEST, NO CLEAR SUPPORT that frequency or alternative count alone explains "
        "the part ordering.",
        "Does unchanged source species organize what remains after exact values?",
    ),
    "fb-r8": (
        "After centering each part/source-value/donor-value combination, mean residuals "
        "still span roughly 20-35 raw-z units across the 50 source species in every part.",
        "The descriptive species means can also absorb pose or repeated-row composition, "
        "and they are not a causal body manipulation.",
        "Check whether species is recoverable from held-out concept vectors and whether "
        "species improves held-out margin prediction.",
        "ACCEPTED FOR an observational source-species association beyond exact values.",
        "Is species information actually present in the learned concept representation?",
    ),
    "fb-r8b": (
        "Species accuracy is 0.758 from all raw logits versus 0.02 chance; tail alone "
        "reaches 0.667 and wing 0.563, with every part block above chance.",
        "FunnyBird concepts help define species, so decodability shows available species "
        "information but not that species caused a particular concept error.",
        "Ask whether adding source species improves held-out prediction of final margin.",
        "ACCEPTED FOR substantial species information in the learned concept vector.",
        "How much of the swap margin generalizes from the proposed explanatory blocks?",
    ),
    "fb-r9": (
        "Held-out RMSE improves only from 11.467 to 11.106 when visibility is added. It "
        "then worsens to 11.225 with exact values and 12.062 with source species.",
        "High-cardinality categorical means may be too sparse or poorly regularized, but "
        "that possibility cannot be counted as positive evidence.",
        "Use seed replication or a preregistered hierarchical/regularized predictor "
        "before assigning generalizing explanatory credit to exact values or species.",
        "VALID TEST, NO SUPPORT from this predictor that exact values or source species "
        "account for held-out margin variance; only visibility gives a small improvement.",
        "Does the concept-layer margin have a large downstream class consequence?",
    ),
    "fb-r10": (
        "Mean donor-species probability rises from approximately zero at negative margins "
        "to 0.107 in the most donor-positive bin, with about 500 rows per bin.",
        "A one-part replacement need not make the whole donor species plausible because "
        "the unchanged body and other parts still belong to the source.",
        "Replicate across seeds and compare class-logit changes, not only final probability.",
        "ACCEPTED FOR a monotone but modest single-swap downstream donor-species effect; "
        "the primary harm here is explanation fidelity.",
        "Does minimality change the accepted standard-CBM quantities?",
    ),
    "cub-r1": (
        "The export contains 1,976 images, 70 species, and 112 concepts; the mask join "
        "retains 1,888 images, 67 species, and 107 concepts. Head/body/beak masks are "
        "present in over 92% of joined images, tail in 81%, bilateral wing/leg masks in "
        "about 60-65%, and eye/neck masks in only about 22-23%.",
        "An absent released mask can mean missing/coarse annotation rather than physical "
        "occlusion.",
        "Inspect bilateral counts, areas, and real images with all masks overlaid.",
        "ACCEPTED FOR the stated CUB70 mask-analysis population and coverage limits; 88 "
        "images, three species, and five concepts are outside the mask-matched population.",
        "Is a species/concept shortcut available before looking at model behavior?",
    ),
    "cub-r2": (
        "Exact values vary widely in supporting species and positive images, and attribute "
        "types contain one to six selected alternatives. Several size/shape concepts have "
        "no released-mask mapping.",
        "Uneven label structure only makes a shortcut possible; it does not show model use.",
        "Decode held-out species from the raw concept vector and individual part blocks.",
        "ACCEPTED FOR uneven species/concept structure and an available contextual shortcut.",
        "Does the learned representation actually store species information?",
    ),
    "cub-r2b": (
        "Species accuracy is 0.221 from all 112 logits versus 1/70 chance. Individual "
        "blocks are also above chance, led by wing 0.228, body 0.214, head 0.211, and "
        "tail 0.207.",
        "Concepts naturally define bird species, so decodability is availability evidence, "
        "not proof that species caused any one score.",
        "Hold exact concept and mask state fixed before estimating species differences.",
        "ACCEPTED FOR species information in the CUB70 concept representation.",
        "Are the exact concept outputs healthy enough to interpret?",
    ),
    "cub-r3": (
        "Task accuracy is 0.1412 and concept accuracy 0.7105. Of 112 exact outputs, "
        "has_throat_color::grey is constant-positive and "
        "has_wing_pattern::multi-colored constant-negative; both have zero raw-z spread, "
        "zero label separation, and balanced accuracy 0.5. The other 110 outputs vary.",
        "Low or uneven performance can reflect the CUB70 training setup and label noise; "
        "it does not by itself establish grounding failure.",
        "Keep the two collapsed slots out of positive grounding claims and analyze all "
        "remaining slots with raw z.",
        "ACCEPTED FOR 110 non-collapsed outputs; the two named collapsed outputs are "
        "unusable and remain explicit negative health results.",
        "How often is a positive label paired with no released mapped mask?",
    ),
    "cub-r4": (
        "The positive-label/mask-absence fraction ranges from near zero to above 0.9. "
        "Body concepts are usually low, wing concepts are commonly about 0.13-0.28, and "
        "several throat/tail concepts are much higher.",
        "Figure 12 shows that mask absence sometimes occurs even when the anatomical "
        "region is visibly present, especially for neck and beak mappings.",
        "Inspect real photographs and all masks, then treat v=0 as released-mask absence "
        "rather than guaranteed physical occlusion.",
        "ACCEPTED FOR label/released-mask conflict, not for an exact physical-occlusion rate.",
        "Do positive-labelled raw scores differ when the mapped mask is present?",
    ),
    "cub-r5": (
        "Across 48 eligible exact concepts, visibility_effect ranges from -0.917 to 1.124 "
        "raw-z units. Body and color concepts are often positive, while several bill, "
        "tail, and wing pattern/shape concepts are negative.",
        "Visible and mask-absent photographs differ in species, pose, background, and mask "
        "quality; negative effects need not be inverse pixel use.",
        "Test bilateral/area dose response, species matching, same-image model robustness, "
        "and real-image mask quality.",
        "VALID OBSERVATIONAL TEST with mixed support: some concepts score higher with the "
        "mask present, but there is no universal CUB visibility response.",
        "When the mapped mask is absent, does contextual label separation remain?",
    ),
    "cub-r6": (
        "For 50 eligible exact concepts, context_gap is nonnegative and is positive for 48; "
        "it reaches 8.267 for yellow throat, 7.454 for buff throat, and above 4 for some "
        "tail patterns. The two zero gaps are the collapsed outputs.",
        "Released-mask absence is a noisy proxy: species, pose, background, annotation "
        "quality, and visibly present but unmasked regions can all create separation.",
        "Match species support, center within exact concept/mask state, and inspect the "
        "selected photographs and masks.",
        "ACCEPTED FOR contextual label separation under released-mask absence; this is "
        "observational and is not a donor/source margin or causal CUB backwash proof.",
        "Can bilateral visibility or region area explain the score patterns more simply?",
    ),
    "cub-r7": (
        "Mean raw z is not monotone in zero/one/two visible sides for eye, leg, or wing. "
        "Within-concept area effects also span positive and negative values in every major "
        "group.",
        "Species and pose composition can overwhelm a natural-image area comparison, and "
        "small masks may be missing rather than physically absent.",
        "Hold exact concept and species fixed and evaluate held-out row-level prediction.",
        "VALID TEST, NO SUPPORT for bilateral count or area as a sufficient universal "
        "explanation; local visual evidence may still matter for individual concepts.",
        "Does performance differ by species after raw-label support is matched?",
    ),
    "cub-r8": (
        "All 221,312 rows align to original per-image CUB labels; all 112 concepts yield "
        "eligible species and 5,190 matched pairs. Mean absolute positive-recall gaps reach "
        "about 0.53, and positive-row raw-z gaps range from zero to about 13.",
        "Species still differ in pose, background, annotation certainty, and image quality; "
        "some matched supports are as small as three positives.",
        "Replicate at the seed level and test species after exact concept and mask state "
        "with held-out images.",
        "ACCEPTED FOR observational species-dependent concept performance after raw-label "
        "support matching, not for causal species backwash.",
        "Do conflict, support, and alternatives organize the concept-level effects?",
    ),
    "cub-r9": (
        "The supplied render used different populations for the two outcomes (87 concepts "
        "for visibility_effect versus 48 for context_gap), so its RMSE curves are not a "
        "valid linked comparison of the contributor sequence. The revised cell fixes both "
        "outcomes to the same non-collapsed population with at least ten visible positives, "
        "ten hidden positives, and ten hidden negatives.",
        "Changing eligibility can change both baselines and apparent predictor gains, so the "
        "old numerical comparison cannot be carried forward.",
        "Rerender this single corrected shared-population analysis, then compare the two outcomes.",
        "INCOMPLETE: code and population are corrected; rerendered Figure 9 must be inspected before assigning contributor credit.",
        "Does species-dependent raw-z variation remain within concept and mask state?",
    ),
    "cub-r10": (
        "After centering within exact concept and mask state, species residuals retain wide "
        "ranges in all eight groups: approximately -31.6 to 10.2 for head, -24.2 to 8.6 "
        "for tail, and -20.5 to 10.5 for neck, with smaller but nonzero ranges elsewhere.",
        "Small or uneven concept/state/species cells and correlated pose/background can "
        "produce extreme descriptive residuals.",
        "Require held-out image prediction and shrunken estimates before giving species "
        "generalizing explanatory credit.",
        "ACCEPTED FOR a descriptive species association after exact concept and mask state.",
        "Does species reduce held-out row-level prediction error?",
    ),
    "cub-r11": (
        "Held-out raw-z RMSE changes from 3.285 with exact concept alone to 3.262 after "
        "visibility/area and 3.104 after species; MAE changes from 1.869 to 1.855 to 1.700.",
        "Species can proxy pose, habitat, background, and collection effects, so predictive "
        "gain does not isolate a biological species-to-concept causal path.",
        "A matched relabel/retrain or valid same-image intervention would be needed for a "
        "causal claim; neither is accepted for CUB yet.",
        "ACCEPTED FOR generalizing contextual organization by species beyond exact concept "
        "and released-mask state; the causal source remains unresolved.",
        "Do real images show true occlusion, missing masks, or pose artifacts at the extremes?",
    ),
    "cub-r12": (
        "The supplied render is defective: the high-conflict/high-gap statistic names "
        "has_throat_color::white, but its visible panel displays the collapsed "
        "has_throat_color::grey example. The old grid also omits the mapped-mask-only view "
        "and complete per-image records. The revised cell excludes collapsed concepts and "
        "asserts that every displayed record matches the selected exact concept.",
        "Because the photograph/statistic join was wrong, that pair cannot distinguish "
        "physical occlusion from missing annotation.",
        "Rerender the corrected six-panel-per-case audit and inspect every selected pair.",
        "INCOMPLETE: corrected Figure 12 must be rendered and every photograph/mask pair inspected before a verdict.",
        "Do context and visibility patterns transfer between CUB70 and full-CUB training?",
    ),
    "cub-r12b": (
        "The supplied render puts unstandardized raw-logit effects from two separately "
        "trained models against an identity line. Since their logit scales differ, distance "
        "from that line has no valid magnitude interpretation. The revised cell standardizes "
        "raw z within exact concept separately for each model before computing effects.",
        "Sign agreement in the old plot remains descriptive, but diagonal distance cannot "
        "support a transfer claim.",
        "Rerender the standardized same-image comparison before judging effect-size stability.",
        "INCOMPLETE: the standardized same-image Figure 12b must be rendered before judging robustness.",
        "What can be concluded directly across FunnyBird and CUB?",
    ),
}


def review(tag: str, figure: str) -> dict:
    literal, alternative, test, conclusion, next_question = REVIEWS[tag]
    return md(tag, f"""
    ### Review record for {figure}

    - **Literal observation:** {literal}
    - **Strongest alternative explanation:** {alternative}
    - **Discriminating test:** {test}
    - **Limited conclusion:** `{conclusion}`
    - **Next question:** {next_question}
    """)


def notebook(cells: list[dict], old_path: Path) -> dict:
    metadata = {}
    if old_path.exists():
        metadata = json.loads(old_path.read_text(encoding="utf-8")).get("metadata", {})
    metadata.setdefault("kernelspec", {
        "display_name": "Python 3", "language": "python", "name": "python3",
    })
    metadata.setdefault("language_info", {"name": "python", "version": "3"})
    return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}


COMMON_MODEL = r"""
## The implemented CBM and the notation used below

For image `i`, the encoder produces a latent concept vector `h_i` with one
slot for each of the `J` exact concepts.  The learned concept head for slot `j`
turns `h_ij` into the raw concept logit `z_ij`:

```text
x_i → image encoder → h_i = (h_i1, …, h_iJ)
                          ├→ learned head q_j(h_ij) → z_ij → sigmoid → p_ij
                          └→ class head on complete h_i       → species prediction
```

The implementation trains with

`L_CBM = L_task + beta × L_concept`.

The class head reads the complete latent vector `h_i`; it does not read a list of
hard 0/1 concept decisions. Thus species loss can shape the same latent slots
that the concept heads read. In these runs each concept head is a learned
`1 → 3 → 1` network, not the identity. The setup cell replays the saved head
weights on saved `h_i` and verifies that `sigmoid(z_ij)` exactly reproduces
the saved probability.

| Symbol | Meaning |
|---|---|
| `x_i` | image `i` |
| `y_i` | species label |
| `c_ij` | processed 0/1 label for exact concept `j` |
| `h_ij` | encoder's latent slot for concept `j`; also read by the class head |
| `z_ij = q_j(h_ij)` | raw concept logit after the learned head; primary grounding quantity |
| `p_ij = sigmoid(z_ij)` | bounded probability; used only for thresholded performance |
| `c_hat_ij = 1[z_ij>0]` | predicted concept presence |
| `v_ig` | whether mapped part mask `g` is visible |
| `a_ig` | visible area of mask `g` |

`L_task` is the species-classification loss. `L_concept` is the sum of the
per-concept label losses. `beta` controls their relative weight. No later plot
uses the encoder slot `h_ij` while calling it a concept logit: grounding plots
use the post-head raw score `z_ij`.

Ordinary accuracy and recall answer whether predictions agree with labels. They
do **not** answer whether the prediction came from the named pixels.
"""


FB_DATA_DESIGN = r"""
## Dataset design and report population

FunnyBird is synthetic, so the relevant objects are known exactly rather than
estimated from photographs.

| Item | Value used here | Why it matters |
|---|---:|---|
| species | 50 | unchanged species/body appearance is the possible contextual signal |
| named parts | `tail`, `wing`, `beak`, `foot`, `eye` | these are the only five FunnyBird part names used below |
| exact concepts | 26 part values across the five parts | for example, `tail::blue`; a part and its exact value are not interchangeable |
| held-out model-health population | 5,000 test images | used for Figure 1 and the species decoder |
| controlled swap population | accepted fixed-render seed-1 CSV | the same validated rendered images are reused across model comparisons |

Species determine part values in FunnyBird, so species context can predict a
concept label even when the named part is hard to see. That makes contextual
prediction possible, but it does not prove the trained CBM used context. The
controlled replacement in Figures 2–4 supplies that stronger test.
"""


FB_PROOF_ROADMAP = r"""
## What this notebook must prove, and how the figures build the claim

The final claim is deliberately narrower than “the model is inaccurate.” A
FunnyBird **backwash event** requires both of the following on the *same
controlled replacement*:

1. the donor part moves the raw concept comparison toward the donor
   (`response_delta > 0`); and
2. after that movement, the old source concept is still higher
   (`m_cf < 0`).

Example: replacing a red tail with a blue tail raises the blue-tail score
relative to red by 24 units, but red still finishes 6 units above blue. The
model reacted to the new tail pixels, yet its final concept answer remained
attached to the old bird. That is the event tested in Figure 4.

Before accepting it, the report must establish these components in order:

| Step | Needed fact | Figure(s) | Why it is needed |
|---|---|---|---|
| 1 | the trained concept outputs are usable | 1 | a constant or broken output cannot support grounding analysis |
| 2 | the renderer really changed only the named part | 2 | otherwise a score change cannot be assigned to that part |
| 3 | the inserted pixels cause donorward movement | 3 | proves the model saw some evidence in the new part |
| 4 | the old source can still win after that movement | 4 | this is the controlled backwash predicate |
| 5 | the event is not a direction-averaging artifact | 5 | checks forward and reverse replacements separately |
| 6 | test proposed contributors | 6–8 | visibility/occlusion, conflicting labels, exact-value difficulty, support/alternatives, and source species |
| 7 | measure what those contributors predict and what remains | 9 | prevents claiming that a plausible story explains all rows |
| 8 | measure downstream class impact | 10 | separates explanation failure from species-classification harm |

### The three contributor hypotheses carried into both reports

The linked comparison tests the same three proposed reasons in the same order:

1. **visibility/occlusion:** the named pixels may be absent or too small;
2. **label–visibility conflict:** training may call a concept positive when its
   mapped region is not visible; and
3. **exact-value difficulty:** some variants may be intrinsically harder, rarer,
   or drawn from a larger alternative set.

Only after those are measured do we ask whether unchanged source species/body
context organizes the remaining raw-score error.  That fourth term is a
residual association, not a promise that the three measured reasons sum to the
whole phenomenon.

The implementation retains the complete renderer audit, all exact values,
species residuals, recall/model-health controls, and provenance inherited from
the earlier curated report and the original renderer-swap and recall notebooks.

### Capabilities and limits that determine this design

FunnyBird supplies an exact renderer mask and a clean donor-part replacement:
body, pose, camera, and background can remain unchanged while one part changes.
That makes Figures 3–4 causal tests of the changed part pixels. Visibility,
training-label conflict, exact value, support, and species are then investigated
as possible contributors. Except for the later matched RLv2 retraining, those
contributor analyses are observational and are not allowed to erase the
controlled event or claim that every cause has been found.

### Predictions stated before the results

- If the concept is locally grounded, replacement should produce
  `response_delta > 0` and usually `m_cf > 0`.
- If backwash occurs, a nontrivial set should have `response_delta > 0` but
  `m_cf < 0`.
- If visibility/occlusion is sufficient, the event should disappear for large,
  clearly visible inserted parts.
- If label–visibility conflict contributes, parts with more positive labels on
  invisible parts should later improve most under matched RLv2 training.
- If exact-value difficulty or species context contributes, matched rows should
  retain systematic value- or species-linked differences.
- None of these predictions requires the measured contributors to reduce the
  remaining error to zero.
"""


CUB_PROOF_ROADMAP = r"""
## What this notebook can establish, and how it approaches the FunnyBird question

The broad research question is the same: does a concept score depend only on its
named region, or does surrounding species/body context help predict it? The
strongest FunnyBird result cannot be copied mechanically because CUB has no
renderer that replaces one part while holding the rest of the photograph fixed.
Therefore this notebook does **not** invent a CUB donor/source margin.

The CUB conclusion must instead be assembled from explicitly observational
predicates, in this order:

1. the available photograph, species, concept, and mask populations are known;
2. species/concept structure makes contextual shortcuts available;
3. positive labels and mapped-mask visibility sometimes disagree;
4. each exact concept output is healthy enough to interpret;
5. natural visibility of the mapped region changes raw concept `z`;
6. positive and negative labels remain separable in `z` when the mapped region
   is absent (`context_gap > 0`);
7. species still organizes `z` after exact concept and mask state are held fixed;
8. measured visibility, conflict, difficulty, support, and species account for
   some—but not necessarily all—held-out variation.

### The same three contributors, with CUB-valid substitutions

1. **visibility/occlusion** uses the released mapped mask, its area, and
   bilateral alternatives; this is a natural-image comparison, not a swap;
2. **label–visibility conflict** is the positive-label/mapped-mask-absent rate
   for every exact concept, retaining coverage counts; and
3. **exact-value difficulty** uses raw-logit health, recall, exact-value support,
   number of alternatives, and species support.

Species/body context is then tested as the remaining observational organizer.
The recall analysis restores the original standard-CBM CUB question while
borrowing only the matching and bootstrap refinements from `mcbm_recallv4`;
no MCBM numerical result is imported here.

| FunnyBird scientific question | CUB operation | Figure(s) | Claim boundary |
|---|---|---|---|
| Are the data/model outputs usable? | inventory, masks, conflict, raw-`z` health | 1–4 | same health question |
| Is species context available? | label structure and held-out species decoding | 2, 4b | availability, not causal use |
| Do named-region pixels matter? | compare naturally visible and hidden positive-labelled photographs | 5, 7 | weaker than a same-image swap |
| Does context retain concept information? | hidden-positive minus hidden-negative raw `z` | 6 | contextual prediction, not donor/source backwash |
| Are scores species-dependent? | matched recall/raw-`z` gaps and within-concept species residuals | 8, 10 | observational species association |
| What proposed contributors organize the result? | concept- and row-level held-out accounting | 9, 11 | prediction, not causal subtraction |
| Could masks be misleading? | rule-selected photographs with all masks | 12 | separates true occlusion from annotation limits |
| Does the pattern depend on CUB70 training? | same-image full-CUB guard | 12b | robustness check |

### CUB capabilities and drawbacks used in the design

CUB provides 112 exact labelled concepts, species labels, real photographs, and
11 released anatomical masks. It permits raw-score health, natural visibility,
area, bilateral-mask, recall, species, and support analyses. Its masks are
coarser than many named attributes and can be absent even when a human can see
the region. It has no accepted clean deletion or donor swap. Consequently:

- `visibility_effect` asks whether visible positive-labelled photographs score
  differently from hidden positive-labelled photographs;
- `context_gap` asks whether context distinguishes positive from negative labels
  when the mapped region is absent;
- neither quantity is the FunnyBird final margin;
- photographs and mask examples must be inspected before interpreting extremes;
- converging results support context-dependent prediction, but only FunnyBird's
  controlled swap establishes the exact backwash event.

### Predictions stated before the results

- Healthy outputs should have nonzero raw-`z` spread, positive label separation,
  and above-chance balanced accuracy/recall.
- If local visibility helps, `visibility_effect` should usually be positive and
  larger visible areas should usually accompany higher `z`.
- If context predicts the concept without the mapped region, `context_gap`
  should remain positive and species should explain held-out variation within
  exact concept and mask state.
- If conflict/support/number of alternatives are sufficient explanations, adding
  them should lower held-out concept-level error. If not, a residual remains.
- Mixed or negative visibility effects must be investigated as pose, mask
  quality, collapse, or composition before being called evidence for backwash.
"""


def build_funnybird() -> dict:
    cells: list[dict] = [
        md("fb-title", r"""
        # 02 · Standard FunnyBird CBM: controlled discovery of concept backwash

        **Report question.** When one FunnyBird part is replaced while body, pose,
        camera, and background stay fixed, does the corresponding concept answer
        follow the inserted part or remain attached to the old bird?

        **Population.** Standard non-RL CBM. This notebook contains no MCBM and no
        visibility-aware relabelled model. Seed-level replication is shown where
        accepted outputs exist; the fixed-render causal analysis begins with seed 1.

        **Claims available here.** FunnyBird's renderer permits a controlled
        donor-part replacement. Therefore a validated positive donor response plus
        a remaining source preference can establish the CBM backwash event. Proposed
        explanations are weaker unless independently manipulated.
        """),
        md("fb-roadmap", FB_PROOF_ROADMAP),
        md("fb-model", COMMON_MODEL),
        md("fb-data-design", FB_DATA_DESIGN),
        code("fb-setup", r"""
        import os, json, re, glob, sys, hashlib, subprocess
        from pathlib import Path
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from IPython.display import display, Image as DisplayImage

        CURATED = Path(os.environ["CURATED_DATA"])
        CWD = Path.cwd()
        REPO = CWD if (CWD/"analysis").is_dir() else CWD.parent
        sys.path.insert(0, str(REPO/"data"/"funnybirds"))
        plt.rcParams.update({"figure.dpi": 120, "axes.grid": False})
        pd.set_option("display.max_rows", 250)
        pd.set_option("display.max_columns", 40)
        ORDER = ["tail", "wing", "beak", "foot", "eye"]
        COLORS = {"tail":"#6A0DAD", "wing":"#0072B2", "beak":"#E69F00",
                  "foot":"#009E73", "eye":"#CC79A7"}

        def require(path, command):
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}\nProduce it with: {command}")
            return path

        swap_candidates = [
            CURATED/"swap_fixed_v3_matched"/"funnybirds-cbm-s1.csv",
            CURATED/"swap_fixed_v2_attempt2"/"funnybirds-cbm-s1.csv",
            CURATED/"swap_fixed_v2"/"funnybirds-cbm-s1.csv",
        ]
        SWAP = next((p for p in swap_candidates if p.exists()), None)
        if SWAP is None:
            raise FileNotFoundError("No accepted fixed-render standard-CBM swap CSV")
        S = pd.read_csv(SWAP)
        # Legacy column names start with z_, but the renderer driver stored
        # model output["c_logits"] after the learned concept head.  In report
        # notation these are z_source and z_donor, not latent h values.
        if "response_delta" not in S:
            S["response_delta"] = S.margin - (S.z_new_orig - S.z_old_orig)
        S["responded_but_source_wins"] = (S.response_delta > 0) & (S.margin < 0)
        print("fixed-render input:", SWAP)
        print("rows:", len(S), "parts:", sorted(S.part.unique()))

        PRED_DIR = REPO/"external"/"minimal_cbm"/"results"/"funnybirds-cbm"/"1"/"predictions"
        PRED = require(PRED_DIR/"epoch_100.pth", "train or restore funnybirds-cbm seed 1 epoch 100")
        import torch
        saved = torch.load(PRED, map_location="cpu", weights_only=False)
        latent_h_saved = saved["z"].detach().cpu().numpy().reshape(len(saved["z"]), -1)
        p_saved = saved["c_preds"].detach().cpu().numpy().reshape(len(saved["c_preds"]), -1)
        c_saved = saved["c"].detach().cpu().numpy().reshape(len(saved["c"]), -1)
        MODEL = require(REPO/"external"/"minimal_cbm"/"results"/"funnybirds-cbm"/"1"/"models"/"epoch_100.pt",
                        "train or restore funnybirds-cbm seed 1 epoch 100")
        sys.path.insert(0, str(REPO/"analysis"))
        from minimal_cbm_scores import concept_logits_from_saved_latent, validate_saved_probabilities
        logit_tensor = concept_logits_from_saved_latent(saved["z"], MODEL, c_saved.shape[1])
        head_error = validate_saved_probabilities(logit_tensor, saved["c_preds"])
        z_saved = logit_tensor.numpy()
        print(f"[CONCEPT-HEAD REPLAY PASS] max |sigmoid(raw_logit)-saved_prob|={head_error:.3g}")

        FB_ROOT = Path(os.environ.get("FUNNYBIRDS_ROOT", CURATED/"FunnyBirds"))
        import funnybirds_concepts as fbc
        parts = fbc.load_parts(FB_ROOT)
        CONCEPT_NAMES = fbc.concept_names(parts)
        SPANS = fbc.group_slices(parts)
        if len(CONCEPT_NAMES) != z_saved.shape[1]:
            raise RuntimeError("parts.json concept width does not match saved predictions")
        CONCEPT_PART = {name: part for part,(a,b) in SPANS.items() for name in CONCEPT_NAMES[a:b]}
        print("checkpoint:", PRED, "concepts:", len(CONCEPT_NAMES), "species:", len(np.unique(saved["y"])))
        """),
    ]

    cells += [
        question("fb-q1", "1", "Did training produce a usable, non-collapsed CBM?",
                 "For every exact concept `j`, measure raw-score spread, positive-versus-negative label separation, balanced accuracy, and positive recall.",
                 "A usable slot has nonzero spread, positive label separation, and above-chance thresholded performance.",
                 "Compute all quantities from the epoch-100 held-out predictions. Recall is a health statistic, not grounding evidence."),
        code("fb-f1", r"""
        def balanced_accuracy(y, pred):
            y=np.asarray(y).astype(int); pred=np.asarray(pred).astype(int)
            tpr=(pred[y==1]==1).mean() if (y==1).any() else np.nan
            tnr=(pred[y==0]==0).mean() if (y==0).any() else np.nan
            return np.nanmean([tpr,tnr])

        rows=[]
        for j,name in enumerate(CONCEPT_NAMES):
            z=z_saved[:,j]; c=c_saved[:,j].astype(int); pred=(z>0).astype(int)
            rows.append({"concept":name,"part":CONCEPT_PART[name],
                         "spread":np.quantile(z,.95)-np.quantile(z,.05),
                         "label_separation":np.median(z[c==1])-np.median(z[c==0]),
                         "balanced_accuracy":balanced_accuracy(c,pred),
                         "positive_recall":pred[c==1].mean(),
                         "n_positive":int(c.sum()),"n_negative":int((c==0).sum())})
        HEALTH=pd.DataFrame(rows).sort_values(["part","concept"])
        y_true=np.asarray(saved["y"]).reshape(-1).astype(int)
        y_scores=np.asarray(saved["y_preds"])
        if y_scores.ndim>2: y_scores=y_scores.reshape(len(y_scores),-1)
        task_accuracy=float((y_scores.argmax(1)==y_true).mean())
        concept_accuracy=float(((z_saved>0)==c_saved).mean())
        display(pd.DataFrame([{"images":len(y_true),"species":len(np.unique(y_true)),
                              "task_accuracy":task_accuracy,"concept_accuracy":concept_accuracy}]).round(4))
        display(HEALTH.round(3))
        metrics=["spread","label_separation","balanced_accuracy","positive_recall"]
        fig,axes=plt.subplots(1,4,figsize=(15,max(5,.24*len(HEALTH))),sharey=True)
        y=np.arange(len(HEALTH))
        for ax,m in zip(axes,metrics):
            ax.scatter(HEALTH[m],y,c=HEALTH.part.map(COLORS).fillna("#BBBBBB"),s=24)
            ax.set_xlabel(m.replace("_"," "))
            if m in ["label_separation"]: ax.axvline(0,color="black",lw=.8)
            if m in ["balanced_accuracy","positive_recall"]: ax.axvline(.5,color="gray",ls="--",lw=.8)
        axes[0].set_yticks(y); axes[0].set_yticklabels(HEALTH.concept,fontsize=7)
        axes[0].invert_yaxis(); fig.suptitle("Figure 1 · Exact-concept model-health guard")
        plt.tight_layout(); plt.show()
        """, "Four aligned dot plots showing raw-score spread, label separation, balanced accuracy, and positive recall for every FunnyBird concept."),
        review("fb-r1", "Figure 1"),

        question("fb-q2", "2", "Did the renderer change only the intended part?",
                 "Inspect the semantic preflight and original/swap/delete/part-map examples for all five parts.",
                 "A valid intervention visibly changes the target part, preserves the rest of the scene, and has nonzero target-mask pixels.",
                 "Use artifacts from the accepted fixed-render root before reading any model response."),
        code("fb-f2", r"""
        ROOT = SWAP.parent
        preflight_candidates=[ROOT/"renderer_preflight"/"renderer_semantic_preflight.png",
                              CURATED/"swap_fixed_v2_attempt2"/"renderer_preflight"/"renderer_semantic_preflight.png"]
        preflight=next((p for p in preflight_candidates if p.exists()),preflight_candidates[0])
        example_candidates=[ROOT/"examples",CURATED/"swap_fixed_v2_attempt2"/"examples"]
        examples=next((p for p in example_candidates if p.is_dir()),example_candidates[0])
        from PIL import Image
        if preflight.exists():
            im0=Image.open(preflight).convert("RGB")
            fig0,ax0=plt.subplots(figsize=(14,3.2))
            ax0.imshow(im0); ax0.axis("off")
            ax0.set_title("Figure 2a · Semantic preflight: original, swap, delete, original map, swap map")
            plt.tight_layout(); plt.show()
        else:
            print("preflight sheet not stored beside CSV; use accepted job-3330289 audit")
        tags=["orig","swap","delete","swap_partmap"]
        fig,axes=plt.subplots(len(ORDER),len(tags),figsize=(12,13))
        for r,part in enumerate(ORDER):
            for c,tag in enumerate(tags):
                ax=axes[r,c]; files=sorted(examples.glob(f"{part}_*_{tag}.png"))
                if files: ax.imshow(Image.open(files[0]).convert("RGB"))
                else: ax.text(.5,.5,"missing",ha="center",va="center")
                ax.set_title(f"{part} · {tag}"); ax.axis("off")
        fig.suptitle("Figure 2b · Complete intervention audit: original, replacement, deletion, and target mask")
        plt.tight_layout(); plt.show()
        """, "Complete FunnyBird intervention audit showing original, swapped, deleted, and part-map images for tail, wing, beak, foot, and eye."),
        review("fb-r2", "Figure 2"),

        question("fb-q3", "3", "Did the inserted pixels move the comparison toward the donor?",
                 "`response_delta = (z_donor-z_source)_cf - (z_donor-z_source)_orig`. Legacy CSV columns named `z_*` contain these post-head raw logits.",
                 "Values above zero mean that replacement pixels moved the model toward the donor concept.",
                 "Plot the complete distribution for every part and report the positive-response rate."),
        code("fb-f3", r"""
        fig,axes=plt.subplots(1,2,figsize=(12,4.2))
        vals=[S.loc[S.part==p,"response_delta"].dropna() for p in ORDER]
        bp=axes[0].boxplot(vals,tick_labels=ORDER,showfliers=False,whis=(5,95),patch_artist=True)
        for box,p in zip(bp["boxes"],ORDER): box.set_facecolor(COLORS[p]); box.set_alpha(.55)
        axes[0].axhline(0,color="black",lw=1); axes[0].set_ylabel("response_delta (raw logit units)")
        axes[0].set_title("A · Distribution of donorward movement")
        rate=S.groupby("part").response_delta.apply(lambda x:(x>0).mean()).reindex(ORDER)
        axes[1].bar(rate.index,rate.values,color=[COLORS[p] for p in rate.index])
        axes[1].set_ylim(0,1)
        axes[1].set_ylabel("fraction with response_delta > 0"); axes[1].set_title("B · Positive donor-response rate")
        counts=S.groupby("part").size().reindex(ORDER)
        for x,(p,v) in enumerate(rate.items()): axes[1].text(x,v+.025,f"n={counts[p]:,}",ha="center",fontsize=8)
        fig.suptitle("Figure 3 · Does the replacement produce the predicted within-image response?")
        plt.tight_layout(); plt.show(); display(rate.rename("positive_response_rate").to_frame().round(3))
        """, "FunnyBird response-delta distributions and positive donor-response rates for all five parts."),
        review("fb-r3", "Figure 3"),

        question("fb-q4", "4", "After responding, does the donor finish above the old source?",
                 "The final margin is `m_cf=z_donor,cf-z_source,cf`. The primary event is `response_delta>0` with `m_cf<0`.",
                 "A lower-right quadrant point means the inserted pixels had an effect but the old source still wins.",
                 "Show final-margin distributions and the joint response/margin plane for every part."),
        code("fb-f4", r"""
        fig,axes=plt.subplots(1,2,figsize=(14,4.8))
        vals=[S.loc[S.part==p,"margin"].dropna() for p in ORDER]
        bp=axes[0].boxplot(vals,tick_labels=ORDER,showfliers=False,whis=(5,95),patch_artist=True)
        for box,p in zip(bp["boxes"],ORDER): box.set_facecolor(COLORS[p]); box.set_alpha(.55)
        axes[0].axhline(0,color="black",lw=1); axes[0].set_ylabel("final margin m_cf (donor − source)")
        axes[0].set_title("A · Final donor-minus-source margin")
        for p in ORDER:
            d=S[S.part==p]
            axes[1].scatter(d.response_delta,d.margin,s=10,alpha=.22,color=COLORS[p],label=p)
        axes[1].axvline(0,color="black",lw=1); axes[1].axhline(0,color="black",lw=1)
        axes[1].set_xlabel("response_delta"); axes[1].set_ylabel("final margin m_cf")
        axes[1].set_title("B · Lower-right = responds, but old source still wins")
        axes[1].legend(ncol=5,fontsize=8)
        fig.suptitle("Figure 4 · Controlled FunnyBird backwash predicate")
        plt.tight_layout(); plt.show()
        summary=S.groupby("part").agg(n=("margin","size"),median_response=("response_delta","median"),
            median_final_margin=("margin","median"),positive_response_rate=("response_delta",lambda x:(x>0).mean()),
            responded_but_source_wins_rate=("responded_but_source_wins","mean")).reindex(ORDER)
        display(summary.round(3))
        """, "Final donor-minus-source margin distributions and joint response-delta versus final-margin plot for all FunnyBird parts."),
        review("fb-r4", "Figure 4"),

        question("fb-q5", "5", "Could opposite swap directions create the result?",
                 "Compare forward and backward rates of `response_delta>0 and final margin<0`, together with median margins.",
                 "A genuine part pattern should appear in both directions rather than cancel when pooled.",
                 "Keep directions separate and show their denominators."),
        code("fb-f5", r"""
        D=(S.groupby(["part","direction"]).agg(n=("margin","size"),median_margin=("margin","median"),
             responded_but_source_wins_rate=("responded_but_source_wins","mean")).reset_index())
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        x=np.arange(len(ORDER))
        for direction,marker,offset in [("fwd","o",-.10),("bwd","s",.10)]:
            d=D[D.direction==direction].set_index("part").reindex(ORDER)
            axes[0].scatter(x+offset,d.responded_but_source_wins_rate,marker=marker,label=direction,s=45)
            axes[1].scatter(x+offset,d.median_margin,marker=marker,label=direction,s=45)
        for ax in axes: ax.set_xticks(x,ORDER)
        axes[0].set_ylim(0,1); axes[0].set_ylabel("fraction: donorward response, but source still wins")
        axes[1].axhline(0,color="black",lw=.8); axes[1].set_ylabel("median final margin")
        axes[0].legend(); axes[1].legend(); fig.suptitle("Figure 5 · Forward and backward directions")
        plt.tight_layout(); plt.show(); display(D.round(3))
        """, "Forward and backward FunnyBird rates where the donor changes the margin but the old source remains larger, alongside final margins for every part."),
        review("fb-r5", "Figure 5"),

        question("fb-q6", "6", "How much of the result is associated with target visibility?",
                 "Use `pixel_count_cf` from the exact swapped-part map and the same final-margin and `response_delta>0, margin<0` definition.",
                 "If visibility is sufficient, highly visible replacements should remove the part gap; a remaining gap requires another explanation.",
                 "Use declared bins and print the number of swap rows in every bin."),
        code("fb-f6", r"""
        if "pixel_count_cf" not in S: raise RuntimeError("fixed swap CSV lacks pixel_count_cf")
        bins=[0,20,50,100,200,500,np.inf]; labels=["0–19","20–49","50–99","100–199","200–499","500+"]
        V=S.copy(); V["visibility_bin"]=pd.cut(V.pixel_count_cf,bins=bins,labels=labels,right=False)
        T=V.groupby(["part","visibility_bin"],observed=True).agg(
            n=("margin","size"),median_margin=("margin","median"),responded_but_source_wins_rate=("responded_but_source_wins","mean")).reset_index()
        fig,axes=plt.subplots(1,2,figsize=(14,4.5))
        for p in ORDER:
            d=T[T.part==p].set_index("visibility_bin").reindex(labels)
            axes[0].plot(labels,d.median_margin,"o-",label=p,color=COLORS[p])
            axes[1].plot(labels,d.responded_but_source_wins_rate,"o-",label=p,color=COLORS[p])
        axes[0].axhline(0,color="black",lw=.8); axes[0].set_ylabel("median final margin")
        axes[1].set_ylim(0,1); axes[1].set_ylabel("fraction: donorward response, but source still wins")
        for ax in axes: ax.tick_params(axis="x",rotation=45); ax.legend(fontsize=8,ncol=2)
        fig.suptitle("Figure 6 · Same-render visibility analysis")
        VISIBLE_ONLY=(V[V.pixel_count_cf>0].groupby("part").agg(
            n_visible_rows=("margin","size"),median_margin=("margin","median"),
            responded_but_source_wins_rate=("responded_but_source_wins","mean")).reindex(ORDER))
        plt.tight_layout(); plt.show(); display(T.round(3)); display(VISIBLE_ONLY.round(3))
        """, "FunnyBird final margin and responded-but-source-still-wins rate across exact swapped-part visibility bins for all parts."),
        review("fb-r6", "Figure 6"),

        question("fb-q6b", "6b", "How often did the original training label conflict with visible part evidence?",
                 "Compare the standard and visibility-aware training records for the same images; count positive concept labels changed to zero within each part group.",
                 "A large conflict count identifies a plausible training signal that can reward contextual prediction, but its causal effect belongs to notebook 03rl.",
                 "Require identical ordered image/class records and allow only `attribute_label` to differ."),
        code("fb-f6b", r"""
        import pickle
        std_path=CURATED/"funnybirds_processed_trainval"/"train.pkl"
        rl_path=CURATED/"funnybirds_processed_rl_trainval"/"train.pkl"
        if not (std_path.exists() and rl_path.exists()):
            print("INCOMPLETE: matched standard/RLv2 training records are not both present")
        else:
            std=pickle.loads(std_path.read_bytes()); rl=pickle.loads(rl_path.read_bytes())
            if len(std)!=len(rl): raise RuntimeError("standard/RLv2 train lengths differ")
            positive=np.zeros(len(CONCEPT_NAMES),dtype=int); changed=np.zeros(len(CONCEPT_NAMES),dtype=int)
            for a,b in zip(std,rl):
                for key in a:
                    if key=="attribute_label": continue
                    av,bv=a[key],b[key]
                    equal=np.array_equal(np.asarray(av),np.asarray(bv)) if isinstance(av,(list,tuple,np.ndarray)) else av==bv
                    if not bool(equal): raise RuntimeError(f"non-label record field differs: {key}")
                ca=np.asarray(a["attribute_label"]); cb=np.asarray(b["attribute_label"])
                positive += (ca==1); changed += ((ca==1)&(cb==0))
            CONFLICT_EXACT=pd.DataFrame({"concept":CONCEPT_NAMES,"part":[CONCEPT_PART[n] for n in CONCEPT_NAMES],
                "n_positive":positive,"n_changed":changed})
            CONFLICT_EXACT["conflict_rate"]=CONFLICT_EXACT.n_changed/CONFLICT_EXACT.n_positive.replace(0,np.nan)
            PART_CONFLICT=(CONFLICT_EXACT.groupby("part").agg(n_positive=("n_positive","sum"),
                n_changed=("n_changed","sum")).reindex(ORDER))
            PART_CONFLICT["conflict_rate"]=PART_CONFLICT.n_changed/PART_CONFLICT.n_positive
            q=CONFLICT_EXACT.sort_values(["part","concept"]); y=np.arange(len(q))
            fig,ax=plt.subplots(figsize=(10,max(6,.24*len(q))))
            ax.barh(y,q.conflict_rate,color=q.part.map(COLORS)); ax.set_yticks(y,q.concept,fontsize=7)
            ax.invert_yaxis(); ax.set_xlim(0,1); ax.set_xlabel("positive-label / invisible-mask conflict rate")
            ax.set_title("Figure 6b · Exact-concept training label–visibility conflict")
            plt.tight_layout(); plt.show(); display(q.round(3)); display(PART_CONFLICT.round(3))
        """, "FunnyBird training-image counts whose positive part-concept labels change under the matched visibility-aware relabeling rule."),
        review("fb-r6b", "Figure 6b"),

        question("fb-q7", "7", "Do exact source and donor values explain the failures?",
                 "For every part, compare the inserted donor value with the concept value that has the largest post-swap raw score.",
                 "A clean diagonal means exact visual values are distinguished; recurring bright columns indicate default answers.",
                 "Display all parts and all values with row-normalized counts."),
        code("fb-f7", r"""
        available=[p for p in ORDER if any(c.startswith(f"z_cf_{p}_") for c in S.columns)]
        if set(available)!=set(ORDER): raise RuntimeError(f"missing all-part post-swap concept logits: have {available}")
        fig,axes=plt.subplots(2,5,figsize=(18,7),constrained_layout=True)
        diag={}
        value_rows=[]
        for col,p in enumerate(ORDER):
            ax=axes[0,col]; bax=axes[1,col]
            cols=sorted([c for c in S if c.startswith(f"z_cf_{p}_")],key=lambda x:int(x.rsplit("_",1)[1]))
            d=S[S.part==p].dropna(subset=cols); donor=d.var_donor.astype(int).to_numpy(); pred=d[cols].to_numpy().argmax(1)
            M=np.zeros((len(cols),len(cols)))
            for a,b in zip(donor,pred):
                if 0<=a<len(cols): M[a,b]+=1
            M=M/np.maximum(M.sum(1,keepdims=True),1); diag[p]=(donor==pred).mean()
            im=ax.imshow(M,vmin=0,vmax=1,cmap="magma"); ax.set_title(f"{p}\ndiagonal={diag[p]:.2f}")
            ax.set_xlabel("highest-scoring value"); ax.set_ylabel("inserted value")
            groups=[]; labels=[]
            for v,g in d.groupby("var_donor"):
                groups.append(g.margin.to_numpy()); labels.append(str(int(v)))
                value_rows.append({"part":p,"donor_value":int(v),"n":len(g),"median_margin":g.margin.median(),
                    "q25_margin":g.margin.quantile(.25),"q75_margin":g.margin.quantile(.75),
                    "event_rate":g.responded_but_source_wins.mean()})
            bax.boxplot(groups,tick_labels=labels,showfliers=False,whis=(5,95)); bax.axhline(0,color="black",lw=.8)
            bax.set_xlabel("inserted value"); bax.set_ylabel("final margin"); bax.set_title(f"{p}: value-wise margins")
            for x0,(lab,g) in enumerate(zip(labels,groups),1): bax.text(x0,bax.get_ylim()[1],f"n={len(g)}",ha="center",va="bottom",fontsize=6,rotation=90)
        fig.colorbar(im,ax=list(axes[0]),fraction=.015); fig.suptitle("Figure 7 · Exact-value attribution and final-margin distributions")
        plt.show(); display(pd.Series(diag,name="diagonal_rate").to_frame().round(3)); display(pd.DataFrame(value_rows).round(3))
        """, "Five row-normalized confusion matrices comparing inserted and highest-scoring FunnyBird part values."),
        review("fb-r7", "Figure 7"),

        question("fb-q7b", "7b", "Are difficult values simply rare or drawn from a larger alternative set?",
                 "For every donor value, compare its source-species support with the rate where `response_delta>0` but the final margin remains negative; also report the total number of alternatives for its part.",
                 "An association supports frequency or choice-set difficulty, but five part-level counts cannot establish a stable correlation.",
                 "Label every exact value and show its number of swap rows."),
        code("fb-f7b", r"""
        VS=(S.groupby(["part","var_donor"]).agg(n_rows=("margin","size"),species_support=("sid_donor","nunique"),
             responded_but_source_wins_rate=("responded_but_source_wins","mean"),median_margin=("margin","median")).reset_index())
        VS["alternatives_in_part"]=VS.part.map({p:hi-lo for p,(lo,hi) in SPANS.items()})
        fig,ax=plt.subplots(figsize=(9,6))
        for p,d in VS.groupby("part"):
            ax.scatter(d.species_support,d.responded_but_source_wins_rate,s=35,color=COLORS[p],label=p)
            for r in d.itertuples(): ax.annotate(f"{p}_{int(r.var_donor)}",(r.species_support,r.responded_but_source_wins_rate),fontsize=6,xytext=(3,3),textcoords="offset points")
        ax.set_xlabel("source species carrying donor value"); ax.set_ylabel("fraction: donorward response, but source still wins")
        ax.set_ylim(-.02,1.02); ax.legend(); ax.set_title("Figure 7b · Exact-value support versus controlled backwash events")
        plt.tight_layout(); plt.show(); display(VS.round(3))
        """, "Labelled FunnyBird donor-value plot of species support versus the rate where donor pixels move the margin but the old source remains larger."),
        review("fb-r7b", "Figure 7b"),

        question("fb-q8", "8", "Does source species organize the remaining error after exact values?",
                 "Subtract the mean margin for each `(part, source value, donor value)` combination, then summarize the residual by source species.",
                 "Persistent species differences support an additional unchanged-body/species association, but remain observational.",
                 "Show every part and require at least five rows per displayed species estimate."),
        code("fb-f8", r"""
        R=S.copy(); R["value_pair_mean"]=R.groupby(["part","var_src","var_donor"]).margin.transform("mean")
        R["margin_after_value_pair"]=R.margin-R.value_pair_mean
        SP=(R.groupby(["part","sid_src"]).agg(n=("margin","size"),residual=("margin_after_value_pair","mean"))
              .reset_index().query("n>=5"))
        fig,axes=plt.subplots(1,5,figsize=(18,4),sharey=True)
        for ax,p in zip(axes,ORDER):
            d=SP[SP.part==p].sort_values("residual")
            ax.scatter(np.arange(len(d)),d.residual,color=COLORS[p],s=18)
            ax.axhline(0,color="black",lw=.8); ax.set_title(f"{p} (n species={len(d)})")
            ax.set_xlabel("source species, sorted")
        axes[0].set_ylabel("mean margin residual after exact value pair")
        fig.suptitle("Figure 8 · Source-species residual after exact source/donor values")
        plt.tight_layout(); plt.show(); display(SP.groupby("part").residual.agg(["min","median","max","std","count"]).round(3))
        """, "Per-source-species FunnyBird margin residuals after controlling exact source and donor values, shown for all parts."),
        review("fb-r8", "Figure 8"),

        question("fb-q8b", "8b", "How much species identity is recoverable from the learned concept vector?",
                 "Train a held-out linear species probe on the full raw-logit vector and on each part block separately.",
                 "Accuracy above the 1/50 chance level shows stored species information; it does not identify the pixels responsible or prove backward causal flow.",
                 "Use one fixed stratified 70/30 split of the held-out prediction population."),
        code("fb-f8b", r"""
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        y_saved=np.asarray(saved["y"]).reshape(-1).astype(int)
        idx=np.arange(len(y_saved)); tr,te=train_test_split(idx,test_size=.30,random_state=20260803,stratify=y_saved)
        blocks={"complete raw logits":np.arange(z_saved.shape[1])}
        blocks.update({p:np.arange(lo,hi) for p,(lo,hi) in SPANS.items()})
        probe=[]
        for name,cols in blocks.items():
            model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,C=1.0,random_state=20260803))
            model.fit(z_saved[tr][:,cols],y_saved[tr]); probe.append({"block":name,"species_accuracy":accuracy_score(y_saved[te],model.predict(z_saved[te][:,cols])),"dimensions":len(cols)})
        PROBE=pd.DataFrame(probe)
        fig,ax=plt.subplots(figsize=(8,4)); ax.bar(PROBE.block,PROBE.species_accuracy,color=["#333333"]+[COLORS.get(x,"#999999") for x in PROBE.block.iloc[1:]])
        ax.axhline(1/len(np.unique(y_saved)),color="black",ls="--",label="chance = 1/50")
        ax.set_ylim(0,1); ax.set_ylabel("held-out species accuracy"); ax.set_title("Figure 8b · Species decoded from learned concept representations")
        ax.legend(); plt.tight_layout(); plt.show(); display(PROBE.round(3))
        """, "Held-out FunnyBird species-decoding accuracy from the complete raw concept vector and each individual part block."),
        review("fb-r8b", "Figure 8b"),

        question("fb-q9", "9", "How much does each observed block account for?",
                 "Predict the raw final margin on held-out render IDs using progressively richer categorical blocks.",
                 "Lower held-out error means the added block organizes the outcome; remaining error is the measured residual.",
                 "Use stable five-fold image-level splits, training-fold group means with shrinkage, and no RLv2 variables."),
        code("fb-f9", r"""
        import hashlib
        A=S.copy(); A["vis_bin"]=pd.cut(A.pixel_count_cf,[-1,19,49,99,199,499,np.inf],labels=False)
        unit=(A["render_id"].astype(str) if "render_id" in A else
              A.get("li",pd.Series(np.arange(len(A)),index=A.index)).astype(str))
        A["fold"]=unit.map(lambda x:int(hashlib.sha1(x.encode()).hexdigest(),16)%5)
        stages=[("part only",["part"]),("+ visibility",["part","vis_bin"]),
                ("+ exact values",["part","vis_bin","var_src","var_donor"]),
                ("+ source species",["part","vis_bin","var_src","var_donor","sid_src"])]
        rows=[]
        for stage,cols in stages:
            pred=pd.Series(index=A.index,dtype=float)
            for fold in range(5):
                tr=A[A.fold!=fold]; te=A[A.fold==fold]
                prior=tr.margin.mean(); stats=tr.groupby(cols).margin.agg(["mean","count"]).reset_index()
                stats["estimate"]=(stats["mean"]*stats["count"]+prior*10)/(stats["count"]+10)
                joined=te[cols].merge(stats[cols+["estimate"]],on=cols,how="left")
                pred.loc[te.index]=joined.estimate.fillna(prior).to_numpy()
            rows.append({"stage":stage,"rmse":float(np.sqrt(np.mean((A.margin-pred)**2))),
                         "mae":float(np.mean(np.abs(A.margin-pred)))})
        ACCOUNT=pd.DataFrame(rows)
        desc=[]
        for p in ORDER:
            d=A[A.part==p]
            for label,mask in [("all rows",np.ones(len(d),dtype=bool)),("visible >0 px",d.pixel_count_cf>0),("clearly visible ≥100 px",d.pixel_count_cf>=100)]:
                g=d[mask]; desc.append({"part":p,"selection":label,"n":len(g),"median_margin":g.margin.median()})
        DESC=pd.DataFrame(desc)
        fig,axes=plt.subplots(1,2,figsize=(14,4.5))
        for label,marker,offset in [("all rows","o",-.15),("visible >0 px","s",0),("clearly visible ≥100 px","^",.15)]:
            d=DESC[DESC.selection==label].set_index("part").reindex(ORDER)
            axes[0].scatter(np.arange(len(ORDER))+offset,d.median_margin,label=label,marker=marker,s=45)
        axes[0].axhline(0,color="black",lw=.8); axes[0].set_xticks(np.arange(len(ORDER)),ORDER)
        axes[0].set_ylabel("median final margin"); axes[0].set_title("A · Descriptive visibility selections"); axes[0].legend(fontsize=8)
        axes[1].plot(ACCOUNT.stage,ACCOUNT.rmse,"o-",color="#0072B2")
        axes[1].set_ylabel("held-out RMSE of final margin"); axes[1].tick_params(axis="x",rotation=25)
        axes[1].set_title("B · Sequential held-out accounting")
        fig.suptitle("Figure 9 · What measured contributors organize, and what remains")
        plt.tight_layout(); plt.show(); display(DESC.round(3)); display(ACCOUNT.round(3))
        """, "Held-out final-margin prediction error after adding FunnyBird visibility, exact values, and source species sequentially."),
        review("fb-r9", "Figure 9"),

        question("fb-q10", "10", "Does the concept-layer error materially alter species prediction?",
                 "Relate final concept margin to the model's donor-species probability, which is a different downstream quantity.",
                 "A small downstream change would limit the harm to explanation reliability rather than widespread class failure.",
                 "Use independent final-margin bins and print bin counts."),
        code("fb-f10", r"""
        prob_col=next((c for c in ["p_cf_donor","p_donor_cf","donor_species_prob"] if c in S),None)
        if prob_col is None:
            print("INCOMPLETE: swap CSV has no donor-species probability column")
        else:
            D=S.copy(); D["margin_bin"]=pd.qcut(D.margin,10,duplicates="drop")
            Q=D.groupby("margin_bin",observed=True).agg(n=(prob_col,"size"),mean_margin=("margin","mean"),mean_donor_species_prob=(prob_col,"mean")).reset_index()
            fig,ax=plt.subplots(figsize=(7,4)); ax.plot(Q.mean_margin,Q.mean_donor_species_prob,"o-")
            for r in Q.itertuples(): ax.annotate(f"n={r.n}",(r.mean_margin,r.mean_donor_species_prob),fontsize=7)
            ax.axvline(0,color="black",lw=.8); ax.set_xlabel("mean final concept margin in bin")
            ax.set_ylabel("mean donor-species probability"); ax.set_title("Figure 10 · Downstream consequence of the concept margin")
            plt.tight_layout(); plt.show(); display(Q.round(3))
        """, "Binned relationship between FunnyBird final concept margin and downstream donor-species probability."),
        review("fb-r10", "Figure 10"),

        md("fb-conclusion", r"""
        ## 11 · Standard-CBM evidence ledger

        Figures 1–10 were displayed and reviewed together on 2026-08-04.

        | Predicate or explanation | Direct measurement | Status after review |
        |---|---|---|
        | model outputs are usable | Figure 1 | `ACCEPTED FOR seed-1 model health` |
        | interventions are valid | Figure 2 | `ACCEPTED FOR validated fixed renders` |
        | inserted pixels cause donorward movement | Figure 3 | `ACCEPTED FOR all five parts` |
        | old source can remain stronger after that movement | Figure 4 | `ACCEPTED; strongest for tail, beak, eye` |
        | direction artifact excluded | Figure 5 | `ACCEPTED` |
        | visibility contribution | Figure 6 | `ACCEPTED AS CONTRIBUTOR, NOT SUFFICIENT` |
        | training label/mask conflict measured | Figure 6b | `MEASURED; causal effect deferred to RLv2` |
        | exact-value contribution | Figure 7 | `ACCEPTED AS GRADED CONTRIBUTOR` |
        | frequency/alternative-count explanation | Figure 7b | `VALID TEST, NO CLEAR SUPPORT AS SOLE EXPLANATION` |
        | source-species residual | Figure 8 | `OBSERVATIONAL ASSOCIATION` |
        | species information in learned representation | Figure 8b | `ACCEPTED FOR AVAILABILITY` |
        | sequential descriptive accounting | Figure 9 | `VISIBILITY IMPROVES; EXACT VALUES/SPECIES DO NOT` |
        | downstream class consequence | Figure 10 | `MONOTONE BUT MODEST DONOR-PROBABILITY EFFECT` |

        **Next report question.** Notebook 03 asks whether the MCBM minimality
        penalty changes these same accepted quantities. It must not replace the
        standard-CBM result established here.
        """),
        md("fb-appendix", r"""
        # Methods appendix · measurements not used in the main claim

        The reciprocal mask-deletion and randomized-patch experiments are retained
        as method-development history. They did not reproduce the clean FunnyBird
        control sufficiently to transfer their causal interpretation to CUB.

        - reciprocal mask deletion: `METHOD NOT CALIBRATED FOR CROSS-DATASET CAUSAL COMPARISON`;
        - randomized patch V1/V2: local pixel response was measurable in selected
          examples, but the all-part control was not calibrated and wing coverage was
          inadequate;
        - none of these outcomes invalidates the validated renderer swap above.

        Full artifacts and scripts remain under `analysis/paired_mask_deletion.py`,
        `analysis/randomized_patch_masking.py`, and their output directories. They
        are not rerun by this notebook.
        """),
        md("fb-prov", r"""
        # Provenance appendix

        The table below records the live Git commit, input paths and SHA-256
        hashes, row counts, seed, and the accepted fixed-render root. It is part
        of the report: a stale HTML is not synchronized evidence.
        """),
        code("fb-prov-code", r"""
        def sha256_file(path):
            h=hashlib.sha256()
            with open(path,"rb") as f:
                for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
            return h.hexdigest()
        commit=subprocess.run(["git","rev-parse","HEAD"],cwd=REPO,capture_output=True,text=True,check=True).stdout.strip()
        prov=[]
        for role,path in [("fixed-render swap CSV",SWAP),("prediction export",PRED),("model checkpoint",MODEL)]:
            prov.append({"role":role,"path":str(path),"sha256":sha256_file(path)})
        display(pd.DataFrame(prov)); display(pd.DataFrame([{"git_commit":commit,"seed":1,
            "swap_rows":len(S),"prediction_images":len(c_saved),"exact_concepts":len(CONCEPT_NAMES),
            "excluded_swap_rows":0,"accepted_render_root":str(SWAP.parent)}]))
        """),
    ]
    return notebook(cells, NOTEBOOKS/"02_funnybirds_cbm.ipynb")


def build_cub() -> dict:
    cells: list[dict] = [
        md("cub-title", r"""
        # 05 · Standard CUB70 CBM: observational test of context-dependent concepts

        **Report question.** On real bird photographs, do raw concept scores depend
        on the visibility of the named region and on species context after exact
        concept identity is held fixed?

        **Causal boundary.** CUB has no accepted clean donor-part replacement.
        Therefore this notebook cannot reproduce the FunnyBird donor/source
        backwash predicate. It tests converging or contrary observational evidence:
        natural visibility, hidden-context scores, matched recall/raw-score gaps,
        and within-concept species effects.

        **Population.** Standard non-RL CUB70 CBM, seed 1, epoch 100. Full-CUB CBM
        is used only as a clearly labelled same-image robustness guard.
        """),
        md("cub-roadmap", CUB_PROOF_ROADMAP),
        md("cub-model", COMMON_MODEL),
        code("cub-setup", r"""
        import os, sys, hashlib, subprocess
        from pathlib import Path
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from IPython.display import display

        CURATED=Path(os.environ["CURATED_DATA"]); CWD=Path.cwd()
        REPO=CWD if (CWD/"analysis").is_dir() else CWD.parent
        sys.path.insert(0,str(REPO/"data"/"cub70"))
        from cub70_parts import CUB70_PARTS, ATTRIBUTE_TYPE_TO_MASK
        from relabel_cub_with_cub70 import coarse_visibility
        COLORS={"head":"#56B4E9","eye":"#CC79A7","beak":"#E69F00","neck":"#009E73",
                "body":"#0072B2","wing":"#D55E00","leg":"#777777","tail":"#F0E442"}
        COARSE_ORDER=["head","eye","beak","neck","body","wing","leg","tail"]
        COLLAPSE_TOL=1e-8
        pd.set_option("display.max_rows", 250)
        pd.set_option("display.max_columns", 40)

        def require(path,command):
            path=Path(path)
            if not path.exists(): raise FileNotFoundError(f"Missing {path}\nProduce it with: {command}")
            return path
        def family(name): return str(name).split("::",1)[0]
        def add_mapping(E):
            E=E.copy(); E["attribute_type"]=E.concept_name.map(family)
            E["mask_group"]=E.attribute_type.map(ATTRIBUTE_TYPE_TO_MASK); return E
        def attach(E,V):
            local=add_mapping(E); V=V.rename(columns={"image_name":"image","coarse":"mask_group"})
            return local[local.mask_group.notna()].merge(
                V[["image","mask_group","pixel_count","area_frac","visible"]],
                on=["image","mask_group"],how="inner",validate="many_to_one")
        def balanced_accuracy(y,pred):
            y=np.asarray(y).astype(int); pred=np.asarray(pred).astype(int)
            tpr=(pred[y==1]==1).mean() if (y==1).any() else np.nan
            tnr=(pred[y==0]==0).mean() if (y==0).any() else np.nan
            return np.nanmean([tpr,tnr])

        VIS=require(CURATED/"cub70_visibility.parquet","bash data/cub70/prepare_all.sh")
        E70P=require(CURATED/"cub70_eval"/"cub70-cbm-s1.parquet","CONFIGS='cub70-cbm' SEEDS='1' bash analysis/cub70_prepare_analysis.sh")
        EFULLP=require(CURATED/"cub70_eval"/"cub-cbm-s1.parquet","CONFIGS='cub-cbm' SEEDS='1' bash analysis/cub70_prepare_analysis.sh")
        RAWVIS=pd.read_parquet(VIS); V=coarse_visibility(RAWVIS,threshold=.001)
        E70=add_mapping(pd.read_parquet(E70P)); EFULL=add_mapping(pd.read_parquet(EFULLP))
        J70=attach(E70,V); JFULL=attach(EFULL,V)
        identity_error=float(np.nanmax(np.abs(E70.prob.to_numpy()-1/(1+np.exp(-E70.z.clip(-50,50).to_numpy())))))
        if identity_error>1e-5: raise RuntimeError(f"exported z is not the concept logit: max probability mismatch={identity_error}")
        print(f"[EXPORTED RAW-LOGIT PASS] max |prob-sigmoid(z)|={identity_error:.3g}")
        print("CUB70 rows:",len(E70),"images:",E70.image.nunique(),"species:",E70.y_true.nunique(),"concepts:",E70.concept_name.nunique())
        print("mask-matched images:",J70.image.nunique(),"fine masks:",sorted(RAWVIS.part.unique()))
        """),
    ]

    cells += [
        question("cub-q1", "1", "What population and mask evidence are available?",
                 "Count prediction images, mask-matched images, species, exact concepts, 11 released masks, and eight coarse groups.",
                 "Coverage losses must be explicit before any visible-versus-hidden comparison.",
                 "Report fine-mask visibility and bilateral left/right support without inventing left/right concepts. A mask is visible when area/image area is at least 0.001; the denominator is all 1,888 mask-matched photographs."),
        code("cub-f1", r"""
        inventory=pd.DataFrame([
            {"population":"CUB70 prediction export","images":E70.image.nunique(),"species":E70.y_true.nunique(),"concepts":E70.concept_name.nunique()},
            {"population":"mask-matched CUB70","images":J70.image.nunique(),"species":J70.y_true.nunique(),"concepts":J70.concept_name.nunique()},
        ])
        fine=RAWVIS.groupby("part").agg(images=("image_name","nunique"),visible_rate=("visible","mean"),median_area=("area_frac","median")).reindex(CUB70_PARTS)
        display(inventory); display(fine.round(4))
        fig,axes=plt.subplots(1,2,figsize=(13,4.5))
        axes[0].bar(fine.index,fine.visible_rate,color="#0072B2"); axes[0].tick_params(axis="x",rotation=55)
        axes[0].set_ylabel("fraction of images with visible mask"); axes[0].set_title("A · Visibility of all 11 released masks")
        axes[1].bar(fine.index,fine.median_area,color="#E69F00"); axes[1].tick_params(axis="x",rotation=55)
        axes[1].set_ylabel("median mask area / image area"); axes[1].set_title("B · Visible-region size")
        fig.suptitle("Figure 1 · CUB70 mask population and coverage")
        plt.tight_layout(); plt.show()
        """, "CUB70 inventory with visibility rates and median area for all 11 released part masks."),
        review("cub-r1", "Figure 1"),

        question("cub-q2", "2", "Is species–concept structure available before model behavior?",
                 "For each exact selected concept, count supporting species, positive images, and the number of alternatives in its attribute type.",
                 "Uneven support and species association make contextual prediction possible but do not prove model use.",
                 "Use labels only; no model score appears in this figure."),
        code("cub-f2", r"""
        LABEL=(E70.groupby(["attribute_type","concept_name","y_true"]).gt_label.mean().reset_index())
        support=(LABEL.assign(supports=lambda d:d.gt_label>=.5).groupby(["attribute_type","concept_name"])
                 .agg(species_support=("supports","sum"),species_total=("y_true","nunique")).reset_index())
        pos=E70.groupby(["attribute_type","concept_name"]).gt_label.agg(positive_images="sum",total_images="size").reset_index()
        support=support.merge(pos); support["alternatives_in_type"]=support.groupby("attribute_type").concept_name.transform("nunique")
        support["mask_group"]=support.attribute_type.map(ATTRIBUTE_TYPE_TO_MASK)
        support=support.sort_values(["attribute_type","concept_name"]).reset_index(drop=True)
        y=np.arange(len(support)); fig,ax=plt.subplots(figsize=(12,max(16,.24*len(support))))
        color_value=np.log1p(support.positive_images)
        sizes=28+18*support.alternatives_in_type
        edge=np.where(support.mask_group.isna(),"#555555","white")
        sc=ax.scatter(support.species_support,y,c=color_value,s=sizes,cmap="viridis",
                      edgecolors=edge,linewidths=.8)
        ax.set_yticks(y); ax.set_yticklabels(support.concept_name,fontsize=7); ax.invert_yaxis()
        ax.set_xlabel("number of CUB70 species carrying this exact concept value")
        cb=fig.colorbar(sc,ax=ax,pad=.01)
        cb.set_label("number of positive photographs (log color scale)")
        from matplotlib.lines import Line2D
        size_values=sorted(set([int(support.alternatives_in_type.min()),
                                int(support.alternatives_in_type.median()),
                                int(support.alternatives_in_type.max())]))
        handles=[Line2D([0],[0],marker="o",linestyle="",markerfacecolor="#888888",
                        markeredgecolor="white",markersize=np.sqrt(28+18*n)/1.5,
                        label=f"{n} alternatives") for n in size_values]
        handles.append(Line2D([0],[0],marker="o",linestyle="",markerfacecolor="#888888",
                              markeredgecolor="#555555",label="no released-mask mapping"))
        ax.legend(handles=handles,loc="lower right",fontsize=8,title="dot size / outline")
        fig.suptitle("Figure 2 · Exact-concept structure before model behavior")
        plt.tight_layout(); plt.show(); display(support.round(3))
        """, "Named CUB70 label-only plot showing species support, positive-image support, and number of alternatives for every exact concept; outlined dots mark concepts without a released-mask mapping."),
        review("cub-r2", "Figure 2"),

        question("cub-q2b", "4b", "How much species identity is recoverable from the learned CUB70 concept vector?",
                 "Decode species from all raw concept logits and from each coarse mask-linked block on a held-out split.",
                 "Accuracy above the 1/70 chance level shows that the learned representation stores species information; it does not prove that species caused a particular concept score.",
                 "Build one image-by-concept matrix and use a fixed stratified 70/30 split."),
        code("cub-f2b", r"""
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        X=E70.pivot_table(index="image",columns="concept_name",values="z",aggfunc="first")
        y=E70[["image","y_true"]].drop_duplicates().set_index("image").loc[X.index,"y_true"]
        tr,te=train_test_split(np.arange(len(X)),test_size=.30,random_state=20260803,stratify=y)
        cmap=E70[["concept_name","mask_group"]].drop_duplicates().set_index("concept_name").mask_group
        blocks={"complete z":list(X.columns)}
        blocks.update({g:[c for c in X.columns if cmap.get(c)==g] for g in COARSE_ORDER})
        rows=[]
        for name,cols in blocks.items():
            if not cols: continue
            model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=4000,C=1.0,random_state=20260803))
            model.fit(X.iloc[tr][cols],y.iloc[tr]); rows.append({"block":name,"species_accuracy":accuracy_score(y.iloc[te],model.predict(X.iloc[te][cols])),"dimensions":len(cols)})
        SPECIES_PROBE=pd.DataFrame(rows)
        fig,ax=plt.subplots(figsize=(9,4)); ax.bar(SPECIES_PROBE.block,SPECIES_PROBE.species_accuracy,color=["#333333"]+[COLORS.get(x,"#BBBBBB") for x in SPECIES_PROBE.block.iloc[1:]])
        ax.axhline(1/y.nunique(),color="black",ls="--",label="chance = 1/70"); ax.set_ylim(0,1)
        ax.set_ylabel("held-out species accuracy"); ax.set_title("Figure 4b · Species decoded from CUB70 raw concept logits")
        ax.legend(); plt.tight_layout(); plt.show(); display(SPECIES_PROBE.round(3))
        """, "Held-out CUB70 species-decoding accuracy from the complete raw concept vector and each coarse mask-linked concept block."),
        review("cub-r2b", "Figure 4b"),

        question("cub-q3", "4", "Did the standard CUB70 CBM produce usable exact-concept outputs?",
                 "For every concept, compute raw-score spread, label separation, balanced accuracy, and positive recall.",
                 "Exact collapse means `Q95(z)-Q05(z) <= 1e-8`; rounded probabilities are not used to diagnose collapse.",
                 "Evaluate all 112 outputs and mark mask-testable concepts separately."),
        code("cub-f3", r"""
        rows=[]
        for (t,c),d in E70.groupby(["attribute_type","concept_name"]):
            pos=d[d.gt_label==1].z; neg=d[d.gt_label==0].z
            spread=np.quantile(d.z,.95)-np.quantile(d.z,.05)
            rows.append({"attribute_type":t,"concept_name":c,"mask_group":d.mask_group.iloc[0],
                         "spread":spread,"collapsed":spread<=COLLAPSE_TOL,
                         "label_separation":pos.median()-neg.median() if len(pos) and len(neg) else np.nan,
                         "balanced_accuracy":balanced_accuracy(d.gt_label,d.z>0),
                         "positive_recall":((pos>0).mean() if len(pos) else np.nan),
                         "n_positive":len(pos),"n_negative":len(neg)})
        HEALTH=pd.DataFrame(rows).sort_values(["attribute_type","concept_name"]).reset_index(drop=True)
        images=E70[["image","y_true","y_pred"]].drop_duplicates("image")
        display(pd.DataFrame([{"images":len(images),"species":images.y_true.nunique(),
                              "task_accuracy":(images.y_true==images.y_pred).mean(),
                              "concept_accuracy":(E70.gt_label==E70.pred_label).mean()}]).round(4))
        y=np.arange(len(HEALTH)); metrics=["spread","label_separation","balanced_accuracy","positive_recall"]
        fig,axes=plt.subplots(1,4,figsize=(16,max(16,.24*len(HEALTH))),sharey=True)
        colors=HEALTH.mask_group.map(COLORS).fillna("#BBBBBB")
        for ax,m in zip(axes,metrics):
            ax.scatter(HEALTH[m],y,c=colors,s=17); ax.set_xlabel(m.replace("_"," "))
            if m=="label_separation": ax.axvline(0,color="black",lw=.8)
            if m in ["balanced_accuracy","positive_recall"]: ax.axvline(.5,color="gray",ls="--",lw=.8)
        axes[0].set_yticks(y); axes[0].set_yticklabels(HEALTH.concept_name,fontsize=7); axes[0].invert_yaxis()
        fig.suptitle("Figure 4 · Raw-score health guard for every exact CUB70 concept")
        plt.tight_layout(); plt.show(); display(HEALTH[HEALTH.collapsed])
        print("exact collapsed slots:",int(HEALTH.collapsed.sum()),"tolerance:",COLLAPSE_TOL)
        """, "Four aligned raw-score and thresholded-health plots for every CUB70 exact concept, with exact collapsed slots reported."),
        review("cub-r3", "Figure 4"),

        question("cub-q4", "3", "How often is a positive label paired with no visible mapped region?",
                 "For concept `j`, conflict is `P(v_ig=0 | c_ij=1)`.",
                 "High conflict means training/evaluation labels can be predicted without visible named-region evidence; it does not prove model use.",
                 "Plot every exact mask-testable concept at a named y-position with its denominator."),
        code("cub-f4", r"""
        exact=[]
        for (t,c),d in J70.groupby(["attribute_type","concept_name"]):
            pos=d[d.gt_label==1]; vis=pos[pos.visible]; hid=pos[~pos.visible]
            neg_hid=d[(d.gt_label==0)&(~d.visible)]
            exact.append({"attribute_type":t,"concept_name":c,"mask_group":d.mask_group.iloc[0],
                          "n_positive":len(pos),"n_visible":len(vis),"n_hidden":len(hid),
                          "label_mask_conflict":len(hid)/len(pos) if len(pos) else np.nan,
                          "z_visible":vis.z.mean() if len(vis) else np.nan,
                          "z_hidden":hid.z.mean() if len(hid) else np.nan,
                          "visibility_effect":vis.z.mean()-hid.z.mean() if len(vis) and len(hid) else np.nan,
                          "context_gap":hid.z.mean()-neg_hid.z.mean() if len(hid) and len(neg_hid) else np.nan,
                          "n_hidden_negative":len(neg_hid)})
        # `support` carries a plotting-only mask_group column. Keep the
        # row-level mask_group above instead of creating mask_group_x/y.
        EXACT=pd.DataFrame(exact).merge(
            support.drop(columns=["mask_group"],errors="ignore"),
            on=["attribute_type","concept_name"],how="left"
        )
        EXACT=EXACT.sort_values(["attribute_type","concept_name"]).reset_index(drop=True)
        y=np.arange(len(EXACT)); fig,ax=plt.subplots(figsize=(12,max(16,.24*len(EXACT))))
        ax.scatter(EXACT.label_mask_conflict,y,c=EXACT.mask_group.map(COLORS).fillna("#BBBBBB"),s=24)
        tick=[f"{r.concept_name}  (hidden/positive={int(r.n_hidden)}/{int(r.n_positive)})" for r in EXACT.itertuples()]
        ax.set_yticks(y); ax.set_yticklabels(tick,fontsize=7); ax.invert_yaxis()
        ax.set_xlim(-.02,1.02); ax.set_xlabel("fraction of positive labels with mapped mask absent")
        ax.set_title("Figure 3 · Label/mask conflict for every exact testable concept")
        plt.tight_layout(); plt.show(); display(EXACT[["concept_name","mask_group","n_positive","n_hidden","label_mask_conflict"]].round(3))
        """, "Aligned named dot plot of positive-label/mask conflict rates and denominators for every testable CUB70 exact concept."),
        review("cub-r4", "Figure 3"),

        question("cub-q5", "5", "Does natural visibility change the raw score of a positive-labelled concept?",
                 "`visibility_effect_j = mean(z|c=1,v=1)-mean(z|c=1,v=0)`.",
                 "Positive values mean visible examples score higher; negative values require investigation rather than automatic backwash language.",
                 "Require at least ten visible and ten hidden positive examples and show every eligible exact concept."),
        code("cub-f5", r"""
        VE=EXACT[(EXACT.n_visible>=10)&(EXACT.n_hidden>=10)&EXACT.visibility_effect.notna()].copy()
        VE=VE.sort_values(["mask_group","attribute_type","concept_name"]).reset_index(drop=True)
        y=np.arange(len(VE)); fig,ax=plt.subplots(figsize=(11,max(9,.23*len(VE))))
        ax.scatter(VE.visibility_effect,y,c=VE.mask_group.map(COLORS).fillna("#BBBBBB"),s=30)
        ax.axvline(0,color="black",lw=1); ax.set_yticks(y); ax.set_yticklabels(VE.concept_name,fontsize=6); ax.invert_yaxis()
        ax.set_xlabel("visibility_effect in raw z units (visible − hidden)")
        ax.set_title("Figure 5 · Natural-visibility effect for every eligible exact concept")
        plt.tight_layout(); plt.show(); display(VE[["concept_name","mask_group","n_visible","n_hidden","z_hidden","z_visible","visibility_effect"]].round(3))
        """, "Zero-centered raw-logit visibility effects for every eligible CUB70 exact concept with visible and hidden counts."),
        review("cub-r5", "Figure 5"),

        question("cub-q6", "6", "Does contextual concept information remain when the named region is hidden?",
                 "`context_gap_j = mean(z|c=1,v=0)-mean(z|c=0,v=0)`.",
                 "A positive gap means outside-region information distinguishes the label while the mapped region is hidden; it is not a donor/source margin.",
                 "Require at least ten hidden positives and ten hidden negatives."),
        code("cub-f6", r"""
        CG=EXACT[(EXACT.n_hidden>=10)&(EXACT.n_hidden_negative>=10)&EXACT.context_gap.notna()].copy()
        CG=CG.sort_values(["mask_group","attribute_type","concept_name"]).reset_index(drop=True)
        y=np.arange(len(CG)); fig,ax=plt.subplots(figsize=(11,max(9,.23*len(CG))))
        ax.scatter(CG.context_gap,y,c=CG.mask_group.map(COLORS).fillna("#BBBBBB"),s=30)
        ax.axvline(0,color="black",lw=1); ax.set_yticks(y); ax.set_yticklabels(CG.concept_name,fontsize=6); ax.invert_yaxis()
        ax.set_xlabel("context_gap in raw z units (hidden positive − hidden negative)")
        ax.set_title("Figure 6 · Hidden-region contextual separation")
        plt.tight_layout(); plt.show(); display(CG[["concept_name","mask_group","n_hidden","n_hidden_negative","context_gap"]].round(3))
        """, "Zero-centered raw-logit hidden-context gaps for every eligible CUB70 exact concept."),
        review("cub-r6", "Figure 6"),

        question("cub-q7", "7", "Do bilateral visibility and visible area offer simpler explanations?",
                 "For eye, wing, and leg, retain left/right masks and compare zero, one, or two visible sides. Separately estimate within-concept area dose response.",
                 "A monotone increase supports local visual evidence; non-monotone patterns motivate pose or species controls.",
                 "Use only positive-labelled rows and raw `z`."),
        code("cub-f7", r"""
        pairmap={"eye":["left_eye","right_eye"],"wing":["left_wing","right_wing"],"leg":["left_leg","right_leg"]}
        side=[]
        for group,parts2 in pairmap.items():
            d=RAWVIS[RAWVIS.part.isin(parts2)]
            pv=d.pivot(index="image_name",columns="part",values="visible").fillna(False)
            pa=d.pivot(index="image_name",columns="part",values="area_frac").fillna(0)
            for image in pv.index:
                side.append({"image":image,"mask_group":group,"visible_sides":int(pv.loc[image].sum()),"bilateral_area":float(pa.loc[image].sum())})
        SIDE=pd.DataFrame(side)
        B=J70[(J70.gt_label==1)&J70.mask_group.isin(pairmap)].merge(SIDE,on=["image","mask_group"])
        BS=B.groupby(["mask_group","visible_sides"]).agg(n=("z","size"),mean_z=("z","mean")).reset_index()
        dose=[]
        for (t,c),d in J70[(J70.gt_label==1)&(J70.area_frac>0)].groupby(["attribute_type","concept_name"]):
            if len(d)<20 or d.area_frac.nunique()<4: continue
            q=pd.qcut(d.area_frac,4,duplicates="drop")
            if q.nunique()<2: continue
            lo=d.loc[q==q.cat.categories[0],"z"].mean(); hi=d.loc[q==q.cat.categories[-1],"z"].mean()
            dose.append({"attribute_type":t,"concept_name":c,"mask_group":d.mask_group.iloc[0],"area_effect":hi-lo,"n":len(d)})
        DOSE=pd.DataFrame(dose)
        fig,axes=plt.subplots(1,2,figsize=(13,4.5))
        for g,d in BS.groupby("mask_group"): axes[0].plot(d.visible_sides,d.mean_z,"o-",label=g)
        axes[0].set_xticks([0,1,2]); axes[0].set_xlabel("visible left/right masks"); axes[0].set_ylabel("mean raw z"); axes[0].legend()
        for g,d in DOSE.groupby("mask_group"): axes[1].scatter([g]*len(d),d.area_effect,label=g,alpha=.65)
        axes[1].axhline(0,color="black",lw=.8); axes[1].set_ylabel("largest-area quartile z − smallest-area quartile z")
        fig.suptitle("Figure 7 · Bilateral visibility and area dose response")
        plt.tight_layout(); plt.show(); display(BS.round(3)); display(DOSE.round(3))
        """, "CUB70 raw-logit response by number of visible bilateral masks and by within-concept visible-area quartiles."),
        review("cub-r7", "Figure 7"),

        question("cub-q8", "8", "Does concept performance differ between species after support is matched?",
                 "Join the original CUB per-image attribute labels to the CBM raw `z` predictions. For each exact concept, compare species that each contain at least three raw positive and three raw negative images. Equalize positive and negative support, then measure both recall gap and positive-row raw-z gap.",
                 "Persistent gaps support species-dependent representation but remain observational.",
                 "Use the refined CUB matching rule from `mcbm_recallv4`: raw image-level labels, deterministic vectorized bootstrap, at most 50 species pairs per exact concept, and explicit alignment/eligibility counts."),
        code("cub-f8", r"""
        if "attribute_id" not in E70.columns:
            raise RuntimeError(
                "ERROR: CUB export lacks attribute_id; rerun cub70_export_eval.py "
                "after pulling the current repository"
            )
        cub_root=CURATED/"CUB_200_2011"
        raw_candidates=[cub_root/"attributes"/"image_attribute_labels.txt",
                        cub_root/"image_attribute_labels.txt"]
        raw_path=next((p for p in raw_candidates if p.exists()),None)
        images_path=cub_root/"images.txt"
        if raw_path is None or not images_path.exists():
            raise FileNotFoundError(
                f"ERROR: raw CUB annotations missing under {cub_root}; need "
                "image_attribute_labels.txt and images.txt"
            )
        raw=pd.read_csv(raw_path,sep=r"\s+",header=None,usecols=[0,1,2,3])
        raw.columns=["image_id","attribute_id","raw_label","certainty"]
        raw=raw[raw.certainty>=1].drop_duplicates(["image_id","attribute_id"])
        image_rows=[]
        for line in images_path.read_text().splitlines():
            image_id,relative=line.split(maxsplit=1)
            image_rows.append({"image_id":int(image_id),"image":Path(relative).stem})
        image_ids=pd.DataFrame(image_rows)
        raw_eval=(E70.merge(image_ids,on="image",how="left",validate="many_to_one")
                  .merge(raw[["image_id","attribute_id","raw_label","certainty"]],
                         on=["image_id","attribute_id"],how="inner",validate="one_to_one"))
        alignment_rate=len(raw_eval)/len(E70)
        if alignment_rate<0.98:
            raise RuntimeError(
                f"ERROR: raw-label alignment covered only {alignment_rate:.1%} of E70 rows"
            )
        rng=np.random.default_rng(20260803); rows=[]; eligibility=[]; B=100
        for (t,c),d in raw_eval.groupby(["attribute_type","concept_name"]):
            eligible=[]
            for sid,g in d.groupby("y_true"):
                pos=g[g.raw_label==1].z.to_numpy(); neg=g[g.raw_label==0].z.to_numpy()
                if len(pos)>=3 and len(neg)>=3:
                    eligible.append((int(sid),pos,neg))
            eligibility.append({"attribute_type":t,"concept_name":c,
                                "eligible_species":len(eligible)})
            pairs=[(eligible[a],eligible[b]) for a in range(len(eligible)) for b in range(a+1,len(eligible))]
            if len(pairs)>50:
                pairs=[pairs[i] for i in rng.choice(len(pairs),50,replace=False)]
            for (sa,za,na),(sb,zb,nb) in pairs:
                mpos=min(len(za),len(zb)); mneg=min(len(na),len(nb))
                aa=za[rng.integers(len(za),size=(B,mpos))]
                bb=zb[rng.integers(len(zb),size=(B,mpos))]
                recall_gaps=np.abs((aa>0).mean(axis=1)-(bb>0).mean(axis=1))
                z_gaps=np.abs(aa.mean(axis=1)-bb.mean(axis=1))
                rows.append({"attribute_type":t,"concept_name":c,"species_a":sa,"species_b":sb,
                             "matched_positive_n":mpos,"matched_negative_n":mneg,
                             "recall_gap":recall_gaps.mean(),"recall_gap_lo":np.quantile(recall_gaps,.025),
                             "recall_gap_hi":np.quantile(recall_gaps,.975),
                             "raw_z_gap":z_gaps.mean(),"raw_z_gap_lo":np.quantile(z_gaps,.025),
                             "raw_z_gap_hi":np.quantile(z_gaps,.975)})
        RECALL=pd.DataFrame(rows,columns=["attribute_type","concept_name","species_a","species_b",
            "matched_positive_n","matched_negative_n","recall_gap","recall_gap_lo","recall_gap_hi",
            "raw_z_gap","raw_z_gap_lo","raw_z_gap_hi"])
        ELIGIBILITY=pd.DataFrame(eligibility)
        if RECALL.empty:
            raise RuntimeError(
                "ERROR: raw image-level CUB labels produced no eligible matched species pairs"
            )
        RS=(RECALL.groupby(["attribute_type","concept_name"]).agg(n_species_pairs=("recall_gap","size"),
             mean_recall_gap=("recall_gap","mean"),mean_raw_z_gap=("raw_z_gap","mean"),
             min_matched_positive_n=("matched_positive_n","min"),
             min_matched_negative_n=("matched_negative_n","min")).reset_index())
        fig,axes=plt.subplots(1,2,figsize=(14,max(12,.24*len(RS))),sharey=True)
        RS=RS.sort_values(["attribute_type","concept_name"]).reset_index(drop=True); y=np.arange(len(RS))
        axes[0].scatter(RS.mean_recall_gap,y,c="#0072B2",s=24); axes[1].scatter(RS.mean_raw_z_gap,y,c="#E69F00",s=24)
        axes[0].set_yticks(y); axes[0].set_yticklabels(RS.concept_name,fontsize=7); axes[0].invert_yaxis()
        axes[0].set_xlabel("matched absolute positive-recall gap"); axes[1].set_xlabel("matched absolute positive-row raw-z gap")
        fig.suptitle("Figure 8 · Species-matched concept differences")
        plt.tight_layout(); plt.show()
        display(pd.DataFrame([{"raw_alignment_rate":alignment_rate,"raw_rows":len(raw_eval),
                               "eligible_concepts":int((ELIGIBILITY.eligible_species>=2).sum()),
                               "matched_pairs":len(RECALL)}]).round(3))
        display(RS.round(3))
        display(RECALL.nlargest(25,"raw_z_gap")[["concept_name","species_a","species_b",
            "matched_positive_n","matched_negative_n","recall_gap","recall_gap_lo","recall_gap_hi",
            "raw_z_gap","raw_z_gap_lo","raw_z_gap_hi"]].round(3))
        """, "Aligned CUB70 exact-concept plots of matched per-species positive-recall gaps and raw-logit gaps using original per-image CUB attribute labels; alignment and eligibility counts are displayed."),
        review("cub-r8", "Figure 8"),

        question("cub-q9", "9", "Do conflict, support, and number of alternatives organize the exact-concept effects?",
                 "At the concept level, relate `visibility_effect` and `context_gap` to label/mask conflict, image support, species support, and alternatives in the attribute type.",
                 "Held-out predictive improvement supports an organizing association, not a causal contribution.",
                 "Use standardized numeric predictors and repeated five-fold ridge regression."),
        code("cub-f9", r"""
        from sklearn.model_selection import RepeatedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        FEATURES=["label_mask_conflict","n_positive","species_support","alternatives_in_type"]
        collapsed_names=set(HEALTH.loc[HEALTH.collapsed,"concept_name"])
        ACCOUNT_BASE=EXACT[(EXACT.n_visible>=10)&(EXACT.n_hidden>=10)&(EXACT.n_hidden_negative>=10)
                           & ~EXACT.concept_name.isin(collapsed_names)].copy()
        rows=[]
        for outcome in ["visibility_effect","context_gap"]:
            d=ACCOUNT_BASE.dropna(subset=[outcome]).copy()
            cv=RepeatedKFold(n_splits=5,n_repeats=10,random_state=20260803)
            baseline=np.sqrt(np.mean((d[outcome]-d[outcome].mean())**2))
            for k in range(1,len(FEATURES)+1):
                model=make_pipeline(SimpleImputer(),StandardScaler(),Ridge(alpha=5.0))
                mse=-cross_val_score(model,d[FEATURES[:k]],d[outcome],cv=cv,scoring="neg_mean_squared_error")
                rows.append({"outcome":outcome,"stage":" + ".join(FEATURES[:k]),"rmse":float(np.sqrt(mse.mean())),"n_concepts":len(d)})
            rows.append({"outcome":outcome,"stage":"intercept only","rmse":baseline,"n_concepts":len(d)})
        CONCEPT_ACCOUNT=pd.DataFrame(rows)
        fig,axes=plt.subplots(1,2,figsize=(14,4.5))
        for ax,outcome in zip(axes,["visibility_effect","context_gap"]):
            d=CONCEPT_ACCOUNT[CONCEPT_ACCOUNT.outcome==outcome]
            order=["intercept only"]+[" + ".join(FEATURES[:k]) for k in range(1,len(FEATURES)+1)]
            d=d.set_index("stage").reindex(order); ax.plot(range(len(d)),d.rmse,"o-")
            ax.set_xticks(range(len(d))); ax.set_xticklabels(["baseline","+ conflict","+ image support","+ species support","+ alternatives"],rotation=25,ha="right")
            ax.set_ylabel("cross-validated RMSE"); ax.set_title(outcome.replace("_"," "))
        fig.suptitle("Figure 9 · Concept-level sequential observational accounting")
        plt.tight_layout(); plt.show()
        display(pd.DataFrame([{"shared_eligible_concepts":len(ACCOUNT_BASE),
            "excluded_collapsed":len(collapsed_names),"minimum_visible_positive":10,
            "minimum_hidden_positive":10,"minimum_hidden_negative":10}]))
        display(CONCEPT_ACCOUNT.round(3))
        """, "Cross-validated concept-level error after sequentially adding label conflict, image support, species support, and number of alternatives."),
        review("cub-r9", "Figure 9"),

        question("cub-q10", "10", "Does species explain raw-score variation within the same exact concept and visibility state?",
                 "First center `z` within each exact concept and visibility state, then summarize residual means by species.",
                 "Persistent spread shows species-dependent contextual prediction beyond the current mask state.",
                 "Require at least three rows for every displayed concept/state/species estimate."),
        code("cub-f10", r"""
        R=J70.copy(); R["concept_visibility_mean"]=R.groupby(["concept_name","visible"]).z.transform("mean")
        R["z_after_concept_visibility"]=R.z-R.concept_visibility_mean
        SP=(R.groupby(["mask_group","concept_name","visible","y_true"]).agg(n=("z","size"),residual=("z_after_concept_visibility","mean"))
              .reset_index().query("n>=3"))
        fig,axes=plt.subplots(2,4,figsize=(16,8),sharey=True); axes=axes.ravel()
        for ax,g in zip(axes,COARSE_ORDER):
            d=SP[SP.mask_group==g].sort_values("residual")
            ax.scatter(np.arange(len(d)),d.residual,s=12,color=COLORS[g],alpha=.7)
            ax.axhline(0,color="black",lw=.8); ax.set_title(f"{g}: {len(d)} estimates"); ax.set_xlabel("concept/state/species, sorted")
        axes[0].set_ylabel("mean raw-z residual"); axes[4].set_ylabel("mean raw-z residual")
        fig.suptitle("Figure 10 · Species variation after exact concept and mask state")
        plt.tight_layout(); plt.show(); display(SP.groupby("mask_group").residual.agg(["min","median","max","std","count"]).round(3))
        """, "CUB70 species-level raw-logit residuals after centering within exact concept and visibility state for all eight coarse groups."),
        review("cub-r10", "Figure 10"),

        question("cub-q11", "11", "What remains after row-level visibility and species are added sequentially?",
                 "Predict raw `z` on stable held-out image folds: exact concept baseline, then mask visibility/area, then species.",
                 "A reduction in held-out error shows organization by that block; remaining error is the residual, not proof of an unknown cause.",
                 "Use training-fold shrunken group means and identical rows at every stage."),
        code("cub-f11", r"""
        A=J70.copy(); A["area_bin"]=pd.qcut(A.area_frac,4,labels=False,duplicates="drop")
        A["fold"]=A.image.map(lambda x:int(hashlib.sha1(str(x).encode()).hexdigest(),16)%5)
        stages=[("exact concept",["concept_name"]),("+ visibility and area",["concept_name","visible","area_bin"]),
                ("+ species",["concept_name","visible","area_bin","y_true"])]
        rows=[]
        for stage,cols in stages:
            pred=pd.Series(index=A.index,dtype=float)
            for fold in range(5):
                tr=A[A.fold!=fold]; te=A[A.fold==fold]; prior=tr.z.mean()
                st=tr.groupby(cols).z.agg(["mean","count"]).reset_index(); st["estimate"]=(st["mean"]*st["count"]+prior*10)/(st["count"]+10)
                j=te[cols].merge(st[cols+["estimate"]],on=cols,how="left")
                pred.loc[te.index]=j.estimate.fillna(prior).to_numpy()
            rows.append({"stage":stage,"rmse":float(np.sqrt(np.mean((A.z-pred)**2))),"mae":float(np.mean(np.abs(A.z-pred)))})
        ROW_ACCOUNT=pd.DataFrame(rows)
        fig,ax=plt.subplots(figsize=(7,4)); ax.plot(ROW_ACCOUNT.stage,ROW_ACCOUNT.rmse,"o-",color="#0072B2")
        ax.set_ylabel("held-out RMSE of raw z"); ax.set_title("Figure 11 · Row-level sequential observational accounting")
        plt.tight_layout(); plt.show(); display(ROW_ACCOUNT.round(3))
        """, "Held-out CUB70 raw-logit prediction error after sequentially adding visibility, area, and species to exact concept identity."),
        review("cub-r11", "Figure 11"),

        question("cub-q12", "12", "Do the numerical extremes correspond to pose, coarse masks, collapse, or contextual prediction?",
                 "Select cases by declared numerical rules: high conflict/high context gap, high conflict/low gap, strong positive visibility effect, and negative visibility effect.",
                 "The photograph and all 11 masks must be inspected before assigning an explanation.",
                 "Display original image, complete mask overlay, exact variables, species, and sample counts."),
        code("cub-f12", r"""
        from PIL import Image
        mask_root=CURATED/"cub70"/"masks"/"AnnotationMasksPerclass"
        if not mask_root.is_dir(): mask_root=CURATED/"cub70"/"masks"
        image_root=CURATED/"CUB_200_2011"/"images"; image_lookup={p.stem:p for p in image_root.rglob("*.jpg")}
        collapsed_names=set(HEALTH.loc[HEALTH.collapsed,"concept_name"])
        eligible=EXACT[(EXACT.n_visible>=10)&(EXACT.n_hidden>=10)&(EXACT.n_hidden_negative>=10)
            & EXACT.context_gap.notna()&EXACT.visibility_effect.notna()
            & ~EXACT.concept_name.isin(collapsed_names)].copy()
        conflict_q75=float(eligible.label_mask_conflict.quantile(.75))
        high=eligible[eligible.label_mask_conflict>=conflict_q75]
        picks=[("high conflict + high context gap",high.nlargest(1,"context_gap").iloc[0]),
               ("high conflict + low context gap",high.nsmallest(1,"context_gap").iloc[0]),
               ("strong positive visibility effect",eligible.nlargest(1,"visibility_effect").iloc[0]),
               ("negative visibility effect",eligible.nsmallest(1,"visibility_effect").iloc[0])]
        mask_colors={p:plt.cm.tab20(i/20) for i,p in enumerate(CUB70_PARTS)}
        mapped_parts={"eye":["left_eye","right_eye"],"wing":["left_wing","right_wing"],
                      "leg":["left_leg","right_leg"]}
        def choose(row,state):
            d=J70[(J70.concept_name==row.concept_name)&(J70.gt_label==1)]
            d=d[d.visible] if state=="visible" else d[~d.visible]
            return d.iloc[(d.z-row.z_visible).abs().argmin()] if len(d) and state=="visible" else (d.iloc[(d.z-row.z_hidden).abs().argmin()] if len(d) else None)
        def overlays(stem,group):
            rgb=np.asarray(Image.open(image_lookup[stem]).convert("RGB")); all_ov=rgb.astype(float)/255; mapped_ov=all_ov.copy()
            rr=RAWVIS[RAWVIS.image_name==stem]; cid=int(rr.class_idx.iloc[0])+1; present=[]
            for p in CUB70_PARTS:
                f=mask_root/str(cid)/f"{stem}_{p}.png"
                if not f.exists(): continue
                m=np.asarray(Image.open(f).convert("L"))>0
                if m.shape!=rgb.shape[:2]: m=np.asarray(Image.fromarray(m.astype("uint8")*255).resize((rgb.shape[1],rgb.shape[0]),Image.Resampling.NEAREST))>0
                all_ov[m]=.4*all_ov[m]+.6*np.array(mask_colors[p][:3]); present.append(p)
                if p in mapped_parts.get(group,[group]): mapped_ov[m]=.3*mapped_ov[m]+.7*np.array(mask_colors[p][:3])
            return rgb,mapped_ov,all_ov,present
        # Two rows per case keep each photograph large enough to inspect in HTML:
        # hidden original/mapped/all masks, then visible original/mapped/all masks.
        records=[]; fig,axes=plt.subplots(8,3,figsize=(13,28))
        for r,(label,row) in enumerate(picks):
            for offset,state in [(0,"hidden"),(1,"visible")]:
                rr=2*r+offset
                rec=choose(row,state)
                for k in range(3): axes[rr,k].axis("off")
                if rec is None: axes[rr,0].text(.5,.5,"no example",ha="center"); continue
                if rec.concept_name!=row.concept_name: raise RuntimeError("example/concept mismatch")
                rgb,mapped,all_ov,present=overlays(rec.image,row.mask_group)
                axes[rr,0].imshow(rgb); axes[rr,1].imshow(mapped); axes[rr,2].imshow(all_ov)
                axes[rr,0].set_title(f"case {r+1}: {label}\n{state}: {rec.image}; species {rec.y_true}\n{row.concept_name}\nc={int(rec.gt_label)}, c_hat={int(rec.pred_label)}, z={rec.z:.3f}, area={rec.area_frac:.4f}",fontsize=9)
                axes[rr,1].set_title(f"mapped {row.mask_group} mask",fontsize=10)
                axes[rr,2].set_title("all available masks\n"+", ".join(present),fontsize=8)
                records.append({"case":r+1,"rule":label,"state":state,"image":rec.image,"species":rec.y_true,
                    "concept_name":row.concept_name,"mask_group":row.mask_group,"c":int(rec.gt_label),
                    "c_hat":int(rec.pred_label),"z":rec.z,"area_frac":rec.area_frac,
                    "label_mask_conflict":row.label_mask_conflict,"visibility_effect":row.visibility_effect,
                    "context_gap":row.context_gap,"n_visible":row.n_visible,"n_hidden":row.n_hidden,
                    "n_hidden_negative":row.n_hidden_negative})
        fig.suptitle("Figure 12 · Rule-selected photographs and complete mask overlays")
        plt.tight_layout(); plt.show()
        display(pd.DataFrame([{"selection_rule":label,"conflict_q75_threshold":conflict_q75,**row.to_dict()} for label,row in picks])
            [["selection_rule","conflict_q75_threshold","concept_name","mask_group","label_mask_conflict","visibility_effect","context_gap","n_visible","n_hidden","n_hidden_negative"]].round(3))
        display(pd.DataFrame(records).round(3))
        """, "Four rule-selected CUB70 cases, each showing hidden and visible photographs beside overlays of all available released masks and exact raw-logit records."),
        review("cub-r12", "Figure 12"),

        question("cub-q12b", "12b", "Do the main observational quantities depend entirely on training with only 70 species?",
                 "On the same mask-matched photographs and exact concepts, compare CUB70-CBM and full-CUB-CBM visibility effects and context gaps.",
                 "Agreement supports robustness to the training species population; disagreement limits transfer between the two models.",
                 "Use identical definitions and plot only concepts measurable in both exports."),
        code("cub-f12b", r"""
        def exact_effects(J):
            rows=[]
            for (t,c),d in J.groupby(["attribute_type","concept_name"]):
                scale=d.z.std(ddof=0)
                if not np.isfinite(scale) or scale<=COLLAPSE_TOL: continue
                d=d.copy(); d["z_standardized"]=(d.z-d.z.mean())/scale
                pos=d[d.gt_label==1]; vis=pos[pos.visible]; hid=pos[~pos.visible]; neg=d[(d.gt_label==0)&(~d.visible)]
                rows.append({"attribute_type":t,"concept_name":c,
                             "visibility_effect":vis.z_standardized.mean()-hid.z_standardized.mean() if len(vis)>=10 and len(hid)>=10 else np.nan,
                             "context_gap":hid.z_standardized.mean()-neg.z_standardized.mean() if len(hid)>=10 and len(neg)>=10 else np.nan})
            return pd.DataFrame(rows)
        F70=exact_effects(J70); F=exact_effects(JFULL); P=F70.merge(F,on=["attribute_type","concept_name"],suffixes=("_cub70","_full"))
        fig,axes=plt.subplots(1,2,figsize=(11,5))
        for ax,m in zip(axes,["visibility_effect","context_gap"]):
            d=P.dropna(subset=[m+"_cub70",m+"_full"]); ax.scatter(d[m+"_full"],d[m+"_cub70"],s=25,alpha=.65)
            lo=min(d[m+"_full"].min(),d[m+"_cub70"].min()); hi=max(d[m+"_full"].max(),d[m+"_cub70"].max())
            ax.plot([lo,hi],[lo,hi],"k--",lw=.8); ax.axhline(0,color="gray",lw=.5); ax.axvline(0,color="gray",lw=.5)
            ax.set_xlabel("full-CUB CBM standardized "+m.replace("_"," ")); ax.set_ylabel("CUB70 CBM standardized "+m.replace("_"," ")); ax.set_title(f"{m.replace('_',' ')} (n={len(d)})")
        fig.suptitle("Figure 12b · Same-image guard: CUB70-trained versus full-CUB-trained CBM")
        plt.tight_layout(); plt.show()
        """, "Same-image comparison of raw-logit visibility effects and context gaps between CUB70-trained and full-CUB-trained CBMs."),
        review("cub-r12b", "Figure 12b"),

        md("cub-compare", r"""
        ## 13 · Direct question-matched FunnyBird/CUB evidence table

        Figures 1–12b and the corresponding FunnyBird figures were displayed and
        reviewed together on 2026-08-04.

        | Scientific question | FunnyBird operation | CUB operation | Same operation? | Allowed conclusion |
        |---|---|---|---|---|
        | Are outputs usable? | all 26 healthy | 110/112 non-collapsed | yes | compare only healthy outputs |
        | Do named pixels matter? | positive controlled `response_delta` for all parts | mixed natural `visibility_effect` | no | causal FunnyBird response; no universal CUB response |
        | Does context remain? | source wins after donorward response | positive released-mask-absent `context_gap` | no | exact backwash predicate in FunnyBird; observational contextual separation in CUB |
        | Does visibility contribute? | same-render target area improves margin | natural mask state/area/sides mixed | weaker in CUB | FunnyBird contributor accepted; CUB result is heterogeneous and mask-limited |
        | Does exact value matter? | post-swap value confusion is strongly graded | natural exact-concept matching still leaves species gaps | no | value difficulty matters in FunnyBird; CUB has related observational variation |
        | Does species matter? | descriptive residual remains, but held-out margin prediction does not improve | residual remains and species lowers held-out raw-z error | observational in both | CUB gives stronger generalizing association; neither is causal species manipulation |
        | Do training labels cause part of it? | conflict measured; matched RLv2 belongs to notebook 03rl | no accepted CUB retraining | no | no causal label conclusion in either standard-CBM report |

        ### CUB causal boundary

        Notebook 05 may conclude that CUB does or does not show converging
        **observational ingredients** of context-dependent concept prediction. It
        may not claim a CUB donor/source backwash event because no accepted donor
        response exists.
        """),
        md("cub-ledger", r"""
        ## 14 · Standard-CUB evidence ledger

        | Predicate or explanation | Direct measurement | Status after review |
        |---|---|---|
        | population and mask coverage understood | Figure 1 | `ACCEPTED WITH MISSING-MASK LIMIT` |
        | species/concept shortcut available | Figure 2 | `ACCEPTED FOR AVAILABILITY` |
        | label/released-mask conflict measured | Figure 3 | `ACCEPTED; NOT PHYSICAL-OCCLUSION RATE` |
        | exact outputs usable | Figure 4 | `110 ACCEPTED; 2 COLLAPSED AND EXCLUDED FROM POSITIVE CLAIMS` |
        | species information in learned representation | Figure 4b | `ACCEPTED FOR AVAILABILITY` |
        | natural visibility effect | Figure 5 | `MIXED; NO UNIVERSAL RESPONSE` |
        | hidden context separation | Figure 6 | `ACCEPTED OBSERVATIONALLY; NOT A DONOR/SOURCE MARGIN` |
        | bilateral/area alternatives | Figure 7 | `VALID TEST, NO SUFFICIENT UNIVERSAL EXPLANATION` |
        | matched recall and raw-z species gaps | Figure 8 | `ACCEPTED OBSERVATIONALLY` |
        | concept-level accounting | Figure 9 | `INCOMPLETE: SHARED-ELIGIBILITY RERENDER/REVIEW REQUIRED` |
        | species residual | Figure 10 | `DESCRIPTIVE ASSOCIATION` |
        | row-level accounting | Figure 11 | `SPECIES LOWERS HELD-OUT ERROR` |
        | visual explanations inspected | Figure 12 | `INCOMPLETE: CORRECTED GRID REQUIRES IMAGE-BY-IMAGE REVIEW` |
        | same-image full-CUB robustness guard | Figure 12b | `INCOMPLETE: STANDARDIZED RERENDER/REVIEW REQUIRED` |

        **Next report question.** Only after this ledger is reviewed may notebook
        06 ask whether CUB MCBM changes the accepted observational quantities.
        """),
        md("cub-appendix", r"""
        # Methods appendix · CUB edit proxies not used in the main claim

        These completed attempts are preserved because they delimit what CUB's
        available masks can support:

        1. **Reciprocal whole-part deletion:** `METHOD NOT CALIBRATED FOR
           CROSS-DATASET CAUSAL COMPARISON`. The shared edit did not reproduce the
           clean FunnyBird deletion and sometimes damaged meaningful control regions.
        2. **Randomized patch masking V1/V2:** selected examples supported local
           pixel response, but the all-part calibration and wing coverage were not
           sufficient for a population-level cross-dataset claim.
        3. **Beak/tail paste pilot:** `VALID TEST, NO SUPPORT FOR POSITIVE DONOR
           RESPONSE`. Therefore its negative final margins cannot be interpreted as
           retained-source backwash.

        These outcomes reject the proposed edit measurements for their intended
        causal use. They do not reject the observational analyses in Figures 1–12
        and do not weaken the validated FunnyBird renderer swap.

        Full code and artifacts remain in the repository and `CURATED_DATA`; this
        report does not rerun them.
        """),
        md("cub-prov", r"""
        # Provenance appendix

        The table below records the live Git commit, prediction and mask inputs,
        SHA-256 hashes, population counts, collapse tolerance, and exclusions.
        """),
        code("cub-prov-code", r"""
        def sha256_file(path):
            h=hashlib.sha256()
            with open(path,"rb") as f:
                for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
            return h.hexdigest()
        commit=subprocess.run(["git","rev-parse","HEAD"],cwd=REPO,capture_output=True,text=True,check=True).stdout.strip()
        prov=[]
        for role,path in [("CUB70 prediction export",E70P),("full-CUB prediction export",EFULLP),("visibility parquet",VIS)]:
            prov.append({"role":role,"path":str(path),"sha256":sha256_file(path)})
        display(pd.DataFrame(prov)); display(pd.DataFrame([{"git_commit":commit,"seed":1,"epoch":100,
            "prediction_images":E70.image.nunique(),"mask_matched_images":J70.image.nunique(),
            "species":E70.y_true.nunique(),"exact_concepts":E70.concept_name.nunique(),
            "collapsed_concepts":int(HEALTH.collapsed.sum()),"collapse_tolerance":COLLAPSE_TOL,
            "visibility_threshold_area_fraction":.001}]))
        """),
    ]
    # The scientific order is data structure -> label/mask conflict -> model
    # health -> species decoding.  These blocks are authored together above
    # because they share setup objects, then placed here in report order.
    def tag_of(cell):
        return cell["id"].rsplit("-", 1)[0]
    desired = [
        "cub-q4", "cub-f4", "cub-r4",
        "cub-q3", "cub-f3", "cub-r3",
        "cub-q2b", "cub-f2b", "cub-r2b",
    ]
    positions = [i for i,c in enumerate(cells) if tag_of(c) in desired]
    selected = {tag_of(c): c for c in cells if tag_of(c) in desired}
    if len(positions) != len(desired) or len(selected) != len(desired):
        raise RuntimeError("CUB core report blocks are missing or duplicated")
    for i in reversed(positions):
        cells.pop(i)
    insert_at = min(positions)
    cells[insert_at:insert_at] = [selected[tag] for tag in desired]
    return notebook(cells, NOTEBOOKS/"05_cub_cbm.ipynb")


def write(name: str, obj: dict) -> None:
    path = NOTEBOOKS/name
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    print(f"wrote {path}: {len(obj['cells'])} cells")


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--only",choices=["02","05"])
    args=ap.parse_args()
    if args.only in (None,"02"): write("02_funnybirds_cbm.ipynb",build_funnybird())
    if args.only in (None,"05"): write("05_cub_cbm.ipynb",build_cub())


if __name__=="__main__":
    main()
