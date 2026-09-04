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
    Figure 2a is the renderer's semantic preflight: for each part it shows the
    original, replacement, deletion, original part map, and replacement part
    map. In a visible example, the named part should change while body, pose,
    camera, and background remain fixed. A cached row with identical original
    and replacement RGB pixels is not called visibly changed; it is retained for
    the later exact visibility analysis. This is a pixel-operation gate; it
    contains no model result.
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
    Example: a replacement with 120 target-part pixels enters the `100--199`
    bin; binning records visibility already present in the render and does not
    add pixels to the image or information to the CBM.
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
    Each labelled point is one exact donor value. The x-axis is its species
    support: the number of the 50 FunnyBird species that naturally carry that
    value in an unmodified bird. It is not an image count or swap count. The
    count comes from the renderer's species-to-part-value definition: if six
    species ordinarily have donor value 2, its support is 6 even when the swap
    table contains hundreds of value-2 rows. The
    three panels partition all swaps into donor wins, donorward movement that
    remains source-negative, and no donorward movement while source-negative.
    The three fractions sum to one within each value. A consistent relationship
    with support would make rarity a plausible organizer. The number of
    alternatives is reported but cannot be cleanly separated with only five
    parts.
    """,
    "fb-q8": """
    First remove the average margin for the same part, source value, and donor
    value. Each remaining cell is a source-species mean residual. The mean is
    forced to zero inside each exact value pair, not inside each species. A
    species can therefore remain above or below zero if it repeatedly lies above
    or below the appropriate exact-pair average. Rows retain source-species IDs
    and columns retain parts; blue is more source-retaining, red is more
    donor-receptive, and white is zero. This is association with the unchanged
    bird context, not an independent species manipulation.
    """,
    "fb-q8b": """
    The y-axis is held-out species-classification accuracy. For every block, grey
    uses processed 0/1 concept labels, color uses learned raw logits, and the
    outlined bar uses raw logits after the training-fold mean for the same 0/1
    label has been removed. The grey bar is the structural control: with balanced
    FunnyBird species and `K` mutually exclusive values for one part, it is
    approximately `K/50` (tail has 9 values, so 9/50=0.18), not 1/50. The residual
    bar asks whether score magnitudes still identify species after the nominal
    label bucket is removed. This diagnoses available information only; it does
    not prove that the saved CBM uses it or that it caused backwash.
    """,
    "fb-q9": """
    Panel A compares the median final raw-logit margin for all rows, rows with a
    nonzero inserted-part mask, and rows with at least 100 inserted-part pixels;
    markers are unconnected because these are nested descriptive selections, not
    a trajectory. Panel B is held-out RMSE when predicting final margin; lower is
    better. Starting from part alone, blocks are added in order: visibility,
    exact source/donor values, then source species. A decrease means the new block
    predicts swaps from unseen original source images better. Every swap derived
    from one original image stays in the same fold. The final nonzero error is the residual;
    an increase is negative evidence for that proposed organizer. RMSE 10 to 8
    is improvement; RMSE 10 to 11 is not.
    """,
    "fb-q9b": """
    All four panels use the same y-axis part order. Panel A is the fraction of
    all swaps satisfying `response_delta>0 and m_cf<0`. Panel B repeats that
    fraction only when the inserted target occupies at least 100 pixels. Panel C
    is the fraction of original positive training labels removed by the matched
    visibility rule. Panel D is one minus the post-swap inserted-value recognition
    rate. Larger is worse in every panel, but the denominators and meanings differ,
    so the bar heights must not be added. The shared ordering asks whether the
    proposed contributors align with the controlled outcome.
    """,
    "fb-q10": """
    Swaps are divided into ten non-overlapping, approximately equal-count bins by final donor-minus-source concept
    margin on the x-axis. The y-axis is the model's mean probability for the donor
    species, with the number of rows printed per bin. This asks whether concept
    grounding failure is associated with a downstream class quantity; it is intentionally the one
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
    The y-axis is held-out species accuracy. Paired bars use learned raw logits
    versus processed 0/1 concept labels on the same train/test split. The dashed
    line is 1/70 blind chance; the dotted line is the saved CUB70 CBM's own task
    accuracy. Unlike FunnyBird, `dimensions/70` is not a valid bucket baseline:
    CUB region blocks contain multiple simultaneous attribute types and labels
    vary within species. The label-only probe is therefore the valid structural
    control. Raw-z accuracy above it is extra species information in the learned
    representation, not proof of causal backwash.
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
    "cub-q11a": """
    Every row is one exact mask-testable CUB concept, such as
    `has_tail_pattern::striped`, and the row order is identical across all five
    panels. Color identifies the coarse mask group. Panel A is the fraction of
    positive labels with the mapped mask absent. Panel B is ordinary
    classification error `1-balanced_accuracy`. Panel C is visible-minus-hidden
    raw `z` among positive labels. Panel D is hidden-positive minus hidden-negative
    raw `z`. Panel E is the standard deviation of species residuals after exact
    concept and mask state are centered. Blank positions mean that exact concept
    lacked the required visible/hidden/species support; they are not zeros.
    """,
    "cub-q11b": """
    Every panel uses the same coarse-group order: tail, wing, beak, leg, eye,
    neck, body, head. Panel A is the positive-label/mapped-mask-absence fraction.
    Panel B is `1 - median balanced accuracy` across non-collapsed exact concepts.
    Panel C is the median visible-minus-hidden raw-z association. Panel D is the
    median hidden-positive-minus-hidden-negative context gap. Panel E is the
    standard deviation of species residuals after exact concept and mask state
    are centered. Higher means more of the named quantity, but these quantities
    have different units and none is a controlled CUB swap failure rate.
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


def figure_method(tag: str, source: str) -> dict:
    """Place one operator-readable method line immediately below a figure."""
    return md(tag, f"- **Method in one line:** {source}")


REVIEWS = {
    "fb-r1": (
        "Across 500 held-out images, all 26 exact outputs vary: raw-z spread is "
        "5.907-13.616, label separation is 8.069-12.483, balanced accuracy is "
        "0.969-1.000, and positive recall is 0.940-1.000. Species accuracy is "
        "0.992 and concept accuracy is 0.9968.",
        "Excellent label prediction can still come from species context rather than "
        "the named part, so this figure establishes health but not grounding.",
        "Use the same-image controlled replacement in Figures 3-4.",
        "ACCEPTED FOR seed-1 standard-CBM model health; no exact output is collapsed.",
        "Is the fixed renderer intervention itself valid?",
    ),
    "fb-r2": (
        "For tail, wing, beak, foot, and eye, the displayed replacement and deletion "
        "alter the named part while the body, pose, camera, and background remain fixed; "
        "the target part map contains that region. Across the complete cache, 98.3% of "
        "replacement RGB images differ visibly from their originals. The remaining 1.7% "
        "are retained and identified by the later pixel-visibility measurement rather "
        "than described as visibly changed.",
        "The five displayed rows are representative semantic checks, not by themselves "
        "proof about every cached image.",
        "Retain the semantic preflight plus the accepted fixed-render hash/diversity "
        "validation across all evaluated models and render IDs.",
        "ACCEPTED FOR the validated FunnyBird fixed-render intervention, with visibly "
        "unchanged rows retained explicitly for the visibility analysis.",
        "Do those inserted pixels move the raw concept comparison toward the donor?",
    ),
    "fb-r3": (
        "The complete response distributions are donorward for nearly every swap. "
        "Positive-response rates are tail 0.919, wing 1.000, beak 0.989, foot "
        "0.997, and eye 0.986, with 1,000 swaps per part.",
        "A positive movement alone does not say that the inserted donor finishes above "
        "the old source.",
        "Inspect the final donor-minus-source margin jointly with response_delta.",
        "ACCEPTED FOR a causal within-image response of the controlled part-replacement "
        "intervention in all five part groups. This is a part-level statement: the observed "
        "positive-response rates are 0.919-1.000, not a claim that every individual row "
        "responded. Tail, beak, and eye have smaller typical movement than wing and foot; "
        "tail is not the mechanism and is not the only comparatively weak response.",
        "Did the parts start equally far behind, and did donor rise versus source release "
        "contribute differently?",
    ),
    "fb-r4": (
        "Median final margins are tail -0.819, wing 6.483, beak 2.551, foot "
        "5.158, and eye 3.511 raw-logit units. On the same 1,000 swaps per part, "
        "the donorward-response-but-source-wins rates are 0.502, 0.019, 0.200, "
        "0.032, and 0.089, respectively.",
        "Starting preference, swap direction, target visibility, exact value difficulty, "
        "and source species could organize the unequal rates.",
        "Test those alternatives separately in Figures 5-9 without changing the event "
        "definition.",
        "ACCEPTED FOR the seed-1 controlled FunnyBird backwash predicate across a graded "
        "part ordering: strongest for tail, beak, and eye, with minority events also in "
        "wing and foot. This is not a tail-specific mechanism claim.",
        "Can swap direction create the pooled pattern?",
    ),
    "fb-r5": (
        "Forward and backward results preserve the ordering. Tail rates are 0.528 "
        "and 0.476; beak is 0.200 in both; eye is 0.090 and 0.088; foot is "
        "0.022 and 0.042; wing is 0.010 and 0.028. Each direction has 500 swaps.",
        "Individual source/donor value pairs can still be asymmetric even when pooled "
        "directions agree.",
        "Inspect every exact inserted value and both direction-specific denominators.",
        "ACCEPTED FOR excluding opposite-direction cancellation as the main explanation.",
        "How does exact target visibility change the event?",
    ),
    "fb-r6": (
        "Visibility helps but is not sufficient. Tail's median margin changes from "
        "-0.819 over all rows to 0.057 for any visible target and 1.416 for targets "
        "with at least 100 pixels, while its event rate remains 0.372 in that clear-"
        "visibility population. Beak and eye rates generally fall with visibility; "
        "tail is non-monotone and its 500+ bin has only 23 rows.",
        "Pixel count is associated with pose, source/donor value, and species, so bins "
        "do not isolate visibility causally by themselves.",
        "Hold exact values and species fixed, and test the visibility-aware label change "
        "later with matched RLv2 training.",
        "ACCEPTED FOR visibility as a contributor, not a sufficient explanation.",
        "Did ordinary training assign positive labels when the part was not visible?",
    ),
    "fb-r6b": (
        "Across the 45,000 training and 5,000 validation images, the visibility "
        "rule removes 8,184 of 188,461 positive labels. By part, it removes "
        "7,489/37,707 tail labels (0.199), 367/37,617 beak (0.010), "
        "268/37,726 eye (0.007), 48/37,723 foot (0.001), and 12/37,688 wing "
        "(less than 0.001).",
        "These are training-signal counts, not measured causal effects on the trained "
        "standard model.",
        "Compare otherwise matched standard and RLv2 checkpoints on the same fixed renders.",
        "ACCEPTED FOR a measured, part-specific label/visibility conflict. It is extremely "
        "large for tail, small but nonzero for beak/eye, and near zero for wing/foot. This "
        "can explain tail's excess severity but cannot by itself explain backwash in every "
        "part; causal credit remains deferred to notebook 02rl.",
        "Are some exact visual variants much harder than others?",
    ),
    "fb-r7": (
        "Post-swap donor-value recognition is graded: diagonal rates are tail "
        "0.395, wing 0.977, beak 0.780, foot 0.965, and eye 0.900. Tail value 7 "
        "has only 35 swaps, a median final margin of -4.796, and event rate 0.800; "
        "beak value 2 is the next conspicuous difficult value at rate 0.363.",
        "Different parts have different numbers and frequencies of variants, so raw "
        "diagonal rates are not directly interchangeable.",
        "Relate each donor value to species support and its part's alternative count.",
        "ACCEPTED FOR exact-value difficulty as an additional graded contributor across "
        "all five parts, not as a tail-only explanation.",
        "Do rarity or a larger choice set organize those value-level failures?",
    ),
    "fb-r7b": (
        "All three outcome panels use every swap for each donor value and close to "
        "one. Within tail's nine values, support has a strong descriptive rank "
        "association with donor wins (Spearman about +0.95) and an inverse association "
        "with controlled failures (about -0.78). Tail value 7 has support from two "
        "species and controlled-event rate 0.800. The cross-part comparison is less "
        "identifiable because tail supplies nearly all low-support values, while values "
        "with overlapping support still differ sharply across parts.",
        "The number of alternatives is constant within a part and therefore remains "
        "confounded with all other part-level differences.",
        "Use more independent part families or a design that changes choice-set size "
        "while holding pixels and species fixed.",
        "VALID TEST WITH SUPPORT FOR A GRADED WITHIN-TAIL RARITY CONTRIBUTION, BUT NOT "
        "FOR SUPPORT OR ALTERNATIVE COUNT AS A SUFFICIENT CROSS-PART EXPLANATION.",
        "Does unchanged source species organize what remains after exact values?",
    ),
    "fb-r8": (
        "After centering each part/source-value/donor-value combination, source-"
        "species mean residuals remain nonzero in every part. Their standard "
        "deviations are tail 2.043, beak 1.733, eye 1.466, foot 1.342, and wing "
        "1.341 raw-logit units; tail ranges from -4.173 to 8.259.",
        "The descriptive species means can also absorb pose or repeated-row composition, "
        "and they are not a causal body manipulation.",
        "Check whether species is recoverable from held-out concept vectors and whether "
        "species improves held-out margin prediction.",
        "ACCEPTED FOR observational source-species variation beyond exact values. The "
        "common heatmap preserves identity across parts, but neither repeated nor "
        "part-specific color patterns establish a causal species effect.",
        "Is species information actually present in the learned concept representation?",
    ),
    "fb-r8b": (
        "On the same held-out split, the complete five-part recipe identifies species at 1.000 from official labels "
        "and 0.993 from all 26 raw logits. This does not mean one part identifies every species: individual raw-z blocks greatly "
        "exceed their label controls: beak 0.407 versus 0.080, eye 0.233 versus "
        "0.060, foot 0.347 versus 0.080, tail 0.953 versus 0.180, and wing "
        "0.700 versus 0.120. Even after the training-fold mean for each 0/1 label bucket is "
        "removed, held-out accuracy remains 0.260 for beak, 0.127 for eye, 0.180 for foot, "
        "0.727 for tail, 0.333 for wing, and 0.947 for all 26 scores.",
        "Species decodability is not grounding: a score block can identify species while "
        "still responding correctly to its named pixels, as the controlled wing swaps show.",
        "Judge grounding from response_delta and the final donor-minus-source margin, then "
        "relate those outcomes to visibility, conflict, and exact-value recognition.",
        "ACCEPTED FOR species-information availability beyond the official 0/1 label buckets. "
        "This is not a grounding test and does not show that leakage alone causes backwash.",
        "After a controlled replacement, do the replaced-part scores retain the unchanged source species?",
    ),
    "fb-r9": (
        "With all swaps from one original image kept in one fold (250 original "
        "images), held-out RMSE improves from 3.333 to 3.098 when visibility is "
        "added. It then worsens to 3.472 with exact values and 3.801 with source "
        "species; MAE follows the same pattern.",
        "There are many exact-value and species combinations but only 250 original "
        "images, so some training-fold groups are small. The simple group-average "
        "predictor may therefore be a poor model, but that possibility cannot be "
        "counted as positive evidence.",
        "Repeat with independent seeds or predeclare a different predictor before "
        "assigning generalizing explanatory credit to exact values or species.",
        "VALID TEST, NO SUPPORT from this predictor that exact values or source species "
        "account for held-out margin variance; only visibility gives a small improvement.",
        "Is the final concept margin associated with downstream donor-species probability?",
    ),
    "fb-r10": (
        "Mean donor-species probability rises monotonically from approximately "
        "zero in negative-margin bins to 0.110 in the most donor-positive bin, "
        "with 499-501 rows per bin.",
        "A one-part replacement need not make the whole donor species plausible because "
        "the unchanged body and other parts still belong to the source.",
        "Replicate across seeds and compare class-logit changes, not only final probability.",
        "ACCEPTED FOR a monotone but modest single-swap downstream association; "
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
        "The grey and colored bars use the same held-out species classifier and split. "
        "Grey uses the known binary concept labels c; colored uses the learned raw scores z.",
        "High species accuracy from a block does not show whether that block uses its named "
        "pixels. It only shows that species can be recovered from the supplied numbers.",
        "Compare this availability diagnostic with visibility_effect, context_gap, and the "
        "held-out row-level species contribution; CUB has no controlled grounding outcome.",
        "ACCEPTED FOR paired label-versus-raw-z species decodability; not a causal grounding "
        "test and not a CUB donor/source margin.",
        "Does natural visibility change the raw score of a positive-labelled concept?",
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


PLAIN_RESULTS = {
    "fb-r1": (
        "The model is not broken or stuck. Every concept score changes across "
        "images, and the model almost always agrees with the ordinary labels. "
        "That makes the later replacement test meaningful, but it still does "
        "not tell us which pixels the model used."
    ),
    "fb-r2": (
        "The pictures and full-file checks agree that the operation targets the "
        "named part rather than silently replacing the whole bird or scene. Most "
        "cached replacements visibly change RGB pixels; the small unchanged group "
        "is kept and measured later instead of being treated as a visible intervention."
    ),
    "fb-r3": (
        "The model nearly always notices the new part: its answer moves toward "
        "the inserted value in at least 91.9% of swaps for every part. The next "
        "question is whether that movement is large enough to change the final answer."
    ),
    "fb-r4": (
        "Yes, backwash occurs. Tail is the clearest case: in about half the tail "
        "replacements, the model reacts in the correct direction but still favors "
        "the tail value belonging to the original bird. The same event also occurs "
        "less often for beak, eye, foot, and wing."
    ),
    "fb-r5": (
        "The ordering is not created by averaging an easy direction with a hard "
        "direction. Replacing A with B and replacing B with A give similar part "
        "rankings, especially for tail, beak, and eye."
    ),
    "fb-r6": (
        "Making the inserted part clearly visible helps, especially for tail, "
        "beak, and eye. It does not solve the problem: even among clearly visible "
        "tail replacements, roughly 37 of every 100 still react toward the donor "
        "but finish with the old source answer higher. The visibility bins contain "
        "different exact values and species, so small non-monotonic steps between "
        "neighboring bins are not evidence that extra pixels themselves hurt."
    ),
    "fb-r6b": (
        "The ordinary training labels often say a tail value is present when the "
        "tail pixels are not visible. This happens for about one tail label in "
        "five but is almost absent for wing and foot. Such supervision could teach "
        "the model to infer tail from the rest of the bird; notebook 02rl tests "
        "that causal proposal by changing the labels and retraining."
    ),
    "fb-r7": (
        "After a tail is inserted, the model names the inserted tail value as its "
        "top tail answer only 39.5% of the time. It is much better for wing, foot, "
        "and eye. This establishes model-level difficulty choosing among tail's "
        "nine exact alternatives; it does not by itself establish that tails are "
        "visually ambiguous to a person. Visibility, label conflict, the number "
        "of alternatives, and source context remain separate candidate contributors."
    ),
    "fb-r7b": (
        "The three clouds distinguish three possible rarity stories. A rare value "
        "can win less often, be noticed but fail to overcome the source, or fail "
        "to move donorward at all. Upper-left points support a rarity concern; "
        "upper-right and lower-left counterexamples show that support is not a "
        "complete rule. Tail remains worse than wing and foot at overlapping "
        "support, so another part-specific contributor is required."
    ),
    "fb-r8": (
        "Even for the same source and donor values, some source species shift the "
        "margin upward and others downward. This says the unchanged bird background "
        "is associated with the answer, but this particular plot reuses the same "
        "rows to estimate and summarize the shift, so it is descriptive."
    ),
        "fb-r8b": (
        "The complete official five-part recipe already identifies the synthetic "
        "species, but one shared part alone does not. Within each one-part test, "
        "the model's score magnitudes reveal more species identity than that part's "
        "official answers. For example, tail scores identify species 95.3% of the "
        "time although the nine tail answers alone reach only 18.0%. This extra "
        "information is present, but presence alone does not prove it caused a "
        "replacement failure. Wing is the decisive control: its raw scores decode "
        "species well, yet its exact donor recognition and donorward movement are "
        "strong enough that backwash is rare. The emerging theory is competition "
        "between retained source-associated structure and local donor evidence."
    ),
    "fb-r9": (
        "Visibility is the only added block that predicts unseen original images "
        "better. Adding exact values or source species makes predictions worse, "
        "which means this declared test gives them no generalizing explanatory "
        "credit even though earlier descriptive plots show associations."
    ),
    "fb-r10": (
        "When the donor concept finishes farther ahead, the model becomes more "
        "willing to predict the donor species. The probability still reaches only "
        "11% in the strongest bin because the other four parts and the body still "
        "belong to the source bird."
    ),
}


PLAIN_CAPTIONS = {
    "fb-r1": "All 26 concept outputs vary and classify their ordinary labels well, so the swap analysis is not being driven by a collapsed or unusable model.",
    "fb-r2": "The renderer targets the named part while preserving the rest of the bird and scene; 98.3% of cached replacements visibly change RGB pixels, while the remainder are retained for visibility analysis.",
    "fb-r3": "Inserted part pixels move the donor-versus-source comparison toward the donor for nearly every swap, although this does not yet say which concept wins.",
    "fb-r4": "Controlled backwash is graded: the model responds to the new pixels but the old source answer still wins most often for tail and less often for every other part.",
    "fb-r5": "The part ordering appears in both replacement directions rather than being created by pooling one easy and one difficult direction.",
    "fb-r6": "Greater target-part visibility usually helps, but clearly visible replacements still leave controlled backwash events, especially for tail.",
    "fb-r6b": "Original supervision frequently marks a tail concept present when its renderer pixels are invisible, while this conflict is rare for wing and foot.",
    "fb-r7": "Exact inserted-value recognition is weakest for tail and graded across the other parts, showing that value-level visual difficulty accompanies the swap failures.",
    "fb-r7b": "Within tail, values carried by more species usually win more often; across parts, rarity is still bundled with part identity and does not determine the outcome by itself.",
    "fb-r8": "After matching the exact source and donor values, source species still organize final margins descriptively, but the plot does not isolate a causal species effect.",
    "fb-r8b": "The complete five-part recipe identifies species, while one part alone leaves several possible species. Raw concept-score magnitudes distinguish some species inside those shared-part groups; this is information availability, not yet use or grounding failure.",
    "fb-r9": "Visibility improves prediction for unseen source images, while this particular categorical lookup gives exact values and source species no held-out explanatory credit.",
    "fb-r10": "A more donor-positive concept margin is associated with a modestly larger saved donor-species probability; this binned analysis does not intervene on the margin.",
}


REFERENCE_TERMS = {
    "fb-r1": "`spread` is the 95th minus 5th percentile of raw `z`; `label separation` is the positive-label median minus the negative-label median; balanced accuracy weights positive and negative recognition equally; positive recall uses positive-labelled images as its denominator.",
    "fb-r2": "Rows are named parts; columns are original, replacement, deletion, and part-map roles. The colors are image pixels or renderer masks, not model scores.",
    "fb-r3": "`response_delta=m_cf-m_orig` is measured in raw-logit units. Above zero means donorward movement. The box shows the middle 50%, whiskers show the 5th--95th percentiles, and each rate uses all 1,000 swaps for that part.",
    "fb-r4": "`m_cf=z_donor,cf-z_source,cf`. A negative final margin means the old source remains higher. The controlled event requires both `response_delta>0` and `m_cf<0` on the same row.",
    "fb-r5": "Forward and backward name the two replacement directions. Every displayed rate is controlled-event rows divided by the 500 swaps in that direction; markers are not connected because direction is categorical.",
    "fb-r6": "The x-axis is visible target-mask pixels after replacement. The left outcome is median final margin; the right outcome is the controlled-event fraction. Every table row prints its own denominator.",
    "fb-r6b": "The numerator is original positive labels changed to zero by the visibility rule; the denominator is all original positive labels for that concept or part. This is a data-conflict rate, not a model probability.",
    "fb-r7": "A heatmap row is the inserted exact value and a column is the highest-scoring value. Row-normalized color is the fraction of that inserted-value population. The lower boxes show final margins and print swap counts.",
    "fb-r7b": "Species support is the number of the 50 species naturally carrying an exact value. The three y-axes are mutually exclusive fractions with all swaps for that donor value as denominator; together they sum to one.",
    "fb-r8": "An exact pair is `(part, source value, donor value)`. A row residual is its final margin minus that pair's pooled mean. Species residual is the average of those row residuals for one source species; within-pair residuals average to zero, but within-species residuals need not.",
    "fb-r8b": "Each row says whether the probe receives all five parts together or only one named part. Grey uses official yes/no answers; solid color uses model raw scores; outline uses the raw-score remainder after subtracting the average for the same yes/no answer. The x-axis is the percentage of 150 held-out images whose species is identified correctly.",
    "fb-r9": "A post-hoc prediction rule maps known row fields to a predicted final margin. Five folds keep every swap from one original image together. Panel A reports RMSE; lower is better, and small fold points are diagnostics rather than seed uncertainty. Panel B reports exact training-group coverage for held-out rows; lower coverage means the richer lookup is increasingly sparse. Nothing is added to the image or CBM.",
    "fb-r10": "The x-axis is mean final concept margin inside one of ten disjoint bins. The y-axis is the saved model's mean donor-species probability. This is the only main figure using class probability rather than raw concept `z`.",
}


def review(tag: str, figure: str) -> dict:
    literal, alternative, test, conclusion, next_question = REVIEWS[tag]
    if not tag.startswith("fb-"):
        return md(tag, f"""
        ### Review record for {figure}

        - **Literal observation:** {literal}
        - **Strongest alternative explanation:** {alternative}
        - **Discriminating test:** {test}
        - **Verdict:** **KEEP**.
        - **Limited conclusion:** `{conclusion}`
        - **Next question:** {next_question}
        """)
    plain = PLAIN_RESULTS.get(tag)
    caption = PLAIN_CAPTIONS.get(tag, plain or "")
    terms = REFERENCE_TERMS.get(tag, "See the definitions immediately before the figure.")
    plain_line = f"**Interpretation.** {plain}\n\n" if plain else ""
    return md(tag, f"""
    ### Plain-language reference for {figure}

    **Plain caption.** {caption}

    **Terms and how to read it.** {terms}

    **Literal values.** {literal}

    {plain_line}**Strongest alternative explanation.** {alternative}

    **Discriminating test.** {test}

    **Verdict.** **KEEP**.

    **Proof ledger.** `{conclusion}`

    **Next question.** {next_question}
    """)


def draft_review(tag: str, figure: str) -> dict:
    """Create a cold-review slot without importing an earlier model's result."""
    _, alternative, test, _, next_question = REVIEWS[tag]
    return md(tag, f"""
    ### First-pass review slot for {figure}

    This slot is intentionally **INCOMPLETE** until the figure above has executed
    from the accepted Koh Joint manifest and has been displayed in chat.

    - **Literal observation:** Fill from the rendered axes and printed denominators;
      do not copy values from the superseded `minimal_cbm` report.
    - **Strongest alternative to test:** {alternative}
    - **Discriminating test:** {test}
    - **Verdict:** choose `KEEP`, `REVISE`, `REMOVE`, or `MISSING EVIDENCE` only
      after visual inspection.
    - **Limited conclusion:** state only what this output directly supports.
    - **Next question if this step is interpretable:** {next_question}
    """)


def notebook(cells: list[dict], old_path: Path, preserve_outputs: bool = False) -> dict:
    metadata = {}
    old = None
    if old_path.exists():
        old = json.loads(old_path.read_text(encoding="utf-8"))
        metadata = old.get("metadata", {})
    if preserve_outputs and old is not None:
        old_code = {c.get("id"): c for c in old.get("cells", [])
                    if c.get("cell_type") == "code" and c.get("id")}
        preserved = 0
        for cell in cells:
            previous = old_code.get(cell.get("id"))
            if cell.get("cell_type") != "code" or previous is None:
                continue
            cell["outputs"] = previous.get("outputs", [])
            cell["execution_count"] = previous.get("execution_count")
            preserved += 1
        print(f"preserved outputs for {preserved} matching code cells in {old_path.name}")
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


FB_KOH_MODEL = r"""
## The implemented standard CBM and the notation used below

This report uses the accepted **ResNet-50 Koh-architecture Joint CBM**, not the
CBM class from `minimal_cbm` and not an MCBM. For image `i`, the ResNet encoder
emits one raw logit for each of the 26 exact FunnyBird concepts. The single
linear species head reads those same 26 raw logits:

CBM means **concept bottleneck model**: instead of predicting species directly
from unspecified image features, it first produces named concept scores and
then predicts species from that bottleneck of scores. “Koh architecture” names
the published CBM design whose concept and class path is preserved here.

In ordinary language, ResNet-50 is the image-processing network. “Joint” means
the image-to-concept part and concept-to-species part are trained together
rather than in separate stages. A “linear species head” is one weighted sum per
species; it receives only the 26 concept scores, with no hidden nonlinear layer.

```text
image x_i
   |
   v
ResNet-50 image encoder
   |
   v
26 raw concept logits z_i = (z_i1, ..., z_i26)
   |                         |
   |                         +--> sigmoid only for thresholded concept metrics
   |
   +--> one linear 26-to-50 species head --> species logits
```

Training minimizes Koh Joint's normalized task-plus-concept loss:

`L = L_task + 0.01 * L_concept`.

`L_task` penalizes wrong species answers. `L_concept` penalizes disagreement
with the 26 supplied concept labels. The factor 0.01 controls their numerical
weight during training; it is not a statement that concepts matter only 1% to
the final prediction.

The class head receives raw `z`; it does not receive probabilities or hard 0/1
concept decisions. There is no learned `1 -> 3 -> 1` concept decoder in this
model. The image encoder is the professor-approved ResNet-50 substitution for
Koh's Inception-v3 encoder. The accepted training description is
`ResNet-50 Koh-architecture Joint CBM, accelerated_v1`, followed by the matched
low-learning-rate convergence continuation recorded in the manifest.
`accelerated_v1` names the declared optimizer, batch, precision, and
learning-rate schedule used to finish training more quickly. It does not replace
the Koh Joint concept bottleneck with an MCBM.

| Symbol | Plain meaning | Use below |
|---|---|---|
| `x_i` | image `i` | model input |
| `y_i` | species label | species-task health |
| `c_ij` | processed 0/1 label for exact concept `j` | concept supervision and health |
| `z_ij` | raw logit emitted for concept `j` | primary grounding quantity |
| `p_ij = sigmoid(z_ij)` | bounded probability | thresholded performance only |
| `c_hat_ij = 1[z_ij>0]` | predicted present/absent concept | recall and balanced accuracy |
| `v_ig` | whether renderer mask `g` is visible | visibility analysis |
| `a_ig` | visible mask area | visibility-strength analysis |

Example: `z_blue_tail=+4` means the model favors “blue tail”; `z_blue_tail=-4`
means it disfavors it. The size of a raw-logit difference is measured in logit
units and is not a probability-point difference.

Ordinary accuracy and recall answer whether the model agrees with labels on
ordinary images. They do **not** establish which pixels produced `z`.
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
| held-out model-health population | 500 test images | used for Figure 1 and the species decoder |
| controlled swap population | accepted fixed-render seed-1 CSV | the same validated rendered images are reused across model comparisons |

Species determine part values in FunnyBird, so species context can predict a
concept label even when the named part is hard to see. That makes contextual
prediction possible, but it does not prove the trained CBM used context. The
controlled replacement in Figures 2–4 supplies that stronger test.
"""


FB_BEGINNER_GUIDE = r"""
## A new reader's guide: one complete replacement in ordinary language

Suppose the original bird has a **red tail** and we replace only that tail with
a **blue tail** taken from another species.

- The bird receiving the replacement is the **source** bird.
- The species that supplied the blue tail is the **donor**.
- “Red tail” and “blue tail” are two **exact concepts**: specific possible
  values of the broader part “tail.”
- The unchanged picture is the **original**. The otherwise identical picture
  containing the blue tail is the **replacement** or **counterfactual**.
- A **mask** is an image marking which pixels belong to one part. If the blue
  tail mask contains 150 pixels, its visible size is 150 pixels.

The model gives every exact concept an unbounded numerical score called a
**raw logit**, written `z`. Larger means “the model favors this answer more”;
smaller means “it favors it less.” A raw logit is not a percentage. For example,
`z_blue=+4` and `z_red=+1` means blue is favored over red by three logit
units. Applying `sigmoid(z)` produces a probability-like number only when a
thresholded yes/no performance question requires it.

The **margin** compares the two relevant answers:

`margin = blue-tail score - red-tail score`.

- margin `+3`: blue finishes three units above red, so the inserted answer wins;
- margin `-3`: red remains three units above blue, so the old answer wins.

The **response change** (`response_delta`) asks how much that margin moved
toward blue after the pixels changed. Worked example:

1. Before replacement, blue scores `-7` and red scores `+3`, so the starting
   margin is `-7 - 3 = -10`.
2. After replacement, blue scores `+1` and red scores `+2`, so the final
   margin is `+1 - 2 = -1`.
3. The margin moved from `-10` to `-1`, so
   `response_delta = -1 - (-10) = +9`.

The model plainly reacted to the blue pixels because the comparison moved nine
units toward blue, but it still answered red more strongly because the final
margin is negative. That combination—positive response change and negative
final margin—is the report's controlled **backwash event**.

### Other terms used later

| Term | Ordinary meaning | Small example |
|---|---|---|
| rate or fraction | count satisfying a rule divided by all eligible rows | 20 events among 100 swaps gives 0.20 or 20% |
| median | middle value after sorting | the median of 1, 3, 9 is 3 |
| percentile | a location in a sorted distribution | Q95 is greater than or equal to 95% of observed values |
| balanced accuracy | average of success on positive and negative labels | 90% positive recall and 70% negative recall gives 80% |
| visibility bin | replacements grouped by target-part pixel count | 100–199 means the inserted part contains from 100 through 199 pixels |
| label/mask conflict | label says the concept is present while its renderer mask says its pixels are not visible | “red tail=1” but zero red-tail-region pixels |
| exact-value recognition | whether the inserted value receives the largest score among alternatives for that part | blue is highest among nine tail values |
| species support | how many of the 50 species naturally carry an exact value in an unmodified bird; it is not the number of images or swaps | support 4 for blue tail means four species normally have blue tails, even if the experiment renders many blue-tail swaps |
| species decoder | a separate diagnostic classifier trained after the CBM; it asks whether species can be guessed from concept numbers | 70% means 70 of 100 held-out species labels are guessed correctly |
| held-out | rows not used to fit the diagnostic rule being evaluated | fit on four folds and score on the fifth |
| fold | one non-overlapping held-out subset | five-fold testing uses each of five subsets once as the test set |
| RMSE | typical prediction error, with large mistakes penalized more | lower RMSE is better; 3.1 is better than 3.8 |
| residual | what remains after subtracting the comparison group's expected value | observed margin 5 minus expected margin 3 leaves residual +2 |
| association | two measurements vary together; the cause is not isolated | larger visible tails tend to have better margins |
| causal evidence | changing one thing while holding the relevant alternatives fixed changes the outcome | the renderer replaces one part in the same scene |
| grounding | the named concept score actually follows the pixels of that named part | blue-tail score follows replacement blue-tail pixels |
| model health | basic check that an output changes and agrees with ordinary labels | a constant score is unhealthy even if one class is common |
| collapsed output | a score that is effectively identical for every image | always returning `z=2` cannot distinguish presence from absence |
| seed 1 | one fixed random initialization/run identifier | other seeds are independent replications, not extra images in an error bar |
| RLv2 | the later matched model trained after changing positive labels to zero when their part is invisible | used for the causal label test in notebook 02rl, not for the discovery result here |

Figures 3–4 provide causal evidence about the inserted pixels because the
renderer holds the rest of the scene fixed. Later comparisons of visibility,
value frequency, or species are mostly associations: they can identify a
plausible contributor without proving that contributor alone caused the event.
"""


FB_SERIES_INTRO = r"""
## How this chapter fits the complete investigation

This report series deliberately moves from a setting with unusually complete
counterfactual information to settings where less can be known:

1. **FunnyBird Standard CBM (this chapter):** use the renderer to establish a
   precise controlled event and learn which observable warning signs accompany
   it.
2. **FunnyBird MCBM:** test whether compressing unnecessary information in the
   concept representation changes the same fixed-render outcome.
3. **FunnyBird RLv2 and relabelled MCBM:** test whether positive labels attached
   to invisible parts are a causal contributor and whether relabelling and
   minimality address different routes.
4. **CUB70:** carry only the FunnyBird-calibrated questions into natural bird
   photographs. Released masks permit visibility and context comparisons, but
   there is no same-image donor-part replacement.
5. **Full CUB:** test whether the CUB70 observations survive 200 species and
   weaker matched support, and report which mechanisms are no longer
   identifiable from the available data.

FunnyBird is therefore a **calibration laboratory**, not an estimate of how
often backwash occurs in ordinary photographs. CUB70 is the natural-image
bridge, and Full CUB is the robustness and identifiability test. As information
decreases, the claims narrow: FunnyBird can establish the controlled event;
CUB can only test explicitly labelled observational signatures.

### Correction hypotheses carried into later chapters

| FunnyBird warning sign | Proposed correction | Later test |
|---|---|---|
| positive concepts remain labelled when their part is invisible | visibility-aware RLv2 labels | matched Standard/RLv2 fixed swaps |
| raw-score magnitudes contain species information beyond the named labels | MCBM minimality/compression | gamma-dependent compression and fixed-swap response |
| both routes operate | combine relabelling and minimality | relabelled MCBM comparison |
| a residual remains after both | direct spatial or swap-consistency supervision may be needed | future method hypothesis, not a result of this chapter |

No later model is allowed to replace the Standard-CBM discovery below. Each
later chapter inherits a question from this one and must state which operation
its own dataset actually permits.
"""


FB_PROOF_ROADMAP = r"""
## Investigation map: what would count as backwash?

We begin without assuming that backwash exists. A FunnyBird replacement will
count as a **backwash event** only if both of the following occur on the *same
controlled replacement*:

1. the donor part moves the raw concept comparison toward the donor
   (`response_delta > 0`); and
2. after that movement, the old source concept is still higher
   (`m_cf < 0`).

Numerical example—not a reported result: replacing a red tail with a blue tail
raises the blue-tail score relative to red by 24 units, but red still finishes
6 units above blue. The model reacted to the new tail pixels, yet its final
concept answer remained attached to the old bird. Figure 4 asks whether this
pattern actually appears in the accepted data.

**Part names are outcomes, not mechanisms.** The proposed general mechanism is
competition between the original context-driven source preference and the
response caused by the inserted part pixels. Visibility/label conflict,
exact-value difficulty, alternative frequency, and source-species organization
may change that balance for any part. FunnyBird tail is the most severe observed
example, but all five parts are measured and CUB must establish its own ordering.

The investigation stops or changes direction if an earlier gate fails. It asks:

| Step | Needed fact | Figure(s) | Why it is needed |
|---|---|---|---|
| 1 | the trained concept outputs are usable | 1 | a constant or broken output cannot support grounding analysis |
| 2 | the renderer really changed only the named part | 2 | otherwise a score change cannot be assigned to that part |
| 3 | the inserted pixels cause donorward movement | 3 | proves the model saw some evidence in the new part |
| 4 | starting preference, donor rise, and old-source decrease are separated | 3b | distinguishes starting context from response magnitude |
| 5 | the old source can still win after that movement | 4, 4b | this is the controlled backwash predicate and its complementary outcomes |
| 6 | the event is not a direction-averaging artifact | 5 | checks forward and reverse replacements separately |
| 7 | test proposed contributors | 6–8 | visibility/occlusion, conflicting labels, exact-value difficulty, support/alternatives, and source species |
| 8 | separate species-information availability, saved-head use, and swap-time consequence | 8b–8d | prevents calling decodable leakage a causal mechanism and directly tests what off-target scores do downstream |
| 9 | measure what those contributors predict and what remains | 9 | prevents claiming that a plausible story explains all rows |
| 10 | measure downstream class impact | 10 | separates explanation failure from species-classification harm |

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

The final mechanistic follow-up distinguishes three different statements that
must not be collapsed into one:

1. species can be decoded from the raw concept scores (**information is
   available**);
2. the unchanged saved CBM species head changes when within-label magnitudes are
   removed (**the trained head actually uses some of that information**);
3. after a physical swap, the off-target same-part scores contribute
   source-over-donor evidence, and erasing only those scores changes the frozen
   species prediction (**a direct downstream consequence**).

The third statement is stronger than decoding or correlation, but it remains a
downstream intervention. The architecture runs from concept scores to species;
the species head cannot feed backward and cause the upstream concept margin.
Species also remains bundled with body shape and pose.

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

### Fast reader path and evidence ladder

The shortest main path is Figures **1 -> 2 -> 3 -> 4 -> 6 -> 7 -> 8 -> 8b ->
8c -> 8d -> 9 -> 10**. Figure 8b establishes ordinary-image information
availability, Figure 8c separates equal-width availability from actual
saved-head use, and Figure 8d measures swap-time source evidence and directly
erases it before rerunning the frozen species head.
MCBM compression, RLv2 relabelling, and their combination are separate later
chapters and cannot replace the Standard-CBM discovery chain.

| Evidence level | Operation or quantity | Strongest permitted conclusion |
|---|---|---|
| controlled grounding test | replace one named part in the same rendered scene; measure `response_delta` and final `m_cf` | the inserted pixels moved the named comparison, yet the old source sometimes remained higher: controlled FunnyBird backwash |
| contributor test | compare visibility, label/mask conflict, exact-value difficulty, support, and source fingerprint | a factor organizes or predicts failures; it is not automatically an isolated cause |
| matched causal follow-up | retrain the matched RLv2 model after changing only the declared visibility-aware labels | whether that label intervention changes the same fixed-swap outcome |
| CUB/CUB70 approximation | compare natural visible/hidden raw logits, mask conflict, exact-value error, matched species effects, and residuals | whether observational signatures recur in photographs; this is not a donor/source swap and cannot by itself prove CUB backwash |
| future source constraint | add swap-consistency or explicit spatial routing only if the accepted results require it | whether forcing the score to follow the named region reduces the controlled event |

### Relation to existing CBM intervention and leakage work

The original [Concept Bottleneck Models paper](https://proceedings.mlr.press/v119/koh20a.html)
intervenes by editing a predicted concept value before the task head. That tests
whether changing the bottleneck changes the final task output; it does not test
whether image-to-concept prediction used the named pixels. Work on
[whether CBMs learn concepts from the intended input features](https://arxiv.org/abs/2105.04289)
motivates the named-pixel grounding test. Work on
[information leakage in soft CBMs](https://arxiv.org/abs/2211.03656) motivates
the label-versus-raw-score control because soft scores can contain information
beyond their nominal concepts. Spatially aware CBMs explicitly introduce local
concept maps and region editing, illustrating why a spatial constraint is a
different method from merely compressing a scalar bottleneck
([Benou et al., 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Benou_Show_and_Tell_Visually_Explainable_Deep_Neural_Nets_via_Spatially-Aware_CVPR_2025_paper.html)).

Our renderer swap therefore supplies a distinct base-case intervention: it
changes the named pixels first and observes the concept score afterward. The
leakage tests are supporting mechanism diagnostics and cannot replace that
controlled grounding test.
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


MEASUREMENT_TEXTBOOK = r"""
## Textbook guide: the measurements are related questions, not interchangeable scores

The aligned figures deliberately put anatomical groups in the same row order,
but the panels do **not** all measure the same thing. FunnyBird has controlled
part replacement; CUB has natural photographs and released masks. We therefore
match the scientific question while naming the weaker CUB approximation.

| Scientific question | FunnyBird measurement | CUB measurement | Same operation? |
|---|---|---|---|
| Are labels present without visible part evidence? | renderer-derived label/visibility conflict | positive label with mapped mask absent | related; CUB masks are noisier |
| Is the concept output usable? | raw-`z` spread, balanced accuracy, positive recall | the same health checks | yes |
| Do named pixels affect the score? | controlled `response_delta` after donor insertion | visible-minus-hidden raw-`z` difference | no; CUB compares different photographs |
| Does context remain after local evidence is limited? | donorward response occurs but old source still wins | hidden positive-minus-negative raw-`z` gap | no; only FunnyBird has a donor/source margin |
| Does species still organize the score? | source-species residual after exact source/donor values | species residual after exact concept and mask state | related and observational |
| Is the exact inserted value recognized? | controlled post-swap value confusion | no clean equivalent | unavailable in CUB |

### Model health comes before grounding

For exact concept `j`, the model predicts positive when `z_ij>0`. Balanced
accuracy gives positive and negative examples equal weight:

`balanced_accuracy = (positive recall + negative recall) / 2`.

If 70% of positive examples and 80% of negative examples are correct, balanced
accuracy is `(0.70+0.80)/2 = 0.75`; the aligned summary plots ordinary concept
error `1-0.75 = 0.25`. A large error says the output is difficult. It does not
say whether the error came from context, weak pixels, or noisy labels.

An output is **collapsed** when its raw score is effectively constant across all
images: `Q95(z)-Q05(z) <= 1e-8`. For example, returning `z=+2.1` for every image
always predicts “present.” Positive recall would misleadingly equal 1, negative
recall would equal 0, and balanced accuracy would equal 0.5. Such an output did
not learn a usable image distinction and cannot support a grounding claim.

The CUB70 model has two exactly collapsed outputs:
`has_throat_color::grey` is constant-positive and
`has_wing_pattern::multi-colored` is constant-negative. They remain visible as
negative health results and are excluded from positive grounding summaries.

### Direction of each CUB panel

| Panel type | A larger value means | Interpretation |
|---|---|---|
| **Data check: positive label / mask absent** | more positive labels lack a usable mapped mask | possible label/visibility conflict, but also possible missing annotation |
| **Health check: ordinary concept error** | worse positive/negative prediction | weak or difficult output; not automatically backwash |
| **Local evidence: visible - hidden raw `z`** | positive examples score higher when the region is visible | evidence that local pixels help; usually a good grounding sign |
| **Context evidence: hidden positive - negative raw `z`** | labels remain separated when the mapped mask is absent | context or unmeasured pixels remain informative |
| **Species context: residual spread** | species shift `z` after exact concept and mask state are centered | species-associated organization remains |

These quantities have different units and directions. They must not be added
into a synthetic “CUB backwash score.” Repeatedly unusual groups are stronger
observational candidates; only a controlled outcome can measure causal
backwash directly.
"""


def funnybird_source_retention_cells() -> list[dict]:
    """Separate species information, saved-head use, and swap-time source evidence."""
    return [
        md("fb-q8c-source", r"""
        ## 8c · From information available to information actually used

        Figure 8b showed that a newly fitted diagnostic can recover species from
        raw concept scores. That is **information availability**. It did not show
        that the CBM's own saved species head uses that information.

        This section asks two narrower questions:

        1. Does tail still contain more extra species information when every part
           is allowed exactly three raw-score coordinates?
        2. If image-specific magnitudes are removed while every official 0/1
           concept answer is preserved, how much does the unchanged saved species
           head `Wz+b` move?

        ### Sanity table 8c.1 · What “ordinary absent baseline” means

        For exact concept `j`, define

        `mu_j0 = mean(z_ij among ordinary images with c_ij=0)` and
        `mu_j1 = mean(z_ij among ordinary images with c_ij=1)`.

        The table prints `N`, mean, and standard deviation for both populations
        for all 26 coordinates. Raw `z` is not a 0/1 value. For example, two images
        can both have official label `tail_4=1` while receiving raw scores `+2`
        and `+8`. The label records the answer; the magnitude records how strongly
        the model produced it.

        ### Figure 8c · Equal-width species information and unchanged-head sensitivity

        **Panel A: equal-width information test.** For one three-coordinate set
        `S`, fit a labels-only species diagnostic and a labels-plus-residuals
        diagnostic in five folds. The residual is

        `r_ij = z_ij - mean_training(z_j | c_ij)`.

        The plotted gain is

        `G_gS = held-out log loss(labels only) - held-out log loss(labels + r)`.

        Positive `G_gS` means the three magnitudes contain species information
        beyond their three yes/no answers. Tail has 84 possible groups of three;
        40 are selected with a fixed seed. Wing uses all 20, beak and foot all 4,
        and eye its only group. The bar is the subset mean. The vertical line is
        the subset minimum-to-maximum range, not uncertainty across seeds.

        **Panel B: the saved-head test.** In each held-out fold, replace a score by
        the training-fold mean for its own official label:

        `z_tilde_ij = mean_training(z_j | c_ij)`.

        Pass the altered vector through the unchanged saved head
        `class_logit_ik = b_k + sum_j W_kj*z_ij`. No diagnostic classifier and no
        CBM is fitted here. The y-axis is

        `0.5 * sum_k |p_ik - p_tilde_ik|`,

        averaged over 500 held-out images and shown as a percentage. It is the
        amount of 50-species probability redistributed, not accuracy and not a
        backwash rate.

        **Concrete example.** If `tail_4=1`, its raw score is `+8`, and the other
        positive training images average `+5`, the intervention changes only that
        score from `+8` to `+5`. If probabilities change from `[0.70,0.20,0.10]`
        to `[0.65,0.25,0.10]`, the redistributed mass is
        `0.5*(0.05+0.05+0)=0.05`, or 5%.

        The important comparison is tail versus wing: both expose species
        information to a new diagnostic, but the saved head may depend on their
        magnitudes very differently.
        """),
        code("fb-f8c-source", r"""
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import log_loss
        import itertools
        import torch

        SCORE_REFERENCE=[]
        for j,name in enumerate(CONCEPT_NAMES):
            c=c_saved[:,j].astype(int); z=z_saved[:,j]
            absent=z[c==0]; present=z[c==1]
            SCORE_REFERENCE.append({
                "concept":name,"part":CONCEPT_PART[name],
                "N_absent":len(absent),"absent_mean":absent.mean(),
                "absent_SD":absent.std(ddof=1),
                "N_present":len(present),"present_mean":present.mean(),
                "present_SD":present.std(ddof=1)})
        SCORE_REFERENCE=pd.DataFrame(SCORE_REFERENCE)
        display(SCORE_REFERENCE.round(3))

        fivefold=StratifiedKFold(n_splits=5,shuffle=True,random_state=20260903)
        def residual_from_training(z_train,c_train,z_test,c_test):
            residual=np.empty_like(z_test,dtype=float)
            for column in range(z_train.shape[1]):
                for label in [0,1]:
                    reference=z_train[c_train[:,column]==label,column]
                    if not len(reference):
                        raise RuntimeError(f"no reference rows for column {column}, label {label}")
                    selected=c_test[:,column]==label
                    residual[selected,column]=z_test[selected,column]-reference.mean()
            return residual
        def conditional_logloss_gain(columns):
            label_probability=np.zeros((len(y_saved),50))
            combined_probability=np.zeros((len(y_saved),50))
            for train_index,test_index in fivefold.split(z_saved,y_saved):
                ztr=z_saved[train_index][:,columns]; zte=z_saved[test_index][:,columns]
                ctr=c_saved[train_index][:,columns].astype(int)
                cte=c_saved[test_index][:,columns].astype(int)
                rtr=residual_from_training(ztr,ctr,ztr,ctr)
                rte=residual_from_training(ztr,ctr,zte,cte)
                label_probe=make_pipeline(StandardScaler(),LogisticRegression(
                    max_iter=5000,C=1.0,random_state=20260903))
                combined_probe=make_pipeline(StandardScaler(),LogisticRegression(
                    max_iter=5000,C=1.0,random_state=20260903))
                label_probe.fit(ctr,y_saved[train_index])
                combined_probe.fit(np.column_stack([ctr,rtr]),y_saved[train_index])
                label_probability[test_index]=label_probe.predict_proba(cte)
                combined_probability[test_index]=combined_probe.predict_proba(
                    np.column_stack([cte,rte]))
            return float(log_loss(y_saved,label_probability)-
                         log_loss(y_saved,combined_probability))

        subset_rows=[]
        for part,(lo,hi) in SPANS.items():
            combinations=list(itertools.combinations(range(lo,hi),3))
            if len(combinations)>40:
                rng=np.random.default_rng(20260903)
                chosen=rng.choice(len(combinations),size=40,replace=False)
                combinations=[combinations[index] for index in sorted(chosen)]
            gains=np.array([conditional_logloss_gain(np.asarray(combo))
                            for combo in combinations])
            subset_rows.append({"part":part,"coordinates_used":3,
                                "subsets_evaluated":len(gains),
                                "mean_logloss_gain":gains.mean(),
                                "minimum_subset_gain":gains.min(),
                                "maximum_subset_gain":gains.max()})
        EQUAL_WIDTH_INFORMATION=(pd.DataFrame(subset_rows).set_index("part")
                                 .reindex(ORDER).reset_index())

        sys.path.insert(0,str(REPO/"compat"))
        sys.path.insert(0,str(REPO/"external"/"ConceptBottleneck"))
        try:
            saved_model=torch.load(MODEL,map_location="cpu",weights_only=False)
        except TypeError:
            saved_model=torch.load(MODEL,map_location="cpu")
        head=saved_model.sec_model.linear
        W=head.weight.detach().cpu().numpy(); b=head.bias.detach().cpu().numpy()
        if W.shape!=(50,26) or b.shape!=(50,):
            raise RuntimeError(f"unexpected saved class-head shapes {W.shape}, {b.shape}")
        raw_class_logits=z_saved@W.T+b
        if not np.array_equal(raw_class_logits.argmax(1),y_pred_saved):
            raise RuntimeError("reconstructed saved linear-head predictions disagree with export")
        def stable_softmax(values):
            shifted=values-values.max(axis=1,keepdims=True)
            exp=np.exp(shifted); return exp/exp.sum(axis=1,keepdims=True)
        raw_probability=stable_softmax(raw_class_logits)
        specifications=[("all 26",np.arange(26))]
        specifications.extend((part,np.arange(lo,hi)) for part,(lo,hi) in SPANS.items())
        altered_logits={name:np.full_like(raw_class_logits,np.nan) for name,_ in specifications}
        for train_index,test_index in fivefold.split(z_saved,y_saved):
            means=np.empty((26,2),dtype=float)
            for j in range(26):
                for label in [0,1]:
                    reference=z_saved[train_index][c_saved[train_index,j].astype(int)==label,j]
                    if not len(reference): raise RuntimeError(f"missing fold mean for {j}, {label}")
                    means[j,label]=reference.mean()
            expected=means[np.arange(26)[None,:],c_saved[test_index].astype(int)]
            for name,columns in specifications:
                altered=z_saved[test_index].copy(); altered[:,columns]=expected[:,columns]
                altered_logits[name][test_index]=altered@W.T+b
        raw_accuracy=float((raw_class_logits.argmax(1)==y_saved).mean())
        head_rows=[]
        for name,columns in specifications:
            logits=altered_logits[name]
            if not np.isfinite(logits).all(): raise RuntimeError(f"incomplete replacement for {name}")
            prediction=logits.argmax(1); probability=stable_softmax(logits)
            head_rows.append({"replaced_block":name,"coordinates_replaced":len(columns),
                              "raw_accuracy":raw_accuracy,
                              "accuracy_after_replacement":float((prediction==y_saved).mean()),
                              "top1_change_rate":float((prediction!=raw_class_logits.argmax(1)).mean()),
                              "mean_probability_mass_moved":float(
                                  (0.5*np.abs(probability-raw_probability).sum(axis=1)).mean())})
        HEAD_USE=pd.DataFrame(head_rows)

        fig,axes=plt.subplots(1,2,figsize=(14,5.2))
        info=EQUAL_WIDTH_INFORMATION
        axes[0].bar(info.part,info.mean_logloss_gain,color=[COLORS[p] for p in info.part])
        axes[0].vlines(info.part,info.minimum_subset_gain,info.maximum_subset_gain,
                       color="black",lw=1.2)
        axes[0].set_ylabel("held-out log-loss improvement")
        axes[0].set_title("A · Species information using exactly three scores per part")
        used=(HEAD_USE.set_index("replaced_block").loc[["all 26"]+ORDER].reset_index())
        probability_percent=100*used.mean_probability_mass_moved
        axes[1].bar(used.replaced_block,probability_percent,
                    color=["#333333"]+[COLORS[p] for p in ORDER])
        axes[1].set_ylabel("mean class-probability mass moved (%)")
        axes[1].set_title("B · Frozen species-head sensitivity when magnitudes are removed")
        axes[1].tick_params(axis="x",rotation=25)
        for index,value in enumerate(probability_percent):
            axes[1].text(index,value+.025,f"{value:.2f}%",ha="center",fontsize=8)
        fig.suptitle("Figure 8c · Information available is not the same as information used")
        plt.tight_layout(); plt.show()
        display(EQUAL_WIDTH_INFORMATION.round(3)); display(HEAD_USE.round(4))
        """, "Two panels separating equal-width species information available to a new held-out diagnostic from probability movement in the unchanged saved CBM species head after within-label magnitudes are removed."),
        figure_method("fb-m8c-source", "Panel A fitted held-out species diagnostics using exactly three concept coordinates per part. Panel B fitted nothing: it replaced raw scores by training-fold means for the same official label and passed them through the unchanged saved Koh linear species head. The preceding table prints the actual absent/present score populations used by the centering."),
        code("fb-r8c-source", r'''
        equal_text=EQUAL_WIDTH_INFORMATION.set_index("part").mean_logloss_gain.round(3).to_dict()
        use_text=(HEAD_USE.set_index("replaced_block").mean_probability_mass_moved
                  .mul(100).round(3).to_dict())
        all_row=HEAD_USE.set_index("replaced_block").loc["all 26"]
        display(Markdown(f"""
        ### Plain-language reference for Figure 8c

        **Plain caption.** Raw magnitudes expose species information beyond the
        official answers, but availability and actual use are different: tail
        remains most informative when every part gets three scores, while the
        frozen species head is much more sensitive to tail magnitudes than wing
        magnitudes.

        **Literal result.** Equal-three-coordinate log-loss gains are
        `{equal_text}`. The black lines are coordinate-subset ranges, not sampling
        uncertainty. Removing within-label magnitudes moves mean class-probability
        mass by these percentages: `{use_text}`. Across all 26 coordinates,
        accuracy changes from `{all_row.raw_accuracy:.3f}` to
        `{all_row.accuracy_after_replacement:.3f}` and the top-one prediction
        changes for `{all_row.top1_change_rate:.3f}` of 500 images.

        **Numerical interpretation.** Wing has substantial recoverable species
        information, yet removing wing magnitudes moves only
        `{use_text['wing']:.3f}%` probability mass. Tail's corresponding movement
        is `{use_text['tail']:.3f}%`. This is why species decoding alone is not a
        backwash measure: information may be present without the saved head relying
        on it to the same degree.

        **Alternative.** The equal-width test controls the number of supplied
        coordinates, not their frequency, visibility, difficulty, or correlations.
        The magnitude-removal vectors are analysis-time interventions and may not
        resemble vectors produced by another trained model.

        **Limited conclusion.** Tail's extra species information is not explained
        only by its nine-coordinate width. The saved head is numerically more
        sensitive to tail than wing magnitudes on ordinary images, but neither fact
        proves that this information causes the upstream concept-margin failure.

        **Next question.** After an actual part swap, do the other same-part scores
        create source-species evidence, and does erasing only those scores reduce
        the saved head's source preference?

        **Verdict.** **KEEP as the separation between information available and
        information used.**
        """))
        ''', "Executed plain-language review separating equal-width recoverable species information from actual sensitivity of the unchanged saved head."),

        md("fb-q8d-source", r"""
        ## 8d · After the physical swap, what source evidence do the other same-part scores create?

        This is the direct continuation from Figure 8c. It does **not** correlate a
        downstream species quantity with the upstream concept margin as its main
        test. Instead it measures the saved head's source evidence and directly
        removes that evidence at the bottleneck.

        **Concrete `tail_2 -> tail_7` example.** The CBM always outputs all nine
        fixed tail scores. Its species head always reads all nine. For this
        diagnostic only, define the seven off-target coordinates as

        `J_off = {tail_0,...,tail_8} minus {tail_2,tail_7}`.

        Position 2 always remains `tail_2`; positions never change meaning. The
        subset differs by swap only because the old and inserted values differ.

        For every off-target concept `j`, subtract its ordinary absent mean
        `mu_j0`. Then use the actual saved source-minus-donor species weights:

        `e_i = sum over j in J_off of (W_source,j - W_donor,j)*(z_cf,ij - mu_j0)`.

        If an absent `tail_5` normally scores `-4`, scores `-1` after this swap,
        and its source-minus-donor weight is `+0.5`, its contribution is
        `(+0.5)*(-1-(-4)) = +1.5` source-over-donor class-logit units. Summing all
        seven contributions gives `e_i`.

        - `e_i > 0`: off-target scores push the saved head toward source species.
        - `e_i < 0`: they push it toward donor species.
        - one `e_i` is one complete swap, not one tail value.

        **Panel A.** The distribution of `e_i` is compared across parts. All are in
        the same class-logit units. Tail sums seven off-target coordinates, wing
        four, beak and foot two, and eye one. The total is primary because it is the
        head's actual total contribution; contribution per coordinate is printed
        only as a descriptive scale check.

        **Direct bottleneck intervention for Panel B.** Keep all 26 scores except
        set only the off-target scores to their ordinary absent means:

        `z_erased,ij = mu_j0 for j in J_off`.

        The old `tail_2` and inserted `tail_7` scores remain untouched. Pass both
        vectors through the same frozen `Wz+b` head. Define source probability
        advantage as `p_source - p_donor`. Panel B plots

        `(p_source - p_donor)_before - (p_source - p_donor)_after`.

        Positive values mean erasing the off-target fingerprint reduces the
        source's probability advantage. This is a direct intervention on what the
        saved species head reads, not a claim that the species head feeds backward
        and causes the concept margin.

        The accepted CSV stores the complete score block for the part being
        replaced, but leaves the other four blocks blank on that row. Therefore
        the complete 26-score vector is obtained by replaying each unique accepted
        replacement image once through the frozen checkpoint on CUDA. Before the
        intervention, the replay is compared with every stored old/donor score.
        The accepted CSV remains authoritative for the original outcome labels;
        replayed sign categories are reported only as a numerical-sensitivity
        audit because tiny changes around exactly zero can flip a strict sign.
        This is inference on existing images, not training or a new experiment.

        ### Figure 8d · Off-target source evidence and its frozen-head consequence
        """),
        code("fb-f8d-source", r"""
        from torchvision import transforms as tv_transforms
        absent_means=np.array([z_saved[c_saved[:,j].astype(int)==0,j].mean()
                               for j in range(26)])

        replay_device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if replay_device.type!="cuda":
            raise RuntimeError(
                "Figure 8d needs CUDA to reproduce the accepted CUDA swap inference "
                "before changing the bottleneck; no training is performed")
        saved_model=saved_model.to(replay_device).eval()
        koh_image_transform=tv_transforms.Compose([
            tv_transforms.CenterCrop(299),tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.5,0.5,0.5],std=[2.0,2.0,2.0])])
        def replay_concept_logits(path_text):
            path=Path(path_text)
            if not path.is_file(): raise FileNotFoundError(f"missing accepted render {path}")
            tensor=koh_image_transform(Image.open(path).convert("RGB")).unsqueeze(0).to(replay_device)
            with torch.no_grad(): outputs=saved_model(tensor)
            if not isinstance(outputs,(list,tuple)) or len(outputs)!=27:
                raise RuntimeError("unexpected frozen Koh output contract during Figure 8d replay")
            return torch.cat([value.reshape(-1,1) for value in outputs[1:]],dim=1)[0].cpu().numpy()
        replacement_records=(S[["image_cf_sha256","image_cf_path"]]
                             .drop_duplicates("image_cf_sha256").reset_index(drop=True))
        if len(replacement_records)!=3040:
            raise RuntimeError(f"expected 3040 unique accepted replacement images, found {len(replacement_records)}")
        replacement_z={str(record.image_cf_sha256):replay_concept_logits(record.image_cf_path)
                       for record in replacement_records.itertuples(index=False)}
        z_cf_all=np.vstack([replacement_z[str(value)] for value in S.image_cf_sha256])
        if not np.isfinite(z_cf_all).all():
            raise RuntimeError("Figure 8d post-swap score matrix contains non-finite values")

        replay_source=[]; replay_donor=[]; accepted_outcome=[]; replayed_outcome=[]
        for position,row in enumerate(S.itertuples()):
            lo,hi=SPANS[str(row.part)]
            source_local=int(row.var_src); donor_local=int(row.var_donor)
            replay_source.append(z_cf_all[position,lo+source_local])
            replay_donor.append(z_cf_all[position,lo+donor_local])
            accepted_outcome.append("donor wins" if row.m_cf>0 else
                                    "donorward, source wins" if row.response_delta>0 else
                                    "no donorward move")
            replay_margin=z_cf_all[position,lo+donor_local]-z_cf_all[position,lo+source_local]
            replay_original_margin=float(row.m_orig)
            replay_response=replay_margin-replay_original_margin
            replayed_outcome.append("donor wins" if replay_margin>0 else
                                    "donorward, source wins" if replay_response>0 else
                                    "no donorward move")
        replay_source=np.asarray(replay_source); replay_donor=np.asarray(replay_donor)
        outcome_same=np.asarray(accepted_outcome)==np.asarray(replayed_outcome)
        conservative_boundary_distance=np.minimum(
            np.abs(S.m_cf.to_numpy()),np.abs(S.response_delta.to_numpy()))
        changed_boundary_distance=conservative_boundary_distance[~outcome_same]
        replay_audit={
            "device":str(replay_device),"unique_replacement_images":len(replacement_records),
            "source_coordinate_median_absolute_difference":float(np.median(np.abs(replay_source-S.z_old))),
            "source_coordinate_maximum_absolute_difference":float(np.max(np.abs(replay_source-S.z_old))),
            "donor_coordinate_median_absolute_difference":float(np.median(np.abs(replay_donor-S.z_new))),
            "donor_coordinate_maximum_absolute_difference":float(np.max(np.abs(replay_donor-S.z_new))),
            "accepted_outcome_agreement":float(outcome_same.mean()),
            "boundary_sensitive_rows":int((~outcome_same).sum()),
            "changed_rows_median_distance_to_nearest_strict_boundary":float(
                np.median(changed_boundary_distance)) if len(changed_boundary_distance) else 0.0,
            "changed_rows_maximum_distance_to_nearest_strict_boundary":float(
                np.max(changed_boundary_distance)) if len(changed_boundary_distance) else 0.0}
        print("Figure 8d matched-replay audit:",replay_audit)
        MAX_REPLAY_COORDINATE_DIFFERENCE=0.02
        if (replay_audit["source_coordinate_maximum_absolute_difference"]>
                MAX_REPLAY_COORDINATE_DIFFERENCE or
            replay_audit["donor_coordinate_maximum_absolute_difference"]>
                MAX_REPLAY_COORDINATE_DIFFERENCE):
            raise RuntimeError(
                "Figure 8d replay exceeds the explicit post-hoc 0.02 raw-logit engineering tolerance")

        before_logits=z_cf_all@W.T+b
        before_probability=stable_softmax(before_logits)
        erased_z=z_cf_all.copy(); evidence_rows=[]
        for position,row in enumerate(S.itertuples()):
            part=str(row.part); lo,hi=SPANS[part]
            source_local=int(row.var_src); donor_local=int(row.var_donor)
            off_local=np.ones(hi-lo,dtype=bool)
            off_local[[source_local,donor_local]]=False
            off_global=np.arange(lo,hi)[off_local]
            residual=z_cf_all[position,off_global]-absent_means[off_global]
            weight_difference=W[int(row.sid_src),off_global]-W[int(row.sid_donor),off_global]
            evidence=float(weight_difference@residual)
            erased_z[position,off_global]=absent_means[off_global]
            evidence_rows.append({"row_index":S.index[position],"part":part,
                                  "source_species":int(row.sid_src),
                                  "donor_species":int(row.sid_donor),
                                  "source_value":source_local,"donor_value":donor_local,
                                  "original_image":str(row.orig_render_id),
                                  "off_target_coordinates":len(off_global),
                                  "off_target_source_evidence":evidence,
                                  "m_cf":float(row.m_cf),
                                  "controlled_event":bool(row.responded_but_source_wins)})
        EVIDENCE_ROWS=pd.DataFrame(evidence_rows)
        after_logits=erased_z@W.T+b
        after_probability=stable_softmax(after_logits)
        source_index=EVIDENCE_ROWS.source_species.to_numpy(int)
        donor_index=EVIDENCE_ROWS.donor_species.to_numpy(int)
        row_index=np.arange(len(S))
        before_source_advantage=(before_probability[row_index,source_index]-
                                 before_probability[row_index,donor_index])
        after_source_advantage=(after_probability[row_index,source_index]-
                                after_probability[row_index,donor_index])
        source_logit_reduction=((before_logits[row_index,source_index]-before_logits[row_index,donor_index])-
                                (after_logits[row_index,source_index]-after_logits[row_index,donor_index]))
        if not np.allclose(source_logit_reduction,EVIDENCE_ROWS.off_target_source_evidence,
                           rtol=1e-7,atol=1e-7):
            raise RuntimeError("erasure logit change does not equal computed off-target evidence")
        EVIDENCE_ROWS["source_probability_advantage_before"]=before_source_advantage
        EVIDENCE_ROWS["source_probability_advantage_after"]=after_source_advantage
        EVIDENCE_ROWS["source_probability_advantage_reduction"]=(
            before_source_advantage-after_source_advantage)
        EVIDENCE_ROWS["top1_changed"]=(before_logits.argmax(1)!=after_logits.argmax(1))
        EVIDENCE_ROWS["source_to_donor_pair_flip"]=(
            (before_source_advantage>0)&(after_source_advantage<=0))
        EVIDENCE_SUMMARY=(EVIDENCE_ROWS.groupby("part").agg(
            n_swaps=("row_index","size"),n_original_images=("original_image","nunique"),
            off_target_coordinates=("off_target_coordinates","first"),
            mean_e=("off_target_source_evidence","mean"),
            median_e=("off_target_source_evidence","median"),
            fraction_e_positive=("off_target_source_evidence",lambda x:float((x>0).mean())),
            mean_e_per_coordinate=("off_target_source_evidence",lambda x:float(
                x.mean()/EVIDENCE_ROWS.loc[x.index,"off_target_coordinates"].iloc[0])),
            mean_source_probability_advantage_reduction=(
                "source_probability_advantage_reduction","mean"),
            top1_change_rate=("top1_changed","mean"),
            source_to_donor_pair_flip_rate=("source_to_donor_pair_flip","mean")).reindex(ORDER).reset_index())

        fig,axes=plt.subplots(1,2,figsize=(14,5.4))
        evidence_data=[EVIDENCE_ROWS.loc[EVIDENCE_ROWS.part==part,
                                        "off_target_source_evidence"].to_numpy()
                       for part in ORDER]
        boxes=axes[0].boxplot(evidence_data,labels=ORDER,showfliers=False,patch_artist=True)
        for patch,part in zip(boxes["boxes"],ORDER): patch.set_facecolor(COLORS[part])
        axes[0].axhline(0,color="black",lw=.8)
        axes[0].set_ylabel("off-target source evidence e_i (class-logit units)")
        axes[0].set_title("A · Actual saved-head source evidence after the swap")
        probability_data=[100*EVIDENCE_ROWS.loc[EVIDENCE_ROWS.part==part,
                            "source_probability_advantage_reduction"].to_numpy()
                          for part in ORDER]
        boxes=axes[1].boxplot(probability_data,labels=ORDER,showfliers=False,patch_artist=True)
        for patch,part in zip(boxes["boxes"],ORDER): patch.set_facecolor(COLORS[part])
        axes[1].axhline(0,color="black",lw=.8)
        axes[1].set_ylabel("reduction in source-minus-donor probability (percentage points)")
        axes[1].set_title("B · Erase only the off-target scores; frozen head rerun")
        fig.suptitle("Figure 8d · Does the post-swap fingerprint push the species head toward source?")
        plt.tight_layout(); plt.show(); display(EVIDENCE_SUMMARY.round(4))
        """, "Two box-plot panels comparing total off-target source-over-donor class evidence across parts and the change in frozen-head source-minus-donor probability after only those off-target scores are reset to ordinary absent baselines."),
        figure_method("fb-m8d-source", "We replayed each unique accepted replacement image once through the frozen checkpoint to recover its complete 26-score vector and compared every stored old/donor score under an explicit post-hoc 0.02-logit engineering tolerance. Strict-sign outcome differences are printed as numerical sensitivity; the accepted CSV remains authoritative. For every swap, we excluded the old and inserted coordinates, centered the remaining same-part logits at their ordinary absent means, applied the frozen source-minus-donor class weights, reset only those off-target scores, and reran the unchanged 26-to-50 head. No model or diagnostic classifier was fitted."),
        code("fb-r8d-source", r'''
        summary_text=(EVIDENCE_SUMMARY.set_index("part")[["mean_e","median_e",
            "fraction_e_positive","mean_source_probability_advantage_reduction",
            "top1_change_rate","source_to_donor_pair_flip_rate"]].round(4).to_dict("index"))
        display(Markdown(f"""
        ### Plain-language reference for Figure 8d

        **Plain caption.** Panel A measures how much the other same-part scores
        push the frozen species head toward the unchanged source after the pixels
        are replaced. Panel B directly erases only that off-target pattern and
        measures how much the source's probability advantage falls.

        **Literal result.** Every part has 1,000 swaps and all 250 original images.
        The complete part summaries are `{summary_text}`. Boxes show the middle
        50% of swap rows and their median; whiskers use the standard 1.5-IQR rule;
        individual outliers are suppressed visually but retained in every summary.
        These rows are repeated measurements from one seed and are not independent
        uncertainty estimates.

        **Replay audit.** The full-vector replay used
        `{replay_audit['unique_replacement_images']}` unique accepted replacement
        images. Maximum absolute differences from the stored source and donor
        coordinates were
        `{replay_audit['source_coordinate_maximum_absolute_difference']:.6f}` and
        `{replay_audit['donor_coordinate_maximum_absolute_difference']:.6f}` raw-
        logit units. `{replay_audit['boundary_sensitive_rows']}` of 5,000 strict-
        sign outcome labels changed under replay. Those replayed labels are not
        used to redefine the accepted outcomes; the count is reported to expose
        threshold sensitivity near zero.

        **How to read the sign.** In Panel A, positive is source-species evidence.
        In Panel B, positive means erasing that evidence reduces source-over-donor
        probability. Tail and wing must be compared using both panels: information
        availability in Figure 8c is not enough if the saved head converts little
        of it into source preference after a swap.

        **Scale caveat.** Tail sums seven off-target coordinates while eye has one.
        The total `e_i` is nevertheless the primary quantity because it is the
        actual total class-logit contribution received by the saved head. The table
        prints mean contribution per coordinate only to show how much block width
        participates; it is not a substitute causal metric.

        **Limited conclusion.** This intervention can establish downstream use:
        changing only the off-target bottleneck values changes the frozen species
        output. It cannot establish that the downstream species head caused
        `m_cf`, because the architecture flows from concept scores to species,
        not backward from species logits to concept scores.

        **Alternative.** Resetting scores to separate ordinary absent means creates
        an artificial bottleneck vector. Body, pose, visibility, and source species
        remain bundled in the original representation.

        **Next question.** Do visibility, label conflict, exact-value difficulty,
        support, and source identity jointly predict held-out swap outcomes, and
        how much remains unexplained?

        **Verdict.** **KEEP if the executed table shows a meaningful tail-versus-
        wing difference or a nontrivial frozen-head probability change; otherwise
        retain the valid negative result in the appendix.**
        """))
        ''', "Executed plain-language review of the across-part off-target evidence distribution and direct frozen-head erasure intervention."),
    ]


def funnybird_source_retention_appendix_cells() -> list[dict]:
    """Keep the weak within-part evidence/margin association as secondary context."""
    return [
        md("fb-app-evidence-correlation", r"""
        ## Appendix A · Secondary check: does more off-target source evidence accompany a worse concept margin?

        This was the earlier main follow-up after defining `e_i`. It is retained
        because it asks a valid question, but it is **not** the primary mechanism
        test. The main text now compares the level of `e_i` between parts and
        directly erases the off-target pattern before rerunning the saved species
        head.

        Here the two quantities belong to different stages of the model:

        - `e_i` is the source-minus-donor species evidence created by the
          off-target same-part scores under the saved species head;
        - `m_cf = z_donor,cf - z_source,cf` is the direct old-versus-inserted
          concept comparison before the species head.

        To prevent a common old-to-new pair from determining both numbers, center
        each quantity within the same `(part, old value, inserted value)` group:

        `e_within_i = e_i - mean(e for the same exact pair)`

        `m_within_i = m_cf_i - mean(m_cf for the same exact pair)`.

        Example: if the `tail_2 -> tail_7` swaps have mean `e=+3`, then swaps with
        `e=+1` and `e=+5` receive within-pair values `-2` and `+2`. This removes
        the average behavior of that exact replacement; it does not force each
        species or image average to zero.

        Spearman correlation asks only whether swaps with unusually high `e`
        inside their own exact-pair group also tend to have unusually low
        `m_cf`. A negative value matches that prediction. It does not measure how
        much species information a part contains, how much the head uses on
        average, or whether the species head causes concept backwash.

        ### Appendix table A1 · Within-pair correlation and evidence fifths
        """),
        code("fb-app-evidence-correlation-code", r"""
        PAIR_KEYS=["part","source_value","donor_value"]
        EVIDENCE_ROWS["e_within_pair"]=(
            EVIDENCE_ROWS.off_target_source_evidence-
            EVIDENCE_ROWS.groupby(PAIR_KEYS).off_target_source_evidence.transform("mean"))
        EVIDENCE_ROWS["margin_within_pair"]=(
            EVIDENCE_ROWS.m_cf-
            EVIDENCE_ROWS.groupby(PAIR_KEYS).m_cf.transform("mean"))
        correlation_rows=[]; fifth_rows=[]
        for part in ORDER:
            part_rows=EVIDENCE_ROWS[EVIDENCE_ROWS.part==part].copy()
            rho=float(part_rows.e_within_pair.rank().corr(
                part_rows.margin_within_pair.rank()))
            correlation_rows.append({"part":part,"n_swaps":len(part_rows),
                                     "n_original_images":part_rows.original_image.nunique(),
                                     "within_pair_spearman_rho":rho})
            part_rows["evidence_fifth"]=pd.qcut(
                part_rows.e_within_pair.rank(method="first"),5,labels=False)+1
            for fifth,group in part_rows.groupby("evidence_fifth"):
                fifth_rows.append({"part":part,"evidence_fifth":int(fifth),
                    "n_swaps":len(group),
                    "n_original_images":group.original_image.nunique(),
                    "mean_e_within_pair":group.e_within_pair.mean(),
                    "mean_margin_within_pair":group.margin_within_pair.mean(),
                    "controlled_event_rate":group.controlled_event.mean()})
        EVIDENCE_CORRELATION=pd.DataFrame(correlation_rows)
        EVIDENCE_FIFTHS=pd.DataFrame(fifth_rows)
        display(EVIDENCE_CORRELATION.round(4)); display(EVIDENCE_FIFTHS.round(4))
        """, "Two audit tables retaining the earlier within-exact-pair rank-correlation and evidence-fifths analysis as secondary exploratory evidence; no model is fitted and no causal claim is made."),
        code("fb-app-evidence-correlation-result", r'''
        rho_text=EVIDENCE_CORRELATION.set_index("part").within_pair_spearman_rho.round(3).to_dict()
        display(Markdown(f"""
        **Literal result.** The within-pair rank correlations are `{rho_text}`.
        Each part contains 1,000 swaps from 250 original images. Fifth 1 contains
        the lowest within-pair `e`; fifth 5 contains the highest. The complete
        table prints the mean centered margin and controlled-event fraction in
        every fifth.

        **Interpretation.** A weak correlation does not cancel a large difference
        between parts. Tail could operate at a much higher source-evidence level
        than wing even if small swap-to-swap fluctuations inside tail do not track
        the concept margin closely.

        **Verdict.** **KEEP IN APPENDIX as a valid but secondary association.**
        The direct off-target erasure in Figure 8d is the cleaner saved-head test.
        """))
        ''', "Plain-language boundary for the exploratory within-part association: it neither ranks average part regimes nor establishes reverse causation."),
    ]


def build_funnybird(preserve_outputs: bool = False) -> dict:
    cells: list[dict] = [
        md("fb-title", r"""
        # Chapter 1 · Standard FunnyBird CBM: controlled concept backwash

        **Result in one sentence.** Controlled concept backwash exists in this
        seed-1 Standard Koh Joint CBM, but it is graded rather than an all-parts-
        behave-the-same effect: the inserted part moves the raw comparison
        donorward, yet the old source remains higher in 50.2% of tail swaps,
        20.0% of beak, 8.9% of eye, 3.2% of foot, and 1.9% of wing swaps.

        **Required predicates and boundary.** The claim requires both
        `response_delta>0` and final margin `m_cf<0` on the same validated
        replacement. Visibility, label/mask conflict, and exact-value difficulty
        align with the graded ordering, but these measurements do not make the
        residual zero. The report separately tests whether raw magnitudes contain
        species information, whether the saved species head uses it, and whether
        swap-time off-target scores exert source-over-donor evidence downstream.
        None of those downstream tests can establish reverse causation into the
        concept scores. The report therefore concludes that backwash exists, not
        that every cause is fully or causally identified.

        **Why begin with a synthetic dataset?** FunnyBird is deliberately
        contrived. Its renderer lets us change one named part while holding the
        body, pose, camera, and background fixed. That makes it possible to
        define and verify backwash more precisely than a natural photograph
        permits. This chapter uses that privileged setting to establish the
        event, calibrate warning signs, and motivate corrections; it does not
        estimate natural-world prevalence.

        **Starting question.** When one FunnyBird part is replaced while body,
        pose, camera, and background stay fixed, what does the corresponding
        concept output do? We do not label the result “backwash” unless the
        predeclared response and final-margin conditions both hold.

        **Population.** Standard non-RL CBM, seed 1. The discovery chain contains
        no MCBM result and no visibility-aware relabelled model; MCBM appears only
        in the final handoff to notebook 03. No seed-level uncertainty is available
        yet, so reused swap rows are not presented as independent error bars.
        “Standard” here means training with the original concept labels. “Non-RL”
        means those labels were not changed according to part visibility.

        **What this design can establish.** FunnyBird's renderer permits a
        controlled donor-part replacement. A validated positive donor response
        plus a remaining source preference can establish the event. Visibility,
        label conflict, exact value, support, and species are investigated only
        after the event is measured; most remain proposed contributors unless
        independently manipulated.
        """),
        md("fb-series-intro", FB_SERIES_INTRO),
        md("fb-model", FB_KOH_MODEL),
        md("fb-beginner-guide", FB_BEGINNER_GUIDE),
        md("fb-roadmap", FB_PROOF_ROADMAP),
        md("fb-data-design", FB_DATA_DESIGN),
        code("fb-setup", r"""
        import os, json, re, glob, sys, hashlib, subprocess
        from pathlib import Path
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from IPython.display import display, Image as DisplayImage, Markdown

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

        MODEL_ROOT = CURATED/"koh_joint_resnet_accelerated_converged_v1"/"funnybirds"/"standard"/"seed1"
        SWAP_ROOT = CURATED/"swap_koh_joint_resnet_accelerated_converged_v1_seed1"
        MODEL_MANIFEST = require(MODEL_ROOT/"SUCCESS.json", "complete accepted FunnyBird Standard convergence")
        SWAP_MANIFEST = require(SWAP_ROOT/"SUCCESS.json", "complete accepted converged FunnyBird fixed swaps")
        for manifest_path in [MODEL_MANIFEST, SWAP_MANIFEST]:
            subprocess.run([sys.executable, str(REPO/"analysis"/"canonical_manifest.py"),
                            "verify", "--manifest", str(manifest_path)], check=True)
        subprocess.run([sys.executable, str(REPO/"analysis"/"validate_fixed_swaps.py"),
                        "--out", str(SWAP_ROOT)], check=True)
        model_manifest = json.loads(MODEL_MANIFEST.read_text())
        swap_manifest = json.loads(SWAP_MANIFEST.read_text())
        expected_model_meta = {"framework":"koh_joint", "backbone":"resnet50",
                               "dataset":"funnybirds", "labels":"standard", "seed":"1"}
        for key,value in expected_model_meta.items():
            if model_manifest.get("metadata",{}).get(key) != value:
                raise RuntimeError(f"model manifest {key} is not {value!r}")
        if swap_manifest.get("metadata",{}).get("framework") != "koh_joint":
            raise RuntimeError("swap manifest is not Koh Joint")
        SWAP = require(SWAP_ROOT/"funnybirds-cbm-s1.csv", "run accepted converged swaps")
        S = pd.read_csv(SWAP)
        # The Koh Joint model emits these raw concept logits directly. Legacy
        # CSV column names are retained only as a file-schema compatibility layer.
        if "response_delta" not in S:
            S["response_delta"] = S.margin - (S.z_new_orig - S.z_old_orig)
        required_swap_columns={"z_new","z_old","z_new_orig","z_old_orig","margin","response_delta"}
        missing_swap_columns=required_swap_columns-set(S.columns)
        if missing_swap_columns:
            raise RuntimeError(f"accepted swap CSV is missing {sorted(missing_swap_columns)}")
        S["m_orig"] = S.z_new_orig - S.z_old_orig
        S["m_cf"] = S.z_new - S.z_old
        S["donor_gain"] = S.z_new - S.z_new_orig
        S["source_decrease"] = S.z_old_orig - S.z_old
        if not np.allclose(S.m_cf,S.margin):
            raise RuntimeError("stored final margin disagrees with z_new-z_old")
        if not np.allclose(S.m_cf,S.m_orig+S.donor_gain+S.source_decrease):
            raise RuntimeError("starting-margin/response decomposition does not close")
        S["responded_but_source_wins"] = (S.response_delta > 0) & (S.margin < 0)
        print("fixed-render input:", SWAP)
        print("rows:", len(S), "parts:", sorted(S.part.unique()))

        PRED = require(MODEL_ROOT/"final_test.parquet", "complete accepted FunnyBird Standard evaluation")
        MODEL = require(MODEL_ROOT/"final_model_1.pth", "complete accepted FunnyBird Standard convergence")
        EVAL = pd.read_parquet(PRED)
        required_eval_columns={"image","y_true","y_pred","concept_index","concept_name","z","prob","gt_label"}
        missing_eval_columns=required_eval_columns-set(EVAL.columns)
        if missing_eval_columns:
            raise RuntimeError(f"Koh evaluation is missing {sorted(missing_eval_columns)}")
        if len(EVAL) != EVAL.image.nunique()*26:
            raise RuntimeError("Koh evaluation is not one row per image and exact concept")
        concept_order=(EVAL[["concept_index","concept_name"]].drop_duplicates()
                       .sort_values("concept_index"))
        if concept_order.concept_index.tolist() != list(range(26)):
            raise RuntimeError("Koh evaluation concept indices are not exactly 0..25")
        image_order=EVAL.image.drop_duplicates().tolist()
        z_saved=(EVAL.pivot(index="image",columns="concept_index",values="z")
                 .reindex(image_order).to_numpy())
        p_saved=(EVAL.pivot(index="image",columns="concept_index",values="prob")
                 .reindex(image_order).to_numpy())
        c_saved=(EVAL.pivot(index="image",columns="concept_index",values="gt_label")
                 .reindex(image_order).to_numpy())
        image_labels=(EVAL[["image","y_true","y_pred"]].drop_duplicates("image")
                      .set_index("image").reindex(image_order))
        y_saved=image_labels.y_true.to_numpy(dtype=int)
        y_pred_saved=image_labels.y_pred.to_numpy(dtype=int)
        if not np.allclose(1/(1+np.exp(-z_saved)),p_saved,rtol=1e-5,atol=1e-6):
            raise RuntimeError("Koh evaluation probability does not equal sigmoid(raw z)")

        FB_ROOT = Path(os.environ.get("FUNNYBIRDS_ROOT", CURATED/"FunnyBirds"))
        import funnybirds_concepts as fbc
        parts = fbc.load_parts(FB_ROOT)
        CONCEPT_NAMES = fbc.concept_names(parts)
        SPANS = fbc.group_slices(parts)
        if concept_order.concept_name.tolist() != CONCEPT_NAMES:
            raise RuntimeError("FunnyBird concept names/order disagree with the Koh evaluation")
        if len(CONCEPT_NAMES) != z_saved.shape[1]:
            raise RuntimeError("parts.json concept width does not match saved predictions")
        CONCEPT_PART = {name: part for part,(a,b) in SPANS.items() for name in CONCEPT_NAMES[a:b]}
        print("framework: Koh Joint; backbone: ResNet-50; minimal_cbm: rejected")
        print("checkpoint:", MODEL)
        print("evaluation:", PRED, "images:", len(y_saved), "concepts:", len(CONCEPT_NAMES),
              "species:", len(np.unique(y_saved)))
        """),
    ]

    cells += [
        question("fb-q1", "1", "Did training produce a usable, non-collapsed CBM?",
                 "For every exact concept `j`, measure raw-score spread, positive-versus-negative label separation, balanced accuracy, and positive recall.",
                 "A usable slot has nonzero spread, positive label separation, and above-chance thresholded performance.",
                 "Compute all quantities from the accepted converged checkpoint's held-out predictions. Recall is a health statistic, not grounding evidence."),
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
        y_true=y_saved
        task_accuracy=float((y_pred_saved==y_true).mean())
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
        figure_method("fb-m1", "We computed four health statistics directly from the frozen CBM's 26 raw concept logits on 500 held-out images; no classifier or model was fitted."),
        review("fb-r1", "Figure 1"),

        question("fb-q2", "2", "Did the renderer change only the intended part?",
                 "Inspect the semantic preflight and original/swap/delete/part-map examples for all five parts.",
                 "For a visible replacement, the target part should change while the rest of the scene is preserved. Rows whose rendered RGB image does not change must remain identifiable and be handled by the later visibility analysis, not counted as visibly changed.",
                 "Use the accepted full-cache validation plus representative all-part examples before reading any model response."),
        code("fb-f2a", r"""
        ROOT = SWAP.parent
        preflight_candidates=[ROOT/"renderer_preflight"/"renderer_semantic_preflight.png"]
        preflight=next((p for p in preflight_candidates if p.exists()),preflight_candidates[0])
        example_candidates=[ROOT/"examples",CURATED/"swap_fixed_v2_attempt2"/"examples"]
        examples=next((p for p in example_candidates if p.is_dir()),example_candidates[0])
        from PIL import Image
        if preflight.exists():
            im0=Image.open(preflight).convert("RGB")
            width,height=im0.size
            fig_width=18
            fig_height=max(8,fig_width*height/width)
            fig0,ax0=plt.subplots(figsize=(fig_width,fig_height),dpi=120)
            ax0.imshow(im0); ax0.axis("off")
            ax0.set_title("Figure 2a · Semantic preflight: original, swap, delete, original map, swap map")
            plt.tight_layout(); plt.show()
        else:
            raise FileNotFoundError("accepted converged swap root lacks the semantic preflight sheet")
        """, "FunnyBird semantic renderer preflight showing the intended one-part replacement and deletion for every part."),
        figure_method("fb-m2a", "We displayed the renderer's saved original, one-part replacement, deletion, and part-mask outputs for all five parts; this is a pixel-operation audit with no model fitting."),
        md("fb-r2a", r"""
        **Plain caption.** This preflight sheet checks that the renderer can replace
        or remove one named part without intentionally changing the remaining bird,
        pose, camera, or background.

        **How this image was obtained.** Before evaluating the CBM, the renderer
        saved, for each part, the original image, the one-part replacement, the
        deletion, the original part map, and the replacement part map. No model
        score or fitted classifier appears in this figure.

        **Literal observation.** Every named part has the required image and mask
        roles, and the highlighted map follows the requested part rather than the
        whole bird. This is a semantic operation check; Figure 2b makes the
        within-row pixel comparison easier to inspect.

        **Alternative explanation still open.** A montage cannot prove that every
        cached row changed only the intended RGB pixels. The full-file hash and
        intervention checks reported around Figure 2b address cache completeness
        and diversity, while visible target area is analyzed later.

        **Limited conclusion.** KEEP as renderer preflight evidence. It validates
        the meaning of the requested operation, not the CBM's response and not the
        entire cache by itself.

        **Next question.** Do representative stored outputs for all five parts show
        the same intended operation clearly enough to inspect row by row?
        """),
        md("fb-q2b", r"""
        ### Figure 2b · Do saved examples confirm the operation for every part?

        **Question.** Does the accepted swap output contain a visually inspectable
        original, replacement, deletion, and replacement-part map for tail, wing,
        beak, foot, and eye?

        **Variables and prediction.** Each row is one named part and each column is
        one image role. A valid visible example changes the named part and its map
        while leaving the remaining bird and scene unchanged. Across the complete
        cache, 98.3% of replacement RGB images differ from their original; the
        remaining 1.7% are retained for the later visibility analysis rather than
        described as visibly changed. “Missing” is an error, not evidence.

        **Method.** Select the first stored audit example by filename order for each
        part. This is a complete five-part semantic check, not a hand-picked model
        success/failure gallery.

        **How to read the figure.** Compare columns within a row, then compare the
        visible changed pixels with the highlighted replacement-part map. No axis or
        color encodes a model score.
        """),
        code("fb-f2b", r"""
        ROOT = SWAP.parent
        example_candidates=[ROOT/"examples"]
        examples=next((p for p in example_candidates if p.is_dir()),example_candidates[0])
        if not examples.is_dir():
            raise FileNotFoundError("accepted converged swap root lacks example images")
        tags=["orig","swap","delete","target_mask"]
        segmentation_colors={"beak":(255,255,0),"eye":(255,255,253),
                             "wing":(0,255,1),"foot":(255,0,1),"tail":(0,0,255)}
        fig,axes=plt.subplots(len(ORDER),len(tags),figsize=(12,13))
        for r,part in enumerate(ORDER):
            for c,tag in enumerate(tags):
                ax=axes[r,c]
                file_tag="swap_partmap" if tag=="target_mask" else tag
                files=sorted(examples.glob(f"{part}_*_{file_tag}.png"))
                if files and tag=="target_mask":
                    segmentation=np.asarray(Image.open(files[0]).convert("RGB"))
                    target=np.all(segmentation==np.asarray(segmentation_colors[part]),axis=2)
                    ax.imshow(target,cmap="gray",vmin=0,vmax=1)
                elif files:
                    ax.imshow(Image.open(files[0]).convert("RGB"))
                else:
                    ax.text(.5,.5,"missing",ha="center",va="center")
                ax.set_title(f"{part} · {tag}"); ax.axis("off")
        fig.suptitle("Figure 2b · Representative five-part intervention audit")
        plt.tight_layout(); plt.show()
        """, "Complete five-part FunnyBird intervention audit showing original, replacement, deletion, and replacement-part map."),
        figure_method("fb-m2b", "We selected one accepted saved audit row per part and displayed its original, replacement, deletion, and isolated target mask; no result was estimated from these examples."),
        review("fb-r2", "Figure 2b"),

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
        axes[1].set_ylim(0,1.05)
        axes[1].set_ylabel("fraction with response_delta > 0"); axes[1].set_title("B · Positive donor-response rate")
        counts=S.groupby("part").size().reindex(ORDER)
        for x,(part,value) in enumerate(rate.items()):
            axes[1].text(x,value+.02,f"n={int(counts.loc[part])}",ha="center",fontsize=8)
        fig.suptitle("Figure 3 · Does the replacement produce the predicted within-image response?")
        plt.tight_layout(rect=[0,0,1,.94]); plt.show()
        display(pd.DataFrame({"eligible_swaps":counts,"positive_response_rate":rate}).round(3))
        """, "FunnyBird response-delta distributions and positive donor-response rates for all five parts."),
        figure_method("fb-m3", "For each of 5,000 paired swaps, we subtracted the original donor-minus-source margin from the counterfactual margin, then summarized those paired raw-logit changes by part; nothing was trained."),
        review("fb-r3", "Figure 3"),

        md("fb-q3b", r"""
        ## 3b — Where did each part start, and which score changed after replacement?

        **Question.** Does a part finish poorly because its donor began far below
        the source, because the donor rose too little, because the removed source
        fell too little, or because several of these occurred together?

        **Variables and exact identity.** For every swap:

        `m_orig = z_donor,orig - z_source,orig`

        `donor_gain = z_donor,cf - z_donor,orig`

        `source_decrease = z_source,orig - z_source,cf`

        `response_delta = donor_gain + source_decrease`

        `m_cf = m_orig + response_delta`

        **Score scale.** Every quantity here uses the post-head raw logit
        `z=q(h)`, which is unbounded. This standard CBM has no MCBM gamma penalty
        and no `±3` target. Notebook 03 applies the soft `±3` target to internal
        `h`, not to the plotted `z`.

        `m_orig` is the starting preference on the unchanged original image. It
        is **not** a pure context measurement because the source part is still
        visible there. Species/body context is tested separately later.

        Example: the donor starts 20 units below the source, then rises by 9 while
        the old source falls by 6. Total donorward response is 15, so the final
        margin is `-20+9+6=-5`: the swap helped, but the source still wins.

        **Method.** Average each raw-logit quantity over all 1,000 validated swaps
        for each part, including both directions. Verify the exact row-wise
        identity before displaying any mean.

        ### Figure 3b — Starting preference, donor rise, source release, response, and final result

        **How to read the figure.** All panels use the same raw-logit y-axis and
        part colors. Panel A below zero means the future donor starts behind.
        Panels B and C above zero are the two ways replacement helps. Panel D is
        their sum. Panel E above zero means the donor finally wins. Part names
        identify observed outcomes, not mechanisms.
        """),
        code("fb-f3b", r"""
        component_columns=["m_orig","donor_gain","source_decrease","response_delta","m_cf"]
        component_means=S.groupby("part")[component_columns].mean().reindex(ORDER)
        decomposition_error=np.max(np.abs(S.m_cf-(S.m_orig+S.donor_gain+S.source_decrease)))
        if decomposition_error>1e-8: raise RuntimeError(f"decomposition error {decomposition_error}")
        titles=[("m_orig","A. Before swap: donor minus source"),
                ("donor_gain","B. Inserted donor score rises"),
                ("source_decrease","C. Removed source score falls"),
                ("response_delta","D. Total donorward movement"),
                ("m_cf","E. After swap: donor minus source")]
        lim=float(np.nanmax(np.abs(component_means.values)))*1.12
        fig,axes=plt.subplots(1,5,figsize=(21,4.4),sharey=True)
        for ax,(column,title) in zip(axes,titles):
            values=component_means[column]
            ax.bar(ORDER,values.values,color=[COLORS[p] for p in ORDER],alpha=.75)
            ax.axhline(0,color="black",lw=.9); ax.set_title(title,fontsize=10)
            ax.tick_params(axis="x",rotation=45); ax.set_ylim(-lim,lim)
        axes[0].set_ylabel("mean raw-logit units")
        fig.suptitle("Figure 3b — What creates each part's final donor-versus-source result?")
        plt.tight_layout(); plt.show(); display(component_means.round(3))
        print("maximum row-wise decomposition error:",decomposition_error)
        """, "Standard FunnyBird CBM starting margin, donor-score gain, removed-source decrease, total response, and final margin for all five parts."),
        figure_method("fb-m3b", "We arithmetically decomposed every paired swap into starting margin, donor-score rise, source-score fall, total response, and final margin, verified the identity row by row, and averaged each term by part."),
        md("fb-r3b", r"""
        ### Plain-language reference for Figure 3b

        **Plain caption.** The final donor-versus-source result combines a
        starting preference for the source with the donor's rise and the
        source's fall after replacement.

        **Terms.** Starting margin is donor score minus source score before
        replacement. Inserted donor score rises is the increase in the donor
        concept's score. Removed source score falls is the decrease in the old
        source concept's score. Total donorward movement is donor rise plus
        source decrease. Final margin is starting margin plus total movement. A
        negative final margin means the old source concept remains higher. All
        five panels use mean raw-logit units over 1,000 swaps per part.

        **Literal mean values.**

        | Part | Starting margin | Donorward movement | Final margin |
        |---|---:|---:|---:|
        | tail | `-10.756` | `+9.800` | `-0.956` |
        | wing | `-9.914` | `+16.390` | `+6.476` |
        | beak | `-9.167` | `+11.778` | `+2.611` |
        | foot | `-8.847` | `+13.831` | `+4.984` |
        | eye | `-8.481` | `+11.749` | `+3.268` |

        The row-wise arithmetic identity closes to numerical error below
        `1.8e-15`. Tail's donor rise is `4.599` and old-source decrease is
        `5.201`, both the smallest part means.

        **Interpretation.** Every part starts with a strong source preference.
        Wing, beak, foot, and eye usually produce enough movement to overcome
        it. Tail produces a large response too, but its average response is
        insufficient to erase the starting preference. Tail does not mainly
        fail because it began uniquely far behind; its total correction is
        smaller.

        **Alternative.** Tail could be smaller, hidden more often, harder to
        distinguish exactly, or more strongly associated with species context.
        Means can also hide direction and exact-value asymmetry, and the original
        margin still contains genuine source-part pixels.

        **Discriminating test.** Separate direction, visibility, exact values,
        and source-species organization in the following figures.

        **Verdict.** **KEEP**.

        **Proof ledger.** The competition producing the final margin is
        separated arithmetically. The causes of the starting preference and
        unequal replacement response remain unresolved.

        **Next question.** How often does the donor actually finish higher?
        """),

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
        figure_method("fb-m4", "We applied the predeclared row-level predicate `response_delta>0 and m_cf<0` to all 1,000 validated swaps per part and plotted the two raw quantities jointly; no threshold was learned."),
        review("fb-r4", "Figure 4"),

        md("fb-q4b", r"""
        ## 4b — How often does the donor win, help but still lose, or fail to move donorward?

        Every validated swap is placed into exactly one outcome:

        1. `m_cf > 0`: the donor concept finishes higher;
        2. `m_cf <= 0 and response_delta > 0`: the new pixels help, but the old
           source concept remains higher;
        3. `m_cf <= 0 and response_delta <= 0`: the source remains higher and the
           replacement does not move the comparison toward the donor.

        These fractions sum to one for every part. Thus Figure 4's controlled-
        backwash rate is not the donor-win rate.

        Example: if 20 of 100 swaps end donor-positive, 50 move donorward but
        remain source-negative, and 30 do not move donorward, the three displayed
        fractions are 0.20, 0.50, and 0.30. The denominator is all 100 swaps.

        ### Figure 4b — Three mutually exclusive outcomes for every part

        **How to read the figure.** Every panel contains all five parts and uses a
        fraction from zero to one. Higher is desirable only in Panel A. Panel B
        is the controlled backwash event. Panel C is a different failure: no
        positive response. Both swap directions are included. Bar colors use the
        shared part palette defined at the start of the notebook.
        """),
        code("fb-f4b", r"""
        outcomes=pd.DataFrame(index=ORDER,dtype=float)
        outcomes["donor_wins"]=(S.m_cf>0).groupby(S.part).mean().reindex(ORDER)
        outcomes["helped_but_source_wins"]=((S.m_cf<=0)&(S.response_delta>0)).groupby(S.part).mean().reindex(ORDER)
        outcomes["no_donorward_move_and_source_wins"]=((S.m_cf<=0)&(S.response_delta<=0)).groupby(S.part).mean().reindex(ORDER)
        if not np.allclose(outcomes.sum(axis=1).values,1):
            raise RuntimeError("three outcome fractions do not sum to one")
        panels=[("donor_wins","A. Donor finishes higher"),
                ("helped_but_source_wins","B. New pixels help, but source stays higher"),
                ("no_donorward_move_and_source_wins","C. No donorward movement; source stays higher")]
        fig,axes=plt.subplots(1,3,figsize=(15,5.4),sharey=True)
        for ax,(column,title) in zip(axes.flat,panels):
            values=outcomes[column]
            y=np.arange(len(ORDER))
            ax.barh(y,values.values,color=[COLORS[p] for p in ORDER],alpha=.78)
            ax.set_title(title,fontsize=10); ax.set_xlim(0,1.08)
            ax.set_yticks(y,ORDER); ax.invert_yaxis()
            for yy,value in enumerate(values): ax.text(value+.015,yy,f"{value:.3f}",va="center",fontsize=8)
            ax.set_xlabel("fraction of all swaps")
        fig.suptitle("Figure 4b — Final outcome categories for standard CBM")
        plt.tight_layout(rect=[0,0,1,.94]); plt.show(); display(outcomes.round(3))
        """, "Standard FunnyBird CBM donor-win, donorward-but-source-still-wins, and no-donorward-movement fractions for all five parts."),
        figure_method("fb-m4b", "We assigned every swap to exactly one of three predeclared outcomes from the signs of `m_cf` and `response_delta`, then divided each count by all 1,000 swaps for that part."),
        md("fb-r4b", r"""
        ### Plain-language reference for Figure 4b

        **Plain caption.** Every replacement is assigned to exactly one final
        outcome, separating successful donor wins from insufficient donorward
        corrections and from complete failures to move donorward.

        **Terms and denominator.** Donor wins means `m_cf>0`. Helped but source
        wins means `response_delta>0` and `m_cf<=0`, the controlled backwash
        event. No donorward move means both quantities are non-positive. Each
        fraction uses all 1,000 swaps for that part, and the three bars sum to one.

        **Literal values.** Donor-win/helped-but-source-wins/no-donorward-move
        fractions are tail `0.417/0.502/0.081`, wing `0.981/0.019/0.000`, beak
        `0.789/0.200/0.011`, foot `0.965/0.032/0.003`, and eye
        `0.900/0.089/0.011`.

        **Interpretation.** Tail usually notices the replacement: only 8.1% of
        tail swaps fail to move toward the donor. The larger problem is that the
        correction is insufficient—50.2% move the right way but retain the old
        answer. Most beak and eye failures have the same form.

        **Alternative.** A positive response may be tiny, and pooled outcomes
        may hide direction or exact-value asymmetry.

        **Discriminating test.** Retain Figure 3b's response magnitudes and next
        separate directions and exact donor values.

        **Verdict.** **KEEP**.

        **Proof ledger.** The controlled event is distinguished from a model
        that simply did not react to the inserted pixels.

        **Next question.** Does the pattern occur in both swap directions?
        """),

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
            for k,part in enumerate(ORDER):
                axes[0].annotate(f"n={int(d.loc[part,'n'])}",
                                 (x[k]+offset,d.loc[part,"responded_but_source_wins_rate"]),
                                 xytext=(0,7 if direction=="fwd" else -12),
                                 textcoords="offset points",ha="center",fontsize=6)
        for ax in axes: ax.set_xticks(x,ORDER)
        axes[0].set_ylim(0,1); axes[0].set_ylabel("fraction: donorward response, but source still wins")
        axes[1].axhline(0,color="black",lw=.8); axes[1].set_ylabel("median final margin")
        axes[0].legend(); axes[1].legend(); fig.suptitle("Figure 5 · Forward and backward directions")
        plt.tight_layout(); plt.show(); display(D.round(3))
        """, "Forward and backward FunnyBird rates where the donor changes the margin but the old source remains larger, alongside final margins for every part."),
        figure_method("fb-m5", "We split each part's 1,000 fixed swaps into its 500 forward and 500 backward replacements and recomputed the same controlled-event rate and final-margin summary independently; nothing was fitted."),
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
            axes[0].plot(labels,d.median_margin,"o-",label=p,color=COLORS[p],lw=1.2)
            axes[1].plot(labels,d.responded_but_source_wins_rate,"o-",label=p,color=COLORS[p],lw=1.2)
            for k,label in enumerate(labels):
                if pd.notna(d.loc[label,"n"]):
                    axes[0].annotate(f"n={int(d.loc[label,'n'])}",(k,d.loc[label,"median_margin"]),
                                     xytext=(2,5),textcoords="offset points",fontsize=5,color=COLORS[p])
        axes[0].axhline(0,color="black",lw=.8); axes[0].set_ylabel("median final margin")
        axes[1].set_ylim(0,1); axes[1].set_ylabel("fraction: donorward response, but source still wins")
        for ax in axes: ax.tick_params(axis="x",rotation=45); ax.legend(fontsize=8,ncol=2)
        fig.suptitle("Figure 6 · Same-render visibility analysis")
        VISIBLE_ONLY=(V[V.pixel_count_cf>0].groupby("part").agg(
            n_visible_rows=("margin","size"),median_margin=("margin","median"),
            responded_but_source_wins_rate=("responded_but_source_wins","mean")).reindex(ORDER))
        plt.tight_layout(); plt.show(); display(T.round(3)); display(VISIBLE_ONLY.round(3))
        """, "FunnyBird final margin and responded-but-source-still-wins rate across exact swapped-part visibility bins for all parts."),
        figure_method("fb-m6", "We grouped the same fixed swaps by the number of visible inserted-part mask pixels and recomputed median final margin and controlled-event fraction inside each declared bin; the images and CBM were unchanged."),
        review("fb-r6", "Figure 6"),

        question("fb-q6b", "6b", "How often did the original training label conflict with visible part evidence?",
                 "Compare the standard and visibility-aware label views for every image used in final training (train plus validation); count positive concept labels changed to zero within each exact concept and part group.",
                 "A large conflict count identifies a plausible training signal that can reward contextual prediction, but its causal effect belongs to notebook 02rl.",
                 "Require identical ordered image/class records in both splits and allow only `attribute_label` to differ. This cell compares data labels, not Standard and RLv2 model predictions."),
        code("fb-f6b", r"""
        import pickle
        standard_input=CURATED/"koh_joint_inputs"/"funnybirds"/"standard"
        visibility_input=CURATED/"koh_joint_inputs"/"funnybirds"/"rlv2"
        pairs=[]
        for split in ["train","val"]:
            std_path=standard_input/f"{split}.pkl"
            visibility_path=visibility_input/f"{split}.pkl"
            if not (std_path.exists() and visibility_path.exists()):
                raise RuntimeError(f"missing matched {split} label views: {std_path} or {visibility_path}")
            std=pickle.loads(std_path.read_bytes())
            visibility=pickle.loads(visibility_path.read_bytes())
            if len(std)!=len(visibility):
                raise RuntimeError(f"standard/visibility-aware {split} lengths differ")
            pairs.extend((split,a,b) for a,b in zip(std,visibility))
        positive=np.zeros(len(CONCEPT_NAMES),dtype=int); changed=np.zeros(len(CONCEPT_NAMES),dtype=int)
        split_rows=[]
        for split in ["train","val"]:
            split_positive=np.zeros(len(CONCEPT_NAMES),dtype=int)
            split_changed=np.zeros(len(CONCEPT_NAMES),dtype=int)
            for _,a,b in [row for row in pairs if row[0]==split]:
                for key in a:
                    if key=="attribute_label": continue
                    av,bv=a[key],b[key]
                    equal=np.array_equal(np.asarray(av),np.asarray(bv)) if isinstance(av,(list,tuple,np.ndarray)) else av==bv
                    if not bool(equal): raise RuntimeError(f"non-label record field differs in {split}: {key}")
                ca=np.asarray(a["attribute_label"]); cb=np.asarray(b["attribute_label"])
                split_positive += (ca==1); split_changed += ((ca==1)&(cb==0))
            positive += split_positive; changed += split_changed
            split_rows.append({"split":split,"images":sum(row[0]==split for row in pairs),
                               "positive_labels":int(split_positive.sum()),
                               "positive_to_zero_conflicts":int(split_changed.sum())})
        CONFLICT_EXACT=pd.DataFrame({"concept":CONCEPT_NAMES,"part":[CONCEPT_PART[n] for n in CONCEPT_NAMES],
            "n_positive":positive,"n_changed":changed})
        CONFLICT_EXACT["conflict_rate"]=CONFLICT_EXACT.n_changed/CONFLICT_EXACT.n_positive.replace(0,np.nan)
        PART_CONFLICT=(CONFLICT_EXACT.groupby("part").agg(n_positive=("n_positive","sum"),
            n_changed=("n_changed","sum")).reindex(ORDER))
        PART_CONFLICT["conflict_rate"]=PART_CONFLICT.n_changed/PART_CONFLICT.n_positive
        q=CONFLICT_EXACT.sort_values(["part","concept"]); y=np.arange(len(q))
        fig,ax=plt.subplots(figsize=(10,max(6,.24*len(q))))
        ax.barh(y,q.conflict_rate,color=q.part.map(COLORS)); ax.set_yticks(y,q.concept,fontsize=7)
        ax.invert_yaxis()
        conflict_axis_max=max(.05,min(1.0,float(q.conflict_rate.max())*1.15))
        ax.set_xlim(0,conflict_axis_max)
        ax.set_xlabel("fraction of positive training labels removed by visibility rule")
        ax.set_title("Figure 6b · Exact-concept label/mask conflict in train + validation")
        plt.tight_layout(); plt.show()
        print("Figure 6b denominators by split:")
        display(pd.DataFrame(split_rows))
        print("Figure 6b exact-concept and part totals:")
        display(q.round(3)); display(PART_CONFLICT.round(3))
        """, "FunnyBird training-image counts whose positive part-concept labels change under the matched visibility-aware relabeling rule."),
        figure_method("fb-m6b", "We joined the Standard and visibility-aware train-plus-validation label records image by image and counted only positive labels changed to zero by the visibility rule; this is a data audit, not a model comparison."),
        review("fb-r6b", "Figure 6b"),

        question("fb-q7", "7", "Do exact source and donor values explain the failures?",
                 "For every part, compare the inserted donor value with the concept value that has the largest post-swap raw score.",
                 "A clean diagonal means exact visual values are distinguished; recurring bright columns indicate default answers.",
                 "Display all parts and all values with row-normalized counts."),
        code("fb-f7", r"""
        available=[p for p in ORDER if any(c.startswith(f"z_cf_{p}_") for c in S.columns)]
        if set(available)!=set(ORDER): raise RuntimeError(f"missing all-part post-swap concept logits: have {available}")
        fig,axes=plt.subplots(2,5,figsize=(20,9.5),constrained_layout=True)
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
            ax.set_xticks(np.arange(len(cols))); ax.set_yticks(np.arange(len(cols)))
            ax.set_xlabel("highest-scoring value"); ax.set_ylabel("inserted value")
            groups=[]; labels=[]
            for v,g in d.groupby("var_donor"):
                groups.append(g.margin.to_numpy()); labels.append(str(int(v)))
                value_rows.append({"part":p,"donor_value":int(v),"n":len(g),"median_margin":g.margin.median(),
                    "q25_margin":g.margin.quantile(.25),"q75_margin":g.margin.quantile(.75),
                    "event_rate":g.responded_but_source_wins.mean()})
            bax.boxplot(groups,tick_labels=labels,showfliers=False,whis=(5,95)); bax.axhline(0,color="black",lw=.8)
            upper=max(float(np.nanmax(np.concatenate(groups))),float(bax.get_ylim()[1]))
            for xpos,g in enumerate(groups,start=1):
                bax.text(xpos,upper,f"n={len(g)}",ha="center",va="bottom",fontsize=6,rotation=90)
            bax.set_ylim(top=upper+max(1,.08*abs(upper)))
            bax.set_xlabel("inserted value"); bax.set_ylabel("final margin"); bax.set_title(f"{p}: value-wise margins")
        colorbar=fig.colorbar(im,ax=list(axes[0]),fraction=.015)
        colorbar.set_label("fraction within inserted-value row")
        fig.suptitle("Figure 7 · Exact-value attribution and final-margin distributions")
        plt.show(); display(pd.Series(diag,name="diagonal_rate").to_frame().round(3)); display(pd.DataFrame(value_rows).round(3))
        """, "Five row-normalized confusion matrices comparing inserted and highest-scoring FunnyBird part values."),
        figure_method("fb-m7", "For each swapped part, we took the highest post-swap raw logit within that part block, cross-tabulated it against the inserted exact value, normalized each inserted-value row, and retained every swap."),
        review("fb-r7", "Figure 7"),

        md("fb-q7a", r"""
        ### Figure 7a · What do the nine tail values actually look like?

        The confusion matrix says which values the model mixes up, but it cannot
        tell a reader whether two tail shapes look similar. This compact visual
        audit shows one large, clearly visible accepted replacement for each of
        the nine inserted tail values. Only pixels inside the renderer's tail
        mask are retained, so body shape, pose, and background do not dominate
        the comparison. The examples are selected mechanically by largest tail
        mask area, not by model success or failure.

        This is a human-readable visual check, not a new backwash measurement.
        Apparent similarity can motivate a hypothesis—for example, that values
        0 and 4 are easy to confuse—but the confusion matrix and controlled
        margins remain the quantitative evidence.
        """),
        code("fb-f7a", r"""
        from PIL import Image
        tail_examples=[]
        for value,group in S[S.part=="tail"].groupby("var_donor"):
            row=group.sort_values("pixel_count_cf",ascending=False).iloc[0]
            rgb_path=Path(row.image_cf_path)
            mask_path=rgb_path.parents[1]/"part_map"/rgb_path.name
            if not rgb_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(f"missing accepted RGB/part-map pair for tail value {int(value)}")
            rgb=np.asarray(Image.open(rgb_path).convert("RGB"))
            seg=np.asarray(Image.open(mask_path).convert("RGB"))
            mask=np.all(seg==np.asarray([0,0,255]),axis=2)
            if not mask.any(): raise RuntimeError(f"tail value {int(value)} has an empty target mask")
            yy,xx=np.where(mask); pad=8
            y0=max(0,int(yy.min())-pad); y1=min(rgb.shape[0],int(yy.max())+pad+1)
            x0=max(0,int(xx.min())-pad); x1=min(rgb.shape[1],int(xx.max())+pad+1)
            isolated=np.full_like(rgb,255); isolated[mask]=rgb[mask]
            tail_examples.append((int(value),isolated[y0:y1,x0:x1],int(mask.sum())))
        tail_examples.sort()
        fig,axes=plt.subplots(1,len(tail_examples),figsize=(16,2.6))
        for ax,(value,crop,pixels) in zip(axes,tail_examples):
            ax.imshow(crop); ax.axis("off"); ax.set_title(f"tail {value}\n{pixels} pixels",fontsize=9)
        fig.suptitle("Figure 7a · One isolated, clearly visible example of every tail value")
        plt.tight_layout(); plt.show()
        """, "Nine isolated renderer-tail crops, one mechanically selected large-mask example per exact tail value."),
        figure_method("fb-m7a", "We selected the largest accepted tail mask for each exact inserted tail value and displayed only its masked RGB pixels; no score, classifier, or outcome was used for selection."),
        md("fb-r7a", r"""
        **Plain caption.** These nine isolated crops let the reader compare the
        renderer's exact tail shapes directly rather than infer visual similarity
        from model errors.

        **How to use this figure.** Look for pairs with similar outline, area, or
        attachment geometry, then check whether Figure 7 confuses that same pair.
        A resemblance visible to a reader is a hypothesis, not a measured cause;
        it would require an independently defined image-similarity measure or a
        targeted rendering intervention before receiving explanatory credit.

        **Verdict.** **KEEP as a compact visual audit; do not count it as new proof.**
        """),

        question("fb-q7b", "7b", "Are difficult values simply rare or drawn from a larger alternative set?",
                 "For every exact donor value, compare its species support with all three mutually exclusive outcomes from Figure 4b; also report the total number of alternatives for its part.",
                 "If rarity organizes the result, lower-support values should systematically win less or fail more. A mixed pattern rejects support as a sufficient explanation.",
                 "Label every exact value, print its swap-row denominator, and verify that its three outcome fractions sum to one."),
        code("fb-f7b", r"""
        VALUE_OUTCOMES=S.assign(
            donor_wins=S.m_cf>0,
            helped_but_source_wins=(S.m_cf<=0)&(S.response_delta>0),
            no_donorward_move_and_source_wins=(S.m_cf<=0)&(S.response_delta<=0),
        )
        VS=(VALUE_OUTCOMES.groupby(["part","var_donor"]).agg(
             n_rows=("margin","size"),species_support=("sid_donor","nunique"),
             donor_wins_rate=("donor_wins","mean"),
             responded_but_source_wins_rate=("helped_but_source_wins","mean"),
             no_donorward_move_rate=("no_donorward_move_and_source_wins","mean"),
             median_margin=("margin","median")).reset_index())
        VS["alternatives_in_part"]=VS.part.map({p:hi-lo for p,(lo,hi) in SPANS.items()})
        support_correlations=[]
        for part,group in VS.groupby("part"):
            support_correlations.append({
                "part":part,"exact_values":len(group),
                "support_vs_donor_wins_spearman":group.species_support.corr(group.donor_wins_rate,method="spearman"),
                "support_vs_controlled_event_spearman":group.species_support.corr(group.responded_but_source_wins_rate,method="spearman"),
            })
        SUPPORT_CORRELATIONS=pd.DataFrame(support_correlations).set_index("part").reindex(ORDER)
        sums=VS[["donor_wins_rate","responded_but_source_wins_rate","no_donorward_move_rate"]].sum(axis=1)
        if not np.allclose(sums,1): raise RuntimeError("value-level outcome fractions do not sum to one")
        panels=[("donor_wins_rate","A · Donor finishes higher"),
                ("responded_but_source_wins_rate","B · Donorward, but source stays higher"),
                ("no_donorward_move_rate","C · No donorward move; source stays higher")]
        fig,axes=plt.subplots(1,3,figsize=(18,6),sharex=True,sharey=True)
        for ax,(column,title) in zip(axes,panels):
            for part in ORDER:
                d=VS.query("part == @part").sort_values("var_donor")
                ax.scatter(d.species_support,d[column],s=52,color=COLORS[part],label=part)
                for k,r in enumerate(d.itertuples()):
                    vertical=5 if k%2==0 else -14
                    ax.annotate(f"{part[0]}{int(r.var_donor)}",
                                (r.species_support,getattr(r,column)),fontsize=7,
                                xytext=(4,vertical),textcoords="offset points")
            ax.set_title(title,fontsize=10)
            ax.set_xlabel("species support\n(number of 50 species)")
            ax.set_ylim(-.04,1.04); ax.set_xlim(0,22); ax.grid(alpha=.18)
        axes[0].set_ylabel("fraction of swaps for that exact donor value")
        axes[2].legend(title="part",fontsize=8,loc="upper right")
        fig.suptitle("Figure 7b · Exact-value support versus all three swap outcomes")
        plt.tight_layout(); plt.show(); display(VS.round(3))
        print("Within-part rank associations (descriptive; very few exact values per part):")
        display(SUPPORT_CORRELATIONS.round(3))
        """, "Three compact labelled FunnyBird exact-value panels showing all parts together: species support against donor wins, donorward-but-source-still-wins events, and no-donorward-movement failures."),
        figure_method("fb-m7b", "We counted how many of the 50 species naturally carry each donor value, then grouped all swap rows for that value into the three exhaustive Figure 4b outcomes; no correlation model was fitted."),
        review("fb-r7b", "Figure 7b"),

        md("fb-q8", r"""
        ## 8 · Does source species organize the remaining error after exact values?

        **Question.** If two birds receive the same exact replacement, do their
        source species still accompany systematically different final margins?

        **Why exact-pair centering is needed.** A species could appear resistant
        merely because it happens to receive difficult replacements. We first
        compare each row only with swaps having the same part, old exact value,
        and inserted exact value. This removes that composition difference before
        species are summarized.

        **Complete procedure.**

        ```text
        for each swap row:
            exact_pair = (part, source_value, donor_value)

        for each exact_pair:
            pair_mean = average final margin across all its rows

        for each row:
            row_residual = row final margin - its pair_mean

        for each (part, source_species):
            species_residual = average row_residual

        display species only when it has at least five rows
        ```

        **Numerical example.** Suppose red-tail to blue-tail margins are `-5`
        and `-3` for Species A and `+1` and `+3` for Species B. Their exact-pair
        mean is `-1`. The residuals are therefore `-4,-2,+2,+4`: they average to
        zero over the pair, but Species A averages `-3` and Species B averages
        `+3`. Species A is more source-retaining than the exact-pair average.

        **Prediction and limit.** Persistent species residuals support an
        unchanged-body/species association after exact values. They do not prove
        that species causes the difference because species remains bundled with
        body shape, pose tendencies, visibility, and other renderer properties.

        **What the color does and does not explain.** A blue cell means that this
        species/part combination finished more source-favouring than other rows
        receiving the same old-to-new value replacement; red means more
        donor-favouring. Color does not itself say *why*. The table printed below
        the heatmap therefore shows, for the most extreme cells, the average
        starting margin, donorward movement, final margin, visible-pixel count,
        and exact-pair-centred residual. Frequency of the old/new values has
        already been controlled by exact-pair centering. Explaining the remaining
        color would require independently measured body shape, pose, mask geometry,
        or another context variable—not merely reusing the species name.

        ### Figure 8 · Source-species residual after exact source/donor values

        **How to read the figure.** Every row keeps one source-species identity
        and every column keeps one replaced part. Blue cells are more
        source-retaining than other swaps with the same exact source and donor
        values; red cells are more donor-receptive; white is the exact-pair
        average. Blank cells lack five eligible rows. Reading across one row asks
        whether that unchanged bird context accompanies similar or different
        residuals for several parts; no correlation threshold is imposed.
        """),
        code("fb-f8", r"""
        R=S.copy(); R["value_pair_mean"]=R.groupby(["part","var_src","var_donor"]).margin.transform("mean")
        R["margin_after_value_pair"]=R.margin-R.value_pair_mean
        pair_zero=R.groupby(["part","var_src","var_donor"]).margin_after_value_pair.mean().abs().max()
        if pair_zero>1e-10: raise RuntimeError(f"exact-pair residual means do not close: {pair_zero}")
        SP=(R.groupby(["part","sid_src"]).agg(n=("margin","size"),residual=("margin_after_value_pair","mean"))
              .reset_index().query("n>=5"))
        species_matrix=SP.pivot(index="sid_src",columns="part",values="residual").reindex(columns=ORDER)
        species_spread=(SP.groupby("part").residual.agg(["min","median","max","std","count"])
                        .reindex(ORDER))
        species_transition=(R.groupby(["part","sid_src"]).agg(
            n=("margin","size"),mean_starting_margin=("m_orig","mean"),
            mean_donorward_movement=("response_delta","mean"),
            mean_final_margin=("margin","mean"),mean_visible_pixels=("pixel_count_cf","mean"),
            exact_pair_centered_residual=("margin_after_value_pair","mean")).reset_index())
        transition_detail=(R.groupby(["part","sid_src","var_src","var_donor"]).agg(
            n=("margin","size"),mean_starting_margin=("m_orig","mean"),
            mean_donorward_movement=("response_delta","mean"),
            mean_final_margin=("margin","mean"),mean_visible_pixels=("pixel_count_cf","mean"),
            exact_pair_centered_residual=("margin_after_value_pair","mean")).reset_index())
        species_extremes=(species_transition.groupby("part",group_keys=False)
                          .apply(lambda frame: pd.concat([
                              frame.nsmallest(2,"exact_pair_centered_residual"),
                              frame.nlargest(2,"exact_pair_centered_residual")]))
                          .sort_values(["part","exact_pair_centered_residual"]))
        extreme_keys=species_extremes[["part","sid_src"]].drop_duplicates()
        extreme_transitions=transition_detail.merge(extreme_keys,on=["part","sid_src"],how="inner")
        extreme_transitions["absolute_residual"]=extreme_transitions.exact_pair_centered_residual.abs()
        extreme_transitions=(extreme_transitions.sort_values(
            ["part","sid_src","absolute_residual"],ascending=[True,True,False])
            .groupby(["part","sid_src"],as_index=False).head(2)
            .drop(columns="absolute_residual"))
        species_matrix=species_matrix.loc[species_matrix.mean(axis=1,skipna=True).sort_values().index]
        lim=float(np.nanmax(np.abs(species_matrix.to_numpy())))
        fig,ax=plt.subplots(figsize=(10,max(8,.24*len(species_matrix))))
        im=ax.imshow(species_matrix.to_numpy(),aspect="auto",cmap="coolwarm",vmin=-lim,vmax=lim)
        ax.set_xticks(np.arange(len(ORDER)),ORDER)
        ax.set_yticks(np.arange(len(species_matrix)),
                      [f"species {int(x)}" for x in species_matrix.index],fontsize=7)
        ax.set_xlabel("replaced part"); ax.set_ylabel("unchanged source species")
        ax.set_title("Figure 8 · Source-species residual after exact source/donor values")
        fig.colorbar(im,ax=ax,fraction=.035,pad=.02).set_label(
            "mean final-margin residual (blue=source-retaining, red=donor-receptive)")
        plt.tight_layout(); plt.show()
        display(species_spread.round(3))
        print("Most source-retaining and donor-receptive species/part cells, with before-to-after components:")
        display(species_extremes.round(3))
        print("Two strongest old-value to inserted-value transitions inside each printed extreme cell:")
        display(extreme_transitions.round(3))
        print("maximum absolute within-exact-pair residual mean:",pair_zero)
        """, "Common source-species-by-part heatmap of final-margin residuals after exact-pair centering, preserving every displayed species identity."),
        figure_method("fb-m8", "We subtracted each ordered `(part, source value, donor value)` pair's pooled mean margin and averaged the remaining residuals by unchanged source species; for the most extreme cells we also print their starting margin, donorward movement, final margin, visibility, and strongest exact old-to-new transitions. This is descriptive centering, not a fitted causal model."),
        review("fb-r8", "Figure 8"),

        question("fb-q8b", "8b", "How much species identity is recoverable from the learned concept vector?",
                 "After the CBM is finished, train three read-only diagnostic classifiers. Repeat the test once with the complete five-part recipe and once with each part alone. Grey receives official yes/no answers c; solid color receives model raw scores z; outline receives each raw score after subtracting the training-fold average for the same yes/no answer.",
                 "Raw or residual accuracy above the known-label control means score magnitudes reveal species beyond the nominal concept pattern. It does not say which pixels produced them, whether the saved class head uses them, or whether they caused a swap failure.",
                 "Use one fixed stratified 70/30 split of the held-out prediction population."),
        md("fb-f8b-explain", r"""
        ### Before Figure 8b: what exactly are the grey and colored bars?

        Each image has 26 official yes/no answers, written `c`. The word
        **binary** means only that an answer is 0 for “absent” or 1 for “present.”
        For example, a bird can have `beak_0=1` and the other beak values equal
        to 0. Across the five parts, these 26 answers are simply a long way to
        record the bird's complete tail + wing + beak + foot + eye recipe. They
        are dataset facts, not model scores.

        **Why can all official answers identify 100% of species while one part
        cannot?** The 50 synthetic species have distinct *combinations* of the
        five part values. Several species can share the same tail, so tail alone
        leaves several possible species. Adding wing, beak, foot, and eye can
        make the complete combination unique. It is like a five-digit code:
        one digit is shared by many records, while all five digits together can
        identify one record. Therefore the 100% bar for all five parts is
        expected dataset structure; it does not mean any single concept head is
        making a 50-species prediction.

        We train separate diagnostic classifiers whose target is the species `y`:

        - **grey bar:** input is the corresponding official yes/no answers `c`;
        - **solid colored bar:** input is the corresponding learned raw scores `z`;
        - **outlined bar:** input is `z` after subtracting the training-fold mean
          for the same exact concept and 0/1 label. This asks whether magnitudes
          still identify species *within* the official label buckets;
        - **bar height:** held-out species accuracy of that diagnostic classifier.

        Thus “species information is present” means only that a classifier can guess
        species from the supplied numbers better than chance. It does **not** mean the
        saved CBM classified the image with that accuracy, and it does **not** measure
        whether a concept used its named pixels.

        Example: suppose a purple tail is shared by species 4, 12, 19, 31, and
        44. The official tail answer can narrow the choice to those five species
        but cannot say which of the five is present. If the nine raw tail scores
        nevertheless differ systematically among those species, a diagnostic can
        do better than the official tail answers. That extra performance is the
        within-bucket species information being tested.

        **One three-bar numerical example.** Imagine all five species in that
        purple-tail group have the same official tail answer:

        - grey sees only “purple tail=yes,” so it cannot distinguish the five;
        - solid color sees the actual nine scores, for example
          `[-4,-3,+6,-2,...]`, and may recognize a species-specific score pattern;
        - outline first subtracts the average nine-score pattern of all
          purple-tail training birds. If the remaining deviations still identify
          species, the score magnitudes contain information beyond “purple=yes.”

        The bars therefore mean **nominal answer**, **all learned numerical
        detail**, and **numerical detail left after the nominal answer is removed**.

        A single part block is supplied as several numbers to a multinomial
        logistic regression. For example, the nine tail scores become nine input
        columns, and the diagnostic fits 50 weighted sums—one per species. We are
        not zeroing other weights in the saved CBM because this is a new diagnostic
        classifier. Figure 8c instead asks the swap-specific question that this
        ordinary-image probe cannot answer: whether post-swap scores retain the
        unchanged source species after exact source and donor values are controlled.

        The held-out probe population is only 30% of 500 images: 150 images, or
        roughly three per species. Therefore small differences between part bars
        can correspond to only a few images. Use this test to establish that
        information is available, not to claim a precise causal ranking.

        > **IMPORTANT: Species leakage makes backwash possible, but leakage alone does
        > not cause it. Wing is the clearest counterexample: wing `z` reveals species,
        > yet the controlled swaps show strong grounding.**

        **Connection to the MCBM/new-loss hypothesis.** A minimality loss should
        reduce the outlined bar by making images with the same official concept
        answer produce more similar internal representations. Notebook 03 tests
        that prediction; it is not assumed here. A post-hoc simulation could
        replace each raw score with its label-conditioned mean or an ideal
        `-3/+3` code and pass that vector through the unchanged species head. That
        would test downstream sensitivity to removing within-label detail, but it
        would not prove that a trainable model looks at the correct pixels. A
        stronger future loss would use the known swap directly: raise the inserted
        value, lower the removed value, and keep unrelated coordinates stable.
        That is a swap-consistency hypothesis, not an MCBM result in this chapter.

        What predicts grounding is measured separately: `response_delta`, final margin
        `m_cf`, target-part visibility, label/mask conflict, and exact donor-value
        recognition. Figure 8b is an availability/control diagnostic, not that outcome.
        """),
        code("fb-f8b", r"""
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        idx=np.arange(len(y_saved)); tr,te=train_test_split(idx,test_size=.30,random_state=20260803,stratify=y_saved)
        blocks={"all five parts together\n(26 outputs)":np.arange(z_saved.shape[1])}
        blocks.update({p:np.arange(lo,hi) for p,(lo,hi) in SPANS.items()})
        label_means=np.zeros((z_saved.shape[1],2),dtype=float)
        for j in range(z_saved.shape[1]):
            for label in [0,1]:
                rows=tr[c_saved[tr,j].astype(int)==label]
                if not len(rows): raise RuntimeError(f"no training-fold rows for concept {j}, label {label}")
                label_means[j,label]=z_saved[rows,j].mean()
        residual_z=z_saved.copy()
        for j in range(z_saved.shape[1]):
            residual_z[:,j]-=label_means[j,c_saved[:,j].astype(int)]
        probe=[]
        for name,cols in blocks.items():
            raw_model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=4000,C=1.0,random_state=20260803))
            label_model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=4000,C=1.0,random_state=20260803))
            residual_model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=4000,C=1.0,random_state=20260803))
            raw_model.fit(z_saved[tr][:,cols],y_saved[tr]); label_model.fit(c_saved[tr][:,cols],y_saved[tr])
            residual_model.fit(residual_z[tr][:,cols],y_saved[tr])
            probe.append({"block":name,
                          "raw_z_accuracy":accuracy_score(y_saved[te],raw_model.predict(z_saved[te][:,cols])),
                          "within_label_residual_accuracy":accuracy_score(y_saved[te],residual_model.predict(residual_z[te][:,cols])),
                          "processed_label_accuracy":accuracy_score(y_saved[te],label_model.predict(c_saved[te][:,cols])),
                          "label_patterns":len(np.unique(c_saved[tr][:,cols],axis=0)),"dimensions":len(cols)})
        PROBE=pd.DataFrame(probe)
        plot_order=["all five parts together\n(26 outputs)"]+ORDER
        P=PROBE.set_index("block").reindex(plot_order).reset_index()
        y=np.arange(len(P)); h=.23
        block_colors=["#333333"]+[COLORS[p] for p in ORDER]
        fig,ax=plt.subplots(figsize=(12,7))
        ax.barh(y-h,100*P.processed_label_accuracy,h,label="official yes/no answers",color="#BBBBBB")
        ax.barh(y,100*P.raw_z_accuracy,h,label="model's raw scores",color=block_colors)
        ax.barh(y+h,100*P.within_label_residual_accuracy,h,
                label="raw-score remainder within the same yes/no answer",
                facecolor="white",edgecolor=block_colors,hatch="//")
        ax.set_yticks(y,P.block); ax.invert_yaxis(); ax.set_xlim(0,105)
        ax.axvline(2,color="#666666",ls=":",lw=1,label="50-species chance (2%)")
        ax.set_xlabel("species identified correctly among 150 held-out images (%)")
        ax.set_title("Figure 8b · All five parts together versus one shared part alone")
        ax.legend(fontsize=9,loc="lower right")
        for yy,row in P.iterrows():
            for offset,value in [(-h,row.processed_label_accuracy),(0,row.raw_z_accuracy),(h,row.within_label_residual_accuracy)]:
                ax.text(100*value+1,yy+offset,f"{100*value:.1f}%",va="center",fontsize=8)
        ax.text(52,-.58,"Complete five-part recipe",ha="center",va="center",fontsize=10,fontweight="bold")
        ax.axhline(.5,color="#777777",lw=.8)
        ax.text(102,3.0,"Each row below\nuses one part only",ha="right",va="center",fontsize=9)
        plt.tight_layout(); plt.show(); display(P.round(3))
        """, "One horizontal comparison showing held-out species decoding from the complete five-part recipe versus each part alone, using official answers, raw scores, and within-answer raw-score remainders."),
        figure_method("fb-m8b", "We fitted three new multinomial logistic-regression diagnostic classifiers per concept block on 70% of the frozen CBM's held-out images and tested them on the same remaining 30%: one used binary labels, one raw logits, and one within-label residual logits; the CBM itself was not retrained or altered."),
        review("fb-r8b", "Figure 8b"),
        code("fb-r8b-compare", r'''
        grounding_comparison=[]
        probe_by_block=PROBE.set_index("block")
        for part in ORDER:
            d=S[S.part==part]
            grounding_comparison.append({
                "part":part,
                "species_decoding_from_raw_scores":probe_by_block.loc[part,"raw_z_accuracy"],
                "mean_donorward_movement":d.response_delta.mean(),
                "inserted_value_recognition":diag[part],
                "controlled_backwash_rate":d.responded_but_source_wins.mean(),
            })
        GROUNDING_COMPARISON=pd.DataFrame(grounding_comparison)
        display(Markdown("""
        ### Comparison table 8b.1 · Why species decoding is not backwash

        The four columns answer different questions. Species decoding asks what
        information a new diagnostic can recover. Donorward movement asks how far
        the named concept comparison changes after the pixels change. Recognition
        asks whether the exact inserted value becomes the largest value in its
        part. Controlled backwash requires a positive movement but a final source
        win. If species decoding alone caused backwash, these columns would follow
        the same part ordering.
        """))
        display(GROUNDING_COMPARISON.round(3))
        display(Markdown("""
        **Literal comparison.** Wing is the decisive counterexample: its scores
        decode species well, but wing has the largest donorward movement, nearly
        perfect inserted-value recognition, and very little controlled backwash.
        Therefore species information is available but is not sufficient to
        produce the failure.
        """))
        ''', "Numbered comparison table placing species decoding beside donorward movement, exact-value recognition, and controlled backwash for every part."),
        figure_method("fb-m8b-compare", "We placed already-computed part summaries side by side without fitting a model, specifically to test whether species decoding, donorward response, exact-value recognition, and controlled backwash share the same part ordering."),

        md("fb-q8c", r"""
        ## 8c · Does the saved CBM class head actually use within-label score magnitudes?

        **Question.** Figure 8b fits new diagnostic classifiers and therefore
        proves only that species information is available. Does the already-trained
        CBM's own linear species head rely on the same information?

        **Method without retraining.** The saved head computes `Wz+b`. On the
        held-out split, calculate the training-fold mean raw score for every exact
        concept separately when its label is 0 and 1. Replace a test image's score
        by the mean for its own known label, then pass that altered vector through
        the unchanged saved `W` and `b`. This removes within-label magnitude while
        preserving the binary concept pattern.

        Example: if positive blue-tail scores average `+5` but one image has
        `+8`, the replacement changes only that coordinate from `+8` to `+5`.
        No class-head weight is refitted. If the species decision changes, the
        existing CBM—not merely a new probe—was using that extra magnitude.

        Top-1 accuracy alone is blunt: scores can change confidence without changing
        the winning species. We therefore also measure **probability mass
        redistributed**, half the sum of the absolute changes across all 50 species
        probabilities. It ranges from 0 (no probability changed) to 1 (all assigned
        probability moved elsewhere). For example, changing probabilities from
        `[0.8, 0.2]` to `[0.6, 0.4]` redistributes
        `(abs(-0.2)+abs(+0.2))/2 = 0.2`.

        ### Figure 8c · Existing-head reliance on within-label magnitudes

        **How to read the figure.** Panel A compares saved-head accuracy using the
        untouched raw vector with accuracy after all 26 coordinates are replaced
        by label-conditioned means. Panel B reports mean probability mass
        redistributed, which can be nonzero even when the winning species does
        not change. The first bar removes all 26 magnitudes; later bars remove
        only the named part block. The zero top-1-change result is printed in the
        table rather than occupying an empty plot panel. Every result uses the
        unchanged saved head and the same held-out images.
        """),
        code("fb-f8c", r"""
        import torch
        sys.path.insert(0,str(REPO/"compat"))
        sys.path.insert(0,str(REPO/"external"/"ConceptBottleneck"))
        try:
            saved_model=torch.load(MODEL,map_location="cpu",weights_only=False)
        except TypeError:
            saved_model=torch.load(MODEL,map_location="cpu")
        head=saved_model.sec_model.linear
        W=head.weight.detach().cpu().numpy(); b=head.bias.detach().cpu().numpy()
        if W.shape!=(50,26): raise RuntimeError(f"unexpected saved class-head shape {W.shape}")
        raw_class_logits=z_saved@W.T+b
        if not np.array_equal(raw_class_logits.argmax(1),y_pred_saved):
            raise RuntimeError("reconstructed saved linear-head predictions disagree with export")
        expected_z=np.empty_like(z_saved)
        for j in range(z_saved.shape[1]): expected_z[:,j]=label_means[j,c_saved[:,j].astype(int)]
        def stable_softmax(values):
            shifted=values-values.max(axis=1,keepdims=True)
            exp=np.exp(shifted)
            return exp/exp.sum(axis=1,keepdims=True)
        raw_eval_logits=raw_class_logits[te]
        raw_eval_prob=stable_softmax(raw_eval_logits)
        raw_eval_pred=raw_eval_logits.argmax(1)
        raw_acc=float((raw_eval_pred==y_saved[te]).mean())
        mean_logits=(expected_z@W.T+b)[te]
        mean_pred=mean_logits.argmax(1)
        mean_acc=float((mean_pred==y_saved[te]).mean())
        mean_probability_shift=float(
            (0.5*np.abs(raw_eval_prob-stable_softmax(mean_logits)).sum(axis=1)).mean()
        )
        mean_prediction_change=float((raw_eval_pred!=mean_pred).mean())
        ablations=[]
        for part,(lo,hi) in SPANS.items():
            altered=z_saved.copy(); altered[:,lo:hi]=expected_z[:,lo:hi]
            altered_logits=(altered@W.T+b)[te]
            altered_pred=altered_logits.argmax(1)
            acc=float((altered_pred==y_saved[te]).mean())
            ablations.append({"part":part,"accuracy_after_removing_within_label_magnitude":acc,
                              "accuracy_drop":raw_acc-acc,
                              "top1_prediction_change_rate":float((raw_eval_pred!=altered_pred).mean()),
                              "mean_probability_redistributed":float(
                                  (0.5*np.abs(raw_eval_prob-stable_softmax(altered_logits)).sum(axis=1)).mean()
                              ),"dimensions":hi-lo})
        HEAD_USE=pd.DataFrame(ablations).set_index("part").reindex(ORDER).reset_index()
        fig,axes=plt.subplots(1,2,figsize=(13,4.8))
        axes[0].bar(["raw z","label-conditioned means"],[raw_acc,mean_acc],color=["#333333","#BBBBBB"])
        axes[0].set_ylim(0,1); axes[0].set_ylabel("held-out accuracy of unchanged saved head")
        axes[0].set_title("A · Remove all within-label magnitudes")
        labels=["all 26"]+HEAD_USE.part.tolist()
        shifts=[mean_probability_shift]+HEAD_USE.mean_probability_redistributed.tolist()
        colors=["#333333"]+[COLORS[p] for p in HEAD_USE.part]
        axes[1].bar(labels,shifts,color=colors)
        axes[1].set_ylim(bottom=0); axes[1].set_ylabel("mean probability mass redistributed")
        axes[1].set_title("B · Confidence movement without a changed winner")
        axes[1].tick_params(axis="x",rotation=25)
        fig.suptitle("Figure 8c · Does the existing CBM head use within-label magnitude?")
        plt.tight_layout(); plt.show()
        display(pd.DataFrame([{"raw_saved_head_accuracy":raw_acc,
                               "all_within_label_magnitude_removed_accuracy":mean_acc,
                               "accuracy_change":raw_acc-mean_acc,
                               "top1_prediction_change_rate":mean_prediction_change,
                               "mean_probability_redistributed":mean_probability_shift}]).round(4))
        display(HEAD_USE.round(4))
        """, "Accuracy, top-one decision changes, and probability redistribution in the unchanged saved Koh linear species head after within-label raw-score magnitudes are removed globally or one part block at a time."),
        figure_method("fb-m8c", "We passed original logits and label-conditioned mean-replaced logits through the unchanged saved linear class head `Wz+b`; no new classifier was trained, and only this analysis-time input vector was replaced."),
        code("fb-r8c", r'''
        part_shift_text=", ".join(
            f"{r.part} `{r.mean_probability_redistributed:.4f}`"
            for r in HEAD_USE.itertuples()
        )
        display(Markdown(f"""
        ### Plain-language reference for Figure 8c

        **Plain caption.** Removing image-specific within-label magnitudes does not
        change any held-out top-1 species decision, although it redistributes some
        probability inside the unchanged saved head.

        **Terms.** `Wz+b` is the unchanged saved 26-to-50 linear species head.
        Label-conditioned mean replacement preserves whether each concept is 0 or
        1 while removing its image-specific magnitude. Probability mass
        redistributed is 0 when no class probability changes and 1 when all
        assigned probability moves to other classes.

        **Literal values.** Raw and all-magnitudes-removed top-1 accuracy are both
        `{raw_acc:.3f}` on `{len(te)}` held-out images, and the top-1 prediction
        change rate is `{mean_prediction_change:.3f}`. Removing all 26 within-label
        magnitudes redistributes `{mean_probability_shift:.4f}` probability mass on
        average. Removing one block at a time redistributes: {part_shift_text}.
        Every part-specific top-1 change rate is also zero.

        **Interpretation.** Figure 8b shows that a newly fitted decoder can extract
        species from score magnitudes. Figure 8c now shows that those magnitudes are
        not necessary for the saved CBM's top-1 decisions on this split. Nonzero
        probability redistribution means the head is numerically sensitive to them,
        but that is weaker than changing its chosen species.

        **Strongest alternative explanation.** Replacing scores by means creates
        artificial vectors, and a 150-image split may miss rare decision changes.

        **Discriminating test.** Connect the unchanged head's weighted off-target
        evidence directly to the controlled swap outcome in Figure 8d.

        **Verdict.** **VALID TEST, NO SUPPORT that within-label magnitudes are needed
        for held-out top-1 species decisions; ACCEPTED only for any measured
        confidence redistribution.**

        **Next question.** Does retained source-species evidence on the controlled
        replacement accompany a more source-negative concept margin?
        """))
        ''', "Numbered plain-language review of Figure 8c using its executed held-out accuracy, decision-change, and probability-redistribution values."),

        md("fb-q8d", r"""
        ## 8d · Do off-target scores carry source-species evidence that is linked to swap failure?

        **Final goal.** Determine whether distributed species information is a
        mechanism relevant to the accepted controlled outcome, rather than an
        interesting but irrelevant decoding fact.

        **Hypothesis.** After replacing one part, the other exact-value outputs
        in that same part block may retain a soft fingerprint of the unchanged
        source species. If the saved class head reads that fingerprint, stronger
        retained source-over-donor evidence should accompany a more negative
        donor-minus-source concept margin.

        **What “off-target” means.** In a red-to-blue tail swap, red and blue are
        the target pair. The other seven tail outputs are off-target coordinates.
        They are not irrelevant to the linear species head: every one has a saved
        weight for every species. We subtract the expected score for each
        counterfactual 0/1 label, then apply the existing source-species weight
        minus donor-species weight to only those off-target residual scores.

        Example: suppose the seven off-target residuals contribute `+2.0` to the
        source class logit and `-0.5` to the donor class logit. Their retained
        source-over-donor evidence is `+2.5` class-logit units. No probe or model
        is trained.

        **Controls and stopping rule.** Center both this evidence and the final
        concept margin within the same exact `(part, source value, donor value)`
        pair. If their within-pair association is absent, stop: species leakage
        remains available but this test does not support it as an explanation.
        If it is present, it is still an association because body, pose, and
        visibility were not independently manipulated.

        ### Figure 8d · Existing-head off-target source fingerprint versus controlled outcome

        **How to read the figure.** Panel A shows the within-exact-pair rank
        correlation for every part; negative is the predicted direction because
        more source-class evidence should accompany a more negative final concept
        margin. Rank correlation compares order rather than raw scale: `-1` means
        higher source evidence always accompanies lower margin, `0` means no
        ordered relationship, and `+1` means they rise together. Panel B orders
        rows into five equal-count evidence groups within each part and shows the
        controlled-backwash fraction. Increasing lines support the hypothesis.
        Every group prints its denominator in the table.
        """),
        code("fb-f8d", r"""
        D=R.copy()
        off_evidence=[]; pair_evidence=[]
        for row in D.itertuples():
            part=row.part; lo,hi=SPANS[part]
            block_columns=[f"z_cf_{part}_{k}" for k in range(hi-lo)]
            values=np.asarray([getattr(row,column) for column in block_columns],dtype=float)
            if not np.isfinite(values).all(): raise RuntimeError(f"non-finite {part} counterfactual block")
            source_local=int(row.var_src); donor_local=int(row.var_donor)
            labels=np.zeros(hi-lo,dtype=int); labels[donor_local]=1
            expected=label_means[np.arange(lo,hi),labels]
            residual=values-expected
            source_species=int(row.sid_src); donor_species=int(row.sid_donor)
            if not (0<=source_species<W.shape[0] and 0<=donor_species<W.shape[0]):
                raise RuntimeError("swap species ID falls outside saved class-head rows")
            weight_difference=W[source_species,lo:hi]-W[donor_species,lo:hi]
            pair_mask=np.zeros(hi-lo,dtype=bool); pair_mask[[source_local,donor_local]]=True
            pair_evidence.append(float(weight_difference[pair_mask]@residual[pair_mask]))
            off_evidence.append(float(weight_difference[~pair_mask]@residual[~pair_mask]))
        D["pair_source_over_donor_evidence"]=pair_evidence
        D["offtarget_source_over_donor_evidence"]=off_evidence
        pair_keys=["part","var_src","var_donor"]
        D["offtarget_evidence_after_pair"]=(D.offtarget_source_over_donor_evidence-
            D.groupby(pair_keys).offtarget_source_over_donor_evidence.transform("mean"))
        D["margin_after_pair"]=D.margin-D.groupby(pair_keys).margin.transform("mean")
        corr=[]; bins=[]
        for part,d in D.groupby("part"):
            rank_corr=float(d.offtarget_evidence_after_pair.rank().corr(d.margin_after_pair.rank()))
            corr.append({"part":part,"rank_correlation":rank_corr,"n":len(d)})
            d=d.copy(); d["evidence_group"]=pd.qcut(d.offtarget_evidence_after_pair,5,labels=False,duplicates="drop")
            for group,g in d.groupby("evidence_group"):
                bins.append({"part":part,"evidence_group":int(group)+1,"n":len(g),
                             "mean_evidence":g.offtarget_evidence_after_pair.mean(),
                             "controlled_backwash_rate":g.responded_but_source_wins.mean(),
                             "median_margin_after_pair":g.margin_after_pair.median()})
        FINGERPRINT_CORR=pd.DataFrame(corr).set_index("part").reindex(ORDER).reset_index()
        FINGERPRINT_BINS=pd.DataFrame(bins)
        fig,axes=plt.subplots(1,2,figsize=(14,4.8))
        axes[0].bar(FINGERPRINT_CORR.part,FINGERPRINT_CORR.rank_correlation,
                    color=[COLORS[p] for p in FINGERPRINT_CORR.part])
        axes[0].axhline(0,color="black",lw=.8)
        axes[0].set_ylabel("rank correlation with exact-pair-centered final margin")
        axes[0].set_title("A · Negative is the predicted direction")
        for part,d in FINGERPRINT_BINS.groupby("part"):
            axes[1].plot(d.evidence_group,d.controlled_backwash_rate,"o-",label=part,color=COLORS[part])
        axes[1].set_xticks(range(1,6)); axes[1].set_xlabel("off-target source-evidence group (low to high)")
        axes[1].set_ylabel("controlled-backwash fraction"); axes[1].set_ylim(0,1)
        axes[1].set_title("B · Does retained source evidence accompany the event?")
        axes[1].legend(fontsize=8)
        fig.suptitle("Figure 8d · Off-target source-species fingerprint in the saved CBM head")
        plt.tight_layout(); plt.show(); display(FINGERPRINT_CORR.round(3)); display(FINGERPRINT_BINS.round(3))
        """, "Within-exact-pair association between off-target source-over-donor evidence from the saved class head and the controlled FunnyBird outcome."),
        figure_method("fb-m8d", "Using the unchanged saved class-head weights, we converted off-target within-part residual logits into source-minus-donor class evidence, centered evidence and margin within exact swap pair, and measured rank association; no probe or model was fitted."),
        md("fb-r8d", r"""
        ### Plain-language reference for Figure 8d

        **Plain caption.** This is the first test that can connect distributed
        species leakage to the controlled swap outcome using the saved CBM head
        rather than a newly trained species decoder.

        **Terms.** Off-target coordinates are the other exact values in the
        replaced part block. Source-over-donor evidence is their contribution to
        the saved source class logit minus their contribution to the saved donor
        class logit. Both evidence and final margin are centered within exact
        source/donor value pair before association is measured.

        **Literal values.** Within exact source/donor value pairs, the rank
        correlations between off-target source evidence and final margin are
        tail `-0.181`, eye `-0.096`, wing `-0.076`, foot `-0.076`, and beak
        `-0.061`, with 1,000 swaps per part. Negative is the predicted direction,
        but every magnitude is weak. From the lowest to highest evidence fifth,
        event rates change from `0.380` to `0.515` for tail, `0.150` to `0.220`
        for beak, `0.045` to `0.135` for eye, `0.020` to `0.050` for foot, and
        `0.015` to `0.030` for wing. The paths are not strictly increasing: tail
        and beak both fall in the final group, and several middle groups reverse.

        **Interpretation.** All five correlations point in the predicted direction,
        and the highest-evidence group has a higher event rate than the lowest for
        every part. That is weak, consistent association—not a complete or causal
        explanation. Together with Figure 8c, it does not justify saying that
        distributed leakage controls the saved model's top-1 species decision.

        **Alternative.** Source species remains bundled with body shape, pose,
        and visibility. Association here cannot assign independent causal credit.

        **Verdict.** **ACCEPTED FOR A WEAK WITHIN-PAIR ASSOCIATION ONLY; REVISE any
        claim that this is an established mechanism or a sufficient explanation.**

        **Next question.** Does the off-target fingerprint actually move from
        source-favouring toward donor-favouring after the replacement, and does
        that transition differ between successful and failing swaps?
        """),

        md("fb-q8e", r"""
        ## 8e · Does a donor replacement remove the source fingerprint and insert a donor fingerprint?

        **Why this is the professor-facing fingerprint test.** A decoder showing
        that species information exists is not enough. The direct questions are:

        1. Is source-favouring information present before replacement?
        2. After donor pixels are inserted, does that information move toward the
           donor or remain attached to the unchanged source bird?
        3. Is the transition different when the donor wins, when the new pixels
           help but the source still wins, and when there is no donorward move?
        4. Does the pattern remain after removing the expected score for the
           official old/new concept labels?
        5. Is this measured through the saved CBM head rather than a newly fitted
           species probe?

        **Fingerprint definition.** For one replacement, take the saved linear
        species-head weights for the source species and subtract the weights for
        the donor species. Multiply that difference by the part block's raw-score
        deviations from their ordinary 0/1-label means. The result is one signed
        class-logit contribution:

        - positive: the score pattern favours the source species;
        - zero: it favours neither side;
        - negative: it favours the donor species.

        We exclude the removed source coordinate and inserted donor coordinate.
        Thus the figure asks whether the *other values in the same part block*
        retain source/donor identity. For a red-to-blue tail replacement, the
        other seven tail outputs form this off-target fingerprint.

        **Before and after.** This cell evaluates both sides through one matched
        inference session: each of the 250 saved original renders and each of the
        3,040 unique saved replacement images pass through the same accepted
        frozen Koh checkpoint on CUDA, one image at a time. The accepted CSV was
        produced in an earlier CUDA session, so tiny floating-point replay
        differences are printed rather than mistaken for different inputs. The
        fail-closed scientific gate is that all 5,000 rows must retain the same
        donor-win/donorward-but-source-wins/no-donorward-move assignment as the
        accepted CSV. No model is trained or changed.

        **Prediction.** A clean donor transfer should move the signed fingerprint
        downward, from source-favouring toward donor-favouring. Controlled
        backwash rows may move downward less or remain more source-positive.

        ### Figure 8e · Off-target source-to-donor fingerprint transition

        **How to read the figure.** Each small panel is one replaced part. The
        left point is the mean off-target fingerprint before replacement and the
        right point is after replacement. Lines separate the three complete swap
        outcomes. Downward means movement toward the donor fingerprint; remaining
        above zero means the off-target pattern still favours the source species.
        The table prints every outcome denominator and mean change.
        """),
        code("fb-f8e", r'''
        from torchvision import transforms as tv_transforms
        replay_device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if replay_device.type != "cuda":
            raise RuntimeError(
                "Figure 8e exact replay requires CUDA because the accepted swap "
                "CSV was generated by CUDA inference; CPU replay is not an exact "
                "numerical reproduction"
            )
        saved_model=saved_model.to(replay_device).eval()
        koh_image_transform=tv_transforms.Compose([
            tv_transforms.CenterCrop(299),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.5,0.5,0.5],std=[2.0,2.0,2.0]),
        ])
        def replay_concept_logits(path_text):
            path=Path(path_text)
            if not path.is_file():
                raise FileNotFoundError(f"missing accepted render {path}")
            tensor=koh_image_transform(Image.open(path).convert("RGB")).unsqueeze(0).to(replay_device)
            with torch.no_grad():
                outputs=saved_model(tensor)
            if not isinstance(outputs,(list,tuple)) or len(outputs)!=27:
                raise RuntimeError("unexpected frozen Koh output contract during matched fingerprint replay")
            return torch.cat([value.reshape(-1,1) for value in outputs[1:]],dim=1)[0].cpu().numpy()

        original_records=(S[["orig_render_id","image_orig_path"]]
                          .drop_duplicates("orig_render_id").reset_index(drop=True))
        if len(original_records)!=250:
            raise RuntimeError(f"expected 250 unique original renders, found {len(original_records)}")
        replacement_records=(S[["image_cf_sha256","image_cf_path"]]
                             .drop_duplicates("image_cf_sha256").reset_index(drop=True))
        if len(replacement_records)!=3040:
            raise RuntimeError(f"expected 3040 unique replacement RGB images, found {len(replacement_records)}")
        original_z={str(record.orig_render_id):replay_concept_logits(record.image_orig_path)
                    for record in original_records.itertuples(index=False)}
        replacement_z={str(record.image_cf_sha256):replay_concept_logits(record.image_cf_path)
                       for record in replacement_records.itertuples(index=False)}
        replay_source_orig=[]; replay_donor_orig=[]
        replay_source_cf=[]; replay_donor_cf=[]
        accepted_outcomes=[]; replayed_outcomes=[]
        fingerprint_rows=[]
        for row in S.itertuples():
            part=row.part; lo,hi=SPANS[part]
            before=np.asarray(original_z[str(row.orig_render_id)][lo:hi],dtype=float)
            after=np.asarray(replacement_z[str(row.image_cf_sha256)][lo:hi],dtype=float)
            source_local=int(row.var_src); donor_local=int(row.var_donor)
            replay_source_orig.append(before[source_local]); replay_donor_orig.append(before[donor_local])
            replay_source_cf.append(after[source_local]); replay_donor_cf.append(after[donor_local])
            before_labels=np.zeros(hi-lo,dtype=int); before_labels[source_local]=1
            after_labels=np.zeros(hi-lo,dtype=int); after_labels[donor_local]=1
            global_columns=np.arange(lo,hi)
            before_residual=before-label_means[global_columns,before_labels]
            after_residual=after-label_means[global_columns,after_labels]
            weight_difference=W[int(row.sid_src),lo:hi]-W[int(row.sid_donor),lo:hi]
            off_target=np.ones(hi-lo,dtype=bool)
            off_target[[source_local,donor_local]]=False
            before_fingerprint=float(weight_difference[off_target]@before_residual[off_target])
            after_fingerprint=float(weight_difference[off_target]@after_residual[off_target])
            if row.m_cf>0:
                outcome="donor wins"
            elif row.response_delta>0:
                outcome="donorward, source wins"
            else:
                outcome="no donorward move"
            replay_m_orig=before[donor_local]-before[source_local]
            replay_m_cf=after[donor_local]-after[source_local]
            replay_delta=replay_m_cf-replay_m_orig
            if replay_m_cf>0:
                replay_outcome="donor wins"
            elif replay_delta>0:
                replay_outcome="donorward, source wins"
            else:
                replay_outcome="no donorward move"
            accepted_outcomes.append(outcome); replayed_outcomes.append(replay_outcome)
            fingerprint_rows.append({"part":part,"outcome":outcome,
                                     "before_source_minus_donor":before_fingerprint,
                                     "after_source_minus_donor":after_fingerprint,
                                     "change_toward_source":after_fingerprint-before_fingerprint})
        coordinate_checks={
            "original source":(np.asarray(replay_source_orig),S.z_old_orig.to_numpy()),
            "original donor":(np.asarray(replay_donor_orig),S.z_new_orig.to_numpy()),
            "replacement source":(np.asarray(replay_source_cf),S.z_old.to_numpy()),
            "replacement donor":(np.asarray(replay_donor_cf),S.z_new.to_numpy()),
        }
        replay_diagnostics={"device":str(replay_device)}
        for name,(current,accepted) in coordinate_checks.items():
            if not np.isfinite(current).all():
                raise RuntimeError(f"non-finite values in current replay: {name}")
            error=np.abs(current-accepted)
            replay_diagnostics[f"{name} median absolute difference"]=float(np.median(error))
            replay_diagnostics[f"{name} maximum absolute difference"]=float(error.max())
        outcome_agreement=float(np.mean(np.asarray(accepted_outcomes)==np.asarray(replayed_outcomes)))
        replay_diagnostics["accepted outcome agreement"]=outcome_agreement
        print("Figure 8e matched-replay audit:",replay_diagnostics)
        if outcome_agreement != 1.0:
            changed=int(np.sum(np.asarray(accepted_outcomes)!=np.asarray(replayed_outcomes)))
            raise RuntimeError(f"matched replay changes {changed} of 5000 accepted swap outcomes")
        FINGERPRINT_TRANSITION=(pd.DataFrame(fingerprint_rows).groupby(["part","outcome"])
            .agg(n=("before_source_minus_donor","size"),
                 before=("before_source_minus_donor","mean"),
                 after=("after_source_minus_donor","mean"),
                 change=("change_toward_source","mean")).reset_index())
        outcome_order=["donor wins","donorward, source wins","no donorward move"]
        outcome_colors={"donor wins":"#009E73","donorward, source wins":"#D55E00","no donorward move":"#777777"}
        fig,axes=plt.subplots(2,3,figsize=(15,9),sharex=True)
        for ax,part in zip(axes.flat,ORDER):
            d=FINGERPRINT_TRANSITION[FINGERPRINT_TRANSITION.part==part].set_index("outcome")
            for outcome in outcome_order:
                if outcome not in d.index: continue
                values=d.loc[outcome]
                ax.plot([0,1],[values.before,values.after],"o-",color=outcome_colors[outcome],
                        label=f"{outcome}; n={int(values.n)}")
            ax.axhline(0,color="black",lw=.8)
            ax.set_xticks([0,1],["before\nsource bird","after\ndonor part inserted"])
            ax.set_ylabel("off-target fingerprint\n+ source / - donor")
            ax.set_title(part); ax.legend(fontsize=7)
        axes.flat[-1].axis("off")
        fig.suptitle("Figure 8e · Does the off-target fingerprint transfer from source toward donor?")
        plt.tight_layout(rect=[0,0,1,.96]); plt.show(); display(FINGERPRINT_TRANSITION.round(3))
        ''', "Before-versus-after off-target source-minus-donor fingerprint through the unchanged saved class head, separated by part and all three swap outcomes."),
        figure_method("fb-m8e", "We replayed all accepted original and unique replacement images in one matched CUDA session through the frozen Standard CBM, verified that all 5,000 accepted outcome categories were unchanged, excluded the source/donor coordinates, and applied the unchanged saved-head weights; nothing was trained."),
        code("fb-r8e", r'''
        transition_wide=FINGERPRINT_TRANSITION.pivot(index="part",columns="outcome",values="change")
        donor_change=transition_wide.get("donor wins",pd.Series(dtype=float))
        backwash_change=transition_wide.get("donorward, source wins",pd.Series(dtype=float))
        comparison=(backwash_change-donor_change).dropna()
        n_more_source=int((comparison>0).sum())
        display(Markdown(f"""
        ### Plain-language reference for Figure 8e

        **Plain caption.** This figure directly checks whether the off-target
        pattern changes from source-favouring toward donor-favouring after the
        named pixels are replaced, and whether that transition differs between
        successful and failing swaps.

        **Literal result.** The complete table above reports the before value,
        after value, change, and denominator for every part/outcome combination.
        In `{n_more_source}` of `{len(comparison)}` parts, controlled-backwash
        rows retain a more sourceward change than donor-win rows when the two are
        compared on this signed fingerprint scale.

        **Interpretation rule.** A negative change is movement toward the donor;
        a positive change is movement toward the source. An after-value above
        zero means the other outputs in the replaced part still favour the source
        species after the donor pixels arrive. This is stronger than decoding
        availability because it uses the frozen saved head before and after the
        actual intervention.

        **Strongest alternative.** The source and donor weight rows were learned
        jointly with the same concept scores, and source species remains bundled
        with body and pose. This test links a fingerprint to the intervention but
        does not independently manipulate the fingerprint.

        **Verdict.** Accept only the literal before/after transition and its
        outcome contrast. Call it a mechanism candidate only if controlled-
        backwash rows consistently retain more sourceward evidence; never call it
        a complete causal explanation.

        **Next question.** Would a representation objective that compresses each
        concept toward fixed label-dependent values reduce this fingerprint?
        """))
        ''', "Executed plain-language review of the source-to-donor fingerprint transition and its controlled-outcome contrast."),

        md("fb-r8e-professor", r"""
        ### What a professor can ask about the fingerprint result

        | Likely question | Where the answer appears | What may be claimed |
        |---|---|---|
        | Was the source fingerprint already present before the replacement? | The left point and `before` column in Figure 8e. | Its signed saved-head contribution can be measured before the swap. |
        | Did inserting the donor part replace it with a donor fingerprint? | The before-to-after line and `change` column. Negative movement is donorward; the `after` sign says which species is still favoured. | Report the literal direction and size separately for every part and outcome. |
        | Is retained source identity specifically associated with backwash? | Compare **donor wins** with **donorward, source wins** within the same part. | A consistent contrast supports a mechanism candidate; an inconsistent contrast does not. |
        | Is this merely the removed source score staying high? | No. Both the removed-source and inserted-donor coordinates are excluded. | The result concerns the other outputs in that part block. |
        | Is this merely ordinary 0/1 concept information? | No. Each raw score has its expected mean for its own official label subtracted first. | The remainder is within-label magnitude variation, not the ordinary label pattern. |
        | Was a new classifier trained to manufacture the result? | No. The calculation uses the frozen accepted CBM's existing linear species-head rows. | It is an audit of the deployed prediction rule, not a fitted diagnostic probe. |
        | Does this prove that the fingerprint causes backwash? | No. The fingerprint itself was not independently intervened on, and species is still bundled with body and pose. | At most: direct intervention-linked association and a concrete next mechanism test. |
        | Why examine MCBM next? | The post-ledger handoff states the exact compression penalty and the prediction it creates. | MCBM tests whether reducing within-label magnitude freedom also reduces the fingerprint and controlled backwash; it cannot be assumed. |

        This table is the boundary for presenting Figure 8e: show the direct
        before/after evidence, then stop at the strongest claim the design
        permits.
        """),

        md("fb-mcbm-bridge", r"""
        ## Post-ledger handoff · What MCBM changes, and what this Standard-CBM evidence predicts

        This section explains the next model without inserting MCBM numerical
        results into the Standard-CBM discovery.

        **Standard CBM in this notebook.** Each exact concept has one unrestricted
        raw logit `z_j`. The linear species head reads all 26 logits. Concept loss
        teaches the correct 0/1 answer, while species loss can still reward useful
        differences in magnitude among images with the same answer.

        **Official MCBM used in this project.** MCBM gives each concept a
        representation `z_j`, predicts the concept and species from those
        representations, and adds a representation penalty. In the pinned source,
        the label-dependent target is exactly:

        `target(c_j) = 6*c_j - 3`

        so an absent concept targets `-3` and a present concept targets `+3`.
        The per-concept source implementation contributes

        `0.2 * mean((z_j - target(c_j))^2)`

        and the total loss is

        `task loss + beta * concept loss + gamma * representation loss`.

        `gamma` controls how strongly the model is penalized for retaining
        image-specific variation around `-3/+3`. Gamma zero is still the MCBM
        architecture with this penalty switched off; it is not the Koh Joint
        Standard CBM.

        **What this does not guarantee.** The penalty says what number a concept
        representation should approach. It does not require the model to obtain
        that number from the named part pixels. A model could use the body/species
        and still output `+3` for the correct ordinary-image label.

        **Prediction established here for notebook 03.** If within-label
        magnitude variation is important to the fingerprint, increasing gamma
        should reduce within-label species decoding and source-fingerprint
        retention. Whether it also improves the controlled donor/source margin is
        an empirical question for the fixed-render MCBM sweep—not something to
        assume from compression alone.

        The comparison must interpret those two measurements separately rather
        than treating gamma as a generic improvement knob:

        | MCBM result across gamma | What it would mean | Next discriminating branch |
        |---|---|---|
        | fingerprint falls and controlled backwash falls | compression is consistent with weakening this candidate pathway | verify the change for each part and exact value; do not call it complete mediation |
        | fingerprint falls but controlled backwash remains | extra magnitude information was compressed, but it was not sufficient to remove the grounding failure | test local pixel response and visibility/label mechanisms before adding a spatial constraint |
        | fingerprint remains but controlled backwash falls | the diagnostic fingerprint was not the route that mattered, or another MCBM change altered swap response | compare the frozen saved-head contribution and exact-value response directly |
        | neither falls | the `-3/+3` representation penalty does not address this Standard-CBM mechanism | only then consider swap-consistency or named-region routing as a separately declared experiment |

        This table prevents a post-hoc story in which any MCBM outcome is called
        confirmation. A spatially constrained model remains a later
        discriminating test, not evidence silently imported into notebook 02.

        ### Handoff schematic · Representation penalty used by MCBM (not a result)

        The x-axis is a possible one-dimensional concept representation. The
        y-axis is the additional representation penalty before multiplication by
        gamma. The absent-label curve is minimized at `-3`; the present-label
        curve is minimized at `+3`. Standard Koh Joint has no corresponding
        representation penalty, shown by the zero line. This is an equation
        visualization, not trained-model evidence.
        """),
        code("fb-f8f", r'''
        grid=np.linspace(-8,8,401)
        absent=.2*(grid+3)**2
        present=.2*(grid-3)**2
        fig,ax=plt.subplots(figsize=(9,4.8))
        ax.plot(grid,absent,label="MCBM label 0: target -3",color="#0072B2")
        ax.plot(grid,present,label="MCBM label 1: target +3",color="#D55E00")
        ax.axhline(0,color="#555555",ls="--",label="Koh Joint: no representation penalty")
        ax.axvline(-3,color="#0072B2",ls=":",lw=1); ax.axvline(3,color="#D55E00",ls=":",lw=1)
        ax.set_ylim(0,12); ax.set_xlabel("one concept representation value z_j")
        ax.set_ylabel("representation penalty before multiplying by gamma")
        ax.set_title("Handoff schematic · What the MCBM -3/+3 representation penalty rewards")
        ax.legend(); plt.tight_layout(); plt.show()
        display(pd.DataFrame([
            {"model":"Koh Joint Standard CBM","representation_target":"none","extra_weight":"none"},
            {"model":"MCBM","representation_target":"-3 absent / +3 present","extra_weight":"gamma"},
        ]))
        ''', "Schematic squared MCBM representation penalties around minus three and plus three, contrasted with the absence of this penalty in Koh Joint."),
        figure_method("fb-m8f", "We plotted the representation-penalty equation from the pinned MCBM source over hypothetical scalar values; this is a schematic calculation, not a trained-model result."),
        md("fb-r8f", r"""
        ### Plain-language reference for the MCBM handoff schematic

        **Plain caption.** MCBM explicitly discourages two images with the same
        concept label from using very different representation magnitudes, while
        the Standard Koh Joint model has no matching compression penalty.

        **Limited conclusion.** MCBM supplies a principled test of whether
        within-label magnitude freedom matters. It does not automatically enforce
        named-pixel grounding. Notebook 03 must compare the same controlled
        response, final margin, exact-value recognition, and fingerprint metrics
        across gamma before assigning a grounding benefit.

        **Verdict.** **KEEP AFTER THE STANDARD-CBM LEDGER AS A METHOD/PREDICTION
        HANDOFF; no MCBM result is claimed in notebook 02.**
        """),

        md("fb-q9", r"""
        ## 9 · How accurately can progressively richer grouping information predict unseen margins?

        **Question.** Do visibility, exact values, and source species predict the
        final raw-logit margin for original source images that were not used to
        build the prediction rule?

        **What a prediction rule is.** It is a lookup learned from training folds,
        not another neural network. For example, the rule can learn that visible
        tail swaps in its training rows have mean margin `-2`, then predict `-2`
        for a held-out visible tail swap. The prediction is compared with the
        held-out observed margin. The target is the final donor-minus-source
        concept margin—not species and not a yes/no backwash label. If the true
        held-out margin is `-4` and the rule predicts `-1`, its absolute error is
        `3` raw-logit units.

        **Five-fold procedure.**

        ```text
        assign each original source image to one of five folds
        keep every swap from that image in the same fold

        for each held-out fold:
            use the other four folds to estimate group means
            blend each group mean with 10 virtual rows at the overall mean
            predict the untouched fold

        combine predictions from all five held-out folds
        calculate RMSE and MAE
        ```

        This split prevents the rule from seeing another swap made from the same
        original image. It does **not** hold out entire species: a source species
        present in the test fold can also appear through different original images
        in the training folds. Therefore the `+ source species` stage asks whether
        a species-specific lookup helps for additional images of already observed
        species. It cannot show generalization to a previously unseen species or
        rule out species-category memorization. A leave-one-source-species-out test
        would be required for that stronger claim.

        **What the three summary columns mean.** `MAE` is the average absolute
        miss. If three absolute misses are `1, 2, 6`, MAE is `3`. `RMSE` squares
        misses before averaging, so the large miss of `6` is punished more; both
        are measured in raw-logit-margin units and lower is better. `Coverage` is
        the fraction of held-out rows whose exact lookup key had appeared in the
        four training folds. Coverage `62.3%` means `37.7%` of held-out rows lacked
        that exact key and had to fall back to a broader average. Figure 9 asks
        whether a rule generalizes to unseen original images. Figure 9b only puts
        earlier summaries beside one another; it fits nothing and adds no new
        evidence.

        The ten virtual rows shrink tiny groups toward the overall mean so one or
        two unusual rows cannot create an extreme lookup. They are a declared
        regularization choice, not additional observations.

        **What the x-axis means.** “Part only” gives one learned mean per part.
        “+ visibility” learns separate means for the declared pixel-count bins.
        “+ exact values” additionally separates old and inserted values. “+ source
        species” additionally separates the unchanged source species. Each stage
        contains every field from the previous stage.

        **Outcome and prediction.** RMSE is the square root of the mean squared
        difference between predicted and observed final margins, in raw-logit
        units. Lower is better. A change from 4 to 3 means a typical held-out
        error reduction of roughly one logit unit; 3 to 4 is worsening and gives
        that added block no explanatory credit. The remaining error is a measured
        residual, not automatically an unknown causal mechanism.

        ### Figure 9 · Held-out margin prediction using progressively richer grouping information

        **How to read the figure.** Panel A gives the four held-out errors. Large
        points combine all folds; small translucent points diagnose whether the
        direction is confined to one fold and are not seed-level error bars.
        Lower is better. Panel B gives exact-group coverage: the percentage of
        held-out rows whose full grouping key was observed in the training folds.
        Falling coverage and small training groups demonstrate when a richer rule
        becomes sparse rather than merely asserting that explanation afterward.
        The tables print RMSE, MAE, coverage, group counts, split unit, and all
        fold-level diagnostics. Figure 6 already answers the separate descriptive
        visibility-selection question.
        """),
        code("fb-f9", r"""
        import hashlib
        A=S.copy(); A["vis_bin"]=pd.cut(A.pixel_count_cf,[-1,19,49,99,199,499,np.inf],labels=False)
        original_id_column=next((c for c in ["orig_render_id","source_render_id","li","image_orig","orig_image"] if c in A),None)
        if original_id_column is None:
            raise RuntimeError("Figure 9 requires an original source-image identity; swap-row render_id is not an independent split unit")
        unit=A[original_id_column].astype(str)
        A["fold"]=unit.map(lambda x:int(hashlib.sha1(x.encode()).hexdigest(),16)%5)
        if A.groupby(original_id_column).fold.nunique().max()!=1:
            raise RuntimeError("one original source image was assigned to more than one fold")
        stages=[("part only",["part"]),("+ visibility",["part","vis_bin"]),
                ("+ exact values",["part","vis_bin","var_src","var_donor"]),
                ("+ source species",["part","vis_bin","var_src","var_donor","sid_src"])]
        rows=[]; fold_rows=[]
        for stage,cols in stages:
            pred=pd.Series(index=A.index,dtype=float)
            for fold in range(5):
                tr=A[A.fold!=fold]; te=A[A.fold==fold]
                prior=tr.margin.mean(); stats=tr.groupby(cols).margin.agg(["mean","count"]).reset_index()
                stats["estimate"]=(stats["mean"]*stats["count"]+prior*10)/(stats["count"]+10)
                joined=te[cols].merge(stats[cols+["estimate"]],on=cols,how="left")
                matched=joined.estimate.notna().to_numpy()
                fold_prediction=joined.estimate.fillna(prior).to_numpy()
                pred.loc[te.index]=fold_prediction
                fold_rows.append({
                    "stage":stage,"fold":fold,"n_test_rows":len(te),
                    "heldout_group_coverage":float(matched.mean()),
                    "training_groups":len(stats),
                    "median_training_group_count":float(stats["count"].median()),
                    "rmse":float(np.sqrt(np.mean((te.margin.to_numpy()-fold_prediction)**2))),
                    "mae":float(np.mean(np.abs(te.margin.to_numpy()-fold_prediction))),
                })
            rows.append({"stage":stage,"rmse":float(np.sqrt(np.mean((A.margin-pred)**2))),
                         "mae":float(np.mean(np.abs(A.margin-pred)))})
        ACCOUNT=pd.DataFrame(rows)
        ACCOUNT_FOLDS=pd.DataFrame(fold_rows)
        coverage=(ACCOUNT_FOLDS.groupby("stage",sort=False).agg(
            mean_group_coverage=("heldout_group_coverage","mean"),
            min_group_coverage=("heldout_group_coverage","min"),
            median_training_group_count=("median_training_group_count","median"),
            mean_training_groups=("training_groups","mean")).reindex(ACCOUNT.stage).reset_index())
        ACCOUNT=ACCOUNT.merge(coverage,on="stage",how="left")
        ACCOUNT["split_unit"]=original_id_column
        ACCOUNT["n_original_images"]=unit.nunique()
        fig,axes=plt.subplots(1,2,figsize=(15,5.5))
        ax=axes[0]
        ax.plot(ACCOUNT.stage,ACCOUNT.rmse,"o-",color="#0072B2",lw=2)
        for fold,d in ACCOUNT_FOLDS.groupby("fold"):
            ax.scatter(d.stage,d.rmse,color="#0072B2",alpha=.28,s=22)
        ax.set_ylabel("held-out prediction error, RMSE (raw-logit units)")
        ax.tick_params(axis="x",rotation=20)
        ax.set_xlabel("information available to the post-hoc prediction rule")
        ax.set_title("A · Held-out error (large points=all folds)\nsmall points diagnose individual folds")
        for row in ACCOUNT.itertuples():
            ax.annotate(f"{row.rmse:.3f}",(row.stage,row.rmse),xytext=(0,8),textcoords="offset points",ha="center")
        ax=axes[1]
        ax.plot(ACCOUNT.stage,100*ACCOUNT.mean_group_coverage,"o-",color="#D55E00",lw=2)
        ax.set_ylim(0,105); ax.set_ylabel("held-out rows with a matching training group (%)")
        ax.tick_params(axis="x",rotation=20); ax.set_xlabel("same sequential information stages")
        ax.set_title("B · Exact-group coverage\nlow coverage reveals sparse lookup groups")
        for row in ACCOUNT.itertuples():
            ax.annotate(f"{100*row.mean_group_coverage:.1f}%",(row.stage,100*row.mean_group_coverage),
                        xytext=(0,8),textcoords="offset points",ha="center")
        fig.suptitle("Figure 9 · Can added information predict unseen final margins, and are its groups supported?")
        plt.tight_layout(); plt.show(); display(ACCOUNT.round(3))
        print("Fold-level diagnostic (folds are not training-seed error bars):")
        display(ACCOUNT_FOLDS.round(3))
        """, "Held-out final-margin prediction error when a post-hoc rule receives progressively richer FunnyBird grouping information."),
        figure_method("fb-m9", "We fitted a five-fold cross-validated, shrinkage-regularized group-mean lookup—not a neural network—while keeping all swaps from each original image in one fold, then calculated held-out RMSE after adding visibility, exact pair, and source species sequentially. Species were not held out, so this tests additional images of observed species and does not rule out species-category memorization."),
        review("fb-r9", "Figure 9"),

        md("fb-measurement-textbook", MEASUREMENT_TEXTBOOK),
        question("fb-q9b", "9b", "Synthesis only: do the already measured contributors line up with the controlled part ordering?",
                 "Place four separately defined part-level quantities in aligned panels: the controlled backwash-candidate rate, the same rate among swaps with at least 100 target pixels, the training label/mask conflict rate, and one minus exact donor-value recognition.",
                 "Tail should be high across several contributor panels while wing and foot should be low if the proposed explanation matches the controlled outcome. The panels use different units and must not be added together.",
                 "Use the same five-part order in every panel and print the exact table."),
        code("fb-f9b", r"""
        if "PART_CONFLICT" not in globals():
            raise RuntimeError("Figure 9b requires the matched standard/RLv2 label records used in Figure 6b")
        if "diag" not in globals():
            raise RuntimeError("Figure 9b requires the exact-value recognition results from Figure 7")
        FB_SYN=pd.DataFrame(index=ORDER)
        FB_SYN.index.name="part"
        FB_SYN["controlled_backwash_rate"]=(S.groupby("part").responded_but_source_wins.mean().reindex(ORDER))
        FB_SYN["clear_visible_backwash_rate"]=(S[S.pixel_count_cf>=100].groupby("part")
                                                  .responded_but_source_wins.mean().reindex(ORDER))
        FB_SYN["label_mask_conflict_rate"]=PART_CONFLICT.conflict_rate.reindex(ORDER)
        FB_SYN["donor_value_error_rate"]=pd.Series({p:1-diag[p] for p in ORDER}).reindex(ORDER)
        panels=[
            ("controlled_backwash_rate","A · OUTCOME: old source still wins"),
            ("clear_visible_backwash_rate","B · VISIBILITY CHECK: target ≥100 px"),
            ("label_mask_conflict_rate","C · DATA CHECK: label/mask conflict"),
            ("donor_value_error_rate","D · VISUAL DIFFICULTY: value misidentified"),
        ]
        fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
        for ax,(column,title) in zip(axes.flat,panels):
            ax.barh(np.arange(len(ORDER)),FB_SYN[column],color=[COLORS[p] for p in ORDER])
            ax.set_xlim(0,1); ax.set_title(title,fontsize=10); ax.set_xlabel("fraction")
            ax.set_yticks(np.arange(len(ORDER)),ORDER); ax.invert_yaxis()
            for y,value in enumerate(FB_SYN[column]):
                ax.text(value+.015,y,f"{value:.3f}",va="center",fontsize=8)
        fig.suptitle("Figure 9b · Synthesis of earlier measurements in one part order\n(no new causal evidence)")
        plt.tight_layout(); plt.show(); display(FB_SYN.round(3))
        """, "Four aligned FunnyBird part-level panels comparing the controlled backwash outcome with clear-visibility residuals, label/mask conflict, and exact donor-value error."),
        figure_method("fb-m9b", "We aligned four previously computed part-level summaries in one fixed order without fitting or adding them, so only their descriptive ordering—not percent explained—can be compared."),
        md("fb-r9b", r"""
        ### Plain-language reference for Figure 9b

        **Plain caption.** The observed part ordering lines up across controlled
        backwash, clearly visible swaps, training label/mask conflict, and exact-
        value error, but the four bars are not pieces that can be added.

        **Terms and denominators.** Panel A uses all 1,000 swaps per part. Panel B
        uses only swaps with at least 100 inserted-part pixels. Panel C divides
        removed positive training labels by all original positives. Panel D is
        one minus the inserted-value recognition rate. Every panel is a fraction,
        but the populations and questions differ.

        **Literal values.** Controlled event rates are tail `0.502`, beak
        `0.200`, eye `0.089`, foot `0.032`, and wing `0.019`. With target area at
        least 100 pixels they remain `0.372`, `0.131`, `0.052`, `0.017`, and
        `0.010`. Label/mask conflict rates are `0.199`, `0.010`, `0.007`,
        `0.001`, and less than `0.001`; donor-value error rates are `0.605`,
        `0.220`, `0.100`, `0.035`, and `0.023` in the same part order.

        **Interpretation.** The hardest controlled part is also the part with the
        most invisible-positive supervision and wrong exact-value answers. That
        agreement makes the proposed story plausible, but five correlated part
        summaries cannot measure how much each cause contributed.

        **Alternative.** Alignment can arise from correlated part properties,
        and there are only five anatomical units.

        **Discriminating test.** Use the same-source-image held-out prediction in Figure 9
        and the matched RLv2 intervention for label-conflict causality.

        **Verdict.** **KEEP**.

        **Proof ledger.** Visibility-resistant events, label conflict, and exact-
        value difficulty align descriptively with the controlled ordering. They
        are not additive and do not fully explain the residual.

        **Next question.** Does the concept-layer behavior have a meaningful
        downstream species-prediction consequence?
        """),

        question("fb-q10", "10", "Is final concept margin associated with donor-species probability?",
                 "Relate final concept margin to the model's donor-species probability, which is a different downstream quantity; this is an association, not an intervention on the margin.",
                 "A small downstream change would limit the harm to explanation reliability rather than widespread class failure.",
                 "Use independent final-margin bins and print bin counts."),
        md("fb-q10-plain", r"""
        **Concrete example.** Suppose a red tail is replaced by a blue tail.
        “Donor-positive concept margin” means the blue-tail raw score finishes
        above the red-tail raw score. The **donor species** is the complete species
        that supplied the blue tail. Its saved probability is the CBM class head's
        probability for that whole donor bird—not a probability that the tail is
        blue. Because the body and the other four parts still belong to the source
        bird, a correct blue-tail win need not make the complete donor species
        likely. This figure asks only whether the two quantities move together.
        """),
        code("fb-f10", r"""
        prob_col=next((c for c in ["p_cf_donor","p_donor_cf","donor_species_prob"] if c in S),None)
        if prob_col is None:
            print("INCOMPLETE: swap CSV has no donor-species probability column")
        else:
            D=S.copy(); D["margin_bin"]=pd.qcut(D.margin,10,duplicates="drop")
            Q=D.groupby("margin_bin",observed=True).agg(n=(prob_col,"size"),mean_margin=("margin","mean"),mean_donor_species_prob=(prob_col,"mean")).reset_index()
            fig,ax=plt.subplots(figsize=(7,4)); ax.plot(Q.mean_margin,Q.mean_donor_species_prob,"o-")
            for k,r in enumerate(Q.itertuples()):
                ax.annotate(f"n={r.n}",(r.mean_margin,r.mean_donor_species_prob),
                            fontsize=7,xytext=(3,8 if k%2==0 else -12),
                            textcoords="offset points")
            ax.axvline(0,color="black",lw=.8); ax.set_xlabel("mean final concept margin in bin")
            ax.set_ylabel("mean donor-species probability"); ax.set_title("Figure 10 · Association between concept margin and donor-species probability")
            plt.tight_layout(); plt.show(); display(Q.round(3))
        """, "Binned relationship between FunnyBird final concept margin and downstream donor-species probability."),
        figure_method("fb-m10", "We divided all swaps into ten disjoint equal-count bins by final concept margin and averaged the frozen CBM's saved donor-species probability within each bin; no regression or new classifier was fitted."),
        review("fb-r10", "Figure 10"),

        md("fb-conclusion", r"""
        ## 11 · Standard-CBM evidence ledger

        | Predicate or explanation | Direct measurement | Status after review |
        |---|---|---|
        | model outputs are usable | Figure 1 | `ACCEPTED FOR SEED-1 MODEL HEALTH` |
        | interventions are valid | Figure 2 | `ACCEPTED FOR CONTROLLED ONE-PART REPLACEMENT` |
        | controlled part replacement causes donorward movement | Figure 3 | `ACCEPTED AT PART LEVEL; POSITIVE-RESPONSE RATES 0.919-1.000, NOT EVERY ROW` |
        | starting preference versus donor rise/source release | Figure 3b | `ACCEPTED ARITHMETIC DECOMPOSITION` |
        | old source can remain stronger after that movement | Figure 4 | `ACCEPTED FOR GRADED CONTROLLED BACKWASH` |
        | donor wins versus two distinct failure states | Figure 4b | `ACCEPTED OUTCOME PARTITION` |
        | direction artifact excluded | Figure 5 | `ACCEPTED; ORDERING HOLDS BOTH DIRECTIONS` |
        | visibility contribution | Figure 6 | `ACCEPTED AS CONTRIBUTOR, NOT SUFFICIENT` |
        | training label/mask conflict measured | Figure 6b | `ACCEPTED DATA ASSOCIATION; CAUSAL TEST IS 02RL` |
        | exact-value difficulty | Figure 7 | `ACCEPTED AS GRADED ASSOCIATION/CANDIDATE CONTRIBUTOR` |
        | frequency/alternative-count explanation | Figure 7b | `MIXED; NO SUFFICIENT MONOTONE EXPLANATION` |
        | source-species residual | Figure 8 | `DESCRIPTIVE ASSOCIATION ONLY` |
        | species information beyond concept-label buckets | Figure 8b | `ACCEPTED FOR AVAILABILITY, NOT GROUNDING` |
        | equal-width information and saved-head magnitude use | Figure 8c | `READ-ONLY TEST; INTERPRET EXECUTED VALUES, NOT DECODING ALONE` |
        | post-swap off-target source evidence and direct erasure | Figure 8d | `READ-ONLY DOWNSTREAM INTERVENTION; DOES NOT CAUSE THE UPSTREAM CONCEPT MARGIN` |
        | progressively richer held-out grouping predictor | Figure 9 | `VISIBILITY IMPROVES HELD-OUT ERROR; EXACT VALUE/SPECIES DO NOT` |
        | aligned contributor view | Figure 9b | `ACCEPTED DESCRIPTIVELY; NOT ADDITIVE OR CAUSAL` |
        | downstream class association | Figure 10 | `ACCEPTED FOR MODEST MONOTONE ASSOCIATION; NOT A MARGIN INTERVENTION` |

        ### Limited conclusion

        **Backwash exists in this seed-1 Standard CBM.** It is not necessary for
        all parts to fail identically: the controlled predicate is row-level,
        and its prevalence is graded from tail through wing. The renderer targets
        one declared part while preserving the scene; 98.3% of cached replacement
        RGB images visibly differ from their originals, while the remaining 1.7%
        are retained for the visibility analysis. At the part-summary level,
        positive donorward-response rates range from 91.9% to 100%; this is not a
        claim that every individual swap responds. Within the measured subset that
        responds but finishes with a negative margin, the final concept answer
        nevertheless remains attached to the old source.

        The proposed contributors and alternatives were investigated. Visibility
        accounts for some held-out organization but leaves many clearly visible
        tail events. Label/mask conflict and exact-value error closely match the
        part ordering, with tail highest and wing/foot lowest, but their current
        standard-model analyses are associations. Rarity/support is mixed.
        Source species strongly appears in the learned concept representation
        and in descriptive residuals. Figure 8b asks what species information a
        newly fitted diagnostic can recover. Figure 8c controls the number of
        supplied coordinates and then asks whether the unchanged saved species
        head actually uses within-label magnitudes. Figure 8d moves to the accepted
        swaps: it computes the source-over-donor class evidence contributed by the
        off-target scores, resets only those scores to ordinary absent baselines,
        and reruns the same frozen head. The executed values determine whether this
        downstream fingerprint meaningfully distinguishes tail from wing. The old
        weak within-part correlation is retained only in the methods appendix.
        Even a positive direct-erasure result would establish downstream use, not
        that the species head caused the upstream concept-margin failure.
        Therefore the evidence does **not** support saying that backwash is fully
        explained or that the measured contributors exhaust every causal pathway.

        ### Why the official MCBM `-3/+3` targets are not automatically the answer

        This Standard CBM does not use an MCBM representation target. Notebook 03
        asks that separate numerical question. The pinned MCBM source targets
        `-3` for an absent concept and `+3` for a present concept. In an ideal
        red-to-blue replacement, the donor coordinate would move `-3 -> +3`
        (`+6`) while the removed source coordinate moves `+3 -> -3` (another
        `+6` contribution to donor-minus-source margin), for total margin movement
        `+12`. A hypothetical `-5/+5` target would analogously move the margin by
        `+20`, not `+10`. But a representation penalty does not specify which
        pixels must produce the target. A network can recognize species/body and
        output the correct target on ordinary training images. Achieving the
        desired counterfactual movement may still require swap-aware or spatial
        grounding supervision; compression alone does not guarantee it.

        ### Explicit handoff through the decreasing-information chapters

        | Requested follow-up | Where it belongs | What it must establish |
        |---|---|---|
        | Standard MCBM gamma sweep | notebook 03 | whether compression removes within-label species information and reduces the same fixed-swap event while health remains usable |
        | relabelled CBM | notebook 02rl | causal effect of changing positive labels when the named part is invisible |
        | relabelled MCBM | notebook 03rl | whether minimality and relabelling address distinct or overlapping routes |
        | CUB70 Standard CBM/MCBM | notebooks 05/06 | whether FunnyBird-calibrated visibility, conflict, exact-value, support, and species signatures recur observationally in photographs |
        | Full CUB | final 200-species stage | whether supported CUB70 signatures survive realistic scale and which matched questions lose adequate support |
        | segmentation or spatial routing | later method decision | only pursue if the accepted evidence requires an explicit named-pixel path constraint |

        The controlled FunnyBird swap remains the definition-quality base case
        and calibration laboratory. CUB70 is the natural-image bridge: it can
        repeat scientific questions using natural visibility, mask conflict,
        exact-concept difficulty, support, and species-conditioned raw scores,
        but those are weaker observational approximations—not donor/source
        margins. Full CUB then asks whether the surviving signatures remain
        detectable with 200 species and weaker matched support. As information
        decreases, the claim must narrow rather than the metric being silently
        weakened.

        **Next report question.** Only after this ledger is reviewed may notebook
        03 ask whether MCBM minimality changes the accepted standard-CBM
        quantities. MCBM cannot replace this discovery.
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
    # Remove the earlier multi-figure fingerprint/saved-head detour. The replacement
    # separates information availability, saved-head use, and a direct downstream
    # erasure at the replaced concept block. MCBM remains in notebook 03.
    removed_prefixes=(
        "fb-q8c-","fb-f8c-","fb-m8c-","fb-r8c-",
        "fb-q8d-","fb-f8d-","fb-m8d-","fb-r8d-",
        "fb-q8e-","fb-f8e-","fb-m8e-","fb-r8e-",
        "fb-mcbm-bridge-","fb-f8f-","fb-m8f-","fb-r8f-",
    )
    cells=[cell for cell in cells if not cell["id"].startswith(removed_prefixes)]
    q9_index=next(i for i,cell in enumerate(cells) if cell["id"].startswith("fb-q9-"))
    cells[q9_index:q9_index]=funnybird_source_retention_cells()
    appendix_index=next(i for i,cell in enumerate(cells) if cell["id"].startswith("fb-appendix-"))
    cells[appendix_index:appendix_index]=funnybird_source_retention_appendix_cells()
    return notebook(cells, NOTEBOOKS/"02_funnybirds_cbm.ipynb", preserve_outputs)


def build_cub(preserve_outputs: bool = False) -> dict:
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
        from cub70_parts import CUB70_PARTS, ATTRIBUTE_TYPE_TO_MASK, COARSE_TO_CUB70
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
        mask_map=[]
        for group in COARSE_ORDER:
            families=sorted(k for k,v in ATTRIBUTE_TYPE_TO_MASK.items() if v==group)
            mask_map.append({"analysis_group":group,
                             "released_mask_sources":", ".join(COARSE_TO_CUB70[group]),
                             "mapped_attribute_families":", ".join(families)})
        MASK_MAP=pd.DataFrame(mask_map)
        display(inventory); display(fine.round(4)); display(MASK_MAP)
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
                 "On the same held-out split, decode species from each raw-logit block and from the corresponding processed 0/1 label block; also show the saved CBM's own task accuracy.",
                 "A bar height is the fraction of held-out photographs whose species a separate diagnostic classifier guesses correctly. This measures recoverability from the supplied numbers, not grounding and not the saved CBM's task accuracy.",
                 "Build one image-by-concept matrix and use a fixed stratified 70/30 split."),
        md("cub-f2b-explain", r"""
        ### Before Figure 4b: what exactly are the grey and colored bars?

        Each photograph has 112 processed binary labels `c`. A small portion of one
        row might read `black bill=1`, `grey bill=0`, `striped tail=1`. This complete
        row is the photograph's **attribute pattern**: its collection of known yes/no
        concept answers after preprocessing.

        We train two separate diagnostic classifiers after the CBM is finished:

        - **grey bar:** the classifier receives known 0/1 labels `c`;
        - **colored bar:** the classifier receives learned raw scores `z`;
        - **bar height:** the fraction of held-out photographs whose species it guesses.

        For example, grey `complete = 1.0` means this diagnostic classifier identified
        every held-out species correctly from all 112 known yes/no answers. It does not
        mean the saved CUB70 CBM has 100% task accuracy; that separate accuracy is the
        dotted line.

        “Species information” therefore means **species is statistically recoverable
        from these numbers**. It does not identify the responsible pixels. A part block
        can reveal species and still be well grounded, so later visibility/context tests
        remain necessary.

        > **IMPORTANT: Species leakage makes backwash possible, but leakage alone does
        > not cause it. FunnyBird wing proves this: wing `z` reveals species while its
        > controlled swaps remain strongly grounded. CUB has no equivalent controlled
        > swap, so this figure cannot rank CUB grounding.**
        """),
        code("cub-f2b", r"""
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        X=E70.pivot_table(index="image",columns="concept_name",values="z",aggfunc="first")
        C=E70.pivot_table(index="image",columns="concept_name",values="gt_label",aggfunc="first").loc[X.index,X.columns]
        image_rows=E70[["image","y_true","y_pred"]].drop_duplicates().set_index("image").loc[X.index]
        y=image_rows.y_true
        saved_task_accuracy=float((image_rows.y_true==image_rows.y_pred).mean())
        tr,te=train_test_split(np.arange(len(X)),test_size=.30,random_state=20260803,stratify=y)
        cmap=E70[["concept_name","mask_group"]].drop_duplicates().set_index("concept_name").mask_group
        blocks={"complete z":list(X.columns)}
        blocks.update({g:[c for c in X.columns if cmap.get(c)==g] for g in COARSE_ORDER})
        rows=[]
        for name,cols in blocks.items():
            if not cols: continue
            raw_model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=4000,C=1.0,random_state=20260803))
            label_model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=4000,C=1.0,random_state=20260803))
            raw_model.fit(X.iloc[tr][cols],y.iloc[tr]); label_model.fit(C.iloc[tr][cols],y.iloc[tr])
            rows.append({"block":name,
                         "raw_z_accuracy":accuracy_score(y.iloc[te],raw_model.predict(X.iloc[te][cols])),
                         "processed_label_accuracy":accuracy_score(y.iloc[te],label_model.predict(C.iloc[te][cols])),
                         "dimensions":len(cols)})
        SPECIES_PROBE=pd.DataFrame(rows)
        x=np.arange(len(SPECIES_PROBE)); w=.36; fig,ax=plt.subplots(figsize=(11,5))
        ax.bar(x-w/2,SPECIES_PROBE.processed_label_accuracy,w,label="known 0/1 label probe",color="#BBBBBB")
        ax.bar(x+w/2,SPECIES_PROBE.raw_z_accuracy,w,label="learned raw-z probe",
               color=["#333333"]+[COLORS.get(x,"#BBBBBB") for x in SPECIES_PROBE.block.iloc[1:]])
        ax.set_xticks(x); ax.set_xticklabels(SPECIES_PROBE.block,rotation=30,ha="right")
        ax.axhline(1/y.nunique(),color="black",ls="--",label="chance = 1/70")
        ax.axhline(saved_task_accuracy,color="#D55E00",ls=":",label=f"saved CUB70 CBM task accuracy = {saved_task_accuracy:.3f}")
        ax.set_ylim(0,1)
        ax.set_ylabel("held-out species accuracy"); ax.set_title("Figure 4b · Species decoded from CUB70 raw concept logits")
        ax.legend(); plt.tight_layout(); plt.show(); display(SPECIES_PROBE.round(3))
        """, "Held-out CUB70 species-decoding accuracy from raw concept logits versus corresponding processed labels, with blind chance and saved-model task accuracy."),
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

        md("cub-measurement-textbook", MEASUREMENT_TEXTBOOK),
        question("cub-q11a", "11a", "Which exact CUB concepts carry each measured problem?",
                 "Align the same exact-concept rows across mask absence, ordinary concept error, visible-minus-hidden raw-z difference, hidden context gap, and within-concept species residual spread.",
                 "If one anatomical family repeatedly contains the largest values, its coarse ranking reflects consistent exact concepts. Mixed rows show that coarse aggregation hides value-specific behavior.",
                 "Retain all mask-testable exact concepts; leave unsupported measurements blank and show the exact denominators in the table."),
        code("cub-f11a", r"""
        CUB_GROUP_ORDER=["tail","wing","beak","leg","eye","neck","body","head"]
        collapsed_names=set(HEALTH.loc[HEALTH.collapsed,"concept_name"])
        species_exact=(SP.groupby(["mask_group","concept_name"]).agg(
            species_residual_sd=("residual","std"),n_species_cells=("residual","size")).reset_index())
        health_exact=HEALTH[["concept_name","balanced_accuracy","collapsed","n_positive","n_negative"]].copy()
        health_exact=health_exact.rename(columns={"n_positive":"health_n_positive","n_negative":"health_n_negative"})
        health_exact["classification_error"]=1-health_exact.balanced_accuracy
        CUB_EXACT_SYN=(EXACT.merge(health_exact,on="concept_name",how="left")
            .merge(species_exact,on=["mask_group","concept_name"],how="left"))
        CUB_EXACT_SYN.loc[CUB_EXACT_SYN.concept_name.isin(collapsed_names),
                          ["classification_error","visibility_effect","context_gap","species_residual_sd"]]=np.nan
        CUB_EXACT_SYN.loc[(CUB_EXACT_SYN.health_n_positive<10)|(CUB_EXACT_SYN.health_n_negative<10),
                          "classification_error"]=np.nan
        CUB_EXACT_SYN.loc[(CUB_EXACT_SYN.n_visible<10)|(CUB_EXACT_SYN.n_hidden<10),
                          "visibility_effect"]=np.nan
        CUB_EXACT_SYN.loc[(CUB_EXACT_SYN.n_hidden<10)|(CUB_EXACT_SYN.n_hidden_negative<10),
                          "context_gap"]=np.nan
        CUB_EXACT_SYN.loc[CUB_EXACT_SYN.n_species_cells<10,"species_residual_sd"]=np.nan
        CUB_EXACT_SYN["group_order"]=CUB_EXACT_SYN.mask_group.map({g:i for i,g in enumerate(CUB_GROUP_ORDER)})
        CUB_EXACT_SYN=CUB_EXACT_SYN.sort_values(
            ["group_order","context_gap","concept_name"],ascending=[True,False,True]).reset_index(drop=True)
        y=np.arange(len(CUB_EXACT_SYN)); row_colors=CUB_EXACT_SYN.mask_group.map(COLORS).fillna("#888888")
        panels=[
            ("label_mask_conflict","A · DATA CHECK: label / mask absent",(0,1)),
            ("classification_error","B · HEALTH CHECK: concept error",(0,1)),
            ("visibility_effect","C · LOCAL EVIDENCE: visible − hidden z",None),
            ("context_gap","D · CONTEXT: hidden positive − negative z",None),
            ("species_residual_sd","E · SPECIES CONTEXT: residual spread",None),
        ]
        fig,axes=plt.subplots(1,5,figsize=(20,max(18,.225*len(CUB_EXACT_SYN))),sharey=True)
        for ax,(column,title,limits) in zip(axes,panels):
            d=CUB_EXACT_SYN[column].notna()
            ax.scatter(CUB_EXACT_SYN.loc[d,column],y[d],c=row_colors[d],s=20)
            if column in ["visibility_effect","context_gap"]: ax.axvline(0,color="black",lw=.8)
            if limits: ax.set_xlim(*limits)
            ax.set_title(title,fontsize=10); ax.grid(axis="x",alpha=.2)
        axes[0].set_yticks(y); axes[0].set_yticklabels(CUB_EXACT_SYN.concept_name,fontsize=6)
        axes[0].invert_yaxis()
        boundaries=CUB_EXACT_SYN.groupby("mask_group",sort=False).size().cumsum().iloc[:-1]-0.5
        for ax in axes:
            for boundary in boundaries: ax.axhline(boundary,color="#BBBBBB",lw=.7)
        fig.suptitle("Figure 11a · Exact CUB concepts aligned across measurement, health, context, and species questions", y=.998)
        plt.tight_layout(rect=[0,0,1,.985]); plt.show()
        display(CUB_EXACT_SYN[["mask_group","attribute_type","concept_name","n_positive","n_hidden",
            "label_mask_conflict","health_n_positive","health_n_negative","classification_error","n_visible","visibility_effect",
            "n_hidden_negative","context_gap","n_species_cells","species_residual_sd"]].round(3))
        """, "Five aligned panels retaining every mask-testable exact CUB concept and showing unsupported quantities as missing rather than zero."),

        question("cub-q11b", "11b", "How are the available CUB contributors distributed across coarse anatomical groups?",
                 "Use the same anatomical order wherever possible and report five distinct quantities: positive-label/mask-absence rate, median exact-concept classification difficulty, median natural visibility effect, median hidden context gap, and species-residual spread.",
                 "If CUB behaves like a simple diluted copy of FunnyBird, the same groups should repeatedly rank as difficult. If rankings differ, the contributors are distributed across concepts and cannot be reduced to one tail-to-wing grounding order.",
                 "Do not sum the panels, do not call any panel a CUB donor/source margin, and print eligibility counts."),
        code("cub-f11b", r"""
        CUB_GROUP_ORDER=["tail","wing","beak","leg","eye","neck","body","head"]
        collapsed_names=set(HEALTH.loc[HEALTH.collapsed,"concept_name"])
        conflict_group=(EXACT.groupby("mask_group").agg(n_hidden=("n_hidden","sum"),n_positive=("n_positive","sum")))
        conflict_group["label_mask_absence_rate"]=conflict_group.n_hidden/conflict_group.n_positive.replace(0,np.nan)
        difficulty=(HEALTH[(~HEALTH.collapsed)&(HEALTH.n_positive>=10)&(HEALTH.n_negative>=10)].groupby("mask_group").agg(
            median_balanced_accuracy=("balanced_accuracy","median"),n_health_concepts=("concept_name","nunique")))
        difficulty["median_classification_error"]=1-difficulty.median_balanced_accuracy
        eligible=EXACT[~EXACT.concept_name.isin(collapsed_names)].copy()
        vis=(eligible[(eligible.n_visible>=10)&(eligible.n_hidden>=10)].groupby("mask_group")
             .agg(median_visibility_effect=("visibility_effect","median"),n_visibility_concepts=("concept_name","nunique")))
        ctx=(eligible[(eligible.n_hidden>=10)&(eligible.n_hidden_negative>=10)].groupby("mask_group")
             .agg(median_context_gap=("context_gap","median"),n_context_concepts=("concept_name","nunique")))
        species=(SP.groupby("mask_group").agg(species_residual_sd=("residual","std"),
                                                n_species_cells=("residual","size")))
        CUB_SYN=(conflict_group.join(difficulty,how="outer").join(vis,how="outer")
                 .join(ctx,how="outer").join(species,how="outer").reindex(CUB_GROUP_ORDER))
        panels=[
            ("label_mask_absence_rate","A · DATA CHECK: label / mask absent",(0,1)),
            ("median_classification_error","B · HEALTH CHECK: concept error",(0,1)),
            ("median_visibility_effect","C · LOCAL EVIDENCE: visible − hidden z",None),
            ("median_context_gap","D · CONTEXT: hidden positive − negative z",None),
            ("species_residual_sd","E · SPECIES CONTEXT: residual spread",None),
        ]
        fig,axes=plt.subplots(1,5,figsize=(19,4.8),sharey=True)
        colors=[COLORS.get(g,"#888888") for g in CUB_GROUP_ORDER]
        for ax,(column,title,limits) in zip(axes,panels):
            ax.barh(np.arange(len(CUB_GROUP_ORDER)),CUB_SYN[column],color=colors)
            if column in ["median_visibility_effect","median_context_gap"]: ax.axvline(0,color="black",lw=.8)
            if limits: ax.set_xlim(*limits)
            ax.set_title(title,fontsize=9); ax.set_yticks(np.arange(len(CUB_GROUP_ORDER)),CUB_GROUP_ORDER)
            ax.invert_yaxis()
        fig.suptitle("Figure 11b · CUB observational contributors by coarse anatomical group; no controlled backwash outcome")
        plt.tight_layout(); plt.show(); display(CUB_SYN.round(3))
        from IPython.display import Markdown
        def rank_text(column,ascending=False):
            return " > ".join(CUB_SYN[column].dropna().sort_values(ascending=ascending).index.tolist())
        display(Markdown(
            "**How to read this comparison.** Each panel answers a different question and "
            "uses different units. Larger values mean more mask disagreement in A, worse "
            "ordinary concept classification in B, a larger visible-minus-hidden association "
            "in C, more separation without the mapped mask in D, and more species-to-species "
            "variation after concept/visibility centering in E. None is a CUB swap failure rate.\n\n"
            f"**Observed rank orders after coarse grouping.** Mask absence: {rank_text('label_mask_absence_rate')}. "
            f"Concept error: {rank_text('median_classification_error')}. "
            f"Hidden context gap: {rank_text('median_context_gap')}. "
            f"Species residual spread: {rank_text('species_residual_sd')}.\n\n"
            "**Limited conclusion.** Agreement across several panels would identify repeatedly "
            "affected groups. Disagreement means CUB's contributors are distributed rather than "
            "conveniently concentrated in one part. Even agreement cannot create the missing "
            "controlled CUB backwash outcome."
        ))
        """, "Five aligned coarse-group CUB panels showing mask disagreement, concept difficulty, natural visibility association, hidden context separation, and species residual variation without inventing a swap outcome."),

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
        | Do training labels cause part of it? | conflict measured; matched CBM-RLv2 belongs to notebook 02rl | no accepted CUB retraining | no | no causal label conclusion in either standard-CBM report |

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
        | species information beyond processed-label structure | Figure 4b | `INCOMPLETE: PAIRED RAW-Z/LABEL CONTROL REQUIRES REVIEW` |
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
    return notebook(cells, NOTEBOOKS/"05_cub_cbm.ipynb", preserve_outputs)


def write(name: str, obj: dict) -> None:
    path = NOTEBOOKS/name
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    print(f"wrote {path}: {len(obj['cells'])} cells")


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--only",choices=["02","05"])
    ap.add_argument("--preserve-outputs",action="store_true",
                    help="retain outputs from code cells whose stable IDs match")
    args=ap.parse_args()
    if args.only in (None,"02"): write("02_funnybirds_cbm.ipynb",build_funnybird(args.preserve_outputs))
    if args.only in (None,"05"): write("05_cub_cbm.ipynb",build_cub(args.preserve_outputs))


if __name__=="__main__":
    main()
