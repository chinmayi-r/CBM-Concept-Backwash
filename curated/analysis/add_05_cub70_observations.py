#!/usr/bin/env python3
"""Insert inspected, simple-language observations into executed notebook 05.

This script changes markdown only and preserves every executed code output.
It is idempotent: observation cells are replaced using stable metadata IDs.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from textwrap import dedent


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "05_cub_cbm.ipynb"


def lines(text: str) -> list[str]:
    return (dedent(text).strip("\n") + "\n").splitlines(keepends=True)


def observation(obs_id: str, text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": "cub05-obs-" + hashlib.sha1(obs_id.encode("utf-8")).hexdigest()[:10],
        "metadata": {"cub05_observation_id": obs_id},
        "source": lines(text),
    }


OBSERVATIONS = [
    (
        'concepts=(E70[["concept_name","attribute_type","mask_group"]].drop_duplicates()',
        observation("inventory", r"""
        ### What Figure 1 says

        **Literal observation.** The model has 112 selected concept values spread across
        all 28 CUB attribute types. Twenty-six types can be linked to a released mask.
        The five individual values belonging to whole-bird `size` and `shape` cannot be
        given an honest local mask, so 107 exact concepts remain for local analysis.

        Breast, belly, back, upperparts, and underparts remain separate concept types,
        even though all five must use the same coarse body mask. Eye, wing, and leg
        concepts are not left/right concepts, even though their masks are lateralized.

        **Alternative explanation to guard against.** Pooling all body concepts or all
        wing concepts now would hide real differences between color, pattern, and shape.

        **Limited conclusion.** CUB must be analysed first at exact-concept and
        28-type resolution. The available mask is recorded as a limitation, not treated
        as the definition of the concept.

        **Next question.** Which species and photographs actually have masks?
        """),
    ),
    (
        'all_images=E70[["image","y_true"]].drop_duplicates()',
        observation("population", r"""
        ### What Figure 2 says

        **Literal observation.** The CUB70 prediction export contains 1,976 photographs
        from 70 species. Only 1,888 photographs join to the released masks, and those
        photographs cover 67 model species. Covered species contribute 11–30 images,
        with a median of 30.

        **Odd result.** Three predicted species disappear completely after the mask
        join. This may be an archive-coverage or filename-alignment issue; it is not a
        model result.

        **Limited conclusion.** Every visibility result below describes the 1,888-image,
        67-species masked population, not the complete 70-species prediction set.

        **Next question.** Within that population, which fine masks are usually visible?
        The missing-species identities must also be audited before claiming full CUB70
        coverage.
        """),
    ),
    (
        'fine=(RAWVIS.groupby("part").agg(',
        observation("fine_visibility", r"""
        ### What Figure 3 says

        **Literal observation.** Head (99.3%), body (97.8%), and beak (93.0%) are almost
        always visible. Tail is visible in 81.1% of photographs. Each wing is visible in
        about 59.8%, and each leg in about 64.7%. Each eye and the neck are visible in
        only about 22–23%.

        For bilateral regions, neither eye is visible in 54.8% of photographs and both
        eyes are visible in only 14 photographs. Wings have useful zero/one/two-side
        variation (15.5%/49.4%/35.1%). Legs most often show both sides (54.9%).

        **Alternative explanation.** A missing mask can mean occlusion, viewpoint, or an
        annotation limitation. Mask area is also not identical to diagnostic color or
        pattern evidence.

        **Limited conclusion.** Natural visibility provides a strong test opportunity
        for eye and neck, a moderate opportunity for wing/leg/tail, and very little
        hidden-case evidence for head/body/beak.

        **Next question.** Before looking at predictions, how strongly does each exact
        concept already identify a subset of species?
        """),
    ),
    (
        'species_concept=(E70.groupby(["attribute_type","concept_name","y_true"]).gt_label.mean()',
        observation("species_support", r"""
        ### What Figure 4 says

        **Literal observation.** Exact concepts vary from being positive in almost no
        species to being positive in 63 of 70. Eye color is shared by 63 species, while
        many color/pattern values occur in only a few species. This variation exists
        across body, head, beak, tail, wing, and leg types.

        **Why this is important.** A rare concept is a stronger clue to species identity
        than a widely shared concept. That creates different shortcut opportunities for
        different exact concepts, even inside the same anatomical region.

        **Limited conclusion.** Species–concept structure is widespread and cannot be
        summarized as a tail-only property. This graph does not prove that the model
        reads species pixels.

        **Next question.** When these species-linked labels are positive, how often is
        the corresponding mask actually absent?
        """),
    ),
    (
        "EXACT70=exact_visibility_metrics(J70)",
        observation("label_mask", r"""
        ### What Figure 5 says

        **Literal observation.** Label/mask disagreement is largest for throat colors
        (roughly 70–97%) and eye color (51%). Leg colors range roughly 21–46%.
        Some individual tail concepts reach 87%, while other tail concepts are much
        lower. Wing concepts range from near zero to roughly 33%. Most body, head, and
        beak concepts have low disagreement because their masks are usually visible.

        **Alternative explanation.** A coarse body or head mask cannot certify that the
        smaller named region—belly, breast, crown, or nape—is visible. Low measured
        disagreement for those types may therefore be falsely reassuring.

        **Limited conclusion.** Visibility-conflicting supervision is a plausible risk
        for throat, eye, leg, and selected tail/wing concepts. It is not uniformly a
        tail problem, and this data count does not yet show what the model learned.

        **Next question.** Did the CUB70-trained CBM learn usable, noncollapsed concept
        outputs before visibility is interpreted?
        """),
    ),
    (
        "display(task_and_concept_accuracy(E70).round(4))",
        observation("model_health", r"""
        ### What Figure 6 says

        **Literal observation.** On its native 1,976-image test set, the CUB70-trained
        CBM has 14.1% species accuracy and 71.1% overall concept accuracy. Median positive
        recall varies substantially across attribute types. Most exact concepts show a
        broad score distribution, but at least one throat-color output has only one
        distinct rounded score and one wing-pattern output has only two.

        **Odd result.** High or low thresholded recall can be produced by a nearly
        constant score. Those collapsed exact concepts cannot count as grounded merely
        because their score falls on one side of 0.5.

        **Limited conclusion.** The model learned many nonconstant concepts, but it is a
        weak classifier and contains some failed concept slots. Every later claim must
        keep the collapse guard and ordinary performance visible.

        **Next question.** For the non-global concepts, does seeing the available named
        mask increase the prediction?
        """),
    ),
    (
        'T=(EXACT70.dropna(subset=["prob_visible","prob_hidden"])',
        observation("visible_hidden", r"""
        ### What Figure 7 says

        **Literal observation.** Exact concepts move in both directions. Eye color,
        wing color, bill color, underparts color, and several body colors are higher on
        visible photographs. Leg color and many tail measurements are nearly flat.
        Wing pattern and several head-related types are lower when their mask is visible.

        The head result is especially fragile because the head mask is absent in only
        about 0.7% of photographs. It cannot carry the same weight as eye or wing results.

        **Alternative explanations.** Visible and hidden groups can contain different
        species and poses. Some exact concepts are weak or collapsed. One shared body
        mask is also a poor local test for five smaller body regions.

        **Limited conclusion.** CUB shows selective visual sensitivity, not one common
        “part visibility” effect. A flat line does not prove occlusion was irrelevant:
        training-time label conflict could have taught a contextual rule that is now
        insensitive to test-time visibility.

        **Next question.** Do concepts with more label/mask conflict show more hidden-part
        model violations?
        """),
    ),
    (
        "X=EXACT70.merge(support,on=[\"attribute_type\",\"concept_name\"],how=\"left\")",
        observation("conflict_violation", r"""
        ### What Figure 8 says

        **Literal observation.** There is no simple one-to-one pattern. Concepts with
        little measured label conflict can still have high hidden-part prediction, and
        high-conflict throat or tail concepts range from moderate to nearly complete
        hidden prediction.

        **Alternative explanation.** The y-axis uses a 0.5 threshold, so compressed or
        collapsed scores can create extreme rates. Species support and mask quality also
        differ across points.

        **Limited conclusion.** Label/mask conflict remains a plausible cause, but it is
        not sufficient by itself to explain which exact concepts remain positive while
        hidden.

        **Next question.** Ignoring the binary threshold, does a larger visible region
        continuously strengthen the concept score?
        """),
    ),
    (
        "dose=[]",
        observation("area_dose", r"""
        ### What Figure 9 says

        **Literal observation.** Eye color, wing color, and primary-feather color tend
        to rise from the smallest to largest visible-area quartile. Many body and head
        concepts contain both positive and negative exact values. Wing shape, leg color,
        and bill length do not show a consistent positive area response.

        **Alternative explanation.** More pixels do not necessarily mean that the
        diagnostic color, edge, or pattern is clearer. Area also changes with pose and
        distance.

        **Limited conclusion.** Visible area matters for selected concept types, not as
        a universal grounding law.

        **Next question.** For bilateral regions, can zero, one, and two visible sides
        give a more direct visibility dose without inventing left/right concepts?
        """),
    ),
    (
        'B=J70[(J70.gt_label==1)&J70.mask_group.isin(["eye","wing","leg"])].copy()',
        observation("bilateral", r"""
        ### What Figure 10 says

        **Literal observation.** Eye-color probability rises from 0.58 with no visible
        eye to 0.72 with one visible eye. The two-eye estimate is based on only 14 rows
        and is not dependable. Wing shape rises from zero to two visible wings. Wing
        color is nearly flat, primary color changes little, and wing pattern decreases.
        Leg color is flat across zero, one, and two visible legs.

        Left-only and right-only means differ modestly for wing color and primary color,
        warning that viewpoint or side composition matters. These are not lateralized
        concept labels.

        **Limited conclusion.** Natural visibility evidence is convincing for eye color
        from zero to one side and suggestive for wing shape. Other bilateral concept
        types do not follow the same pattern.

        **Next question.** Does the apparent visibility effect remain when the species
        and exact concept are held fixed?
        """),
    ),
    (
        "MATCH70=matched_effects(J70)",
        observation("species_matched", r"""
        ### What Figure 11 says

        **Literal observation.** After comparing within the same species and exact
        concept, every attribute type still contains positive and negative groups.
        Several body-color types have positive median effects around 0.10–0.14. Eye
        color is only slightly positive (median about 0.02). Tail types are mostly near
        zero. Wing color is near zero, wing shape slightly positive, primary color
        slightly negative, and wing pattern near zero. Nape and crown remain slightly
        negative but have few matched groups.

        **What changed.** Several large pooled differences became much smaller after
        species matching. Species composition therefore explained part of the earlier
        visible-versus-hidden comparison.

        **Limited conclusion.** Some body-region sensitivity remains observationally,
        but there is no single CUB region with a universal response. Pose, viewpoint,
        and background remain uncontrolled within species.

        **Next question.** Is the remaining movement specific to positive concepts, or
        does visibility move negative outputs in the same way?
        """),
    ),
    (
        'for (t,lab),d in J70.groupby(["attribute_type","gt_label"]):',
        observation("specificity", r"""
        ### What Figure 12 says

        **Literal observation.** Several body-color types show the desired specific
        pattern: positive-label scores rise while negative-label scores fall. Eye scores
        rise for both positive and negative labels, which looks more like a general
        visibility/pose signal than a clean eye-color-specific effect. Head-related
        positive and negative rows often move in opposite, unexpected directions.
        Wing types remain mixed: wing shape is positive-specific, while wing pattern is
        negative for positive labels.

        **Limited conclusion.** The strongest current specificity evidence belongs to
        selected body colors and wing shape, not to every concept sharing those masks.
        Eye visibility affects the model, but this plot does not show that it affects
        only the correct eye-color concept.

        **Next question.** Does training the same CBM architecture on 70 rather than 200
        species improve these exact-concept behaviors on identical masked images?
        """),
    ),
    (
        'print("Native test populations (reported separately; not a direct accuracy comparison):")',
        observation("direct_comparison", r"""
        ### What Figures 13 and 14 say

        **Identity check.** Both CBMs are compared on exactly the same 1,888 masked image
        identities and the same 107 locally maskable exact concepts. Native full-test
        accuracies are printed separately and are not used as a direct comparison.

        **Literal observation.** On the identical masked population, the CUB70-trained
        CBM has 14.25% task accuracy and 70.80% concept accuracy; the full-CUB-trained
        CBM has 33.10% and 84.05%. Many CUB70 hidden-violation points fall below the
        equality line, but others rise above it or become exactly zero/one. CUB70 score
        spreads are usually smaller, with some collapsed points. Species-matched
        visibility effects show weak agreement and several large outliers rather than a
        uniform improvement.

        **Alternative explanation.** A lower hidden-violation rate can result from a
        weaker or compressed concept output falling below 0.5. The collapse/spread panel
        confirms that this occurs for some concepts.

        **Limited conclusion.** Restricting training to 70 species changes concept
        behavior, but does not establish a general grounding improvement. The CUB70 CBM
        is substantially weaker on the same photographs, and effects remain
        concept-specific.

        **Next question.** Before MCBM, identify the collapsed exact concepts and display
        real image/mask examples for the unexplained extremes: high conflict/high
        violation, high conflict/low violation, positive area response, and negative area
        response. Also resolve why three prediction species have no joined masks.
        """),
    ),
]


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    ids = {cell[1]["metadata"]["cub05_observation_id"] for cell in OBSERVATIONS}
    nb["cells"] = [
        c for c in nb["cells"]
        if c.get("metadata", {}).get("cub05_observation_id") not in ids
    ]
    for marker, cell in OBSERVATIONS:
        index = next(
            i for i, existing in enumerate(nb["cells"])
            if marker in "".join(existing.get("source", []))
        )
        nb["cells"].insert(index + 1, cell)
    NOTEBOOK.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"inserted {len(OBSERVATIONS)} inspected observation cells into {NOTEBOOK}")


if __name__ == "__main__":
    main()
