"""Research continuation cells for Notebook 05. No invented model results."""
from __future__ import annotations


OPENING = r"""
# 05 · CUB concept backwash: distinguish annotation errors, local evidence, and context

**Question.** When a concept's named visual value changes, does the saved CBM
follow that evidence, or retain the value suggested by the rest of the bird?
If it retains context, which training incentive or label conflict explains it,
and does changing that supervision improve grounding?

**Completion standard.** This chapter is not complete merely because its code
runs. The requested continuation needs (1) a verified anatomical segmentation
method, (2) a reviewed and calibrated CUB edit, (3) frozen-model measurements
that discriminate mechanisms, and (4) a matched relabeling result if the
required training-mask population is valid and training is authorized.
These new empirical results are currently **INCOMPLETE**. Previously accepted
observational results below remain useful baselines; they do not fill these gaps.

**Model.** Standard non-RL CUB70 Koh Joint ResNet-50, 112 concept logits and 70
species, seed 1. Existing checkpoints, labels, splits, and accepted results are
preserved. No training runs from this notebook. Full CUB is a separate stage.

**How to read the investigation.** First establish what the labels and masks
actually record, then what the model predicts. Use the unexpected patterns to
choose between annotation failure, initial source preference, weak local
response, and context-dependent response. The intervention and relabeling
sections define measurements before data are inspected. An absent result is
shown as incomplete, never as zero or an expected benefit.
"""


def inventory_cells(md, code):
    return [md("cub-mask-inventory-question", r"""
    ## Figure 0 · Does a missing segmentation file mean an invisible part?

    **Question.** Can the released masks alone justify changing a positive
    concept label to negative? The competing explanations for a missing mask
    are physical invisibility and incomplete annotation. File inventory cannot
    decide between them, but it reveals whether the release distinguishes them.

    **Variables and denominator.** An archive image has at least one supplied
    mask. For each of its eleven anatomical parts define an annotation state:
    `file_nonempty`, `file_empty`, or `file_missing`. This is distinct from the
    earlier mask-derived visibility variable `v`: neither missing nor empty
    annotation by itself establishes physical absence. The denominator is the
    number of represented images, not the full CUB70 test population.

    **How to read.** Each horizontal bar is one anatomical part; the x-axis is
    the fraction of represented images. Green denotes a supplied nonempty
    mask, orange an empty mask, and grey no file. Moving right within the grey
    section means a larger annotation gap, not greater model backwash. For
    example, 75 missing files out of 100 images gives 0.75 annotation missingness;
    it does not establish that 75 parts are invisible. This is a complete
    archive count, so no independent-seed uncertainty is attached.
    """), code("cub-mask-inventory", r"""
    import os, sys
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display, Markdown
    cwd = Path.cwd()
    curated_here = cwd if (cwd / "analysis").is_dir() else cwd.parent
    sys.path.insert(0, str(curated_here / "analysis"))
    from cub_mask_inventory import inventory
    mask_source = os.environ.get("CUB70_MASK_SOURCE")
    if mask_source is None and os.environ.get("CURATED_DATA"):
        mask_source = str(Path(os.environ["CURATED_DATA"]) / "cub70" / "masks")
    if mask_source is None:
        raise FileNotFoundError("Set CUB70_MASK_SOURCE to the released archive or mask directory; do not infer visibility from missing inputs.")
    MASK_FILES, MASK_STATES, MASK_INVENTORY, MASK_INVENTORY_PROVENANCE = inventory(mask_source)
    mask_table = pd.DataFrame(MASK_INVENTORY)
    fig, ax = plt.subplots(figsize=(10, 5))
    left = pd.Series(0., index=mask_table.index)
    for field, label, color in [("file_nonempty", "Supplied nonempty mask", "#20826A"),
                                 ("file_empty", "Supplied empty mask", "#D78625"),
                                 ("file_missing", "No supplied file", "#B8BCC2")]:
        fraction = mask_table[field] / mask_table.images_in_archive
        ax.barh(mask_table.part, fraction, left=left, color=color, label=label)
        left += fraction
    ax.set(xlim=(0,1), xlabel="Fraction of archive-represented images", ylabel="CUB anatomical part")
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(.5,1.15), ncol=3, frameon=False)
    fig.suptitle("Figure 0 · Annotation availability is not physical visibility", y=1.02)
    plt.tight_layout(); plt.show()
    display(mask_table)
    display(Markdown(f"**Executed inventory:** {MASK_INVENTORY_PROVENANCE['mask_count']:,} files, "
        f"{MASK_INVENTORY_PROVENANCE['image_count']:,} represented images, "
        f"{MASK_INVENTORY_PROVENANCE['folder_count']} released class folders. "
        "The release must be joined to the official image/split tables before claiming training coverage. "
        "Missing-mask labels are not changed by this audit."))
    """, "Stacked counts of supplied, empty, and omitted mask files for eleven CUB anatomical parts."),
    md("cub-mask-inventory-inference", r"""
    **What this resolves.** File availability and physical visibility are
    different variables. A segmentation replacement must be evaluated against
    visual anatomy, not rewarded simply for reproducing every omission.
    **Next test.** Use exact image identities to join the official split and
    keypoints; inspect photographs where a keypoint is visible but its mask is
    missing. Compare these with genuinely occluded and clearly segmented cases.
    Mask overlaps, anatomical coarseness, and boundary quality are audited
    separately; one defect does not invalidate the entire release.
    """)]


def mechanism_cells(md, code):
    return [md("cub-mechanism", r"""
    ## 13 · Why could a supervised concept learn context?

    **Question.** What does the actual objective reward, and which outcome
    would distinguish a label conflict from a weak visual response? This is a
    derivation from Koh's objective, not a newly measured training history.

    Let `t = Wz+b` be species logits, `s = softmax(t)`, `e_y` the one-hot true
    species, `λ=0.01`, and `w_j` Koh's scalar BCE weight for exact concept `j`.
    The main-output, per-image loss is

    `L = [CE(t,y) + λ Σ_j w_j BCEWithLogits(z_j,c_j)] / (1+λJ)`.

    Koh's implementation uses BCE **weight**, multiplying both label terms,
    not a positive-only `pos_weight`. Training also has an auxiliary branch
    with coefficient 0.4; the equation above isolates the main-output gradient
    and is not a replacement trainer. Its derivative is

    `∂L/∂z_j = { [Wᵀ(s−e_y)]_j + λ w_j [sigmoid(z_j)−c_j] } / (1+λJ)`.

    A positive derivative means gradient descent locally pushes the logit
    down; a negative derivative pushes it up. For example, with `c_j=1`,
    `sigmoid(z_j)=0.2`, and `w_j=1`, the concept term before normalization is
    `−0.008`: it pushes the score up whether the named part is visible or not.
    Context can satisfy this supervision. The task term may reinforce or oppose
    that movement, and the shared encoder determines which pixels change it.
    This gradient makes context use possible; it does not prove it occurred.

    **Why relabeling might help, and why it might not.** At a fixed model state,
    changing a truly invisible positive to zero changes that concept gradient
    by `+λw_j/(1+λJ)`. It removes a specific upward pressure. It does not
    remove task pressure or guarantee a more local image-to-concept mapping.
    Recomputing imbalance weights would change another ingredient and must be
    declared rather than called a labels-only comparison.

    **Identifiability example.** If every training bird of a species has the
    same wing value, a wing-pixel rule and a context rule can predict identical
    labels. High concept accuracy cannot choose between them. The discriminating
    evidence changes local value while retaining context, or changes context
    while retaining identical local evidence.

    | Explanation | Prediction on an informative test | Result that weakens it |
    |---|---|---|
    | Annotation omission drives the apparent hidden score | Many allegedly hidden positives have visibly present parts; corrected visibility reduces the contrast | Contrast survives independently verified absence |
    | Strong initial source preference | Large negative original margin; substantial donorward response insufficient to cross zero | Failure persists with matched initial margins but different response |
    | Weak local sensitivity | Donorward movement remains small for clearly recognizable inserted values | Large robust response on matched visible edits |
    | Context suppresses local response | The same target pixels evoke different increments under changed context | Target response is stable across matched context conditions |
    | Visibility-conflicting supervision contributes | Matched relabeling improves donor wins and response while retaining health | Event count falls only because donor response disappears, or nothing improves |

    CUB observations select hypotheses; they do not identify these mechanisms
    by themselves. The experiment below must measure the predicted contrasts.
    """), md("cub-segmentation-design", r"""
    ## 14 · Obtain anatomical masks that support the experiment

    **Recovered source.** The September 6 conversation explicitly names
    [From Segments to Concepts, v1](https://arxiv.org/abs/2510.04180v1).
    Its later [v2, Spatially Grounded Concept-Based Image Classification](https://arxiv.org/html/2510.04180v2)
    describes offline CLIP concept ranking, GroundingDINO boxes and SAM masks
    (Algorithm 1). CUB uses CLIP ViT-B/16, top 32 concepts, maximum mask-area
    fraction 0.5 and merge IoU 0.5. The displayed implementation table leaves
    the minimum-pixel threshold symbolic; any numeric choice is ours, not a
    recovered paper setting. This pipeline proposes regions; it does not supply
    a validated donor-part swap or certify physical absence after no detection.

    **User's correction.** Use released CUB70 masks directly as the trusted
    segmentation for covered images. No extra annotator is required merely to
    recreate them. The new engineering question is replacement. For uncovered
    images, reproduce the paper's region-proposal procedure and compare its
    coverage and anatomy with released masks. Do not replace Koh with the
    paper's different MIL classifier inside this Standard-CBM investigation.

    **Evaluation population.** Split images, not mask files. Keep fitting,
    method selection, and final evaluation identities separate. Report whether
    a pretrained segmenter has seen any evaluation photographs. An eleven-part
    benchmark cannot certify finer belly/back/throat distinctions it never labels.

    **Quantities defined before testing.** For predicted part mask `P_ig` and
    independently reviewed reference `G_ig`, `IoU=|P∩G|/|P∪G|` on nonempty
    unions; count both-empty cases separately. Report per-part IoU, false absence
    among visibly present parts, false presence among verified absent parts,
    and all unreviewed/ambiguous cases. Never score a missing reference file as
    confirmed negative. Use released masks as one source of reference evidence,
    with independent review for contradictions and annotation gaps.

    **Decision.** Mask quality must be adequate for the specific operation:
    a swap needs clean boundaries and value-preserving alignment; relabeling
    needs reliable presence/absence on training images. A good average IoU
    establishes neither requirement alone. The segmenter identity and criteria
    will be frozen before inspecting its held-out CBM outcomes.
    """), md("cub-paired-edit-design", r"""
    ## 15 · Can a CUB edit change the named value without changing the question?

    **Protocol under development, not an accepted swap.** Choose source images
    with an independently visible exact concept and select both a same-value
    donor and a different-value donor of the same attribute family. Match pose,
    size, view, and anatomic footprint before consulting model scores. Apply the
    same transformation, insertion, and boundary handling to both donors. A
    shape-changing concept cannot be tested by warping both shapes to the same
    silhouette; a color edit cannot quietly change several named attributes.

    Include original, self-reconstruction, same-value insertion, and different-
    value insertion images. Record source/donor identities, transform, pixel
    hashes, affected concept labels, and changed-pixel map. Visually review
    inserted value, old-part removal, and off-target damage while blinded to
    the model outcome. Report eligibility and all exclusions. Natural-image
    composites remain a weaker operation than renderer swaps unless calibrated.

    For a fixed source/donor concept pair let
    `m_orig=z_D(x_orig)−z_S(x_orig)`, `m_same=z_D(x_same)−z_S(x_same)`, and
    `m_diff=z_D(x_diff)−z_S(x_diff)`, in raw logit units.
    Define `response_delta=m_diff−m_orig`,
    `control_adjusted_response=m_diff−m_same`, and
    `same_value_edit_drift=m_same−m_orig`.

    Example: margins `−10, −8, −2` imply raw response `+8`, control-adjusted
    response `+6`, and edit drift `+2`. The donor still loses (`m_diff<0`).
    A positive adjusted response shows an effect beyond this same-value edit
    control; it does not erase all realism and co-changing-attribute confounds.

    **Report all reviewed trials**, separated into donor win (`m_diff>0`),
    positive response with source retention (`response_delta>0, m_diff<0`),
    nonpositive response with source retention (`response_delta≤0, m_diff<0`),
    and ties. Report the adjusted-response companion and edit drift separately.
    Do not select only responding images and then call the resulting percentage
    a population backwash rate. Counts are descriptive within a trained seed;
    model replication uncertainty requires independently trained seeds.

    **Calibration.** Test the proposed editing procedure on positive controls
    with known values and inspect source/donor recognition. Compare with the
    renderer reference where transferable. If the insertion cannot produce
    recognizable donor evidence, that method is not calibrated; the model's
    original observational results remain usable. The new procedure preserves donor silhouette under isotropic alignment, whereas
    the old pilot stretched donor texture into the source silhouette. Alignment and controls must be
    separately identified and evaluated.
    """), md("cub-relabel-design", r"""
    ## 16 · Would relabeling improve grounding, and by how much?

    **Training feasibility comes first.** Join mask/review records to official
    training image identities and exact concepts. Let `u_ig` be independently
    established visible, absent, or unknown state. A potential label intervention
    is `c'_ij=0` only when `c_ij=1` and `u_ig=absent`; keep other labels, retaining
    unknown status rather than assuming absence. Report counts for every
    image/concept affected. Released test masks alone cannot label the training set.

    This changes the target from a species-associated attribute toward visible
    evidence. Evaluate original-label performance and verified-visible-label
    performance separately. Do not claim a benefit solely because the evaluation
    target was changed to match the intervention.

    **Matched comparison.** Freeze the data split, architecture, encoder weights
    family, optimizer, schedule, image augmentations, and paired seeds. Declare
    whether imbalance weights are frozen or recomputed. No CUB70 relabeling
    training is launched by this chapter. The FunnyBird accelerated protocol is
    not automatically authorized for CUB70.

    **Magnitude of benefit.** Evaluate Standard and relabeled models on exactly
    the same reviewed edit images and a common ordinary test set. Report paired
    changes in final margin, response, donor-win fraction, and all response-state
    transitions by concept, anatomical group, and seed. A reduction in positive-
    response/source-retention cases is not sufficient: count those that become
    donor wins separately from those that merely lose donorward response.
    Check ordinary concept and species health so a less informative model is not
    called better grounded. Independently trained matched seeds are the units
    for replicated training-effect uncertainty.

    **Current answer:** the gradient predicts removal of one incentive, not an
    effect size. No new CUB relabeling benefit has been measured in this
    reconstruction. The actual benefit can be positive, zero, or negative.
    """), md("cub-intervention-readout", r"""
    ## Figure 18 · Response, retained source preference, and edit controls

    The horizontal axis is `response_delta` in raw logit units; rightward means
    more donorward change from the original photograph. The vertical axis is
    `m_diff`; upward means stronger donor preference after insertion. Dashed
    zero lines separate direction of response and final preference. The lower
    right quadrant therefore shows positive response with source retention.
    Example: response +8 and final margin −2 belongs in that quadrant despite
    the substantial movement. Each point is one reviewed trial, colors identify
    anatomical groups, and panels identify trained seeds. No jitter or image-level
    confidence interval is used. Reused photographs are not independent replicas.

    The companion table reports selected, eligible and excluded counts; fractions
    use all eligible trials within that seed/group, including nonresponders.
    Medians of `m_orig`, `m_diff`, `response_delta`, `control_adjusted_response`,
    and `same_value_edit_drift` use that same population. Counts of the four
    response states use the predicates defined in section 15. Model calibration
    remains separate: this plot is a diagnostic, not automatic causal acceptance.
    """), code("cub-intervention-availability", r"""
    from IPython.display import display, Markdown
    from pathlib import Path
    import os
    experiment_root = os.environ.get("CUB_INTERVENTION_RESULTS")
    if experiment_root is None:
        display(Markdown("**INCOMPLETE: no reviewed new CUB intervention results are supplied.** "
            "No CUB swap rate or relabeling effect is computed. The protocol and tested outcome-accounting "
            "code are ready for the identified segmentation method and accepted model artifacts."))
    else:
        from cub_intervention_analysis import load_reviewed_results, summarize_interventions
        import matplotlib.pyplot as plt
        scored_edits, inference_manifest = load_reviewed_results(experiment_root)
        eligible_edits = scored_edits[scored_edits.review_eligible]
        display(Markdown("**Figure 18 · Reviewed paired-edit diagnostics.** "
            f"{len(eligible_edits)} of {len(scored_edits)} trials have all five explicit review checks. "
            "These checks do not certify method calibration."))
        display(summarize_interventions(scored_edits))
        if len(eligible_edits):
            seeds = sorted(eligible_edits.seed.unique())
            fig, axes = plt.subplots(1, len(seeds), figsize=(6*len(seeds),5), squeeze=False)
            for ax, seed in zip(axes[0], seeds):
                for group, rows in eligible_edits[eligible_edits.seed == seed].groupby('mask_group'):
                    ax.scatter(rows.response_delta, rows.margin_different_value, label=group, alpha=.7)
                ax.axhline(0, color='grey', linestyle='--'); ax.axvline(0, color='grey', linestyle='--')
                ax.set(xlabel='Donorward response from original (raw logits)',
                       ylabel='Final donor minus source score (raw logits)', title=f'Seed {seed}')
                ax.legend(title='CUB region')
            fig.suptitle('Figure 18 · Paired-edit diagnostic; calibration remains separate')
            plt.tight_layout(); plt.show()
        else:
            display(Markdown("**INCOMPLETE:** review the image grids while blinded to predictions. "
                "No response fraction is interpreted from unreviewed edits."))
    """, "New CUB intervention result availability; missing empirical evidence is never represented as zero.")]


CONCLUSION = r"""
## 17 · What has been resolved, and what has not?

The preceding accepted natural-image results establish model health and
observational organization; their displayed methods retain their individual
scope. They do not establish the new segmentation, edit, or relabeling claims.

| User's research question | Evidence needed | Current status |
|---|---|---|
| Do released masks establish physical absence? | Annotation-state inventory plus photograph/keypoint review | File inventory available; physical absence cannot be inferred from omission alone |
| Does a better segmenter support anatomical interventions? | Identified source method, split/ontology audit, held-out quality and examples | Named paper recovered; automatic region extraction and real-data evaluation pending |
| Can the edited concept overcome retained source evidence? | Reviewed paired CUB edits and frozen accepted-CBM predictions | INCOMPLETE: no accepted new CUB swap result |
| Which mechanism explains failure? | Tests separating initial margin, local response, context interaction, and annotation errors | Objective-derived predictions stated; new causal tests not yet measured |
| Does relabeling help, and how much? | Valid training visibility, matched training intervention, identical evaluation edits and health controls | INCOMPLETE: no quantified CUB relabeling benefit |

**Has the user's problem with Notebook 05 been fully solved? No.** The code and
writing must not turn these missing experiments into claims. The next decisive
step is to evaluate the reconstructed edit procedure against renderer controls, then run
the declared intervention on accepted frozen models. Relabeling is a separate
matched training decision after visibility coverage is established.
"""


def extend(cells, md, code):
    def starts(cell, tag):
        return cell.get("id", "").startswith(tag + "-")
    revised = []
    for cell in cells:
        if starts(cell, "cub-title"):
            revised.append(md("cub-title", OPENING))
        elif starts(cell, "cub-roadmap"):
            revised.append(md("cub-roadmap", """
            ## Evidence route

            Annotation inventory → existing model/data baseline → anomalies
            needing explanation → objective-derived mechanisms → segmentation
            validation → paired value-changing edits → matched relabeling test.
            Existing observational figures retain their original numbers.
            Figure 0 audits the recovered masks; sections 13–17 contain the
            research continuation and explicit unfinished empirical questions.
            """))
        elif starts(cell, "cub-compare"):
            historical = "".join(cell["source"])
            historical = historical.replace("## 13 ·", "## Retained baseline ·", 1)
            revised.append(md("cub-compare", "**Provenance:** the following table and numerical summary belong to the earlier reviewed baseline. They are retained, not newly verified by this reconstruction; current outputs must be reviewed before reacceptance.\n\n" + historical))
            revised.extend(mechanism_cells(md, code))
            revised.append(md("cub-research-conclusion", CONCLUSION))
        elif starts(cell, "cub-ledger"):
            historical = "".join(cell["source"]).replace("## 14 ·", "## Historical baseline ·", 1)
            revised.append(md("cub-ledger", "**Historical review ledger:** records prior acceptance, not review of the newly generated report.\n\n" + historical))
            revised.append(md("cub-baseline-ledger-note", """
            ### Scope of the retained baseline

            The previously reviewed observational figures remain baseline
            evidence from their stated checkpoints and populations. Source
            review records do not certify newly generated figures or the new
            intervention. Check each current output against its exact inputs
            and code before accepting a revised render.
            """))
        else:
            revised.append(cell)
        if starts(cell, "cub-model"):
            revised.extend(inventory_cells(md, code))
    return revised
