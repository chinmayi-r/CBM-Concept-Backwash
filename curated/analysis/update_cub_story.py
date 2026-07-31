#!/usr/bin/env python3
"""Add the cross-dataset evidence chain and targeted CUB diagnostics.

The CUB notebooks are executed on Adroit, but their scientific structure is
maintained here. This updater is idempotent: cells are identified by a stable
``cub_story_id`` metadata field and replaced rather than duplicated.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


HERE = Path(__file__).resolve().parent
NOTEBOOKS = HERE.parent / "notebooks"


def source(text: str) -> list[str]:
    return (dedent(text).strip("\n") + "\n").splitlines(keepends=True)


def markdown(cell_id: str, text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {"cub_story_id": cell_id},
        "source": source(text),
    }


def code(cell_id: str, text: str) -> dict:
    return {
        "cell_type": "code",
        "id": cell_id,
        "execution_count": None,
        "metadata": {"cub_story_id": cell_id},
        "outputs": [],
        "source": source(text),
    }


def replace_after(cells: list[dict], marker: str, additions: list[dict]) -> None:
    ids = {c["metadata"]["cub_story_id"] for c in additions}
    cells[:] = [
        c for c in cells if c.get("metadata", {}).get("cub_story_id") not in ids
    ]
    index = next(
        i for i, cell in enumerate(cells)
        if marker in "".join(cell.get("source", []))
    )
    cells[index + 1:index + 1] = additions


def update_04(nb: dict) -> None:
    replace_after(
        nb["cells"],
        "# 04",
        [markdown(
            "cub04_funnybird_map",
            r"""
            ## How this repeats the FunnyBird data question

            Notebook 01 showed that FunnyBird concept targets are largely determined by species,
            then asked whether image-level visibility contradicts those targets. This notebook asks
            the same *data* question on CUB before any CUB model is interpreted.

            | FunnyBird evidence step | Closest CUB test here | What a match would mean |
            |---|---|---|
            | concept labels are species-associated | measure within-species CUB attribute variation | species identity is not a complete image-level concept label |
            | preprocessing hides image variation | count raw labels overwritten by species-majority targets | training supervision can erase real within-species differences |
            | visibility can contradict a positive target | compare CUB70 labels with held-out part masks | the named part can be absent while the processed target remains positive |

            **Pre-result prediction.** If the same supervision problem exists in CUB, raw labels
            should vary within species, majority preprocessing should overwrite a nontrivial number
            of image labels, and mask disagreement should remain after obvious invisible-negative
            annotations are excluded.

            **Boundary.** These are data facts. They establish an opportunity for backwash but do
            not show that a trained CUB model actually uses species or body context. That model test
            begins in notebook 05.
            """,
        )],
    )


def update_05(nb: dict) -> None:
    replace_after(
        nb["cells"],
        "# 05",
        [markdown(
            "cub05_funnybird_map",
            r"""
            ## Direct comparison with the FunnyBird CBM evidence chain

            The aim is to test whether the *pattern* found causally in FunnyBirds also appears in
            real CUB photographs. The measurements are kept as parallel as the datasets allow.

            | FunnyBird CBM question | CUB test | Directly comparable? | CUB limitation |
            |---|---|---|---|
            | Does a present concept score high on ordinary images? | positive-label concept output | yes, descriptively | CUB labels may already be species-majority targets |
            | Does removing the named part lower its concept? | mask-visible versus naturally mask-absent photographs | same question, weaker test | different photographs also differ in pose, background, and species |
            | Does more visible part evidence strengthen the output? | within-part mask-area dose response | approximately | pixel area is not the same as diagnostic visual information |
            | Does donor beat source after a fixed swap? | unavailable | no | CUB has no paired renderer intervention |
            | Does the effect remain after species is fixed? | within-(species, concept) visible-minus-occluded comparison | yes, observationally | pose and viewpoint remain uncontrolled |
            | Do visibility-aware training labels repair grounding? | unavailable with current masks | no | CUB70 masks cover evaluation images, not the training set |

            **Pre-result prediction.** If CUB has the same nonlocal-grounding problem, positive
            outputs should often remain high with the named mask absent, visibility dose response
            should be weak for at least some parts, and a smaller effect may remain after species
            and concept are fixed. If the FunnyBird result were only a synthetic-renderer artifact,
            CUB outputs should instead track natural mask visibility strongly.

            **Conclusion rule.** CUB can corroborate the same observational pattern. It cannot by
            itself prove that species caused an individual prediction; the paired FunnyBird
            swap/deletion experiment remains the causal source test.
            """,
        )],
    )

    replace_after(
        nb["cells"],
        "Same masked images: full-CUB versus CUB70-trained CBM",
        [
            markdown(
                "cub05_eye_collapse_md",
                r"""
                ## 5 · Odd result: is the eye output grounded or collapsed?

                The full-CUB eye violation bar is almost zero, but the preceding tables place its
                mean probability near 0.5 for both visible and hidden eyes. A near-zero thresholded
                violation is not evidence of good grounding if the predictor simply emits one
                constant value below 0.5.

                **Discriminating test.** For each model and part, measure probability standard
                deviation, range, and number of distinct rounded values among originally-positive
                rows. Then inspect the complete eye distribution split by mask visibility.

                - broad, separated eye distributions would support genuine visibility response;
                - a narrow spike at one value would identify output collapse or a mapping problem.
                """,
            ),
            code(
                "cub05_eye_collapse_code",
                r"""
                collapse_rows=[]; task_rows=[]
                model_frames=[]
                if J_FULL is not None:
                    model_frames.append(("full-CUB trained",J_FULL))
                if "J70" in globals() and J70 is not None:
                    model_frames.append(("CUB70 trained",J70))
                for model,J in model_frames:
                    images=J[["image","y_true","y_pred"]].drop_duplicates("image")
                    task_rows.append({
                        "model":model,"n_images":len(images),
                        "task_accuracy":(images.y_true==images.y_pred).mean(),
                    })
                    for part,d in J[J.gt_label==1].groupby("part"):
                        collapse_rows.append({
                            "model":model,"part":part,"n":len(d),
                            "prob_mean":d.prob.mean(),"prob_std":d.prob.std(),
                            "prob_min":d.prob.min(),"prob_max":d.prob.max(),
                            "unique_prob_6dp":d.prob.round(6).nunique(),
                        })
                MODEL_TASK=pd.DataFrame(task_rows)
                COLLAPSE=pd.DataFrame(collapse_rows)
                display(MODEL_TASK.round(4))
                display(COLLAPSE.sort_values(["part","model"]).round(6))

                if model_frames:
                    fig,axes=plt.subplots(1,len(model_frames),figsize=(6*len(model_frames),3.7),
                                          squeeze=False,sharex=True,sharey=True)
                    bins=np.linspace(0,1,51)
                    for ax,(model,J) in zip(axes[0],model_frames):
                        eye=J[(J.part=="eye")&(J.gt_label==1)]
                        for visible,color in [(False,"#CC79A7"),(True,CBM_C)]:
                            d=eye[eye.visible==visible]
                            ax.hist(d.prob,bins=bins,alpha=.55,color=color,
                                    label=f"visible={visible}, n={len(d)}")
                        ax.axvline(.5,color="black",ls="--",lw=.8)
                        ax.set_title(model);ax.set_xlabel("eye concept probability")
                        ax.legend(fontsize=8)
                    axes[0,0].set_ylabel("rows")
                    fig.suptitle("Eye output distribution: grounding response or collapsed score?")
                    fig.tight_layout();plt.show()
                """,
            ),
            markdown(
                "cub05_eye_collapse_boundary",
                r"""
                **Conclusion boundary.** A collapsed eye output must be reported as model failure or
                calibration/mapping failure, not as successful grounding. The CUB70-trained eye
                result is interpreted separately because changing the number of species may alter
                whether the eye concept is learned at all.
                """,
            ),
        ],
    )

    replace_after(
        nb["cells"],
        "Is the visibility response specific to positive concepts?",
        [
            markdown(
                "cub05_reversal_md",
                r"""
                ## 8 · Odd result: why do tail and wing reverse after species matching?

                In the pooled positive-label comparison, tail and wing can score slightly *lower*
                when visible. In the within-species control, both become slightly positive. This is
                a possible species-composition reversal: naturally visible and hidden photographs
                contain different mixtures of species.

                **Discriminating test.** Put the pooled and species-matched estimates side by side,
                then show every eligible `(species, concept)` visibility effect. If the matched
                distribution is centered above zero while the pooled estimate is negative, species
                composition explains the sign reversal. Wide positive and negative matched effects
                would instead show genuine species-specific heterogeneity.
                """,
            ),
            code(
                "cub05_reversal_code",
                r"""
                if J_FULL is not None:
                    pooled=(CONTROL[CONTROL.gt_label==1]
                            .set_index("part").visible_minus_occluded.rename("pooled"))
                    matched=(MATCHED.set_index("part").visible_minus_occluded
                             .rename("species_matched"))
                    REVERSAL=pd.concat([pooled,matched],axis=1).reset_index()
                    REVERSAL["sign_reversal"]=(
                        np.sign(REVERSAL.pooled)!=np.sign(REVERSAL.species_matched)
                    )
                    display(REVERSAL.round(4))

                    pos=J_FULL[J_FULL.gt_label==1]
                    effects=(pos.groupby(["part","y_true","concept_idx","visible"]).prob.mean()
                             .unstack("visible").dropna())
                    effects["effect"]=effects[True]-effects[False]
                    effects=effects.reset_index()
                    EFFECT_SUMMARY=(effects.groupby("part").effect.agg(
                        n_groups="size",mean="mean",median="median",
                        q25=lambda x:x.quantile(.25),q75=lambda x:x.quantile(.75),
                        fraction_positive=lambda x:(x>0).mean(),
                    ).reset_index())
                    display(EFFECT_SUMMARY.round(4))
                    fig,ax=plt.subplots(figsize=(8,4))
                    rng=np.random.default_rng(20260731)
                    for i,part in enumerate(sorted(effects.part.unique())):
                        values=effects.loc[effects.part==part,"effect"].to_numpy()
                        ax.scatter(i+rng.uniform(-.16,.16,len(values)),values,
                                   s=13,alpha=.35,color=CBM_C)
                    ordered=sorted(effects.part.unique())
                    med=effects.groupby("part").effect.median().reindex(ordered)
                    ax.scatter(range(len(ordered)),med,s=65,marker="_",color="black",
                               linewidth=2,label="matched-group median")
                    ax.axhline(0,color="black",lw=.8)
                    ax.set_xticks(range(len(ordered)));ax.set_xticklabels(ordered,rotation=30,ha="right")
                    ax.set_ylabel("visible − occluded probability within (species, concept)")
                    ax.set_title("Species-matched visibility effects: distribution behind the mean")
                    ax.legend(fontsize=8);plt.show()
                """,
            ),
            markdown(
                "cub05_reversal_boundary",
                r"""
                **Limited conclusion rule.** Species matching can diagnose composition bias, but it
                remains observational. A causal source-species claim still requires changing body
                or species context while holding the target part fixed—an intervention available in
                FunnyBirds but not in these natural CUB photographs.
                """,
            ),
        ],
    )

    for cell in nb["cells"]:
        text = "".join(cell.get("source", []))
        if "## 7 · Seed support and final decision" in text:
            cell["source"] = source(text.replace(
                "## 7 · Seed support and final decision",
                "## 9 · Seed support and final decision",
            ))
        if "## Conclusion boundary" in text:
            cell["source"] = source(r"""
                ## Integrated conclusion boundary

                Notebook 05 repeats the FunnyBird CBM grounding questions wherever CUB permits:
                ordinary positive outputs, natural part absence, visibility dose response,
                species matching, negative-label specificity, collapse checks, and seed support.

                A matching CUB pattern strengthens external validity: the concept can remain high
                without visible named-part evidence in real photographs, not only synthetic birds.
                But CUB cannot turn that observation into the same causal statement because its
                visible and hidden cases are different photographs. It also cannot reproduce RLv2
                training with the current test-only masks. The causal claims therefore remain
                anchored by fixed FunnyBird deletion/swap interventions; CUB asks whether the same
                warning signs generalize to a natural dataset.
                """)


def update_06(nb: dict) -> None:
    replace_after(
        nb["cells"],
        "# 06",
        [markdown(
            "cub06_funnybird_map",
            r"""
            ## How this repeats the FunnyBird MCBM question

            Notebook 03 asked whether increasing minimality `γ` removes backwash. The direct
            FunnyBird endpoint was a fixed part swap. Here the closest endpoint is

            `P(c_pred_j ≥ 0.5 | original c_j=1, named mask absent)`

            plus the within-species visible-minus-occluded effect from notebook 05.

            **Prediction if minimality improves grounding:** as `γ` rises, absent-part violation
            should fall, the species-matched visibility effect should become more positive, and
            task/concept behavior should remain noncollapsed.

            **Prediction if minimality only compresses `z`:** raw scores may narrow while violation
            and species-matched visibility dependence remain unchanged or become erratic.

            | What stays directly comparable to FunnyBirds | What does not |
            |---|---|
            | same CBM/MCBM architecture family and γ ordering | CUB uses natural occlusion rather than a fixed rendered edit |
            | same question: can the concept stay positive without its part? | hidden and visible CUB photographs differ in pose/background |
            | same collapse and task-accuracy guards | current CUB masks cannot relabel training data like RLv2 |
            | same requirement for seeds and a nonmonotone γ warning | CUB has no donor-versus-source swap margin |

            Thus notebook 06 tests whether the *γ-dependent pattern generalizes*. It cannot replace
            the FunnyBird causal minimality experiment.
            """,
        )],
    )

    replace_after(
        nb["cells"],
        "CUB70 MCBM: visibility violation versus",
        [
            markdown(
                "cub06_collapse_guard_md",
                r"""
                ## Collapse guard across γ

                A lower absent-part violation rate is not improved grounding if the model outputs a
                constant value below 0.5 or loses task accuracy. For every exported CUB70 MCBM,
                report task accuracy and positive-label probability spread by part before reading
                the γ curve.
                """,
            ),
            code(
                "cub06_collapse_guard_code",
                r"""
                collapse_rows=[];task_rows=[]
                for path in sorted((CURATED/"cub70_eval").glob("cub70-mcbm-g*-s*.parquet")):
                    m=re.match(r"cub70-mcbm-g([0-9p]+)-s(\d+)",path.stem)
                    if not m:continue
                    gamma=float(m.group(1).replace("p","."));seed=int(m.group(2))
                    E=pd.read_parquet(path)
                    images=E[["image","y_true","y_pred"]].drop_duplicates("image")
                    task_rows.append({"gamma":gamma,"seed":seed,"n":len(images),
                                      "task_accuracy":(images.y_true==images.y_pred).mean()})
                    for part,d in E[(E.part!="")&(E.gt_label==1)].groupby("part"):
                        collapse_rows.append({
                            "gamma":gamma,"seed":seed,"part":part,"n":len(d),
                            "prob_mean":d.prob.mean(),"prob_std":d.prob.std(),
                            "prob_min":d.prob.min(),"prob_max":d.prob.max(),
                            "unique_prob_6dp":d.prob.round(6).nunique(),
                        })
                if task_rows:
                    TASK_GAMMA=pd.DataFrame(task_rows);COLLAPSE_GAMMA=pd.DataFrame(collapse_rows)
                    display(TASK_GAMMA.sort_values(["gamma","seed"]).round(4))
                    display(COLLAPSE_GAMMA.sort_values(["part","gamma","seed"]).round(6))
                    fig,axes=plt.subplots(1,2,figsize=(11,3.8))
                    for seed,d in TASK_GAMMA.groupby("seed"):
                        axes[0].plot(d.gamma.replace(0,.03),d.task_accuracy,"o-",label=f"seed {seed}")
                    axes[0].set_xscale("log");axes[0].set_ylim(0,1)
                    axes[0].set_xlabel("γ (0 shown at 0.03)");axes[0].set_ylabel("task accuracy")
                    axes[0].set_title("Task-performance guard");axes[0].legend(fontsize=8)
                    for part,d in COLLAPSE_GAMMA.groupby("part"):
                        g=d.groupby("gamma").prob_std.mean().reset_index()
                        axes[1].plot(g.gamma.replace(0,.03),g.prob_std,"o-",label=part)
                    axes[1].set_xscale("log");axes[1].set_xlabel("γ (0 shown at 0.03)")
                    axes[1].set_ylabel("std of positive-label probability")
                    axes[1].set_title("Concept-output collapse guard");axes[1].legend(ncol=2,fontsize=7)
                    fig.tight_layout();plt.show()
                else:
                    print("[pending] export CUB70 MCBM evaluation tables before collapse audit")
                """,
            ),
        ],
    )


def update(path: Path, updater) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    updater(nb)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"updated {path}")


def main() -> None:
    update(NOTEBOOKS / "04_cub_analysis.ipynb", update_04)
    update(NOTEBOOKS / "05_cub_cbm.ipynb", update_05)
    update(NOTEBOOKS / "06_cub_mcbm.ipynb", update_06)


if __name__ == "__main__":
    main()
