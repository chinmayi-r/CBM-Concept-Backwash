#!/usr/bin/env python3
"""Build the CUB70 MCBM observational report from normalized accepted exports."""
from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path

CURATED = Path(__file__).resolve().parents[1]
OUT = CURATED / "notebooks" / "06_cub_mcbm.ipynb"


def source(text: str) -> list[str]:
    return (textwrap.dedent(text).strip("\n") + "\n").splitlines(keepends=True)


def cell(kind: str, tag: str, text: str) -> dict:
    body = textwrap.dedent(text).strip("\n") + "\n"
    base = {"cell_type": kind, "id": f"cub06-{tag}-{hashlib.sha1(body.encode()).hexdigest()[:10]}",
            "metadata": {}, "source": body.splitlines(keepends=True)}
    if kind == "code":
        base.update(execution_count=None, outputs=[])
    return base


def md(tag: str, text: str) -> dict:
    return cell("markdown", tag, text)


def code(tag: str, text: str) -> dict:
    return cell("code", tag, text)


def before(number: str, title: str, question: str, formula: str, inputs: str,
           reading: str, example: str) -> dict:
    return md(f"before-{number}", f"""
    ## {number} · {title}

    **Question and prediction.** {question}

    **Exact quantity.** {formula}

    **Inputs and model.** {inputs}

    **How to read the figure.** {reading}

    **Concrete example.** {example}
    """)


def after(number: str, support: str, alternative: str, test: str, next_q: str) -> dict:
    return md(f"after-{number}", f"""
    ### What Figure {number} can establish

    - **Literal result:** read the complete table printed above; the plot contains no additional hidden calculation.
    - **What it supports:** {support}
    - **Plausible alternative:** {alternative}
    - **What would distinguish it:** {test}
    - **Limited conclusion:** CUB70 supplies observational evidence, not a clean donor-part intervention.
    - **Next question:** {next_q}
    """)


cells = [
md("title", r"""
# 06 · CUB70 MCBM — what did minimality change on natural photographs?

Notebook 05 studies the official Koh Joint ResNet-50 Standard CBM first. This
chapter then asks whether the official `minimal_cbm` gamma sweep changes the
same CUB70 observational quantities.

The practical question is not merely whether a representation became smaller.
It is whether the named concept score became more tied to its named region
without breaking ordinary concept or species prediction.

There is no accepted CUB donor-part swap. Therefore this chapter never calls a
visible/hidden difference a donor margin or a measured backwash rate.
"""),
md("model", r"""
## Model and loss, in ordinary language and mathematics

For exact concept `j` in image `i`:

- `c_ij` is the known 0/1 label;
- `h_ij` is MCBM's internal scalar slot;
- `z_ij = q_j(h_ij)` is the learned raw concept logit;
- `p_ij = sigmoid(z_ij)` is used only for thresholded classification checks.

The saved MCBM species network reads the complete internal vector `h`, while a
separate learned `1→3→1` head `q_j` turns each scalar `h_j` into the concept
logit `z_j`.

The training objective is

`L = L_species + beta L_concept + gamma L_rep`,

`L_rep = 0.2 sum_j mean_i[(h_ij - (6 c_ij - 3))^2]`.

Thus a negative label has target `-3` and a positive label has target `+3`.
These are soft targets, not clipping bounds. The loss never names a mask or a
pixel location. A score can approach its target while still being inferred
from body/species context.

Two comparisons must remain separate:

1. **Koh Standard → MCBM gamma 0** changes architecture, noise, heads, and recipe.
2. **MCBM gamma 0 → positive gamma** changes the weight on the minimality term
   within the MCBM family.
"""),
code("setup", r"""
import os,re,sys
from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,log_loss

CURATED=Path(os.environ["CURATED_DATA"]); CWD=Path.cwd()
REPO=CWD if (CWD/"analysis").is_dir() else CWD.parent
sys.path.insert(0,str(REPO/"data"/"cub70"))
from cub70_parts import ATTRIBUTE_TYPE_TO_MASK
from relabel_cub_with_cub70 import coarse_visibility
ORDER=["head","eye","beak","neck","body","wing","leg","tail"]
COLORS={"head":"#56B4E9","eye":"#CC79A7","beak":"#E69F00","neck":"#009E73",
        "body":"#0072B2","wing":"#D55E00","leg":"#777777","tail":"#F0E442"}
GAMMAS=[0.,.1,.3,1.]

def require(path,why):
    path=Path(path)
    if not path.is_file(): raise FileNotFoundError(f"Missing {path}\n{why}")
    return path
def family(name): return str(name).split("::",1)[0]
def normalize(E,model,gamma):
    E=E.copy(); E["image"]=E.image.map(lambda x:Path(str(x)).stem)
    if "concept_idx" in E and "concept_index" not in E: E=E.rename(columns={"concept_idx":"concept_index"})
    if "class_label" in E and "y_true" not in E: E=E.rename(columns={"class_label":"y_true"})
    E["pred_label"]=(E.z>0).astype(int)
    E["attribute_type"]=E.concept_name.map(family)
    E["part"]=E.attribute_type.map(ATTRIBUTE_TYPE_TO_MASK)
    E["model"]=model; E["gamma"]=gamma
    return E

KROOT=CURATED/"koh_joint_resnet_v1"/"cub70"/"standard"/"seed1"
KOH=normalize(pd.read_parquet(require(KROOT/"final_test.parquet","Run notebook 05 prerequisites.")),"Koh Standard",np.nan)
tables={"Koh Standard":KOH}
tags={0.:"0",.1:"0p1",.3:"0p3",1.:"1"}
for g in GAMMAS:
    path=require(CURATED/"cub70_eval"/f"cub70-mcbm-g{tags[g]}-s1.parquet",
                 f"Export accepted CUB70 MCBM gamma={g} seed 1 with cub70_export_eval.py")
    tables[f"MCBM gamma={g:g}"]=normalize(pd.read_parquet(path),f"MCBM gamma={g:g}",g)

names=set(KOH.concept_name)
for name,E in tables.items():
    if set(E.concept_name)!=names: raise RuntimeError(f"{name} exact concept set differs from Koh Standard")
    if E.image.nunique()!=1976: raise RuntimeError(f"{name} has {E.image.nunique()} images, expected 1976")
RAWVIS=pd.read_parquet(require(CURATED/"cub70_visibility.parquet","Run data/cub70 preparation."))
V=coarse_visibility(RAWVIS,.001).rename(columns={"image_name":"image","coarse":"part"})
V["image"]=V.image.map(lambda x:Path(str(x)).stem)
joined={name:E[E.part.notna()].merge(V[["image","part","area_frac","visible"]],
        on=["image","part"],how="inner",validate="many_to_one") for name,E in tables.items()}
print("models:",list(tables)); print("images per model:",{k:v.image.nunique() for k,v in tables.items()})
"""),

before("1", "Are all model outputs usable before discussing grounding?",
       "If gamma only produces collapsed or inaccurate outputs, a lower hidden score is not an improvement.",
       "For each model: task accuracy = P(y_pred=y_true); concept balanced accuracy = (TPR+TNR)/2; spread_j = Q95(z_j)-Q05(z_j).",
       "The official Koh CUB70 export and accepted seed-1 MCBM exports at gamma 0, 0.1, 0.3, and 1. No model is retrained.",
       "Higher accuracy is healthier. A raw-z spread of zero within 1e-8 is exact collapse. MCBM gamma 3/5 original-recipe errors are not plotted as results.",
       "Balanced accuracy 0.80 means average positive and negative recognition is 80%, even if one label is rarer."),
code("fig1", r"""
health=[]
for name,E in tables.items():
    images=E[["image","y_true","y_pred"]].drop_duplicates("image")
    exact=[]
    for concept,d in E.groupby("concept_name"):
        pos=d.gt_label==1; pred=d.z>0
        tpr=pred[pos].mean() if pos.any() else np.nan; tnr=(~pred[~pos]).mean() if (~pos).any() else np.nan
        exact.append({"balanced_accuracy":np.nanmean([tpr,tnr]),
                      "spread":d.z.quantile(.95)-d.z.quantile(.05)})
    exact=pd.DataFrame(exact)
    health.append({"model":name,"task_accuracy":(images.y_true==images.y_pred).mean(),
                   "median_concept_balanced_accuracy":exact.balanced_accuracy.median(),
                   "median_raw_z_spread":exact.spread.median(),
                   "collapsed_exact_concepts":int((exact.spread<=1e-8).sum())})
HEALTH=pd.DataFrame(health); display(HEALTH.round(4))
fig,axes=plt.subplots(1,3,figsize=(14,4))
for ax,col,title in zip(axes,["task_accuracy","median_concept_balanced_accuracy","median_raw_z_spread"],
                        ["species accuracy","median concept balanced accuracy","median raw-z spread"]):
    ax.bar(HEALTH.model,HEALTH[col],color=["#333333"]+["#D55E00"]*4);ax.set_title(title);ax.tick_params(axis="x",rotation=55)
fig.suptitle("Figure 1 · Ordinary model-health guards before grounding claims");plt.tight_layout();plt.show()
"""),
after("1", "Only non-collapsed, ordinarily usable model outputs may enter later interpretation.",
      "A model can retain average accuracy while a small set of exact concepts collapses.",
      "Inspect every exact-concept health row and repeat with independent seeds.",
      "Does minimality change the association between named-region visibility and raw z?"),

before("2", "Does a positive concept score rise when its named mask is naturally visible?",
       "If named pixels matter, positive-labelled images should usually score higher when the mapped region is visible.",
       "For each exact concept j: V_j = mean(z_ij | c_ij=1, visible=1) - mean(z_ij | c_ij=1, visible=0). Before pooling, z is standardized within each model and exact concept so gamma-dependent scale alone cannot create the comparison.",
       "The same CUB70 images, labels, masks, and visibility threshold for Koh and every MCBM gamma. No new classifier is trained.",
       "Positive is the predicted direction. Every colored line is one CUB group. The complete exact-concept table gives n_visible and n_hidden.",
       "If visible mean z is 2 and hidden mean z is -1, V=+3 raw logits. If that concept's overall SD is 2, the standardized effect is +1.5."),
code("fig2", r"""
visibility_rows=[]
for name,J in joined.items():
    J=J.copy(); J["z_std"]=J.groupby("concept_name").z.transform(lambda s:(s-s.mean())/s.std(ddof=0) if s.std(ddof=0)>1e-8 else np.nan)
    for concept,d in J[J.gt_label==1].groupby("concept_name"):
        vis=d[d.visible].z_std; hid=d[~d.visible].z_std
        visibility_rows.append({"model":name,"gamma":d.gamma.iloc[0],"concept_name":concept,"part":d.part.iloc[0],
          "n_visible":len(vis),"n_hidden":len(hid),"visibility_effect_std":vis.mean()-hid.mean() if len(vis)>=5 and len(hid)>=5 else np.nan})
VIS_EFFECT=pd.DataFrame(visibility_rows); display(VIS_EFFECT.round(4))
Q=VIS_EFFECT.groupby(["model","part"],observed=True).visibility_effect_std.median().unstack("part").reindex(columns=ORDER)
fig,ax=plt.subplots(figsize=(12,5));x=np.arange(len(Q));
for p in ORDER:
    if p in Q: ax.plot(x,Q[p],"o-",label=p,color=COLORS[p])
ax.axhline(0,color="black",lw=.8);ax.set_xticks(x);ax.set_xticklabels(Q.index,rotation=45,ha="right")
ax.set_ylabel("median visible-minus-hidden z, in concept SDs");ax.legend(ncol=4);ax.set_title("Figure 2 · Natural visibility association by model and CUB group")
plt.tight_layout();plt.show()
"""),
after("2", "A positive group value supports an observational link between the named visible region and its concept score.",
      "Visible and hidden photographs differ in pose, image quality, and possibly species composition.",
      "Repeat inside the same species and exact concept, then compare gradient localization.",
      "Does positive-vs-negative separation remain when the mask is absent?"),

before("3", "When the named mask is absent, does context still separate positive from negative labels?",
       "A large hidden-context gap means the score still distinguishes labels without released visible evidence for the named region.",
       "H_j = mean(z_std | c=1, visible=0) - mean(z_std | c=0, visible=0), standardized within model and exact concept.",
       "Exactly the same normalized tables and masks as Figure 2. No diagnostic is trained.",
       "Positive means hidden positive-labelled photographs score above hidden negative-labelled photographs. That is contextual prediction, not proof that species is the cue.",
       "Hidden positives averaging +0.8 and hidden negatives -0.4 give H=+1.2 concept SDs."),
code("fig3", r"""
context=[]
for name,J in joined.items():
    J=J.copy();J["z_std"]=J.groupby("concept_name").z.transform(lambda s:(s-s.mean())/s.std(ddof=0) if s.std(ddof=0)>1e-8 else np.nan)
    for concept,d in J.groupby("concept_name"):
        pos=d[(d.gt_label==1)&(~d.visible)].z_std;neg=d[(d.gt_label==0)&(~d.visible)].z_std
        context.append({"model":name,"gamma":d.gamma.iloc[0],"concept_name":concept,"part":d.part.iloc[0],
          "n_hidden_positive":len(pos),"n_hidden_negative":len(neg),"context_gap_std":pos.mean()-neg.mean() if len(pos)>=5 and len(neg)>=5 else np.nan})
CONTEXT=pd.DataFrame(context);display(CONTEXT.round(4))
Q=CONTEXT.groupby(["model","part"],observed=True).context_gap_std.median().unstack("part").reindex(columns=ORDER)
fig,ax=plt.subplots(figsize=(12,5));x=np.arange(len(Q))
for p in ORDER:
    if p in Q:ax.plot(x,Q[p],"o-",label=p,color=COLORS[p])
ax.axhline(0,color="black",lw=.8);ax.set_xticks(x);ax.set_xticklabels(Q.index,rotation=45,ha="right")
ax.set_ylabel("median hidden positive-minus-negative z, concept SDs");ax.legend(ncol=4)
ax.set_title("Figure 3 · Contextual label separation while released mask is absent");plt.tight_layout();plt.show()
"""),
after("3", "A persistent positive gap says the model retains label-predictive context under released-mask absence.",
      "The part can be physically present but missed by a coarse or incomplete released mask.",
      "Inspect photographs and compare concept-specific Grad-CAM with masks; a future segment-routed model can provide exact contribution evidence.",
      "How much species identity is available in the concept scores?"),

before("4", "How much species information remains beyond the known 0/1 concept labels?",
       "If raw magnitudes encode extra species structure, they should predict held-out species better than the same concepts' known labels.",
       "On one fixed 70/30 stratified image split, fit diagnostic multinomial logistic regression. Report accuracy and log-loss gain: logloss(labels)-logloss(raw z). Positive gain means raw z carries extra held-out species information.",
       "One newly trained diagnostic per model/input type. It is not the model's saved species head. The split is identical across models.",
       "Bars compare raw-z and known-label accuracy. The table includes blind chance 1/70 and the saved model's own task accuracy.",
       "If labels give log loss 2.0 and raw z gives 1.5, gain=+0.5. This is information availability, not grounding failure."),
code("fig4", r"""
probe=[]
for name,E in tables.items():
    Z=E.pivot(index="image",columns="concept_name",values="z").sort_index(axis=1)
    C=E.pivot(index="image",columns="concept_name",values="gt_label").loc[Z.index,Z.columns]
    y=E[["image","y_true"]].drop_duplicates("image").set_index("image").loc[Z.index].y_true
    tr,te=train_test_split(np.arange(len(Z)),test_size=.30,random_state=20260913,stratify=y)
    values={"known 0/1 labels":C.to_numpy(),"raw z":Z.to_numpy()};fits={}
    for kind,X in values.items():
        fit=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=5000,random_state=20260913))
        fit.fit(X[tr],y.iloc[tr]);pred=fit.predict(X[te]);prob=fit.predict_proba(X[te])
        fits[kind]={"accuracy":accuracy_score(y.iloc[te],pred),"logloss":log_loss(y.iloc[te],prob,labels=fit[-1].classes_)}
    images=E[["image","y_true","y_pred"]].drop_duplicates("image")
    probe.append({"model":name,"label_accuracy":fits["known 0/1 labels"]["accuracy"],"raw_z_accuracy":fits["raw z"]["accuracy"],
      "conditional_logloss_gain":fits["known 0/1 labels"]["logloss"]-fits["raw z"]["logloss"],
      "saved_task_accuracy":(images.y_true==images.y_pred).mean(),"blind_chance":1/70})
PROBE=pd.DataFrame(probe);display(PROBE.round(4))
x=np.arange(len(PROBE));w=.38;fig,ax=plt.subplots(figsize=(11,5))
ax.bar(x-w/2,PROBE.label_accuracy,w,label="known 0/1 labels",color="#999999")
ax.bar(x+w/2,PROBE.raw_z_accuracy,w,label="raw z",color="#D55E00")
ax.axhline(1/70,color="black",ls="--",label="blind 1/70 chance")
ax.set_xticks(x);ax.set_xticklabels(PROBE.model,rotation=45,ha="right");ax.set_ylabel("held-out species accuracy")
ax.set_title("Figure 4 · Species information available in complete concept vectors");ax.legend();plt.tight_layout();plt.show()
"""),
after("4", "Raw-z improvement over the known-label control establishes extra recoverable species information.",
      "A diagnostic may exploit information that the saved MCBM species network does not use, and lower decoding can also reflect poorer optimization.",
      "Replay the saved MCBM h-based species head under a declared internal-code replacement; do not infer use from this probe alone.",
      "Do all measured quantities improve together as gamma rises?"),

before("5", "Does increasing gamma jointly improve health, visibility association, hidden context, and species compression?",
       "A useful loss should not receive credit for shrinking one number while worsening the others.",
       "Join the exact tables already calculated. For each MCBM gamma and group report health, median V_j, median H_j, and complete-vector conditional species log-loss gain. No quantities are added together.",
       "Figures 1-4 only; no new model and no new diagnostic fit.",
       "Each panel retains its own units. Read gamma 0→positive gamma as the minimality comparison; Koh→gamma0 is a separate architecture/recipe contrast.",
       "A smaller context gap with lower concept accuracy is not an accepted repair; it may simply be a weaker model."),
code("fig5", r"""
mcbm_names=[f"MCBM gamma={g:g}" for g in GAMMAS]
summary=[]
for name in mcbm_names:
    row=HEALTH.set_index("model").loc[name].to_dict();row.update(PROBE.set_index("model").loc[name].to_dict())
    row["median_visibility_effect_std"]=VIS_EFFECT[VIS_EFFECT.model==name].visibility_effect_std.median()
    row["median_context_gap_std"]=CONTEXT[CONTEXT.model==name].context_gap_std.median()
    row["gamma"]=float(name.split("=")[1]);summary.append(row)
SYNTH=pd.DataFrame(summary).sort_values("gamma");display(SYNTH.round(4))
fig,axes=plt.subplots(1,4,figsize=(16,4));spec=[
 ("median_concept_balanced_accuracy","concept balanced accuracy"),
 ("median_visibility_effect_std","visible-minus-hidden z"),
 ("median_context_gap_std","hidden context gap"),
 ("conditional_logloss_gain","extra species log-loss gain")]
for ax,(col,title) in zip(axes,spec):
    ax.plot(SYNTH.gamma,SYNTH[col],"o-",color="#D55E00");ax.set_xlabel("gamma");ax.set_title(title)
fig.suptitle("Figure 5 · MCBM gamma changes four distinct quantities; they are not one score")
plt.tight_layout();plt.show()
"""),
after("5", "Only a joint pattern can motivate a loss change: ordinary health retained, stronger named-region association, and less hidden/contextual or species-coded structure.",
      "With one independently initialized model per gamma, a jagged curve can be initialization noise rather than a gamma mechanism.",
      "Train independent seeds under the frozen recipe before calling a gamma trend stable; do not silently merge the stabilized gamma 1/3/5 lane.",
      "What practical loss follows from the measured failure?"),

md("conclusion", r"""
## Final decision table: what loss should be tried next?

| Measured failure | What the current MCBM loss does | Better targeted experiment |
|---|---|---|
| positive `z_j` remains separated when the named mask is absent | pushes `h_j` toward a label target but never names pixels | add a mask/segment sufficiency-and-necessity term, then retain the same health and visibility checks |
| species information remains in raw magnitudes | compresses `h`, but does not directly minimize conditional species information or guarantee location | first test whether the saved head uses it; remove only demonstrated harmful use, not all species-correlated visual detail |
| a concept Grad-CAM lies outside its mapped mask | no localization term | compare with a segment-routed model whose class/concept score is an exact sum of segment contributions |
| poor concept accuracy or collapse | stronger compression may worsen it | reject the repair; grounding is not improved by destroying the concept output |

The justified next loss is therefore **not merely a larger gamma**. It is a
spatially targeted objective that rewards evidence inside the named region and
penalizes reliance outside it, while preserving the same ordinary-health guard.
This is a proposal until trained and tested; the present chapter does not claim
that segmentation has already removed CUB backwash.
"""),
md("provenance", r"""
## Provenance and limitations

- CUB70 MCBM gamma 0, 0.1, 0.3, and 1 are accepted completed runs with the
  documented uncontrolled-initialization limitation.
- Gamma 3/5 original-recipe attempts are `ERROR`, not negative scientific
  results. The stabilized gamma 1/3/5 lane uses a different FP32/lower-LR recipe
  and is not silently joined to this sweep.
- Completed CUB70 MCBMs use the recorded 299-pixel historical CUB preprocessing.
- Natural mask absence is observational and released masks are incomplete or
  coarse for some named attributes.
- Seed 1 cannot provide seed-level uncertainty.
"""),
]

notebook={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
          "language_info":{"name":"python","version":"3.10"}},"nbformat":4,"nbformat_minor":5}
OUT.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+"\n",encoding="utf-8")
print(f"wrote {OUT} with {len(cells)} cells")
