#!/usr/bin/env python3
"""Build the CUB70-CBM-only exploratory notebook.

The notebook deliberately starts from the 28 CUB attribute families and the 11
CUB70 masks.  It does not use relabelled models or MCBM results, and it leaves
literal observations to be written only after the executed figures are viewed.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "05_cub_cbm.ipynb"


def src(text: str) -> list[str]:
    return (dedent(text).strip("\n") + "\n").splitlines(keepends=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": src(text),
    }


cells = [
md(r"""
# 05 · CUB70 CBM exploration — what does each named concept use?

This notebook studies **CBMs only**. It does not use MCBM and it does not use
visibility-relabelled training.

We also do not assume CUB must repeat FunnyBirds. FunnyBirds gives us careful
questions to ask. CUB70 is allowed to answer differently.

The order is fixed:

1. list the dataset objects;
2. count species and concepts;
3. count visibility for all 11 masks;
4. describe label/species structure;
5. check that the CBM works normally;
6. inspect visibility behavior at the full attribute-type and exact-concept
   levels;
7. investigate any odd result;
8. only at the end compare the CUB70-trained CBM with the full-CUB-trained CBM.

Every plot begins with a question and a decision rule. Literal observations are
added only after the executed image is displayed and inspected.
"""),
md(r"""
## 0 · The complete non-RL reasoning inherited from FunnyBirds

The standard CBM notebook first checked training and untouched images. It then
deleted one part, validated a one-part swap visually, compared the inserted and
old concept scores, checked both directions and coverage, separated visual
response from final victory, tested visibility, tested exact variants, and only
then asked whether source body/species still mattered. Species decoding and task
effects were kept separate from grounding.

The standard MCBM notebook came next. It first verified that `gamma` really
compressed the representation, then repeated the same swap, visibility,
deletion, variant, direction, seed, and species controls. Its central lesson was
simple: a small bottleneck can still read the wrong pixels. Relabeling was only a
later proposed cause test; it is not part of either discovery notebook.

The full minute-by-minute chain is frozen in
`CUB70_CBM_EXPLORATION_PLAN.md`.
"""),
code(r"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CURATED = Path(os.environ["CURATED_DATA"])
CWD = Path.cwd()
REPO = CWD if (CWD/"analysis").is_dir() else CWD.parent
sys.path.insert(0, str(REPO/"analysis"))
sys.path.insert(0, str(REPO/"data"/"cub70"))

from cub70_parts import (
    CUB70_PARTS, COARSE_TO_CUB70, ATTRIBUTE_TYPE_TO_MASK, attribute_type,
)
from relabel_cub_with_cub70 import coarse_visibility

COLORS = {
    "beak":"#E69F00", "eye":"#CC79A7", "head":"#56B4E9",
    "neck":"#009E73", "body":"#0072B2", "wing":"#D55E00",
    "leg":"#999999", "tail":"#F0E442", None:"#333333",
}

def require(path, command):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}\nProduce it with: {command}")
    return path

def family(name):
    return str(name).split("::", 1)[0]

def add_mapping(E):
    E = E.copy()
    E["attribute_type"] = E.concept_name.map(family)
    E["mask_group"] = E.attribute_type.map(ATTRIBUTE_TYPE_TO_MASK)
    return E

def attach(E, coarse):
    E = add_mapping(E)
    local = E[E.mask_group.notna()].copy()
    V = coarse.rename(columns={"image_name":"image", "coarse":"mask_group"})
    return local.merge(V[["image","mask_group","pixel_count","area_frac","visible"]],
                       on=["image","mask_group"], how="inner")

def exact_visibility_metrics(J):
    rows=[]
    for (ctype,cname),d in J[J.gt_label==1].groupby(["attribute_type","concept_name"]):
        vis=d[d.visible].prob; hid=d[~d.visible].prob
        rows.append({
            "attribute_type":ctype,"concept_name":cname,"mask_group":d.mask_group.iloc[0],
            "n_positive":len(d),"n_visible":len(vis),"n_hidden":len(hid),
            "label_hidden_rate":len(hid)/len(d),
            "prob_visible":vis.mean() if len(vis) else np.nan,
            "prob_hidden":hid.mean() if len(hid) else np.nan,
            "visibility_effect":vis.mean()-hid.mean() if len(vis) and len(hid) else np.nan,
            "visible_recall":(vis>=.5).mean() if len(vis) else np.nan,
            "hidden_violation":(hid>=.5).mean() if len(hid) else np.nan,
            "prob_std":d.prob.std(),"unique_prob_6dp":d.prob.round(6).nunique(),
        })
    return pd.DataFrame(rows)

def matched_effects(J):
    P=J[J.gt_label==1]
    g=(P.groupby(["attribute_type","concept_name","y_true","visible"]).prob.mean()
       .unstack("visible"))
    if not {False,True}.issubset(g.columns): return pd.DataFrame()
    g=g.dropna(subset=[False,True]).copy()
    g["effect"]=g[True]-g[False]
    return g.reset_index()

def task_and_concept_accuracy(E):
    images=E[["image","y_true","y_pred"]].drop_duplicates("image")
    return pd.DataFrame([{
        "n_images":len(images), "n_species":images.y_true.nunique(),
        "task_accuracy":(images.y_true==images.y_pred).mean(),
        "concept_accuracy":(E.gt_label==E.pred_label).mean(),
    }])

SEED=1
VIS_PATH=require(CURATED/"cub70_visibility.parquet", "bash data/cub70/prepare_all.sh")
E70_PATH=require(CURATED/"cub70_eval"/f"cub70-cbm-s{SEED}.parquet",
                 "CONFIGS='cub70-cbm' SEEDS='1' bash analysis/cub70_prepare_analysis.sh")
EFULL_PATH=require(CURATED/"cub70_eval"/f"cub-cbm-s{SEED}.parquet",
                   "CONFIGS='cub-cbm' SEEDS='1' bash analysis/cub70_prepare_analysis.sh")
RAWVIS=pd.read_parquet(VIS_PATH)
COARSE=coarse_visibility(RAWVIS, threshold=.001)
E70=add_mapping(pd.read_parquet(E70_PATH))
EFULL=add_mapping(pd.read_parquet(EFULL_PATH))
J70=attach(E70,COARSE)
JFULL=attach(EFULL,COARSE)
print("ready:",len(E70),"CUB70-CBM concept rows;",RAWVIS.image_name.nunique(),"masked images")
"""),
md(r"""
## 1 · What concepts and masks actually exist?

**Question.** What are we measuring before anything is pooled?

**Variables.** CUB concept names have an attribute type before `::` and a value
after it. CUB70 separately supplies 11 masks.

**Decision rule.** Print every type, its selected values, and its mask. Whole-bird
size and shape have no valid local mask and are excluded from local grounding.
The five body-region types are kept distinct even though the released data gives
them the same body mask.
"""),
code(r"""
concepts=(E70[["concept_name","attribute_type","mask_group"]].drop_duplicates()
          .sort_values(["attribute_type","concept_name"]))
inventory=(concepts.groupby(["attribute_type","mask_group"],dropna=False)
           .agg(n_selected_concepts=("concept_name","size"),
                selected_values=("concept_name",lambda s:", ".join(x.split("::",1)[1] for x in s)))
           .reset_index())
display(inventory)
print("attribute types:",inventory.attribute_type.nunique())
print("selected concept values:",len(concepts))
print("fine masks:",len(CUB70_PARTS),CUB70_PARTS)

fig,ax=plt.subplots(figsize=(8,8))
q=inventory.sort_values("n_selected_concepts")
ax.barh(q.attribute_type.str.replace("has_","",regex=False),q.n_selected_concepts,
        color=[COLORS.get(x,"#333333") for x in q.mask_group])
ax.set_xlabel("number of selected concept values")
ax.set_title("All selected CUB concept types — before coarse pooling")
plt.tight_layout();plt.show()
"""),
md(r"""
## 2 · How many species and images are in this exact analysis?

**Question.** Is this a 200-species or 70-species result, and is coverage balanced?

**Prediction.** None. We print the population before model behavior so a later
part result cannot quietly be caused by using only a few species or images.
"""),
code(r"""
images=E70[["image","y_true"]].drop_duplicates()
species_counts=images.groupby("y_true").size().rename("n_images")
display(pd.DataFrame({
    "n_images":[len(images)],"n_species":[images.y_true.nunique()],
    "min_images_per_species":[species_counts.min()],
    "median_images_per_species":[species_counts.median()],
    "max_images_per_species":[species_counts.max()],
}))
fig,ax=plt.subplots(figsize=(10,3))
ax.bar(range(len(species_counts)),species_counts.values,color="#0072B2")
ax.set_xlabel("CUB70 species index");ax.set_ylabel("masked test images")
ax.set_title("Image coverage for every species")
plt.tight_layout();plt.show()
"""),
md(r"""
## 3 · Visibility inventory for all 11 masks

**Question.** Which regions are commonly absent in these photographs?

**Variables.** `visible` means the fine mask covers at least 0.1% of the image.
For eyes, wings, and legs, we also count whether zero, one, or both sides are
visible. We do not yet use concept labels or predictions.

**Why this comes first.** A concept cannot be tested often under natural
occlusion if its mask is almost always visible. Left/right asymmetry can also be
lost by immediately merging the two sides.
"""),
code(r"""
fine=(RAWVIS.groupby("part").agg(
    n_images=("image_name","nunique"), visible_rate=("visible","mean"),
    mean_area=("area_frac","mean"), median_area=("area_frac","median"),
).reindex(CUB70_PARTS).reset_index())
display(fine.round(4))

side_rows=[]
for group,parts in {"eye":["left_eye","right_eye"],
                    "wing":["left_wing","right_wing"],
                    "leg":["left_leg","right_leg"]}.items():
    d=RAWVIS[RAWVIS.part.isin(parts)].pivot(index="image_name",columns="part",values="visible")
    n=d.fillna(False).sum(axis=1).astype(int)
    for k,v in n.value_counts(normalize=True).sort_index().items():
        side_rows.append({"group":group,"visible_sides":k,"fraction_images":v,"n":(n==k).sum()})
SIDES=pd.DataFrame(side_rows)
display(SIDES.round(4))

fig,axes=plt.subplots(1,2,figsize=(13,4))
axes[0].bar(fine.part,fine.visible_rate,color="#0072B2")
axes[0].set_ylim(0,1);axes[0].tick_params(axis="x",rotation=45)
axes[0].set_ylabel("fraction of images with visible mask")
axes[0].set_title("All 11 CUB70 masks")
S=SIDES.pivot(index="group",columns="visible_sides",values="fraction_images").fillna(0)
S.plot.bar(stacked=True,ax=axes[1],color=["#CC79A7","#56B4E9","#009E73"])
axes[1].set_ylim(0,1);axes[1].set_ylabel("fraction of images")
axes[1].set_title("Bilateral regions: zero, one, or two sides visible")
axes[1].legend(title="visible sides")
plt.tight_layout();plt.show()
"""),
md(r"""
## 4 · Concept structure before model behavior

**Question.** How tightly is each concept tied to species identity?

**Variables.** For every exact concept, count the number of CUB70 species whose
processed label is positive. A concept positive in very few species gives a
stronger species clue than one shared by most species.

**Decision rule.** This does not prove that the model uses species pixels. It
only measures how easy a species shortcut would be.
"""),
code(r"""
species_concept=(E70.groupby(["attribute_type","concept_name","y_true"]).gt_label.mean()
                 .reset_index())
support=(species_concept.assign(pos=lambda d:d.gt_label>=.5)
         .groupby(["attribute_type","concept_name"]).pos.sum()
         .rename("n_positive_species").reset_index())
display(support.groupby("attribute_type").n_positive_species.agg(
    n_concepts="size",minimum="min",median="median",maximum="max").round(2))

types=sorted(support.attribute_type.unique())
fig,ax=plt.subplots(figsize=(10,7))
rng=np.random.default_rng(20260731)
for i,t in enumerate(types):
    v=support.loc[support.attribute_type==t,"n_positive_species"]
    mask=ATTRIBUTE_TYPE_TO_MASK.get(t)
    ax.scatter(i+rng.uniform(-.15,.15,len(v)),v,s=18,alpha=.6,color=COLORS.get(mask,"#333333"))
ax.set_xticks(range(len(types)));ax.set_xticklabels([x.replace("has_","") for x in types],rotation=65,ha="right")
ax.set_ylabel("number of CUB70 species with positive label")
ax.set_title("Species support for every exact concept")
plt.tight_layout();plt.show()
"""),
md(r"""
## 5 · Label/mask disagreement — still before model interpretation

**Question.** When a processed concept label is positive, how often is its named
mask absent?

This is not a relabel experiment. It is a count of the labels the existing CBM
was evaluated against.

**Prediction.** None. We calculate the rate separately for every exact concept.
A high rate creates more opportunity to learn a contextual shortcut, but it does
not prove the model took that shortcut.
"""),
code(r"""
EXACT70=exact_visibility_metrics(J70)
display(EXACT70.groupby("attribute_type").agg(
    n_concepts=("concept_name","size"),
    total_positive=("n_positive","sum"),
    median_hidden_rate=("label_hidden_rate","median"),
    min_hidden_rate=("label_hidden_rate","min"),
    max_hidden_rate=("label_hidden_rate","max"),
).sort_values("median_hidden_rate",ascending=False).round(3))

types=sorted(EXACT70.attribute_type.unique())
fig,ax=plt.subplots(figsize=(11,6))
rng=np.random.default_rng(5)
for i,t in enumerate(types):
    d=EXACT70[EXACT70.attribute_type==t]
    ax.scatter(i+rng.uniform(-.14,.14,len(d)),d.label_hidden_rate,
               color=COLORS.get(d.mask_group.iloc[0],"#333333"),s=20,alpha=.7)
ax.set_xticks(range(len(types)));ax.set_xticklabels([x.replace("has_","") for x in types],rotation=65,ha="right")
ax.set_ylim(0,1);ax.set_ylabel("P(mask absent | processed concept label = 1)")
ax.set_title("Label/mask disagreement for every exact concept")
plt.tight_layout();plt.show()
"""),
md(r"""
## 6 · Did the CUB70-trained CBM work normally?

**Question.** Can this model support a grounding analysis at all?

**Variables.** Species accuracy, overall concept accuracy, positive-concept
recall, and prediction spread for every exact concept.

**Decision rule.** A concept with nearly constant probability is marked as a
collapsed output. A threshold below 0.5 is not evidence of grounding if the
model simply emits the same value for visible and hidden images.
"""),
code(r"""
display(task_and_concept_accuracy(E70).round(4))
ordinary=(E70.groupby(["attribute_type","concept_name"]).agg(
    n=("prob","size"),prob_std=("prob","std"),
    concept_accuracy=("pred_label",lambda x:np.nan),
).reset_index())
acc=(E70.assign(correct=E70.pred_label==E70.gt_label)
     .groupby(["attribute_type","concept_name"]).correct.mean().rename("concept_accuracy").reset_index())
ordinary=ordinary.drop(columns="concept_accuracy").merge(acc)
pos=(E70[E70.gt_label==1].groupby(["attribute_type","concept_name"])
     .agg(positive_recall=("pred_label","mean"),positive_prob_mean=("prob","mean"),
          positive_prob_std=("prob","std"),unique_prob_6dp=("prob",lambda x:x.round(6).nunique()))
     .reset_index())
ordinary=ordinary.merge(pos,how="left")
display(ordinary.groupby("attribute_type").agg(
    n_concepts=("concept_name","size"),median_accuracy=("concept_accuracy","median"),
    median_positive_recall=("positive_recall","median"),
    median_positive_std=("positive_prob_std","median"),
    minimum_unique_scores=("unique_prob_6dp","min"),
).round(4))

fig,axes=plt.subplots(1,2,figsize=(12,5))
axes[0].scatter(ordinary.positive_prob_std,ordinary.positive_recall,s=22,alpha=.65)
axes[0].set_xlabel("score spread on positive rows");axes[0].set_ylabel("positive recall")
axes[0].set_title("Collapse guard for every exact concept")
types=sorted(ordinary.attribute_type.unique())
med=ordinary.groupby("attribute_type").positive_recall.median().reindex(types)
axes[1].bar(range(len(types)),med,color=[COLORS.get(ATTRIBUTE_TYPE_TO_MASK.get(t),"#333333") for t in types])
axes[1].set_xticks(range(len(types)));axes[1].set_xticklabels([x.replace("has_","") for x in types],rotation=70,ha="right")
axes[1].set_ylim(0,1);axes[1].set_ylabel("median positive recall")
axes[1].set_title("Ordinary positive-concept performance")
plt.tight_layout();plt.show()
"""),
md(r"""
## 7 · First model question: does visibility change the named concept?

**Question.** For a processed positive concept, is its prediction higher when
the corresponding mask is visible than when it is absent?

**Decision rule.** Each line is one exact concept. Moving upward from hidden to
visible supports visual sensitivity. A flat high line means the concept stays
high without visible named-region evidence. A flat low line may instead be a
weak or collapsed concept and must be checked against Section 6.
"""),
code(r"""
T=(EXACT70.dropna(subset=["prob_visible","prob_hidden"])
   .sort_values(["attribute_type","concept_name"]).reset_index(drop=True))
fig,ax=plt.subplots(figsize=(8,7))
for r in T.itertuples():
    c=COLORS.get(r.mask_group,"#333333")
    ax.plot([0,1],[r.prob_hidden,r.prob_visible],color=c,alpha=.28,lw=1)
    ax.scatter([0,1],[r.prob_hidden,r.prob_visible],color=c,s=9,alpha=.5)
ax.set_xticks([0,1]);ax.set_xticklabels(["mask absent","mask visible"])
ax.set_ylim(0,1);ax.set_ylabel("mean predicted probability")
ax.set_title("Every exact positive concept: hidden versus visible")
plt.tight_layout();plt.show()

display(T.groupby("attribute_type").agg(
    n_testable=("concept_name","size"),median_hidden_prob=("prob_hidden","median"),
    median_visible_prob=("prob_visible","median"),
    median_visibility_effect=("visibility_effect","median"),
).sort_values("median_visibility_effect").round(4))
"""),
md(r"""
## 8 · Odd-result test A: label conflict versus model violation

**Question.** Are concepts that are often labelled positive while hidden also
the concepts most likely to stay predicted positive while hidden?

**Variables.** The x-axis is a data property. The y-axis is model behavior.
Point size shows how many species carry the positive concept.

**Decision rule.** A relationship would make label/mask conflict a plausible
explanation. No relationship would tell us to look elsewhere. Eight pooled
parts are not used; every point is an exact concept.
"""),
code(r"""
X=EXACT70.merge(support,on=["attribute_type","concept_name"],how="left")
fig,ax=plt.subplots(figsize=(8,6))
for mask,d in X.groupby("mask_group"):
    ax.scatter(d.label_hidden_rate,d.hidden_violation,
               s=18+2*d.n_positive_species,alpha=.65,label=mask,color=COLORS.get(mask))
ax.set_xlim(-.02,1.02);ax.set_ylim(-.02,1.02)
ax.set_xlabel("data: P(mask absent | positive label)")
ax.set_ylabel("model: P(predicted positive | positive label, mask absent)")
ax.set_title("Every concept: label conflict versus hidden-part prediction")
ax.legend(title="available mask",fontsize=8)
plt.tight_layout();plt.show()
"""),
md(r"""
## 9 · Odd-result test B: amount of visible area

**Question.** When the part is visible, does more mask area strengthen the exact
concept prediction?

**Decision rule.** For every concept with enough rows, compare its lowest and
highest area quartiles. Positive `Q4-Q1` supports an area response. Near zero
does not prove context use: mask area may be a poor measure for fine color or
pattern visibility.
"""),
code(r"""
dose=[]
for (t,c),d in J70[(J70.gt_label==1)&(J70.area_frac>0)].groupby(["attribute_type","concept_name"]):
    if len(d)<20 or d.area_frac.nunique()<4: continue
    d=d.copy();d["q"]=pd.qcut(d.area_frac.rank(method="first"),4,labels=False)+1
    q=d.groupby("q").prob.mean()
    dose.append({"attribute_type":t,"concept_name":c,"n":len(d),"q4_minus_q1":q.get(4,np.nan)-q.get(1,np.nan)})
DOSE=pd.DataFrame(dose)
display(DOSE.groupby("attribute_type").q4_minus_q1.agg(n_concepts="size",median="median",minimum="min",maximum="max").round(4))
types=sorted(DOSE.attribute_type.unique())
fig,ax=plt.subplots(figsize=(11,6));rng=np.random.default_rng(9)
for i,t in enumerate(types):
    d=DOSE[DOSE.attribute_type==t]
    ax.scatter(i+rng.uniform(-.14,.14,len(d)),d.q4_minus_q1,s=20,alpha=.7,
               color=COLORS.get(ATTRIBUTE_TYPE_TO_MASK.get(t),"#333333"))
ax.axhline(0,color="black",lw=.8)
ax.set_xticks(range(len(types)));ax.set_xticklabels([x.replace("has_","") for x in types],rotation=65,ha="right")
ax.set_ylabel("mean probability in Q4 − mean probability in Q1")
ax.set_title("Visible-area response for every testable exact concept")
plt.tight_layout();plt.show()
"""),
md(r"""
## 10 · Odd-result test C: species composition

**Question.** Did a visible/hidden difference appear only because the two sets
contain different species?

**Test.** Compare visible and hidden photographs inside the same species and
exact concept, retaining only groups with both states.

**Boundary.** This removes the species main effect. It is still observational:
pose, viewpoint, and background can differ between photographs.
"""),
code(r"""
MATCH70=matched_effects(J70)
display(MATCH70.groupby("attribute_type").effect.agg(
    n_groups="size",mean="mean",median="median",
    fraction_positive=lambda x:(x>0).mean()).sort_values("median").round(4))
types=sorted(MATCH70.attribute_type.unique())
fig,ax=plt.subplots(figsize=(11,6));rng=np.random.default_rng(10)
for i,t in enumerate(types):
    d=MATCH70[MATCH70.attribute_type==t]
    ax.scatter(i+rng.uniform(-.16,.16,len(d)),d.effect,s=13,alpha=.35,
               color=COLORS.get(ATTRIBUTE_TYPE_TO_MASK.get(t),"#333333"))
ax.axhline(0,color="black",lw=.8)
ax.set_xticks(range(len(types)));ax.set_xticklabels([x.replace("has_","") for x in types],rotation=65,ha="right")
ax.set_ylabel("visible − hidden probability within species and exact concept")
ax.set_title("Species-matched visibility effects — every eligible group")
plt.tight_layout();plt.show()
"""),
md(r"""
## 11 · Odd-result test D: positive-label specificity

**Question.** Does visibility move the named positive concept specifically, or
does it move positive and negative labels together like a general pose/quality
signal?

**Decision rule.** A positive effect for `label=1` but not the same movement for
`label=0` is more consistent with named-region evidence.
"""),
code(r"""
rows=[]
for (t,lab),d in J70.groupby(["attribute_type","gt_label"]):
    v=d[d.visible].prob;h=d[~d.visible].prob
    rows.append({"attribute_type":t,"label":int(lab),"n_visible":len(v),"n_hidden":len(h),
                 "effect":v.mean()-h.mean() if len(v) and len(h) else np.nan})
SPEC=pd.DataFrame(rows)
display(SPEC.pivot(index="attribute_type",columns="label",values="effect").round(4))
P=SPEC.pivot(index="attribute_type",columns="label",values="effect")
fig,ax=plt.subplots(figsize=(11,5));x=np.arange(len(P));w=.38
ax.bar(x-w/2,P.get(0),w,label="processed label 0",color="#999999")
ax.bar(x+w/2,P.get(1),w,label="processed label 1",color="#0072B2")
ax.axhline(0,color="black",lw=.8);ax.set_xticks(x)
ax.set_xticklabels([s.replace("has_","") for s in P.index],rotation=65,ha="right")
ax.set_ylabel("visible − hidden mean probability")
ax.set_title("Is visibility response specific to positive concepts?");ax.legend()
plt.tight_layout();plt.show()
"""),
md(r"""
## 12 · Direct CBM comparison on the same masked images

Only now compare two CBMs:

- one trained on the 70 CUB species;
- one trained on all 200 CUB species.

Both are evaluated on the same 1,888 masked photographs. We repeat the same
collapse, exact-concept violation, and species-matched checks. This comparison
does not involve MCBM or relabeling.

**Boundary.** Their classification heads solve different species tasks, so task
accuracy is reported as a guard, not treated as the causal explanation for a
concept difference.
"""),
code(r"""
display(pd.concat([
    task_and_concept_accuracy(E70).assign(model="CUB70-trained CBM"),
    task_and_concept_accuracy(EFULL).assign(model="full-CUB-trained CBM"),
],ignore_index=True).set_index("model").round(4))

EXACTFULL=exact_visibility_metrics(JFULL)
PAIR=(EXACT70.merge(EXACTFULL,on=["attribute_type","concept_name","mask_group"],
                    suffixes=("_70","_full")))
fig,axes=plt.subplots(1,2,figsize=(12,5))
for mask,d in PAIR.groupby("mask_group"):
    axes[0].scatter(d.hidden_violation_full,d.hidden_violation_70,
                    s=24,alpha=.65,label=mask,color=COLORS.get(mask))
axes[0].plot([0,1],[0,1],"k--",lw=.8)
axes[0].set_xlabel("full-CUB CBM hidden violation")
axes[0].set_ylabel("CUB70 CBM hidden violation")
axes[0].set_title("Same exact concepts and masked images")

axes[1].scatter(PAIR.prob_std_full,PAIR.prob_std_70,s=24,alpha=.6,color="#0072B2")
lim=max(PAIR.prob_std_full.max(),PAIR.prob_std_70.max())
axes[1].plot([0,lim],[0,lim],"k--",lw=.8)
axes[1].set_xlabel("full-CUB CBM positive-score spread")
axes[1].set_ylabel("CUB70 CBM positive-score spread")
axes[1].set_title("Collapse/spread guard")
axes[0].legend(fontsize=8)
plt.tight_layout();plt.show()

MFULL=matched_effects(JFULL)
M70=(MATCH70.groupby(["attribute_type","concept_name"]).effect.mean().rename("effect_70"))
MF=(MFULL.groupby(["attribute_type","concept_name"]).effect.mean().rename("effect_full"))
MP=pd.concat([M70,MF],axis=1).dropna().reset_index()
fig,ax=plt.subplots(figsize=(6,6))
ax.scatter(MP.effect_full,MP.effect_70,s=24,alpha=.6,color="#0072B2")
lo=min(MP.effect_full.min(),MP.effect_70.min());hi=max(MP.effect_full.max(),MP.effect_70.max())
ax.plot([lo,hi],[lo,hi],"k--",lw=.8)
ax.axhline(0,color="black",lw=.5);ax.axvline(0,color="black",lw=.5)
ax.set_xlabel("full-CUB CBM matched visibility effect")
ax.set_ylabel("CUB70 CBM matched visibility effect")
ax.set_title("Same exact-concept control in both CBMs")
plt.tight_layout();plt.show()
"""),
md(r"""
## 13 · Stop here before MCBM

The executed notebook must now be reviewed one image at a time.

For every figure record, in simple language:

1. the question;
2. what was compared;
3. what is literally visible;
4. another possible explanation;
5. the next test that separates those explanations;
6. the smallest conclusion currently allowed.

Do not write “tail is the CUB problem” unless the exact CUB plots establish it.
Do not call a constant score grounding. Do not discuss relabeling or minimality
until this CBM-only exploration is complete.
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {OUT} with {len(cells)} cells")

