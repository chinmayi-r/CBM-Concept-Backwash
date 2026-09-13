#!/usr/bin/env python3
"""Build the fail-closed official full-CUB Koh Joint report."""
from __future__ import annotations
import hashlib,json,textwrap
from pathlib import Path

CURATED=Path(__file__).resolve().parents[1]
OUT=CURATED/"notebooks"/"07_full_cub_cbm.ipynb"

def make(kind,tag,text):
    body=textwrap.dedent(text).strip("\n")+"\n"
    d={"cell_type":kind,"id":f"cub07-{tag}-{hashlib.sha1(body.encode()).hexdigest()[:10]}","metadata":{},"source":body.splitlines(True)}
    if kind=="code":d.update(execution_count=None,outputs=[])
    return d
def md(tag,text):return make("markdown",tag,text)
def code(tag,text):return make("code",tag,text)
def before(n,title,q,formula,inputs,read,example):return md(f"before-{n}",f"""
## {n} · {title}

**Question and prediction.** {q}

**Exact quantities.** {formula}

**Inputs/model.** {inputs}

**Axes, groups, and exclusions.** {read}

**Example.** {example}
""")
def after(n,support,alt,test,nextq):return md(f"after-{n}",f"""
### Review chain for Figure {n}

- **Literal observation:** the complete numerical table is printed above the figure.
- **What it supports:** {support}
- **Plausible alternative:** {alt}
- **Discriminating test:** {test}
- **Limited conclusion:** Full CUB has no accepted donor-part swap, so this is not a measured donor/source backwash rate.
- **Next question:** {nextq}
""")

cells=[
md("title",r"""
# 07 · Full CUB Standard CBM — what survives at 200 species?

This is a separate final stage, not extra rows appended to CUB70. It requires the
official Koh-architecture Joint ResNet-50 continuation through epoch 600 and
its successful manifest. It never loads the legacy `minimal_cbm` CBM.

All 5,794 test photographs support ordinary model health and species-information
analyses. Released segmentation masks exist only for the CUB70 subset, so every
spatial or visible/hidden result prints the smaller joined population.

The chapter asks three practical questions:

1. Is the full 200-species model healthy?
2. Do its concept scores contain and use extra species information?
3. On the mask-covered subset, are its concept scores spatially concentrated on
   the named region, and do natural visibility/context patterns resemble CUB70?
"""),
md("variables",r"""
## Model and notation

The image encoder emits 112 raw concept logits `z`. The unchanged saved species
head is one linear map `Wz+b` over all 112 raw logits. Training uses Koh Joint's
normalized task loss plus `0.01 × concept loss`; no sigmoid is inserted before
the species head.

- `c`: known 0/1 concept label.
- `z`: raw concept logit. This is the grounding quantity.
- `c_hat = 1[z>0]`: thresholded concept prediction, used for accuracy only.
- `v`: released mapped mask is visible; this exists only on the CUB70 mask subset.

Natural visibility and Grad-CAM remain observational/post-hoc. The failed CUB
paste/deletion pilots remain in the methods appendix and are not revived here.
"""),
code("setup",r"""
import os,sys,json
from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from IPython.display import display,Image as DisplayImage
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,log_loss
CURATED=Path(os.environ["CURATED_DATA"]);CWD=Path.cwd();REPO=CWD if (CWD/"analysis").is_dir() else CWD.parent
sys.path.insert(0,str(REPO/"data"/"cub70"))
from cub70_parts import ATTRIBUTE_TYPE_TO_MASK
from relabel_cub_with_cub70 import coarse_visibility
ORDER=["head","eye","beak","neck","body","wing","leg","tail"]
COLORS={"head":"#56B4E9","eye":"#CC79A7","beak":"#E69F00","neck":"#009E73","body":"#0072B2","wing":"#D55E00","leg":"#777777","tail":"#F0E442"}
def need(p,msg):
 p=Path(p)
 if not p.is_file():raise FileNotFoundError(f"{p}\n{msg}")
 return p
def norm(E):
 E=E.copy();E["image"]=E.image.map(lambda x:Path(str(x)).stem);E["pred_label"]=(E.z>0).astype(int)
 E["attribute_type"]=E.concept_name.map(lambda x:str(x).split("::",1)[0]);E["part"]=E.attribute_type.map(ATTRIBUTE_TYPE_TO_MASK);return E

FULL_ROOT=CURATED/"koh_joint_resnet_decay_continuation_v1"/"cub"/"standard"/"seed1"
manifest=json.loads(need(FULL_ROOT/"SUCCESS.json","Complete entry 11 before rendering Full CUB.").read_text())
meta=manifest.get("metadata",{});expected={"framework":"koh_joint","backbone":"resnet50","dataset":"cub","labels":"standard","seed":"1"}
bad={k:(meta.get(k),v) for k,v in expected.items() if str(meta.get(k))!=str(v)}
if bad:raise RuntimeError(f"full-CUB manifest mismatch: {bad}")
FULL=norm(pd.read_parquet(need(FULL_ROOT/"final_test.parquet","Official full-CUB export is incomplete.")))
K70_ROOT=CURATED/"koh_joint_resnet_v1"/"cub70"/"standard"/"seed1"
K70=norm(pd.read_parquet(need(K70_ROOT/"final_test.parquet","Render notebook 05 prerequisites first.")))
SP=need(CURATED/"cub_koh_spatial_v1"/"full_cub_standard_s1"/"SUCCESS.json","Run notebooks/run_cub_koh_spatial_audit.sh full").parent
GRAD=pd.read_csv(need(SP/"gradcam_summary.csv","rerun spatial audit"));HEAD=pd.read_csv(need(SP/"saved_head_use.csv","rerun spatial audit"));EXAMPLES=need(SP/"gradcam_examples.png","rerun spatial audit")
RAWVIS=pd.read_parquet(need(CURATED/"cub70_visibility.parquet","prepare CUB70 masks"));V=coarse_visibility(RAWVIS,.001).rename(columns={"image_name":"image","coarse":"part"});V["image"]=V.image.map(lambda x:Path(str(x)).stem)
JF=FULL[FULL.part.notna()].merge(V[["image","part","area_frac","visible"]],on=["image","part"],how="inner",validate="many_to_one")
J70=K70[K70.part.notna()].merge(V[["image","part","area_frac","visible"]],on=["image","part"],how="inner",validate="many_to_one")
print("official full-CUB metadata:",meta);print("full images",FULL.image.nunique(),"species",FULL.y_true.nunique(),"concepts",FULL.concept_name.nunique());print("mask-joined full-CUB images",JF.image.nunique())
"""),

before("1","Is the completed 200-species model healthy?","Ordinary task/concept behavior must be usable before spatial interpretation.",
"spread_j=Q95(z_j)-Q05(z_j); label separation_j=median(z|c=1)-median(z|c=0); balanced accuracy_j=(TPR+TNR)/2.",
"All 5,794 official Full-CUB test photographs. No masks and no diagnostic model are needed.",
"Every row is one of 112 exact concepts. Zero spread within 1e-8 is exact collapse; positive separation and balanced accuracy above 0.5 are healthy directions.",
"If 90 of 100 positives and 80 of 100 negatives are correct, balanced accuracy=(.90+.80)/2=.85."),
code("fig1",r"""
rows=[]
for concept,d in FULL.groupby("concept_name"):
 pos=d.gt_label==1;pred=d.z>0;tpr=pred[pos].mean();tnr=(~pred[~pos]).mean()
 rows.append({"concept_name":concept,"part":d.part.iloc[0],"spread":d.z.quantile(.95)-d.z.quantile(.05),"label_separation":d[pos].z.median()-d[~pos].z.median(),"balanced_accuracy":np.mean([tpr,tnr]),"positive_recall":tpr})
HEALTH=pd.DataFrame(rows);display(HEALTH.round(4));images=FULL[["image","y_true","y_pred"]].drop_duplicates("image");print("task accuracy",(images.y_true==images.y_pred).mean())
fig,axes=plt.subplots(1,3,figsize=(15,10));y=np.arange(len(HEALTH))
for ax,col,title in zip(axes,["spread","label_separation","balanced_accuracy"],["raw-z spread","positive-minus-negative median z","balanced accuracy"]):
 ax.scatter(HEALTH[col],y,s=8);ax.set_title(title);ax.set_yticks([])
axes[1].axvline(0,color="black",ls="--");axes[2].axvline(.5,color="black",ls="--");fig.suptitle("Figure 1 · Full-CUB exact-concept health");plt.tight_layout();plt.show()
"""),
after("1","The later analysis is interpretable only for healthy, non-collapsed exact outputs.","High pooled health can hide a small number of failed concepts.","Use the printed exact table and independent seeds.","How much extra species information exists in raw z?"),

before("2","Do raw concept magnitudes reveal species beyond known labels?","Raw z above the known-label control means the learned representation contains extra species structure.",
"Fit identical held-out species probes; conditional log-loss gain=logloss(labels)-logloss(raw z). Positive means raw z predicts species better.",
"All Full-CUB test images; newly trained diagnostic classifiers only. The saved CBM head is tested separately in Figure 3.",
"Grey is known labels; blue is raw z; dashed is blind 1/200 chance. The same split is used for both.",
"Label log loss 3.0 and raw-z log loss 2.4 gives gain +0.6."),
code("fig2",r"""
Z=FULL.pivot(index="image",columns="concept_name",values="z").sort_index(axis=1);C=FULL.pivot(index="image",columns="concept_name",values="gt_label").loc[Z.index,Z.columns]
y=FULL[["image","y_true"]].drop_duplicates("image").set_index("image").loc[Z.index].y_true;tr,te=train_test_split(np.arange(len(Z)),test_size=.30,random_state=20260913,stratify=y)
probe=[]
for label,X in [("known 0/1 labels",C.to_numpy()),("raw z",Z.to_numpy())]:
 fit=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=5000,random_state=20260913));fit.fit(X[tr],y.iloc[tr]);pred=fit.predict(X[te]);prob=fit.predict_proba(X[te]);probe.append({"input":label,"accuracy":accuracy_score(y.iloc[te],pred),"log_loss":log_loss(y.iloc[te],prob,labels=fit[-1].classes_)})
PROBE=pd.DataFrame(probe);display(PROBE.round(4));print("conditional log-loss gain",PROBE.set_index("input").loc["known 0/1 labels","log_loss"]-PROBE.set_index("input").loc["raw z","log_loss"])
fig,ax=plt.subplots(figsize=(6,4));ax.bar(PROBE.input,PROBE.accuracy,color=["#999999","#0072B2"]);ax.axhline(1/200,color="black",ls="--");ax.set_ylabel("held-out 200-species accuracy");ax.set_title("Figure 2 · Species information available in Full-CUB concept vectors");plt.show()
"""),
after("2","It measures information availability beyond the official label pattern.","A newly fitted probe can exploit information that the saved class head ignores.","Reuse the original saved Wz+b head after removing within-label magnitudes.","Does the actual saved head use it?"),

before("3","Does the unchanged saved species head use within-label magnitude variation?","If it does, replacing magnitudes while preserving every 0/1 side will move class logits or probabilities.",
"Five cross-fitted image folds replace each z by the other folds' mean for the same concept and label; rerun saved Wz+b. Probability mass moved=.5 sum_k |p_before,k-p_after,k|.",
"Precomputed from the official checkpoint and all 5,794 test images; the original saved head is reused, not retrained.",
"The black all-112 bar replaces every concept. Colored bars replace one coarse group. Larger means more actual saved-head reliance on within-label magnitude.",
"Changing one z from 7 to its positive-label mean 3 preserves c=1 but removes four units of within-label variation."),
code("fig3",r"""
display(HEAD.round(5));q=HEAD.copy();q["order"]=q.block.map({"all 112":-1,**{p:i for i,p in enumerate(ORDER)}});q=q.sort_values("order");colors=["#333333" if b=="all 112" else COLORS[b] for b in q.block]
fig,axes=plt.subplots(1,2,figsize=(13,4));axes[0].bar(q.block,q.mean_probability_mass_moved,color=colors);axes[1].bar(q.block,q.top1_change_rate,color=colors)
axes[0].set_ylabel("mean probability mass moved");axes[1].set_ylabel("top-species change rate")
for ax in axes:ax.tick_params(axis="x",rotation=45)
fig.suptitle("Figure 3 · Actual use of within-label magnitudes by saved Wz+b");plt.tight_layout();plt.show()
"""),
after("3","This directly measures use by the saved class head, which is distinct from decodability.","Class probabilities can be saturated, making probability movement small despite logit changes.","Read probability, top-one, and absolute class-logit changes together.","Where are concept scores spatially sensitive?"),

before("4","Do concept-specific gradients concentrate on the named released mask?","A localized concept should place more positive sensitivity inside its named region than a uniform map would.",
"Grad-CAM G=max(0,sum_k alpha_k A_k), alpha_k=mean_uv(d z_j/d A_kuv); enrichment=(mass of normalized G inside mask)/(mask area fraction).",
"Frozen official Full-CUB model on deterministic positive-labelled, visibly masked photographs from the CUB70 mask subset. No training.",
"Examples show photograph, mask, positive map, absolute map. Population bars report enrichment, pointing, and equal-area overlap with exact counts.",
"A 5%-area mask receiving 25% of positive map mass has enrichment 5; value 1 equals a uniform map."),
code("fig4",r"""
display(DisplayImage(filename=str(EXAMPLES)));display(GRAD.round(4));q=GRAD.set_index("mask_group").reindex(ORDER).dropna(how="all")
fig,axes=plt.subplots(1,3,figsize=(15,4));
for ax,col,title in zip(axes,["median_positive_enrichment","positive_pointing_rate","median_positive_equal_area_iou"],["mask enrichment","maximum inside mask","equal-area overlap"]):
 ax.bar(q.index,q[col],color=[COLORS[x] for x in q.index]);ax.set_title(title);ax.tick_params(axis="x",rotation=45)
 if col=="median_positive_enrichment":ax.axhline(1,color="black",ls="--")
fig.suptitle("Figure 4 · Full-CUB concept Grad-CAM on CUB70 mask-covered photographs");plt.tight_layout();plt.show()
"""),
after("4","It provides post-hoc spatial evidence at the exact concept output.","Grad-CAM resolution and coarse masks can reduce apparent overlap.","A segment-routed model with exact forward contributions and insertion/deletion validation is stronger.","Do natural visibility and hidden context agree with CUB70?"),

before("5","Do mask-covered Full-CUB photographs reproduce the CUB70 visibility/context pattern?","Agreement would show that observations are not unique to training on 70 species; disagreement limits transfer.",
"Within each model/concept standardize z. V_j=mean(z_std|c=1,v=1)-mean(z_std|c=1,v=0). H_j=mean(z_std|c=1,v=0)-mean(z_std|c=0,v=0).",
"Official CUB70 and Full-CUB Koh models, identical released masks, and only exact concepts/images with the required groups. No diagnostic is trained.",
"Points compare the same exact-concept definition between models. The diagonal is equality in within-concept SD units; sign disagreements cross axes.",
"Full V=+0.8 and CUB70 V=-0.2 is a sign disagreement, not a small scale difference."),
code("fig5",r"""
def effects(J,label):
 J=J.copy();J["z_std"]=J.groupby("concept_name").z.transform(lambda s:(s-s.mean())/s.std(ddof=0) if s.std(ddof=0)>1e-8 else np.nan);rows=[]
 for concept,d in J.groupby("concept_name"):
  pv=d[(d.gt_label==1)&d.visible].z_std;ph=d[(d.gt_label==1)&(~d.visible)].z_std;nh=d[(d.gt_label==0)&(~d.visible)].z_std
  rows.append({"concept_name":concept,"part":d.part.iloc[0],f"V_{label}":pv.mean()-ph.mean() if len(pv)>=5 and len(ph)>=5 else np.nan,f"H_{label}":ph.mean()-nh.mean() if len(ph)>=5 and len(nh)>=5 else np.nan})
 return pd.DataFrame(rows)
PAIR=effects(JF,"full").merge(effects(J70,"cub70"),on=["concept_name","part"]);display(PAIR.round(4))
fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,x,y,title in [(axes[0],"V_cub70","V_full","visibility association"),(axes[1],"H_cub70","H_full","hidden context gap")]:
 for p,d in PAIR.groupby("part"):ax.scatter(d[x],d[y],label=p,color=COLORS[p],alpha=.7)
 lo=np.nanmin(PAIR[[x,y]].to_numpy());hi=np.nanmax(PAIR[[x,y]].to_numpy());ax.plot([lo,hi],[lo,hi],"k--");ax.axhline(0,color="grey",lw=.5);ax.axvline(0,color="grey",lw=.5);ax.set_xlabel("CUB70 model");ax.set_ylabel("Full-CUB model");ax.set_title(title)
axes[0].legend(ncol=2,fontsize=8);fig.suptitle("Figure 5 · Same observational quantities, separate official Koh models");plt.tight_layout();plt.show()
"""),
after("5","Agreement supports transfer of a named observational pattern across training populations.","Both models share data conventions and many photographs, so agreement is not independent replication.","Train independent seeds and obtain a calibrated localized intervention.","What can a researcher do with the result?"),

md("conclusion",r"""
## Final practical answer

The report separates three knobs:

1. **Availability:** can a new probe recover species from raw concept scores
   beyond known labels?
2. **Use:** does the model's own saved `Wz+b` class head change when that extra
   within-label magnitude is removed?
3. **Localization:** does concept-specific spatial sensitivity concentrate in
   the named mask?

High availability alone is not backwash. Low use can protect the task head while
the concept explanation remains questionable. Good Grad-CAM overlap is useful
support but not a causal donor-part test.

The actionable next model is a spatially routed or spatially regularized CBM:
make each concept depend on its named segment, retain the ordinary concept/task
losses, and compare against this frozen baseline. The strongest design inspired
by SEG-MIL-CBM makes the prediction an exact sum of learned segment
contributions, following
[SEG-MIL-CBM](https://arxiv.org/abs/2510.04180v2). Its deletion/insertion validation can then test whether the
ranked segments truly control the output. That is stronger than merely drawing
a post-hoc heatmap.
"""),
md("limits",r"""
## Causal and completion boundary

- Full CUB is complete only if the official continuation manifest and final
  export loaded at the top of this notebook.
- The CUB70 masks cover only a subset of Full-CUB photographs and are coarse for
  many attributes.
- Natural visibility is observational. Grad-CAM is post-hoc.
- The failed deletion, randomized-patch, and beak/tail paste methods remain
  `METHOD NOT CALIBRATED` or `VALID TEST, NO SUPPORT`; they are not silently
  promoted into the main result.
- Seed-level replication is still required for uncertainty.
"""),
]

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.10"}},"nbformat":4,"nbformat_minor":5}
OUT.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+"\n",encoding="utf-8")
print(f"wrote {OUT} with {len(cells)} cells")
