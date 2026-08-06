#!/usr/bin/env python3
"""Build the standard FunnyBird MCBM report from the locked 03/06 roadmap."""
from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks/03_funnybirds_mcbm.ipynb"


def _src(s: str) -> list[str]:
    return (textwrap.dedent(s).strip("\n") + "\n").splitlines(keepends=True)


def md(tag: str, s: str) -> dict:
    s = textwrap.dedent(s).strip("\n") + "\n"
    return {"cell_type": "markdown", "id": f"03-{tag}-{hashlib.sha1(s.encode()).hexdigest()[:8]}",
            "metadata": {}, "source": s.splitlines(keepends=True)}


def code(tag: str, s: str, alt: str) -> dict:
    s = "# ALT: " + alt + "\n" + textwrap.dedent(s).strip("\n") + "\n"
    return {"cell_type": "code", "id": f"03-{tag}-{hashlib.sha1(s.encode()).hexdigest()[:8]}",
            "metadata": {"alt": alt}, "execution_count": None, "outputs": [],
            "source": s.splitlines(keepends=True)}


REVIEWS = {
1: """- **Literal observation:** Every gamma from `0` through `5` has exactly one validated seed, 5,000 directed replacements, all five parts, both directions, and all 50 source and donor species. All rows come from `swap_fixed_v2_attempt2`.
- **Alternative explanations:** Equal row counts alone would not prove equal pixels.
- **Discriminating test:** The fixed-render validator already required shared render hashes, intervention diversity, and the semantic renderer preflight before these CSVs were accepted.
- **Limited conclusion:** The seed-1 gamma comparison uses the same accepted counterfactual population. It is not multi-seed causal replication.
- **Next question:** Did gamma actually change the intended internal representation without breaking the model?""",
2: """- **Literal observation:** Mean target RMSE falls from `20.07` at `gamma=0` to `1.16` at `0.1` and about `0.39` at `3–5`; within-label slot spread falls similarly. Species accuracy stays near `0.74–0.75` and concept balanced accuracy near `0.99`. The unfinished `gamma=5, seed=2` artifact contains non-finite slots and is explicitly excluded as **INVALID OUTPUT**.
- **Alternative explanations:** Smaller raw values alone do not imply better grounding.
- **Discriminating test:** Figures 4–8 apply the same controlled pixel replacements after confirming ordinary model health here.
- **Limited conclusion:** Gamma strongly enforces the implemented minimality target without generally destroying task or concept prediction.
- **Next question:** Does the MCBM architecture already change the standard-CBM baseline before gamma is turned on?""",
3: """- **Literal observation:** MCBM `gamma=0` has lower tail (`0.506` versus `0.608`) and beak (`0.415` versus `0.537`) controlled-backwash rates than standard CBM, nearly identical eye, and similarly low wing/foot rates.
- **Alternative explanations:** MCBM `gamma=0` is not mathematically identical to standard CBM; architecture and optimization differ even without the representation penalty.
- **Discriminating test:** Treat `gamma=0` as the MCBM starting point and test changes across gamma on identical fixed renders.
- **Limited conclusion:** The MCBM architecture modestly changes the baseline but retains the same broad pattern: tail/eye are difficult and wing/foot are strong.
- **Next question:** Does increasing gamma preserve or weaken the direct donorward response?""",
4: """- **Literal observation:** Mean `response_delta` is positive for every part and gamma, so every model responds to the inserted pixels. Tail response falls from `18.20` to `6.27` raw-logit units; its median fraction of the original deficit closed falls from `0.82` to `0.45`. Wing and foot remain large and usually close more than their full starting deficit.
- **Alternative explanations:** Raw-logit scales change with gamma, but the within-model deficit-closed panel shows the same tail weakening.
- **Discriminating test:** Inspect the final margin and the exact controlled-backwash predicate rather than equating positive movement with success.
- **Limited conclusion:** Minimality preserves some tail sensitivity but suppresses it relative to the model’s own starting deficit.
- **Next question:** After this donorward movement, which concept finishes higher?""",
5: """- **Literal observation:** Tail median `m_cf` is negative at every gamma (`-4.40` to `-7.31`), and tail controlled backwash rises from `0.506` at gamma zero to `0.666–0.780` after minimality. Wing and foot remain strongly positive with low backwash. Beak improves at gamma `1–5`; eye improves most at gamma `5`.
- **Alternative explanations:** A pooled direction error or occluded inserted parts could inflate these rates.
- **Discriminating test:** Figures 6 and 7 separate direction and visibility; Figure 8 checks the exact inserted value.
- **Limited conclusion:** Minimality is not a general grounding repair. In seed 1 it worsens the tail outcome while improving some beak/eye settings.
- **Next question:** Is the tail result present in both reciprocal swap directions?""",
6: """- **Literal observation:** Forward and backward panels have the same qualitative structure at every gamma: high tail backwash, low wing/foot backwash, and intermediate beak/eye. Tail differs by at most about seven percentage points between directions at a given gamma.
- **Alternative explanations:** Some beak and eye direction differences remain, so their pooled values should not be read as exact symmetric effects.
- **Discriminating test:** The tail conclusion requires both directions to remain high, which they do.
- **Limited conclusion:** Opposite swap directions are not cancelling to create the tail result, and reversed bookkeeping is not a plausible explanation.
- **Next question:** Does restricting to clearly visible inserted parts remove the failures?""",
7: """- **Literal observation:** Requiring at least 100 inserted pixels lowers gamma-zero tail backwash from `0.506` to `0.397`, so visibility explains part of the baseline. It does not remove the problem: large visible tails still have rates `0.644–0.787` for gamma `0.1–5`. Large-part filtering helps beak and eye at several settings and leaves wing/foot low.
- **Alternative explanations:** Visibility strata are selected subsets and can differ in species/value composition; the higher visible-only tail rate is not itself a causal visibility effect.
- **Discriminating test:** Exact-value recognition and value/species adjustment are examined next.
- **Limited conclusion:** Occlusion contributes, but it cannot explain why minimality leaves or increases tail backwash among large visible insertions.
- **Next question:** Is the donor merely losing to the source, or is the model confused about the exact inserted value?""",
8: """- **Literal observation:** Tail exact-value recognition is `0.278` at gamma zero and only `0.109–0.202` after minimality. Wing remains `0.811–0.881` and foot `0.926–0.991`. Beak reaches about `0.75` at gamma `1–3`; eye reaches `0.735` at gamma `5`.
- **Alternative explanations:** Exact-value argmax is stricter than the two-slot margin and could expose confusion with a third value.
- **Discriminating test:** Because wing and foot processed by the same code remain strongly diagonal, the tail decline is not a universal argmax or compression artifact.
- **Limited conclusion:** Minimality has part-specific effects: it worsens exact tail attribution while sometimes improving beak and eye.
- **Next question:** After exact source/donor value difficulty is removed, does unchanged source species still organize the margin?""",
9: """- **Literal observation:** Raw residual spreads shrink with gamma partly because the entire score scale shrinks. After standardization, tail source-species spread does not decline (`0.425` at gamma zero and `0.494` at gamma five). Beak remains around `0.54–0.61`, and eye is often larger (`0.58–0.72`). Wing and foot show some reduction at high gamma.
- **Alternative explanations:** Source species is tied to the unchanged body, pose, and other parts; this analysis does not manipulate species independently.
- **Discriminating test:** Exact source and donor values are matched before residualizing, and the right panel removes gamma/part scale.
- **Limited conclusion:** Minimality does not generally erase source-species/body organization after exact value difficulty. The residual is observational, not a standalone causal species effect.
- **Next question:** Is species information still recoverable from the learned concept representation?""",
10: """- **Literal observation:** The full learned concept-output vector predicts held-out species at `0.978–0.998` for every gamma. Tail-only accuracy is `0.186–0.226` at low gamma, near the approximate nine-tail-value structural control `9/50=0.18`, and reaches `0.268–0.272` at gamma `3–5`.
- **Alternative explanations:** Tail values themselves identify species buckets, so tail accuracy above blind `1/50` is not automatically extra leakage. High-gamma tail results currently have one seed.
- **Discriminating test:** Compare tail with its processed-label structural control and use the controlled swap, not decoding alone, for grounding.
- **Limited conclusion:** The full concept representation remains strongly species-informative under minimality. Extra tail-within-bucket information is modest and provisional.
- **Next question:** Does the developed matched-recall diagnostic independently show species-dependent recognition?""",
11: """- **Literal observation:** No authoritative FunnyBird MCBM recall-v4 table was present, so no recall figure or numerical gamma claim was produced.
- **Alternative explanations:** Reusing older recall outputs would mix pairing rules or present CUB/MCBM numbers under the wrong method.
- **Discriminating test:** Generate the all-positive-species FunnyBird pairing with the recall-v4 vectorized bootstrap before adding this supporting diagnostic.
- **Limited conclusion:** This item is **INCOMPLETE**. The missing optional recall diagnostic does not invalidate the accepted fixed-render swap evidence.
- **Next question:** Does final concept success have a measurable effect on the donor species head?""",
12: """- **Literal observation:** For every part, donor-species probability is near zero when `m_cf` is negative and generally rises as the donor concept margin becomes strongly positive. Tail occupies mostly negative or modest positive margins and its binned donor-species probability stays below about `0.08`; wing reaches about `0.16`, with beak/foot/eye intermediate.
- **Alternative explanations:** Replacing one part does not turn the whole bird into the donor species, and bins are descriptive rather than a randomized dose.
- **Discriminating test:** The monotone association is shown separately by part so strong wing/foot rows cannot hide tail.
- **Limited conclusion:** Successful concept attribution is associated with more donor-class support, but the absolute downstream probability is small. Grounding failure is clearer than its class-level cost.
- **Next question:** Which findings have independent model-seed replication?""",
13: """- **Literal observation:** Health has three valid seeds at gamma `0, 0.3, 1, 3`, two at `0.1`, and one at `5`. Every validated fixed-render gamma result has only seed 1. The unfinished gamma-five seed-two artifact is not counted.
- **Alternative explanations:** Thousands of swaps reduce within-model sampling noise but do not replace independently trained models.
- **Discriminating test:** Replicate the fixed-render replay on independently trained seeds before estimating gamma uncertainty.
- **Limited conclusion:** Compression/health is replicated for most gammas; the causal gamma pattern is a broad seed-1 result and must be labelled provisional numerically.
- **Next question:** Taken together, did compression and grounding move in the predicted repair direction?""",
14: """- **Literal observation:** Gamma reduces target RMSE from `20.07` to below `1.2` while concept balanced accuracy remains about `0.99`. At the same time, tail donorward response drops from `18.20` to `6.27`, controlled backwash rises from `0.51` to `0.67–0.78`, and exact-value error rises from `0.72` to `0.80–0.89`. Wing and foot remain strong; beak and eye improve at selected higher gammas.
- **Alternative explanations:** The exact numerical gamma ordering may change with new fixed-render seeds, and the source-species residual is observational.
- **Discriminating test:** Compression, health, response, final margin, exact-value recognition, direction, and visibility all have to agree before calling minimality a repair.
- **Limited conclusion:** **Accepted for the limited seed-1 claim:** minimality successfully compresses the representation but is not sufficient to create part grounding. It suppresses the already weak tail response while leaving a source/body preference; its effects differ by part.
- **Next question:** Notebook 06 now asks the same questions on CUB using raw-logit natural-visibility and species-matched approximations, explicitly stopping where no clean swap exists.""",
}


def review(n: int) -> dict:
    return md(f"review{n}", f"""
    **Figure {n} review.**

    {REVIEWS[n]}
    """)


cells: list[dict] = []
cells += [md("title", r"""
# 03 · FunnyBird standard MCBM — does minimality repair concept grounding?

Notebook 02 discovered controlled concept backwash in a standard CBM. This
notebook changes one training ingredient: MCBM penalizes information in each
internal concept slot that is unnecessary for its binary label. The question is
not merely whether scores become smaller. It is whether the *same validated
part replacements* become more correctly attributed.

This is the non-RLv2 MCBM stage. RLv2 is a later causal label test.
"""), code("setup", r"""
import os, re, glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display
CURATED=Path(os.environ["CURATED_DATA"])
REPO=Path.cwd() if (Path.cwd()/"analysis").is_dir() else Path.cwd().parent
sys.path.insert(0,str(REPO/"analysis"))
from minimal_cbm_scores import concept_logits_from_saved_latent, validate_saved_probabilities
ORDER=["tail","wing","beak","foot","eye"]
COLORS=dict(tail="#7B3294",wing="#0080C6",beak="#E66101",foot="#009E73",eye="#CC79A7")
GAMMAS=[0.,.1,.3,1.,3.,5.]
plt.rcParams.update({"figure.dpi":120,"axes.grid":False})

def heat(ax, table, title, cbar, vmin=None, vmax=None, cmap="viridis", fmt=".2f"):
    a=table.astype(float).values
    if vmin is None: vmin=np.nanmin(a)
    if vmax is None: vmax=np.nanmax(a)
    im=ax.imshow(a,aspect="auto",cmap=cmap,vmin=vmin,vmax=vmax)
    ax.set_xticks(range(len(table.columns))); ax.set_xticklabels(table.columns)
    ax.set_yticks(range(len(table.index))); ax.set_yticklabels(table.index)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if np.isfinite(a[i,j]): ax.text(j,i,format(a[i,j],fmt),ha="center",va="center",fontsize=8)
    ax.set_title(title); plt.colorbar(im,ax=ax,label=cbar,fraction=.046)
""", "Imports and shared plotting definitions; no scientific figure."),
md("model", r"""
## Model, loss, and notation

For image `x_i` and exact concept `j`, the encoder emits an internal scalar slot
`h_ij`. The minimal-CBM source code calls this tensor `z`, but it is **not yet
the concept logit**. The learned concept head produces the raw concept logit

`z_ij = concept_head_j(h_ij)`.

`p_ij=sigmoid(z_ij)` is a probability and
`c_hat_ij=1[z_ij>0]` is the thresholded prediction. Grounding figures use raw
`z`; probabilities appear only for thresholded performance or the species head.
The species head reads the vector of internal slots `h`.

MCBM trains with

`L = L_species + beta L_concept + gamma L_rep`, where

`L_rep = sum_j 0.2 mean((h_ij-(6c_ij-3))^2)`.

Thus a positive label pulls `h_ij` toward `+3` and a negative label toward `-3`.
This removes within-label detail, but it does not specify which pixels produced
the slot. A body/species shortcut can satisfy this loss perfectly.

For the controlled replacement:

- `m_orig=z_donor,orig-z_source,orig`;
- `m_cf=z_donor,cf-z_source,cf`;
- `response_delta=m_cf-m_orig`;
- controlled backwash is `response_delta>0 and m_cf<0`.

Example: `m_orig=-20`, `m_cf=-5` gives `response_delta=+15`. The donor pixels
changed the answer in the right direction, but the old source still wins by 5.
""")]

cells += [md("f1", r"""
## Figure 1 — What data, checkpoints, renders, gammas, and seeds are actually compared?

**Question.** Is every gamma evaluated on the same validated pixels, and how many
independent seeds support each result?

**How to read.** Each row is one gamma/seed CSV. `rows` is the number of directed
part replacements; `render_ids` counts distinct fixed counterfactual images.
The checkpoint column is the model recorded by the evaluator. This is an
inventory, not a result. The fixed-render gamma curve is currently seed 1 unless
additional validated CSVs are present.
"""), code("f1", r"""
FIXED=CURATED/"swap_fixed_v2_attempt2"
if not FIXED.exists(): raise FileNotFoundError(f"validated fixed-render directory missing: {FIXED}")
rows=[]
for fp in sorted(FIXED.glob("funnybirds-mcbm-g*-s*.csv")):
    m=re.fullmatch(r"funnybirds-mcbm-g([0-9p]+)-s(\d+)\.csv",fp.name)
    if not m: continue
    d=pd.read_csv(fp); g=float(m.group(1).replace("p",".")); seed=int(m.group(2))
    d["gamma"]=g; d["seed"]=seed; rows.append(d)
if not rows: raise FileNotFoundError("no validated standard-MCBM fixed-render CSVs")
SW=pd.concat(rows,ignore_index=True)
needed={"part","direction","z_new","z_old","z_new_orig","z_old_orig","margin","sid_src","sid_donor"}
missing=needed-set(SW.columns)
if missing: raise RuntimeError(f"fixed-render CSV schema missing {sorted(missing)}")
SW["m_orig"]=SW.z_new_orig-SW.z_old_orig
SW["m_cf"]=SW.z_new-SW.z_old
SW["response_delta"]=SW.m_cf-SW.m_orig
SW["backwash"]=(SW.response_delta>0)&(SW.m_cf<0)
inv=(SW.groupby(["gamma","seed"]).agg(rows=("part","size"),parts=("part","nunique"),
       directions=("direction","nunique"),source_species=("sid_src","nunique"),
       donor_species=("sid_donor","nunique")).reset_index())
if "render_id_cf" in SW: inv=inv.merge(SW.groupby(["gamma","seed"]).render_id_cf.nunique().rename("render_ids"),on=["gamma","seed"])
display(inv)
print("fixed render root:",FIXED)
""", "Figure 1. Inventory of validated fixed-render MCBM comparisons by gamma and seed."), review(1)]

cells += [md("f2", r"""
## Figure 2 — Did gamma compress the intended internal slots without breaking prediction?

**Variables.** `target RMSE` is the root mean squared distance from each saved
internal slot `h_ij` to its `+3/-3` label target. `within-label spread` is the
median across concepts and labels of `Q95(h)-Q05(h)`. Lower means stronger
compression. Species accuracy and concept balanced accuracy are health checks.

**Prediction.** Gamma should lower the first two quantities. A grounding claim is
interpretable only if species/concept health remains usable. These panels may use
all available checkpoints; dots are independent seeds and lines connect only
gamma means.
"""), code("f2", r"""
import torch
health=[]; excluded_health=[]
for g,tag in [(0,"g0"),(.1,"g0p1"),(.3,"g0p3"),(1,"g1"),(3,"g3"),(5,"g5")]:
  base=REPO/"external/minimal_cbm/results"/f"funnybirds-mcbm-{tag}"
  for sd in sorted(base.glob("[0-9]*")) if base.exists() else []:
    pp=sd/"predictions/epoch_100.pth"
    ck=sd/"models/epoch_100.pt"
    if not (pp.exists() and ck.exists()): continue
    d=torch.load(pp,map_location="cpu",weights_only=False); h=d["z"].float().reshape(len(d["z"]),-1); c=d["c"].float().reshape(len(h),-1)
    if not (torch.isfinite(h).all() and torch.isfinite(c).all()):
      excluded_health.append(dict(gamma=g,seed=int(sd.name),status="INVALID OUTPUT",reason="non-finite saved internal slots or labels")); continue
    logits=concept_logits_from_saved_latent(h,ck,c.shape[1])
    if not torch.isfinite(logits).all():
      excluded_health.append(dict(gamma=g,seed=int(sd.name),status="INVALID OUTPUT",reason="non-finite replayed concept logits")); continue
    err=validate_saved_probabilities(logits,d["c_preds"])
    target=6*c-3; rmse=float(((h-target)**2).mean().sqrt())
    spreads=[]
    for j in range(c.shape[1]):
      for lab in [0,1]:
        q=h[c[:,j]==lab,j]
        if len(q)>5: spreads.append(float(torch.quantile(q,.95)-torch.quantile(q,.05)))
    pred=(logits>0); tpr=((pred)&(c==1)).sum(0)/(c==1).sum(0).clamp(min=1); tnr=((~pred)&(c==0)).sum(0)/(c==0).sum(0).clamp(min=1)
    yp=d["y_preds"]; ya=float((yp.argmax(-1)==d["y"]).float().mean())
    health.append(dict(gamma=g,seed=int(sd.name),target_rmse=rmse,within_label_spread=np.median(spreads),species_accuracy=ya,concept_balanced_accuracy=float(((tpr+tnr)/2).mean()),replay_error=err))
H=pd.DataFrame(health)
if H.empty: raise FileNotFoundError("no standard MCBM prediction/checkpoint pairs")
display(H.round(4))
if excluded_health:
 print("Excluded unfinished/corrupt artifacts; these are not seeds:")
 display(pd.DataFrame(excluded_health))
fig,ax=plt.subplots(1,4,figsize=(15,3.5)); metrics=[("target_rmse","target RMSE h vs ±3"),("within_label_spread","within-label h spread"),("species_accuracy","species accuracy"),("concept_balanced_accuracy","concept balanced accuracy")]
for a,(metric,title) in zip(ax,metrics):
  for _,r in H.iterrows(): a.scatter(r.gamma,r[metric],color="#D55E00",alpha=.55)
  q=H.groupby("gamma")[metric].mean(); a.plot(q.index,q.values,"o-",color="black"); a.set_xlabel("gamma"); a.set_title(title)
plt.tight_layout()
""", "Figure 2. Compression and ordinary prediction health across gamma; dots are independently trained seeds."), review(2)]

cells += [md("f3", r"""
## Figure 3 — Does MCBM gamma 0 reproduce the standard-CBM discovery?

**Question.** Before crediting minimality, compare standard CBM and MCBM
`gamma=0` on the identical fixed renders and the identical controlled-backwash
predicate. The y-axis is the fraction satisfying `response_delta>0 and m_cf<0`;
higher means more responded-but-source-wins cases. This is not expected to be
identical because the architectures differ, but the part ordering establishes
the starting point.
"""), code("f3", r"""
cbfs=sorted(FIXED.glob("funnybirds-cbm-s*.csv")); CB=pd.concat([pd.read_csv(f).assign(seed=int(re.search(r"s(\d+)",f.stem).group(1))) for f in cbfs],ignore_index=True) if cbfs else None
parts=[]
if CB is not None:
  CB["m_orig"]=CB.z_new_orig-CB.z_old_orig; CB["m_cf"]=CB.z_new-CB.z_old; CB["response_delta"]=CB.m_cf-CB.m_orig; CB["backwash"]=(CB.response_delta>0)&(CB.m_cf<0)
  parts.append(CB.groupby("part").backwash.mean().rename("standard CBM"))
parts.append(SW[SW.gamma==0].groupby("part").backwash.mean().rename("MCBM gamma=0"))
T=pd.concat(parts,axis=1).reindex(ORDER); display(T.round(3)); ax=T.plot.bar(figsize=(8,4),color=["#0072B2","#D55E00"]); ax.set_ylabel("fraction: response_delta>0 and m_cf<0"); ax.set_xlabel("part"); ax.set_title("Same fixed renders and same backwash predicate"); plt.tight_layout()
""", "Figure 3. Standard CBM and MCBM gamma zero controlled-backwash rates by part."), review(3)]

cells += [md("f4", r"""
## Figure 4 — Do the inserted pixels move the margin donorward as gamma changes?

Each cell is mean `response_delta=m_cf-m_orig` in raw-logit units. Positive means
the insertion moves donor relative to source. Because raw scale changes with
gamma, the second panel divides by the original donor deficit for rows where the
donor started below the source; 1 means the full starting deficit was closed.
Neither panel says that the donor finished above zero.
"""), code("f4", r"""
R=SW.copy(); R["gap_closed"]=np.where(R.m_orig<0,R.response_delta/(-R.m_orig),np.nan)
A=R.groupby(["gamma","part"]).response_delta.mean().unstack().reindex(columns=ORDER)
B=R.groupby(["gamma","part"]).gap_closed.median().unstack().reindex(columns=ORDER)
fig,ax=plt.subplots(1,2,figsize=(12,4)); heat(ax[0],A,"Mean donorward movement","raw logit units",cmap="coolwarm"); heat(ax[1],B,"Median original deficit closed","fraction",cmap="coolwarm"); plt.tight_layout(); display(pd.concat({"mean_response_delta":A,"median_gap_closed":B}).round(3))
""", "Figure 4. Raw and scale-normalized donorward response to the inserted part across gamma."), review(4)]

cells += [md("f5", r"""
## Figure 5 — After moving donorward, does the donor actually finish above the old source?

The left cell value is median final margin `m_cf`; positive means donor wins and
negative means old source wins. The right value is the controlled-backwash rate
`P(response_delta>0 and m_cf<0)`. A repair requires final margins to rise and
backwash rates to fall while Figure 4 retains a genuine response.
"""), code("f5", r"""
A=SW.groupby(["gamma","part"]).m_cf.median().unstack().reindex(columns=ORDER)
B=SW.groupby(["gamma","part"]).backwash.mean().unstack().reindex(columns=ORDER)
fig,ax=plt.subplots(1,2,figsize=(12,4)); lim=np.nanmax(abs(A.values)); heat(ax[0],A,"Median final donor-minus-source margin","raw logit units",-lim,lim,"coolwarm"); heat(ax[1],B,"Responded but old source still wins","fraction",0,1,"magma_r"); plt.tight_layout(); display(pd.concat({"median_m_cf":A,"backwash_rate":B}).round(3))
""", "Figure 5. Final concept margin and controlled-backwash fraction by part and gamma."), review(5)]

cells += [md("f6", r"""
## Figure 6 — Is the result present in both swap directions?

Forward and backward are reciprocal body/part constructions. Each cell is the
controlled-backwash fraction, not simple donor-win accuracy. Similar values rule
out a pooled mean hiding opposite directions; disagreement would require pair
construction inspection before interpretation.
"""), code("f6", r"""
fig,ax=plt.subplots(1,2,figsize=(12,4)); tables=[]
for a,direction in zip(ax,["fwd","bwd"]):
 T=SW[SW.direction==direction].groupby(["gamma","part"]).backwash.mean().unstack().reindex(columns=ORDER); tables.append(T); heat(a,T,f"{direction} controlled backwash","fraction",0,1,"magma_r")
plt.tight_layout(); display(pd.concat({"forward":tables[0],"backward":tables[1]}).round(3))
""", "Figure 6. Controlled-backwash rate separated into forward and backward swaps."), review(6)]

cells += [md("f7", r"""
## Figure 7 — Does visible inserted-part area explain the failures or the gamma trend?

Rows are gamma; columns are parts. Panels report controlled-backwash rates for
all swaps, for `pixel_count_cf>0`, and for `pixel_count_cf>=100`. These are nested
descriptive selections using the same thresholds at every gamma. If visibility
were the whole cause, the >=100-pixel panel should approach zero for every part.
"""), code("f7", r"""
if "pixel_count_cf" not in SW: print("INCOMPLETE: fixed CSV has no pixel_count_cf")
else:
 fig,ax=plt.subplots(1,3,figsize=(16,4)); tabs=[]
 for a,(label,d) in zip(ax,[("all rows",SW),("visible: pixels > 0",SW[SW.pixel_count_cf>0]),("large: pixels >= 100",SW[SW.pixel_count_cf>=100])]):
  T=d.groupby(["gamma","part"]).backwash.mean().unstack().reindex(columns=ORDER); tabs.append(T); heat(a,T,label,"backwash fraction",0,1,"magma_r")
 plt.tight_layout(); display(pd.concat(dict(zip(["all","visible","large"],tabs))).round(3))
""", "Figure 7. Backwash rates before and after common inserted-part visibility thresholds."), review(7)]

cells += [md("f8", r"""
## Figure 8 — Does the model name the exact inserted value, not merely beat the source value?

For each part and gamma, the value is the fraction of swaps where the inserted
donor value has the largest post-swap raw logit among all values of that part.
One means perfect exact-value attribution. This is stricter than `m_cf>0` and
detects confusion with a third value.
"""), code("f8", r"""
diag=pd.DataFrame(index=sorted(SW.gamma.unique()),columns=ORDER,dtype=float)
for g in diag.index:
 for p in ORDER:
  d=SW[(SW.gamma==g)&(SW.part==p)]; cols=sorted([c for c in SW if c.startswith(f"z_cf_{p}_")],key=lambda c:int(c.rsplit("_",1)[1]))
  if cols and "var_donor" in d:
   q=d.dropna(subset=cols); donor=q.var_donor.astype(int).values; pred=q[cols].values.argmax(1); valid=(donor>=0)&(donor<len(cols)); diag.loc[g,p]=(pred[valid]==donor[valid]).mean()
fig,ax=plt.subplots(figsize=(8,4)); heat(ax,diag,"Exact inserted-value recognition","fraction correct",0,1,"RdYlGn"); plt.tight_layout(); display(diag.round(3))
""", "Figure 8. Exact inserted-value recognition for every FunnyBird part and gamma."), review(8)]

cells += [md("f9", r"""
## Figure 9 — After exact source/donor value difficulty, does source species still organize the final margin?

For each row, subtract the mean `m_cf` for the same gamma, part, source value,
and donor value. Then average the residual by source species. The left panel is
the standard deviation of those species means in raw-logit units and is valid
within a gamma. Because gamma changes score scale, the right panel divides each
residual by the overall `m_cf` standard deviation for that gamma and part. That
standardized panel is the valid cross-gamma comparison. Zero would mean no
remaining source-species organization after exact values. This is observational
because body/species appearance was not independently manipulated.
"""), code("f9", r"""
if {"var_src","var_donor","sid_src"}.issubset(SW):
 D=SW.copy(); D["matched_mean"]=D.groupby(["gamma","part","var_src","var_donor"]).m_cf.transform("mean"); D["residual"]=D.m_cf-D.matched_mean
 D["part_scale"]=D.groupby(["gamma","part"]).m_cf.transform("std"); D["standardized_residual"]=D.residual/D.part_scale.replace(0,np.nan)
 Q=(D.groupby(["gamma","part","sid_src"]).residual.mean().groupby(["gamma","part"]).std().unstack().reindex(columns=ORDER))
 Z=(D.groupby(["gamma","part","sid_src"]).standardized_residual.mean().groupby(["gamma","part"]).std().unstack().reindex(columns=ORDER))
 fig,ax=plt.subplots(1,2,figsize=(13,4)); heat(ax[0],Q,"Raw source-species residual spread","SD in raw z units",0,None,"viridis"); heat(ax[1],Z,"Scale-standardized residual spread","SD / within-part m_cf SD",0,None,"viridis"); plt.tight_layout(); display(pd.concat({"raw":Q,"standardized":Z}).round(3))
else: print("INCOMPLETE: exact value/species columns absent")
""", "Figure 9. Residual source-species organization after matching exact source and donor values."), review(9)]

cells += [md("f10", r"""
## Figure 10 — How much species information remains in the concept representation?

Existing held-out species probes are loaded for every available gamma/seed. The
full vector and tail block are compared with blind chance `1/50`. Tail also has
a structural processed-label control near `9/50`, because its nine mutually
exclusive values divide the 50 species into nine label buckets. Accuracy above
that control is extra within-bucket species information. Leakage makes backwash
possible but does not prove that the model used it in a swap.
"""), code("f10", r"""
probe=[]
for f in sorted((CURATED/"species_probe").glob("funnybirds-mcbm-g*-s*.json")):
 m=re.search(r"-g([0-9p]+)-s(\d+)\.json$",f.name)
 if not m: continue
 q=json.loads(f.read_text()); probe.append(dict(gamma=float(m.group(1).replace("p",".")),seed=int(m.group(2)),full=q["species_from_cpreds"]["acc"],tail=q["species_from_part_cpreds"].get("tail",{}).get("acc",np.nan)))
P=pd.DataFrame(probe)
if P.empty: print("INCOMPLETE: species probes unavailable")
else:
 display(P.round(3)); M=P.groupby("gamma")[["full","tail"]].mean(); ax=M.plot(marker="o",figsize=(7,4)); ax.axhline(1/50,color="black",ls="--",label="blind 1/50"); ax.axhline(9/50,color=COLORS["tail"],ls=":",label="tail labels ~9/50"); ax.set_ylabel("held-out species accuracy"); ax.set_xlabel("gamma"); ax.set_ylim(0,1.02); ax.set_title("Species recoverable from learned concept outputs"); ax.legend(); plt.tight_layout()
""", "Figure 10. Held-out species decoding from all concept outputs and the tail block across gamma."), review(10)]

cells += [md("f11", r"""
## Figure 11 — Is concept recall species-dependent after valid matching?

Recall is a model-health/species-dependence diagnostic, not a substitute for the
swap. This section accepts only precomputed tables produced with the developed
`fb_recallv2` all-positive-species pairing and recall-v4 vectorized bootstrap.
Each value is the absolute recall difference between two matched species for the
same positive concept. Larger means the concept is recognized less uniformly
across species. Bootstrap rows are not independent model seeds.
"""), code("f11", r"""
files=sorted((CURATED/"recall").glob("funnybirds-mcbm-g*-*.csv")) if (CURATED/"recall").exists() else []
if not files: print("INCOMPLETE: no authoritative precomputed FunnyBird MCBM recall-v4 tables; swap evidence remains usable")
else:
 rr=[]
 for f in files:
  m=re.search(r"-g([0-9p]+)",f.name); d=pd.read_csv(f); d["gamma"]=float(m.group(1).replace("p",".")); rr.append(d)
 R=pd.concat(rr); col="gap_mean" if "gap_mean" in R else "recall_gap"; T=R.groupby("gamma")[col].agg(["median","mean","count"]); display(T.round(3)); ax=T[["median","mean"]].plot(marker="o",figsize=(7,4)); ax.set_ylabel("absolute matched-species recall gap"); ax.set_title("Species dependence of concept recall"); plt.tight_layout()
""", "Figure 11. Matched-species concept recall gaps from authoritative FunnyBird recall pairing, when available."), review(11)]

cells += [md("f12", r"""
## Figure 12 — Does the concept margin have a downstream species-class effect?

Within each part and gamma, swaps are divided into independent, approximately
equal-count bins by final margin `m_cf` on the x-axis. The y-axis is mean
species-head probability for the donor species. Separate part panels prevent the
strong wing/foot results from hiding tail. This is intentionally a probability
because the question is downstream classification. A low value means the concept
failure has limited class cost; it does not make the grounding failure unreal.
"""), code("f12", r"""
if "p_cf_donor" not in SW: print("INCOMPLETE: p_cf_donor absent")
else:
 fig,axes=plt.subplots(1,5,figsize=(19,3.8),sharey=True)
 for ax,part in zip(axes,ORDER):
  for g in sorted(SW.gamma.unique()):
   d=SW[(SW.gamma==g)&(SW.part==part)].copy(); d["bin"]=pd.qcut(d.m_cf,8,duplicates="drop"); q=d.groupby("bin",observed=True).agg(m_cf=("m_cf","mean"),p=("p_cf_donor","mean"),n=("p_cf_donor","size")); ax.plot(q.m_cf,q.p,"o-",ms=3,label=f"gamma={g:g}")
  ax.axvline(0,color="black",lw=1); ax.set_xlabel("final margin m_cf"); ax.set_title(part)
 axes[0].set_ylabel("mean P(donor species)"); axes[-1].legend(fontsize=7,ncol=2); fig.suptitle("Downstream class response, separated by replaced part"); plt.tight_layout()
""", "Figure 12. Donor-species probability as a function of final donor-minus-source concept margin."), review(12)]

cells += [md("f13", r"""
## Figure 13 — Which gamma claims have independent-seed support?

The left count is prediction/checkpoint health seeds from Figure 2. The right
count is validated fixed-render causal seeds from Figure 1. Only the latter can
support uncertainty for the gamma grounding curve. Repeated swaps within one
seed are not independent model replications.
"""), code("f13", r"""
A=H.groupby("gamma").seed.nunique().rename("health seeds")
B=SW.groupby("gamma").seed.nunique().rename("fixed-render seeds")
T=pd.concat([A,B],axis=1).reindex(GAMMAS).fillna(0).astype(int); display(T); ax=T.plot.bar(figsize=(8,4),color=["#999999","#D55E00"]); ax.set_ylabel("independently trained seeds"); ax.set_xlabel("gamma"); ax.set_title("Replication actually available for each claim"); plt.tight_layout()
""", "Figure 13. Independent checkpoint and validated fixed-render seed counts by gamma."), review(13)]

cells += [md("f14", r"""
## Figure 14 — Aligned decision table: did compression repair grounding?

All panels share gamma rows and part columns. `compression` is shown once per
gamma; the other panels show donorward response, final backwash, exact-value
error, and ordinary concept health. Larger is not uniformly better: the titles
state the desired direction. The quantities are not added because they have
different units and denominators. Repair requires compression **and** improved
causal grounding without broken health.
"""), code("f14", r"""
compression=H.groupby("gamma").target_rmse.mean().reindex(GAMMAS)
resp=SW.groupby(["gamma","part"]).response_delta.mean().unstack().reindex(index=GAMMAS,columns=ORDER)
bw=SW.groupby(["gamma","part"]).backwash.mean().unstack().reindex(index=GAMMAS,columns=ORDER)
err=1-diag.reindex(index=GAMMAS,columns=ORDER)
health=H.groupby("gamma").concept_balanced_accuracy.mean().reindex(GAMMAS)
fig,ax=plt.subplots(1,5,figsize=(19,4)); C=pd.DataFrame({"target RMSE":compression,"concept BA":health})
heat(ax[0],compression.to_frame("all slots"),"Compression target RMSE (lower)","h units",0,None,"viridis"); heat(ax[1],resp,"Donorward response (higher)","z units",None,None,"coolwarm"); heat(ax[2],bw,"Controlled backwash (lower)","fraction",0,1,"magma_r"); heat(ax[3],err,"Exact-value error (lower)","fraction",0,1,"magma_r"); heat(ax[4],health.to_frame("all concepts"),"Concept health BA (higher)","balanced accuracy",0,1,"RdYlGn"); plt.tight_layout(); display(C.round(3))
""", "Figure 14. Aligned compression, response, backwash, exact-value error, and health summary across gamma."), review(14),
md("end", r"""
## Claim boundary

After Figures 1--14 are visually reviewed, the notebook may conclude whether
minimality compressed the representation and whether the accepted seed-1 gamma
curve supports or fails to support grounding repair. It may not call single-seed
gamma differences stable. It may not claim that visibility, value difficulty,
and source species explain every residual unless the measured residual is near
zero. The next stage is notebook 06, which freezes these questions and states
where CUB offers the same operation, a weaker observational approximation, or no
valid test.
""")]

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.10"}},"nbformat":4,"nbformat_minor":5}
if __name__ == "__main__":
    OUT.write_text(json.dumps(nb,indent=1,ensure_ascii=False),encoding="utf-8")
    print(f"wrote {OUT} with {len(cells)} cells")
