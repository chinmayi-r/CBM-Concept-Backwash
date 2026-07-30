#!/usr/bin/env python3
"""Restore important standard-MCBM analyses lost during the notebook rewrite.

This is deliberately limited to the pre-RL story. It makes notebook 03 prefer
the semantically validated fixed-render output, then restores the useful old
diagnostics that are not already present in the shorter evidence-led rewrite.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "03_funnybirds_mcbm.ipynb"
TAG = "standard-mcbm-catchup"


def cell(kind: str, source: str) -> dict:
    out = {
        "cell_type": kind,
        "id": uuid.uuid4().hex[:8],
        "metadata": {"tags": [TAG]},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        out.update(execution_count=None, outputs=[])
    return out


LOADER = r'''ORDER=["tail","wing","beak","foot","eye"]
def load_swaps():
    # Prefer the semantically validated, byte-matched fixed-render directory.
    # Never mix rows from fixed and legacy renderer runs.
    candidates=[
        CURATED/"swap_fixed_v2_attempt2",
        CURATED/"swap_fixed_v2",
        CURATED/"swap",
    ]
    swap_root=None
    for root in candidates:
        if list(root.glob("funnybirds-mcbm-g*-s*.csv")):
            swap_root=root
            break
    if swap_root is None:
        print("[pending] no standard MCBM combined swap CSVs")
        return None, None

    rows=[]
    for fp in sorted(swap_root.glob("funnybirds-mcbm-g*-s*.csv")):
        m=re.match(r"funnybirds-mcbm-g([0-9p]+)-s(\d+)\.csv$", fp.name)
        if not m:
            continue
        d=pd.read_csv(fp)
        d["gamma"]=float(m.group(1).replace("p","."))
        d["seed"]=int(m.group(2))
        rows.append(d)
    SW=pd.concat(rows,ignore_index=True) if rows else None

    cbs=[f for f in sorted(swap_root.glob("funnybirds-cbm-s*.csv"))
         if re.match(r"funnybirds-cbm-s(\d+)\.csv$", f.name)]
    CB=pd.concat([pd.read_csv(f) for f in cbs],ignore_index=True) if cbs else None
    print("swap source:", swap_root)
    if "fixed_v2" not in swap_root.name:
        print("[PROVISIONAL] legacy independently rendered swaps; do not make exact cross-model causal claims")
    if SW is not None:
        print("loaded swaps:", {
            g:sorted(SW[SW.gamma==g].seed.unique().tolist())
            for g in sorted(SW.gamma.unique())
        })
    return SW, CB

SW,CB=load_swaps()
if SW is not None:
    H=SW.groupby(["gamma","part"]).ordering_correct.mean().unstack().reindex(columns=ORDER)
    display(H.round(3))
    fig,ax=plt.subplots(figsize=(6.2,3.8))
    im=ax.imshow(H.values,cmap="RdYlGn",vmin=0,vmax=1,aspect="auto")
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER)
    ax.set_yticks(range(len(H.index))); ax.set_yticklabels([f"γ={g:g}" for g in H.index])
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            v=H.values[i,j]
            if np.isfinite(v):
                ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=8)
    ax.set_title("Post-swap ordering: part × γ")
    fig.colorbar(im,fraction=0.046,label="P(donor score > source score)")
'''


CELLS = [
    cell(
        "markdown",
        """## 6a · Catch-up: standard MCBM analyses from the original notebook

These panels restore the important pre-RL work that disappeared when notebook 03
was shortened. They do **not** use the relabeled models.

For every panel the order is: question, prediction, literal output, alternative
explanation, discriminating check, limited conclusion, next question. Literal
observations must be written only after the executed figure is inspected.
""",
    ),
    cell(
        "markdown",
        """### 6a.1 · Does the result agree in both swap directions?

**Question.** Does the replacement win for `body_A + part_B` and for
`body_B + part_A`, or is a bookkeeping direction creating the result?

**Prediction.** A real part-grounding pattern should have similar forward and
backward values. Strong disagreement would make the combined average unsafe.

The figure reports `P(margin>0)` separately for both directions. Green means the
replacement concept wins; red means the old source concept wins.
""",
    ),
    cell(
        "code",
        r'''if SW is not None and "direction" in SW:
    frames=[]
    if CB is not None:
        frames.append(("CBM",CB))
    frames += [(f"MCBM γ={g:g}",d) for g,d in SW.groupby("gamma")]
    labels=[x[0] for x in frames]
    fig,axes=plt.subplots(1,2,figsize=(12,0.48*len(labels)+2.2),sharey=True)
    for ax,direction in zip(axes,["fwd","bwd"]):
        H=pd.DataFrame(
            [d[d.direction==direction].groupby("part").ordering_correct.mean()
             for _,d in frames],
            index=labels,
        ).reindex(columns=ORDER)
        im=ax.imshow(H.values,cmap="RdYlGn",vmin=0,vmax=1,aspect="auto")
        ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        ax.set_title(f"{direction}: P(donor score > source score)")
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                if np.isfinite(H.iloc[i,j]):
                    ax.text(j,i,f"{H.iloc[i,j]:.2f}",ha="center",va="center",fontsize=7)
    fig.colorbar(im,ax=axes,fraction=0.025)
else:
    print("[pending] direction-labelled standard swap rows")
''',
    ),
    cell(
        "markdown",
        """**Discriminating check.** Compare each forward cell directly with its
backward partner. If they disagree, inspect pair construction before interpreting
the average.

**Limited conclusion rule.** Agreement can rule out a simple reversed-direction
bug. It cannot by itself prove that the unchanged body caused the failure.

**Next question.** Even when the donor finishes below the source, did the inserted
part move the model in the correct direction?
""",
    ),
    cell(
        "markdown",
        """### 6a.2 · Did the swap move the scores toward the donor?

Define

`response_delta = (z_donor − z_source)_swap − (z_donor − z_source)_original`.

**Question.** Does adding the donor part move the donor-versus-source margin
upward? `response_delta>0` means it did, even if the donor did not finish on top.

**Prediction.** Grounded concepts should usually have positive response. A
positive but small tail response means partial visual sensitivity plus a strong
remaining source/body preference.
""",
    ),
    cell(
        "code",
        r'''if SW is not None:
    D=SW.copy()
    if "response_delta" not in D:
        needed={"margin","z_new_orig","z_old_orig"}
        if needed.issubset(D.columns):
            D["response_delta"]=D["margin"]-(D["z_new_orig"]-D["z_old_orig"])
    if "response_delta" in D:
        R=(D.groupby(["gamma","seed","part"]).response_delta.mean()
             .reset_index())
        fig,ax=plt.subplots(figsize=(7.2,4.2))
        for part in ORDER:
            p=R[R.part==part]
            for seed,s in p.groupby("seed"):
                ax.plot(s.gamma,s.response_delta,marker="o",alpha=.35,linewidth=1)
            m=p.groupby("gamma").response_delta.mean()
            ax.plot(m.index,m.values,marker="o",linewidth=2.3,label=part)
        ax.axhline(0,color="black",linewidth=1)
        ax.set_xscale("symlog",linthresh=.05)
        ax.set_xlabel("γ"); ax.set_ylabel("mean response_delta")
        ax.set_title("Within-image visual response to the inserted part")
        ax.legend(ncol=3,fontsize=8)
        display(R.pivot_table(index=["gamma","seed"],columns="part",
                              values="response_delta").round(3))
    else:
        print("[pending] swap rows with original and counterfactual margins")
''',
    ),
    cell(
        "markdown",
        """**Alternative explanation.** A positive mean could be driven by a few
large rows or one seed. Inspect every seed and the distribution before calling it
stable.

**Limited conclusion rule.** Positive `response_delta` supports visual
sensitivity. It does not imply correct final attribution; that still requires the
post-swap margin to cross zero.

**Next question.** Are failures spread across variants, or caused by one bad
concept slot?
""",
    ),
    cell(
        "markdown",
        """### 6a.3 · Which visible variant does each part report?

**Question.** After inserting donor variant `v`, which concept slot has the
largest score within that part group?

Rows are donor variants and columns are the reported argmax variants. A bright
diagonal means correct attribution. A bright column means many inputs collapse
onto one default answer.
""",
    ),
    cell(
        "code",
        r'''if SW is not None:
    g0=sorted(SW.gamma.unique())[0]
    S0=SW[SW.gamma==g0]
    have=[p for p in ORDER if any(c.startswith(f"z_cf_{p}_") for c in S0.columns)]
    if have:
        fig,axes=plt.subplots(1,len(have),figsize=(3.05*len(have),3.2))
        if len(have)==1: axes=[axes]
        last=None; diags={}
        for ax,p in zip(axes,have):
            cols=sorted([c for c in S0.columns if c.startswith(f"z_cf_{p}_")],
                        key=lambda c:int(c.split("_")[-1]))
            dd=S0[S0.part==p].dropna(subset=cols)
            n=len(cols); pred=dd[cols].values.argmax(1); donor=dd.var_donor.astype(int).values
            M=np.zeros((n,n))
            for a,b in zip(donor,pred):
                if 0<=a<n: M[a,b]+=1
            M=M/M.sum(1,keepdims=True).clip(min=1)
            diags[p]=float((pred==donor).mean())
            last=ax.imshow(M,cmap="magma",vmin=0,vmax=1)
            ax.set_title(f"{p}: diagonal={diags[p]:.2f}",fontsize=9)
            ax.set_xlabel("reported variant")
        axes[0].set_ylabel("inserted donor variant")
        fig.suptitle(f"All-part concept confusion at γ={g0:g}")
        fig.colorbar(last,ax=axes,fraction=.02)
        print("diagonal fractions:",{k:round(v,3) for k,v in diags.items()})
    else:
        print("[pending] z_cf_<part>_<variant> columns")
''',
    ),
    cell(
        "markdown",
        """**Alternative explanation.** Argmax attribution is stricter and
different from the two-slot ordering metric. Read it as a description of
within-part confusion, not a replacement for `margin`.

**Limited conclusion rule.** A repeated off-diagonal column supports
default-concept collapse only when grounded parts processed by the same code show
cleaner diagonals.

**Next question.** Is the remaining tail variation explained by tail variants,
or does the unchanged source body still matter?
""",
    ),
    cell(
        "markdown",
        """### 6a.4 · Variant first, then source body/species

**Question.** Which source–donor tail combinations fail, and is there additional
variation across source species after acknowledging that each species has a fixed
tail variant?

The first panel shows variant-pair follow rates. The second shows raw source
species violation rates, coloured by the source tail variant. The third subtracts
the mean failure rate of each source variant so species sharing a variant can be
compared.
""",
    ),
    cell(
        "code",
        r'''if SW is not None and {"var_src","var_donor","sid_src"}.issubset(SW.columns):
    g0=sorted(SW.gamma.unique())[0]
    t=SW[(SW.gamma==g0)&(SW.part=="tail")].copy()
    P=t.groupby(["var_src","var_donor"]).ordering_correct.mean().unstack()
    raw=(t.assign(violation=~t.ordering_correct.astype(bool))
          .groupby(["sid_src","var_src"]).violation.mean().reset_index()
          .sort_values("violation",ascending=False))
    raw["variant_mean"]=raw.groupby("var_src").violation.transform("mean")
    raw["within_variant_residual"]=raw.violation-raw.variant_mean
    residual=raw.sort_values("within_variant_residual",ascending=False)
    fig,axes=plt.subplots(1,3,figsize=(18,4))
    im=axes[0].imshow(P.values,cmap="RdYlGn",vmin=0,vmax=1,aspect="auto")
    axes[0].set_xticks(range(len(P.columns))); axes[0].set_xticklabels(P.columns)
    axes[0].set_yticks(range(len(P.index))); axes[0].set_yticklabels(P.index)
    axes[0].set_xlabel("donor tail variant"); axes[0].set_ylabel("source tail variant")
    axes[0].set_title("Tail replacement-follow rate by variant pair")
    fig.colorbar(im,ax=axes[0],fraction=.046)
    sc=axes[1].scatter(range(len(raw)),raw.violation,c=raw.var_src,cmap="tab10",s=24)
    axes[1].axhline(t.assign(v=~t.ordering_correct.astype(bool)).v.mean(),
                    color="black",linestyle="--",linewidth=1)
    axes[1].set_xlabel("source species, sorted"); axes[1].set_ylabel("violation rate")
    axes[1].set_title("Raw source-species differences (colour=source variant)")
    fig.colorbar(sc,ax=axes[1],label="source tail variant")
    sc2=axes[2].scatter(range(len(residual)),residual.within_variant_residual,
                        c=residual.var_src,cmap="tab10",s=24)
    axes[2].axhline(0,color="black",linestyle="--",linewidth=1)
    axes[2].set_xlabel("source species, sorted")
    axes[2].set_ylabel("violation rate - source-variant mean")
    axes[2].set_title("Body/species signal remaining after source variant")
    fig.colorbar(sc2,ax=axes[2],label="source tail variant")
    display(residual[["sid_src","var_src","violation","variant_mean",
                      "within_variant_residual"]].head(15).round(3))
''',
    ),
    cell(
        "markdown",
        """**Alternative explanation.** Species differences can merely restate
variant differences because every species has one canonical tail. The third panel
removes each source variant's mean violation rate. Remaining residual spread
compares species that share the same source-tail variant.

**Limited conclusion rule.** Variant-pair differences are established if cells
differ with adequate counts. Consistent within-variant residual spread supports
an additional unchanged-body/species effect, although it remains observational
rather than a controlled body swap.

**Next question.** Does minimality change the full before-to-after distribution,
or only its average?
""",
    ),
    cell(
        "markdown",
        """### 6a.5 · Before and after the swap across γ

**Question.** How does the donor-versus-source margin distribution move from the
original image to the counterfactual image at each `γ`?

The dashed distribution is
`margin_original=z_donor,original−z_source,original`; the solid distribution is
`margin_swap`. Movement to the right is visual response. Crossing zero is correct
final ordering.
""",
    ),
    cell(
        "code",
        r'''if SW is not None and {"margin","z_new_orig","z_old_orig"}.issubset(SW.columns):
    gammas=sorted(SW.gamma.unique())
    fig,axes=plt.subplots(len(gammas),len(ORDER),
                          figsize=(3.0*len(ORDER),2.0*len(gammas)),
                          sharex=False,sharey=False,squeeze=False)
    for i,g in enumerate(gammas):
        for j,p in enumerate(ORDER):
            d=SW[(SW.gamma==g)&(SW.part==p)]
            before=(d.z_new_orig-d.z_old_orig).dropna()
            after=d.margin.dropna()
            vals=np.r_[before.values,after.values]
            if len(vals):
                lo,hi=np.quantile(vals,[.01,.99]); bins=np.linspace(lo,hi,28)
                axes[i,j].hist(before,bins=bins,density=True,histtype="step",
                               linestyle="--",color="gray",label="original")
                axes[i,j].hist(after,bins=bins,density=True,histtype="step",
                               linewidth=1.6,color="#D55E00",label="swap")
            axes[i,j].axvline(0,color="black",linewidth=.7)
            if i==0: axes[i,j].set_title(p)
            if j==0: axes[i,j].set_ylabel(f"γ={g:g}")
    axes[0,0].legend(fontsize=7)
    fig.suptitle("Donor-minus-source margin before and after the part swap",y=1.01)
    fig.tight_layout()
''',
    ),
    cell(
        "markdown",
        """**Alternative explanation.** Reused original images make rows
non-independent; read distribution positions, not apparent sample density.

**Limited conclusion rule.** A right shift shows response to the inserted pixels.
Mass remaining below zero shows that response was insufficient to overcome the
old source/body preference.

**Next question.** With the old standard work restored, notebook 03 can now give
its limited MCBM conclusion before notebook 03rl tests the visibility-label cause.
""",
    ),
    cell(
        "markdown",
        """### 6a.6 · Ordinary-image source and donor scores

**Question.** Before any intervention, is the present source concept high and
the absent donor alternative low for every part?

**Prediction.** Strong separation means the model looks grounded on ordinary
images. A later swap failure would then be intervention-specific rather than a
general inability to identify concepts.
""",
    ),
    cell(
        "code",
        r'''if SW is not None and {"z_old_orig","z_new_orig"}.issubset(SW.columns):
    gd=0.1 if 0.1 in SW.gamma.unique() else sorted(SW.gamma.unique())[0]
    d=SW[SW.gamma==gd]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,col,title in [
        (axes[0],"z_old_orig","present source concept on original image"),
        (axes[1],"z_new_orig","absent donor concept on original image"),
    ]:
        vals=[d[d.part==p][col].dropna().values for p in ORDER]
        ax.boxplot(vals,labels=ORDER,showfliers=False)
        ax.axhline(0,color="black",linewidth=.8)
        ax.set_title(title); ax.set_ylabel("raw z")
    fig.suptitle(f"Ordinary-image grounding control at γ={gd:g}")
''',
    ),
    cell(
        "markdown",
        """**Limited conclusion rule.** Separation supports ordinary-image
grounding only. It does not settle whether swapped pixels or the unchanged body
control the answer.

**Next question.** Are tail failures broad, or concentrated in particular donor
variants?
""",
    ),
    cell(
        "markdown",
        """### 6a.7 · Tail margin distributions by donor variant

**Question.** Does every inserted tail variant have a similar post-swap margin,
or do particular variants fail systematically?

Each panel shows `margin=z_donor,swap−z_source,swap`. Values right of zero mean
the inserted donor tail wins.
""",
    ),
    cell(
        "code",
        r'''if SW is not None and "var_donor" in SW:
    gd=0.1 if 0.1 in SW.gamma.unique() else sorted(SW.gamma.unique())[0]
    t=SW[(SW.gamma==gd)&(SW.part=="tail")]
    variants=sorted(t.var_donor.dropna().astype(int).unique())
    ncol=3; nrow=int(np.ceil(len(variants)/ncol))
    fig,axes=plt.subplots(nrow,ncol,figsize=(11,2.7*nrow),squeeze=False)
    for ax,v in zip(axes.flat,variants):
        x=t[t.var_donor.astype(int)==v].margin.dropna()
        ax.hist(x,bins=24,color="#6a0dad",alpha=.75)
        ax.axvline(0,color="black",linestyle="--",linewidth=1)
        ax.set_title(f"donor tail {v}: win={(x>0).mean():.2f}, n={len(x)}")
        ax.set_xlabel("post-swap margin")
    for ax in axes.flat[len(variants):]: ax.axis("off")
    fig.suptitle(f"Tail margin by inserted donor variant, γ={gd:g}",y=1.01)
    fig.tight_layout()
''',
    ),
    cell(
        "markdown",
        """**Alternative explanation.** Rows reuse source images and are not
independent. Compare positions and variant consistency, not histogram density.

**Limited conclusion rule.** Unequal panels establish variant-specific
difficulty. They do not alone explain why a variant fails.

**Next question.** Do forward and backward margins move together or oppose one
another?
""",
    ),
    cell(
        "markdown",
        """### 6a.8 · Forward versus backward mean margin

**Question.** For each part and `γ`, is mean forward margin similar to mean
backward margin?

Points near the positive diagonal indicate direction agreement. Points near the
negative diagonal indicate direction cancellation, which produced the misleading
0.50 averages in the original broken run.
""",
    ),
    cell(
        "code",
        r'''if SW is not None and "direction" in SW:
    M=(SW.groupby(["gamma","part","direction"]).margin.mean()
         .unstack("direction").dropna())
    fig,ax=plt.subplots(figsize=(6,5.5))
    colors={p:c for p,c in zip(ORDER,plt.cm.tab10.colors)}
    for (g,p),r in M.iterrows():
        ax.scatter(r["fwd"],r["bwd"],s=45,color=colors[p])
        ax.annotate(f"{p}, γ={g:g}",(r["fwd"],r["bwd"]),fontsize=7,
                    xytext=(3,3),textcoords="offset points")
    lim=max(abs(M[["fwd","bwd"]].values).max(),1)
    ax.plot([-lim,lim],[-lim,lim],"--",color="green",label="same direction")
    ax.plot([-lim,lim],[lim,-lim],":",color="red",label="cancellation")
    ax.axhline(0,color="gray",linewidth=.7); ax.axvline(0,color="gray",linewidth=.7)
    ax.set_xlabel("mean forward margin"); ax.set_ylabel("mean backward margin")
    ax.set_title("Direction agreement after validated swaps"); ax.legend(fontsize=8)
''',
    ),
    cell(
        "markdown",
        """**Limited conclusion rule.** Positive-diagonal agreement rules out
the original cancellation artifact. It still does not identify the causal source
of a low margin.

**Next question.** Which individual concept slots remain worst as `γ` changes?
""",
    ),
    cell(
        "markdown",
        """### 6a.9 · Worst concept slots across γ

**Question.** Is poor grounding confined to one corrupted slot, or repeated
across several part variants and minimality settings?

For each `γ`, rank `(part, donor variant)` by violation rate. Bars are
descriptive; their row counts must be checked before comparing close values.
""",
    ),
    cell(
        "code",
        r'''if SW is not None and "var_donor" in SW:
    gammas=sorted(SW.gamma.unique())
    ncol=3; nrow=int(np.ceil(len(gammas)/ncol))
    fig,axes=plt.subplots(nrow,ncol,figsize=(14,3.6*nrow),squeeze=False)
    colors={p:c for p,c in zip(ORDER,plt.cm.tab10.colors)}
    for ax,g in zip(axes.flat,gammas):
        q=(SW[SW.gamma==g].assign(violation=lambda x:~x.ordering_correct.astype(bool))
             .groupby(["part","var_donor"]).agg(rate=("violation","mean"),
                                                 n=("violation","size"))
             .reset_index().sort_values("rate",ascending=False).head(15))
        labels=[f"{p}_{int(v)} (n={n})" for p,v,n in
                zip(q.part,q.var_donor,q.n)]
        ax.barh(range(len(q)),q.rate,color=[colors[p] for p in q.part])
        ax.set_yticks(range(len(q))); ax.set_yticklabels(labels,fontsize=7)
        ax.invert_yaxis(); ax.set_xlim(0,1); ax.set_title(f"γ={g:g}")
        ax.set_xlabel("violation rate")
    for ax in axes.flat[len(gammas):]: ax.axis("off")
    fig.suptitle("Worst-grounded concept slots at each γ",y=1.01)
    fig.tight_layout()
''',
    ),
    cell(
        "markdown",
        """**Limited conclusion rule.** Repeated tail, beak, or eye slots support
a graded multi-part problem. A single high bar with small `n` is not enough.

**Next question.** Does visible pixel area explain the remaining tail failures?
""",
    ),
    cell(
        "markdown",
        """### 6a.10 · Tail visibility versus margin

**Question.** Among validated swaps, does a larger visible inserted tail produce
a larger margin, and do failures disappear above a reasonable visibility level?

The left panel plots pixels against margin. The right compares visibility
distributions for donor wins and violations.
""",
    ),
    cell(
        "code",
        r'''if SW is not None and "pixel_count_cf" in SW:
    gd=0.1 if 0.1 in SW.gamma.unique() else sorted(SW.gamma.unique())[0]
    t=SW[(SW.gamma==gd)&(SW.part=="tail")].copy()
    t["outcome"]=np.where(t.ordering_correct.astype(bool),"donor wins","violation")
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].scatter(t.pixel_count_cf,t.margin,s=12,alpha=.35,
                    c=np.where(t.ordering_correct.astype(bool),"#2ca02c","#d62728"))
    axes[0].axhline(0,color="black",linewidth=1)
    axes[0].set_xlabel("visible inserted-tail pixels"); axes[0].set_ylabel("margin")
    axes[0].set_title(f"Tail visibility versus margin, γ={gd:g}")
    vals=[t[t.outcome==o].pixel_count_cf.values for o in ["donor wins","violation"]]
    axes[1].boxplot(vals,labels=["donor wins","violation"],showfliers=False)
    axes[1].set_ylabel("visible inserted-tail pixels")
    axes[1].set_title("Visibility by outcome")
''',
    ),
    cell(
        "markdown",
        """**Limited conclusion rule.** Overlap at high visibility rejects
occlusion as the only explanation. A shift between groups still supports
visibility as a partial explanation.

**Next question.** Why are CBM and MCBM raw `z` values numerically different even
when `γ=0`?
""",
    ),
    cell(
        "markdown",
        """### 6a.11 · CBM versus MCBM raw-score scale

**Question.** Can raw CBM and MCBM margin magnitudes be compared directly?

No: their bottleneck parameterizations differ. This control shows their
ordinary-image `z_source` distributions. Ordering signs and within-model changes
remain comparable; raw magnitudes do not.
""",
    ),
    cell(
        "code",
        r'''if SW is not None and CB is not None and "z_old_orig" in CB:
    g0=sorted(SW.gamma.unique())[0]
    a=CB.z_old_orig.dropna().values
    b=SW[SW.gamma==g0].z_old_orig.dropna().values
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].hist(a,bins=45,density=True,alpha=.6,label="CBM")
    axes[0].hist(b,bins=45,density=True,alpha=.5,label=f"MCBM γ={g0:g}")
    axes[0].set_xlabel("z_source on original image"); axes[0].set_ylabel("density")
    axes[0].set_title("Raw score distributions"); axes[0].legend()
    axes[1].bar(["CBM",f"MCBM γ={g0:g}"],[np.std(a),np.std(b)],
                color=[CBM_C,MCBM_C])
    axes[1].set_ylabel("std(z_source,original)")
    axes[1].set_title("Different scales: compare signs/changes, not magnitude")
''',
    ),
    cell(
        "markdown",
        """**Limited conclusion.** This scale difference explains why raw margin
magnitudes cannot be compared directly across CBM and MCBM. It does not explain
within-model part differences or the sign of the swap response.

Notebook 03 now retains the useful questions from the original 20 figures while
recomputing them on the validated renders and excluding plots whose only content
was the known broken 0.50/zero-visibility artifact.
""",
    ),
    cell(
        "markdown",
        """### 6a.12 · Visibility across every part

**Question.** Is visibility related only to tail failure, or do beak, eye, wing,
and foot show their own visibility-response patterns as `γ` changes?

For each `γ`, each line is one part. The x-axis uses fixed visible-pixel bins and
the y-axis is `P(margin>0)`. The printed table includes `n`, because bins with few
swaps should not drive a conclusion.
""",
    ),
    cell(
        "code",
        r'''if SW is not None and "pixel_count_cf" in SW:
    bins=[-1,20,50,100,200,500,np.inf]
    labels=["0-20","21-50","51-100","101-200","201-500",">500"]
    V=SW.copy()
    V["visibility_bin"]=pd.cut(V.pixel_count_cf,bins=bins,labels=labels)
    Q=(V.groupby(["gamma","part","visibility_bin"],observed=True)
         .agg(follow=("ordering_correct","mean"),n=("ordering_correct","size"))
         .reset_index())
    gammas=sorted(V.gamma.unique()); ncol=3
    nrow=int(np.ceil(len(gammas)/ncol))
    fig,axes=plt.subplots(nrow,ncol,figsize=(14,3.7*nrow),
                          sharex=True,sharey=True,squeeze=False)
    colors={p:c for p,c in zip(ORDER,plt.cm.tab10.colors)}
    for ax,g in zip(axes.flat,gammas):
        for p in ORDER:
            d=Q[(Q.gamma==g)&(Q.part==p)]
            x=[labels.index(str(v)) for v in d.visibility_bin]
            ax.plot(x,d.follow,marker="o",label=p,color=colors[p])
        ax.axhline(.5,color="gray",linestyle=":",linewidth=1)
        ax.set_title(f"γ={g:g}"); ax.set_ylim(0,1)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels,rotation=35)
    for ax in axes.flat[len(gammas):]: ax.axis("off")
    axes.flat[0].legend(ncol=3,fontsize=7)
    fig.supxlabel("visible inserted-part pixels")
    fig.supylabel("P(donor score > source score)")
    fig.suptitle("All-part visibility response across minimality settings",y=1.01)
    fig.tight_layout()
    display(Q.pivot_table(index=["gamma","part","visibility_bin"],
                          values=["follow","n"]).round(3))
''',
    ),
    cell(
        "markdown",
        """**Interpretation rule.** A rising curve supports visibility as one
cause for that part. A curve that remains low in well-populated high-visibility
bins rejects visibility as the only cause. Differences between parts must not be
read from empty or tiny bins.

**Next question.** After mapping these standard-model explanations, notebook
03rl tests whether visibility-aware training labels change the same responses.
""",
    ),
    cell(
        "markdown",
        """## Evidence chain: observation, explanation, test

1. **Did minimality actually change the model?** Yes: `γ` sharply compresses
   `z`, while ordinary concept and species accuracy stay high.
2. **Did that improve visual grounding?** No: tail replacement wins become less
   frequent when minimality turns on, while wing and foot remain strong.
3. **That is the odd result.** `response_delta` remains positive. Inserting a
   tail moves the donor-versus-source scores in the correct direction, so the
   model is not simply blind to the tail. The movement is weaker than for
   grounded parts and often does not cross zero.
4. **Could visibility explain it?** Visible-only filtering helps, but tail stays
   far below foot. Visibility is partial, not sufficient.
5. **Could one bad tail slot explain it?** The all-part confusion matrix,
   donor-variant margins, and worst-slot plots test this. Variants are unequal,
   but the problem is not confined to one slot.
6. **Could variant composition masquerade as species?** The variant-pair panel
   comes first. The within-variant residual panel then asks whether source-body
   or species differences remain after subtracting each source variant's mean.
7. **Could raw-score scale explain it?** CBM and MCBM use different `z` scales,
   so magnitudes cannot be compared across model families. The within-model
   ordering and response conclusions survive that control.
8. **Limited conclusion.** Minimality changes and compresses the bottleneck but
   does not guarantee that named slots follow their named visible parts. The
   strongest demonstrated failure is explanatory grounding; downstream
   species-probability changes remain small.
9. **Next causal question.** Training labels remain positive when a
   non-placeholder part is effectively invisible, especially for tails.
   Notebook 03rl changes those labels while holding images and training
   membership fixed. It is the next test, not a replacement for this evidence.
""",
    ),
]


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    nb["cells"] = [
        c for c in nb["cells"] if TAG not in c.get("metadata", {}).get("tags", [])
    ]

    loader_index = next(
        i
        for i, c in enumerate(nb["cells"])
        if c["cell_type"] == "code" and "def load_swaps()" in "".join(c["source"])
    )
    nb["cells"][loader_index]["source"] = LOADER.splitlines(keepends=True)
    nb["cells"][loader_index]["outputs"] = []
    nb["cells"][loader_index]["execution_count"] = None

    insert_at = next(
        i
        for i, c in enumerate(nb["cells"])
        if c["cell_type"] == "markdown"
        and "## 7 · Conclusion" in "".join(c["source"])
    )
    nb["cells"][insert_at:insert_at] = CELLS
    NOTEBOOK.write_text(
        json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"updated {NOTEBOOK} with {len(CELLS)} standard-MCBM catch-up cells")


if __name__ == "__main__":
    main()
