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
species violation rates, coloured by the source tail variant. The raw species
plot is descriptive because species and canonical variant are linked.
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
    fig,axes=plt.subplots(1,2,figsize=(13,4))
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
    display(raw.head(15).round(3))
''',
    ),
    cell(
        "markdown",
        """**Alternative explanation.** Species differences can merely restate
variant differences because every species has one canonical tail. The
discriminating test is variation among species sharing the same source variant;
this plot motivates that test but does not complete it.

**Limited conclusion rule.** Variant-pair differences are established if cells
differ with adequate counts. A separate body/species effect remains provisional
until compared within source-variant groups.

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
