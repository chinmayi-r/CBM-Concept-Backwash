#!/usr/bin/env python3
"""Backport useful MCBM discriminating tests into the standard CBM notebook.

The script is idempotent and intentionally does not execute the notebook.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "02_funnybirds_cbm.ipynb"
TAG = "mcbm-backport"


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


CELLS = [
    cell(
        "markdown",
        """## 10a · Stronger causal response and species controls

The MCBM analysis exposed two tests that should also be applied backward to the
standard CBM.

First define

`response_delta = (z_donor-z_source)_swap - (z_donor-z_source)_original`.

This is stronger than looking only at donor-score rise because the source score
may also move. Positive values mean the complete donor-versus-source comparison
moved toward the inserted part.
""",
    ),
    cell(
        "code",
        r'''if S is not None:
    D=S.copy()
    if "response_delta" not in D:
        D["response_delta"]=D.margin-(D.z_new_orig-D.z_old_orig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    vals=[D[D.part==p].response_delta.dropna().values for p in ORDER]
    axes[0].boxplot(vals,labels=ORDER,showfliers=False)
    axes[0].axhline(0,color="black",linewidth=1)
    axes[0].set_ylabel("response_delta")
    axes[0].set_title("Did the full margin move toward the inserted part?")

    M=(D.groupby(["part","direction"]).margin.mean()
         .unstack("direction").dropna())
    colors={p:c for p,c in zip(ORDER,plt.cm.tab10.colors)}
    for p,r in M.iterrows():
        axes[1].scatter(r["fwd"],r["bwd"],s=65,color=colors[p],label=p)
        axes[1].annotate(p,(r["fwd"],r["bwd"]),xytext=(4,4),
                         textcoords="offset points",fontsize=8)
    lim=max(abs(M[["fwd","bwd"]].values).max(),1)
    axes[1].plot([-lim,lim],[-lim,lim],"--",color="green",
                 label="same direction")
    axes[1].plot([-lim,lim],[lim,-lim],":",color="red",
                 label="cancellation")
    axes[1].axhline(0,color="gray",linewidth=.7)
    axes[1].axvline(0,color="gray",linewidth=.7)
    axes[1].set_xlabel("mean forward margin")
    axes[1].set_ylabel("mean backward margin")
    axes[1].set_title("Do both swap directions tell the same story?")
    axes[1].legend(fontsize=7,ncol=2)
''',
    ),
    cell(
        "markdown",
        """**Interpretation rule.** Positive `response_delta` means the inserted
pixels influenced the correct comparison. A negative final margin means that
movement was still insufficient to overcome the old source/body preference.
Forward/backward points near the positive diagonal rule out cancellation as the
reason for the average.

The second backport completes the species test. Raw source-species failure can
merely restate source-tail variant because every species has one canonical tail.
Subtract each source variant's mean failure rate before asking whether species
differences remain.
""",
    ),
    cell(
        "code",
        r'''if S is not None and {"sid_src","var_src"}.issubset(S.columns):
    t=S[S.part=="tail"].copy()
    raw=(t.assign(violation=~t.ordering_correct.astype(bool))
          .groupby(["sid_src","var_src"]).violation.mean().reset_index())
    raw["variant_mean"]=raw.groupby("var_src").violation.transform("mean")
    raw["within_variant_residual"]=raw.violation-raw.variant_mean
    raw=raw.sort_values("within_variant_residual",ascending=False)
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    sc=axes[0].scatter(range(len(raw)),raw.violation,
                       c=raw.var_src,cmap="tab10",s=28)
    axes[0].set_xlabel("source species, sorted by within-variant residual")
    axes[0].set_ylabel("raw violation rate")
    axes[0].set_title("Raw source-species failure")
    fig.colorbar(sc,ax=axes[0],label="source tail variant")
    sc2=axes[1].scatter(range(len(raw)),raw.within_variant_residual,
                        c=raw.var_src,cmap="tab10",s=28)
    axes[1].axhline(0,color="black",linestyle="--",linewidth=1)
    axes[1].set_xlabel("source species, same order")
    axes[1].set_ylabel("violation rate - source-variant mean")
    axes[1].set_title("Species/body difference remaining within variant")
    fig.colorbar(sc2,ax=axes[1],label="source tail variant")
    display(raw.head(15).round(3))
''',
    ),
    cell(
        "markdown",
        """**Limited conclusion.** Residual spread supports an additional
source-body/species association beyond source-tail variant. It remains
observational because the body itself was not independently swapped.

### Updated CBM evidence chain

1. The CBM is accurate and appears grounded on ordinary images.
2. Controlled swaps reveal a graded problem: tail is worst, while beak and eye
   are also imperfect; wing and foot are stronger controls.
3. `response_delta` asks whether inserted pixels moved the complete
   donor-versus-source comparison. Positive movement with a negative final margin
   means “the model saw the part, but the old context remained stronger.”
4. Visibility is tested next and is only a partial explanation.
5. Variant confusion and donor-pair tests show unequal variants rather than one
   universally broken slot.
6. Within-variant residuals test whether source body/species still matters after
   accounting for canonical variant.
7. The full concept vector stores substantial species information, but downstream
   species-probability effects are modest.
8. Notebook 03 then asks whether minimality removes this grounding problem.
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
        if c["cell_type"] == "code"
        and 'SW = CURATED/"swap"/"funnybirds-cbm-s1.csv"' in "".join(c["source"])
    )
    source = "".join(nb["cells"][loader_index]["source"])
    source = source.replace(
        'SW = CURATED/"swap"/"funnybirds-cbm-s1.csv"',
        '''_swap_candidates = [
    CURATED/"swap_fixed_v2_attempt2"/"funnybirds-cbm-s1.csv",
    CURATED/"swap_fixed_v2"/"funnybirds-cbm-s1.csv",
    CURATED/"swap"/"funnybirds-cbm-s1.csv",
]
SW = next((p for p in _swap_candidates if p.exists()), _swap_candidates[0])
print("CBM swap source:", SW)
if "fixed_v2" not in str(SW):
    print("[PROVISIONAL] using legacy independently rendered swaps")''',
    )
    nb["cells"][loader_index]["source"] = source.splitlines(keepends=True)
    nb["cells"][loader_index]["outputs"] = []
    nb["cells"][loader_index]["execution_count"] = None

    insert_at = next(
        i
        for i, c in enumerate(nb["cells"])
        if c["cell_type"] == "markdown"
        and "## 11 · What is established" in "".join(c["source"])
    )
    nb["cells"][insert_at:insert_at] = CELLS
    NOTEBOOK.write_text(
        json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"updated {NOTEBOOK} with {len(CELLS)} MCBM-to-CBM backport cells")


if __name__ == "__main__":
    main()
