"""Install a shared predicate ledger and honest residual decomposition in notebooks."""

from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
SECTION = "predicate_proof_ladder_v1"


def cell(kind: str, source: str, name: str) -> dict:
    source = dedent(source).strip() + "\n"
    return {
        "cell_type": kind,
        "execution_count": None if kind == "code" else None,
        "metadata": {"proof_section": SECTION, "proof_name": name},
        "outputs": [] if kind == "code" else None,
        "source": source.splitlines(keepends=True),
        "id": hashlib.sha1(f"{SECTION}:{name}:{source}".encode()).hexdigest()[:8],
    }


def clean(c: dict) -> dict:
    if c["cell_type"] == "markdown":
        c.pop("execution_count", None)
        c.pop("outputs", None)
    return c


FB_HEADER = clean(cell("markdown", r"""
## Predicate-first decision gate

Before calling a negative post-swap margin backwash, require both:

1. `response_delta > 0`: the inserted donor pixels moved the model toward the donor;
2. `margin_after_swap < 0`: despite that response, the old source still wins.

Model health, semantic renderer validity, forward/backward agreement, fixed image
hashes, and seed scope are separate prerequisites. See
`../PREDICATE_PROOF_LEDGER.md`. A visibility filter is a selection check, not a
causal subtraction. The matched RLv2 comparison is the causal subtraction.
""", "fb_header"))

FB_STANDARD_RESIDUAL = clean(cell("code", r"""
# Standard-CBM discovery only: no RLv2 and no MCBM.
if S is not None:
    fb = S.copy()
    if "response_delta" not in fb:
        fb["response_delta"] = fb.margin - (fb.z_new_orig - fb.z_old_orig)
    fb["candidate_event"] = fb.response_delta.gt(0) & fb.margin.lt(0)
    fb_thresholds = {
        part: max(8, int(d.loc[d.pixel_count_cf.gt(0), "pixel_count_cf"].median()))
        for part, d in fb.groupby("part")
    }

    def loo_brier(frame, outcome, columns, alpha=5.0):
        y = frame[outcome].astype(float)
        p0 = float(y.mean())
        if not columns:
            return float(((y-p0)**2).mean())
        st = frame.assign(_y=y).groupby(columns, dropna=False)._y.agg(["sum", "count"])
        k = frame[columns].merge(st.reset_index(), on=columns, how="left")
        pred = (k["sum"].to_numpy()-y.to_numpy()+alpha*p0)/(k["count"].to_numpy()-1+alpha)
        return float(np.mean((y.to_numpy()-pred)**2))

    fb_rows = []
    for part, all_rows in fb.groupby("part"):
        high = all_rows[all_rows.pixel_count_cf >= fb_thresholds[part]].copy()
        fb_rows.append({
            "part": part, "n_all": len(all_rows),
            "candidate_rate_all": all_rows.candidate_event.mean(),
            "n_high_visibility": len(high),
            "candidate_rate_high_visibility": high.candidate_event.mean(),
            "baseline_error": loo_brier(high, "candidate_event", []),
            "after_variant_direction_error": loo_brier(
                high, "candidate_event", ["var_src", "var_donor", "direction"]),
            "after_source_species_error": loo_brier(
                high, "candidate_event", ["var_src", "var_donor", "direction", "sid_src"]),
            "remaining_count": int(high.candidate_event.sum()),
        })
    FB_STANDARD_RESIDUAL = pd.DataFrame(fb_rows).set_index("part").reindex(ORDER)
    display(FB_STANDARD_RESIDUAL.round(4))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    x = np.arange(len(FB_STANDARD_RESIDUAL)); w=.36
    axes[0].bar(x-w/2, FB_STANDARD_RESIDUAL.candidate_rate_all, w, label="all valid swaps")
    axes[0].bar(x+w/2, FB_STANDARD_RESIDUAL.candidate_rate_high_visibility, w,
                label="high-visibility swaps")
    axes[0].set_xticks(x); axes[0].set_xticklabels(FB_STANDARD_RESIDUAL.index)
    axes[0].set_ylim(0,1); axes[0].set_ylabel("candidate-event rate")
    axes[0].set_title("Selection check: what remains with a clearly visible donor part?")
    axes[0].legend()

    stages=["baseline_error","after_variant_direction_error","after_source_species_error"]
    labels=["no grouping","+ exact variants/direction","+ source species"]
    for j,(stage,label) in enumerate(zip(stages,labels)):
        axes[1].plot(x, FB_STANDARD_RESIDUAL[stage], "o-", label=label)
    axes[1].set_xticks(x); axes[1].set_xticklabels(FB_STANDARD_RESIDUAL.index)
    axes[1].set_ylabel("leave-one-out prediction error (lower is better)")
    axes[1].set_title("Observational organization of the high-visibility remainder")
    axes[1].legend()
    plt.tight_layout(); plt.show()
else:
    print("[pending] validated standard-CBM swap CSV")
""", "fb_standard_residual"))

FB_STANDARD_EXPLANATION = clean(cell("markdown", r"""
### What was and was not subtracted

This is **standard CBM only**.

**Axes.** In the left panel, the x-axis is bird part and the y-axis is the
fraction of strict candidate events; blue uses every valid swap and orange keeps
only clearly visible donor parts. In the right panel, the x-axis is bird part
and the y-axis is leave-one-out prediction error (lower is better). The blue,
orange, and green lines add no grouping, exact variants/direction, and source
species in that order.

1. The candidate event requires a positive donor response and a negative final
   donor-minus-source margin.
2. Keeping only high-visibility swaps shows whether failures concentrate among
   poorly visible edits. This selects rows; it does not causally repair them.
3. Lower prediction error after adding exact variants/direction means these
   variables organize the remainder.
4. A further decrease after source species means species/body context adds
   predictive structure after those controls.
5. `remaining_count` is what is still present. Variant and species are not called
   causes until independently manipulated.

Use `../CBM_CROSS_DATASET_PROOF_MAP.md` to compare every stage with notebook 05.
""", "fb_standard_explanation"))

FB_MCBM_GATE = clean(cell("markdown", r"""
## Predicate-first MCBM decision gate

The CBM phenomenon must first pass the FunnyBird clean-swap predicates. This
notebook then asks whether increasing gamma changes that same event on the same
kind of intervention. A smaller raw `z` scale is not a repair by itself. A repair
requires a more positive donor response/final margin or fewer candidate events,
with model health retained and the direction replicated across the gamma/seed
scope being claimed. See `../PREDICATE_PROOF_LEDGER.md`.
""", "fb_mcbm_gate"))

RL_CODE = clean(cell("code", r"""
# Same-row waterfall. This never adds percentages from different populations.
proof_pairs = {"CBM": PAIRED["CBM"]}
if "broad_pairs" in globals() and broad_pairs:
    for gamma, q0 in sorted(broad_pairs.items()):
        q = q0.copy()
        for col in ["var_src", "var_donor"]:
            if col not in q and f"{col}_standard" in q:
                if not (q[f"{col}_standard"] == q[f"{col}_rl"]).all():
                    raise ValueError(f"all-gamma proof column differs: {col}, gamma={gamma}")
                q[col] = q[f"{col}_standard"]
        proof_pairs[f"MCBM γ={gamma:g}"] = q
else:
    proof_pairs.update({k:v for k,v in PAIRED.items() if k != "CBM"})
proof_model_order = list(proof_pairs)

proof_rows = []
remaining_rows = {}
for model, q0 in proof_pairs.items():
    q = q0.copy()
    q["candidate_standard"] = (
        q["swap_moved_toward_donor_standard"].astype(bool)
        & q["margin_standard"].lt(0)
    )
    q["candidate_rl"] = (
        q["swap_moved_toward_donor_rl"].astype(bool)
        & q["margin_rl"].lt(0)
    )
    for part, d_all in q.groupby("part"):
        threshold = visibility_thresholds.get(part, 1)
        d = d_all.loc[d_all.pixel_count_cf_standard >= threshold].copy()
        if d.empty:
            continue
        resolved = d.candidate_standard & ~d.candidate_rl
        introduced = ~d.candidate_standard & d.candidate_rl
        remaining = d.candidate_standard & d.candidate_rl
        remaining_rows[(model, part)] = d.loc[remaining].copy()
        proof_rows.append({
            "model": model, "part": part,
            "n_all": len(d_all),
            "standard_rate_all": d_all.candidate_standard.mean(),
            "n_high_visibility": len(d),
            "standard_rate_high_visibility": d.candidate_standard.mean(),
            "rl_rate_same_high_visibility": d.candidate_rl.mean(),
            "resolved_1_to_0": resolved.mean(),
            "introduced_0_to_1": introduced.mean(),
            "remaining_1_to_1": remaining.mean(),
            "remaining_count": int(remaining.sum()),
        })

PROOF_WATERFALL = pd.DataFrame(proof_rows)
display(PROOF_WATERFALL.round(3))

stages = [
    ("standard_rate_all", "standard: all valid"),
    ("standard_rate_high_visibility", "standard: high visibility"),
    ("rl_rate_same_high_visibility", "RLv2: same high-vis rows"),
]
resid_vmax = RESIDUAL_ORGANIZATION[[c for c,_ in stages]].max().max()
fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
for ax, (column, title) in zip(axes, stages):
    matrix = (PROOF_WATERFALL.pivot(index="model", columns="part", values=column)
              .reindex(index=proof_model_order, columns=ORDER))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="magma_r", aspect="auto")
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER, rotation=30)
    ax.set_yticks(range(len(matrix))); ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=axes, label="candidate-event rate", shrink=.82)
plt.tight_layout(); plt.show()
""", "rl_waterfall"))

RL_RESIDUAL_CODE = clean(cell("code", r"""
# Observational organization of the RLv2 remainder. Lower Brier error means the
# added grouping variables predict where candidate events remain more reliably.
def smoothed_loo_brier(d, outcome, group_cols, alpha=5.0):
    y = d[outcome].astype(float)
    p0 = float(y.mean())
    if not group_cols:
        return float(((y - p0) ** 2).mean())
    stats = d.assign(_y=y).groupby(group_cols, dropna=False)._y.agg(["sum", "count"])
    keyed = d[group_cols].merge(stats.reset_index(), on=group_cols, how="left")
    pred = (keyed["sum"].to_numpy() - y.to_numpy() + alpha * p0) / (
        keyed["count"].to_numpy() - 1 + alpha
    )
    return float(np.mean((y.to_numpy() - pred) ** 2))

organization_rows = []
for model, q0 in proof_pairs.items():
    q = q0.copy()
    q["candidate_rl"] = (
        q.swap_moved_toward_donor_rl.astype(bool) & q.margin_rl.lt(0)
    )
    for part, d0 in q.groupby("part"):
        threshold = visibility_thresholds.get(part, 1)
        d = d0.loc[d0.pixel_count_cf_standard >= threshold].copy()
        if len(d) < 10:
            continue
        b0 = smoothed_loo_brier(d, "candidate_rl", [])
        bv = smoothed_loo_brier(
            d, "candidate_rl", ["var_src", "var_donor", "direction"]
        )
        bs = smoothed_loo_brier(
            d, "candidate_rl", ["var_src", "var_donor", "direction", "sid_src"]
        )
        organization_rows.append({
            "model": model, "part": part, "n": len(d),
            "remaining_rate": d.candidate_rl.mean(),
            "baseline_brier": b0,
            "after_variant_direction_brier": bv,
            "after_source_species_brier": bs,
            "variant_direction_gain": b0 - bv,
            "additional_species_gain": bv - bs,
        })

RESIDUAL_ORGANIZATION = pd.DataFrame(organization_rows)
display(RESIDUAL_ORGANIZATION.round(4))

stages = [
    ("baseline_brier", "no grouping"),
    ("after_variant_direction_brier", "+ exact variants/direction"),
    ("after_source_species_brier", "+ source species"),
]
fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
for ax, (column, title) in zip(axes, stages):
    matrix = (RESIDUAL_ORGANIZATION.pivot(index="model", columns="part", values=column)
              .reindex(index=proof_model_order, columns=ORDER))
    im = ax.imshow(matrix, vmin=0, vmax=resid_vmax, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER, rotation=30)
    ax.set_yticks(range(len(matrix))); ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if value > resid_vmax/2 else "black")
fig.colorbar(im, ax=axes, label="leave-one-out prediction error", shrink=.82)
plt.tight_layout(); plt.show()
""", "rl_residual_organization"))

RL_EXPLANATION = clean(cell("markdown", r"""
### How to read this figure

**First heatmap axes.** Rows are CBM followed by every MCBM γ. Columns are bird
parts. Each cell's number and color are the fraction of strict candidate events.
The three panels are: every valid standard swap; high-visibility standard swaps;
and the matched RLv2 model on those exact same high-visibility images.

- Moving from panel 1 to panel 2 is a visibility **selection** check, not a
  causal percentage removed.
- Moving from panel 2 to panel 3 is the matched label intervention.
- The table reports `1 -> 0` resolved events, `0 -> 1` introduced events, and
  `1 -> 1` remaining events. These transitions use the same rendered rows.
- `1 -> 1` is what remains. Variant pair and source species may organize this
  remainder, but they do not become causes until separately manipulated.

**Second heatmap axes.** Rows again identify CBM/MCBM γ and columns identify bird
parts. Each cell is leave-one-out prediction error, so smaller numbers/darker
movement across panels mean the added information predicts where RLv2 failures
remain. The panels add: no grouping; exact source/donor variants plus direction;
then source species. This is deliberately not shown as another causal percentage
removed. The final residual is the remaining `1 -> 1` count plus whatever
prediction error remains after the last grouping step.

The final conclusion must report the remaining count. It must not imply that the
listed influences add to 100% or that all causes have been identified.
""", "rl_explanation"))

CUB_GATE = clean(cell("markdown", r"""
## Predicate-first causal boundary

The CUB70 edit tests stop before a causal backwash conclusion:

| Predicate | Current result |
|---|---|
| Healthy ordinary model | available as a guard |
| Clean one-part intervention | failed for current deletion, patch, and paste proxies |
| Positive donor response | failed in the beak/tail pilot (40% positive; median about zero) |
| Old source wins after positive response | not interpretable because the earlier predicates failed |
| Same-image residual waterfall | not licensed |

Therefore the natural visibility, label-conflict, exact-concept, and species
patterns are observational convergence only. They motivate a better CUB
intervention; they do not prove that CUB has the FunnyBird causal phenomenon.
""", "cub_gate"))

CUB_STANDARD_RESIDUAL = clean(cell("code", r"""
# CUB observational analogue. This is not the FunnyBird causal event.
cub = J70[(J70.gt_label == 1) & J70.mask_group.notna()].copy()
cub["model_positive"] = cub.pred_label.astype(bool)
cub["mask_state"] = np.where(
    cub.pixel_count.eq(0), "absent",
    np.where(cub.visible, "visible", "tiny/non-visible"),
)
conflict_lookup = EXACT70[
    ["attribute_type", "concept_name", "label_hidden_rate"]
].drop_duplicates()
cub = cub.merge(
    conflict_lookup, on=["attribute_type", "concept_name"],
    how="left", validate="many_to_one",
)
cub["label_conflict_bin"] = pd.cut(
    cub.label_hidden_rate,
    bins=[-0.001, 0.10, 0.25, 0.50, 0.75, 1.001],
    labels=["≤10%", "10–25%", "25–50%", "50–75%", ">75%"],
)

def cub_loo_brier(frame, outcome, columns, alpha=5.0):
    y = frame[outcome].astype(float)
    p0 = float(y.mean())
    if not columns:
        return float(((y-p0)**2).mean())
    st = frame.assign(_y=y).groupby(columns, dropna=False)._y.agg(["sum", "count"])
    k = frame[columns].merge(st.reset_index(), on=columns, how="left")
    pred = (k["sum"].to_numpy()-y.to_numpy()+alpha*p0)/(k["count"].to_numpy()-1+alpha)
    return float(np.mean((y.to_numpy()-pred)**2))

cub_rows=[]
for part, d in cub.groupby("mask_group"):
    hidden=d[~d.visible]
    cub_rows.append({
        "part":part, "n_positive_labels":len(d), "n_hidden":len(hidden),
        "hidden_prediction_rate":hidden.model_positive.mean(),
        "baseline_error":cub_loo_brier(d,"model_positive",[]),
        "after_visibility_error":cub_loo_brier(d,"model_positive",["mask_state"]),
        "after_label_conflict_error":cub_loo_brier(
            d,"model_positive",["mask_state","label_conflict_bin"]),
        "after_exact_concept_error":cub_loo_brier(
            d,"model_positive",["mask_state","label_conflict_bin","concept_name"]),
        "after_species_error":cub_loo_brier(
            d,"model_positive",["mask_state","label_conflict_bin","concept_name","y_true"]),
        "remaining_hidden_positive_count":int(hidden.model_positive.sum()),
    })
CUB_STANDARD_RESIDUAL=pd.DataFrame(cub_rows).set_index("part").sort_index()
display(CUB_STANDARD_RESIDUAL.round(4))

fig,axes=plt.subplots(1,2,figsize=(15,4.5))
x=np.arange(len(CUB_STANDARD_RESIDUAL))
axes[0].bar(x,CUB_STANDARD_RESIDUAL.hidden_prediction_rate,color="#0072B2")
axes[0].set_xticks(x);axes[0].set_xticklabels(CUB_STANDARD_RESIDUAL.index)
axes[0].set_ylim(0,1);axes[0].set_ylabel("predicted-positive rate")
axes[0].set_title("Positive label but naturally hidden: model still says present")
for stage,label in [
    ("baseline_error","no grouping"),
    ("after_visibility_error","+ mask visibility"),
    ("after_label_conflict_error","+ label/mask conflict"),
    ("after_exact_concept_error","+ exact concept"),
    ("after_species_error","+ species"),
]:
    axes[1].plot(x,CUB_STANDARD_RESIDUAL[stage],"o-",label=label)
axes[1].set_xticks(x);axes[1].set_xticklabels(CUB_STANDARD_RESIDUAL.index)
axes[1].set_ylabel("leave-one-out prediction error (lower is better)")
axes[1].set_title("Which observed variables organize CUB predictions?")
axes[1].legend()
plt.tight_layout();plt.show()
""", "cub_standard_residual"))

CUB_STANDARD_EXPLANATION = clean(cell("markdown", r"""
### Direct relation to the FunnyBird standard-CBM figure

The layout is intentionally similar, but the outcomes are not equal:

**Axes.** In the left panel, the x-axis is CUB mask/part group and the y-axis is
the fraction of naturally hidden positive-labelled rows that the model still
calls positive. In the right panel, the x-axis is the same part group and the
y-axis is leave-one-out prediction error (lower is better). Lines sequentially
add current mask visibility, concept-level label/mask conflict, exact concept,
and species.

- FunnyBird counts validated swap events: new pixels move the donor score, yet
  the old source still wins.
- CUB counts naturally hidden, positive-labelled photographs where the model
  predicts the concept as present.

The CUB lines show, in the requested order, whether current visibility,
concept-level label/mask conflict, exact concept identity, and species help
predict the model output. They do **not** show causal percentages removed. Label
conflict is binned before exact identity so its broad pattern is tested before
letting every exact concept have a separate rate. The final hidden-positive count
is an observational remainder. Because the CUB edit tests failed and no matched
CUB relabel intervention exists, it cannot be promoted to causal backwash.

See `../CBM_CROSS_DATASET_PROOF_MAP.md` for the figure-by-figure pairing with
notebook 02.
""", "cub_standard_explanation"))

MCBM_CUB_GATE = clean(cell("markdown", r"""
## Scope inherited from the CUB CBM predicate gate

This notebook may ask whether gamma changes CUB prediction, compression, natural
visibility associations, or species-linked residuals. It may not say that gamma
repairs or worsens causal CUB backwash, because notebook 05's current CUB edits
failed the clean-intervention and positive-donor-response predicates.
""", "cub_mcbm_gate"))


def install(path: Path, cells: list[dict]) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    nb["cells"] = [
        c for c in nb["cells"]
        if c.get("metadata", {}).get("proof_section") != SECTION
    ]
    nb["cells"].extend(cells)
    seen = set()
    for index, c in enumerate(nb["cells"]):
        cid = c.get("id")
        if not cid or cid in seen:
            payload = f"{path.name}:{index}:{c['cell_type']}:{''.join(c.get('source', []))}"
            cid = hashlib.sha1(payload.encode()).hexdigest()[:8]
            salt = 0
            while cid in seen:
                salt += 1
                cid = hashlib.sha1(f"{payload}:{salt}".encode()).hexdigest()[:8]
            c["id"] = cid
        seen.add(cid)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"installed predicate proof section in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["02", "03", "03rl", "05", "06"])
    args = parser.parse_args()
    jobs = {
        "02": (ROOT / "notebooks/02_funnybirds_cbm.ipynb",
               [FB_HEADER, FB_STANDARD_RESIDUAL, FB_STANDARD_EXPLANATION]),
        "03": (ROOT / "notebooks/03_funnybirds_mcbm.ipynb", [FB_MCBM_GATE]),
        "03rl": (ROOT / "notebooks/03rl_funnybirds_mcbm_relabeled.ipynb",
                 [FB_HEADER, RL_CODE, RL_RESIDUAL_CODE, RL_EXPLANATION]),
        "05": (ROOT / "notebooks/05_cub_cbm.ipynb",
               [CUB_GATE, CUB_STANDARD_RESIDUAL, CUB_STANDARD_EXPLANATION]),
        "06": (ROOT / "notebooks/06_cub_mcbm.ipynb", [MCBM_CUB_GATE]),
    }
    selected = [args.only] if args.only else list(jobs)
    for key in selected:
        install(*jobs[key])


if __name__ == "__main__":
    main()
