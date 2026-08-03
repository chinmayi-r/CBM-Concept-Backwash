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
proof_rows = []
remaining_rows = {}
for model, q0 in PAIRED.items():
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

fig, axes = plt.subplots(1, len(MODEL_FILES), figsize=(17, 4.5), sharey=True)
for ax, model in zip(axes, MODEL_FILES):
    d = PROOF_WATERFALL.query("model == @model").set_index("part").reindex(ORDER)
    x = np.arange(len(d)); w = .25
    ax.bar(x-w, d.standard_rate_all, w, label="standard: all valid")
    ax.bar(x, d.standard_rate_high_visibility, w, label="standard: high visibility")
    ax.bar(x+w, d.rl_rate_same_high_visibility, w, label="RLv2: same high-vis rows")
    ax.set_xticks(x); ax.set_xticklabels(ORDER, rotation=25)
    ax.set_title(model); ax.set_ylim(0, 1)
    ax.set_ylabel("candidate-event rate")
axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
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
for model, q0 in PAIRED.items():
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

fig, axes = plt.subplots(1, len(MODEL_FILES), figsize=(17, 4.5), sharey=True)
for ax, model in zip(axes, MODEL_FILES):
    d = RESIDUAL_ORGANIZATION.query("model == @model").set_index("part").reindex(ORDER)
    x = np.arange(len(d)); w = .25
    ax.bar(x-w, d.baseline_brier, w, label="no grouping")
    ax.bar(x, d.after_variant_direction_brier, w, label="+ variant/direction")
    ax.bar(x+w, d.after_source_species_brier, w, label="+ source species")
    ax.set_xticks(x); ax.set_xticklabels(ORDER, rotation=25)
    ax.set_title(model); ax.set_ylabel("leave-one-out prediction error (lower is better)")
axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
plt.tight_layout(); plt.show()
""", "rl_residual_organization"))

RL_EXPLANATION = clean(cell("markdown", r"""
### How to read this figure

- First bar: all valid standard-training swaps.
- Second bar: only high-visibility swaps. Its change is **selection**, not a
  causal percentage removed by visibility.
- Third bar: the matched RLv2 model on those exact same images. The `1 -> 0`,
  `0 -> 1`, and `1 -> 1` columns are the causal label-intervention accounting.
- `1 -> 1` is what remains. Variant pair and source species may organize this
  remainder, but they do not become causes until separately manipulated.

The second figure asks whether exact variant/direction and then source species
help predict which same-image RLv2 candidate events remain. Falling prediction
error means that factor **organizes** the remainder. It is deliberately not
shown as “another percentage causally removed.” The final residual is the
remaining `1 -> 1` count plus whatever prediction error remains after the last
grouping step.

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
        "02": (ROOT / "notebooks/02_funnybirds_cbm.ipynb", [FB_HEADER]),
        "03": (ROOT / "notebooks/03_funnybirds_mcbm.ipynb", [FB_MCBM_GATE]),
        "03rl": (ROOT / "notebooks/03rl_funnybirds_mcbm_relabeled.ipynb",
                 [FB_HEADER, RL_CODE, RL_RESIDUAL_CODE, RL_EXPLANATION]),
        "05": (ROOT / "notebooks/05_cub_cbm.ipynb", [CUB_GATE]),
        "06": (ROOT / "notebooks/06_cub_mcbm.ipynb", [MCBM_CUB_GATE]),
    }
    selected = [args.only] if args.only else list(jobs)
    for key in selected:
        install(*jobs[key])


if __name__ == "__main__":
    main()
