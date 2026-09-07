#!/usr/bin/env python3
"""Build the standard FunnyBird MCBM report from the locked 03/06 roadmap."""
from __future__ import annotations

import hashlib
import json
import argparse
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks/03_funnybirds_mcbm.ipynb"


def _src(s: str) -> list[str]:
    return (textwrap.dedent(s).strip("\n") + "\n").splitlines(keepends=True)


def md(tag: str, s: str) -> dict:
    tag = re.sub(r"[^a-zA-Z0-9_-]", "-", tag)[:48]
    s = textwrap.dedent(s).strip("\n") + "\n"
    return {"cell_type": "markdown", "id": f"03-{tag}-{hashlib.sha1(s.encode()).hexdigest()[:8]}",
            "metadata": {}, "source": s.splitlines(keepends=True)}


def code(tag: str, s: str, alt: str) -> dict:
    tag = re.sub(r"[^a-zA-Z0-9_-]", "-", tag)[:48]
    s = "# ALT: " + alt + "\n" + textwrap.dedent(s).strip("\n") + "\n"
    return {"cell_type": "code", "id": f"03-{tag}-{hashlib.sha1(s.encode()).hexdigest()[:8]}",
            "metadata": {"alt": alt}, "execution_count": None, "outputs": [],
            "source": s.splitlines(keepends=True)}




def review(n: int | str) -> dict:
    # Historical observations remain in git, not as current figure conclusions.
    return pending_review(str(n))


def pending_review(label: str) -> dict:
    return md(f"review-{label}", f"""
    ### Review record for Figure {label}

    **INCOMPLETE: current output requires execution and visual review.**

    - **Literal result:** Record the actual values, sample sizes and exceptions.
    - **What it supports:** State only the claim measured by this figure.
    - **Plausible alternative:** Give a concrete competing explanation.
    - **Discriminating test:** State which observation would distinguish it.
    - **Next question:** Explain why the next analysis follows from this result.

    Display the complete current figure in chat before filling this record.
    """)


def verify_implementation_contract() -> None:
    """Fail before notebook generation if the pinned model/loss sources drift."""
    required = {
        ROOT / "external/ConceptBottleneck/CUB/train.py": [
            "args.attr_loss_weight * attr_criterion",
            "total_loss = total_loss / (1 + args.attr_loss_weight * args.n_attributes)",
            "criterion = torch.nn.CrossEntropyLoss()",
        ],
        ROOT / "external/ConceptBottleneck/CUB/models.py": [
            "def ModelXtoCtoY",
            "return End2EndModel",
        ],
        ROOT / "external/minimal_cbm/src/models/mcbm.py": [
            "z_logits = torch.unsqueeze(6 * c - 3, -1)",
            "loss_z_j = 0.2 *",
            "loss = y_loss + self.beta * c_loss + self.gamma * z_loss",
            "sampled_z = z + self.var_z * torch.randn_like(z) if sampling else z",
        ],
        ROOT / "external/minimal_cbm/src/models/cbm.py": [
            "self.mlp_c = nn.ModuleList",
            "c_logits = torch.stack",
        ],
        ROOT / "external/minimal_cbm/src/models/vanilla.py": [
            "self.mlp_y, self.act_y = self._build_head",
            "y_logits = self.mlp_y(z)",
        ],
        ROOT / "train/configs/funnybirds-mcbm.yaml": [
            "hidden_dims_y: 256",
            "var_z: 1",
            "hidden_dims_c: [3]",
            "beta: 1.0",
        ],
    }
    for path, snippets in required.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"canonical source is unavailable: {path}; initialize pinned submodules"
            )
        source = path.read_text(encoding="utf-8")
        missing = [snippet for snippet in snippets if snippet not in source]
        if missing:
            raise RuntimeError(f"canonical architecture/loss source drift in {path}: {missing}")
    print("[MCBM REPORT SOURCE CONTRACT PASS] Koh Joint and minimal_cbm wiring/losses verified")


cells: list[dict] = []
cells += [md("title", r"""
# 03 · FunnyBird standard MCBM — does minimality repair concept grounding?

**Report question.** Notebook 02 discovered controlled concept backwash in a
standard CBM. When MCBM increasingly penalizes information in each internal
concept slot beyond its binary label, do the same validated part replacements
become more correctly attributed?

**Population.** Standard non-RLv2 MCBM at gamma `0, 0.1, 0.3, 1, 3, 5`, with
standard CBM retained only as the discovery reference. The all-gamma fixed-render
causal comparison currently has one independently trained seed per gamma.

**Claims available here.** This notebook can test whether gamma compresses the
implemented internal representation and whether that compression repairs the
already-defined FunnyBird controlled event. It cannot attribute standard-CBM
versus MCBM-gamma-zero differences to minimality, call one causal seed a stable
gamma curve, or replace the earlier standard-CBM discovery with an MCBM result.

This is the non-RLv2 MCBM stage. RLv2 is a later causal label test.

**Part names are outcomes, not mechanisms.** The general hypothesis is a
competition between the original-image source advantage and a response driven
by the changed part pixels. The starting advantage is not pure context because
the original source part is still present; later species-residual tests ask
whether context helps maintain it after replacement. The report
must evaluate all five parts without presupposing their ordering. Notebook 06 must establish its own CUB ordering
from all exact concepts and masks; it must not presume that CUB tail is special.

More precisely, the proposed contributors are properties of each
part/concept and its data: the original source-versus-donor margin, the size of
the response to the inserted pixels, positive-label/visibility conflict,
exact-value difficulty, the number and frequency of alternatives, and residual
source-species organization. Backwash should be strongest wherever these
properties combine unfavourably. Whether these contributors predict the observed
ordering is a question for the current outputs, not an assumed result.
"""), md("roadmap", r"""
## What this notebook must prove, and how it continues notebook 02

Notebook 02 established the standard-CBM event on a controlled replacement:

`response_delta > 0 and m_cf < 0`.

This report does not rediscover or replace that result. It asks whether the
MCBM training change repairs the same event on the same rendered images.

### What would count as an MCBM grounding repair?

Compression alone is not success. The report will call MCBM a grounding repair
only if the following measurements move together relative to the MCBM
`gamma=0` baseline, without broken concept outputs:

| Required result | Exact measurement | Repair direction |
|---|---|---|
| the added loss did what it was designed to do | distance of `h_ij` from `+3/-3` and within-label `h` spread | lower |
| the inserted donor pixels still affect the model | `response_delta=m_cf-m_orig`, alongside no-response and collapse counts | genuine response is retained; raw magnitude need not match another model's scale |
| the donor concept finishes above the old source | `m_cf=z_donor,cf-z_source,cf` | rises above zero more often |
| responded-but-source-wins events become rarer | `P(response_delta>0 and m_cf<0)` | lower |
| the model identifies the exact inserted value | argmax over every value belonging to the replaced part | higher |
| the result is not a single-model accident | independent trained-model seeds with the same fixed-render replay | replicated |

Improvement in one selected part or gamma is reported as a part-specific result,
not a general repair. Thousands of swaps from one checkpoint do not satisfy the
last row.

### Visual parity rule inherited from notebook 02rl

- When Standard and all MCBM gammas use the identical formula, population, and
  axes, Standard is one additional row, bar, or line in the same figure.
- When adding six gammas would change the construction enough to hide the
  original result, display the unchanged Standard graph itself, followed by
  separate gamma graphs with the same construction. A reference alone does
  not satisfy parity.
- MCBM-only quantities such as internal-slot distance from plus/minus three,
  local concept-head slope, and use by the nonlinear species MLP have no Koh
  equivalent and therefore receive separate figures.
- No historical minimal_cbm-CBM checkpoint is used as a visual convenience.
  Standard always means the accepted Koh Joint ResNet-50 model.

| Step | Needed fact | Output | Why it is needed |
|---|---|---|---|
| 1 | every input, checkpoint, render ID, and hash is valid | 1 | unequal pixels or populations invalidate a gamma comparison |
| 2 | gamma changes the quantity named by the MCBM loss without hiding broken exact outputs | 2, 2b | compression must be demonstrated before it explains anything |
| 3 | compression and the learned `h -> z` head are separated, and the one collapsed output is bounded | 2c, 2d | MCBM-specific mechanisms must not be confused with grounding or allowed to create a pooled result |
| 4 | MCBM gamma 0 is compared fairly with standard CBM | 3 | gamma 0 is the architecture/noise baseline, not evidence for minimality |
| 5 | original donor/source scores, starting preference, donor-score gain, and source-score decrease are separated | 4, 4b, 4c | the final outcome depends on both the original source advantage and pixel-driven movement |
| 6 | the final donor/source outcome and all three outcome states are explicit | 5, 5b | this is the primary grounding endpoint and removes ambiguity about the failure predicate |
| 7 | direction and visibility alternatives are tested | 6, 7 | pooling or tiny target parts must not create the result |
| 8 | training conflict and exact-value difficulty are carried forward | 7b, 8, 8b | notebook-02 contributors must not disappear from the MCBM story |
| 9 | source species is tested after exact-value matching, then proposed contributors are tested on held-out rows | 9, 9b | plausible associations are not automatically explanations |
| 10 | species decoding and recall use structural controls | 10, 11 | species information is opportunity, not grounding proof |
| 11 | the four notebook-02 measurements are aligned without being added | 11b | the same proposed reasons are compared on their real denominators |
| 12 | downstream class cost and independent-seed coverage are explicit | 12, 13 | grounding failure, class harm, and reproducibility are different claims |
| 13 | every preregistered repair requirement is shown together | 14 | compression is not credited merely because one number shrank |

### Capabilities and limits that determine this MCBM design

- **Same causal operation as notebook 02:** every MCBM gamma is replayed on the
  same validated FunnyBird donor-part replacements. Body, pose, camera, and
  background remain fixed, so `response_delta` and `m_cf` retain the same causal
  meaning.
- **New MCBM mechanism available:** saved ordinary-image predictions contain the
  internal slots `h`, and saved checkpoints contain the learned `h -> z` heads.
  This permits compression and local-head tests that standard CBM did not need.
- **Mechanism boundary:** the accepted swap CSVs contain counterfactual raw logits
  `z`, but not counterfactual internal slots `h`. Figure 2c can characterize the
  head locally on ordinary images; it cannot decide whether a weak swap response
  arose in the encoder, the learned head, or both.
- **Contributor boundary:** visibility, label conflict, exact value, support, and
  source species are measured associations. They are not independent causal
  manipulations and are not added as percentages.
- **Replication boundary:** model health has several seeds at most gammas, but the
  accepted all-gamma fixed-render causal replay currently has one seed per gamma.

### Predictions stated before the results

1. Increasing `gamma` should reduce the distance of internal slot `h_ij` from
   its label target `+3` or `-3`, and reduce within-label variation in `h`.
2. The loss does **not** mention part pixels. A species/body shortcut can still
   produce the correct ±3 target.
3. If minimality repairs grounding, `m_cf` should rise, controlled-backwash
   rates should fall, and exact donor-value recognition should rise across
   parts as gamma increases.
4. If minimality only compresses, model health can remain high while grounding
   remains unchanged or worsens. Weak local visual variation may be suppressed
   because within-label variation is exactly what the penalty removes.
5. Label/mask conflict is a property of the unchanged training records. It is
   constant across gamma; gamma may interact with it but cannot change its count.
6. Fixed-render gamma trends remain provisional while only seed 1 has causal
   replay. Repeated swaps are not independent trained models.

### The same contributor questions as notebook 02

- **visibility/occlusion:** do failures remain when the inserted part is large?
- **label/visibility conflict:** did training call a concept positive while its
  part was not visible?
- **exact-value difficulty and support:** are some inserted values consistently
  confused, rare, or selected from more alternatives?
- **source species/body residual:** after exact values are matched, does the
  unchanged source bird still organize the final raw-logit margin?

These quantities have different denominators and are never added into a single
backwash score. They are tested against the controlled outcome.
"""), md("architecture", r"""
## Standard CBM versus MCBM: different wiring, related questions

The two primary papers do **not** implement the same network with one extra
loss term.

**Accepted Koh Joint Standard CBM from notebook 02**

```text
image x -> ResNet-50 -> 26 raw concept logits z
                              |-> sigmoid(z_j) for concept j
                              `-> one linear layer Wz+b -> 50 species logits
```

The species layer reads the exact 26 raw concept logits studied in notebook 02.
The accepted training objective is

`L_Koh = [L_species(Wz+b,y) + 0.01 * sum_j L_concept,j(z_j,c_j)] / [1 + 0.01*26]`.

Here `L_species` is 50-class cross-entropy and each `L_concept,j` is Koh's
weighted binary-logit loss. During the accepted `use_aux` training, each species
and concept term also includes 0.4 times its auxiliary-output loss; the displayed
equation abbreviates that main-plus-auxiliary term, not an auxiliary-free recipe.
The final denominator is the behavior selected by
Koh's `normalize_loss` option. This is Koh's Joint CBM with the approved
ResNet-50 encoder substitution. It is not the model Koh separately called
`Standard`, which has no concept loss.

**Official minimal_cbm MCBM used here**

```text
image x -> ResNet-50 -> 26 internal scalars h
                              |-> one learned 1 -> 3 -> 1 concept head per j
                              |      q_j(h_j) = raw concept logit z_j
                              |      sigmoid(z_j) = concept probability
                              `-> learned 26 -> 256 -> 50 species MLP
```

During MCBM training only, the implementation feeds
`h_tilde = h + epsilon`, `epsilon ~ Normal(0,I)`, to both readers. Evaluation
uses `h` without sampled noise. The pinned implementation minimizes

`L_MCBM = L_species + beta * L_concept + gamma * L_rep`, with `beta=1`,

`L_rep = 0.2 * sum_j mean_i[(h_ij - (6*c_ij-3))^2]`.

Thus `c_ij=0` gives target `-3`, and `c_ij=1` gives target `+3`. This is the
repository's concrete mean-squared-error form of the MCBM paper's variational
regularizer. The paper derives a KL penalty and fixes binary prototypes at
`-lambda/+lambda` with `lambda=3`.

Primary sources: [Koh et al. (2020)](https://proceedings.mlr.press/v119/koh20a.html)
and [Almudévar et al. (2026)](https://arxiv.org/abs/2506.04877). The implemented
equations are additionally checked against the pinned `train.py`, `mcbm.py`,
`cbm.py`, and `vanilla.py` source files before this report is built.

| Symbol | Meaning |
|---|---|
| `x_i` | image `i` |
| `y_i` | species label for image `i` |
| `c_ij` | processed binary label for exact concept `j` |
| `h_ij` | MCBM encoder's internal scalar slot for concept `j`; the MCBM species MLP reads the complete vector `h_i` |
| `q_j` | learned `1 -> 3 -> 1` concept head for exact concept `j` |
| `z_ij=q_j(h_ij)` | post-head raw concept logit; the primary grounding score |
| `p_ij=sigmoid(z_ij)` | bounded probability, used only for thresholded performance |
| `c_hat_ij=1[z_ij>0]` | thresholded concept prediction |
| `gamma` | weight on MCBM's representation-compression loss |

The minimal-CBM source code calls the internal tensor `z`, but this report calls
it `h` because it is not yet the post-head concept logit. Every grounding figure
uses the post-head raw score `z_ij`. Ordinary accuracy and recall measure label
agreement; they do not reveal which image pixels produced the score.

Gamma pushes each MCBM internal slot toward its binary-label target. It does
**not** tell the encoder which pixels to use. A species/body shortcut can
predict the right label and be compressed neatly to `-3/+3`. Compression is
therefore evidence that the new loss acted, not evidence that grounding improved.

**Why add h and a separate reader q_j?** MCBM regularizes an internal code h_j,
then learns how that code predicts the named binary answer. Koh instead passes
the concept logit itself to its species reader. These are distinct designs,
not interchangeable variable names. The MCBM paper motivates minimizing
conditional information I(H_j;X|C_j): how much an internal slot can still tell
us about the image once the concept label is already known. For example, among
tail_4-positive images, h=3 for every image leaves no magnitude difference to
identify species; h=2 for one species and4 for another leaves such a clue.
Its Gaussian variational penalty leads to the implemented squared distance
from label-conditioned prototypes. This finite objective and a finite decoder
test do not certify that every possible species clue has disappeared.

The pinned implementation uses the fixed mapping c→6c−3 for its target, not a
learned image-dependent target. The learned q_j maps h_j to z_j. Its separate
species MLP reads all h slots jointly. The paper's internal concept corrections
and information diagnostics test useful representation properties, but do not
replace our physical-image swap test of where the encoder got its evidence.

### Important: neither `h` nor raw `z` is clipped to `[-3,+3]`

The values `-3` and `+3` are **targets in a squared-error penalty**, not hard
bounds. An internal slot `h_ij` may still be smaller than `-3` or larger than
`+3`, especially when gamma is small. The learned `1 -> 3 -> 1` concept head
then maps `h_ij` to the final raw logit `z_ij`; that output is also unbounded.
Only `p_ij=sigmoid(z_ij)` lies between zero and one.

The older exploratory FunnyBird MCBM notebooks used `z` for a different
intermediate quantity and sometimes compared a CBM sigmoid probability with an
MCBM raw score. Their useful questionâ€”separating the pre-swap preference from
the change caused by the swapâ€”is restored below using the current checkpoints
and four verified raw logits. Their raw numerical scales are not reused.

Because separately trained heads can use different raw-logit scales, raw
magnitudes are interpreted primarily **within a model**. Cross-model claims rely
most strongly on predicates, fractions, exact-value ranks, and the fraction of
the model's own starting deficit that the swap closes. A larger raw-logit change
in one model is not automatically a stronger cross-model effect.

There are several baseline differences: MCBM has nonlinear concept heads, a
nonlinear species head, Gaussian training noise, different concept-loss
weighting, and an independently optimized checkpoint. Consequently:

- standard CBM versus MCBM `gamma=0` tests the training-noise/optimization
  baseline because the minimality loss has zero weight;
- MCBM `gamma=0` versus positive gamma tests the added minimality pressure.

An observed Koh-versus-MCBM-gamma-zero difference cannot be credited to
minimality. Only gamma-zero-to-positive-gamma changes within MCBM isolate the
added regularizer.

For a controlled replacement from source value `s` to donor value `d`:

- `m_orig=z_donor,orig-z_source,orig` is the donor-minus-source margin before replacement;
- `m_cf=z_donor,cf-z_source,cf` is the same margin after replacement;
- `response_delta=m_cf-m_orig` is movement caused by the changed image;
- controlled backwash is `response_delta>0 and m_cf<0`.

Example: `m_orig=-20` and `m_cf=-5` gives `response_delta=+15`. The donor pixels
moved the comparison 15 raw-logit units toward the donor, but the old source
still finishes 5 units higher. That is a controlled backwash event.
"""), code("setup", r"""
import os, re, glob, json, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display
CURATED=Path(os.environ["CURATED_DATA"])
REPO=Path.cwd() if (Path.cwd()/"analysis").is_dir() else Path.cwd().parent
KOH_MODEL_ROOT=CURATED/"koh_joint_resnet_accelerated_converged_v1"/"funnybirds"/"standard"/"seed1"
KOH_SWAP_ROOT=CURATED/"swap_koh_joint_resnet_accelerated_converged_v1_seed1"
MCBM_FIXED_ROOT=CURATED/"swap_fixed_v2_attempt2"
VISIBILITY_ROOT=CURATED/"funnybird_visibility_correction_v1"
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
md("population", r"""
## Dataset, population, and causal capability

FunnyBird has 50 species, 26 exact concept values, and five named parts:
`tail`, `wing`, `beak`, `foot`, and `eye`. The accepted renderer changes one
part while holding body, pose, camera, and background fixed. That makes
`response_delta` and `m_cf` causal same-image measurements of the inserted
pixels. Visibility, value support, and source-species residuals remain proposed
contributors unless independently manipulated.

All MCBM gamma comparisons use epoch 100 and the accepted MCBM fixed-render
root `swap_fixed_v2_attempt2`. Whenever Koh Standard is shown, it comes from
the accepted converged Koh root
`swap_koh_joint_resnet_accelerated_converged_v1_seed1`, never from the
`minimal_cbm` repository's different CBM class. Figure 1 verifies that the two
roots identify the same replacement pixels before combining them.

| Item | Value used here | Why it matters |
|---|---:|---|
| species | 50 | unchanged source-species/body appearance is a possible contextual signal |
| named parts | `tail`, `wing`, `beak`, `foot`, `eye` | exactly the same five interventions as notebook 02 |
| exact concepts | 26 part values | exact-value confusion can be separated from coarse part identity |
| ordinary held-out population | 5,000 images per available seed | used for health, compression, decoding, and recall diagnostics |
| fixed-render population | 5,000 directed replacements per gamma at seed 1 | used for causal gamma comparisons |

Notebook 02 already displayed and accepted the semantic renderer preflight for
all five parts. Figure 1 below does not replace that visual inspection: it proves
that every gamma uses those same accepted counterfactual render IDs and byte
hashes. The invalid black-render cache and uncalibrated deletion/patch methods
are not loaded anywhere in this report.
""")]

cells += [md("f1", r"""
## 1 · What data, checkpoints, renders, gammas, and seeds are actually compared?

**Notebook 02 connection.** Notebook 02 first established model health and the
renderer intervention. MCBM adds a gamma sweep, so this report must additionally
prove that every gamma sees the same counterfactual pixels.

**Question.** Is every gamma evaluated on the same validated pixels, and how many
independent seeds support each result?

**Variables and prediction.** A valid row must contain 5,000 unique directed
replacement IDs, all five parts, both directions, all 50 source and donor
species, finite raw logits, and the same render-ID-to-byte-hash mapping as every
other gamma. The stored `margin`, `margin_orig`, and `response_delta` must agree
with values recomputed from the four raw logits.

**Method and exclusions.** The runner first executes the repository's complete
fixed-render validator. This cell then independently checks schema, finiteness,
algebra, identities, checkpoint existence, and file hashes. A non-finite or
unfinished checkpoint is not silently counted as a seed.

### Figure 1 · Are the gamma comparisons mechanically matched?

**How to read the figure.** Each row is one gamma/seed CSV. `rows` is the number
of directed swaps; `render_ids` is the number of unique counterfactual images;
`parts`, `directions`, and the species columns describe coverage. `csv_sha256` and
`checkpoint_sha256` identify the exact inputs. `render_ids` is the number of
unique counterfactual images. `max_algebra_error` is the largest disagreement
between saved and recomputed margins; values near zero are expected. Example:
5,000 rows, five parts, and two directions means 500 swaps per part and direction.
This is an input audit, not a model result.
"""), code("f1", r"""
FIXED=MCBM_FIXED_ROOT
if not FIXED.exists(): raise FileNotFoundError(f"validated fixed-render directory missing: {FIXED}")

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

KOH_CSV=KOH_SWAP_ROOT/"funnybirds-cbm-s1.csv"
if not KOH_CSV.exists():
    raise FileNotFoundError(f"accepted converged Koh swap CSV missing: {KOH_CSV}")
CB=pd.read_csv(KOH_CSV).assign(seed=1,source_csv=KOH_CSV.name)
required_identity={"render_id","image_cf_sha256","image_orig_sha256","part"}
if required_identity-set(CB.columns):
    raise RuntimeError(f"Koh CSV lacks identity columns {sorted(required_identity-set(CB.columns))}")
if CB.render_id.duplicated().any():
    raise RuntimeError("accepted Koh CSV has duplicate render IDs")
reference_render_map={str(r.render_id):(str(r.image_cf_sha256),str(r.image_orig_sha256),str(r.part))
                      for r in CB.itertuples()}
reference_name=KOH_CSV.name

rows=[]; file_meta=[]
for fp in sorted(FIXED.glob("funnybirds-mcbm-g*-s*.csv")):
    m=re.fullmatch(r"funnybirds-mcbm-g([0-9p]+)-s(\d+)\.csv",fp.name)
    if not m: continue
    d=pd.read_csv(fp); g=float(m.group(1).replace("p",".")); seed=int(m.group(2))
    required={"part","direction","z_new","z_old","z_new_orig","z_old_orig",
              "margin","margin_orig","response_delta","sid_src","sid_donor",
              "render_id","image_cf_sha256","image_orig_sha256","var_src","var_donor"}
    missing=required-set(d.columns)
    if missing: raise RuntimeError(f"{fp.name} schema missing {sorted(missing)}")
    numeric=["z_new","z_old","z_new_orig","z_old_orig","margin","margin_orig","response_delta"]
    if not np.isfinite(d[numeric].to_numpy(float)).all():
        raise RuntimeError(f"{fp.name} contains non-finite grounding values")
    m_orig=d.z_new_orig-d.z_old_orig; m_cf=d.z_new-d.z_old; delta=m_cf-m_orig
    algebra=max(float(np.max(np.abs(d.margin-m_cf))),
                float(np.max(np.abs(d.margin_orig-m_orig))),
                float(np.max(np.abs(d.response_delta-delta))))
    if algebra>1e-6: raise RuntimeError(f"{fp.name} stored/recomputed margin mismatch: {algebra}")
    if set(d.part)!=set(ORDER) or set(d.direction)!={"fwd","bwd"}:
        raise RuntimeError(f"{fp.name} has wrong part or direction population")
    if d.render_id.duplicated().any(): raise RuntimeError(f"{fp.name} has duplicate render IDs")
    render_map={str(r.render_id):(str(r.image_cf_sha256),str(r.image_orig_sha256),str(r.part))
                for r in d.itertuples()}
    if render_map!=reference_render_map:
        raise RuntimeError(f"{fp.name} render IDs/bytes/parts differ from accepted {reference_name}")
    tag=m.group(1); ck=REPO/"external/minimal_cbm/results"/f"funnybirds-mcbm-g{tag}"/str(seed)/"models/epoch_100.pt"
    if not ck.exists(): raise FileNotFoundError(f"matching checkpoint missing: {ck}")
    d["gamma"]=g; d["seed"]=seed; d["source_csv"]=fp.name; rows.append(d)
    file_meta.append(dict(gamma=g,seed=seed,csv=fp.name,rows=len(d),
        render_ids=d.render_id.nunique(),parts=d.part.nunique(),directions=d.direction.nunique(),
        source_species=d.sid_src.nunique(),donor_species=d.sid_donor.nunique(),
        csv_sha256=sha256(fp)[:16],checkpoint=str(ck),checkpoint_sha256=sha256(ck)[:16],
        max_algebra_error=algebra))
if not rows: raise FileNotFoundError("no validated standard-MCBM fixed-render CSVs")
SW=pd.concat(rows,ignore_index=True)
SW["m_orig"]=SW.z_new_orig-SW.z_old_orig
SW["m_cf"]=SW.z_new-SW.z_old
SW["response_delta"]=SW.m_cf-SW.m_orig
SW["backwash"]=(SW.response_delta>0)&(SW.m_cf<0)
CB["m_orig"]=CB.z_new_orig-CB.z_old_orig
CB["m_cf"]=CB.z_new-CB.z_old
CB["response_delta"]=CB.m_cf-CB.m_orig
CB["backwash"]=(CB.response_delta>0)&(CB.m_cf<0)

VISIBILITY_TABLE=VISIBILITY_ROOT/"visibility.csv"
if not VISIBILITY_TABLE.exists():
    raise FileNotFoundError(f"corrected visibility table missing: {VISIBILITY_TABLE}")
visibility=pd.read_csv(VISIBILITY_TABLE)
if visibility.duplicated(["render_id","part"]).any():
    raise RuntimeError("corrected visibility key is not unique")
for name,frame in [("Koh",CB),("MCBM",SW)]:
    before=len(frame)
    merged=frame.merge(
        visibility[["render_id","part","legacy_single_instance_pixels",
                    "corrected_all_instance_pixels","added_second_instance_pixels"]],
        on=["render_id","part"],how="left",validate="many_to_one")
    if len(merged)!=before or merged.corrected_all_instance_pixels.isna().any():
        raise RuntimeError(f"corrected visibility does not cover every {name} swap row")
    merged["pixel_count_cf_legacy_single_instance"]=merged["pixel_count_cf"]
    merged["pixel_count_cf"]=merged.corrected_all_instance_pixels.astype(int)
    if name=="Koh": CB=merged
    else: SW=merged
inv=pd.DataFrame(file_meta).sort_values(["gamma","seed"])
if set(inv.gamma)!=set(GAMMAS): raise RuntimeError(f"expected gamma set {GAMMAS}; got {sorted(inv.gamma.unique())}")
display(inv)
print("fixed render root:",FIXED)
print("accepted Koh comparison:",KOH_CSV)
print("corrected visibility:",VISIBILITY_TABLE)
""", "Figure 1. Inventory of validated fixed-render MCBM comparisons by gamma and seed."), review(1)]

cells += [md("f2", r"""
## 2 · Did gamma compress the intended internal slots without breaking prediction?

**Notebook 02 connection.** Notebook 02 checked whether standard-CBM concept
outputs were usable. This section repeats that guard and adds the MCBM-specific
question: did gamma actually enforce the representation penalty?

**Question.** Does increasing gamma move internal slots toward their label
targets and remove within-label variation without breaking ordinary prediction?

**Variables and prediction.** `target RMSE` is the root mean squared distance from each saved
internal slot `h_ij` to its `+3/-3` label target. `within-label spread` is the
median across concepts and labels of `Q95(h)-Q05(h)`. Lower means stronger
compression. Species accuracy and concept balanced accuracy are health checks.
Gamma should lower the first two quantities. A grounding claim is
interpretable only if species/concept health remains usable. These panels may use
all available checkpoints; dots are independent seeds and lines connect only
gamma means.

**Method and exclusions.** Replay every finite epoch-100 prediction/checkpoint
pair. Recompute post-head logits from saved `h`, verify the replayed probabilities,
and exclude unfinished or non-finite artifacts rather than counting them as seeds.

### Figure 2 · Compression and ordinary prediction health across gamma

**How to read the figure.** The x-axis in every panel is gamma. Orange dots are
independently trained seeds; the black point and line are the mean at each gamma.
Panel A is target RMSE in `h` units and Panel B is within-label `h` spread; lower
means stronger compression. Panel C is species accuracy and Panel D is concept
balanced accuracy; higher means healthier prediction. A fall from RMSE 20 to 1
means the slot is much closer to ±3. It does not mean the slot used the right
pixels.
"""), code("f2", r"""
import torch
sys.path.insert(0,str(REPO/"data/funnybirds"))
import funnybirds_concepts as fbc
FB_ROOT=Path(os.environ.get("FUNNYBIRDS_ROOT",CURATED/"FunnyBirds"))
FB_PARTS=fbc.load_parts(FB_ROOT); CONCEPT_NAMES=fbc.concept_names(FB_PARTS); SPANS=fbc.group_slices(FB_PARTS)
CONCEPT_PART={name:part for part,(lo,hi) in SPANS.items() for name in CONCEPT_NAMES[lo:hi]}
health=[]; health_exact=[]; excluded_health=[]; HEALTH_DATA={}
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
    yp=d["y_preds"].reshape(len(h),-1); ya=float((yp.argmax(-1)==d["y"].reshape(-1)).float().mean())
    health.append(dict(gamma=g,seed=int(sd.name),target_rmse=rmse,within_label_spread=np.median(spreads),species_accuracy=ya,concept_balanced_accuracy=float(((tpr+tnr)/2).mean()),replay_error=err))
    HEALTH_DATA[(g,int(sd.name))]=dict(
      h=h.numpy(),c=c.numpy(),z=logits.numpy(),
      y=np.asarray(d["y"]).reshape(-1).astype(int),
      y_probability=np.asarray(d["y_preds"]).reshape(len(h),-1))
    for j,name in enumerate(CONCEPT_NAMES):
      zj=logits[:,j].numpy(); cj=c[:,j].numpy().astype(int); pj=zj>0
      pos=zj[cj==1]; neg=zj[cj==0]
      health_exact.append(dict(gamma=g,seed=int(sd.name),concept=name,part=CONCEPT_PART[name],
        spread=np.quantile(zj,.95)-np.quantile(zj,.05),
        full_range=np.max(zj)-np.min(zj),
        distinct_finite_scores=np.unique(zj[np.isfinite(zj)]).size,
        label_separation=np.median(pos)-np.median(neg),
        balanced_accuracy=.5*((pj[cj==1]).mean()+(~pj[cj==0]).mean()),
        positive_recall=(pj[cj==1]).mean()))
H=pd.DataFrame(health)
HEXACT=pd.DataFrame(health_exact)
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

cells += [md("f2b", r"""
## 2b · Did any exact concept become unusable while the average stayed high?

**Notebook 02 connection.** This is the all-exact-concept health guard from
notebook 02, repeated separately at every MCBM gamma.

**Question.** Figure 2 averages across 26 concepts. Does that hide a constant or
broken exact output?

**Variables and prediction.** For exact concept `j`, `spread_j=Q95(z)-Q05(z)`,
`label_separation_j=median(z|c=1)-median(z|c=0)`, balanced accuracy gives positive
and negative labels equal weight, and `positive_recall_j=P(z>0|c=1)`. Here
`spread_j` describes the middle 90% of scores. It is not enough to prove that
every score is constant. We therefore also print `full_range=max(z)-min(z)` and
the number of distinct finite scores whenever `spread_j <= 1e-8`. Exact collapse
requires `full_range <= 1e-8`. Higher spread is not inherently better; it only
shows that scores vary.

Balanced accuracy is used because most exact concepts are absent from most
images. It is `(positive recall + negative recall)/2`. Example: if a concept is
positive in only 5% of images, predicting “absent” for every image gives 95%
ordinary accuracy but 50% balanced accuracy: it found none of the positives.

**Method and exclusions.** Use seed 1 for the gamma-aligned panels and print all
26 concepts. Non-finite checkpoints were already excluded in Figure 2.

### Figure 2b · Exact-concept health at every gamma

**How to read the figure.** Rows are exact concepts in the same order in all four panels;
columns are all six gammas. Positive label separation, balanced accuracy above
0.5, and positive recall above 0.5 are the expected health directions. These are
health checks, not evidence that the named pixels produced `z`. Example: balanced
accuracy 0.50 means the exact output gives no better-than-chance balanced binary
decision even if another panel shows high overall average accuracy.
"""), code("f2b", r"""
E=HEXACT[HEXACT.seed==1].copy()
metrics=[("spread","raw-z spread"),("label_separation","positive - negative median z"),
         ("balanced_accuracy","balanced accuracy"),("positive_recall","positive recall")]
fig,axes=plt.subplots(1,4,figsize=(18,max(7,.25*len(CONCEPT_NAMES))),sharey=True)
for ax,(metric,title) in zip(axes,metrics):
  T=E.pivot(index="concept",columns="gamma",values=metric).reindex(index=CONCEPT_NAMES,columns=GAMMAS)
  heat(ax,T,title,metric,0 if metric in ["spread","balanced_accuracy","positive_recall"] else None,
       1 if metric in ["balanced_accuracy","positive_recall"] else None,
       "viridis" if metric=="spread" else "coolwarm")
  ax.set_yticklabels(CONCEPT_NAMES,fontsize=7)
plt.tight_layout(); display(E.round(3))
central_zero=E[E.spread.le(1e-8)][
  ["gamma","concept","spread","full_range","distinct_finite_scores",
   "balanced_accuracy","positive_recall"]]
print("gamma/concept cells with zero central-90% spread:")
display(central_zero)
exact_collapsed=E.full_range.le(1e-8)
print("exact full-range-collapsed gamma/concept outputs:",int(exact_collapsed.sum()))
""", "Figure 2b. Exact-concept raw-z spread, label separation, balanced accuracy, and positive recall for all six MCBM gammas."), review("2b")]

cells += [md("f2c", r"""
## 2c · Where does MCBM compression occur, and how does the learned concept head transform it?

**Notebook 02 connection.** Standard CBM has no ±3 representation target, so
this is an MCBM-specific mechanism test rather than a duplicated grounding plot.

**Question.** Does gamma compress every part similarly, and does the learned
concept head `q_j` compensate by amplifying small changes in the compressed
internal slot?

**Variables and prediction.** For each part and gamma, compute: (A) target RMSE
`sqrt(E[(h-(6c-3))^2])`; (B) within-label spread `median(Q95(h)-Q05(h))`; (C)
the mean absolute local head slope `E[|dz/dh|]`; and (D) the fraction of held-out
rows on a locally flat head branch, `P(|dz/dh|<=10^-4)`. The slope is estimated
by a centered finite difference of the saved learned head. If gamma merely
shrinks `h` but the head compensates, Panels A-B should fall while Panel C
stays large or rises. If tail loses head sensitivity, its Panel-C value should
fall and/or its Panel-D flat fraction should rise relative to other parts.

**Method and exclusions.** Use the same held-out seed-1 predictions and finite
checkpoints accepted in Figures 2-2b. Perturb every scalar `h_ij` by `±0.001`
and replay the exact saved concept heads. This measures local head behavior on
ordinary held-out images; it does not recover the unrecorded counterfactual
change in `h` and therefore cannot replace `response_delta`.

### Figure 2c · Per-part compression and concept-head sensitivity

**How to read the figure.** Rows are gamma and columns are the five parts in the
same order used throughout notebooks 02 and 03. Lower values in Panels A-B mean
stronger compression. In Panel C, mean `|dz/dh|=2` means that a local change of
`0.5` in `h` changes `z` by about `1` on average. Panel D is the fraction of
held-out image-concept rows for which the learned head is locally flat; larger
is less locally responsive. The printed table also separates positive,
negative, and flat slopes. This matters because a median slope of zero can hide
a responsive minority on a piecewise-linear ReLU head. These panels explain
where a score scale can change; they do not show which image pixels changed `h`.
"""), code("f2c", r"""
eps=1e-3
mechanism=[]
for g,tag in [(0,"g0"),(.1,"g0p1"),(.3,"g0p3"),(1,"g1"),(3,"g3"),(5,"g5")]:
    d=HEALTH_DATA[(g,1)]
    h=torch.as_tensor(d["h"],dtype=torch.float32)
    c=torch.as_tensor(d["c"],dtype=torch.float32)
    ck=REPO/"external/minimal_cbm/results"/f"funnybirds-mcbm-{tag}"/"1"/"models/epoch_100.pt"
    slope=((concept_logits_from_saved_latent(h+eps,ck,c.shape[1])-
            concept_logits_from_saved_latent(h-eps,ck,c.shape[1]))/(2*eps)).numpy()
    hn=h.numpy(); cn=c.numpy(); target=6*cn-3
    for part in ORDER:
        lo,hi=SPANS[part]; part_spreads=[]
        for j in range(lo,hi):
            for lab in [0,1]:
                q=hn[cn[:,j]==lab,j]
                if len(q)>5: part_spreads.append(np.quantile(q,.95)-np.quantile(q,.05))
        local=slope[:,lo:hi].reshape(-1)
        slope_tol=1e-4
        active=np.abs(local)>slope_tol
        mechanism.append(dict(
            gamma=g,part=part,
            target_rmse=np.sqrt(np.mean((hn[:,lo:hi]-target[:,lo:hi])**2)),
            within_label_h_spread=np.median(part_spreads),
            mean_abs_dz_dh=np.mean(np.abs(local)),
            active_median_abs_dz_dh=(np.median(np.abs(local[active])) if active.any() else 0.0),
            positive_slope_fraction=np.mean(local>slope_tol),
            negative_slope_fraction=np.mean(local < -slope_tol),
            flat_slope_fraction=np.mean(~active)))
MECHANISM=pd.DataFrame(mechanism)
fig,ax=plt.subplots(1,4,figsize=(18,4))
spec=[("target_rmse","A. Distance from ±3 target","h RMSE",0,None,"viridis"),
      ("within_label_h_spread","B. Remaining within-label h variation","Q95-Q05 in h",0,None,"viridis"),
      ("mean_abs_dz_dh","C. Mean learned-head local sensitivity","mean |dz/dh|",0,None,"viridis"),
      ("flat_slope_fraction","D. Locally flat learned-head rows","fraction |dz/dh| <= 1e-4",0,1,"magma_r")]
for a,(metric,title,label,vmin,vmax,cmap) in zip(ax,spec):
    T=MECHANISM.pivot(index="gamma",columns="part",values=metric).reindex(index=GAMMAS,columns=ORDER)
    heat(a,T,title,label,vmin,vmax,cmap)
plt.tight_layout(); display(MECHANISM.round(4))
""", "Figure 2c. Per-part MCBM target compression, within-label internal variation, mean learned-head sensitivity, and locally flat-row fraction."), review("2c")]

cells += [md("f2d", r"""
## 2d · Does the one collapsed tail output create the tail gamma result?

**Notebook 02 connection.** Notebook 02 required exact-output health before
interpreting a part-level swap average. Figure 2b found one exactly constant
output: `tail_7` at gamma zero. This sensitivity analysis prevents that one
broken output from silently determining the MCBM conclusion.

**Question.** Does the tail gamma pattern remain after removing every controlled
swap whose source or donor tail value is 7?

**Variables and prediction.** For the complete tail population and the matched
population excluding value 7, report mean `response_delta`, median final margin
`m_cf`, controlled-backwash rate `P(response_delta>0 and m_cf<0)`, and exact
donor-value recognition. If the collapsed output created the result, removing
value 7 should strongly reduce or reverse the gamma trend. If both lines retain
the same ordering, the tail result is broader than that output.

**Method and exclusions.** Apply the same exclusion to every gamma, even though
only gamma zero has the exact collapse, so every line uses the same set of tail
value pairs. No images, thresholds, or model outputs are changed.

### Figure 2d · Tail gamma results with and without value 7

**How to read the figure.** The blue line includes all tail swaps; the orange
line excludes swaps with source value 7 or donor value 7. Each point is the
seed-1 mean or median over the printed number of fixed-render rows. In Panels A
and B, larger is better. In Panel C, lower controlled backwash is better. In
Panel D, larger exact donor recognition is better. Example: if both Panel-C
lines rise after gamma zero, value 7 cannot be the sole cause of worsening.
"""), code("f2d", r"""
tail=SW[SW.part.eq("tail")].copy()
tail_rows=[]
for g in GAMMAS:
    d=tail[tail.gamma.eq(g)]
    for population,q in [
        ("all tail swaps",d),
        ("exclude source/donor value 7",d[(d.var_src.ne(7)) & (d.var_donor.ne(7))])]:
        cols=sorted([c for c in q if c.startswith("z_cf_tail_")],
                    key=lambda x:int(x.rsplit("_",1)[1]))
        donor=q.var_donor.astype(int).to_numpy()
        pred=q[cols].to_numpy().argmax(1)
        valid=(donor>=0)&(donor<len(cols))
        tail_rows.append(dict(
            gamma=g,population=population,n=len(q),
            mean_response_delta=q.response_delta.mean(),
            median_final_margin=q.m_cf.median(),
            controlled_backwash_rate=q.backwash.mean(),
            exact_donor_recognition=float((pred[valid]==donor[valid]).mean())))
TAIL7_SENSITIVITY=pd.DataFrame(tail_rows)
fig,ax=plt.subplots(1,4,figsize=(16,3.6))
metrics=[
  ("mean_response_delta","A. Donorward movement","mean response_delta"),
  ("median_final_margin","B. Final donor-minus-source margin","median m_cf"),
  ("controlled_backwash_rate","C. Controlled backwash","fraction of swaps"),
  ("exact_donor_recognition","D. Exact donor-value recognition","fraction correct")]
for a,(metric,title,ylabel) in zip(ax,metrics):
    for population,color in [("all tail swaps","#4c78a8"),
                             ("exclude source/donor value 7","#f58518")]:
        q=TAIL7_SENSITIVITY[TAIL7_SENSITIVITY.population.eq(population)]
        a.plot(q.gamma,q[metric],marker="o",label=population,color=color)
    a.set(title=title,xlabel="gamma",ylabel=ylabel)
    a.set_xticks(GAMMAS)
    if metric in {"controlled_backwash_rate","exact_donor_recognition"}:
        a.set_ylim(0,1)
    a.axhline(0,color="black",lw=.7,alpha=.5)
ax[0].legend(frameon=False,fontsize=8)
plt.tight_layout(); display(TAIL7_SENSITIVITY.round(4))
""", "Figure 2d. Tail response, final margin, controlled-backwash rate, and exact donor-value recognition before and after excluding every value-7 swap."), review("2d")]

cells += [md("f11-new", r"""
## 11 · Is recognition of the same positive concept species-dependent?

**Notebook 02 connection.** This restores the authoritative FunnyBird recall
diagnostic as supporting evidence. It does not replace notebook 02's controlled swap.

**Question.** For the same exact positive concept, does recognition differ
between species after positive and negative sample counts are matched?

**Variables and prediction.** The authoritative `fb_recallv2` method has two stages. First, for an exact
concept, it pairs two species only when each contains at least ten positive and
ten negative rows; positive and negative sample counts are matched between the
species. Only if this produces no pairs does it use the all-positive-species
fallback. This notebook prints the selected rule and eligibility coverage.

The current curated validation labels vary within species, so the expected rule
is `matched_positive_negative`, not the fallback. In 300 vectorized bootstrap
runs per pair, `recall gap` is the absolute difference in `P(z>0 | c=1)`.
`balanced-accuracy gap` also uses the matched negatives. `raw-z gap` is the
absolute difference in mean positive `z`, standardized within model/concept.
Zero means equal recognition.

Each heatmap cell is the median across valid concept/species pairs assigned to
that part. Images and pairs are not independent model seeds. Recall is a model-
health/species-dependence diagnostic; the controlled replacement remains the
grounding test.

**Method.** Use 300 vectorized bootstrap draws per eligible species pair, print
the chosen pairing rule and coverage, and keep all models on the same prediction
population. Bootstrap pairs are not independent trained seeds.

### Figure 11 · Matched-species recall, balanced accuracy, and raw-logit gaps

**How to read the figure.** Rows are the six MCBM gammas;
columns are parts. Panel A is the absolute positive-recall difference, Panel B
the absolute balanced-accuracy difference, and Panel C the positive raw-logit
difference measured in within-concept standard deviations. Zero means equal
recognition across the paired species. Example: raw-`z` gap `0.15` means the two
species' mean positive scores differ by 0.15 within-concept standard deviations,
even if both stay above the `z=0` threshold and recall barely changes.
"""), code("f11-new", r"""
from itertools import combinations
rec=[]; coverage=[]; B_RECALL=300
for model,d in MODEL_DATA.items():
 gamma=float(model.split("=")[1]); y=d["y"]; c=d["c"].astype(int); z=d["z"]
 for j,name in enumerate(CONCEPT_NAMES):
  zj=z[:,j]; zstd=(zj-zj.mean())/(zj.std()+1e-12); stats=[]
  for sp in np.unique(y):
   ix=y==sp; n=int(ix.sum()); npos=int(c[ix,j].sum()); stats.append((int(sp),n,npos,n-npos,npos/n))
  eligible=[s for s,n,np_,nn,p in stats if np_>=10 and nn>=10]
  rule="matched_positive_negative"
  pairs=list(combinations(eligible,2))[:200]
  if not pairs:
   eligible=[s for s,n,np_,nn,p in stats if np_>=3 and p>=.9]
   rule="all_positive_fallback"; pairs=list(combinations(eligible,2))[:200]
  coverage.append(dict(model=model,concept=name,part=CONCEPT_PART[name],pairing_rule=rule,
                       eligible_species=len(eligible),pairs=len(pairs),max_species_prevalence=max(s[-1] for s in stats)))
  for pair_index,(a,b) in enumerate(pairs):
   Apos=np.where((y==a)&(c[:,j]==1))[0]; Bpos=np.where((y==b)&(c[:,j]==1))[0]
   Aneg=np.where((y==a)&(c[:,j]==0))[0]; Bneg=np.where((y==b)&(c[:,j]==0))[0]
   mpos=min(len(Apos),len(Bpos)); mneg=min(len(Aneg),len(Bneg))
   rng=np.random.default_rng(20260806+j*1000+pair_index)
   ap=Apos[rng.integers(len(Apos),size=(B_RECALL,mpos))]; bp=Bpos[rng.integers(len(Bpos),size=(B_RECALL,mpos))]
   recA=(zj[ap]>0).mean(1); recB=(zj[bp]>0).mean(1); recall_gaps=np.abs(recA-recB)
   raw_gaps=np.abs(zstd[ap].mean(1)-zstd[bp].mean(1))
   if rule=="matched_positive_negative":
    an=Aneg[rng.integers(len(Aneg),size=(B_RECALL,mneg))]; bn=Bneg[rng.integers(len(Bneg),size=(B_RECALL,mneg))]
    baA=.5*(recA+(zj[an]<=0).mean(1)); baB=.5*(recB+(zj[bn]<=0).mean(1)); ba_gap=float(np.abs(baA-baB).mean())
   else: ba_gap=np.nan
   rec.append(dict(model=model,gamma=gamma,seed=1,concept=name,part=CONCEPT_PART[name],species_a=a,species_b=b,
      pairing_rule=rule,n_positive=mpos,n_negative=mneg,recall_gap=float(recall_gaps.mean()),
      recall_gap_ci_low=float(np.quantile(recall_gaps,.025)),recall_gap_ci_high=float(np.quantile(recall_gaps,.975)),
      balanced_accuracy_gap=ba_gap,standardized_raw_z_gap=float(raw_gaps.mean())))
RECALL=pd.DataFrame(rec)
COVERAGE=pd.DataFrame(coverage)
display(COVERAGE.groupby(["model","pairing_rule"]).agg(concepts=("concept","nunique"),eligible_species_median=("eligible_species","median"),pairs=("pairs","sum"),maximum_prevalence=("max_species_prevalence","max")).round(3))
if RECALL.empty: raise RuntimeError("authoritative two-stage recall pairing produced no pairs; inspect displayed coverage")
model_order=[f"g={g:g}" for g in GAMMAS]
Rg=RECALL.groupby(["model","part"])[["recall_gap","balanced_accuracy_gap","standardized_raw_z_gap"]].median()
R1=Rg.recall_gap.unstack().reindex(index=model_order,columns=ORDER)
RB=Rg.balanced_accuracy_gap.unstack().reindex(index=model_order,columns=ORDER)
R2=Rg.standardized_raw_z_gap.unstack().reindex(index=model_order,columns=ORDER)
fig,ax=plt.subplots(1,3,figsize=(18,4))
heat(ax[0],R1,"Median matched-species positive-recall gap","absolute recall difference",0,1,"magma")
heat(ax[1],RB,"Median matched-species balanced-accuracy gap","absolute BA difference",0,1,"magma")
heat(ax[2],R2,"Median matched-species standardized raw-z gap","within-concept SD units",0,None,"viridis")
plt.tight_layout(); display(RECALL.groupby(["model","part","pairing_rule"]).agg(pairs=("recall_gap","size"),median_recall_gap=("recall_gap","median"),median_balanced_accuracy_gap=("balanced_accuracy_gap","median"),median_raw_z_gap=("standardized_raw_z_gap","median")).round(3))
""", "Figure 11. Two-stage matched-species recall, balanced-accuracy, and standardized raw-logit gaps for every MCBM gamma."), pending_review("11")]

# Assemble the main story from the loss-specific sections and every Standard
# construction. Historical prose stays in git, not in the rendered main report.
from build_mcbm_parity import assemble
cells = assemble(cells, md, code)
nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.10"}},"nbformat":4,"nbformat_minor":5}
if __name__ == "__main__":
    verify_implementation_contract()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preserve-outputs",
        action="store_true",
        help="replace report Markdown while retaining outputs from matching executed code cells",
    )
    args = parser.parse_args()
    for generated_cell in cells:
        if generated_cell["cell_type"] == "code":
            compile("".join(generated_cell["source"]), generated_cell["id"], "exec")
    if args.preserve_outputs and OUT.exists():
        old = json.loads(OUT.read_text(encoding="utf-8"))
        old_code = [c for c in old.get("cells", []) if c.get("cell_type") == "code"]
        new_code = [c for c in nb["cells"] if c.get("cell_type") == "code"]
        # A changed setup cell can invalidate every later figure. Matching an
        # image caption is not evidence that the computation stayed unchanged.
        unchanged = [c["source"] for c in old_code] == [c["source"] for c in new_code]
        matched = 0
        if unchanged:
            for c, previous in zip(new_code, old_code):
                c["outputs"] = previous.get("outputs", [])
                c["execution_count"] = previous.get("execution_count")
                matched += 1
        else:
            print("Code changed: previous outputs were not carried into the new report")
        print(f"preserved outputs for {matched} matching code cells")
    OUT.write_text(json.dumps(nb,indent=1,ensure_ascii=False),encoding="utf-8")
    print(f"wrote {OUT} with {len(cells)} cells")
