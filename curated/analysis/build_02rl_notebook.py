#!/usr/bin/env python3
"""Build the CBM-only FunnyBird RLv2 causal follow-up notebook.

This intentionally does not import or summarize MCBM results.  Notebook 03 and
03rl retain the standard-MCBM and MCBM-RLv2 questions respectively.
"""
from __future__ import annotations

import base64
import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
CURATED = HERE.parent
NOTEBOOKS = CURATED / "notebooks"
SOURCE_02 = NOTEBOOKS / "02_funnybirds_cbm.ipynb"
OUT = NOTEBOOKS / "02rl_funnybirds_cbm_relabeled.ipynb"


def cell_id(tag: str) -> str:
    return f"{tag}-{hashlib.sha1(tag.encode()).hexdigest()[:12]}"


def markdown(tag: str, text: str, attachments=None) -> dict:
    cell = {
        "cell_type": "markdown",
        "id": cell_id(tag),
        "metadata": {},
        "source": text.strip() + "\n",
    }
    if attachments:
        cell["attachments"] = attachments
    return cell


def code(tag: str, text: str, alt: str | None = None) -> dict:
    source = text.strip() + "\n"
    if alt:
        source = f"# ALT: {alt}\n" + source
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id(tag),
        # Nbconvert does not propagate the source comment into the generated
        # image tag.  The HTML postprocessor deliberately reads this metadata
        # and fails closed when an executed image has no description.
        "metadata": {"alt": alt} if alt else {},
        "outputs": [],
        "source": source,
    }


def question(tag: str, number: str, title: str, variables: str, prediction: str, method: str,
             how: str) -> dict:
    return markdown(tag, f"""
## {number} · {title}

**Question.** {title}

**Variables and prediction.** {variables} {prediction}

**Method.** {method}

**Numerical record.** The code cell prints the complete table used to draw the
figure, including the per-part or per-value denominator. The picture is a visual
summary of that table, not a separate calculation. Standard and RLv2 always use
the same row identities, part order, definitions, bins, and axes unless the
section explicitly says otherwise.

### Figure {number} · {title}

**How to read the figure.** {how}
""")


REVIEW_RESULTS = {
    "1": {
        "literal": "Use the executed exact-concept and per-part intervention tables.",
        "alternative": "Intervention size alone does not show that the trained model improved, and the exact tail variants differ substantially (0.111 to 0.391 conflict).",
        "test": "Figure 2 checks every permitted training difference; Figures 3–6 then check model health, identical renders, and behavior.",
        "conclusion": "ACCEPTED FOR identifying the manipulated supervision. The causal behavioral effect is not established by this figure alone.",
    },
    "2": {
        "literal": "Use the executed Koh-manifest and record-parity table.",
        "alternative": "A matched input audit cannot exclude training randomness or a failed checkpoint.",
        "test": "Figures 3–4 check both trained models and prove that their evaluation pixels are identical.",
        "conclusion": "ACCEPTED FOR matched seed-1 training/configuration parity. Seed replication remains separate.",
    },
    "3": {
        "literal": "Use the executed accepted-Koh health tables.",
        "alternative": "The relabeling can still change raw-logit scale or individual variants even though neither model collapsed.",
        "test": "Figures 5–11 use sign-based controlled events, paired component scores, and exact-value recognition on identical renders.",
        "conclusion": "ACCEPTED FOR model health. The small task-accuracy difference does not explain the part-specific controlled pattern by itself.",
    },
    "4": {
        "literal": "Use the executed fixed-render identity table.",
        "alternative": "Renderer validity does not guarantee that relabeling improves grounding.",
        "test": "Figure 5 evaluates both models on these identical pixels.",
        "conclusion": "ACCEPTED FOR fixed-render evaluation parity and the limited one-part replacement claim.",
    },
    "5": {
        "literal": "Use the executed matched-margin and event-rate tables.",
        "alternative": "A lower aggregate rate could hide many newly introduced cases or reciprocal-direction cancellation.",
        "test": "Figure 6 counts every transition; Figures 9–10 separate direction and inserted-part pixel count.",
        "conclusion": "ACCEPTED FOR a seed-1 reduction in the controlled event for tail, beak, and eye. It is negligible for foot and contrary for wing.",
    },
    "6": {
        "literal": "Use the executed resolved/remaining/introduced/never table.",
        "alternative": "The standard-to-RLv2 contrast is one training seed; small net changes may reflect retraining variability or shared-encoder spillover.",
        "test": "Figures 7–11 test the score mechanism and robustness within seed 1; matched seeds 2–3 are still required for reproducibility.",
        "conclusion": "RLv2 resolves a real subset rather than merely shifting an average, but it neither removes all candidates nor avoids new ones.",
    },
    "7": {
        "literal": "Use the executed paired source/donor component table.",
        "alternative": "Raw-logit magnitudes from separately trained models can also reflect scale recalibration, and `response_delta` includes an RLv2-shifted original margin.",
        "test": "Figure 8 checks within-model donorward response and Figure 11 checks sign-free exact donor-value recognition.",
        "conclusion": "ACCEPTED FOR the predicted donor-up/source-down mechanism in tail and directionally in beak/eye, not as a universal RLv2 mechanism.",
    },
    "8": {
        "literal": "Use the executed original-to-swap response table.",
        "alternative": "Because RLv2 changes the original-image margin as well as the counterfactual margin, `response_delta` need not track the primary final-preference change.",
        "test": "Use the paired final margin and event transition as primary endpoints; retain `response_delta` only to prove that inserted pixels are still detected.",
        "conclusion": "RLv2 does not work by making swaps more detectable. It mainly changes the final donor-versus-source preference, especially for tail.",
    },
    "9": {
        "literal": "Use the executed per-part, per-direction table.",
        "alternative": "Different exact variants within a direction could still respond unequally.",
        "test": "Figures 10–11 stratify by visible pixel count and exact inserted value.",
        "conclusion": "ACCEPTED FOR ruling out opposite-direction cancellation as the aggregate explanation.",
    },
    "10": {
        "literal": "Use the executed corrected bilateral-visibility table and denominators.",
        "alternative": "Sparse extreme bins and exact-value composition can make individual bin estimates unstable.",
        "test": "Figure 11 compares exact donor values directly; seed-level replication is required for uncertainty.",
        "conclusion": "The tail/beak improvement is not confined to nearly invisible inserted parts. The visible-pixel result is not universal across parts.",
    },
    "11": {
        "literal": "Use the executed exact-value confusion and rate table.",
        "alternative": "Highest-logit recognition ignores the size of the winning margin and can hide value-specific heterogeneity.",
        "test": "Retain the per-value matrices and compare them with final-margin and source-species residual results.",
        "conclusion": "RLv2 improves donor-value identification most for tail and beak, but does not eliminate exact-value difficulty.",
    },
    "12": {
        "literal": "Use the executed source-species residual table.",
        "alternative": "These residuals are descriptive within the same data and may encode unmeasured renderer/body structure or sparse strata rather than a causal species effect.",
        "test": "Figure 14 asks whether source species improves prediction on held-out renders after the same measured controls.",
        "conclusion": "ACCEPTED FOR a remaining observational source-species association, reduced most for tail. It is not accepted as a held-out causal explanation.",
    },
    "13": {
        "literal": "Use the executed matched species-decoding table.",
        "alternative": "This is one diagnostic split, and decodability measures available species information rather than whether a concept uses the wrong pixels.",
        "test": "Grounding conclusions must continue to come from Figures 5–11; multiple training seeds/splits are needed before treating small decoder changes as stable.",
        "conclusion": "VALID DIAGNOSTIC, HETEROGENEOUS CHANGE. Species information remains abundant and is neither necessary nor sufficient for poor grounding.",
    },
    "14": {
        "literal": "Use the executed held-out predictive table.",
        "alternative": "A high-cardinality group-mean estimator can overfit source species, so worse RMSE does not prove that contextual species information is absent.",
        "test": "Treat Figure 12 as descriptive only; a different preregistered held-out species model or more seeds would be required to revive the predictive claim.",
        "conclusion": "ACCEPTED FOR visibility prediction and RLv2 exact-value prediction. VALID TEST, NO SUPPORT for source species as an added held-out predictive block here.",
    },
    "15": {
        "literal": "Use the executed downstream donor-species table.",
        "alternative": "The class head reads the full latent vector and one replaced part may have little class-level leverage even when concept grounding improves.",
        "test": "Compare this class outcome with the raw-z grounding results rather than using it as their substitute.",
        "conclusion": "VALID TEST, NO UNIFORM DOWNSTREAM SUPPORT. Improved grounding does not consistently raise donor-species probability.",
    },
    "16": {
        "literal": "Use the executed aligned synthesis table; do not add unlike quantities.",
        "alternative": "RLv2 changes the complete label vector and retrains a shared encoder, so part-specific improvements cannot be attributed only to that part's own changed labels; one seed cannot separate stable effects from retraining variability.",
        "test": "Rerun matched CBM-RLv2 seeds 2–3 and replay the same fixed renders before making a seed-general claim.",
        "conclusion": "ACCEPTED FOR a provisional seed-1 causal effect of the complete visibility-aware label intervention: it reduces, but does not eliminate, controlled backwash for tail, beak, and eye. Reproducibility is INCOMPLETE.",
    },
}


def pending_review(tag: str, number: str, next_question: str) -> dict:
    result = REVIEW_RESULTS.get(number)
    if result:
        return markdown(tag, f"""
### Review record for Figure {number}

- **Literal observation:** The complete executed table immediately above is the
  authoritative per-part/per-value result. Values are deliberately not hard-coded
  into the builder before the accepted converged Koh artifacts are executed.
- **Strongest alternative explanation:** {result['alternative']}
- **Discriminating test:** {result['test']}
- **Limited conclusion:** {result['conclusion']}
- **Next question:** {next_question}
""")
    return markdown(tag, f"""
### Review record for Figure {number}

- **Literal observation:** `INCOMPLETE UNTIL THE EXECUTED FIGURE IS DISPLAYED IN CHAT.`
- **Strongest alternative explanation:** `INCOMPLETE UNTIL THE FIGURE IS INSPECTED.`
- **Discriminating test:** `INCOMPLETE UNTIL THE OBSERVATION IS FIXED.`
- **Limited conclusion:** `INCOMPLETE; DO NOT INTERPRET FROM THE TABLE ALONE.`
- **Next question:** {next_question}
""")


def source_png(prefix: str) -> str:
    nb = json.loads(SOURCE_02.read_text(encoding="utf-8"))
    matches = [c for c in nb["cells"] if c.get("id", "").startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one executed notebook-02 cell for {prefix}; got {len(matches)}")
    images = []
    for output in matches[0].get("outputs", []):
        raw = output.get("data", {}).get("image/png")
        if raw:
            images.append("".join(raw) if isinstance(raw, list) else raw)
    if not images:
        raise RuntimeError(f"notebook-02 cell {prefix} has no PNG output")
    return images[0]


def motivating_figure(tag: str, title: str, prefix: str, filename: str, explanation: str) -> dict:
    raw = source_png(prefix)
    # Validate rather than blindly copying malformed notebook data.
    base64.b64decode(raw)
    return markdown(tag, f"""
### {title}

{explanation}

![{title}](attachment:{filename})
""", {filename: {"image/png": raw}})


def notebook() -> dict:
    cells = [
        markdown("02rl-title", r"""
# 02rl · FunnyBird CBM: matched visibility-aware relabeling test

**Report question.** Did positive concept labels attached to invisible parts
cause some of the controlled backwash established for the standard FunnyBird
CBM in notebook 02?

**Population.** Accepted converged Standard CBM versus matched converged
CBM-RLv2, seed 1. Both are ResNet-50 Koh-architecture Joint CBMs. This
notebook contains no MCBM and no gamma sweep. `RLv2` means visibility-aware
relabeling, not reinforcement learning.

**Causal scope.** Training/validation images, class labels, architecture,
optimizer, schedule, seed, epoch, and fixed-render evaluation must match. The
only permitted training-record difference is `attribute_label`. If any parity
check fails, later differences are diagnostics rather than causal evidence.
"""),
        markdown("02rl-roadmap", r"""
## What this notebook must prove

Notebook 02 already proved the standard-CBM event. This notebook does not ask
whether backwash exists again. It tests one proposed cause.

| Step | Needed fact | Why it is needed |
|---|---|---|
| 1 | reproduce the accepted standard-CBM motivation | prevents RLv2 from replacing the discovery story |
| 2 | define and count the label changes | identifies the manipulated training signal |
| 3 | prove record, configuration, checkpoint, and render parity | makes the comparison causal with respect to labels |
| 4 | verify both models are usable | excludes collapse or failed training |
| 5 | compare the controlled event on identical swaps | primary causal result |
| 6 | count resolved, remaining, and introduced events | distinguishes improvement from merely moving averages |
| 7 | decompose original donor/source scores, donor rise, source release, total response, and final margin | identifies where the numerical change occurs without calling the starting margin pure context |
| 8 | repeat direction, visibility, exact-value, species, and downstream checks | tests alternative explanations and the remainder |
| 9 | align the standard and RLv2 summaries | states exactly what changed and what did not |

Every result follows: **question → variables/prediction → method → figure →
literal observation → alternative → discriminating test → limited conclusion →
next question**. No result is accepted until its figure is shown in chat.

Part names are outcomes, not mechanisms. Tail has the largest manipulated label
burden in this dataset and therefore carries the strongest preregistered
prediction, but every component is measured for all five parts. Because RLv2
retrains one shared encoder, changes outside tail are possible and informative.
"""),
        markdown("02rl-model", r"""
## The two CBMs and the notation

Both models implement the same computation:

```text
x_i → ResNet-50 encoder → 26 raw concept logits z_i
                              ↓
                    one linear layer Wz_i+b
                              ↓
                       50 species logits
```

Both use Koh Joint's normalized `L_task + 0.01 × L_concept`. The class head
reads the 26 raw concept logits, not hard 0/1 decisions and not a separate MCBM
latent vector. Grounding analyses use raw `z`, not probability `p`.

Neither model has an MCBM gamma penalty or a `±3` representation target. Their
raw logits are unbounded. Notebook 03's soft targets belong only to MCBM.

| Symbol | Meaning |
|---|---|
| `c_standard[i,j]` | original processed 0/1 concept label |
| `v[i,g(j)]` | whether the renderer says the mapped part is visible |
| `c_RLv2[i,j] = c_standard[i,j] × v[i,g(j)]` | visibility-aware concept label |
| `S_orig`, `D_orig` | source and donor concept logits on the original image |
| `S_cf`, `D_cf` | source and donor concept logits after replacement |
| `m_orig = D_orig-S_orig` | donor-minus-source margin before replacement |
| `m_cf = D_cf-S_cf` | final donor-minus-source margin |
| `response_delta = m_cf-m_orig` | donorward movement caused by replacement |

The controlled candidate event remains exactly the notebook-02 definition:

`response_delta > 0 and m_cf < 0`.

RLv2 is predicted to lower `S_cf` most directly. It may raise `D_cf`, but fewer
positive training labels make that less certain. Because RLv2 can also change
`m_orig`, stronger `response_delta` is secondary; the primary endpoint is the
final event on the identical replacement.
"""),
        markdown("02rl-motivation", r"""
## 0 · Accepted standard-CBM evidence that motivated RLv2

The four images below are copied verbatim from the executed notebook 02. They
are not recomputed with the RLv2 model and are not new results.
"""),
        motivating_figure(
            "02rl-motivation-backwash",
            "Notebook-02 Figure 4 · Controlled standard-CBM backwash predicate",
            "fb-f4-", "notebook02_figure4.png",
            "The standard model usually moves toward the inserted donor, but the old source often remains higher. Tail, beak, and eye contain the largest controlled candidate fractions.",
        ),
        motivating_figure(
            "02rl-motivation-visibility",
            "Notebook-02 Figure 6 · Visibility contributes but is insufficient",
            "fb-f6-", "notebook02_figure6.png",
            "Larger inserted masks generally improve the final margin. Clear visibility does not reduce every part's candidate rate to zero, so visibility alone is not a complete explanation.",
        ),
        motivating_figure(
            "02rl-motivation-conflict",
            "Notebook-02 Figure 6b · Positive labels can conflict with visible evidence",
            "fb-f6b-", "notebook02_figure6b.png",
            "The original training data contain positive concept labels whose named part is not visible. This is a measured property of the targets, not yet a causal model effect.",
        ),
        motivating_figure(
            "02rl-motivation-aligned",
            "Notebook-02 Figure 9b · Outcome and proposed contributors in one part order",
            "fb-f9b-", "notebook02_figure9b.png",
            "Tail is high in the controlled outcome, clear-visibility remainder, label conflict, and donor-value error. The four fractions have different denominators and must not be added.",
        ),
        markdown("02rl-preregister", r"""
## Predictions fixed before reading RLv2 behavior

1. Tail should show the largest reduction in candidate events because it has
   the largest training-label conflict.
2. Beak and eye may improve modestly.
3. Foot and wing should change little.
4. The most direct mechanism is a lower removed-source score `S_cf`.
5. The final margin `m_cf=D_cf-S_cf` should move right.
6. `response_delta` may not rise by the same amount because RLv2 intentionally
   changes low-visibility original images too.
7. Exact-value and source-species residuals may remain; RLv2 manipulates labels,
   not the number of variants or the unchanged source body.
"""),
        code("02rl-setup", r'''
import os, re, json, sys, hashlib, subprocess, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display, Markdown

CURATED = Path(os.environ["CURATED_DATA"])
CWD = Path.cwd()
REPO = CWD if (CWD/"analysis").is_dir() else CWD.parent
ORDER = ["tail", "wing", "beak", "foot", "eye"]
COLORS = {"tail":"#6A0DAD", "wing":"#0072B2", "beak":"#E69F00",
          "foot":"#009E73", "eye":"#CC79A7"}
REGIME_COLORS = {"standard":"#0072B2", "RLv2":"#009E73"}
plt.rcParams.update({"figure.dpi":120, "axes.grid":False})
pd.set_option("display.max_rows",250); pd.set_option("display.max_columns",50)

SEED=1
MODEL_BASE=CURATED/"koh_joint_resnet_accelerated_converged_v1"/"funnybirds"
MODEL_ROOTS={"standard":MODEL_BASE/"standard"/"seed1",
             "RLv2":MODEL_BASE/"rlv2"/"seed1"}
SWAP_DIR=CURATED/"swap_koh_joint_resnet_accelerated_converged_v1_seed1"
VISIBILITY_DIR=CURATED/"funnybird_visibility_correction_v1"
for root in [*MODEL_ROOTS.values(),SWAP_DIR]:
    manifest=root/"SUCCESS.json"
    if not manifest.is_file(): raise FileNotFoundError(manifest)
    subprocess.run([sys.executable,str(REPO/"analysis"/"canonical_manifest.py"),
                    "verify","--manifest",str(manifest)],check=True)
subprocess.run([sys.executable,str(REPO/"analysis"/"validate_fixed_swaps.py"),
                "--out",str(SWAP_DIR)],check=True)

STD=pd.read_csv(SWAP_DIR/"funnybirds-cbm-s1.csv")
RL=pd.read_csv(SWAP_DIR/"funnybirds-cbm-rlv2matched-s1.csv")
for frame in [STD,RL]:
    if "response_delta" not in frame:
        frame["response_delta"]=frame.margin-(frame.z_new_orig-frame.z_old_orig)
    frame["m_orig"]=frame.z_new_orig-frame.z_old_orig
    frame["m_cf"]=frame.z_new-frame.z_old
    frame["donor_gain"]=frame.z_new-frame.z_new_orig
    frame["source_decrease"]=frame.z_old_orig-frame.z_old
    if not np.allclose(frame.m_cf,frame.margin):
        raise RuntimeError("stored margin disagrees with z_new-z_old")
    if not np.allclose(frame.m_cf,frame.m_orig+frame.donor_gain+frame.source_decrease):
        raise RuntimeError("starting-margin/response decomposition does not close")
    frame["candidate"]=(frame.response_delta>0)&(frame.margin<0)
visibility=pd.read_csv(VISIBILITY_DIR/"visibility.csv")
if visibility.duplicated(["render_id","part"]).any():
    raise RuntimeError("corrected visibility key is not unique")
for frame in [STD,RL]:
    original_rows=len(frame)
    merged=frame.merge(visibility[["render_id","part","legacy_single_instance_pixels",
                                   "corrected_all_instance_pixels","added_second_instance_pixels"]],
                       on=["render_id","part"],how="left",validate="many_to_one")
    if len(merged)!=original_rows or merged.corrected_all_instance_pixels.isna().any():
        raise RuntimeError("corrected visibility does not cover all paired rows")
    if not np.array_equal(merged.pixel_count_cf.astype(int),
                          merged.legacy_single_instance_pixels.astype(int)):
        raise RuntimeError("legacy visibility does not match accepted CSV")
    frame.drop(frame.index,inplace=True)
    for column in merged.columns: frame[column]=merged[column]
    frame["pixel_count_cf_legacy_single_instance"]=frame.pixel_count_cf
    frame["pixel_count_cf"]=frame.corrected_all_instance_pixels.astype(int)
keys=["render_id"]
Q=STD.merge(RL,on=keys,validate="one_to_one",suffixes=("_standard","_rl"))
same_cols=["image_cf_sha256","image_orig_sha256","part","direction","sid_src","sid_donor",
           "var_src","var_donor","li","pixel_count_cf"]
for col in same_cols:
    a,b=f"{col}_standard",f"{col}_rl"
    if not Q[a].equals(Q[b]): raise RuntimeError(f"fixed rows disagree on {col}")
    Q[col]=Q[a]
Q["candidate_standard"]=(Q.response_delta_standard>0)&(Q.margin_standard<0)
Q["candidate_rl"]=(Q.response_delta_rl>0)&(Q.margin_rl<0)
Q["pair_id"]=Q.apply(lambda r:f"{r['part']}:{min(r.sid_src,r.sid_donor)}-{max(r.sid_src,r.sid_donor)}",axis=1)
Q["transition"]=np.select([
    Q.candidate_standard&~Q.candidate_rl,
    Q.candidate_standard&Q.candidate_rl,
    ~Q.candidate_standard&Q.candidate_rl],
    ["resolved","remaining","introduced"],default="never candidate")

FB_ROOT=Path(os.environ.get("FUNNYBIRDS_ROOT",CURATED/"FunnyBirds"))
sys.path.insert(0,str(REPO/"data"/"funnybirds"))
import funnybirds_concepts as fbc
parts=fbc.load_parts(FB_ROOT); CONCEPT_NAMES=fbc.concept_names(parts); SPANS=fbc.group_slices(parts)
CONCEPT_PART={n:p for p,(lo,hi) in SPANS.items() for n in CONCEPT_NAMES[lo:hi]}
print("validated fixed-render root:",SWAP_DIR)
print("paired rows:",len(Q),"parts:",sorted(Q.part.unique()))
display(Q.groupby("part").agg(
    swaps=("render_id","size"),
    mean_legacy_pixels=("pixel_count_cf_legacy_single_instance_standard","mean"),
    mean_corrected_pixels=("pixel_count_cf","mean"),
    swaps_with_added_second_instance=("added_second_instance_pixels_standard",lambda x:int((x>0).sum()))
).reindex(ORDER).round(2))
''', "Load only the matched standard-CBM and CBM-RLv2 fixed-render rows and fail if any render identity differs."),
        question(
            "02rl-q1", "1", "What training supervision changed?",
            "For every exact concept, count original positive labels and labels changed from 1 to 0 by RLv2. The part-level conflict rate is changed labels divided by original positive labels.",
            "If the proposed cause is correctly targeted, tail should receive the largest intervention, followed far behind by beak and eye.",
            "Compare the matched training pickle records in their original order, allow only `attribute_label` to differ, and show one changed record per part.",
            "Panel A shows the exact-concept conflict fraction. Panel B shows changed record counts by part on a log scale. The example table prints one real changed image/concept record for every part. These are intervention sizes, not behavioral improvements.",
        ),
        code("02rl-f1", r'''
INPUT_BASE=CURATED/"koh_joint_inputs"/"funnybirds"
STD_PKLS=INPUT_BASE/"standard"; RL_PKLS=INPUT_BASE/"rlv2"
std_train=[]; rl_train=[]
for split in ["train","val"]:
    with open(STD_PKLS/f"{split}.pkl","rb") as f: std_train.extend(pickle.load(f))
    with open(RL_PKLS/f"{split}.pkl","rb") as f: rl_train.extend(pickle.load(f))
if len(std_train)!=len(rl_train): raise RuntimeError("matched train+validation lengths differ")

positive=np.zeros(len(CONCEPT_NAMES),dtype=int); changed=np.zeros(len(CONCEPT_NAMES),dtype=int)
examples={}
for idx,(a,b) in enumerate(zip(std_train,rl_train)):
    ca=np.asarray(a["attribute_label"]); cb=np.asarray(b["attribute_label"])
    positive+=(ca==1); changed+=((ca==1)&(cb==0))
    for j in np.flatnonzero((ca==1)&(cb==0)):
        part=CONCEPT_PART[CONCEPT_NAMES[j]]
        if part not in examples:
            examples[part]={"record_index":idx,"image":a.get("image",a.get("img_path")),
                            "concept":CONCEPT_NAMES[j],"standard_label":int(ca[j]),"RLv2_label":int(cb[j])}
EXACT_CONFLICT=pd.DataFrame({"concept":CONCEPT_NAMES,"part":[CONCEPT_PART[n] for n in CONCEPT_NAMES],
                             "n_positive":positive,"n_changed":changed})
EXACT_CONFLICT["conflict_rate"]=EXACT_CONFLICT.n_changed/EXACT_CONFLICT.n_positive.replace(0,np.nan)
PART_CONFLICT=EXACT_CONFLICT.groupby("part").agg(n_positive=("n_positive","sum"),n_changed=("n_changed","sum")).reindex(ORDER)
PART_CONFLICT["conflict_rate"]=PART_CONFLICT.n_changed/PART_CONFLICT.n_positive

fig,axes=plt.subplots(1,2,figsize=(14,6))
d=EXACT_CONFLICT.sort_values(["part","concept"]); y=np.arange(len(d))
axes[0].barh(y,d.conflict_rate,color=d.part.map(COLORS)); axes[0].set_yticks(y,d.concept,fontsize=7)
axes[0].invert_yaxis(); axes[0].set_xlim(0,1); axes[0].set_xlabel("changed / original-positive labels")
axes[0].set_title("A · Exact-concept label intervention")
axes[1].bar(PART_CONFLICT.index,PART_CONFLICT.n_changed,color=[COLORS[p] for p in PART_CONFLICT.index])
axes[1].set_yscale("log"); axes[1].set_ylabel("changed training records (log scale)")
axes[1].set_title("B · Intervention size by part")
for x,(p,row) in enumerate(PART_CONFLICT.iterrows()): axes[1].text(x,row.n_changed*1.15,f"{row.n_changed:,}",ha="center",fontsize=8)
fig.suptitle("Figure 1 · What RLv2 changed in the matched training labels")
plt.tight_layout();plt.show();display(EXACT_CONFLICT.round(3));display(PART_CONFLICT.round(3));display(pd.DataFrame(examples).T)
''', "Exact and part-level FunnyBird training-label changes plus one concrete changed record for each part."),
        pending_review("02rl-r1", "1", "Did anything besides the concept labels differ between training regimes?"),
        question(
            "02rl-q2", "2", "Is RLv2 the only training difference?",
            "Compare the accepted Koh manifests and every non-label pickle-record field. Permitted manifest differences identify the label view, output path, and independently reached stopping checkpoint; within records, only `attribute_label` may differ.",
            "A causal label claim requires identical ordered images, record IDs, class labels, train/validation membership, and all model/training settings.",
            "Fail closed on the first unexpected difference and print explicit train/validation denominators.",
            "The table has one row for configuration parity and one row for each split. Every Boolean must be true. This is an audit table, not a model-performance result.",
        ),
        code("02rl-f2", r'''
def semantic_differences(a,b,path=()):
    if isinstance(a,dict) and isinstance(b,dict):
        out=[]
        for k in sorted(set(a)|set(b)):
            if k not in a or k not in b: out.append(path+(k,))
            else: out.extend(semantic_differences(a[k],b[k],path+(k,)))
        return out
    if isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b): return [path]
        return [q for i,(x,y) in enumerate(zip(a,b)) for q in semantic_differences(x,y,path+(i,))]
    return [] if a==b else [path]

model_metadata={name:json.loads((root/"SUCCESS.json").read_text()).get("metadata",{})
                for name,root in MODEL_ROOTS.items()}
scientific_keys=["framework","backbone","dataset","seed","architecture",
                 "training_protocol","loss","batch_size","optimizer"]
unexpected_metadata_differences={key:(model_metadata["standard"].get(key),model_metadata["RLv2"].get(key))
                                 for key in scientific_keys
                                 if key in model_metadata["standard"] and key in model_metadata["RLv2"]
                                 and model_metadata["standard"][key]!=model_metadata["RLv2"][key]}
if model_metadata["standard"].get("labels")!="standard" or model_metadata["RLv2"].get("labels")!="rlv2":
    raise RuntimeError("accepted manifests do not identify the Standard and RLv2 label views")
if unexpected_metadata_differences:
    raise RuntimeError(f"unexpected manifest metadata differences: {unexpected_metadata_differences}")
def value_equal(x,y):
    try:return bool(np.array_equal(np.asarray(x),np.asarray(y)))
    except:return x==y
def identity(r): return (str(r.get("image",r.get("img_path"))).replace("\\","/"),int(r["class_label"]),r.get("id"))
parity=[]
for split in ["train","val","test"]:
    with open(STD_PKLS/f"{split}.pkl","rb") as f:a=pickle.load(f)
    with open(RL_PKLS/f"{split}.pkl","rb") as f:b=pickle.load(f)
    same_len=len(a)==len(b); same_ids=same_len and [identity(r) for r in a]==[identity(r) for r in b]
    bad=[]
    if same_len:
        for i,(ra,rb) in enumerate(zip(a,b)):
            keys=set(ra)|set(rb)
            diff=[k for k in keys if k!="attribute_label" and not value_equal(ra.get(k),rb.get(k))]
            if diff: bad.append((i,diff));break
    parity.append({"check":split,"n_standard":len(a),"n_RLv2":len(b),"same_ordered_image_class_id":same_ids,
                   "same_all_non_label_fields":same_len and not bad,"first_unexpected_difference":bad[:1]})
PARITY=pd.DataFrame([{"check":"accepted manifest protocol","n_standard":np.nan,"n_RLv2":np.nan,
                      "same_ordered_image_class_id":True,"same_all_non_label_fields":not unexpected_metadata_differences,
                      "first_unexpected_difference":str(unexpected_metadata_differences)}]+parity)
display(PARITY)
if not PARITY[["same_ordered_image_class_id","same_all_non_label_fields"]].to_numpy().all():
    raise RuntimeError("training/configuration parity failed")
print("ACCEPTED FOR matched training/configuration parity")
''', "Fail-closed standard-CBM versus CBM-RLv2 configuration and record-parity table."),
        pending_review("02rl-r2", "2", "Were both accepted converged models usable rather than collapsed or broken?"),
        question(
            "02rl-q3", "3", "Did both trainings produce usable CBMs?",
            "For every exact concept and each regime, calculate raw-z spread, label separation against that regime's saved target, balanced accuracy, and positive recall. Also report species-task accuracy.",
            "Both models must have non-collapsed outputs and above-chance health; otherwise a change in swap behavior could reflect failed training.",
            "Read each accepted Koh final-test export, reconstruct its 500-by-26 raw-logit and label matrices, and verify that sigmoid(raw z) reproduces the exported probabilities.",
            "Rows are exact concepts and columns are the same four health measures as notebook 02. Standard and RLv2 are shown separately. These are health checks, not grounding evidence.",
        ),
        code("02rl-f3", r'''
def balanced_accuracy(y,p):
    y=np.asarray(y).astype(int);p=np.asarray(p).astype(int)
    return np.nanmean([(p[y==1]==1).mean() if (y==1).any() else np.nan,
                       (p[y==0]==0).mean() if (y==0).any() else np.nan])
PRED_DATA={};health=[];health_summary=[]
for regime,root in MODEL_ROOTS.items():
    pred_path=root/"final_test.parquet"
    if not pred_path.is_file(): raise FileNotFoundError(pred_path)
    d=pd.read_parquet(pred_path)
    needed={"image","y_true","y_pred","concept_index","concept_name","z","prob","gt_label"}
    if missing:=needed-set(d): raise RuntimeError(f"{regime} evaluation lacks {sorted(missing)}")
    image_order=d.image.drop_duplicates().tolist()
    z=d.pivot(index="image",columns="concept_index",values="z").reindex(image_order).to_numpy()
    c=d.pivot(index="image",columns="concept_index",values="gt_label").reindex(image_order).to_numpy()
    labels=(d[["image","y_true","y_pred"]].drop_duplicates("image").set_index("image").reindex(image_order))
    y=labels.y_true.to_numpy(int); yp=labels.y_pred.to_numpy(int)
    probabilities=d.pivot(index="image",columns="concept_index",values="prob").reindex(image_order).to_numpy()
    sigmoid_error=float(np.max(np.abs(probabilities-1/(1+np.exp(-z)))))
    if sigmoid_error>1e-5: raise RuntimeError(f"{regime} sigmoid/raw-z mismatch: {sigmoid_error}")
    PRED_DATA[regime]={"z":z,"c":c,"y":y,"y_pred":yp,"pred_path":pred_path}
    health_summary.append({"regime":regime,"images":len(y),
                           "task_accuracy":(yp==y).mean(),"sigmoid_replay_error":sigmoid_error})
    for j,name in enumerate(CONCEPT_NAMES):
        zz=z[:,j];cc=c[:,j].astype(int);pred=(zz>0).astype(int)
        health.append({"regime":regime,"concept":name,"part":CONCEPT_PART[name],
          "spread":np.quantile(zz,.95)-np.quantile(zz,.05),
          "label_separation":np.median(zz[cc==1])-np.median(zz[cc==0]),
          "balanced_accuracy":balanced_accuracy(cc,pred),"positive_recall":pred[cc==1].mean()})
HEALTH=pd.DataFrame(health);display(pd.DataFrame(health_summary).round(5));display(HEALTH.round(3))
fig,axes=plt.subplots(2,4,figsize=(16,12),sharey="row")
metrics=["spread","label_separation","balanced_accuracy","positive_recall"]
for r,regime in enumerate(["standard","RLv2"]):
    d=HEALTH[HEALTH.regime==regime].sort_values(["part","concept"]);y=np.arange(len(d))
    for ax,m in zip(axes[r],metrics):
        ax.scatter(d[m],y,c=d.part.map(COLORS),s=22);ax.set_title(f"{regime}: {m.replace('_',' ')}")
        if m=="label_separation":ax.axvline(0,color="black",lw=.8)
        if m in ["balanced_accuracy","positive_recall"]:ax.axvline(.5,color="gray",ls="--")
    axes[r,0].set_yticks(y,d.concept,fontsize=7);axes[r,0].invert_yaxis()
fig.suptitle("Figure 3 · Exact-concept health for standard CBM and matched CBM-RLv2")
plt.tight_layout(rect=[0,0,1,.98]);plt.show()
''', "Standard and RLv2 exact-concept raw-score spread, label separation, balanced accuracy, and positive recall."),
        pending_review("02rl-r3", "3", "Were the two models evaluated on exactly the same valid rendered images?"),
        question(
            "02rl-q4", "4", "Are the fixed-render comparisons pixel-identical?",
            "Check one-to-one render IDs, original hashes, counterfactual hashes, part, direction, source/donor identities, exact values, and visible-pixel counts.",
            "Every identity must match before a score difference is attributed to training labels.",
            "Display the accepted renderer preflight and print the full paired-image audit.",
            "The image shows original, replacement, deletion, and part maps for all parts. The table reports 5,000 paired rows and Boolean identity checks. This is an intervention audit, not a behavioral result.",
        ),
        code("02rl-f4", r'''
audit=pd.DataFrame([{"paired_rows":len(Q),"unique_render_ids":Q.render_id.nunique(),
 "unique_counterfactual_hashes":Q.image_cf_sha256_standard.nunique(),
 "all_counterfactual_hashes_match":(Q.image_cf_sha256_standard==Q.image_cf_sha256_rl).all(),
 "all_original_hashes_match":(Q.image_orig_sha256_standard==Q.image_orig_sha256_rl).all(),
 "parts":",".join(sorted(Q.part.unique())),"directions":",".join(sorted(Q.direction.unique()))}])
display(audit)
if len(Q)!=5000 or not audit[["all_counterfactual_hashes_match","all_original_hashes_match"]].to_numpy().all():
    raise RuntimeError("fixed-render parity failed")
preflight=SWAP_DIR/"renderer_preflight"/"renderer_semantic_preflight.png"
if not preflight.exists():raise FileNotFoundError(preflight)
fig,ax=plt.subplots(figsize=(15,4));ax.imshow(Image.open(preflight).convert("RGB"));ax.axis("off")
ax.set_title("Figure 4 · Accepted renderer semantic preflight shared by standard and RLv2")
plt.tight_layout();plt.show()
''', "Validated all-part FunnyBird renderer preflight plus identical-render audit for the standard/RLv2 pair."),
        pending_review("02rl-r4", "4", "Did changing only the labels reduce the controlled candidate event?"),
        question(
            "02rl-q5", "5", "Did RLv2 reduce controlled backwash on identical swaps?",
            "For each part and regime, plot final margin `m_cf` and the rate satisfying both `response_delta>0` and `m_cf<0`.",
            "Tail should improve most; beak and eye may improve modestly; foot and wing should remain similar.",
            "Use the exact same 1,000 swaps per part and pair rows by render ID and image hash.",
            "Panel A shows final-margin distributions; values right of zero favor the donor. Panel B shows the complete candidate rate. A lower RLv2 bar means fewer identical swaps responded to donor pixels yet still preferred the removed source.",
        ),
        code("02rl-f5", r'''
fig,axes=plt.subplots(1,2,figsize=(15,5))
positions=[];values=[];colors=[];ticks=[];labels=[];pos=1
for part in ORDER:
    for regime,suffix in [("standard","standard"),("RLv2","rl")]:
        positions.append(pos);values.append(Q.loc[Q.part==part,f"margin_{suffix}"]);colors.append(REGIME_COLORS[regime]);pos+=1
    ticks.append(pos-1.5);labels.append(part);pos+=.5
bp=axes[0].boxplot(values,positions=positions,widths=.75,showfliers=False,whis=(5,95),patch_artist=True)
for b,c in zip(bp["boxes"],colors):b.set_facecolor(c);b.set_alpha(.6)
axes[0].axhline(0,color="black",lw=.8);axes[0].set_xticks(ticks,labels);axes[0].set_ylabel("final margin m_cf")
axes[0].set_title("A · Final donor-minus-source margin")
rates=[]
for part in ORDER:
    d=Q[Q.part==part];rates.append({"part":part,"standard":d.candidate_standard.mean(),"RLv2":d.candidate_rl.mean(),"n":len(d)})
PRIMARY=pd.DataFrame(rates).set_index("part").reindex(ORDER);x=np.arange(len(ORDER));w=.36
axes[1].bar(x-w/2,PRIMARY.standard,w,color=REGIME_COLORS["standard"],label="standard")
axes[1].bar(x+w/2,PRIMARY.RLv2,w,color=REGIME_COLORS["RLv2"],label="RLv2")
axes[1].set_xticks(x,ORDER);axes[1].set_ylim(0,1);axes[1].set_ylabel("fraction: response_delta>0 and m_cf<0")
axes[1].set_title("B · Controlled candidate rate");axes[1].legend()
fig.suptitle("Figure 5 · Primary matched CBM-RLv2 result")
plt.tight_layout();plt.show();display(PRIMARY.round(3))
''', "Paired standard-CBM and CBM-RLv2 final margins and controlled candidate-event rates for every part."),
        pending_review("02rl-r5", "5", "Which standard candidate events were resolved, remained, or were newly introduced?"),
        question(
            "02rl-q6", "6", "Which individual swap outcomes changed?",
            "Classify every paired row as resolved, remaining, introduced, or never a candidate using the unchanged candidate predicate.",
            "A useful intervention should resolve more standard events than it introduces, especially for tail.",
            "Count transitions on identical render IDs and report both all-row rates and the conditional resolution rate among standard candidates.",
            "Each bar partitions all swaps for a part into four mutually exclusive states. `Resolved` means standard was a candidate and RLv2 was not; `introduced` is the reverse. The table prints the exact denominators.",
        ),
        code("02rl-f6", r'''
cats=["resolved","remaining","introduced","never candidate"]
TRANS=(Q.groupby(["part","transition"]).size().unstack(fill_value=0).reindex(index=ORDER,columns=cats,fill_value=0))
TRANS_RATE=TRANS.div(TRANS.sum(1),axis=0)
TRANS["standard_candidates"]=TRANS.resolved+TRANS.remaining
TRANS["resolution_rate_among_standard_candidates"]=TRANS.resolved/TRANS.standard_candidates.replace(0,np.nan)
TRANS["net_candidate_change"]=TRANS.introduced-TRANS.resolved
fig,ax=plt.subplots(figsize=(10,5));left=np.zeros(len(ORDER));palette={"resolved":"#009E73","remaining":"#D55E00","introduced":"#CC79A7","never candidate":"#BBBBBB"}
for c in cats:
    ax.barh(ORDER,TRANS_RATE[c],left=left,label=c,color=palette[c]);left+=TRANS_RATE[c].to_numpy()
ax.set_xlim(0,1);ax.set_xlabel("fraction of identical swaps");ax.set_title("Figure 6 · Standard-to-RLv2 event transitions")
ax.legend(ncol=4,fontsize=8);plt.tight_layout();plt.show();display(TRANS)
''', "Resolved, remaining, introduced, and never-candidate transitions for every identical FunnyBird swap."),
        pending_review("02rl-r6", "6", "Did RLv2 work by suppressing the removed source, raising the donor, or both?"),
        question(
            "02rl-q7", "7", "Which raw concept score changed?",
            "For every paired swap compute `ΔD_cf=D_cf_RLv2-D_cf_standard`, `ΔS_cf=S_cf_RLv2-S_cf_standard`, and `Δm_cf=m_cf_RLv2-m_cf_standard`.",
            "The proposed mechanism predicts a negative `ΔS_cf` and positive `Δm_cf`, strongest for tail. `ΔD_cf` is less certain.",
            "Average paired changes by part and retain the row-level distributions in the table.",
            "Negative removed-source change means RLv2 turned off the concept that was removed. Positive donor change means it strengthened the inserted concept. Positive margin change means the donor gained relative to the source.",
        ),
        code("02rl-f7", r'''
MECH_ROWS=[]
for part,d in Q.groupby("part"):
    MECH_ROWS.append({"part":part,"delta_donor_cf":(d.z_new_rl-d.z_new_standard).mean(),
      "delta_source_cf":(d.z_old_rl-d.z_old_standard).mean(),
      "delta_final_margin":(d.margin_rl-d.margin_standard).mean(),
      "delta_response":(d.response_delta_rl-d.response_delta_standard).mean()})
MECH=pd.DataFrame(MECH_ROWS).set_index("part").reindex(ORDER);x=np.arange(len(ORDER));w=.22
fig,ax=plt.subplots(figsize=(11,5));ax.bar(x-1.5*w,MECH.delta_donor_cf,w,label="Δ donor score D_cf",color="#E69F00")
ax.bar(x-.5*w,MECH.delta_source_cf,w,label="Δ removed-source score S_cf",color="#CC79A7")
ax.bar(x+.5*w,MECH.delta_final_margin,w,label="Δ final margin",color="#009E73")
ax.bar(x+1.5*w,MECH.delta_response,w,label="Δ response_delta",color="#0072B2")
ax.axhline(0,color="black",lw=.8);ax.set_xticks(x,ORDER);ax.set_ylabel("RLv2 − standard raw-logit units")
ax.set_title("Figure 7 · Paired score mechanism");ax.legend(ncol=2);plt.tight_layout();plt.show();display(MECH.round(3))
''', "Paired changes in inserted-donor score, removed-source score, final margin, and response delta."),
        pending_review("02rl-r7", "7", "Did RLv2 change the complete original-to-swap response, or mainly the final preference?"),
        question(
            "02rl-q7b", "7b", "Which part of the starting preference and swap response did RLv2 change?",
            "For each regime, decompose `m_cf = m_orig + donor_gain + source_decrease`, while also showing `D_orig` and `S_orig` separately. All quantities are post-head raw logits `z`, not internal `h`.",
            "If visibility-inconsistent labels help preserve a removed concept, RLv2 should increase `source_decrease` and improve `m_cf` most where the manipulated conflict was largest. It may also change `D_orig`, `S_orig`, and donor gain because the complete shared encoder is retrained.",
            "Use the same 5,000 paired render IDs for both models, average each component by part, verify the decomposition row by row, and print `RLv2-standard` changes.",
            "Rows are standard and RLv2; columns are the same five parts. Panels A-B show the two original-image scores, Panel C their starting margin, Panels D-E the two swap-response components, Panel F their sum, and Panel G the final margin. Blue means negative and red means positive within the shared raw-logit scale. The table below the figure prints the exact means and paired regime differences.",
        ),
        code("02rl-f7b", r'''
component_specs=[
    ("D_orig","Original absent-donor score"),
    ("S_orig","Original present-source score"),
    ("m_orig","Before swap: donor minus source"),
    ("donor_gain","Inserted donor score rises"),
    ("source_decrease","Removed source score falls"),
    ("response_delta","Total movement toward donor"),
    ("m_cf","After swap: donor minus source"),
]
column_map={
    "D_orig":("z_new_orig_standard","z_new_orig_rl"),
    "S_orig":("z_old_orig_standard","z_old_orig_rl"),
    "m_orig":("m_orig_standard","m_orig_rl"),
    "donor_gain":("donor_gain_standard","donor_gain_rl"),
    "source_decrease":("source_decrease_standard","source_decrease_rl"),
    "response_delta":("response_delta_standard","response_delta_rl"),
    "m_cf":("m_cf_standard","m_cf_rl"),
}
component_tables={}
for key,_ in component_specs:
    table=pd.DataFrame(index=["standard","RLv2"],columns=ORDER,dtype=float)
    standard_col,rl_col=column_map[key]
    table.loc["standard"]=Q.groupby("part")[standard_col].mean().reindex(ORDER)
    table.loc["RLv2"]=Q.groupby("part")[rl_col].mean().reindex(ORDER)
    component_tables[key]=table
for suffix in ["standard","rl"]:
    err=np.max(np.abs(Q[f"m_cf_{suffix}"]-(Q[f"m_orig_{suffix}"]+Q[f"donor_gain_{suffix}"]+Q[f"source_decrease_{suffix}"])))
    if err>1e-8: raise RuntimeError(f"{suffix} decomposition error {err}")
lim=max(float(np.nanmax(np.abs(t.values))) for t in component_tables.values())
fig,axes=plt.subplots(2,4,figsize=(22,8));axes=axes.ravel()
for ax,(key,title) in zip(axes,component_specs):
    table=component_tables[key]
    im=ax.imshow(table.values,aspect="auto",cmap="coolwarm",vmin=-lim,vmax=lim)
    ax.set_xticks(range(len(ORDER)),ORDER,rotation=40,ha="right")
    ax.set_yticks(range(2),table.index);ax.set_title(title,fontsize=10)
    for r in range(2):
        for c in range(len(ORDER)):
            ax.text(c,r,f"{table.iloc[r,c]:.2f}",ha="center",va="center",fontsize=8)
axes[-1].axis("off")
fig.colorbar(im,ax=axes[:-1].tolist(),label="mean raw-logit units",fraction=.02,pad=.02)
fig.suptitle("Figure 7b · Complete standard-to-RLv2 starting-margin and response decomposition")
plt.tight_layout();plt.show()
display(pd.concat(component_tables,names=["component","regime"]).round(3))
paired_changes=pd.DataFrame({key:table.loc["RLv2"]-table.loc["standard"]
                             for key,table in component_tables.items()}).T
display(Markdown("**RLv2 minus standard (positive and negative retain each component's defined direction):**"))
display(paired_changes.round(3))
''', "Matched standard-CBM and RLv2 original donor/source scores, starting margin, donor gain, source decrease, total response, and final margin for every FunnyBird part."),
        pending_review("02rl-r7b", "7b", "Do the sign-based final outcomes confirm the raw-logit decomposition?"),
        markdown("02rl-factor-map", r'''
### What this decomposition can and cannot isolate

| Proposed factor | Component it could affect | Test in this notebook | Causal status |
|---|---|---|---|
| visibility-inconsistent training labels | any component after retraining; predicted most directly to improve removed-source release | matched RLv2 versus standard on identical renders | causal for the complete relabeling package at seed 1; not isolated to one part because the encoder is shared |
| inserted-part visibility/area | mainly donor gain, possibly source release | Figure 10, same rows stratified by exact target pixels | association within validated swaps; area itself was not randomized |
| exact source/donor value | starting margin and both response components | Figure 11 and value-matched residuals | controlled description, not an independent frequency intervention |
| unchanged source species/body | especially removed-source persistence and final margin | Figures 12 and 14 | observational; source species is not independently manipulated |
| retraining randomness | every component | matched seeds 2-3 | currently incomplete |

Figure 7b identifies **where the numerical change occurs**. Only the matched
label intervention identifies a manipulated cause, and even that intervention
retrained the complete shared encoder. It therefore tests whether the label
package contributes to the behavior, not whether each changed tail label acts
only on the tail output.
'''),
        question(
            "02rl-q8", "8", "Did original-to-swap donorward movement change?",
            "Compare standard and RLv2 `response_delta=m_cf-m_orig` distributions and positive-response rates.",
            "RLv2 may strengthen donorward movement, but this is secondary because the original-image margin is also intentionally changed.",
            "Use identical rows and show every part.",
            "The boxes show response_delta; above zero means the replacement moved the comparison toward the donor. The bars show the fraction above zero. This does not by itself say whether the donor finishes above the source.",
        ),
        code("02rl-f8", r'''
fig,axes=plt.subplots(1,2,figsize=(15,5));positions=[];values=[];colors=[];ticks=[];pos=1
for part in ORDER:
    for regime,suffix in [("standard","standard"),("RLv2","rl")]:
        positions.append(pos);values.append(Q.loc[Q.part==part,f"response_delta_{suffix}"]);colors.append(REGIME_COLORS[regime]);pos+=1
    ticks.append(pos-1.5);pos+=.5
bp=axes[0].boxplot(values,positions=positions,widths=.75,showfliers=False,whis=(5,95),patch_artist=True)
for b,c in zip(bp["boxes"],colors):b.set_facecolor(c);b.set_alpha(.6)
axes[0].axhline(0,color="black",lw=.8);axes[0].set_xticks(ticks,ORDER);axes[0].set_ylabel("response_delta")
rates=[]
for part,d in Q.groupby("part"):
    rates.append({"part":part,"standard":(d.response_delta_standard>0).mean(),"RLv2":(d.response_delta_rl>0).mean()})
RESPONSE=pd.DataFrame(rates).set_index("part").reindex(ORDER);x=np.arange(len(ORDER));w=.36
axes[1].bar(x-w/2,RESPONSE.standard,w,color=REGIME_COLORS["standard"],label="standard")
axes[1].bar(x+w/2,RESPONSE.RLv2,w,color=REGIME_COLORS["RLv2"],label="RLv2")
axes[1].set_xticks(x,ORDER);axes[1].set_ylim(0,1);axes[1].set_ylabel("fraction response_delta>0");axes[1].legend()
fig.suptitle("Figure 8 · Original-to-swap response under each label regime")
plt.tight_layout();plt.show();display(RESPONSE.round(3))
''', "Standard and RLv2 response-delta distributions and positive donor-response rates."),
        pending_review("02rl-r8", "8", "Does the primary result hold in both replacement directions?"),
        question(
            "02rl-q9", "9", "Could one swap direction create the RLv2 result?",
            "Calculate the unchanged candidate predicate separately for forward and backward replacements under each regime.",
            "A real part-level label effect should not depend entirely on one arbitrary direction.",
            "Keep direction and regime separate and print 500-row denominators.",
            "Circles and squares distinguish standard and RLv2; solid and dashed lines distinguish forward and backward swaps. Lower candidate rates are better.",
        ),
        code("02rl-f9", r'''
rows=[]
for (part,direction),d in Q.groupby(["part","direction"]):
    rows.append({"part":part,"direction":direction,"n":len(d),"standard":d.candidate_standard.mean(),"RLv2":d.candidate_rl.mean()})
DIRECTION=pd.DataFrame(rows);fig,ax=plt.subplots(figsize=(10,5))
for regime,marker in [("standard","o"),("RLv2","s")]:
    for direction,ls in [("fwd","-"),("bwd","--")]:
        d=DIRECTION[DIRECTION.direction==direction].set_index("part").reindex(ORDER)
        ax.plot(ORDER,d[regime],marker=marker,ls=ls,color=REGIME_COLORS[regime],label=f"{regime} {direction}")
ax.set_ylim(0,1);ax.set_ylabel("candidate-event fraction");ax.set_title("Figure 9 · Direction-specific controlled candidate rates")
ax.legend(ncol=2);plt.tight_layout();plt.show();display(DIRECTION.round(3))
''', "Forward and backward controlled candidate-event rates for standard CBM and matched CBM-RLv2."),
        pending_review("02rl-r9", "9", "Is any improvement confined to swaps where the inserted part has little or no visible area?"),
        question(
            "02rl-q10", "10", "How does exact inserted-part visibility change the RLv2 result?",
            "Within the same declared pixel-count bins as notebook 02, compare standard and RLv2 candidate rates.",
            "If RLv2 only changes zero-pixel cases, it did not improve recognition of visible donor evidence. Improvement among clearly visible rows supports a broader label-training effect.",
            "Use exact counterfactual part-map pixel counts and print row denominators for every bin.",
            "Each panel is one part. The x-axis is visible inserted-part pixels and the y-axis is the complete candidate rate. Standard and RLv2 share the same rows in every bin.",
        ),
        code("02rl-f10", r'''
bins=[0,20,50,100,200,500,np.inf];labels=["0–19","20–49","50–99","100–199","200–499","500+"]
W=Q.copy();W["visibility_bin"]=pd.cut(W.pixel_count_cf,bins=bins,labels=labels,right=False)
rows=[]
for (part,v),d in W.groupby(["part","visibility_bin"],observed=True):
    rows.append({"part":part,"visibility_bin":v,"n":len(d),"standard":d.candidate_standard.mean(),"RLv2":d.candidate_rl.mean()})
VIS=pd.DataFrame(rows);fig,axes=plt.subplots(1,5,figsize=(20,4),sharey=True)
for ax,part in zip(axes,ORDER):
    d=VIS[VIS.part==part].set_index("visibility_bin").reindex(labels)
    ax.plot(labels,d.standard,"o-",color=REGIME_COLORS["standard"],label="standard")
    ax.plot(labels,d.RLv2,"s-",color=REGIME_COLORS["RLv2"],label="RLv2")
    ax.set_title(part);ax.tick_params(axis="x",rotation=60);ax.set_ylim(0,1)
axes[0].set_ylabel("candidate-event fraction");axes[-1].legend()
fig.suptitle("Figure 10 · Matched RLv2 effect across exact visibility bins")
plt.tight_layout(rect=[0,0,1,.95]);plt.show();display(VIS.round(3))
''', "Per-part standard/RLv2 candidate rates across exact swapped-part visibility bins."),
        pending_review("02rl-r10", "10", "Did RLv2 improve exact donor-value recognition, or do particular variants remain difficult?"),
        question(
            "02rl-q11", "11", "Did exact inserted-value recognition improve?",
            "For every part and regime, compare the inserted donor value with the highest post-swap raw concept logit.",
            "If RLv2 repairs broad label conflict but not visual-value difficulty, tail recognition may improve while particular rows remain poor.",
            "Display all exact values in two aligned rows: standard and RLv2.",
            "Each heatmap row is the inserted value and each column is the highest-scoring value. A brighter diagonal means better exact donor recognition. The title prints the full diagonal rate.",
        ),
        code("02rl-f11", r'''
def concept_columns(frame,part):
    return sorted([c for c in frame if c.startswith(f"z_cf_{part}_")],key=lambda x:int(x.rsplit("_",1)[1]))
VALUE_ROWS=[];DIAG={}
fig,axes=plt.subplots(2,5,figsize=(18,7))
for r,(regime,frame) in enumerate([("standard",STD),("RLv2",RL)]):
    for c,part in enumerate(ORDER):
        d=frame[frame.part==part];cols=concept_columns(d,part);pred=d[cols].to_numpy().argmax(1);donor=d.var_donor.astype(int).to_numpy()
        n=len(cols);M=np.zeros((n,n))
        for a,b in zip(donor,pred):M[a,b]+=1
        M=M/np.maximum(M.sum(1,keepdims=True),1);DIAG[(regime,part)]=(pred==donor).mean()
        axes[r,c].imshow(M,cmap="magma",vmin=0,vmax=1);axes[r,c].set_title(f"{part} · {regime}\ndiagonal={DIAG[(regime,part)]:.2f}")
        axes[r,c].set_xlabel("highest-scoring value");
        if c==0:axes[r,c].set_ylabel("inserted value")
        for value,g in d.assign(pred=pred).groupby("var_donor"):
            VALUE_ROWS.append({"regime":regime,"part":part,"donor_value":int(value),"n":len(g),
              "recognition_rate":(g.pred==g.var_donor).mean(),"candidate_rate":g.candidate.mean()})
fig.suptitle("Figure 11 · Exact donor-value recognition before and after RLv2")
plt.tight_layout(rect=[0,0,1,.96]);plt.show();VALUE=pd.DataFrame(VALUE_ROWS);display(VALUE.round(3))
''', "All-part exact donor-value confusion matrices for standard CBM and matched CBM-RLv2."),
        pending_review("02rl-r11", "11", "After exact values, direction, and visibility are controlled, does source species still organize the margin?"),
        question(
            "02rl-q12", "12", "Does source species still organize the remaining margin?",
            "For each regime, subtract the mean margin within `(part, source value, donor value, direction, visibility bin)` and average the residual by source species.",
            "If RLv2 removes label conflict but not unchanged-body context, species residual spread should remain even if it becomes smaller for tail.",
            "Require at least five rows per displayed source-species estimate and use identical control strata.",
            "Each panel is a part. Source species are ordered by the standard residual; blue is standard and green is RLv2 for the same species. Zero means that species matches its controlled-stratum average.",
        ),
        code("02rl-f12", r'''
R=Q.copy();R["vis_bin"]=pd.cut(R.pixel_count_cf,[-1,19,49,99,199,499,np.inf],labels=False)
SPECIES_TABLES={};spread=[]
for regime,suffix in [("standard","standard"),("RLv2","rl")]:
    margin=f"margin_{suffix}";d=R.copy();d["expected"]=d.groupby(["part","var_src","var_donor","direction","vis_bin"])[margin].transform("mean")
    d["residual"]=d[margin]-d.expected
    g=d.groupby(["part","sid_src"]).agg(n=("residual","size"),residual=("residual","mean")).reset_index().query("n>=5")
    SPECIES_TABLES[regime]=g
    for part,x in g.groupby("part"):spread.append({"regime":regime,"part":part,"species":len(x),"residual_sd":x.residual.std()})
SPECIES_SPREAD=pd.DataFrame(spread)
fig,axes=plt.subplots(1,5,figsize=(20,4),sharey=True)
for ax,part in zip(axes,ORDER):
    a=SPECIES_TABLES["standard"].query("part==@part").set_index("sid_src").residual.sort_values()
    b=SPECIES_TABLES["RLv2"].query("part==@part").set_index("sid_src").residual.reindex(a.index)
    x=np.arange(len(a));ax.scatter(x,a,s=16,color=REGIME_COLORS["standard"],label="standard")
    ax.scatter(x,b,s=16,color=REGIME_COLORS["RLv2"],label="RLv2");ax.axhline(0,color="black",lw=.8);ax.set_title(part)
axes[0].set_ylabel("mean residual final margin");axes[-1].legend()
fig.suptitle("Figure 12 · Source-species residual after exact value, direction, and visibility controls")
plt.tight_layout(rect=[0,0,1,.94]);plt.show();display(SPECIES_SPREAD.pivot(index="part",columns="regime",values="residual_sd").reindex(ORDER).round(3))
''', "Matched source-species margin residuals after exact value, direction, and visibility controls."),
        pending_review("02rl-r12", "12", "Did RLv2 reduce species information in the learned concept representation?"),
        question(
            "02rl-q13", "13", "Did species decodability change inside the concept vector?",
            "For each regime, train the same held-out diagnostic classifier on known labels `c` and learned raw logits `z`, using the same image split.",
            "If RLv2 reduces contextual coding, raw-z species accuracy may decrease most for tail. This remains a representation diagnostic, not grounding evidence.",
            "Use one fixed stratified 70/30 split and identical logistic-regression settings.",
            "Each panel is one regime. Grey bars use known labels and colored bars use learned raw z. Bar height is diagnostic held-out species accuracy, not the saved CBM's task accuracy.",
        ),
        code("02rl-f13", r'''
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
probe=[]
for regime in ["standard","RLv2"]:
    z=PRED_DATA[regime]["z"];c=PRED_DATA[regime]["c"];y=PRED_DATA[regime]["y"]
    idx=np.arange(len(y));tr,te=train_test_split(idx,test_size=.30,random_state=20260803,stratify=y)
    blocks={"complete":np.arange(z.shape[1])};blocks.update({p:np.arange(lo,hi) for p,(lo,hi) in SPANS.items()})
    for block,cols in blocks.items():
        for source,X in [("known labels c",c),("learned raw z",z)]:
            model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=4000,C=1,random_state=20260803))
            model.fit(X[tr][:,cols],y[tr]);probe.append({"regime":regime,"block":block,"source":source,
              "species_accuracy":accuracy_score(y[te],model.predict(X[te][:,cols])),"dimensions":len(cols)})
PROBE=pd.DataFrame(probe);fig,axes=plt.subplots(1,2,figsize=(14,5),sharey=True)
for ax,regime in zip(axes,["standard","RLv2"]):
    d=PROBE[PROBE.regime==regime];blocks=["complete"]+ORDER;x=np.arange(len(blocks));w=.36
    a=d[d.source=="known labels c"].set_index("block").reindex(blocks);b=d[d.source=="learned raw z"].set_index("block").reindex(blocks)
    ax.bar(x-w/2,a.species_accuracy,w,color="#BBBBBB",label="known labels c")
    ax.bar(x+w/2,b.species_accuracy,w,color=["#333333"]+[COLORS[p] for p in ORDER],label="learned raw z")
    ax.axhline(1/50,color="black",ls="--");ax.set_xticks(x,blocks,rotation=25);ax.set_title(regime);ax.set_ylabel("held-out species accuracy");ax.legend(fontsize=8)
fig.suptitle("Figure 13 · Species decoded from each CBM concept representation")
plt.tight_layout(rect=[0,0,1,.94]);plt.show();display(PROBE.round(3))
''', "Matched diagnostic species decoders using known labels and learned raw logits for standard and RLv2 CBMs."),
        pending_review("02rl-r13", "13", "Do visibility, exact values, or source species predict the remaining margin on held-out render IDs?"),
        question(
            "02rl-q14", "14", "What measured structure predicts each regime's margin on held-out renders?",
            "For standard and RLv2 separately, predict final margin using part, then visibility, exact source/donor values, and source species.",
            "If a block organizes unseen rows, held-out RMSE should decrease when it is added. RLv2 need not make every residual disappear.",
            "Use the same deterministic five-fold render-ID split and training-fold shrunken group means as notebook 02.",
            "The y-axis is held-out final-margin RMSE; lower is better. Blue is standard and green is RLv2. An increase is negative evidence for that proposed organizing block.",
        ),
        code("02rl-f14", r'''
def account(frame,margin_col):
    d=frame.copy();d["vis_bin"]=pd.cut(d.pixel_count_cf,[-1,19,49,99,199,499,np.inf],labels=False)
    d["fold"]=d.render_id.astype(str).map(lambda x:int(hashlib.sha1(x.encode()).hexdigest(),16)%5)
    stages=[("part only",["part"]),("+ visibility",["part","vis_bin"]),("+ exact values",["part","vis_bin","var_src","var_donor"]),("+ source species",["part","vis_bin","var_src","var_donor","sid_src"])]
    out=[]
    for stage,cols in stages:
        pred=pd.Series(index=d.index,dtype=float)
        for fold in range(5):
            tr=d[d.fold!=fold];te=d[d.fold==fold];prior=tr[margin_col].mean();stats=tr.groupby(cols)[margin_col].agg(["mean","count"]).reset_index()
            stats["estimate"]=(stats["mean"]*stats["count"]+prior*10)/(stats["count"]+10)
            pred.loc[te.index]=te[cols].merge(stats[cols+["estimate"]],on=cols,how="left").estimate.fillna(prior).to_numpy()
        out.append({"stage":stage,"rmse":np.sqrt(np.mean((d[margin_col]-pred)**2))})
    return pd.DataFrame(out)
ACCOUNT=[]
for regime,col in [("standard","margin_standard"),("RLv2","margin_rl")]:
    x=account(Q,col);x["regime"]=regime;ACCOUNT.append(x)
ACCOUNT=pd.concat(ACCOUNT,ignore_index=True);fig,ax=plt.subplots(figsize=(9,5))
for regime in ["standard","RLv2"]:
    d=ACCOUNT[ACCOUNT.regime==regime];ax.plot(d.stage,d.rmse,"o-",color=REGIME_COLORS[regime],label=regime)
ax.set_ylabel("held-out RMSE of final margin");ax.tick_params(axis="x",rotation=25);ax.set_title("Figure 14 · Sequential held-out margin accounting")
ax.legend();plt.tight_layout();plt.show();display(ACCOUNT.round(3))
''', "Held-out standard and RLv2 final-margin prediction error after sequentially adding visibility, exact values, and source species."),
        pending_review("02rl-r14", "14", "Did the concept-layer change materially alter donor-species prediction?"),
        question(
            "02rl-q15", "15", "Did RLv2 change the downstream donor-species probability?",
            "Compare the mean donor-species probability after each identical swap.",
            "Better concept ordering may modestly increase donor-species probability, but a large task change is not required because the unchanged body still belongs to the source species.",
            "Use the saved post-swap donor-species probability and every part.",
            "Bars show the mean probability assigned to the donor species after a one-part replacement. This is the one section using class probability rather than raw concept z.",
        ),
        code("02rl-f15", r'''
rows=[]
for part,d in Q.groupby("part"):
    rows.append({"part":part,"standard":d.p_cf_donor_standard.mean(),"RLv2":d.p_cf_donor_rl.mean(),"paired_change":(d.p_cf_donor_rl-d.p_cf_donor_standard).mean()})
DOWNSTREAM=pd.DataFrame(rows).set_index("part").reindex(ORDER);x=np.arange(len(ORDER));w=.36
fig,ax=plt.subplots(figsize=(10,5));ax.bar(x-w/2,DOWNSTREAM.standard,w,color=REGIME_COLORS["standard"],label="standard")
ax.bar(x+w/2,DOWNSTREAM.RLv2,w,color=REGIME_COLORS["RLv2"],label="RLv2");ax.set_xticks(x,ORDER)
ax.set_ylabel("mean donor-species probability after swap");ax.set_title("Figure 15 · Downstream species consequence");ax.legend()
plt.tight_layout();plt.show();display(DOWNSTREAM.round(4))
''', "Standard and RLv2 downstream donor-species probability after identical one-part replacements."),
        pending_review("02rl-r15", "15", "How do the outcome, intervention, visual difficulty, and residual context line up in one final view?"),
        question(
            "02rl-q16", "16", "What did RLv2 resolve, and what remains?",
            "Align candidate rate, clearly-visible candidate rate, original label-conflict burden, donor-value error, and source-species residual spread in the same part order.",
            "A causal label contribution predicts the largest standard-to-RLv2 improvement where label conflict was greatest, while value error or species residuals may remain.",
            "Do not add the panels; they use different units and answer different questions.",
            "Blue and green bars compare standard and RLv2 model outcomes where both exist. The label-conflict panel shows the manipulated training burden: blue is the original conflict and green is the fraction of those original conflicts left positive by RLv2. Higher is worse except that the panels are not commensurate.",
        ),
        code("02rl-f16", r'''
SUMMARY=pd.DataFrame(index=ORDER);SUMMARY.index.name="part"
SUMMARY["candidate_standard"]=Q.groupby("part").candidate_standard.mean().reindex(ORDER)
SUMMARY["candidate_RLv2"]=Q.groupby("part").candidate_rl.mean().reindex(ORDER)
clear=Q[Q.pixel_count_cf>=100]
SUMMARY["clear_standard"]=clear.groupby("part").candidate_standard.mean().reindex(ORDER)
SUMMARY["clear_RLv2"]=clear.groupby("part").candidate_rl.mean().reindex(ORDER)
SUMMARY["label_conflict_standard"]=PART_CONFLICT.conflict_rate.reindex(ORDER)
SUMMARY["label_conflict_left_by_RLv2"]=0.0
SUMMARY["value_error_standard"]=[1-DIAG[("standard",p)] for p in ORDER]
SUMMARY["value_error_RLv2"]=[1-DIAG[("RLv2",p)] for p in ORDER]
sp=SPECIES_SPREAD.pivot(index="part",columns="regime",values="residual_sd").reindex(ORDER)
SUMMARY["species_residual_standard"]=sp["standard"];SUMMARY["species_residual_RLv2"]=sp["RLv2"]
panels=[("candidate","A · CONTROLLED OUTCOME: candidate rate"),("clear","B · CLEAR VISIBILITY: candidate rate"),
        ("label_conflict","C · TRAINING INPUT: label conflict"),("value_error","D · VISUAL DIFFICULTY: donor-value error"),
        ("species_residual","E · CONTEXT: source-species residual SD")]
fig,axes=plt.subplots(1,5,figsize=(20,5),sharey=True);y=np.arange(len(ORDER));h=.36
for ax,(prefix,title) in zip(axes,panels):
    a=SUMMARY[f"{prefix}_standard"]
    b=SUMMARY["label_conflict_left_by_RLv2"] if prefix=="label_conflict" else SUMMARY[f"{prefix}_RLv2"]
    ax.barh(y-h/2,a,h,color=REGIME_COLORS["standard"],label="standard")
    ax.barh(y+h/2,b,h,color=REGIME_COLORS["RLv2"],label="RLv2")
    ax.set_title(title,fontsize=9);ax.set_yticks(y,ORDER);ax.invert_yaxis()
axes[-1].legend();fig.suptitle("Figure 16 · Standard-CBM contributors extended by the matched RLv2 intervention")
plt.tight_layout(rect=[0,0,1,.94]);plt.show();display(SUMMARY.round(3))
''', "Five aligned FunnyBird panels comparing the controlled outcome and proposed contributors before and after matched CBM-RLv2 training."),
        pending_review("02rl-r16", "16", "Which causal claims are now accepted, provisional, unsupported, or still observational?"),
        markdown("02rl-ledger", r"""
## 17 · CBM-RLv2 evidence ledger

This table must be completed only after every important figure above has been
displayed and reviewed in chat.

| Claim | Required figure | Status |
|---|---|---|
| only the visibility-aware concept labels changed | 1–2 | `ACCEPTED FOR matched seed-1 training/configuration parity` |
| both models and fixed renders are valid | 3–4 | `ACCEPTED FOR model health and identical fixed-render evaluation` |
| RLv2 reduced the controlled CBM candidate event | 5–6 | `ACCEPTED FOR tail, beak, and eye at seed 1; negligible foot change; contrary wing result` |
| the change followed the predicted raw-score mechanism | 7–8 | `ACCEPTED FOR tail; directional support for beak/eye; not universal` |
| complete starting-margin and response decomposition | 7b | `INCOMPLETE PENDING EXECUTION AND VISUAL REVIEW` |
| the result survives direction and visibility checks | 9–10 | `ACCEPTED FOR tail/beak and mostly eye within seed 1; sparse extreme bins retained` |
| exact-value difficulty after RLv2 | 11 | `ACCEPTED FOR partial tail/beak/eye improvement; substantial difficulty remains` |
| remaining source-species association | 12 | `ACCEPTED FOR descriptive association only; causal/predictive explanation not accepted` |
| species decodability changed | 13 | `VALID DIAGNOSTIC, HETEROGENEOUS CHANGE; not grounding evidence` |
| proposed blocks predict held-out residuals | 14 | `visibility supported; exact values supported only for RLv2; source species: VALID TEST, NO SUPPORT` |
| downstream class consequence | 15 | `VALID TEST, NO UNIFORM SUPPORT` |
| integrated causal conclusion | 16 | `ACCEPTED FOR provisional seed-1 RLv2 label-package effect; seed reproducibility INCOMPLETE` |

Seed 1 supports the bounded intervention result above because parity passed. It
cannot establish training-seed reproducibility. Missing seeds remain
`INCOMPLETE` and are not silently replaced with row-level bootstrap intervals.
"""),
        markdown("02rl-appendix", r"""
# Methods appendix · what this notebook deliberately excludes

- No MCBM or gamma result appears here.
- The earlier uncalibrated deletion/patch methods are not used as causal
  confirmation.
- Row and source/donor-pair resampling do not replace independent training
  seeds.
- Source-species residuals are observational because the source body/species
  was not independently randomized.
- The existing mixed notebook 03rl remains historical/later MCBM-RLv2 work; it
  is not the source of this CBM-only conclusion.
"""),
        markdown("02rl-provenance-title", r"""
# Provenance appendix

The final table records the Git commit, exact checkpoints, training records,
fixed CSVs, hashes, row counts, seed, and epoch. A stale HTML is not accepted
evidence.
"""),
        code("02rl-provenance", r'''
def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()
commit=subprocess.run(["git","rev-parse","HEAD"],cwd=REPO,capture_output=True,text=True,check=True).stdout.strip()
paths=[("standard fixed CSV",SWAP_DIR/"funnybirds-cbm-s1.csv"),("RLv2 fixed CSV",SWAP_DIR/"funnybirds-cbm-rlv2matched-s1.csv"),
       ("standard prediction",PRED_DATA["standard"]["pred_path"]),("RLv2 prediction",PRED_DATA["RLv2"]["pred_path"]),
       ("standard checkpoint",MODEL_ROOTS["standard"]/"final_model_1.pth"),("RLv2 checkpoint",MODEL_ROOTS["RLv2"]/"final_model_1.pth"),
       ("standard train records",STD_PKLS/"train.pkl"),("RLv2 train records",RL_PKLS/"train.pkl")]
display(pd.DataFrame([{"role":role,"path":str(path),"sha256":sha256_file(path)} for role,path in paths]))
display(pd.DataFrame([{"git_commit":commit,"seed":SEED,"training_state":"accepted converged checkpoints","paired_swaps":len(Q),
 "parts":len(ORDER),"exact_concepts":len(CONCEPT_NAMES),"fixed_render_root":str(SWAP_DIR)}]))
''', "SHA-256 provenance for the matched standard-CBM and CBM-RLv2 records, checkpoints, predictions, and fixed-render CSVs."),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
            "language_info": {"name":"python","version":"3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def install_review_text(target: Path) -> None:
    """Update report prose without deleting executed outputs."""
    current = json.loads(target.read_text(encoding="utf-8"))
    desired = notebook()
    current_ids = [cell.get("id") for cell in current["cells"]]
    desired_ids = [cell.get("id") for cell in desired["cells"]]
    if current_ids != desired_ids:
        raise RuntimeError("refusing prose-only update: notebook cell identities changed")
    for old, new in zip(current["cells"], desired["cells"]):
        if old["cell_type"] == "markdown":
            old["source"] = new["source"]
        elif "".join(old.get("source", [])) != "".join(new.get("source", [])):
            raise RuntimeError(f"refusing prose-only update: code changed in {old.get('id')}")
    target.write_text(json.dumps(current, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"updated reviewed prose without changing outputs in {target}")


def preserve_matching_outputs(target: Path, desired: dict) -> dict:
    """Retain outputs only when a code cell ID and source are unchanged."""
    if not target.exists():
        return desired
    current = json.loads(target.read_text(encoding="utf-8"))
    old_code = {cell.get("id"): cell for cell in current.get("cells", [])
                if cell.get("cell_type") == "code" and cell.get("id")}
    preserved = 0
    for cell in desired["cells"]:
        if cell.get("cell_type") != "code":
            continue
        old = old_code.get(cell.get("id"))
        if old is None or "".join(old.get("source", [])) != "".join(cell.get("source", [])):
            continue
        cell["outputs"] = old.get("outputs", [])
        cell["execution_count"] = old.get("execution_count")
        preserved += 1
    print(f"preserved outputs for {preserved} unchanged code cells")
    return desired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review-text-only",
        action="store_true",
        help="install reviewed Markdown into the executed notebook without rebuilding code outputs",
    )
    parser.add_argument(
        "--preserve-outputs",
        action="store_true",
        help="rebuild structure while retaining outputs from unchanged code cells",
    )
    args = parser.parse_args()
    if args.review_text_only:
        install_review_text(OUT)
    else:
        desired = notebook()
        if args.preserve_outputs:
            desired = preserve_matching_outputs(OUT, desired)
        OUT.write_text(json.dumps(desired, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
