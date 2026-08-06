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

# These records were accepted only after the executed outputs were displayed in
# chat and reviewed figure by figure on 2026-08-06.  Keeping them here makes a
# rebuilt notebook retain the scientific review rather than reverting to a
# generic pending placeholder.
REVIEWS.update({
"2b": """- **Literal observation:** Most exact concepts have non-zero central-90% raw-logit spread, positive label separation, balanced accuracy well above `0.5`, and high positive recall. Four gamma/concept cells have `Q95(z)-Q05(z) <= 1e-8`; the visible example is `tail_7` at gamma `0`, where balanced accuracy is `0.50` and positive recall is `0.00`. At some other gammas a zero central-90% spread coexists with high balanced accuracy, so `Q95-Q05=0` alone must not be described as every score being constant.
- **Alternative explanations:** A rare set of non-central scores can support classification even when the middle 90% is identical.
- **Discriminating test:** Print full range and the number of distinct finite scores for every zero-central-spread cell. Only full range within `1e-8` is called exact collapse.
- **Limited conclusion:** Average model health is strong, but exact-concept health is uneven and `tail_7` at gamma zero is unusable by the threshold metrics. The corrected full-range check is required before counting exact collapses at other gammas.
- **Next question:** With that exact-output caution recorded, does MCBM gamma zero reproduce the standard-CBM controlled pattern?""",
3: """- **Literal observation:** On identical renders, MCBM gamma zero lowers controlled backwash relative to standard CBM for tail (`0.506` versus `0.608`) and beak (`0.415` versus `0.537`), while eye is essentially unchanged and wing/foot remain low. Tail still has a negative median final margin (`-4.40`), whereas wing and foot finish strongly donor-positive.
- **Alternative explanations:** Gamma zero has no minimality penalty, but its architecture and independently optimized checkpoint differ from standard CBM.
- **Discriminating test:** Use gamma zero only as the MCBM baseline; credit minimality only to changes from gamma zero across the same fixed renders.
- **Limited conclusion:** MCBM begins with the same qualitative grounding order as standard CBM, but baseline numerical differences cannot be attributed to minimality.
- **Next question:** As gamma increases, do the inserted donor pixels still move the concept margin donorward?""",
7: """- **Literal observation:** Larger inserted regions generally improve final margins, but they do not remove tail failure. With at least 100 inserted pixels, tail controlled backwash is `0.397` at gamma zero and `0.644-0.787` at positive gammas. Wing and foot remain much lower across the same size bins.
- **Alternative explanations:** Pixel-size bins contain different donor values and species, so a bin difference is not a randomized visibility effect.
- **Discriminating test:** Keep the exact-value and source-species variables in the held-out accounting model in Figure 9b, and separately inspect the visible-only endpoint in Figure 11b.
- **Limited conclusion:** Small/occluded insertions explain some failures, especially at gamma zero, but not the high positive-gamma tail failure among clearly visible insertions.
- **Next question:** Did every gamma receive the same conflicting positive labels for invisible parts?""",
"7b": """- **Literal observation:** The shared training data label a tail concept positive while the rendered tail is absent in `6,711/33,929 = 0.198` positive tail rows. The corresponding rates are `0.010` for beak, `0.007` for eye, `0.001` for foot, and effectively `0` for wing. Exact tail values range from `0.111` to `0.391` conflict.
- **Alternative explanations:** This is a property of the training data shared by all gammas; by itself it cannot explain why the gamma curve changes.
- **Discriminating test:** Compare the fixed conflict ordering with visible-only causal failure and exact-value error in Figure 11b.
- **Limited conclusion:** Label/visibility conflict is a credible contributor to the tail-versus-wing/foot ordering, not a complete explanation of the gamma effect.
- **Next question:** When the donor beats the source, does the model also identify the exact inserted value?""",
8: """- **Literal observation:** Exact donor-value recognition is lowest for tail at every gamma: `0.278` at gamma zero and `0.109-0.202` at positive gammas. Wing stays `0.811-0.881` and foot `0.926-0.991`. Beak improves to about `0.75` at gamma `1-3`, and eye reaches `0.735` at gamma `5`.
- **Alternative explanations:** Exact argmax is stricter than donor-versus-source margin because a third value can win.
- **Discriminating test:** The same argmax procedure gives strong diagonals for wing and foot, so the tail result is not a universal consequence of the metric or compression.
- **Limited conclusion:** Minimality has part-specific effects: it weakens exact tail attribution while improving selected beak and eye settings.
- **Next question:** Are the difficult exact donor values simply rare or drawn from more alternatives?""",
"8b": """- **Literal observation:** Failure varies substantially among exact values with similar species support, and tail remains difficult across both low- and high-support values. Wing and foot have low failure over overlapping support ranges. The number of alternatives is fixed within each part, so the five part families do not provide an independent alternative-count experiment.
- **Alternative explanations:** Species support, exact value identity, and part identity are entangled in this generated dataset.
- **Discriminating test:** Match exact source and donor values first, then ask whether source species still shifts the final margin in Figure 9.
- **Limited conclusion:** Rarity/support alone does not explain the part ordering. Exact value difficulty contributes, but the alternative-count hypothesis is not independently identified here.
- **Next question:** After exact-value matching, does unchanged source species/body context still organize the result?""",
9: """- **Literal observation:** Supported source species retain different mean residual margins after exact source/donor value adjustment at every gamma. Raw residual spread shrinks with gamma, but standardized tail spread does not (`0.425` at gamma zero and `0.494` at gamma five); beak stays about `0.54-0.61`, and eye is often `0.58-0.72`.
- **Alternative explanations:** Source species is bundled with the unchanged body, pose, and other parts, so this is not an independent species manipulation.
- **Discriminating test:** Figure 9b tests whether adding source species lowers prediction error on held-out replacements after visibility and exact values are already included.
- **Limited conclusion:** Source-species/body context explains reproducible margin variation beyond exact value difficulty, but the responsible visual component is not isolated.
- **Next question:** How much held-out variation does each proposed contributor explain, and how much remains?""",
"9b": """- **Literal observation:** For every gamma, held-out standardized RMSE falls slightly after visibility is added, more after exact values are added, and again after source species is added. At gamma zero it falls `0.866 -> 0.844 -> 0.738 -> 0.673`; at gamma five it falls `0.670 -> 0.651 -> 0.561 -> 0.503`. A substantial residual of about `0.50-0.67` standard deviations remains.
- **Alternative explanations:** Sequential credit depends on variable order, and predictive improvement is association rather than causal subtraction.
- **Discriminating test:** Each added block is accepted only because it improves five-fold held-out prediction, not because it improves in-sample fit.
- **Limited conclusion:** Visibility, exact values, and source species/body each account for reproducible variation, but they do not fully explain final margins and are not arithmetically additive causes.
- **Next question:** Is species information present in labels, raw concept logits, and MCBM internal slots?""",
10: """- **Literal observation:** Ground-truth concept labels already decode species well from the complete 26-value vector (`0.787`), establishing structural species information. Standard-CBM tail and wing raw logits/internal features decode species far above their label-only controls. Increasing gamma reduces most extra single-part decoding in raw logits, but the complete learned vector remains near the label baseline. Tail internal slots remain notably species-decodable even at high gamma.
- **Alternative explanations:** Species decoding measures information availability, not whether that information caused a particular swap failure. Part blocks also have different dimensions and structural label baselines.
- **Discriminating test:** Compare every learned source with its same-block label control and require the controlled swap for the grounding claim.
- **Limited conclusion:** Minimality reduces some extra part-level species information but does not erase the species structure of the full representation. Because tail backwash worsens while some decoding falls, `more decodable species information` is not a monotone explanation of the gamma curve.
- **Next question:** Does a matched within-concept diagnostic also show species-dependent recognition?""",
11: """- **Literal observation:** All 26 concepts pass the stated pairing rule, with median `9.5` eligible species and `1,358` species pairs per model. Thresholded recall gaps are small because ordinary recognition is near saturation, but the standardized raw-logit gap is consistently largest or near-largest for tail under MCBM (`0.106-0.178`) and falls for wing/foot at high gamma.
- **Alternative explanations:** Species pairs remain observational and differ in visual context; small recall gaps can hide score shifts far from the zero threshold.
- **Discriminating test:** The raw-logit companion preserves score information while the recall and balanced-accuracy panels show whether the shift crosses the decision boundary.
- **Limited conclusion:** Species-dependent score organization persists most clearly for tail, but recall is supporting model-health/species-dependence evidence, not a replacement for the controlled swap.
- **Next question:** Do the four standard-CBM measurements align with every MCBM gamma?""",
"11b": """- **Literal observation:** Tail is worst or near-worst on all four aligned measurements: all-swap backwash `0.506-0.780`, visible-only backwash `0.397-0.787`, training conflict `0.198`, and exact-value error `0.722-0.891`. Wing and foot remain strong; beak and eye are intermediate and improve at selected higher gammas.
- **Alternative explanations:** Training conflict is fixed across gamma and therefore explains ordering better than the worsening tail gamma curve. The four panels also have different denominators and cannot be added as percentages.
- **Discriminating test:** Require agreement across causal swap response, visible-only restriction, training-data conflict, and exact-value recognition; then use Figure 9b for held-out accounting.
- **Limited conclusion:** The standard-CBM contributors extend coherently to MCBM, especially the tail-versus-wing/foot contrast. They do not fully explain why minimality specifically weakens tail response.
- **Next question:** Does successful concept attribution measurably alter the downstream donor-species prediction?""",
12: """- **Literal observation:** Within each part, mean donor-species probability is near zero in bins with negative final donor-minus-source margin and generally rises in strongly positive-margin bins. The absolute values remain small: tail stays below about `0.08`, while the strongest wing/foot bins reach roughly `0.16-0.17`.
- **Alternative explanations:** Replacing one part should not make the complete bird belong to the donor species, and margin bins are descriptive rather than randomized doses.
- **Discriminating test:** Keep parts separate and show the number of swap rows in every bin, preventing large easy-part groups from producing the trend alone.
- **Limited conclusion:** Better concept attribution is associated with more donor-class support, but the downstream class cost is modest. The controlled concept-grounding failure is the clearer result.
- **Next question:** Which gamma conclusions are replicated across independently trained models?""",
})


def review(n: int | str) -> dict:
    return md(f"review{n}", f"""
    **Figure {n} review.**

    {REVIEWS[n]}
    """)


def pending_review(label: str) -> dict:
    return md(f"review-{label}", f"""
    ### Review record for Figure {label}

    **PENDING EXECUTION AND VISUAL REVIEW.** Display this complete output in chat
    before writing its literal observation. Then record the strongest alternative
    explanation, discriminating test, limited conclusion, and next question.
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
"""), md("roadmap", r"""
## What this notebook must prove, and how it continues notebook 02

Notebook 02 established the standard-CBM event on a controlled replacement:

`response_delta > 0 and m_cf < 0`.

This report does not rediscover or replace that result. It asks whether the
MCBM training change repairs the same event on the same rendered images.

| Step | Needed fact | Output | Why it is needed |
|---|---|---|---|
| 1 | every input, checkpoint, render ID, and hash is valid | 1 | unequal pixels or populations invalidate a gamma comparison |
| 2 | gamma changes the quantity named by the MCBM loss without breaking prediction | 2, 2b | compression must be demonstrated before it explains anything |
| 3 | MCBM gamma 0 is compared fairly with standard CBM | 3 | gamma 0 is the sweep baseline, not evidence for minimality |
| 4 | inserted pixels still cause donorward movement | 4 | a source win is backwash only if donor pixels had an effect |
| 5 | the final donor/source outcome changes or does not change | 5 | this is the primary grounding endpoint |
| 6 | direction and visibility alternatives are tested | 6, 7 | pooling or tiny target parts must not create the result |
| 7 | training conflict and exact-value difficulty are carried forward | 7b, 8, 8b | notebook-02 contributors must not disappear from the MCBM story |
| 8 | exact values and source species are accounted for before residual claims | 9, 10 | plausible associations are not automatically explanations |
| 9 | species information and recall use structural controls | 10, 11 | species leakage is opportunity, not grounding proof |
| 10 | unlike measurements are aligned but never added | 10c, 14 | compression is not credited merely because one number shrank |
| 11 | downstream class cost and independent-seed coverage are explicit | 12, 13 | explanation failure and class harm are different claims |

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
## Standard CBM versus MCBM: exact wiring and loss

Both models use the same evaluation-time computation:

```text
x_i -> image encoder -> h_i = (h_i1, ..., h_iJ)
                         |-> learned 1->3->1 head q_j(h_ij) -> raw logit z_ij
                         |                                          -> p_ij=sigmoid(z_ij)
                         `-> species head reads the complete h_i
```

Standard CBM trains with

`L_CBM = L_species + beta L_concept`.

MCBM adds

`L_rep = sum_j 0.2 mean((h_ij - (6c_ij-3))^2)`

and trains with

`L_MCBM = L_species + beta L_concept + gamma L_rep`.

For a positive label, `6c-3=+3`; for a negative label it is `-3`. Gamma pushes
each internal slot toward its binary-label target. It does not tell the encoder
which pixels to use.

There is one additional baseline difference. During training, standard CBM uses
`var_z=0`, while MCBM uses `var_z=1`; the trainer therefore adds Gaussian noise
to `h` for MCBM. Evaluation is deterministic for both. Consequently:

- standard CBM versus MCBM `gamma=0` tests the training-noise/optimization
  baseline because the minimality loss has zero weight;
- MCBM `gamma=0` versus positive gamma tests the added minimality pressure.

An observed CBM-versus-gamma-zero difference cannot be credited to minimality.
"""), code("setup", r"""
import os, re, glob, json, sys, hashlib
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
"""), md("population", r"""
## Dataset, population, and causal capability

FunnyBird has 50 species, 26 exact concept values, and five named parts:
`tail`, `wing`, `beak`, `foot`, and `eye`. The accepted renderer changes one
part while holding body, pose, camera, and background fixed. That makes
`response_delta` and `m_cf` causal same-image measurements of the inserted
pixels. Visibility, value support, and source-species residuals remain proposed
contributors unless independently manipulated.

All gamma comparisons use epoch 100 and the accepted fixed-render root
`swap_fixed_v2_attempt2`. Figure 1 verifies the actual files rather than relying
on that directory name.
""")]

cells += [md("f1", r"""
## Figure 1 — What data, checkpoints, renders, gammas, and seeds are actually compared?

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

**How to read.** Each row is one gamma/seed CSV. `csv_sha256` and
`checkpoint_sha256` identify the exact inputs. `render_ids` is the number of
unique counterfactual images. This is an input audit, not a model result.
"""), code("f1", r"""
FIXED=CURATED/"swap_fixed_v2_attempt2"
if not FIXED.exists(): raise FileNotFoundError(f"validated fixed-render directory missing: {FIXED}")

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

rows=[]; file_meta=[]; reference_render_map=None; reference_name=None
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
    render_map=dict(zip(d.render_id.astype(str),d.image_cf_sha256.astype(str)))
    if reference_render_map is None: reference_render_map,reference_name=render_map,fp.name
    elif render_map!=reference_render_map:
        raise RuntimeError(f"{fp.name} render IDs/bytes differ from {reference_name}")
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
inv=pd.DataFrame(file_meta).sort_values(["gamma","seed"])
if set(inv.gamma)!=set(GAMMAS): raise RuntimeError(f"expected gamma set {GAMMAS}; got {sorted(inv.gamma.unique())}")
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
    yp=d["y_preds"]; ya=float((yp.argmax(-1)==d["y"]).float().mean())
    health.append(dict(gamma=g,seed=int(sd.name),target_rmse=rmse,within_label_spread=np.median(spreads),species_accuracy=ya,concept_balanced_accuracy=float(((tpr+tnr)/2).mean()),replay_error=err))
    HEALTH_DATA[(g,int(sd.name))]=dict(h=h.numpy(),c=c.numpy(),z=logits.numpy(),y=np.asarray(d["y"]).reshape(-1).astype(int))
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
## Figure 2b — Did any exact concept become unusable while the average stayed high?

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

**Method and exclusions.** Use seed 1 for the gamma-aligned panels and print all
26 concepts. Non-finite checkpoints were already excluded in Figure 2.

**How to read.** Rows are exact concepts in the same order in all four panels;
columns are all six gammas. Positive label separation, balanced accuracy above
0.5, and positive recall above 0.5 are the expected health directions. These are
health checks, not evidence that the named pixels produced `z`.
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

cells += [md("f3", r"""
## Figure 3 — Does MCBM gamma 0 reproduce the standard-CBM discovery?

**Question.** Before crediting minimality, compare standard CBM and MCBM
`gamma=0` on identical fixed renders.

**Variables and prediction.** Compare mean `response_delta`, median final margin
`m_cf`, controlled-backwash rate, and exact donor-value error for every part.
The minimality term is zero at gamma 0. Differences from standard CBM therefore
belong to MCBM's training noise and ordinary optimization, not minimality.

**How to read.** All four panels use the same part order and display both models.
Higher response and margin are better; lower controlled backwash and exact-value
error are better. Raw-logit magnitudes may differ between independently trained
models, so the two fraction panels are the strongest baseline comparison.
"""), code("f3", r"""
cbfs=sorted(FIXED.glob("funnybirds-cbm-s*.csv"))
if not cbfs: raise FileNotFoundError("standard-CBM fixed-render CSV missing")
CB=pd.concat([pd.read_csv(f).assign(seed=int(re.search(r"s(\d+)",f.stem).group(1))) for f in cbfs],ignore_index=True)
CB["m_orig"]=CB.z_new_orig-CB.z_old_orig; CB["m_cf"]=CB.z_new-CB.z_old
CB["response_delta"]=CB.m_cf-CB.m_orig; CB["backwash"]=(CB.response_delta>0)&(CB.m_cf<0)

def exact_error(d):
  out={}
  for p in ORDER:
    cols=sorted([c for c in d if c.startswith(f"z_cf_{p}_")],key=lambda x:int(x.rsplit("_",1)[1]))
    q=d[d.part==p].dropna(subset=cols); donor=q.var_donor.astype(int).to_numpy(); pred=q[cols].to_numpy().argmax(1)
    valid=(donor>=0)&(donor<len(cols)); out[p]=1-float((pred[valid]==donor[valid]).mean())
  return pd.Series(out)

G0=SW[SW.gamma==0]
tables={
 "mean response_delta":pd.concat([CB.groupby("part").response_delta.mean().rename("standard CBM"),G0.groupby("part").response_delta.mean().rename("MCBM gamma=0")],axis=1).reindex(ORDER),
 "median final margin":pd.concat([CB.groupby("part").m_cf.median().rename("standard CBM"),G0.groupby("part").m_cf.median().rename("MCBM gamma=0")],axis=1).reindex(ORDER),
 "controlled backwash":pd.concat([CB.groupby("part").backwash.mean().rename("standard CBM"),G0.groupby("part").backwash.mean().rename("MCBM gamma=0")],axis=1).reindex(ORDER),
 "exact donor-value error":pd.concat([exact_error(CB).rename("standard CBM"),exact_error(G0).rename("MCBM gamma=0")],axis=1).reindex(ORDER),
}
fig,axes=plt.subplots(2,2,figsize=(14,8))
for ax,(title,T) in zip(axes.flat,tables.items()):
 T.plot.bar(ax=ax,color=["#0072B2","#D55E00"]); ax.set_title(title); ax.set_xlabel("part")
 if "backwash" in title or "error" in title: ax.set_ylim(0,1); ax.set_ylabel("fraction")
 elif "response" in title or "margin" in title: ax.axhline(0,color="black",lw=.8); ax.set_ylabel("raw-logit units")
plt.tight_layout(); display(pd.concat(tables,names=["measurement","part"]).round(3))
""", "Figure 3. Standard CBM and MCBM gamma-zero response, final margin, controlled backwash, and exact donor-value error on identical renders."), review(3)]

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

**Variables and prediction.** `pixel_count_cf` is the exact number of target-part
pixels in the counterfactual part map. If visibility is sufficient, increasing
target area should move median `m_cf` above zero and the controlled-backwash rate
toward zero for every gamma and part.

**Method.** Use the same declared bins at every gamma and print every nonempty
denominator. These are descriptive selections: size is associated with pose,
value, and species.

**How to read.** Each row is one gamma. The left column plots median final margin
against pixel bin; above zero means donor wins. The right column plots the
controlled-backwash fraction; lower is better. Colors identify all five parts.
"""), code("f7", r"""
if "pixel_count_cf" not in SW: raise RuntimeError("fixed CSV has no pixel_count_cf")
bins=[0,20,50,100,200,500,np.inf]; labels=["0-19","20-49","50-99","100-199","200-499","500+"]
V=SW.copy(); V["visibility_bin"]=pd.cut(V.pixel_count_cf,bins=bins,labels=labels,right=False)
VT=(V.groupby(["gamma","part","visibility_bin"],observed=True)
      .agg(n=("m_cf","size"),median_m_cf=("m_cf","median"),backwash_rate=("backwash","mean")).reset_index())
fig,axes=plt.subplots(len(GAMMAS),2,figsize=(14,3.1*len(GAMMAS)),sharex=True)
for row,g in enumerate(GAMMAS):
 for p in ORDER:
  d=VT[(VT.gamma==g)&(VT.part==p)].set_index("visibility_bin").reindex(labels)
  axes[row,0].plot(labels,d.median_m_cf,"o-",label=p,color=COLORS[p])
  axes[row,1].plot(labels,d.backwash_rate,"o-",label=p,color=COLORS[p])
 axes[row,0].axhline(0,color="black",lw=.8); axes[row,0].set_ylabel(f"gamma={g:g}\nmedian m_cf")
 axes[row,1].set_ylim(0,1); axes[row,1].set_ylabel("backwash fraction")
axes[0,0].set_title("Final donor-minus-source margin"); axes[0,1].set_title("Responded but old source still wins")
for ax in axes[-1]: ax.tick_params(axis="x",rotation=35)
axes[0,1].legend(fontsize=7,ncol=3); plt.tight_layout(); display(VT.round(3))
""", "Figure 7. Final margin and controlled-backwash rate across exact target-pixel bins for every part and gamma."), review(7)]

cells += [md("f7b", r"""
## Figure 7b — How much label/visibility conflict did every gamma receive?

**Question.** Did ordinary training call a concept positive when the renderer
said its named part was not visible?

**Variables and prediction.** For exact concept `j`, the conflict rate is
`P(c_RLv2=0 | c_standard=1)`, calculated from matched records that differ only
in `attribute_label`. This is a training-data rate, not a model probability.
Because every standard MCBM gamma used the same original records, the rate is
identical across gamma. A high-conflict part is where the ±3 penalty could most
strongly reward contextual reproduction of an unsupported positive label.

**Method.** Assert every non-label record field is identical, count every exact
concept, and aggregate with positive-label counts as denominators.

**How to read.** Each exact concept appears once. A rate of 0.25 means 25 of 100
original positive labels were removed by the visibility-aware rule. The part
table is later aligned with every gamma's controlled outcome; the conflict bar
is not repeated as if gamma changed the data.
"""), code("f7b", r"""
import pickle
std_path=CURATED/"funnybirds_processed_trainval"/"train.pkl"
rl_path=CURATED/"funnybirds_processed_rl_trainval"/"train.pkl"
if not (std_path.exists() and rl_path.exists()):
  raise FileNotFoundError("matched standard/RLv2 training records required for label-conflict audit")
std=pickle.loads(std_path.read_bytes()); rl=pickle.loads(rl_path.read_bytes())
if len(std)!=len(rl): raise RuntimeError("standard/RLv2 train lengths differ")
positive=np.zeros(len(CONCEPT_NAMES),dtype=int); changed=np.zeros(len(CONCEPT_NAMES),dtype=int)
for a,b in zip(std,rl):
 for key in a:
  if key=="attribute_label": continue
  av,bv=a[key],b[key]
  equal=np.array_equal(np.asarray(av),np.asarray(bv)) if isinstance(av,(list,tuple,np.ndarray)) else av==bv
  if not bool(equal): raise RuntimeError(f"non-label record field differs: {key}")
 ca=np.asarray(a["attribute_label"]); cb=np.asarray(b["attribute_label"])
 positive+=(ca==1); changed+=((ca==1)&(cb==0))
CONFLICT_EXACT=pd.DataFrame({"concept":CONCEPT_NAMES,"part":[CONCEPT_PART[n] for n in CONCEPT_NAMES],"n_positive":positive,"n_changed":changed})
CONFLICT_EXACT["conflict_rate"]=CONFLICT_EXACT.n_changed/CONFLICT_EXACT.n_positive.replace(0,np.nan)
PART_CONFLICT=CONFLICT_EXACT.groupby("part").agg(n_positive=("n_positive","sum"),n_changed=("n_changed","sum")).reindex(ORDER)
PART_CONFLICT["conflict_rate"]=PART_CONFLICT.n_changed/PART_CONFLICT.n_positive
q=CONFLICT_EXACT.sort_values(["part","concept"]); y=np.arange(len(q)); fig,ax=plt.subplots(figsize=(10,max(6,.24*len(q))))
ax.barh(y,q.conflict_rate,color=q.part.map(COLORS)); ax.set_yticks(y,q.concept,fontsize=7); ax.invert_yaxis(); ax.set_xlim(0,1)
ax.set_xlabel("fraction of original positive labels removed by visibility rule"); ax.set_title("Training label/visibility conflict shared by every standard MCBM gamma")
plt.tight_layout(); display(q.round(3)); display(PART_CONFLICT.round(3))
""", "Figure 7b. Exact-concept and part-level training label/visibility conflict shared by every MCBM gamma."), review("7b")]

cells += [md("f8", r"""
## Figure 8 — Does the model name the exact inserted value, not merely beat the source value?

**Variables and prediction.** For each part, compare the actually inserted donor
value with the value having the largest post-swap raw logit. If minimality repairs
grounding, diagonal recognition should increase with gamma rather than collapse
onto a default value.

**How to read.** The grid has one row per gamma and one column per part. Within a
panel, rows are inserted values and columns are highest-scoring values. Each row
is normalized to one. A bright diagonal means correct exact-value attribution;
a bright off-diagonal column means many donor values collapse onto one answer.
The printed diagonal fraction is stricter than `m_cf>0` because a third value
can beat both donor and source.
"""), code("f8", r"""
diag=pd.DataFrame(index=GAMMAS,columns=ORDER,dtype=float); CONFUSION={}
fig,axes=plt.subplots(len(GAMMAS),len(ORDER),figsize=(16,2.7*len(GAMMAS)))
for ri,g in enumerate(GAMMAS):
 for ci,p in enumerate(ORDER):
  ax=axes[ri,ci]; d=SW[(SW.gamma==g)&(SW.part==p)]
  cols=sorted([c for c in SW if c.startswith(f"z_cf_{p}_")],key=lambda c:int(c.rsplit("_",1)[1]))
  q=d.dropna(subset=cols); donor=q.var_donor.astype(int).to_numpy(); pred=q[cols].to_numpy().argmax(1); valid=(donor>=0)&(donor<len(cols))
  M=np.zeros((len(cols),len(cols)))
  for a,b in zip(donor[valid],pred[valid]): M[a,b]+=1
  M=M/np.maximum(M.sum(1,keepdims=True),1); CONFUSION[(g,p)]=M; diag.loc[g,p]=(donor[valid]==pred[valid]).mean()
  ax.imshow(M,vmin=0,vmax=1,cmap="magma"); ax.set_title(f"gamma={g:g} | {p}\ndiagonal={diag.loc[g,p]:.2f}",fontsize=8)
  ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols))); ax.tick_params(labelsize=6)
  if ri==len(GAMMAS)-1: ax.set_xlabel("highest-scoring value",fontsize=7)
  if ci==0: ax.set_ylabel("inserted value",fontsize=7)
plt.tight_layout(); display(diag.round(3))
""", "Figure 8. Complete exact-value confusion matrices for all five parts at every MCBM gamma."), review(8)]

cells += [md("f8b", r"""
## Figure 8b — Are difficult donor values rare or drawn from more alternatives?

**Question.** Does exact-value support explain the gamma-specific failures?

**Variables and prediction.** Each point is one donor value. `species_support`
is the number of donor species carrying that value. The y-axis is its controlled-
backwash fraction. `alternatives_in_part` is the number of possible values for
that part and is printed in the table. A consistent downward pattern would
support rarity; a part-wide shift shared by values with overlapping support
requires another explanation.

**Method.** Use every exact donor value and all six gammas. Do not infer a stable
effect of alternative count from only five part families.

**How to read.** Each panel is one gamma. Color identifies part and text labels
the exact value index. Lower is better. The same support values can be compared
across parts, but support is not experimentally manipulated.
"""), code("f8b", r"""
VALUE_SUPPORT=(SW.groupby(["gamma","part","var_donor"])
 .agg(n=("backwash","size"),species_support=("sid_donor","nunique"),backwash_rate=("backwash","mean")).reset_index())
VALUE_SUPPORT["alternatives_in_part"]=VALUE_SUPPORT.part.map({p:hi-lo for p,(lo,hi) in SPANS.items()})
fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True)
for ax,g in zip(axes.flat,GAMMAS):
 d=VALUE_SUPPORT[VALUE_SUPPORT.gamma==g]
 for p in ORDER:
  q=d[d.part==p]; ax.scatter(q.species_support,q.backwash_rate,color=COLORS[p],label=p,s=35)
  for r in q.itertuples(): ax.annotate(f"{p[0]}{int(r.var_donor)}",(r.species_support,r.backwash_rate),fontsize=6)
 ax.set_title(f"gamma={g:g}"); ax.set_ylim(-.03,1.03); ax.set_xlabel("species supporting donor value")
axes[0,0].set_ylabel("controlled-backwash fraction"); axes[1,0].set_ylabel("controlled-backwash fraction")
axes[0,2].legend(fontsize=7); plt.tight_layout(); display(VALUE_SUPPORT.round(3))
""", "Figure 8b. Exact donor-value support and controlled-backwash rate for every part and gamma."), review("8b")]

legacy_cells = [md("f9", r"""
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

legacy_cells = [md("f10", r"""
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

legacy_cells = [md("f11", r"""
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

cells += [md("f9-new", r"""
## Figure 9 — After exact source/donor value difficulty, does source species still organize the final margin?

**Question.** If two replacements use the same source value and donor value,
does the unchanged source species/body still predict which one fails?

**Variable.** Within each gamma, part, source value, and donor value:

`species_residual_i = m_cf,i - mean(m_cf | gamma, part, source value, donor value)`.

A residual of `-2` means that replacement finished two raw-logit units more
source-favoring than comparable value pairs. Since gamma changes score scale,
the cross-gamma variable is `species_residual / SD(m_cf | gamma,part)`.

**Method and plot.** A source-species mean is shown only when supported by at
least five rows. Each small-panel dot is one source species; zero is the exact-
value expectation. The heatmaps summarize the standard deviation of the species
means. Larger spread means source species still organizes the outcome after
exact values are accounted for. This remains observational because source
species is tied to body, pose, and every unchanged part.
"""), code("f9-new", r"""
D=SW.copy()
D["matched_mean"]=D.groupby(["gamma","part","var_src","var_donor"]).m_cf.transform("mean")
D["species_residual"]=D.m_cf-D.matched_mean
D["part_scale"]=D.groupby(["gamma","part"]).m_cf.transform("std")
D["standardized_species_residual"]=D.species_residual/D.part_scale.replace(0,np.nan)
S=(D.groupby(["gamma","part","sid_src"])
   .agg(n=("m_cf","size"),raw_mean=("species_residual","mean"),
        standardized_mean=("standardized_species_residual","mean")).reset_index())
S=S[S.n>=5].copy()
Q=S.groupby(["gamma","part"]).raw_mean.std().unstack().reindex(index=GAMMAS,columns=ORDER)
Z=S.groupby(["gamma","part"]).standardized_mean.std().unstack().reindex(index=GAMMAS,columns=ORDER)
fig,axes=plt.subplots(len(GAMMAS),len(ORDER),figsize=(16,13),sharey=True)
for ri,g in enumerate(GAMMAS):
 for ci,p in enumerate(ORDER):
  ax=axes[ri,ci]; q=S[(S.gamma==g)&(S.part==p)].sort_values("standardized_mean")
  ax.scatter(range(len(q)),q.standardized_mean,color=COLORS[p],s=12)
  ax.axhline(0,color="black",lw=.8); ax.set_title(f"gamma={g:g} | {p}\n{len(q)} source species",fontsize=8)
  ax.set_xticks([])
  if ci==0: ax.set_ylabel("mean standardized residual",fontsize=7)
plt.tight_layout()
fig,ax=plt.subplots(1,2,figsize=(13,4))
heat(ax[0],Q,"Raw source-species residual spread","SD in raw-z units",0,None,"viridis")
heat(ax[1],Z,"Scale-standardized residual spread","SD / within-part m_cf SD",0,None,"viridis")
plt.tight_layout(); display(pd.concat({"raw_z_units":Q,"standardized":Z}).round(3))
""", "Figure 9. Every supported source-species residual and its spread after exact-value matching, for all parts and gammas."), review(9)]

cells += [md("f9b", r"""
## Figure 9b — Add proposed explanations one at a time and test them on held-out replacements

**Outcome.** The target is final raw-logit margin `m_cf`. Prediction error is
root mean squared error (RMSE) on held-out render IDs; lower is better. The right
panel divides RMSE by the within-gamma standard deviation of `m_cf` so gamma
scales can be compared.

The predictor starts with part, then adds inserted-pixel bin, exact source/donor
values, and source species. Group averages are learned only from four folds,
shrunken toward the preceding prediction, and tested on the fifth fold. An added
variable gets explanatory credit only if held-out error falls on the same rows.
For example, `8 -> 6` after exact values is useful; `6 -> 7` after species is not.
This is forecasting as an accounting guard, not a new definition of backwash.
"""), code("f9b", r"""
from sklearn.model_selection import KFold
STAGES=[("part",["part"]),("+ visibility",["part","pixel_bin"]),
        ("+ exact values",["part","pixel_bin","var_src","var_donor"]),
        ("+ source species",["part","pixel_bin","var_src","var_donor","sid_src"])]
A=SW.copy(); A["pixel_bin"]=pd.cut(A.pixel_count_cf,[0,20,50,100,200,500,np.inf],right=False,include_lowest=True).astype(str)
acct=[]
for g in GAMMAS:
 d=A[A.gamma==g].copy().reset_index(drop=True); scale=d.m_cf.std()
 pred={name:np.full(len(d),np.nan) for name,_ in STAGES}
 for tr,te in KFold(5,shuffle=True,random_state=20260806).split(d):
  train=d.iloc[tr].copy(); test=d.iloc[te].copy()
  previous_train=np.full(len(train),float(train.m_cf.mean())); previous_test=np.full(len(test),float(train.m_cf.mean()))
  for name,cols in STAGES:
   tmp=train[cols].copy(); tmp["resid"]=train.m_cf.to_numpy()-previous_train
   stat=tmp.groupby(cols,dropna=False).resid.agg(["mean","count"]).reset_index()
   stat["adj"]=stat["mean"]*stat["count"]/(stat["count"]+10)
   previous_train += train[cols].merge(stat[cols+["adj"]],on=cols,how="left").adj.fillna(0).to_numpy()
   previous_test += test[cols].merge(stat[cols+["adj"]],on=cols,how="left").adj.fillna(0).to_numpy()
   pred[name][te]=previous_test
 for name,_ in STAGES:
  rmse=float(np.sqrt(np.mean((d.m_cf.to_numpy()-pred[name])**2)))
  acct.append(dict(gamma=g,stage=name,rmse=rmse,standardized_rmse=rmse/scale))
ACCOUNT=pd.DataFrame(acct)
fig,ax=plt.subplots(1,2,figsize=(13,4))
for g in GAMMAS:
 q=ACCOUNT[ACCOUNT.gamma==g]
 ax[0].plot(q.stage,q.rmse,"o-",label=f"gamma={g:g}")
 ax[1].plot(q.stage,q.standardized_rmse,"o-",label=f"gamma={g:g}")
ax[0].set_ylabel("held-out RMSE in raw-z margin units"); ax[1].set_ylabel("held-out RMSE / SD(m_cf)")
for a in ax: a.tick_params(axis="x",rotation=20); a.set_xlabel("information available to predictor")
ax[1].legend(fontsize=7,ncol=2); plt.tight_layout(); display(ACCOUNT.round(3))
""", "Figure 9b. Five-fold held-out final-margin error as visibility, exact values, and source species are added sequentially."), review("9b")]

cells += [md("f10-new", r"""
## Figure 10 — Species information in known labels, internal slots, and raw concept logits

**Question.** Does minimality remove species information beyond what the known
concept labels structurally reveal?

Grey uses the known binary labels `c`; it measures information built into the
dataset. Blue uses learned raw logits `z`. Orange uses internal slots `h`, the
vector read by the species head. A separate logistic-regression probe trains on
75% of images and tests on the same held-out 25% for every model. Blind chance
is `1/50`.

Example: nine tail labels divide 50 species into tail-value buckets, so the grey
tail bar can exceed blind chance. A blue or orange bar above grey is extra within-
label species information. This makes backwash possible but does not prove that
a replacement used it. Standard CBM is the reference; MCBM gamma zero isolates
the MCBM training-noise baseline; every positive gamma tests minimality.
"""), code("f10-new", r"""
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
CBM_P=REPO/"external/minimal_cbm/results/funnybirds-cbm/1/predictions/epoch_100.pth"
CBM_CK=REPO/"external/minimal_cbm/results/funnybirds-cbm/1/models/epoch_100.pt"
if not (CBM_P.exists() and CBM_CK.exists()): raise FileNotFoundError("standard CBM epoch-100 seed-1 inputs missing")
d0=torch.load(CBM_P,map_location="cpu",weights_only=False); h0=d0["z"].float().reshape(len(d0["z"]),-1); c0=d0["c"].float().reshape(len(h0),-1)
z0=concept_logits_from_saved_latent(h0,CBM_CK,c0.shape[1]); validate_saved_probabilities(z0,d0["c_preds"])
MODEL_DATA={"CBM":dict(h=h0.numpy(),c=c0.numpy(),z=z0.numpy(),y=np.asarray(d0["y"]).reshape(-1).astype(int))}
for g in GAMMAS: MODEL_DATA[f"g={g:g}"]=HEALTH_DATA[(g,1)]
reference_y=MODEL_DATA["CBM"]["y"]
for name,d in MODEL_DATA.items():
 if not np.array_equal(d["y"],reference_y): raise RuntimeError(f"{name} prediction population/order differs")
train_idx,test_idx=train_test_split(np.arange(len(reference_y)),test_size=.25,random_state=20260806,stratify=reference_y)
BLOCKS={"complete":np.arange(len(CONCEPT_NAMES)),**{p:np.arange(*SPANS[p]) for p in ORDER}}
decode=[]
for model,d in MODEL_DATA.items():
 for block,idx in BLOCKS.items():
  for source in ["c","z","h"]:
   X=d[source][:,idx]; clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1500,C=.5,n_jobs=1))
   clf.fit(X[train_idx],reference_y[train_idx])
   decode.append(dict(model=model,block=block,source=source,accuracy=float(clf.score(X[test_idx],reference_y[test_idx])),dimensions=len(idx)))
DECODE=pd.DataFrame(decode); models=list(MODEL_DATA); blocks=list(BLOCKS)
fig,axes=plt.subplots(2,4,figsize=(18,8),sharey=True)
for ax,model in zip(axes.flat,models):
 q=DECODE[DECODE.model==model]; x=np.arange(len(blocks)); w=.25
 for k,(src,color,label) in enumerate([("c","#888888","known labels c"),("z","#0072B2","raw logits z"),("h","#E69F00","internal slots h")]):
  vals=q[q.source==src].set_index("block").loc[blocks]
  ax.bar(x+(k-1)*w,vals.accuracy,width=w,color=color,label=label)
 ax.axhline(1/50,color="black",ls="--",lw=1); ax.set_xticks(x); ax.set_xticklabels(blocks,rotation=45,ha="right"); ax.set_title(model); ax.set_ylim(0,1.03)
axes[0,0].set_ylabel("held-out species accuracy"); axes[1,0].set_ylabel("held-out species accuracy")
axes[0,0].legend(fontsize=7); axes.flat[-1].axis("off"); plt.tight_layout(); display(DECODE.round(3))
""", "Figure 10. Held-out species decoding from known labels, raw logits, and internal slots for standard CBM and all MCBM gammas."), review(10)]

cells += [md("f11-new", r"""
## Figure 11 — Is recognition of the same positive concept species-dependent?

The authoritative `fb_recallv2` method has two stages. First, for an exact
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
"""), code("f11-new", r"""
from itertools import combinations
rec=[]; coverage=[]; B_RECALL=300
for model,d in MODEL_DATA.items():
 gamma=-1 if model=="CBM" else float(model.split("=")[1]); y=d["y"]; c=d["c"].astype(int); z=d["z"]
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
model_order=["CBM"]+[f"g={g:g}" for g in GAMMAS]
Rg=RECALL.groupby(["model","part"])[["recall_gap","balanced_accuracy_gap","standardized_raw_z_gap"]].median()
R1=Rg.recall_gap.unstack().reindex(index=model_order,columns=ORDER)
RB=Rg.balanced_accuracy_gap.unstack().reindex(index=model_order,columns=ORDER)
R2=Rg.standardized_raw_z_gap.unstack().reindex(index=model_order,columns=ORDER)
fig,ax=plt.subplots(1,3,figsize=(18,4))
heat(ax[0],R1,"Median matched-species positive-recall gap","absolute recall difference",0,1,"magma")
heat(ax[1],RB,"Median matched-species balanced-accuracy gap","absolute BA difference",0,1,"magma")
heat(ax[2],R2,"Median matched-species standardized raw-z gap","within-concept SD units",0,None,"viridis")
plt.tight_layout(); display(RECALL.groupby(["model","part","pairing_rule"]).agg(pairs=("recall_gap","size"),median_recall_gap=("recall_gap","median"),median_balanced_accuracy_gap=("balanced_accuracy_gap","median"),median_raw_z_gap=("standardized_raw_z_gap","median")).round(3))
""", "Figure 11. Two-stage matched-species recall, balanced-accuracy, and standardized raw-logit gaps for standard CBM and every MCBM gamma."), review(11)]

cells += [md("f11b", r"""
## Figure 11b — Align the four notebook-02 measurements with every MCBM outcome

Panel A is controlled backwash on every replacement. Panel B uses only clearly
visible insertions (`pixel_count_cf >= 100`). Panel C is the fraction of positive
training labels removed by the visibility-aware rule; it is one shared data
property and is shown once, not copied across gamma. Panel D is exact donor-value
error after replacement. Lower is better in every panel.

`CBM` is the standard-CBM reference. Rows `g=0` through `g=5` use the identical
fixed-render population for MCBM. The panels are aligned to test whether part
orderings agree, but they are not added: their populations and meanings differ.
"""), code("f11b", r"""
model_rows=["CBM"]+[f"g={g:g}" for g in GAMMAS]
ALL_BW=pd.DataFrame(index=model_rows,columns=ORDER,dtype=float)
VIS_BW=pd.DataFrame(index=model_rows,columns=ORDER,dtype=float)
EXACT_ERR=pd.DataFrame(index=model_rows,columns=ORDER,dtype=float)
ALL_BW.loc["CBM"]=CB.groupby("part").backwash.mean().reindex(ORDER)
VIS_BW.loc["CBM"]=CB[CB.pixel_count_cf>=100].groupby("part").backwash.mean().reindex(ORDER)
EXACT_ERR.loc["CBM"]=exact_error(CB).reindex(ORDER)
for g in GAMMAS:
 q=SW[SW.gamma==g]; label=f"g={g:g}"
 ALL_BW.loc[label]=q.groupby("part").backwash.mean().reindex(ORDER)
 VIS_BW.loc[label]=q[q.pixel_count_cf>=100].groupby("part").backwash.mean().reindex(ORDER)
 EXACT_ERR.loc[label]=1-diag.loc[g].reindex(ORDER)
fig,ax=plt.subplots(1,4,figsize=(20,5))
heat(ax[0],ALL_BW,"A. Responded, but source still wins","fraction",0,1,"magma_r")
heat(ax[1],VIS_BW,"B. Same event, inserted part >=100 px","fraction",0,1,"magma_r")
pc=PART_CONFLICT[["conflict_rate"]].T.reindex(columns=ORDER); pc.index=["shared training data"]
heat(ax[2],pc,"C. Positive-label / visibility conflict","fraction",0,1,"magma_r")
heat(ax[3],EXACT_ERR,"D. Wrong exact donor value after swap","fraction",0,1,"magma_r")
plt.tight_layout(); display(pd.concat({"all_backwash":ALL_BW,"visible_backwash":VIS_BW,"training_conflict":pc,"exact_value_error":EXACT_ERR}).round(3))
""", "Figure 11b. Standard CBM and every MCBM gamma aligned on the four notebook-02 measurements."), review("11b")]

legacy_cells = [md("f12", r"""
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

cells += [md("f12-new", r"""
## Figure 12 — Does concept attribution change the donor-species output?

For each part and model, replacements are divided into eight approximately
equal-count bins by final raw-logit margin `m_cf`. The x coordinate is mean
`m_cf` in that bin; the y coordinate is mean species-head probability assigned
to the donor species. A point is therefore a group of swaps, not one swap. The
vertical line marks equal donor/source concept logits.

This is intentionally the one grounding section using a class probability: its
question is downstream classification, not concept grounding. Standard CBM is
shown with every MCBM gamma. A rising line means donor-species support tends to
increase when the inserted donor concept becomes more dominant. The absolute
probability can remain small because replacing one part does not turn the whole
bird into the donor species.
"""), code("f12-new", r"""
DOWN=pd.concat([CB.assign(model="CBM")]+[SW[SW.gamma==g].assign(model=f"g={g:g}") for g in GAMMAS],ignore_index=True)
if "p_cf_donor" not in DOWN: raise RuntimeError("p_cf_donor absent from fixed-render output")
fig,axes=plt.subplots(1,5,figsize=(19,3.8),sharey=True)
palette=plt.cm.viridis(np.linspace(0,1,len(model_rows)))
bin_rows=[]
for ax,part in zip(axes,ORDER):
 for model,color in zip(model_rows,palette):
  d=DOWN[(DOWN.model==model)&(DOWN.part==part)].copy(); d["bin"]=pd.qcut(d.m_cf,8,duplicates="drop")
  q=d.groupby("bin",observed=True).agg(m_cf=("m_cf","mean"),donor_species_probability=("p_cf_donor","mean"),n=("p_cf_donor","size")).reset_index()
  q["model"]=model; q["part"]=part; bin_rows.append(q); ax.plot(q.m_cf,q.donor_species_probability,"o-",ms=3,label=model,color=color)
 ax.axvline(0,color="black",lw=1); ax.set_xlabel("mean final margin m_cf"); ax.set_title(part)
axes[0].set_ylabel("mean P(donor species)"); axes[-1].legend(fontsize=7,ncol=2); fig.suptitle("Downstream class response, separated by replaced part"); plt.tight_layout()
display(pd.concat(bin_rows,ignore_index=True)[["model","part","m_cf","donor_species_probability","n"]].round(3))
""", "Figure 12. Donor-species probability by final concept margin for standard CBM and every MCBM gamma, separated by part."), review(12)]

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
md("ledger", r"""
## Evidence ledger before notebook 06

This table prevents a visually striking panel from silently becoming a stronger
claim than its design supports. It must be updated only after every new output
above has been displayed and reviewed.

| Proposed statement | Required evidence here | Status after complete visual review |
|---|---|---|
| Gamma implements the intended compression | target RMSE and within-label `h` spread fall; exact `z` outputs remain healthy | **ACCEPTED FOR REPRESENTATION COMPRESSION**, with the Figure 2b full-range collapse correction still to execute |
| MCBM begins from the same discovered problem | standard CBM and gamma-zero use identical renders and predicates | **ACCEPTED FOR THE QUALITATIVE BASELINE**; gamma-zero numerical differences are not minimality |
| Inserted pixels affect the model | `response_delta>0`, with both directions and declared visibility strata | **ACCEPTED FOR SEED 1**; positive donorward response occurs for every part/gamma and in both directions |
| Minimality repairs controlled backwash | `m_cf` rises and `P(response_delta>0,m_cf<0)` falls without broken health | **VALID TEST, NO SUPPORT AS A GENERAL REPAIR**; tail worsens, while selected beak/eye settings improve |
| Training conflict contributes | the conflict ordering aligns with causal failure and remains after visibility restriction | **ACCEPTED AS AN ASSOCIATIONAL CONTRIBUTOR**, not an explanation of the gamma curve |
| Exact-value difficulty contributes | donor-value confusion/support predicts held-out margin variation | **ACCEPTED AS AN ASSOCIATIONAL CONTRIBUTOR**; rarity alone is insufficient |
| Source species/body contributes beyond values | supported species residuals remain and source species lowers held-out error | **ACCEPTED AS AN OBSERVATIONAL CONTRIBUTOR**; species is not independently manipulated |
| Species leakage causes backwash | decoding/recall plus controlled swap show the required model behavior | **NOT ESTABLISHED BY DECODING OR RECALL ALONE**; the controlled response/final-margin pair remains the causal evidence |
| The gamma curve is reproducible | fixed-render replay from independent trained seeds | **INCOMPLETE** where Figure 13 shows one causal seed |

Identified contributors are not subtracted arithmetically. The held-out sequence
in Figure 9b is the accounting test: a contributor explains reproducible margin
variation only when adding it lowers error on unseen replacements. Residual error
is reported rather than promised to become zero.
"""),
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
