# STORY — the paper, in full (SOURCE OF TRUTH)

Governs every artifact. Each notebook/script names the claim it serves; default to
the minimal demonstration; correct this file rather than drifting from it.

--------------------------------------------------------------------------------
## 0. Thesis (the whole paper in three sentences)

Concept Bottleneck Models (CBMs) are sold as *interpretable* because a human-named
concept (e.g. "red wing", "notched tail") is predicted first and the class is read
from those concepts — so an explanation like "it's a Herring Gull *because* it has
a grey undertail" is supposed to be faithful. We show that this faithfulness is an
illusion driven by **concept–class backwash**: the concept unit does not read its
part, it reads the **class / whole-bird identity** and reports the concept value
the class is *supposed* to have — so the concept still fires when the part is
**occluded, deleted, or swapped to a different variant**. This backwash is present
in standard CBMs, **survives the "minimal" MCBM bottleneck across its entire
strength range γ** (the model explicitly designed to remove exactly this leakage),
generalizes from the controlled FunnyBirds world to real CUB birds, and is
**invisible to every metric the CBM and MCBM papers report** — because those
metrics are all measured on-distribution, where "reads the part" and "reads the
class" are the same function.

--------------------------------------------------------------------------------
## 1. Background — the two models we test and what they promise

**CBM (Koh et al., ICML 2020).** x → c → y: a backbone predicts a vector of concept
logits `z` (one per human-named attribute), a concept head gives `ĉ = σ(z)`, and a
label head predicts `y` from the concept layer. The selling points:
- *Interpretability*: the prediction is "explained" by the concept vector.
- *Intervention (TTI)*: replace a predicted concept with its ground-truth value and
  the class prediction updates — Fig 4 shows accuracy rising as more concepts are
  corrected.
- *Robustness* (TravelingBirds, Table 3): concepts resist a background shift.
All of this is measured **on the natural test set**, where each species' concepts
take their canonical values.

**MCBM (Almudévar et al., "There Was Never a Bottleneck in CBMs").** Argues CBM's
`z_j` is a *sufficient* but not *minimal* statistic of `c_j` — it leaks extra
information — and adds an Information Bottleneck so each `z_j` encodes **all and
only** `c_j`. The training objective (their code, and our verified copy):

```
L = CE(y_logits, y)                      # task
  + λ_c · BCE(z, c)                       # concept supervision, on CLEAN z
  + γ   · 0.2 · mean_j (q(z_j) − z_j)^2   # IB penalty, on CLEAN z
        q(z_j) = 6·σ(z_j) − 3   (soft target: +3 if c_j=1, −3 if c_j=0)
```
The label head is fed a **noisy** `z_s = z + σ·ε` (the "bottleneck"); concept and IB
losses use **clean** `z`. γ is the minimality strength; **effective force = γ·0.2**.
MCBM's evidence: a leakage metric (URR: predict a *nuisance* from [z,c] vs [c]) and
intervention-reliability curves — again **on-distribution**, and URR needs
*independent nuisances* to exist at all.

**(Optional third family) LF-CBM** (label-free CBM, arXiv 2304.06129): concepts from
CLIP, no concept labels. Same backwash question applies; include only if C1–C6 leave
room. Not on the critical path.

--------------------------------------------------------------------------------
## 2. The phenomenon — concept–class backwash, defined

A concept unit `z_j` is **grounded** if its value is *caused by* the part it names:
change the part (in pixels) and `z_j` follows; hide the part and `z_j` goes quiet.
It is **backwashed** if its value is instead *inferred from the class / global
identity*: `z_j` reports the concept the recognized species is supposed to have,
regardless of what the part actually shows. The two are **behaviorally identical on
the natural test set** (each species shows its canonical part), and **only diverge
off-distribution** — when the part contradicts, is missing from, or is swapped out
of the class it belongs to. That off-distribution divergence is the entire
experimental lever of the paper.

--------------------------------------------------------------------------------
## 3. Why the field hasn't seen it (our core argument)

- **On-distribution identifiability failure.** Because concept = f(class) on the data
  (FunnyBirds: exactly; CUB: strongly), "detect the part" and "look up the class's
  part" produce the *same* concept vector. No correlational metric — concept
  accuracy, recall, a linear probe, TTI, URR — can separate them.
- **Content vs source.** MCBM's minimality constrains *what `z_j` contains*
  (`I(z_j; x | c_j)=0`). Backwash is about *which pixels `z_j` reads* — the
  computational **source**. When `c_j = f(class)`, "encode only `c_j`" is fully
  satisfiable by **class-lookup**, so a content constraint cannot forbid it, and a
  content metric cannot detect it. This is *why* MCBM's bottleneck is expected to
  fail here, not just an empirical surprise.
- **The papers' own evidence is orthogonal or scoped away** (recorded in DECISIONS
  §D.2): CBM interventions (Fig 3/4) *overwrite* `ĉ` with the ground-truth, so they
  test the c→y head, never the x→c grounding; TravelingBirds shifts a *decorrelated*
  artifact (background) while keeping concept↔class intact; MCBM's URR is inert when
  concepts are class-determined (our FunnyBirds run literally prints `nuisances=0`);
  and both intervene only on *visible* concepts, explicitly stepping around the
  occluded case that would expose backwash.

--------------------------------------------------------------------------------
## 4. Discovery arc (the paper's narrative order)

1. **FunnyBirds + standard CBM** — the phenomenon first shows up here (recall gap,
   then the renderer swap). Controlled synthetic world with a 3-D renderer and
   ground-truth part maps ⇒ we can *causally* manipulate one part at a time.
2. **Small subset of CUB** — reproduced on real birds.
3. **Large part of CUB** — it holds at scale ⇒ not a synthetic artifact.
4. **Across model families** — persists in CBM and, crucially, in **MCBM across all
   γ** (the model that promises to remove it), i.e. it is a property of label-only
   concept supervision on class-correlated data, not of a particular architecture.

FunnyBirds is the **centerpiece** (causal ground truth + renderer); CUB/CUB70 is the
**real-data generalization**.

--------------------------------------------------------------------------------
## 5. Method battery (how we make the invisible visible)

Ordered from cheap-indicator to causal-proof:

**(a) Recall gap — cheap INDICATOR, not proof (correlational).** Per concept, compare
recall on species that canonically have it vs. don't. Flags that concepts track
species. Powered only where concepts vary within a species (CUB, if the pkls are
image-level); on FunnyBirds it's n-per-species quantization noise (species-constant),
so it's an appendix/indicator there. Significance via bootstrapping (1000s of trials).

**(b) Renderer part-SWAP / z-ordering — the primary FunnyBirds CAUSAL test (your
`fb_*_renderer_swap` notebooks).** Re-render a species-A bird with one part swapped to
species-B's variant (same camera/light/background, *only the part changes*), run both
through the CBM/MCBM, and read `margin = z_new − z_old` (donor-concept minus
source-concept activation after the swap). `frac_correct = P(margin > 0)` = the model
correctly noticed the new part. **Grounded ⇒ frac_correct ≫ 0.5, margin > 0;
backwashed ⇒ frac_correct ≈ 0.5, margin ≈ 0** (z stayed on the source species' value =
"anchoring"). For tail, also record all 9 tail-dim activations → a confusion/anchoring
matrix (did it stay on the source variant, or misfire on a third?).
  *(Caveat that MADE the restart necessary: the old hand-written MCBM computed the
  concept/IB losses on the NOISY `z_s`, so `z` collapsed to ~0, every margin was 0,
  and every part read frac_correct=0.5 at every γ — a false null. The official
  `minimal_cbm` uses clean `z` for those losses, so the restart fixes this for free.)*

**(c) Occlusion / deletion — CAUSAL, renderer-free, unifies both datasets.** Remove or
mask the part and read the concept probability (or `z`). **Grounded ⇒ it drops toward
absent; backwashed ⇒ it stays high** (the model asserts a part it cannot see —
"is z≈1 even in the occluded image?", the professor's exact question). FunnyBirds:
use the pre-rendered part-deletion interventions (no live renderer). CUB70: mask the
part region using the real segmentation masks. This is the one method that runs on
FunnyBirds *and* real CUB70 with no renderer. (Deletion = the zero-visibility extreme
of occlusion; graded occlusion decomposes occlusion-driven fallback vs pure anchoring.)

**(d) Species probe on z — how much class identity lives in the concept vector.**
Train a probe to predict species from `z` (and compare `z` vs the raw avg-pool
features). High ⇒ the concept layer is largely a class code = backwash substrate.
Also test whether it's the *probe* that's bad (use the CBM's own trained weights, or
GradCAM/LIME/IG saliency — does the concept's evidence land on the part or the whole
bird?).

--------------------------------------------------------------------------------
## 6. FunnyBirds results (the centerpiece)

- **CBM is mostly grounded EXCEPT specific parts.** Deletion test (epoch-150 CBM):
  overall backwash 0.085, but **part-specific** — wing/foot/beak ≈ 0, eye 0.05,
  **tail 0.36** (asserts the species' tail 36% of the time when the tail is gone).
  Non-uniformity ⇒ genuine signal, not global OOD confusion.
- **Tail is the backwash-prone part — candidate causes (both contribute, per your
  analysis; not ranked):** (i) **variant count** — tail has 9 variants (the most) ⇒
  largest/hardest concept space; (ii) **occlusion / low visibility** — tail is often
  small or hidden in renders ⇒ little pixel evidence ⇒ the model leans on the class
  prior. The renderer-swap seg-analysis measured `pixel_count_cf` per swap and found
  occlusion is one of the causes (it was tested, not ruled out). Failing tail
  variants (e.g. 1,2,7) vs grounded (5,8) → the confusion matrix shows **anchoring**
  (stays on the source species' tail).
- **Across MCBM γ (the money result):** run (b) and (c) on the γ-sweep and plot
  backwash / (1−frac_correct) **vs γ**. Prediction: it stays high (minimality trains
  `z_j → 6c_j−3`, a *class-determined* target, so it may even AMPLIFY tail backwash).
  Either "flat/high" or "rises" is a paper-making result the MCBM authors never test.
  Deliverable: `backwash_vs_gamma.png` + the per-part table.

--------------------------------------------------------------------------------
## 7. Generalization to real data (CUB / CUB70)

- **Full CUB (200)** — carries the **recall-gap indicator** *iff* the pkls are
  image-level (attributes vary within a species; the CUB notebook §4 decides). No part
  masks ⇒ no occlusion grounding on full CUB.
- **CUB70 (70)** — the CUB subset **with per-part segmentation masks** (head/eye/beak/
  neck/body/wing/leg/tail). Masks ⇒ we can run method (c) occlusion on *real* birds:
  mask a part, does its concept still fire? This is the real-data version of the
  FunnyBirds deletion result. Train fresh CBM + MCBM on CUB70 (official code).
- **Source-side fix (constructive):** visibility-aware **relabeled** CBM — zero a
  concept's label when its part is occluded in that image — should *reduce* backwash
  vs the label-constant CBM. Compare `z` of each part in occluded vs visible images,
  relabeled vs labeled (professor's step). This closes the loop: backwash is fixable
  only by a **source-side** intervention, consistent with §3.

--------------------------------------------------------------------------------
## 8. Claims → minimal experiment → status

- **C1 CBM backwashes (FunnyBirds).** occlusion/deletion (c) + swap (b). **DONE (deletion; tail 0.36).**
- **C2 Persists across MCBM γ.** (b)+(c) on the γ-sweep → backwash-vs-γ. **sweep running.**
- **C3 Part-specific; tracks variant-count + occlusion.** per-part backwash + candidate features. **notebooks done; per-part pending sweep.**
- **C4 Standard metrics can't see it.** assemble: 99% concept acc, TTI "works", URR≈0, all beside high backwash. **argued; assemble.**
- **C5 Real-data generalization (CUB70 occlusion).** train CBM+MCBM on CUB70; occlude via masks. **TODO (professor priority).**
- **C6 Source-side relabel reduces it.** relabeled CBM; z occluded-vs-visible, relabeled-vs-labeled. **TODO (professor priority).**

--------------------------------------------------------------------------------
## 9. Roadmap (FunnyBirds-first, simplest path)

- **P1 [running] FunnyBirds CBM + MCBM γ-sweep → deletion backwash-vs-γ** (+ per-part).
  C1–C4. Add the renderer-SWAP z-ordering on the same models as the richer causal
  cut *if the Node renderer is available* (it tracks *which* variant, not just
  present/absent).
- **P2 [next; professor's "steps for next meeting"] CUB70 CBM + MCBM → occlusion
  z-firing, labeled vs relabeled.** C5 + C6. Real-data generalization + the fix.
- **P3 [optional] full-CUB recall-gap indicator** (only if attributes vary within
  species; bootstrap for significance).

--------------------------------------------------------------------------------
## 10. Working contract
1. Every artifact names the claim (C1–C6) it serves; if none, don't build it.
2. Default to the minimal demo (reuse outputs, fewest files, CPU where possible);
   flag "nice-to-have" and wait for a yes.
3. No gold-plating off the critical path. Simplicity > elegance.
4. Check each notebook/script against this file before building; state the claim.
5. You may veto/reprioritize any claim; settled ones aren't re-litigated.
