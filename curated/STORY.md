# STORY — the paper, the claims, the minimal path (SOURCE OF TRUTH)

Every notebook/script must serve a claim below. If it serves none, don't build it.
Default to the *minimal* demonstration. This file governs; correct it, don't drift.

## Thesis (one sentence)
Concept Bottleneck Models — and their "minimal"/information-bottleneck successor
(MCBM), which *claims* to fix concept leakage — **do not ground concepts in their
parts**: the concept representation is read off the class / whole bird, so it still
fires for a part that is **occluded, deleted, or swapped**, this concept–class
**backwash persists across the MCBM minimality strength γ**, and it is **invisible
to the metrics both papers report**.

## What we stand on / refute
- **CBM (Koh 2020)**: x→c→y; concepts "interpretable + intervenable." Evidence: concept
  accuracy, test-time intervention (replace ĉ with GT c → acc rises). All **on-distribution**.
- **MCBM (Almudévar)**: CBM z_j leaks info beyond c_j; add an IB so z_j is a *minimal
  sufficient statistic* of c_j. Evidence: URR leakage (predict nuisance from [z,c] vs [c]),
  intervention curves. Needs **independent nuisances**.
- **Our refutation**: neither tests **causal grounding** (does the *part* cause the concept).
  On-distribution, "reads the part" and "reads the class" are the **same function** (part and
  class are correlated). Break the correlation — occlude / delete / swap the part — and the
  concept follows the **class, not the part**, even under maximal minimality.

## Claims (each = ONE minimal experiment) and status
- **C1 — Backwash exists in CBM.** Occlude/delete a part → concept still fires.
  FunnyBirds CBM. **STATUS: DONE** (overall backwash 0.085; tail 0.36).
- **C2 — It persists across MCBM γ.** Same test on the MCBM γ-sweep → `backwash vs γ`
  stays high. **STATUS: sweep running** → `backwash_vs_gamma.png`.
- **C3 — Part-specific; tracks part difficulty/occlusion.** Per-part backwash + candidate
  features (variant count, visibility). **STATUS: data notebooks done; per-part from C1/C2.**
- **C4 — Standard metrics can't see it.** Concept acc ~99%, intervention/TTI "works," MCBM
  leakage delta ≈ 0 (no nuisances on class-determined concepts) — all while backwash is high.
  **STATUS: argued + `nuisances=0` observed; assemble side-by-side.**
- **C5 — Generalizes to real data (CUB70).** Train CBM+MCBM on CUB70; occlude a part using
  the real segmentation masks → concept still fires. **STATUS: TODO (professor priority).**
- **C6 — Source-side fix reduces it (constructive).** Visibility-aware *relabeled* CBM
  (zero a concept when its part is occluded) → lower backwash than the label-constant CBM.
  **STATUS: TODO (professor priority).**

## Why (our mechanism / the punchline)
MCBM's minimality constrains z_j's **content** (I(z_j; x | c_j)=0). Backwash is a **source**
problem (which pixels z_j reads). When c_j = f(class), "encode only c_j" is satisfiable by
**class-lookup**, so minimality can't forbid it and content-based metrics can't detect it.
The fix must be **source-side** (visibility relabeling C6, or spatial grounding).

## Roadmap (priority order = simplest path)
- **P1 [running] FunnyBirds CBM + MCBM γ-sweep → occlusion/deletion backwash vs γ.**
  Covers C1–C4. Deliverable: `backwash_vs_gamma.png` + per-part table.
- **P2 [next; professor's "steps for next meeting"] CUB70 CBM + MCBM → occlusion z-firing,
  labeled vs relabeled.** Covers C5, C6. Real-data generalization + the constructive fix.
- **P3 [optional] full-CUB recall-gap** as the cheap *correlational indicator* — only if
  attributes vary within species (CUB notebook §4 decides). Indicator, not proof.
- **Deferred unless asked: live-renderer part SWAP** (z_new−z_old margin / z-ordering, your
  `fb_*_renderer_swap` notebooks). Richer (tracks *which* variant) but needs the FunnyBirds
  Node renderer. **Occlusion/deletion achieves the same claim renderer-free**, so it is the
  primary method; the swap is a FunnyBirds-only robustness add.

## Method note
Primary grounding metric = **occlusion/deletion**: mask/remove the part, read the concept
prob (or z). Fires when the part is absent = backwash. Renderer-free, works on **FunnyBirds**
(pre-rendered part deletions) *and* **CUB70** (real masks) → one method, both datasets.
NB: official `minimal_cbm` computes the concept loss on **clean z** (not noisy z_s), so it
avoids the z-collapse bug that made your old MCBM 50/50 — the restart fixes that for free.

## Working contract (so I stay on the simplest path)
1. I name the claim (C1–C6) every artifact serves; if none, I don't build it.
2. I default to the minimal demo — reuse existing outputs, fewest files, CPU where possible.
   I flag "nice-to-have" and wait for a yes before building it.
3. I do **not** gold-plate infra off the critical path. Simplicity > elegance.
4. Before writing a notebook/script I check it against this file and say which claim it serves.
5. You may veto/reprioritize any claim; I won't re-litigate settled ones.
