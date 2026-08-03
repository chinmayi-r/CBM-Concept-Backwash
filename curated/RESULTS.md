# RESULTS LOG — measured numbers, provenance, status

> **Supersession notice (2026-08-03):** this table contains historical and
> provisional rows. Use `PREDICATE_PROOF_LEDGER.md` and `CURRENT_STATE.md` for
> current claim status. In particular, no CUB deletion, patch, or insertion row
> currently supports a causal CUB backwash claim. The validated causal centerpiece
> remains the FunnyBird fixed-render swap.

I update this every time you send outputs (pasted, pushed notebook, or figure).
STORY.md quotes nothing until it's marked LOCKED here. Status: 🟡 provisional · 🟢 locked.

| date | dataset | model / γ | metric | value | n / seed / epoch | claim | status |
|------|---------|-----------|--------|-------|------------------|-------|--------|
| 07-10 | FunnyBirds | CBM | deletion backwash (overall) | ~0.085 | 100 imgs · s1 · ep150 | C1 | 🟡 |
| 07-10 | FunnyBirds | CBM | deletion backwash (tail) | ~0.36 | 100 imgs · s1 · ep150 | C1 | 🟡 |
| 07-10 | FunnyBirds | CBM | deletion backwash (wing/foot/beak) | ~0.00 | 100 imgs · s1 · ep150 | C1 | 🟡 |
| 07-10 | FunnyBirds | CBM | task acc / concept acc | 75% / 99.6% | val · s1 · ep150 | C4 | 🟡 |
| 07-12 | FunnyBirds | CBM | tail retained_frac: **all vs VISIBLE-ONLY** | 0.308 → **0.118** (frac_noop 0.324) | s1 · ep150 | C1 | 🟡 |
| 07-12 | FunnyBirds | CBM | deletion retained_frac per part (VISIBLE-only) | tail .118 · eye .037 · beak .032 · foot .010 · wing .005 | s1 · ep150 | C1 | 🟡 |
| 07-12 | FunnyBirds | MCBM | γ-sweep COMPLETE (g0/0.1/0.3/1/3/5, all-rows) | 0.115/0.095/0.108/0.093/0.109/0.113 | 2500 rows · s1 · ep100 | C2 | 🟡 |
| — | method | — | **deletion test gated to VISIBLE-only** (changed_frac>1e-3); tail no-op ~32% | — | — | C1 | — |
| 07-13 | FunnyBirds | CBM | renderer-swap z-ordering_correct per part | tail .37 · beak .49 · eye .60 · wing .84 · foot .89 | swap · s1 · 100 pairs | C1 | 🟡 |
| 07-13 | FunnyBirds | CBM | tail z-ordering: all vs VISIBLE-only(≥50px) | 0.37 → 0.48 (occlusion partial; backwash survives) | swap · s1 | C1 | 🟡 |
| 07-13 | FunnyBirds | CBM | tail concept-confusion diagonal (correctly attributed) | 0.27 (73% mis-attributed) | swap · s1 | C1 | 🟡 |
| 07-13 | FunnyBirds | CBM | on-distribution grounding (donor concept logit, part absent) | strongly NEGATIVE (−23…−36) → §42 pre-swap backwash does NOT replicate | swap · s1 | control | 🟢 |
| — | STORY | — | 3 probes agree: tail most backwashed (deletion .118 / swap .37 / confusion .27); wing/foot grounded | — | s1 | C1 | 🟡 |
| 07-12 | FunnyBirds | MCBM | γ-control (rep_loss / mean\|z\| moved 458→0.15 / 18.4→3.0) | **γ bit → flat retention is a real refutation** | s1 · ep100 | C2 | 🟡 |
| — | naming | all | metric column is `retained_frac` (never "backwash"); backwash = its interpretation | — | — | — | — |
| 07-12 | FunnyBirds | CBM | species←z | 0.990 ± 0.015 | 500 imgs · s1 · **ep150** · chance 0.02 | C4 | 🟡 |
| 07-12 | FunnyBirds | CBM | species←c_preds (26-d concept vector) | 0.996 ± 0.005 | 500 imgs · s1 · ep150 · chance 0.02 · *partly tautological* | C4 | 🟡 |
| 07-12 | FunnyBirds | CBM | species←single part's concepts | tail .182 / wing .120 / beak .082 / foot .080 / eye .064 | 500 imgs · s1 · ep150 · chance 0.02 | C4 | 🟡 |
| 07-12 | FunnyBirds | (data, no model) | class×concept binary frac (train) / test species-constancy | 0.808 / 1.000 | 50k tr · 500 te · 26 concepts | C3 | 🟡 |
| 07-12 | FunnyBirds | (data, no model) | per-part n_variants / frac_visible | tail 9/.635 · wing 6/.756 · beak 4/.749 · foot 4/.757 · eye 3/.753 | train | C3 | 🟡 |
| 07-13 | CUB (raw, no model) | — | within-species variation (all 312 attrs) | 57.4% of (species,attr) pairs vary | 11788 img · 200 sp | C3/P3 | 🟡 |
| 07-13 | CUB (raw) | — | genuinely ambiguous pairs (present 30–70%) | 9.6% (full) / 8.1% (cub70) | raw image-level | C3/P3 | 🟡 |
| 07-13 | CUB (raw) | — | image-attr cells relabeled by CBM majority-vote | 7.8% (full) / 7.3% (cub70) | 312 attrs | C3/§3b | 🟡 |
| — | CUB (raw) | — | 112-CBM-used-attr version | pending re-run (noise-robust) | — | C3/§3b | ◻ |
| — | CUB70 | CBM/MCBM | occlusion z-firing | pending (needs masks + trained model) | — | C5 | ◻ |
| — | CUB70 | CBM (labeled vs relabeled) | Δ backwash | pending | — | C6 | ◻ |
| — | full CUB | CBM/MCBM | recall gap / species-probe | pending | — | C3/P3 | ◻ |

## How a result gets logged
1. You send outputs (paste / push notebook / figure).
2. I add/replace the row, note provenance (n, seed, epoch), map it to a claim.
3. I re-check STORY.md: if the number changes the narrative, I update the story and
   flag it; if it's a placeholder firming up, I flip 🟡→🟢 once it's the final run.
