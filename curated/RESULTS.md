# RESULTS LOG — measured numbers, provenance, status

I update this every time you send outputs (pasted, pushed notebook, or figure).
STORY.md quotes nothing until it's marked LOCKED here. Status: 🟡 provisional · 🟢 locked.

| date | dataset | model / γ | metric | value | n / seed / epoch | claim | status |
|------|---------|-----------|--------|-------|------------------|-------|--------|
| 07-10 | FunnyBirds | CBM | deletion backwash (overall) | ~0.085 | 100 imgs · s1 · ep150 | C1 | 🟡 |
| 07-10 | FunnyBirds | CBM | deletion backwash (tail) | ~0.36 | 100 imgs · s1 · ep150 | C1 | 🟡 |
| 07-10 | FunnyBirds | CBM | deletion backwash (wing/foot/beak) | ~0.00 | 100 imgs · s1 · ep150 | C1 | 🟡 |
| 07-10 | FunnyBirds | CBM | task acc / concept acc | 75% / 99.6% | val · s1 · ep150 | C4 | 🟡 |
| 07-12 | FunnyBirds | CBM | deletion retained_frac per part (=P removed/P intact) | tail .308 · eye .054 · beak .048 · foot .010 · wing .005 | 2500 rows · s1 · ep150 | C1 | 🟡 |
| 07-12 | FunnyBirds | CBM | deletion retained_frac (overall) | 0.085 | 2500 rows · s1 · ep150 | C1 | 🟡 |
| 07-12 | FunnyBirds | MCBM | overall retained_frac vs γ (g0/0.1/0.3/1/5) | 0.115 / 0.095 / 0.108 / 0.093 / 0.113 | 2500 rows · s1 · ep100 | C2 | 🟡 |
| 07-12 | FunnyBirds | MCBM | **tail** retained_frac vs γ (g0/0.1/0.3/1/5) | 0.374 / 0.382 / 0.420 / 0.360 / 0.450 (rising, ≥CBM) | s1 · ep100 · g3 pending | C2 | 🟡 |
| 07-12 | FunnyBirds | MCBM | γ-control (rep_loss / mean\|z\| moved 458→0.15 / 18.4→3.0) | **γ bit → flat retention is a real refutation** | s1 · ep100 | C2 | 🟡 |
| — | naming | all | metric column is `retained_frac` (never "backwash"); backwash = its interpretation | — | — | — | — |
| 07-12 | FunnyBirds | CBM | species←z | 0.990 ± 0.015 | 500 imgs · s1 · **ep150** · chance 0.02 | C4 | 🟡 |
| 07-12 | FunnyBirds | CBM | species←c_preds (26-d concept vector) | 0.996 ± 0.005 | 500 imgs · s1 · ep150 · chance 0.02 · *partly tautological* | C4 | 🟡 |
| 07-12 | FunnyBirds | CBM | species←single part's concepts | tail .182 / wing .120 / beak .082 / foot .080 / eye .064 | 500 imgs · s1 · ep150 · chance 0.02 | C4 | 🟡 |
| 07-12 | FunnyBirds | (data, no model) | class×concept binary frac (train) / test species-constancy | 0.808 / 1.000 | 50k tr · 500 te · 26 concepts | C3 | 🟡 |
| 07-12 | FunnyBirds | (data, no model) | per-part n_variants / frac_visible | tail 9/.635 · wing 6/.756 · beak 4/.749 · foot 4/.757 · eye 3/.753 | train | C3 | 🟡 |
| — | CUB70 | CBM/MCBM | occlusion z-firing | pending | — | C5 | ◻ |
| — | CUB70 | CBM (labeled vs relabeled) | Δ backwash | pending | — | C6 | ◻ |
| — | full CUB | CBM/MCBM | recall gap / species-probe | pending | — | C3/P3 | ◻ |

## How a result gets logged
1. You send outputs (paste / push notebook / figure).
2. I add/replace the row, note provenance (n, seed, epoch), map it to a claim.
3. I re-check STORY.md: if the number changes the narrative, I update the story and
   flag it; if it's a placeholder firming up, I flip 🟡→🟢 once it's the final run.
