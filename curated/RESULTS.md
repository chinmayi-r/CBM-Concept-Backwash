# RESULTS LOG — measured numbers, provenance, status

I update this every time you send outputs (pasted, pushed notebook, or figure).
STORY.md quotes nothing until it's marked LOCKED here. Status: 🟡 provisional · 🟢 locked.

| date | dataset | model / γ | metric | value | n / seed / epoch | claim | status |
|------|---------|-----------|--------|-------|------------------|-------|--------|
| 07-10 | FunnyBirds | CBM | deletion backwash (overall) | ~0.085 | 100 imgs · s1 · ep150 | C1 | 🟡 |
| 07-10 | FunnyBirds | CBM | deletion backwash (tail) | ~0.36 | 100 imgs · s1 · ep150 | C1 | 🟡 |
| 07-10 | FunnyBirds | CBM | deletion backwash (wing/foot/beak) | ~0.00 | 100 imgs · s1 · ep150 | C1 | 🟡 |
| 07-10 | FunnyBirds | CBM | task acc / concept acc | 75% / 99.6% | val · s1 · ep150 | C4 | 🟡 |
| — | FunnyBirds | MCBM γ-sweep | backwash vs γ | **artifact produced** (need csv) | s1 · ep100 · g0..g5 | C2 | 🟡 |
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
