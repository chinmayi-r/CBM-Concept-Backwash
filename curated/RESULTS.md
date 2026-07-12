# RESULTS LOG — measured numbers, provenance, status

I update this every time you send outputs (pasted, pushed notebook, or figure).
STORY.md quotes nothing until it's marked LOCKED here. Status: 🟡 provisional · 🟢 locked.

| date | dataset | model / γ | metric | value | n / seed / epoch | claim | status |
|------|---------|-----------|--------|-------|------------------|-------|--------|
| 07-10 | FunnyBirds | CBM | deletion backwash (overall) | ~0.085 | 100 imgs · s1 · ep150 | C1 | 🟡 |
| 07-10 | FunnyBirds | CBM | deletion backwash (tail) | ~0.36 | 100 imgs · s1 · ep150 | C1 | 🟡 |
| 07-10 | FunnyBirds | CBM | deletion backwash (wing/foot/beak) | ~0.00 | 100 imgs · s1 · ep150 | C1 | 🟡 |
| 07-10 | FunnyBirds | CBM | task acc / concept acc | 75% / 99.6% | val · s1 · ep150 | C4 | 🟡 |
| — | FunnyBirds | MCBM γ-sweep | backwash vs γ | pending | sweep running | C2 | ◻ |
| — | CUB70 | CBM/MCBM | occlusion z-firing | pending | — | C5 | ◻ |
| — | CUB70 | CBM (labeled vs relabeled) | Δ backwash | pending | — | C6 | ◻ |
| — | full CUB | CBM/MCBM | recall gap / species-probe | pending | — | C3/P3 | ◻ |

## How a result gets logged
1. You send outputs (paste / push notebook / figure).
2. I add/replace the row, note provenance (n, seed, epoch), map it to a claim.
3. I re-check STORY.md: if the number changes the narrative, I update the story and
   flag it; if it's a placeholder firming up, I flip 🟡→🟢 once it's the final run.
