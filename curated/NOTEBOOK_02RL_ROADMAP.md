# Notebook 02rl roadmap: matched FunnyBird CBM relabeling test

## Scope

Notebook `02rl_funnybirds_cbm_relabeled.ipynb` answers one causal question:

> Did positive concept labels attached to invisible parts cause some of the
> controlled backwash measured for the standard FunnyBird CBM in notebook 02?

The comparison is standard CBM versus matched CBM-RLv2 only. It contains no
MCBM and no gamma sweep. Notebook 03 remains the standard MCBM minimality test;
notebook 03rl remains the later MCBM-RLv2 extension.

RLv2 means visibility-aware relabeling, not reinforcement learning:

`c_RLv2[i,j] = c_standard[i,j] * v[i,g(j)]`.

## Required proof ladder

| Step | Question | Required evidence |
|---|---|---|
| 0 | Why was RLv2 proposed? | Reproduce accepted notebook-02 Figures 4, 6, 6b, and 9b before any RLv2 result |
| 1 | What changed? | Exact changed-label counts and concrete changed records for all five parts |
| 2 | Is this a fair causal comparison? | Same ordered train/validation images and classes; configuration differs only in `data.pkls_dir`; epoch and seed match |
| 3 | Are the evaluated pixels identical? | One-to-one render IDs plus identical original and counterfactual hashes and renderer preflight |
| 4 | Are both models usable? | Raw-`z` spread, label separation, balanced accuracy, positive recall, and task accuracy |
| 5 | Did RLv2 reduce the controlled event? | Paired standard/RLv2 final margins and rates of `response_delta>0 and m_cf<0` for every part |
| 6 | Which rows changed status? | Resolved, remaining, introduced, and never-candidate counts on identical swaps |
| 7 | What score changed? | Paired change in donor score, removed-source score, final margin, and `response_delta` |
| 8 | Do simpler alternatives survive? | Direction, exact visibility, every donor value, source species, and downstream task checks |
| 9 | What remains? | Final aligned summary and an evidence ledger that separates causal resolution from observational residuals |

## Predictions stated before results

The matched label audit in notebook 02 found the greatest conflict for tail,
with much smaller conflict for beak and eye and almost none for foot and wing.
Therefore, before viewing RLv2 behavior:

1. tail should have the largest reduction in controlled candidate events;
2. beak and eye may improve modestly;
3. foot and wing should change little;
4. the most direct score change should be a lower removed-source logit after
   replacement;
5. final donor-minus-source margin should move right;
6. `response_delta` may or may not increase because RLv2 intentionally changes
   original-image scores as well as replacement-image scores;
7. RLv2 need not eliminate exact-value or source-species residual structure.

## Required reporting pattern

Every result uses notebook 02's sequence:

`question -> variables and prediction -> method -> figure -> literal
observation -> strongest alternative -> discriminating test -> limited
conclusion -> next question`.

Every plotted quantity is defined before it appears. The five FunnyBird parts
always use `tail, wing, beak, foot, eye`. Important figures are shown in chat
before their interpretations are accepted. Seed 1 is explicitly provisional;
row or species-pair resampling is not presented as training-seed uncertainty.

## Final causal boundary

If the matched RLv2 model resolves more standard candidate events than it
introduces, especially for tail, the notebook may conclude that contradictory
training labels caused part of the standard-CBM backwash. Remaining exact-value
and source-species patterns remain observational until independently
manipulated. RLv2 neither creates nor replaces the original notebook-02 proof.
