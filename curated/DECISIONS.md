# Curated pipeline — decisions, deviations, problems (living doc)

The lab notebook for the `curated/` restart. Every non-obvious choice, every
deviation from the official repos, and every bug found lives here, with the
rationale phrased so it can drop straight into the paper's Methods / Limitations.
Keep it updated as things change. Pinned upstream commits: `external/COMMITS.txt`.

Status legend: ✅ done · ⚠️ partial / caveat · ❌ open · 🔬 verified-in-code

---

## A. Deviations from the official repositories (Methods / reproducibility)

Adaptation lives in `curated/`. Where an in-place edit is meaningfully smaller and
easier to verify than a shim, we make it as a **small tracked patch** in
`curated/patches/<submodule>.patch`, applied on top of the pinned SHA by
`setup.sh` (idempotent). The submodule tree stays clean-at-SHA; the patch is the
citable record of the change (SHA + patch = fully reproducible).

| Upstream | What we do differently | Why | Where |
|---|---|---|---|
| minimal_cbm | Add a **FunnyBirds loader** via an explicit `elif` in `get_loader` (a 5-line patch), not a runtime monkeypatch | Repo only ships CUB200/CIFAR10/disentanglement/CelebA/Spirals; explicit dispatch is far easier to verify than dynamic patching | `patches/minimal_cbm.patch`, `compat/mcbm_funnybirds.py` (loader body) |
| minimal_cbm | Force **wandb offline** (`TrainExperiment.wandb_offline=True` + env) | Repo hardcodes an author's wandb key and inits **online**; on a cluster that fails or leaks to a stranger's account | `train/run_mcbm.py` |
| minimal_cbm | Author our own **configs to the real schema**, generated per-run from templates | Bundled examples are `configs/cub12/*`; our datasets/backbones need their own | `train/configs/*`, `train/_paths.sh` |
| minimal_cbm | **Val-based model selection** via a `<pkls>_trainval` dir | Repo evaluates on `test` every epoch (no val); selecting on that leaks test | `data/make_val_split.py`, `train/_paths.sh` |
| ConceptBottleneck (yewsiang) | Kept as a **separate, labeled** official-CBM cross-check, not mixed into the same-backbone comparison | Different codebase/backbone; only for the canonical 3-regime CUB reference | `train/cbm_cub.sh` |

Generated configs/results are written **inside** `external/minimal_cbm/{configs,results}/`
(the trainer keys off those paths). This dirties the submodule *working tree* but
touches no tracked file. ⚠️ open cleanup: redirect results outside `external/`.

---

## B. Facts verified directly in the official code (so results are interpretable)

- 🔬 **Minimality force = `gamma × 0.2 × MSE(z, ±3)`**, target `z = 6c − 3`
  (`src/models/mcbm.py: get_loss_z`, `get_loss`, `q_z_c`). The `0.2` is baked in.
  ⇒ the old hand-written training under-scaled gamma; the sweep goes to `gamma=30`
  (effective 6.0) to reach the strong-bottleneck regime the old runs never did.
- 🔬 **CBM ≠ MCBM at gamma=0.** `MCBM.forward` injects `var_z` noise into z during
  training (`sampled_z = z + var_z*randn`, `sampling=True` in the train loop);
  `CBM.forward` uses z directly. So gamma=0 MCBM is "MCBM architecture, no
  minimality, still stochastic z" — a valid ablation, **not** the CBM reference.
  The true reference is `model_type: cbm` (`train/configs/*-cbm.yaml`).

### B.1 Key theoretical finding — MCBM minimality ≠ causal grounding (the study's thesis)

MCBM's objective (Almudévar Eq. 4) is `min I(Z_j; X | C_j)`, i.e. `z_j` carries no
information about the input **beyond** `c_j` (equivalently the Markov chain
X ↔ C_j ↔ Z_j; `p(z_j|c_j)=p(z_j|x)`). Implemented as `MSE(f_θ(x), 6c_j−3)`.

This constrains the **content** of `z_j`, which is **path-independent**. It does
**not** constrain how `z_j` is computed. On FunnyBirds `c_j` is a deterministic
function of the species, so an encoder that **recognizes the species and looks up
`c_j(s)`** outputs `z_j = g_z(c_j)` — a deterministic function of `c_j` — and thus
satisfies `I(Z_j;X|C_j)=0` **perfectly**, despite never using the part pixels.
"Encode only `c_j`" and "encode species, as far as it determines `c_j`" are
informationally identical when `c = f(species)`, so no content-based bottleneck
can separate them. (Worse: the task head already computes species to predict `y`,
so species-lookup for concepts is the path of least resistance.)

MCBM's minimality genuinely removes **independent nuisances** `n ⟂ c` and other
concepts `c_k` from `z_j` — real and useful on their datasets (MPI3D, Shapes3D,
CUB-12, where attributes vary within class). It is **silent on grounding** exactly
in the class-determined-concept regime their threat model (`p(x,y,c,n)` with `n`
independent) excludes — which is the FunnyBirds regime.

Consequences: (1) identifying part→concept vs species→concept requires an
**intervention** that breaks the `c=f(species)` correlation → the counterfactual
**part swap** is theoretically forced, not just convenient. (2) The gamma sweep
measures, empirically, whether raising γ reduces backwash **in practice anyway** —
a case the MCBM paper never tests. (3) Paper thesis: *MCBM minimality is about
representation content, not causal grounding; when concepts are class-determined,
minimality is achievable by class-lookup and does not prevent concept–class
backwash, which we expose via counterfactual part swaps across bottleneck
strength γ.*

---

### B.2 Empirical γ-sweep result + how backwash is measured (and its caveats)

**Measurement.** `backwash = mean(p_removed)/mean(p_intact)` per (image, present
part): render the part deleted, read `c_preds` for the species-typical concept;
`p_intact` = its prob on the full bird, `p_removed` = its prob with the part gone.
= fraction of the concept's "present" probability that survives deleting the part.
1 = full backwash, 0 = grounded. CSV "overall" averages all 5 parts (diluted; tail
is the signal).

**Result (s1, ep100, official code):** overall backwash is **flat in γ** — cbm
0.085; mcbm g0/0.1/0.3/1/5 = 0.115/0.095/0.108/0.093/0.113 (g3 NaN, re-run). MCBM
sits **at or above** CBM with no downward trend.

**Why the loss predicts this (not a coincidence).** The minimality term is
`γ·0.2·mean((6c−3 − z)²)` — an MSE pulling `z_j(x)` toward the ±3 target set by the
concept **label** `c_j`. It regularizes the code's **saturation**, not its **pixel
source**; and since `c_j=f(class)` on FunnyBirds the target is class-derived, so
raising γ *entrenches* class-reading rather than removing it. Backwash flat/rising
in γ is the predicted outcome, not a null.

**Two caveats (do not skip before locking):**
1. *Saturation confound.* `c_preds` is a learned head on `z`; as γ saturates `z`,
   its outputs polarize, so a ratio-of-probs metric can drift for reasons unrelated
   to grounding. Robustness check = deletion effect on the **z-margin/logit**
   (`z_intact − z_removed`), not the prob ratio.
2. *"γ did nothing" alternative.* Flat backwash only refutes minimality if γ
   actually changed the representation. **Required positive control** (notebook 03
   §4): show a minimality metric that moves with γ — `mean((±3−z)²)`↓ / `mean|z|`↑
   read from the saved test-set `z`. If it moves and grounding stays flat → clean
   refutation; if it's also flat → γ range underpowered, widen before concluding.

---

## C. Methodological decisions (rationale for the paper)

1. **Base-case-first ordering.** vanilla → CBM (fully analyzed) → MCBM gamma sweep.
   The instruments (data analysis, recall, swap, grounding) are validated on the
   reference before the treatment. `train/run_baselines.sh` then
   `train/mcbm_gamma_sweep.sh`.
2. **Same-backbone comparability.** vanilla/CBM/MCBM run through one trainer with
   one encoder, so the only variable is head + gamma. **Primary backbone: resnet50,
   held FIXED.** Optional robustness axis is **capacity (resnet18 vs resnet50)** —
   *not* family: resnet50 ≈ inception_v3 in size, so that pairing would test
   nothing. One knob: `ARCH=`.
3. **FunnyBirds vs CUB division of labor.**
   - FunnyBirds concepts are **species-constant** (verified: within-species std ≈ 0
     on test; notebook 01 §2b). So the matched-pair **recall gap is confounded with
     species and n=10 underpowered** → appendix only. The **causal concept-swap**
     is the FunnyBirds headline: it renders/intervenes the *same species* with a
     *different part variant*, severing the species↔concept correlation, so whether
     the prediction follows the part (grounded) or the species (backwash) is
     identifiable.
   - CUB attributes **vary within species** → correlation broken naturally → CUB
     carries the correlational **recall-gap-vs-gamma** axis.
4. **Probe-grounding confound → resolved by intervention, not a better probe.** At
   layer4 species is ~99% linearly decodable, so a single-layer concept probe can
   succeed by decoding species and table-lookup rather than detecting the part.
   Correlational methods can't escape this on a species-constant dataset; the
   counterfactual **swap** does (see C.3). Optional secondary diagnostic: concept
   emergence *across layers* relative to species.
5. **Model selection = validation-based (standard, Koh et al. 2020).** Hold out a
   val fold, select the checkpoint on val, touch test **once**. minimal_cbm has no
   val, so `data/make_val_split.py` builds `<pkls>_trainval/` where the trainer's
   `test.pkl` **is the val fold**; the real test stays untouched for final eval.
   `_paths.sh` auto-prefers `_trainval` and **warns** if it's absent (else per-epoch
   eval is on test = leak). FunnyBirds: carve 10%/class from 50k train. CUB: reuse
   the official `val.pkl`.
6. **Concept-vector gate.** `data/funnybirds/validate_concept_vectors.py` checks our
   26-slot vectors match the official parts.json indexing **before** training.
   Passed: 0 / 50,500 mismatches.
7. **Absent part = all-zero concept group** (~1/4 of images per part). Treated as a
   legitimate state; documented so reviewers know it's intentional.

---

## D. Bugs found and fixed during curation

| Bug | Impact if shipped | Fix |
|---|---|---|
| sweep `sed` replaced **every** `gamma:` line | clobbered `scheduler.gamma` (LR decay) on every run → wrong LR schedule | token `__GAMMA__`, only `model.gamma` substituted; guard asserts `scheduler.gamma`=0.1 |
| **fabricated config schema** (`name:`/`manifest_dir:`/`${oc.env}`) | crashes in `read_config`/`get_model`; looks plausible | migrated to real schema; `_paths.sh` guard rejects un-migrated templates |
| CBM(inception,27M) vs MCBM(resnet18,12M) | 2.3× capacity gap confounds CBM-vs-MCBM | same-backbone spine (C.2) |
| sbatch: conda not initialized; `set -u` vs conda `_CE_M` | job dies before training | robust activation + `set +u` around it |
| notebook written to `curated/curated/…` | wrong path | moved; species-constancy folded into existing `notebooks/01` |

---

## D.1 The narrative (as the study is actually framed)

**Thesis: concept–class ("species") backwash is universal across CBM variants.**
Discovery order (for the paper's arc):
1. Found with **CBM on FunnyBirds** (recall gap).
2. Reproduced on a **small subset of CUB**.
3. Then on a **large part of CUB** — so it generalizes.
4. Persists across **every flavor**, including **MCBM** (which *promises* separation)
   and across all bottleneck strengths γ. MCBM is the strongest **foil**, not the subject.

**The recall gap is an INDICATOR, not proof.** It is correlational: it flags that
concept predictions track species, but cannot alone establish that species *causes*
the concept prediction (vs. genuine within-part signal). The **counterfactual part
swap** (FunnyBirds) is the causal clincher; the recall gap is the broad, cheap
evidence that motivates looking. Both belong in the paper, clearly labeled as
indicator vs. proof.

Why no loss fix (recorded in B.1): backwash is a **source** problem (which pixels
`z_j` reads), and all CBM-family losses constrain **content** (`z_j`'s information),
so none can fix it. The two real levers are **counterfactual data augmentation**
(break the class↔concept correlation in training; FunnyBirds renderer makes it
exact) and **spatial-source constraints** (attribution-to-part supervision, or
part-local/prototype architectures). This is the Discussion / Future Work.

Training-length note (from the first full run): vanilla & CBM on FunnyBirds
plateau by ~epoch 30 (val task ~75%, concept ~99.6%) then overfit (val loss climbs
4×); n_epochs cut 200→100, save_epochs 50→20. NOTE the minimal_cbm "vanilla" still
bottlenecks to sum(dim_z)=26 dims, so 75% is the 26-dim-latent number, not an
unbottlenecked ResNet-50 ceiling (~1.0 in the FunnyBirds paper) — fair for
vanilla-vs-CBM-vs-MCBM (all 26-dim) but not a task ceiling.

## D.2 Reconciling the CBM/MCBM papers' positive results (pre-empt the reviewer)

- **CBM Fig 3 & 4 (interventions work)**: test the **c→y** path by *replacing* predicted
  `ĉ` with ground-truth `c`. This **overwrites x→c**, where backwash lives, so it is
  structurally blind to it; a fully backwashed model yields identical curves. Orthogonal.
- **CBM Table 3 / TravelingBirds (background robustness)**: a *different* spurious
  feature. CBM resists background because training **decorrelates** concept↔background
  ("a concept spans many backgrounds"). Backwash is concept↔**species**, which training
  **preserves** (concept = f(species)). Same mechanism, opposite outcome. The paper's own
  scope line: CBM helps "when y is more correlated with training artifacts than the
  concepts c" — background qualifies, species does not (species *is* y). TravelingBirds
  shifts background but keeps concept↔label invariant → never tests backwash.
- **CBM §6.2 own caveats**: (i) results "optimistic … assume birds of the same species
  always share the same concept values" = the determinism backwash exploits; (ii) they
  intervene **only on visible concepts**, explicitly avoiding the not-visible case — which
  is exactly the deletion diagnostic.
- **MCBM leakage (URR, Tables 3–4)** needs independent nuisances `n`; inert when concept
  = f(class) (our FunnyBirds `nuisances = 0.0`). MCBM interventions (Fig 5) use Koh's
  on-distribution ground-truth protocol → same blind spot.

Net: none of these measure concept↔class backwash; each is orthogonal or scoped to a
decorrelated artifact. That is the gap our off-manifold tests fill.

## D.3 Cleanest way to SHOW backwash (the two off-manifold tests)

On-distribution, "reads the part" and "reads the part-associated-species-set" are the
**same function** (they agree on every real bird), so the recall gap is only an
**indicator**. To make them disagree you must go **off the 50-species manifold**
(5 parts, 4·3·4·9·6 = 5184 combos, only 50 are species — so most single-part edits are
novel). FunnyBirds ships a renderer (`json_to_image`/`render_class`) + `classes.json` +
`test_interventions`, so both tests are directly runnable.

1. **Deletion (cleanest, no bookkeeping).** Remove part `j` (render bird without it).
   - grounded → predicts the part's concept group **absent** (all-zero; a legal state);
   - backwashed → still predicts the **species-typical** variant (reports a part it cannot see).
   - Metric: **backwash rate** = fraction of part-removed images where the model still
     predicts a present/species-typical variant; plus mean `|z_j|` (confidence). A
     confident concept on a part-less bird is unambiguous backwash.
2. **Novel-combination swap.** Swap part `j` to another variant.
   - **Stratify** by the swap-graph (built from `classes.json`): swaps that land on a real
     species are *uninformative* (grounded & backwashed both correct) → **exclude**; score
     only **novel combos**.
   - Metric on novel combos: concept accuracy on the swapped part (grounded ≈ unchanged;
     backwashed collapses toward species-typical), and whether `z_j` follows the part or
     the species.

Both plotted vs γ across flavors = the money figures. Recall gap = cheap on-distribution
indicator that motivates looking; deletion/swap = the causal proof.

## D.4 First deletion-grounding result (funnybirds-cbm, epoch 150, 100 imgs)

Overall backwash **0.085** (grounding 0.915) — the CBM largely READS parts; deleting
a part collapses its concept prob 1.00→0.085. But **part-specific**:
beak .012, eye .050, foot .002, wing .001, **tail .362**. Tail = most variants (9),
hardest discrimination → substantial backwash (asserts the species' tail 36% when the
tail is absent). Non-uniformity ⇒ genuine signal, not OOD confusion.

Thesis refinement: "pervasive CBM backwash" does NOT hold for a well-trained FunnyBirds
CBM; the real phenomenon is **part-specific backwash where visual evidence is weakest**.
Crux is now `backwash(tail) vs γ`: minimality trains `z_j → 6c_j−3` (species-determined
target), so MCBM may **amplify** tail backwash — the striking version of the result.

Refinement to harden the metric (TODO): on a part-removed image, also confirm the
**other (present) parts stay correctly predicted** — rules out global OOD confusion and
isolates the drop to the removed part. (Different parts already behave very differently,
which argues against a pure-OOD artifact, but the check makes it airtight.)

## E. Known limitations / open items

- ❌ FunnyBirds test = **10 images/species** — recall-gap underpowered (→ appendix).
- ❌ `cub70-*` configs still fabricated schema (guard blocks them until migrated).
- ⚠️ CUB `n_groups_concepts: 28` unverified (must equal #attribute groups or the
  loader raises).
- ⚠️ Env actually used is `cubvision-gpu`, not the shipped `environment-mcbm.yml` —
  snapshot `conda list` of the real env for provenance.
- ⚠️ Generated results written inside `external/` (working-tree pollution).
- ❌ Backbone capacity-robustness (resnet18 vs resnet50) not yet run.
- ❌ Final val→test selection+eval script (picks best-val epoch, evals real test) —
  to be built with the eval-table dump step.

---

## F. Reproducibility anchors

- Upstream SHAs: `external/COMMITS.txt`.
- Seeds: `-s` per run (≥3 for error bars on any "X vs gamma").
- Deterministic config generation from templates via `train/_paths.sh: gen_config`.
