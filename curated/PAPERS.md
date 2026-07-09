# Source papers — approaches end-to-end, and what we borrow

Synthesized from the **primary sources we vendor** (the repo code + READMEs at the
pinned SHAs in `external/COMMITS.txt`) plus the papers. This is the "related work
/ heavy-inspiration" reference for the curated study; keep it for the paper's
Related Work and Methods.

Pinned: CBM `d6353f2`, minimal_cbm `9ba535c`, funnybirds `91b4b46`,
funnybirds-framework `1350bbe`.

---

## 1. CBM — "Concept Bottleneck Models" (Koh, Nguyen, Tang, Mussmann, Pierson, Kim, Liang; ICML 2020)

**Idea.** Don't go pixels→label end-to-end. Go **x → c → y**: first predict a
vector of human-specified concepts `c`, then predict the label `y` from `c` only.
Because `y` sees only `c`, you can **intervene** — edit predicted concepts at test
time and the label update follows.

**Data / CUB attribute subset** (this is the "subset of CUB" you asked about).
CUB-200-2011 ships **312** binary attributes. CBM denoises them (majority-vote per
class, keep attributes present in enough classes) down to **112** attributes,
organized into **28 attribute groups** (`has_bill_shape`, `has_wing_color`, …).
Backbone: **InceptionV3**, ImageNet-pretrained, 299px.

**Three training regimes** (all in `external/ConceptBottleneck`, `CUB/README.md`):
- **Independent**: train `x→c` and `c→y` separately; `c→y` is trained on *ground-truth* concepts.
- **Sequential**: train `x→c`, freeze, then train `c→y` on *predicted* concepts.
- **Joint**: end-to-end, loss = task + `λ`·concept.

**Key evaluation.** Task acc, concept acc, and **TTI (test-time intervention)**:
replace predicted concepts with ground truth and watch task accuracy rise — the
headline "interpretability payoff."

**What we borrow.** The three regimes (we run CBM and state which regime we
report); TTI as an interpretability-vs-γ curve; the 28-group concept structure.

> The known crack this study is about: a *soft* concept scalar can encode **more
> than its concept** ("leakage"/"impurity"; Mahinpei et al. 2021, Margeloiu et
> al. 2021) — i.e. your "backwash." CBM does not prevent it. That motivates MCBM.

---

## 2. MCBM — "There Was Never a Bottleneck in Concept Bottleneck Models" (Almudévar, Hernández-Lobato, Ortega; arXiv 2506.04877)

**This is our study's backbone — its thesis IS our research question.** Direct
quote (README): *"the fact that a component can predict a concept does not
guarantee that it encodes only information about that concept."* CBM has no true
bottleneck.

**Method (verified in `src/models/mcbm.py`, not just the abstract).** Each concept
`j` has a scalar latent `z_j = encoder(x)_j`. An **Information-Bottleneck**
regularizer pulls `z_j` to carry *only* `c_j`:
- fixed-point prior `q(z_j|c_j)`: target `= 6·c_j − 3`  → **+3** if concept present, **−3** if absent;
- IB loss term: `L_z = 0.2 · MSE(z_j, 6c_j−3)`, added to the total as
  `L = L_task + β·L_concept + γ·L_z`  ⇒ **effective minimality force = γ · 0.2**;
- **variational** part: during training `z` is sampled as `z + var_z·ε` (Gaussian
  noise), so it's a stochastic latent, not a point estimate. `var_z=1` in configs.
- `γ` = IB / bottleneck strength = **our sweep axis**.

**Datasets / backbones.** Controlled-factor datasets where the *nuisances are
known*: MPI3D, Shapes3D, CIFAR-10 (CLIP synthetic attributes), and **CUB (they
sample 12 attribute groups as concepts, the rest as nuisances)**. This "known
nuisance" design is central to how they measure leakage — see below.

**Leakage metric (verified in `src/experiments/train.py: evaluate()`).** For each
nuisance attribute, train an sklearn classifier to predict it from **`[z, c]`**
vs. from **`c` alone**, and report the **accuracy delta** and **mutual-information
delta** (`res_joint − res_conc`). If `z` leaks non-concept info, `[z,c]` predicts
nuisances *better* than `c` alone → positive delta. **A good bottleneck drives the
delta toward 0.** The repo also ships a full disentanglement suite in
`src/helpers/disentanglement/` (**DCI, MIG, SAP, modularity, InfoMEC, IRS**).

**Interventions.** MCBM intervenes through the learned `q(z_j|c_j)` — set `z_j` to
its fixed point for the desired concept value (Bayes-consistent), vs. CBM's
quantile-swap intervention.

**Finding.** MCBM lowers the leakage delta and gives more reliable interventions
than CBM at comparable task/concept accuracy.

**What we borrow.**
- **γ is the spine.** Our sweep to γ=30 (effective 6.0) reaches the strong-IB
  regime the old mis-scaled runs never did.
- **Their leakage delta is a ready-made backwash metric** — but it needs *known
  nuisances*. → On **CUB** we use it directly (nuisances = the non-selected
  attribute groups). On **FunnyBirds** our concepts are the whole label, so there
  are **no nuisance attributes** → this metric is inert (that's the `nuisances_* =
  0.0` you saw). FunnyBirds leakage therefore comes from the **causal swap** (§4).
- The **z→±3 snap** is exactly what the smoke/sweep validates; the disentanglement
  suite gives extra leakage axes vs γ.

---

## 3. FunnyBirds — "A Synthetic Vision Dataset for a Part-Based Analysis of XAI" (Hesse, Schaub-Meyer, Roth; ICCV 2023, oral)

**Idea.** XAI has **no ground-truth explanations**, so evaluation is unsolved.
FunnyBirds fixes that with a **synthetic, fully controllable** world: cartoon
birds whose **classes are defined by part attributes** (beak, eye, wing, tail,
foot types), rendered so you can **remove or swap individual parts** and get
counterfactual images with known effects.

**Data.** 50 classes built from combinations of part variants; a renderer produces
each bird and its **part maps** (per-part segmentation → visibility). Because the
class→part map is exact, the dataset gives **ground-truth part importance**.

**Evaluation protocol.** Automatic scoring of the **Co-12** properties —
**Completeness, Correctness, Contrastivity** — via part interventions. The core
mechanism is **controlled part deletion/swap**: intervene on a part in the
renderer, measure the model's output change → ground-truth importance; metrics
like **target sensitivity** check whether an explanation focuses on
class-discriminative parts. Models evaluated are plain classifiers
(**ResNet-50**, VGG16, ViT), ImageNet-pretrained, output dim 50, trained to ≈1.0
accuracy.

**What we borrow.** The **renderer part-swap IS our FunnyBirds causal test.** Swap
a part on a fixed species and ask: does the concept prediction follow the **part**
(grounded) or stay with the **species** (backwash)? This is ground-truth-backed
and, crucially, **breaks the species↔concept correlation** that makes correlational
tests (recall gap, single-layer probes) uninterpretable on a species-constant
dataset. Their plain-ResNet-50 models also match our same-backbone choice.

---

## 4. How this shapes OUR design (the heavy-inspo synthesis)

| Element | Borrowed from | Our use |
|---|---|---|
| Same backbone across models (resnet50) | FunnyBirds (plain resnet50) + comparability | vanilla / CBM / MCBM all resnet50 |
| x→c→y + 3 regimes + TTI | CBM (Koh 2020) | CBM reference; TTI vs γ as interpretability payoff |
| IB regularizer, γ strength, ±3 fixed points | MCBM (Almudévar) | the gamma sweep spine; z-snap validation |
| Leakage = nuisance-pred delta `[z,c]` vs `[c]` | MCBM code | **CUB** leakage-vs-γ (nuisances = unused attr groups) |
| Disentanglement suite (DCI/MIG/SAP…) | MCBM repo | extra leakage/disentanglement axes vs γ |
| Renderer part deletion/swap → ground truth | FunnyBirds | **FunnyBirds** causal backwash test (no nuisances there) |
| Controlled datasets w/ known nuisances | MCBM | why CUB carries the correlational axis, FunnyBirds the causal one |

**The one-line thesis these combine into:** take MCBM's γ-controlled information
bottleneck, and measure — *as a function of γ* — whether concept components stop
encoding species identity, using MCBM's nuisance-leakage delta on CUB (known
nuisances) and FunnyBirds' renderer part-swap on FunnyBirds (known ground truth).
That is the gamma-sweep spine, with each paper supplying one leg.

*Sources: vendored repo code + READMEs (pinned SHAs above); CBM = Koh et al. ICML
2020; MCBM = Almudévar et al. arXiv 2506.04877; FunnyBirds = Hesse et al. ICCV
2023 (arXiv 2308.06248).*
