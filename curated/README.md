# `curated/` — clean-room CBM & MCBM pipeline

This folder is a **restart** of the training and dataset-analysis code. The
earlier top-level code in this repo was written quickly and we are no longer
confident it is faithful to the source papers. `curated/` fixes that by
**building on the official released implementations** instead of
reimplementing them:

| Model | Paper | Official code (submodule) |
|-------|-------|----------------------------|
| **CBM**  | Koh, Nguyen, Tang et al., *Concept Bottleneck Models*, ICML 2020 (arXiv:2007.04612) | [`yewsiang/ConceptBottleneck`](https://github.com/yewsiang/ConceptBottleneck) → `external/ConceptBottleneck` |
| **MCBM** | Almudévar, Hernández-Lobato, Ortega, *There Was Never a Bottleneck in Concept Bottleneck Models* (arXiv:2506.04877) | [`antonioalmudevar/minimal_cbm`](https://github.com/antonioalmudevar/minimal_cbm) → `external/minimal_cbm` |

FunnyBirds itself is also taken from the **official** source, not any hand-written
module in this repo:

| Dataset | Paper | Official code (submodule) |
|---------|-------|----------------------------|
| **FunnyBirds** | Hesse, Schaub-Meyer, Roth, *FunnyBirds: A Synthetic Vision Dataset…*, ICCV 2023 | [`visinf/funnybirds-framework`](https://github.com/visinf/funnybirds-framework) (loader `FunnyBirds`, eval protocols) → `external/funnybirds-framework`; [`visinf/funnybirds`](https://github.com/visinf/funnybirds) (render + custom eval) → `external/funnybirds` |

The FunnyBirds concept schema (parts/variants) and per-image part visibility are
read from the **official** `parts.json`, `dataset_{mode}.json`, and part-map PNGs
(`{mode}_part_map/{class_idx}/{idx:06d}.png`, color→part map from
`funny_birds.py`). The dataset `FunnyBirds.zip` is downloaded separately.

**Nothing in `external/` is edited.** It is a pinned, verbatim copy of the
official release so the paper can say "we used release X at commit Y." Anything
we need to change for compatibility or for our datasets lives in *our* code
(`compat/`, `patches/`, `data/`, `train/`, `analysis/`), which is reviewable
and cited in the appendix.

> **Provenance note for the paper.** After `git submodule update --init`,
> record both commit SHAs (`setup.sh` prints them to `external/COMMITS.txt`).
> Report them in the experimental-setup section.

---

## 0. What runs where

This scaffold was authored in an environment with **no GPU, no datasets, and no
GitHub egress**, so it has not been executed here. It is designed to run on the
cluster (adroit), which has the data and GPUs. Every script documents its
expected inputs/outputs. The order is: `setup.sh` → build data → train →
notebooks.

---

## 1. Setup (run once, on a networked machine)

```bash
# from repo root, on a networked machine (the gitlinks were not committable in the
# offline authoring env, so the FIRST init goes through setup.sh, which falls back
# to `git submodule add` when no pinned commit exists yet):
cd curated
bash setup.sh                                 # populates external/, records commit SHAs, builds envs, import sanity checks
# then commit the newly created submodule gitlinks so future clones only need:
#   git submodule update --init --recursive
```

The two reference repos have **incompatible dependency stacks** (CBM is
2020-era TF/PyTorch; MCBM is recent), so they get **separate environments**:

- `environment-cbm.yml`  → conda env `cbm`   (drives `external/ConceptBottleneck`)
- `environment-mcbm.yml` → conda env `mcbm`  (drives `external/minimal_cbm`)

Known compatibility edits we apply (see `patches/` for the full list and
rationale): old `torchvision` Inception weights API, NumPy/`np.int` aliases, a
hardcoded `wandb_key` in `minimal_cbm/bin/train.py` (we force `WANDB_MODE=offline`).

---

## 2. Datasets & the visibility/occlusion axis

We train and analyze three dataset settings, each with both models:

| Setting | Classes | Why |
|---------|---------|-----|
| **FunnyBirds** | synthetic birds | renders give exact per-part pixel counts → ground-truth visibility. This is where we first observed the occlusion-vs-anchoring split. |
| **CUB-200** | 200 | the real-image benchmark from the CBM paper. 112 denoised attributes. |
| **CUB70** | first 70 test classes of CUB-200 | has **pixel-level part segmentation masks** (Behzadi-Khormouji & Oramas, WACV 2023, [CUB70-PartSegmentationDataset](https://github.com/hamedbehzadi/CUB70-PartSegmentationDataset)) → lets us port the FunnyBirds visibility analysis to real images. |

Data prep entry points (see `data/README.md` for paths):

- `data/cub/prepare_cub.md` — wraps the official `src/data_processing.py` +
  `src/generate_new_data.py` to produce `CUB_processed/class_attr_data_10`.
- `data/funnybirds/build_funnybirds_cbm_data.py` — reads the **official**
  `dataset_{mode}.json` + `parts.json` + part-map PNGs and emits the pickled-list
  format `CUBDataset` expects (so the *same* CBM trainer handles it), plus the
  FunnyBirds visibility table. `--labels image_level` zeroes occluded parts from
  the official part maps (**prof note #1 for FunnyBirds**). Concept schema and
  visibility both come from official files — no hand-written FunnyBirds module.
- `data/funnybirds/build_funnybirds_mcbm_data.py` — converts those pickles into the
  `minimal_cbm` CSV manifest + `concepts.json` (schema from `parts.json`).
- `data/cub70/build_cub70_visibility.py` — computes per-part pixel area from the
  CUB70 masks → `cub70_visibility.parquet` (the ground-truth visibility table).
- `data/cub70/relabel_cub_with_cub70.py` — applies a visibility threshold to flip
  `present→absent` labels that are actually occluded (**prof note #1, relabeling**).

---

## 3. Training (CBM and MCBM, all three settings)

All commands are in `train/`. They are thin wrappers over the official entry
points — we do **not** reimplement the training loop.

**CBM** (`external/ConceptBottleneck/src/experiments.py cub <Mode>`): we run the
three regimes the paper reports, and state which one we use in the paper because
they have different leakage characteristics:

- *Independent* — `Concept_XtoC` then `Independent_CtoY` (label head sees GT concepts)
- *Sequential*  — `Concept_XtoC` frozen, then `Sequential_CtoY` on its predictions
- *Joint*       — `Joint` end-to-end, `-attr_loss_weight λ` trades off the two losses

See `train/cbm_cub.sh`, `train/cbm_cub70.sh`, `train/cbm_funnybirds.sh`.

**MCBM** (`external/minimal_cbm/bin/train.py <config>.yaml -s <seed>`): driven by
YAML configs in `train/configs/`. The minimal regularizer (read straight from
the code) pulls each concept's latent `z` toward a fixed code:

```
z_logits = 6*c - 3          # c=0 -> -3 , c=1 -> +3   (NOT learned)
z_loss   = 0.2 * mean((z_logits - z)^2)
loss     = y_loss + beta*c_loss + gamma*z_loss
```

`gamma` = bottleneck/minimality strength, `beta` = concept-loss weight. The ±3
target is what keeps `z` off the `z=0`/sigmoid-0.5 ambiguous fixed point that
broke the earlier implementation — **watch the `z` distribution during training**
(notebook slot for this). ⚠️ Verify this against the paper's Appendix C before
the final run: the arXiv abstract describes a variational `q(z|c)` with a
mutual-information term, but the released code uses the fixed MSE-to-±3 form
above. Reconcile the two and state in the paper which you used.

See `train/mcbm_cub.sh`, `train/mcbm_cub70.sh`, `train/mcbm_funnybirds.sh`.

**CUB70 models** are trained two ways on purpose (they answer different questions):

1. *Full-200 model evaluated on the CUB70 subset* — tests whether **the model we
   actually report** stays grounded where a mask says a part is occluded (**prof note #2**).
2. *CUB70-only retrain on relabeled, visibility-aware labels* — a causal ablation:
   does cleaning the labels at train time fix grounding, or does it persist
   (**prof notes #3–#4**)? Item 1 cannot answer this.

---

## 4. Data-analysis notebooks (`notebooks/`)

Two notebooks, **publication quality** (every figure/table is paper-ready; cells
fill in numbers when run on adroit). These are dataset characterization +
training/grounding *validation* — the leakage/backwash **results** notebooks are
the explicit next step, not here.

- `01_funnybirds_data_and_validation.ipynb` — CBM + MCBM, 10 figure/table slots.
- `02_cub_data_and_validation.ipynb` — CBM + MCBM. **Part A** full CUB-200 (10
  slots) + **Part B** CUB70 occlusion/relabeling axis (~8 slots). Cap raised for
  this notebook because CUB70 is a genuinely separate analysis axis.

The exact slot list is in `notebooks/README.md`.

---

## 5. How this answers the professor's five notes

| Prof note | Where it is handled |
|-----------|---------------------|
| 1. Relabel on CBM | `data/cub70/relabel_cub_with_cub70.py` → notebook 02 Part B (relabel diagnostic) |
| 2. Is `z`≈1 in occluded relabeled-vs-labeled images | `analysis/occlusion.py` + notebook 02 Part B, on the **full-200** model |
| 3. New CBM/MCBM on CUB70 | `train/{cbm,mcbm}_cub70.sh` |
| 4. Repeat #2 on the CUB70-trained model | notebook 02 Part B, second pass |
| 5. Decide whether full segmentation is worth it | notebook 02 Part B final "verdict" table comparing the three conditions |

The logic, in one line: **before any swap/intervention result can be trusted, we
must show how much of a recall gap is real representation behavior vs. an
artifact of species-constant labels applied to images where the part is
occluded.** CUB70 masks give us the ground-truth visibility to measure that.
