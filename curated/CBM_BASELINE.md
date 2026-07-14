# CBM baseline — `fb_cbm_renderer_swap_v2.ipynb` (86 cells) walked cell by cell

The original CBM notebook = the **baseline** the curated nb02 reproduces on official
`minimal_cbm`. Status: ✅ ported · ✏️ modified · ✂️ replaced · ❌ dropped · 🔁 moved to
SLURM driver (`z_ordering_swap.py` + `renderer_swap.slurm`).

## Setup — cells 0–23
| cell | purpose | curated |
|---|---|---|
| 0 | title / experiment description | — |
| 1 | imports (gc, torch, numpy, matplotlib, requests…) | 🔁 driver + nb02 setup cell |
| 2–3 | §0 paths: `ROOT`, CBM checkpoint, features, FunnyBirds root | 🔁 driver reads from `$CURATED_DATA` |
| 4–5 | §1 `load_species_maps` — species↔variant index from `dataset_test.json` | 🔁 driver (same logic) |
| 6 | `safe_torch_load` — robust checkpoint loader | ✂️ replaced by curated `load_model` (official ckpt format) |
| 7–9 | §2 concept names, class×concept matrix, `load_meta` | ✏️ driver builds these from `parts.json`/`funnybirds_concepts` |
| 10–12 | §3 load **old CBM weights** + precomputed avgpool features | ✂️ replaced — curated loads the official minimal_cbm CBM; no precomputed feature dump |
| 13–17 | §4 rendering params: part params per species, **tail variant distribution** (how many species share each tail) | ✏️ concept/variant maps built in the driver; the "variants per part" count → nb02 §2 (`n_variants`) |
| 18–19 | §5 renderer present-check | 🔁 `renderer_swap.slurm` starts the Node server |
| 20 | `json_to_url` (verbatim from FunnyBirds `create_dataset.py`) | 🔁 driver |
| 21 | start/stop the Node renderer process | 🔁 slurm |
| 22 | eval transform (must match `train_cbm_funnybirds.py`) | 🔁 driver (`_MEAN/_STD` + resize) |
| 23 | **Smoke test A** — stored image z == precomputed z (sanity) | ✅ kept as an assert in the driver |

## Part-swap helpers + visual check — cells 24–29
| cell | purpose | curated |
|---|---|---|
| 24–25 | §6 `swap_part_in_ann` — replace one part's params with a donor's | 🔁 driver |
| 26 | `swap_record_from_ann` — render swapped image, read z, record margins | 🔁 driver (core of the swap) |
| 27 | tuneable occlusion threshold (skip occluded images) | ✅ = curated visible-only gate (`pixel_count`/`changed_frac`) |
| 28–29 | **visual check grid** — orig / swap / delete per part | ✅ nb02 §6c inspection grid |

## Z-ordering experiment — cells 30–41 (the core)
| cell | purpose | curated |
|---|---|---|
| 30–31 | §7 `z_ordering_record` — for a species pair, swap a part both ways, record `z_old/z_new/margin/ordering_correct` | 🔁 driver (this IS the swap metric) |
| 32–36 | §7a **tail primary experiment** — run the swap for tail across species pairs; first result table | 🔁 driver → nb02 §6 (tail row) |
| 37–38 | §7b **extend to all parts** — same swap for wing/beak/foot/eye | 🔁 driver → nb02 §6 (all parts) |
| 39 | per-part `frac_correct` summary table | ✅ nb02 §6 (the fwd/bwd bars) |
| 40 | per-part mean margin | ✅ nb02 §6 / box plot §E |
| 41 | fwd vs bwd split (the direction breakdown) | ✅ nb02 §6 fwd/bwd bars (kept separate from the start) |

## Concept grounding before swap — cells 42–45
| cell | purpose | curated |
|---|---|---|
| 42–43 | **on-distribution grounding**: source concept (present) vs donor concept (absent) on the *original* image | ✅ nb02 §5 ("donor OFF when part absent") |
| 44 | scatter z_source vs z_donor (backwash = donor rides up) | ✏️ nb02 §8b (kept, colored by part not outcome) |
| 45 | margin distribution before swap | ✅ folded into §5 / §8b |

## Good vs failing tail variants — cells 46–49
| cell | purpose | curated |
|---|---|---|
| 46–47 | split tail variants into "good" vs "failing" by median `frac_correct` | ✂️ replaced by nb02 §8c **ranked** top-20 (no median split — the "arbitrary dividing line" you flagged) |
| 48 | per-variant `frac_correct` bars | ✂️ → §8c |
| 49 | hard-coded notes on which variants fail | ❌ dropped (anecdotal) |

## Concept confusion for tail violations — cells 50–54
| cell | purpose | curated |
|---|---|---|
| 50–52 | **tail confusion matrix**: on a mis-ordered swap, which tail concept fires (argmax); `is_anchoring` flag | ✅ nb02 §8 confusion matrix (all tail swaps; diagonal=grounded) |
| 53–54 | per-variant confusion detail / anchoring rate | ✏️ folded into §8 (the matrix + caption). Note: **currently tail-only**; driver now logs all parts → re-run gives the all-part version |

## Source species breakdown — cells 55–61
| cell | purpose | curated |
|---|---|---|
| 55–57 | **per-source-species violation rate** (tail; also beak/wing) | ✅ nb02 §8d (single color; median-split coloring dropped) |
| 58–60 | worst donor species per source; per-species detail | ✏️ condensed into §8d |
| 61 | worst variant-pair violations | ✂️ → §8c top-20 concept slots |

## Diagnostic plots — cells 62–71
| cell | purpose | curated |
|---|---|---|
| 62–64 | **visual swap inspection grid** (n_src × n_donor renders) + worst-donor finder | ✅ nb02 §6c grid |
| 65–66 | §9 header + **downstream probability effect** (does the margin move P(species)?) | ✏️ nb02 §8f (de-tautologized: binned mean + colored by visibility, not margin sign) |
| 67–68 | downstream scatter / regression | ✏️ §8f |
| 69 | print summary columns | ❌ dropped (debug) |
| 70–71 | **violation rate per concept, top-20 worst** | ✅ nb02 §8c |

## Part 2 — occlusion analysis — cells 72–84
| cell | purpose | curated |
|---|---|---|
| 72–74 | §8 header + `z_ordering_record_v2` — same swap but also records `pixel_count_cf` (part-map area) | 🔁 driver v2 (writes `pixel_count_cf`); = curated occlusion control data |
| 75–77 | rerun tail + all parts with v2 (visibility) | 🔁 driver (`--no-v2` off) |
| 78–79 | pixel-count histograms per part | ✅ nb02 §8e visibility per part |
| 80–82 | **before/after occlusion filter** (px ≥ threshold), tail vs others | ✅ nb02 §7 occlusion control + §35 all-parts binned |
| 83–84 | visibility vs margin scatter / by outcome | ✅ nb02 §8 (visibility vs margin + by-outcome hist) |
| 85 | empty | — |

## Baseline vs curated — what changed and why (the honest diffs)
1. **Model**: old hand-loaded CBM (`safe_torch_load` + precomputed features) → **official minimal_cbm** CBM. The whole point of the curated restart.
2. **Heavy compute → SLURM**: all rendering/swap (cells 20–31, 74) moved to `z_ordering_swap.py` + `renderer_swap.slurm`; the notebook only reads CSVs.
3. **Median-split "good vs failing" (46–49) dropped** → ranked top-20; no arbitrary dividing line.
4. **Downstream scatter (65–68) de-tautologized** → binned mean, colored by an independent variable.
5. **Curated ADDS what the baseline lacks**: deletion grounding (§1), species probe (§2), per-species leakage (§3c), the objection-first captions, and the `retained_frac` naming discipline.
6. **Still tail-only in curated**: the confusion matrix (50–54) — driver now logs all parts, needs a swap re-run for the all-part version.
