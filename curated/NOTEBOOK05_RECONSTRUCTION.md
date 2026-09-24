# Notebook 05 reconstruction, 24 September 2026

Research is incomplete until real interventions are calibrated and reviewed. This
change implements the next experiment; it does not claim that CUB backwash or a
CUB relabeling benefit has been measured.

## Recovered method and working convention

The recovered conversation explicitly linked
https://arxiv.org/abs/2510.04180v1 (From Segments to Concepts). Its successor v2
is Spatially Grounded Concept-Based Image Classification. The region proposal
pipeline is CLIP ranking, GroundingDINO boxes, then SAM masks. The paper does
not supply a validated bird-part swap. The user's correction was to use released
CUB70 masks directly on covered images. No additional human annotator is a
prerequisite for using that release.

Notebook construction follows question, defined variables and predictions,
actual figure, literal observation, alternatives, discriminating test, limited
conclusion, and next question. Review every current figure. Do not prewrite new
observations. Existing numerical paragraphs and review ledgers are retained as
historical baseline provenance, not newly verified results.

## What changed

Markdown: added the annotation-state distinction, recovered segmentation source,
Koh loss-gradient derivation, competing explanations, paired edit controls, and
an explicit matched relabeling estimand. Retained the earlier evidence table.

Code: added archive inventory, optional local-weight paper-inspired segmentation,
silhouette-preserving replacement, label/mask-based pilot selection, frozen Koh
inference with original-export replay, blinded review, and outcome accounting.
The notebook can read the resulting reviewed predictions; absent evidence stays
incomplete rather than being plotted as zero.

The old pilot resized donor texture into the source silhouette. The new pilot
uses isotropic alignment and the donor silhouette. It fills exposed old-part
pixels from nearest unchanged context, which can create artifacts. Neither
pixel invariance nor self-reinsertion proves realism. Review pose, inserted value,
old-part removal, model-crop visibility, and co-changing attributes. The default
geometry matching is not a pose estimator. The small deterministic selection is
not a prevalence sample. All eight mapped CUB groups receive a coverage record.

The segmentation option needs user-supplied local pretrained snapshots. It does
not download weights or train. Use transformers 4.57.1 for the checked processor
API. CLIP ViT-B/16, top 32, maximum area 0.5 and merge IoU 0.5 follow the v2 CUB
settings. Minimum pixels, detector thresholds, exact detector/SAM snapshots and
greedy merge implementation are declared reconstruction choices. The paper's
classifier bag selection is not needed for masks and is not reproduced. Empty
detections do not establish physical absence.

## Established Adroit workflow

The existing workflow is Git push, run in the user's Adroit session, then return
outputs for review. No direct SSH access from the restored laptop is required.
Use an isolated checkout for this branch so the user's active checkout and other
research task remain untouched. Reuse existing curated_data and checkpoints.

Environment from recovered messages:

```bash
module load anaconda3/2025.6
conda activate cubvision-gpu
export CURATED_DATA=/scratch/network/cr7998/cv_emergence_project/curated_data
```

From the branch checkout, the small pilot is:

```bash
bash curated/notebooks/run_cub_swap_v2_pilot.sh
```

The runner displays fresh queue/accounting information when Slurm is available,
checks dependencies and accepted manifest hashes, then runs a CPU pilot. It does
not submit a job or train. CPU inference avoids assuming a GPU allocation on the
visualization host. Do not set CUB_SWAP_DEVICE=cuda without an appropriate
existing allocation. Output roots are unique and never replace accepted runs.

First return selection/SELECTION.json, all edits/*/review.png, and edit manifests.
Review the image grids before predictions. The generated blinded_review.csv
starts with every review flag false. Do not auto-accept it. Then inspect the
frozen inference report and scores. Set CUB_INTERVENTION_RESULTS to the reviewed
inference directory when rebuilding the full report.

## Validation and remaining evidence

Local synthetic checks cover gradient differentiation, response-state accounting,
missing-versus-empty masks, duplicate identities, unchanged context, donor shape,
self reconstruction, segmentation merging, and plan-to-four-image execution.
Notebook builder checks compile every code cell and preserve the existing model
and diagnostic boundaries. No real checkpoint inference or pretrained segmenter
inference has run on this restored machine: accepted models and photographs are
not present locally.

Next: run and inspect the small pilot, establish a credible positive control,
then quantify response and retained preference on accepted frozen models. In
parallel, join segmentation/visibility to training identities before designing
a relabeling experiment. No training is authorized by this reconstruction.
