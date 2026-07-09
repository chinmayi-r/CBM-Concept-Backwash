# Preparing CUB-200 for the CBM trainer

We use the **official** processing scripts unchanged. Run inside the `cbm` env
from the submodule root.

```bash
conda activate cbm
cd curated/external/ConceptBottleneck
export CURATED_DATA=/scratch/<you>/curated_data

# 1. raw CUB-200-2011 must be at $CURATED_DATA/CUB_200_2011
# 2. official processing -> pickled train/val/test lists + the denoised 112-attr set
python3 src/data_processing.py   --save_dir $CURATED_DATA/CUB_processed --data_dir $CURATED_DATA/CUB_200_2011
python3 src/generate_new_data.py # produces class_attr_data_10 (majority-vote class-level attrs, 112 attrs >=10 classes)
```

Result: `$CURATED_DATA/CUB_processed/class_attr_data_10/{train,val,test}.pkl`,
the input to every CUB CBM command in `train/cbm_cub.sh`.

> Check the exact flags/paths in `external/ConceptBottleneck/CUB/README.md` and
> `src/data_processing.py --help`; they are the authority. The two lines above
> mirror that README. `-n_attributes 112` everywhere downstream refers to this
> denoised set.

## MCBM on CUB

minimal_cbm reads images + concept vectors via its own loaders. Build a CSV
manifest from the same processed pickles so both models train on identical
labels:

```bash
python3 curated/data/cub/build_cub_mcbm_data.py   # writes $CURATED_DATA/cub_mcbm/{train,val,test}.csv
```

(That builder is the CUB analogue of
`data/funnybirds/build_funnybirds_mcbm_data.py`; it reads
`CUB_processed/class_attr_data_10/*.pkl` and emits the 112-wide concept CSV.)
