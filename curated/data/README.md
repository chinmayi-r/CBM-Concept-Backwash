# Data layout

Set one environment variable and point it at a directory with the raw datasets;
every script reads from it. On adroit this is typically a `/scratch` path.

```bash
export CURATED_DATA=/scratch/<you>/curated_data
```

Expected raw inputs:

```
$CURATED_DATA/
  CUB_200_2011/                 # official CUB-200-2011 release (images, attributes, parts, bounding_boxes)
  CUB_processed/                # produced by data/cub/prepare_cub.md (pickled lists)
    class_attr_data_10/{train,val,test}.pkl
  FunnyBirds/                   # official FunnyBirds release (FunnyBirds.zip), used as-is:
    dataset_train.json  dataset_test.json  classes.json  parts.json
    {train,test}/{class_idx}/{idx:06d}.png            # input images
    {train,test}_part_map/{class_idx}/{idx:06d}.png   # part-map segmentation (official visibility)
  cub70/                        # CUB70-PartSegmentationDataset (per-image, per-part PNG masks for first 70 test classes)
```

Produced artifacts (written back under `$CURATED_DATA`):

```
  funnybirds_processed/         # build_funnybirds_cbm_data.py  -> CBM pickled lists
  funnybirds_visibility.parquet # build_funnybirds_cbm_data.py  -> per-image coarse-part pixel counts (official part maps)
  funnybirds_mcbm/              # build_funnybirds_mcbm_data.py -> MCBM CSV manifest + concepts.json
  cub70_visibility.parquet      # build_cub70_visibility.py     -> per-image,per-part mask area + visible flag
  CUB_processed/class_attr_data_10_cub70_eval_relabeled/  # test-only evaluation diagnostic
```

### CBM pickled-list format (what `CUBDataset` expects)

Each split pickle is a `list[dict]`; one dict per image:

```python
{
  "id": int,
  "img_path": str,                 # absolute or remappable via -data_dir2
  "class_label": int,              # 0-indexed class
  "attribute_label": list[int],    # length n_attributes, 0/1
  "attribute_certainty": list[int],# CUB certainty 1..4 (use 4 for synthetic/known)
}
```

Our FunnyBirds and CUB70 builders emit exactly this so the **same** official CBM
trainer consumes all three datasets unchanged.
