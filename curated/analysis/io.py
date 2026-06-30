"""Normalized evaluation table: the single currency the notebooks consume.

Both frameworks dump per-image, per-concept inference into one long-format table
so all downstream plotting/occlusion code is framework-agnostic.

EVAL_SCHEMA (one row per image x concept):
    image          str   image stem (joins to the visibility tables)
    class_label    int
    concept_idx    int
    concept_name   str
    part           str   coarse body part (CUB) or group (FunnyBirds); may be ""
    z              float pre-sigmoid concept logit / latent (CBM: c_logit; MCBM: z)
    prob           float sigmoid(z)
    gt_label       int   ground-truth concept label (0/1)
    pred_label     int   prob >= 0.5
And image-level columns repeated on every row: y_true, y_pred (int).

`build_eval_table` is the one function to implement per framework -- it runs the
trained model over a split and emits the schema above. Stubs below document the
exact entry points; fill them in on the cluster where the checkpoints live.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import pandas as pd

EVAL_COLUMNS = [
    "image", "class_label", "concept_idx", "concept_name", "part",
    "z", "prob", "gt_label", "pred_label", "y_true", "y_pred",
]


def load_eval_table(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = set(EVAL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"eval table {path} missing columns: {sorted(missing)}")
    return df


def save_eval_table(df: pd.DataFrame, path: str | Path) -> None:
    df[EVAL_COLUMNS].to_parquet(path, index=False)


# ----------------------------------------------------------------------------
# Framework-specific builders (run on the cluster; documented entry points).
# ----------------------------------------------------------------------------
def build_eval_table_cbm(model_ckpt, data_dir, concept_names, part_of=None,
                         device="cuda") -> pd.DataFrame:
    """Run an official ConceptBottleneck checkpoint over a split.

    Implementation notes:
      * load the InceptionV3-based ModelXtoC from external/ConceptBottleneck
        (CUB/models.py / template_model.py) via curated.compat.add_cbm_to_path().
      * forward each image -> per-concept logits (the `z` here), sigmoid -> prob.
      * `part_of`: callable concept_name -> coarse part, e.g.
        curated.data.cub70.cub70_parts.attribute_to_part.
    Returns a DataFrame in EVAL_SCHEMA.
    """
    raise NotImplementedError(
        "Fill in on the cluster: load the x->c checkpoint and dump logits. "
        "See docstring; this is intentionally not run in the authoring env."
    )


def build_eval_table_mcbm(model_ckpt, config, concept_names, part_of=None,
                          device="cuda") -> pd.DataFrame:
    """Run a minimal_cbm MCBM checkpoint over a split.

    Use curated.compat.add_mcbm_to_path(); load MinimalConceptBottleneckModel;
    forward(x, c, sampling=False) -> dict with 'z' and 'c_logits'. Map 'z' to the
    schema's `z` column and sigmoid(c_logits) to `prob`.
    """
    raise NotImplementedError(
        "Fill in on the cluster: load the MCBM checkpoint and dump z/c_logits."
    )


def attach_part(df: pd.DataFrame, part_of) -> pd.DataFrame:
    """(Re)compute the `part` column from concept_name via a mapping callable."""
    df = df.copy()
    df["part"] = df["concept_name"].map(part_of).fillna("")
    return df
