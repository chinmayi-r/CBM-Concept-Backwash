# src/datasets/cub_metadata.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class CUBMetadata:
    images: pd.DataFrame
    parts: pd.DataFrame
    part_locs: pd.DataFrame
    classes: pd.DataFrame

    # Optional part-level metadata
    image_parts_binary: Optional[pd.DataFrame] = None

    # NEW: attribute-level metadata
    attributes: Optional[pd.DataFrame] = None              # one row per attribute_id
    image_attributes_binary: Optional[pd.DataFrame] = None # one row per image_id


def _load_attributes(cub_root: Path) -> Optional[pd.DataFrame]:
    """
    Load attribute definitions.

    We first try the standard CUB path:
        <cub_root>/attributes/attributes.txt

    In your repo, attributes.txt lives in <cub_root>/../attributes.txt (e.g. data/attributes.txt),
    so we also try that as a fallback.
    """
    # Standard CUB location
    attrs_txt = cub_root / "attributes" / "attributes.txt"

    # If not there, try parent directory (your data/attributes.txt)
    if not attrs_txt.exists():
        alt = cub_root.parent / "attributes.txt"
        if alt.exists():
            attrs_txt = alt
        else:
            # No attribute definition file found
            return None

    attrs = pd.read_csv(
        attrs_txt,
        sep=r"\s+",
        header=None,
        names=["attribute_id", "attribute_name"],
    )
    attrs["attribute_id"] = attrs["attribute_id"].astype(int)

    # Split "has_bill_color::black" -> group="has_bill_color", value="black"
    split = attrs["attribute_name"].str.split("::", n=1, expand=True)
    attrs["group"] = split[0]
    attrs["value"] = split[1]

    return attrs



def _load_image_attributes_long(cub_root: Path) -> Optional[pd.DataFrame]:
    """
    Load image-level attribute labels in long format from
    attributes/image_attribute_labels.txt.

    We DO NOT rely on pandas.read_csv here because the file sometimes has
    6 numeric fields per line instead of 5.

    We manually split each line on whitespace and keep:
      image_id, attribute_id, is_present, certainty, time_ms

    If fewer than 3 fields are present, the line is skipped.
    """
    img_attr_txt = cub_root / "attributes" / "image_attribute_labels.txt"
    if not img_attr_txt.exists():
        return None

    rows = []
    with img_attr_txt.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            # We need at least image_id, attribute_id, is_present
            if len(parts) < 3:
                continue

            try:
                image_id = int(parts[0])
                attribute_id = int(parts[1])
                is_present = int(float(parts[2]))  # 0/1 (or >0 treated as present)
            except ValueError:
                # Skip malformed lines
                continue

            # Optional fields
            certainty = int(float(parts[3])) if len(parts) > 3 else 0
            time_ms = float(parts[4]) if len(parts) > 4 else 0.0

            rows.append(
                {
                    "image_id": image_id,
                    "attribute_id": attribute_id,
                    "is_present": 1 if is_present > 0 else 0,
                    "certainty": certainty,
                    "time_ms": time_ms,
                }
            )

    if not rows:
        return None

    df = pd.DataFrame(rows)
    return df



def _make_image_attributes_binary(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long-form attribute annotations into a wide binary table:

      columns: image_id, attr_1_present, ..., attr_312_present
    """
    wide = df_long.pivot_table(
        index="image_id",
        columns="attribute_id",
        values="is_present",
        aggfunc="max",
        fill_value=0,
    )

    wide = wide.reset_index()
    wide.columns = [
        "image_id" if c == "image_id" else f"attr_{int(c)}_present"
        for c in wide.columns
    ]
    return wide


def load_cub_metadata(cub_root: str | Path) -> CUBMetadata:
    """
    Load CUB metadata from preprocessed CSVs in cub_root / 'metadata'
    AND parse original attributes files in cub_root / 'attributes'.

    Assumes you already created:

      metadata/images.csv
      metadata/parts.csv
      metadata/part_locs.csv
      metadata/classes.csv
      [optional] metadata/image_parts_binary.csv
    """
    cub_root = Path(cub_root)
    meta_dir = cub_root / "metadata"

    images = pd.read_csv(meta_dir / "images.csv")
    parts = pd.read_csv(meta_dir / "parts.csv")
    part_locs = pd.read_csv(meta_dir / "part_locs.csv")
    classes = pd.read_csv(meta_dir / "classes.csv")

    image_parts_binary = None
    ipb_path = meta_dir / "image_parts_binary.csv"
    if ipb_path.exists():
        image_parts_binary = pd.read_csv(ipb_path)

    # --- NEW: attributes ---
    attributes = _load_attributes(cub_root)
    image_attributes_binary = None

    img_attr_long = _load_image_attributes_long(cub_root)
    if img_attr_long is not None:
        image_attributes_binary = _make_image_attributes_binary(img_attr_long)

    return CUBMetadata(
        images=images,
        parts=parts,
        part_locs=part_locs,
        classes=classes,
        image_parts_binary=image_parts_binary,
        attributes=attributes,
        image_attributes_binary=image_attributes_binary,
    )
