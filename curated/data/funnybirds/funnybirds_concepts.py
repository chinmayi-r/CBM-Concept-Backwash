"""FunnyBirds concept schema, derived from the OFFICIAL dataset files.

We do not trust any hand-written FunnyBirds module. The concept structure is
read at runtime from the official `parts.json` (shipped in the FunnyBirds
release and used by funnybirds-framework's `datasets/funny_birds.py`), and the
per-image part visibility is read from the official part-map PNGs using the
exact color->part map defined in that loader.

parts.json structure:  { part_name: [ {"model":..., "color":...?}, ... ] }
dataset_{mode}.json:    [ { "class_idx": int, "<part>_model":..., "<part>_color":..? }, ... ]

A concept vector is the concatenation of one-hot groups, one group per part,
sized by the number of variants in parts.json. `placeholder` model => part
absent => all-zero group (matches the official semantics).
"""
from __future__ import annotations
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Official part-map color -> part-instance (from funnybirds-framework
# datasets/funny_birds.py). Two eyes/feet/wings are distinct instances.
# ---------------------------------------------------------------------------
PARTMAP_COLOR_TO_INSTANCE: "OrderedDict[Tuple[int,int,int], str]" = OrderedDict([
    ((255, 255, 253), "eye01"),
    ((255, 255, 254), "eye02"),
    ((255, 255, 0),   "beak"),
    ((255, 0, 1),     "foot01"),
    ((255, 0, 2),     "foot02"),
    ((0, 255, 1),     "wing01"),
    ((0, 255, 2),     "wing02"),
    ((0, 0, 255),     "tail"),
])

# part-instance -> coarse part (the granularity of the concept groups)
INSTANCE_TO_COARSE = {
    "eye01": "eye", "eye02": "eye",
    "beak": "beak",
    "foot01": "foot", "foot02": "foot",
    "wing01": "wing", "wing02": "wing",
    "tail": "tail",
}
COARSE_PARTS = ["beak", "eye", "wing", "foot", "tail"]


# ---------------------------------------------------------------------------
# Concept schema from parts.json
# ---------------------------------------------------------------------------
def load_parts(funnybirds_root: str | Path) -> "OrderedDict[str, list]":
    """Load parts.json preserving order."""
    p = Path(funnybirds_root) / "parts.json"
    with open(p) as f:
        return OrderedDict(json.load(f))


def part_variant_counts(parts: Dict[str, list]) -> "OrderedDict[str, int]":
    return OrderedDict((part, len(variants)) for part, variants in parts.items())


def concept_names(parts: Dict[str, list]) -> List[str]:
    names: List[str] = []
    for part, variants in parts.items():
        names.extend(f"{part}_{i}" for i in range(len(variants)))
    return names


def group_slices(parts: Dict[str, list]) -> "OrderedDict[str, Tuple[int, int]]":
    spans: "OrderedDict[str, Tuple[int, int]]" = OrderedDict()
    cur = 0
    for part, variants in parts.items():
        n = len(variants)
        spans[part] = (cur, cur + n)
        cur += n
    return spans


def _variant_key(vd: Dict[str, Any]) -> tuple:
    kf = {"model": vd["model"]}
    if "color" in vd:
        kf["color"] = vd["color"]
    return tuple(sorted(kf.items()))


def build_part_lookup(parts: Dict[str, list]) -> Dict[str, Dict[tuple, int]]:
    """part -> {variant-key -> index}, mirroring funny_birds.py semantics."""
    lut: Dict[str, Dict[tuple, int]] = {}
    for part, variants in parts.items():
        lut[part] = {_variant_key(vd): i for i, vd in enumerate(variants)}
    return lut


def params_to_concept_vector(parts: Dict[str, list],
                             lut: Dict[str, Dict[tuple, int]],
                             entry: Dict[str, Any]) -> List[int]:
    """One image's params dict -> flat one-hot concept vector. placeholder=absent."""
    vec: List[int] = []
    for part, variants in parts.items():
        n = len(variants)
        onehot = [0] * n
        model = entry.get(f"{part}_model", "placeholder")
        if model != "placeholder":
            kf = {"model": model}
            ck = f"{part}_color"
            if ck in entry:
                kf["color"] = entry[ck]
            idx = lut[part][tuple(sorted(kf.items()))]
            onehot[idx] = 1
        vec.extend(onehot)
    return vec
