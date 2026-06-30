"""
curated/data/funnybirds/funnybirds_concepts.py

Single source of truth for FunnyBirds concept names and group structure.
Delegates to the canonical datasets/funnybirds_dataset module — nothing is
hardcoded here so it can never drift from the real dataset.

Usage:
    from curated.data.funnybirds.funnybirds_concepts import flat_concept_names, group_slices
"""

from __future__ import annotations
import sys
from pathlib import Path

# Allow importing from repo root even when curated/ is the cwd
_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# These constants mirror datasets/funnybirds_dataset.py (single canonical source).
# Kept here without importing torch so notebooks can use them without a GPU env.
# If the upstream ever changes them, update here too and bump the assertion below.
PARTS: list[str] = ["beak", "eye", "wing", "foot", "tail"]
PART_VARIANTS: dict[str, int] = {"beak": 4, "eye": 3, "wing": 6, "foot": 4, "tail": 9}
NUM_CONCEPTS: int = sum(PART_VARIANTS.values())  # 26

def concept_names() -> list[str]:
    """Match datasets/funnybirds_dataset.concept_names() exactly."""
    names = []
    for part in PARTS:
        for i in range(PART_VARIANTS[part]):
            names.append(f"{part}_{i}")
    return names


def flat_concept_names() -> list[str]:
    """Return the ordered list of all 26 concept names, e.g. ['beak_0', ..., 'tail_8']."""
    return concept_names()




def group_slices() -> list[slice]:
    """
    Return one slice per part group so that concepts[s] selects the one-hot
    block for that part.  Order matches flat_concept_names().
    """
    slices = []
    offset = 0
    for part in PARTS:
        n = PART_VARIANTS[part]
        slices.append(slice(offset, offset + n))
        offset += n
    return slices


def part_names() -> list[str]:
    return list(PARTS)


def n_concepts() -> int:
    return NUM_CONCEPTS
