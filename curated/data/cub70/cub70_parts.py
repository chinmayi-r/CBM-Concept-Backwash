"""CUB70 part inventory and the CUB-attribute -> body-part mapping.

CUB70 (Behzadi-Khormouji & Oramas, WACV 2023) provides non-overlapping
pixel-wise masks for 11 parts on the first 70 *test* classes of CUB-200-2011:

    head, right eye, left eye, beak, neck, body,
    right wing, left wing, right leg, left leg, tail

To relabel CUB attributes by visibility we must know which body part each
attribute describes. CUB attribute names look like `has_bill_shape::...`,
`has_wing_color::...`, `has_tail_pattern::...`; the location token after `has_`
identifies the part. The mapping below is the documented default -- VERIFY it
against attributes.txt for your run and edit `TOKEN_TO_PART` only here.
"""
from __future__ import annotations
import re
from typing import Optional

CUB70_PARTS = [
    "head", "right_eye", "left_eye", "beak", "neck", "body",
    "right_wing", "left_wing", "right_leg", "left_leg", "tail",
]

# Coarse parts used for attribute mapping (left/right merged, since a CUB
# attribute like "eye color" is not lateralized).
COARSE_PARTS = ["head", "eye", "beak", "neck", "body", "wing", "leg", "tail"]

# CUB attribute location token  ->  coarse part
TOKEN_TO_PART = {
    "bill": "beak",
    "eye": "eye",
    "forehead": "head", "crown": "head", "nape": "head", "head": "head",
    "wing": "wing", "primary": "wing",
    "tail": "tail",
    "leg": "leg",
    "throat": "neck",
    "back": "body", "breast": "body", "belly": "body",
    "underparts": "body", "upperparts": "body",
    "size": "body", "shape": "body",
}

# Merge each coarse part down to the CUB70 masks that cover it (sum their areas).
COARSE_TO_CUB70 = {
    "head": ["head"],
    "eye": ["right_eye", "left_eye"],
    "beak": ["beak"],
    "neck": ["neck"],
    "body": ["body"],
    "wing": ["right_wing", "left_wing"],
    "leg": ["right_leg", "left_leg"],
    "tail": ["tail"],
}

_HAS = re.compile(r"^has_([a-z]+)")


def attribute_to_part(attr_name: str) -> Optional[str]:
    """Map a CUB attribute name to a coarse body part, or None if unmapped."""
    m = _HAS.match(attr_name.strip().lower())
    if not m:
        return None
    return TOKEN_TO_PART.get(m.group(1))
