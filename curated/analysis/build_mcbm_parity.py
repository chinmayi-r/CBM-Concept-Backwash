"""Notebook 02 figure inventory used by Notebook 03 source preflight.

The former version mechanically adapted every Standard figure once per MCBM
gamma. Notebook 03 no longer uses that construction. This module intentionally
contains only the stable inventory and exact-cell lookup shared by the real-input
preflight and tests.
"""
from __future__ import annotations

import re


TAGS = [
    "f1", "f2a", "f2b", "f3", "f3b", "f4", "f4b", "f5", "f6", "f6b",
    "f6c", "f7", "f7a", "f7b", "f7c", "f8", "f8b", "r8b-compare",
    "f8c-source", "f8d-source", "f9-new", "f9b", "f10",
    "app-evidence-correlation-code",
]

SHARED = {"f2a", "f2b", "f6b", "f7a"}


def find(cells: list[dict], tag: str) -> dict:
    """Return the one exact generated Notebook 02 code cell for ``tag``."""
    selected = [
        cell for cell in cells
        if cell.get("cell_type") == "code"
        and re.fullmatch(r"fb-" + re.escape(tag) + r"-[0-9a-f]+", cell.get("id", ""))
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Standard figure {tag}: expected one source cell, found {len(selected)}"
        )
    return selected[0]
