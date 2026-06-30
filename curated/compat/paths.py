"""Locate the vendored submodules and put them on sys.path."""
from __future__ import annotations
import sys
from pathlib import Path

CURATED_ROOT = Path(__file__).resolve().parents[1]
CBM_ROOT = CURATED_ROOT / "external" / "ConceptBottleneck"
MCBM_ROOT = CURATED_ROOT / "external" / "minimal_cbm"


def _ensure(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{name} submodule not found at {path}. "
            "Run `git submodule update --init --recursive` first."
        )
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)


def add_cbm_to_path() -> Path:
    """Make `import CUB...` resolve to the official ConceptBottleneck repo."""
    _ensure(CBM_ROOT, "ConceptBottleneck")
    return CBM_ROOT


def add_mcbm_to_path() -> Path:
    """Make `import src...` resolve to the official minimal_cbm repo."""
    _ensure(MCBM_ROOT, "minimal_cbm")
    return MCBM_ROOT
