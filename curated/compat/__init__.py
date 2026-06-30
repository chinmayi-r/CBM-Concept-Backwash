"""Import-time compatibility shims for the vendored reference repos.

Importing this package makes the two submodules importable and patches the few
deprecations that the 2020-era CBM code trips over, without editing anything in
external/. Use it at the top of our own scripts:

    from curated.compat import add_cbm_to_path, add_mcbm_to_path
    add_cbm_to_path()
    from CUB.dataset import CUBDataset
"""
from .paths import (
    CURATED_ROOT,
    CBM_ROOT,
    MCBM_ROOT,
    FUNNYBIRDS_ROOT,
    add_cbm_to_path,
    add_mcbm_to_path,
    add_funnybirds_to_path,
)
from . import numpy_compat  # noqa: F401  (applies aliases on import)

__all__ = [
    "CURATED_ROOT",
    "CBM_ROOT",
    "MCBM_ROOT",
    "FUNNYBIRDS_ROOT",
    "add_cbm_to_path",
    "add_mcbm_to_path",
    "add_funnybirds_to_path",
]
