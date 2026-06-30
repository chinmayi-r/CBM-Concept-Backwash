"""Shared analysis utilities for the data-analysis notebooks.

Both notebooks consume one normalized table (see io.EVAL_SCHEMA) regardless of
whether a model came from the CBM or the MCBM framework, so the same plotting and
occlusion code serves both. Keeping this logic here (not in the notebooks) is why
the notebooks stay short and paper-grade.
"""
from . import io, plotting, occlusion  # noqa: F401

__all__ = ["io", "plotting", "occlusion"]
