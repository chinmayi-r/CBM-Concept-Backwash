"""Restore NumPy aliases removed in 1.24 that 2020-era code expects.

No-op when the pinned numpy=1.23 is used, but keeps the code robust if a newer
NumPy sneaks into the env. Never silences a real error -- only re-adds the exact
aliases NumPy itself documented as equivalent to the builtins.
"""
import numpy as np

for _alias, _builtin in (("int", int), ("float", float), ("bool", bool),
                         ("object", object), ("str", str)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _builtin)
