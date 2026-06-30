#!/usr/bin/env python3
"""
curated/patches/run_cbm_funnybirds.py

Drop-in replacement for external/ConceptBottleneck/experiments.py that adds
FunnyBirds support. The official repo hardcodes N_CLASSES = 200 as a module
constant in CUB/config.py, and CUB/train.py does
`from CUB.config import ... N_CLASSES ...` at import time, then uses it
directly as `num_classes=N_CLASSES` when constructing every model (ModelXtoC,
ModelXtoY, ModelXtoCY, etc.). Since that import binds a copy of the name, the
only way to change it without editing external/ (forbidden per
curated/README.md) is to patch CUB.config.N_CLASSES *before* CUB.train (or
anything that imports it, including experiments.parse_arguments) is ever
imported.

`-n_attributes` is already a CLI flag in the official parser (default
N_ATTRIBUTES, overridable), so FunnyBirds' 26 concepts need no patch there --
just pass `-n_attributes 26` on the command line as cbm_funnybirds.sh already
does.

This wrapper otherwise replicates experiments.py's __main__ verbatim: parse
CUB args, seed, run_experiments(dataset, args). The image-path / CUB_200_2011
token trick that makes FunnyBirds images loadable by the unmodified
CUB/dataset.py is handled entirely by the symlink that
build_funnybirds_cbm_data.py creates -- this wrapper only deals with N_CLASSES.

Usage (run from inside external/ConceptBottleneck, same convention as
experiments.py):
    python3 ../../patches/run_cbm_funnybirds.py CUB Concept_XtoC <args...>
"""

from __future__ import annotations
import sys
from pathlib import Path

if "" not in sys.path:
    sys.path.insert(0, "")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import CUB.config as _cfg  # noqa: E402  (must precede CUB.train import)

_cfg.N_CLASSES = 50
_cfg.N_ATTRIBUTES = 26

# Only safe to import after the patch above is in place -- experiments.py's
# own parse_arguments() does `from CUB.train import parse_arguments`, which
# would otherwise bind the unpatched N_CLASSES = 200.
from experiments import parse_arguments, run_experiments  # noqa: E402

if __name__ == "__main__":
    import torch
    import numpy as np

    dataset, args = parse_arguments()

    np.random.seed(args[0].seed)
    torch.manual_seed(args[0].seed)

    run_experiments(dataset, args)
