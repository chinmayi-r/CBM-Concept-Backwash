#!/usr/bin/env python3
"""Validate our FunnyBirds concept vectors against the OFFICIAL indexing.

Run this ON ADROIT once, against the real FunnyBirds release, BEFORE trusting any
downstream training. It is the safety net for the one assumption we could not
check without the data: that a part variant is uniquely identified by
(model, color) and that our absent/placeholder handling matches the official
`FunnyBirds.single_params_to_part_idxs` logic.

The official loader (funnybirds-framework/datasets/funny_birds.py) builds, per
image, a `parts_specification[part]` dict from EVERY param key whose prefix is a
parts.json key (part = key.split('_')[0], attribute = key.split('_')[1]), then
does `self.parts[part].index(parts_specification[part])` -- an exact full-dict
match against the variant list. We replicate that here and compare, one-hot, to
`params_to_concept_vector`.

What this catches:
  * a part variant that needs MORE than (model, color) to be unique
    -> our concept vector would be ambiguous/wrong.
  * how 'placeholder' (absent part) is represented: whether parts.json contains a
    placeholder entry (official gives it an index) vs. our choice of all-zero
    group. We report the count so you can decide + document it.
  * any part whose parts.json key is not one of the 5 coarse parts, which would
    silently break the image_level occlusion labeling.

Exit code is non-zero if any hard mismatch (other than the documented
placeholder-as-absent choice) is found.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from collections import Counter, OrderedDict
from pathlib import Path

from funnybirds_concepts import (
    load_parts, build_part_lookup, params_to_concept_vector, concept_names,
    group_slices, COARSE_PARTS,
)


def official_onehot(parts, entry):
    """Replicate FunnyBirds.single_params_to_part_idxs, then one-hot it.

    Returns (vec, placeholder_parts) where placeholder_parts is the set of parts
    we treated as absent (all-zero group) because their model == 'placeholder'.
    Raises ValueError with a helpful message if a non-placeholder spec is not
    found in parts.json (that is exactly the (model,color)-insufficiency bug).
    """
    parts_keys = list(parts.keys())
    spec = OrderedDict()
    for key, val in entry.items():
        if "_" not in key:
            continue
        part = key.split("_")[0]
        attribute = key.split("_", 1)[1]
        if part in parts_keys:
            spec.setdefault(part, {})[attribute] = val

    spans = group_slices(parts)
    n = sum(len(v) for v in parts.values())
    vec = [0] * n
    placeholder_parts = set()
    for part in parts_keys:
        a, b = spans[part]
        s = spec.get(part)
        if not s or s.get("model") == "placeholder":
            placeholder_parts.add(part)
            continue  # absent -> all-zero group (our modeling choice)
        try:
            idx = parts[part].index(s)
        except ValueError:
            raise ValueError(
                f"part '{part}' spec {s} not found in parts.json variants "
                f"{parts[part]!r}. (model,color) is NOT sufficient to identify "
                f"the variant, OR parts.json entries carry extra attributes.")
        vec[a + idx] = 1
    return vec, placeholder_parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--funnybirds-root", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="check only the first N images per split (0 = all)")
    args = ap.parse_args()
    fb = Path(args.funnybirds_root)

    parts = load_parts(fb)
    lut = build_part_lookup(parts)
    names = concept_names(parts)
    widths = OrderedDict((p, len(v)) for p, v in parts.items())

    print(f"parts.json keys (order): {list(parts.keys())}")
    print(f"per-part variant counts:  {dict(widths)}")
    print(f"total concepts:           {len(names)}")

    # structural checks
    bad_keys = [p for p in parts if p not in COARSE_PARTS]
    if bad_keys:
        print(f"  !! parts.json has keys not in COARSE_PARTS {COARSE_PARTS}: "
              f"{bad_keys} -- image_level occlusion labeling would break.",
              file=sys.stderr)
    # how many attributes define a variant?
    attr_sets = {p: sorted({k for v in vs for k in v}) for p, vs in parts.items()}
    print(f"variant-defining attributes per part: {attr_sets}")
    extra = {p: a for p, a in attr_sets.items() if set(a) - {"model", "color"}}
    if extra:
        print(f"  !! parts with attributes beyond (model,color): {extra} -- our "
              f"params_to_concept_vector only uses (model,color).", file=sys.stderr)

    total = mismatches = 0
    placeholder_counts = Counter()
    hard_fail = bool(bad_keys or extra)

    for mode in ("train", "test"):
        dj = fb / f"dataset_{mode}.json"
        if not dj.exists():
            continue
        params = json.loads(dj.read_text())
        if args.limit:
            params = params[: args.limit]
        for i, entry in enumerate(params):
            ours = params_to_concept_vector(parts, lut, entry)
            try:
                ref, ph = official_onehot(parts, entry)
            except ValueError as e:
                print(f"  [{mode}#{i}] OFFICIAL INDEX FAILED: {e}", file=sys.stderr)
                hard_fail = True
                continue
            total += 1
            for p in ph:
                placeholder_counts[p] += 1
            if ours != ref:
                mismatches += 1
                if mismatches <= 10:
                    diff = [j for j in range(len(ref)) if ours[j] != ref[j]]
                    print(f"  [{mode}#{i}] MISMATCH at concept idxs {diff}: "
                          f"ours={[ours[j] for j in diff]} ref={[ref[j] for j in diff]}",
                          file=sys.stderr)

    print(f"\nchecked {total} images; {mismatches} concept-vector mismatches")
    if placeholder_counts:
        print(f"placeholder (absent) parts, per part: {dict(placeholder_counts)}")
        print("  -> these become all-zero concept groups. Confirm this matches how "
              "you want CBM to treat an absent part, and state it in the paper.")

    if mismatches or hard_fail:
        print("\nVALIDATION FAILED -- do not train until concept vectors match.",
              file=sys.stderr)
        sys.exit(1)
    print("\nVALIDATION OK -- concept vectors match the official indexing.")


if __name__ == "__main__":
    main()
