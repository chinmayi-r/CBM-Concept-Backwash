#!/usr/bin/env python3
"""Copy stored notebook figure descriptions into nbconvert HTML image tags.

Nbconvert's standard lab/classic templates currently emit code-output PNGs
without ``alt`` attributes even when the output or cell metadata contains a
description.  The HTML exporter then inserts a generic placeholder and warns.
This postprocessor replaces only those generic placeholders, in notebook output
order, and fails closed if image counts or descriptions do not match.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path


PLACEHOLDER = 'alt="No description has been provided for this image"'
IMAGE_MIMES = ("image/png", "image/jpeg", "image/svg+xml")


def image_descriptions(notebook: Path) -> list[str]:
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    descriptions: list[str] = []
    for cell in nb.get("cells", []):
        cell_alt = str(cell.get("metadata", {}).get("alt", "")).strip()
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            metadata = output.get("metadata", {})
            for mime in IMAGE_MIMES:
                if mime not in data:
                    continue
                mime_metadata = metadata.get(mime, {})
                mime_alt = (
                    mime_metadata.get("alt", "")
                    if isinstance(mime_metadata, dict)
                    else ""
                )
                alt = str(mime_alt or metadata.get("alt", "") or cell_alt).strip()
                if not alt:
                    raise RuntimeError(
                        f"image output in cell {cell.get('id')!r} has no description"
                    )
                descriptions.append(alt)
    return descriptions


def repair(notebook: Path, html_path: Path) -> None:
    descriptions = image_descriptions(notebook)
    rendered = html_path.read_text(encoding="utf-8")
    placeholders = rendered.count(PLACEHOLDER)
    if placeholders != len(descriptions):
        raise RuntimeError(
            f"image/description mismatch for {html_path}: "
            f"HTML placeholders={placeholders}, notebook descriptions={len(descriptions)}"
        )

    iterator = iter(descriptions)
    rendered = re.sub(
        re.escape(PLACEHOLDER),
        lambda _match: f'alt="{html.escape(next(iterator), quote=True)}"',
        rendered,
    )
    if PLACEHOLDER in rendered:
        raise RuntimeError(f"generic image description remains in {html_path}")
    html_path.write_text(rendered, encoding="utf-8")
    print(f"[HTML ALT-TEXT PASS] {html_path}: {len(descriptions)} images described")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("html", type=Path)
    args = parser.parse_args()
    repair(args.notebook, args.html)


if __name__ == "__main__":
    main()
