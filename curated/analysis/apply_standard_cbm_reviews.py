#!/usr/bin/env python3
"""Install visually reviewed markdown without discarding executed outputs."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import build_standard_cbm_reports as reports


NOTEBOOKS = {
    "02_funnybirds_cbm.ipynb": reports.build_funnybird,
    "05_cub_cbm.ipynb": reports.build_cub,
}


def stable_id(cell: dict) -> str:
    cell_id = cell.get("id", "")
    if "-" not in cell_id:
        raise RuntimeError(f"cell lacks a generated stable id: {cell_id!r}")
    return cell_id.rsplit("-", 1)[0]


def install(path: Path, build) -> None:
    executed = json.loads(path.read_text(encoding="utf-8"))
    fresh = build()

    old_code = {
        stable_id(cell): cell
        for cell in executed["cells"]
        if cell.get("cell_type") == "code"
    }
    fresh_code = [
        cell for cell in fresh["cells"] if cell.get("cell_type") == "code"
    ]
    if len(old_code) != len(fresh_code):
        raise RuntimeError(
            f"code-cell count changed for {path}: executed={len(old_code)}, "
            f"fresh={len(fresh_code)}"
        )

    for cell in fresh_code:
        key = stable_id(cell)
        if key not in old_code:
            raise RuntimeError(f"new code cell {key} has no executed counterpart")
        old = old_code[key]
        if "".join(cell.get("source", [])) != "".join(old.get("source", [])):
            raise RuntimeError(
                f"code changed for {key}; rerun is required instead of copying outputs"
            )
        cell["execution_count"] = old.get("execution_count")
        cell["outputs"] = copy.deepcopy(old.get("outputs", []))
        cell["metadata"] = copy.deepcopy(old.get("metadata", {}))

    old_pngs = sum(
        "image/png" in output.get("data", {})
        for cell in executed["cells"]
        for output in cell.get("outputs", [])
    )
    new_pngs = sum(
        "image/png" in output.get("data", {})
        for cell in fresh["cells"]
        for output in cell.get("outputs", [])
    )
    if old_pngs != new_pngs or old_pngs == 0:
        raise RuntimeError(
            f"PNG preservation error for {path}: old={old_pngs}, new={new_pngs}"
        )

    fresh["metadata"] = copy.deepcopy(executed.get("metadata", {}))
    path.write_text(
        json.dumps(fresh, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(
        f"[REVIEW INSTALL PASS] {path}: {len(fresh['cells'])} cells, "
        f"{new_pngs} PNG outputs preserved"
    )


def main() -> None:
    for name, build in NOTEBOOKS.items():
        install(reports.NOTEBOOKS / name, build)


if __name__ == "__main__":
    main()
