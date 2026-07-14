#!/usr/bin/env python3
"""Execute a notebook and dump EVERY figure to a PNG, so plots can be reviewed as files.

Usage:
    export CURATED_DATA=/scratch/network/cr7998/cv_emergence_project/curated_data
    python analysis/export_figs.py notebooks/03_funnybirds_mcbm.ipynb figs/03
    python analysis/export_figs.py notebooks/02_funnybirds_cbm.ipynb figs/02

Writes figs/<out>/NN_<title>.png (NN = code-cell index) plus an index.md listing them.
Cells with missing data just print [pending] and produce no image — that's fine, you see
whatever the current data supports. Needs: nbformat, nbconvert, and the analysis env
(pandas, matplotlib, torch). No renderer / GPU required (reads CSVs + cpu-mapped ckpts).
"""
import sys, re, base64
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

if len(sys.argv) < 2:
    sys.exit("usage: python analysis/export_figs.py <notebook.ipynb> [out_subdir]")
nb_path = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("figs") / nb_path.stem
out.mkdir(parents=True, exist_ok=True)

nb = nbformat.read(nb_path, as_version=4)
print(f"executing {nb_path.name} ...")
ep = ExecutePreprocessor(timeout=1800, kernel_name="python3", allow_errors=True)
ep.preprocess(nb, {"metadata": {"path": str(nb_path.parent)}})

def title_of(cell):
    m = re.search(r'set_title\(\s*[f]?["\']([^"\']{3,60})', "".join(cell.source))
    if not m:
        m = re.search(r'suptitle\(\s*[f]?["\']([^"\']{3,60})', "".join(cell.source))
    t = m.group(1) if m else "fig"
    return re.sub(r"[^a-zA-Z0-9]+", "-", t).strip("-")[:40]

lines, n = ["# executed figures\n"], 0
for ci, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    imgs = [o for o in cell.get("outputs", []) if "image/png" in o.get("data", {})]
    for k, o in enumerate(imgs):
        name = f"{ci:02d}_{title_of(cell)}{'' if k == 0 else '_'+str(k)}.png"
        (out / name).write_bytes(base64.b64decode(o["data"]["image/png"]))
        lines.append(f"- `{name}`  (code cell {ci})")
        n += 1
    # surface pending/error text so gaps are obvious
    for o in cell.get("outputs", []):
        txt = o.get("text", "") or "".join(o.get("traceback", []))
        if "[pending]" in txt or o.get("output_type") == "error":
            lines.append(f"  - cell {ci}: {txt.strip().splitlines()[0][:80]}")

(out / "index.md").write_text("\n".join(lines) + "\n")
print(f"wrote {n} figures to {out}/  (see {out}/index.md)")
