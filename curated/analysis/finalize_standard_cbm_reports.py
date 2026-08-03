#!/usr/bin/env python3
"""Assign stable IDs and image alternative text after notebook execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


CURATED=Path(__file__).resolve().parents[1]


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("notebook")
    args=ap.parse_args(); path=Path(args.notebook)
    if not path.is_absolute(): path=CURATED/path
    nb=json.loads(path.read_text(encoding="utf-8")); images=0
    for i,cell in enumerate(nb["cells"]):
        source="".join(cell.get("source",[]))
        cell.setdefault("id",hashlib.sha1(f"{path.name}:{i}:{source}".encode()).hexdigest()[:16])
        png=[o for o in cell.get("outputs",[]) if "image/png" in o.get("data",{})]
        if not png: continue
        match=re.search(r"^# ALT: (.+)$",source,re.MULTILINE)
        if not match: raise RuntimeError(f"plot cell {i} has no # ALT description")
        alt=match.group(1).strip(); cell.setdefault("metadata",{})["alt"]=alt
        for out in png:
            meta=out.setdefault("metadata",{}); meta["alt"]=alt
            meta.setdefault("image/png",{})["alt"]=alt; images+=1
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    print(f"[NOTEBOOK METADATA PASS] {path.name}: {len(nb['cells'])} cells, {images} described PNG outputs")


if __name__=="__main__": main()
