#!/usr/bin/env python3
"""Create and verify fail-closed manifests for canonical experiment stages."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_manifest(args: argparse.Namespace) -> None:
    outputs = [Path(p).resolve() for p in args.output]
    missing = [str(p) for p in outputs if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise SystemExit("required output missing or empty:\n" + "\n".join(missing))
    repo = Path(args.repo).resolve()
    metadata = dict(item.split("=", 1) for item in args.meta)
    manifest = {
        "status": "SUCCESS",
        "schema": 1,
        "stage": args.stage,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": args.command,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
        "repository_sha": git_sha(repo),
        "inputs": {
            str(Path(p).resolve()): sha256(Path(p).resolve()) for p in args.input
        },
        "outputs": {str(p): sha256(p) for p in outputs},
        "metadata": metadata,
    }
    framework = metadata.get("framework")
    if framework == "koh_joint":
        koh = repo / "curated/external/ConceptBottleneck"
        manifest.update({
            "koh_sha": git_sha(koh),
            "koh_tracked_diff_sha256": bytes_sha256(subprocess.check_output(
                ["git", "-C", str(koh), "diff", "--binary", "--", "."]
            )),
        })
    elif framework == "minimal_cbm":
        minimal = repo / "curated/external/minimal_cbm"
        manifest.update({
            "minimal_cbm_sha": git_sha(minimal),
            "minimal_cbm_tracked_diff_sha256": bytes_sha256(
                subprocess.check_output(
                    ["git", "-C", str(minimal), "diff", "--binary", "--", "."]
                )
            ),
            "declared_minimal_cbm_patch_sha256": sha256(
                repo / "curated/patches/minimal_cbm.patch"
            ),
        })
    else:
        raise SystemExit(f"unsupported manifest framework: {framework!r}")
    target = Path(args.manifest).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
    print(f"[MANIFEST SUCCESS] {target}")


def verify_manifest(args: argparse.Namespace) -> None:
    path = Path(args.manifest).resolve()
    data = json.loads(path.read_text())
    if data.get("status") != "SUCCESS":
        raise SystemExit(f"manifest is not successful: {path}")
    for name, expected in data.get("inputs", {}).items():
        source = Path(name)
        if not source.is_file() or sha256(source) != expected:
            raise SystemExit(f"manifest input mismatch: {source}")
    for name, expected in data.get("outputs", {}).items():
        output = Path(name)
        if not output.is_file() or sha256(output) != expected:
            raise SystemExit(f"manifest output mismatch: {output}")
    print(f"[MANIFEST VERIFIED] {path}")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="action", required=True)
    write = sub.add_parser("write")
    write.add_argument("--repo", required=True)
    write.add_argument("--stage", required=True)
    write.add_argument("--manifest", required=True)
    write.add_argument("--command", required=True)
    write.add_argument("--input", action="append", default=[])
    write.add_argument("--output", action="append", default=[], required=True)
    write.add_argument("--meta", action="append", default=[])
    check = sub.add_parser("verify")
    check.add_argument("--manifest", required=True)
    return ap


if __name__ == "__main__":
    ns = parser().parse_args()
    write_manifest(ns) if ns.action == "write" else verify_manifest(ns)
