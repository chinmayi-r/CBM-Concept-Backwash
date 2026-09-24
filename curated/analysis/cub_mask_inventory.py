"""Read-only inventory of released CUB70 masks; missing files are not occlusion."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np
from PIL import Image

PARTS = ("head", "left_eye", "right_eye", "beak", "neck", "body",
         "left_wing", "right_wing", "left_leg", "right_leg", "tail")


def parse_name(name):
    path = PurePosixPath(name)
    for part in sorted(PARTS, key=len, reverse=True):
        suffix = "_" + part + ".png"
        if path.name.endswith(suffix):
            return path.name[:-len(suffix)], part, path.parent.name
    raise ValueError(f"Unrecognized mask filename: {name}")


def read_masks(source):
    source = Path(source)
    if source.is_dir():
        for path in sorted(source.rglob("*.png")):
            yield path.relative_to(source).as_posix(), path.read_bytes()
    else:
        with tarfile.open(source, "r:*") as archive:
            for member in archive:
                if member.isfile() and member.name.lower().endswith(".png"):
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError(f"Unreadable mask: {member.name}")
                    yield member.name, stream.read()


def inventory(source):
    rows, images, seen = [], {}, set()
    digest = hashlib.sha256()
    for name, data in read_masks(source):
        image, part, folder = parse_name(name)
        if (image, part) in seen:
            raise ValueError(f"Duplicate image/part mask: {image}, {part}")
        seen.add((image, part))
        with Image.open(io.BytesIO(data)) as png:
            raw = np.asarray(png.convert("L"))
        foreground = raw > 0
        height, width = foreground.shape
        record = images.setdefault(image, {"shape": (height, width), "folder": folder,
                                          "parts": {}, "union": np.zeros_like(foreground),
                                          "overlap": np.zeros_like(foreground)})
        if record["shape"] != (height, width) or record["folder"] != folder:
            raise ValueError(f"Inconsistent image mask geometry/class folder: {image}")
        record["overlap"] |= record["union"] & foreground
        record["union"] |= foreground
        pixels = int(foreground.sum())
        record["parts"][part] = pixels
        mask_hash = hashlib.sha256(data).hexdigest()
        digest.update((name + "\0" + mask_hash + "\n").encode())
        rows.append({"image": image, "part": part, "released_folder": folder,
                     "height": height, "width": width, "positive_pixels": pixels,
                     "area_fraction": pixels / (height * width),
                     "nonbinary_pixels": int(((raw != 0) & (raw != 255)).sum()),
                     "sha256": mask_hash})
    if not images:
        raise ValueError("No recognized masks found")
    states = []
    for image, record in sorted(images.items()):
        for part in PARTS:
            pixels = record["parts"].get(part)
            states.append({"image": image, "part": part, "released_folder": record["folder"],
                           "annotation_state": "file_missing" if pixels is None else
                           ("file_empty" if pixels == 0 else "file_nonempty"),
                           "positive_pixels": pixels})
    summary = []
    for part in PARTS:
        records = [r for r in states if r["part"] == part]
        counts = {state: sum(r["annotation_state"] == state for r in records)
                  for state in ("file_missing", "file_empty", "file_nonempty")}
        summary.append({"part": part, "images_in_archive": len(images), **counts,
                        "file_missing_fraction": counts["file_missing"] / len(images)})
    manifest = {"status": "ACCEPTED FOR released-mask file inventory only",
                "source": str(Path(source).resolve()), "mask_count": len(rows),
                "image_count": len(images), "folder_count": len({r["folder"] for r in images.values()}),
                "ordered_mask_content_sha256": digest.hexdigest(),
                "images_with_overlapping_masks": sum(bool(r["overlap"].any()) for r in images.values()),
                "overlapping_pixels": sum(int(r["overlap"].sum()) for r in images.values()),
                "physical_visibility_verified": False,
                "train_test_identity_verified": False,
                "boundary": "Missing files are unknown annotation state, not confirmed physical absence. "
                            "Denominator includes only images represented by at least one archived mask. "
                            "Training coverage requires joining the official split by exact image identity."}
    return rows, states, summary, manifest


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Preserve previous inventory: output already exists: {args.out}")
    rows, states, summary, manifest = inventory(args.source)
    args.out.mkdir(parents=True)
    write_csv(args.out / "mask_files.csv", rows)
    write_csv(args.out / "annotation_states.csv", states)
    write_csv(args.out / "part_inventory.csv", summary)
    (args.out / "INVENTORY.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
