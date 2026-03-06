#!/usr/bin/env python
"""
Prepare structured metadata CSVs for the CUB dataset.

Input:  CUB root directory containing:
        - classes.txt
        - images.txt
        - image_class_labels.txt
        - train_test_split.txt
        - bounding_boxes.txt
        - parts/parts.txt
        - parts/part_locs.txt

Output:
        - images.csv
        - classes.csv
        - parts.csv
        - part_locs.csv
        - image_parts_binary.csv  (image-level binary presence for each part)
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path


def read_classes(classes_path: Path) -> dict:
    classes = {}
    with classes_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            class_id_str, class_name = line.split(" ", 1)
            classes[int(class_id_str)] = class_name
    return classes


def read_images(images_path: Path) -> dict:
    images = {}
    with images_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_id_str, rel_path = line.split(" ", 1)
            images[int(image_id_str)] = rel_path
    return images


def read_image_class_labels(labels_path: Path) -> dict:
    mapping = {}
    with labels_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_id_str, class_id_str = line.split()
            mapping[int(image_id_str)] = int(class_id_str)
    return mapping


def read_train_test_split(split_path: Path) -> dict:
    mapping = {}
    with split_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_id_str, is_train_str = line.split()
            mapping[int(image_id_str)] = bool(int(is_train_str))
    return mapping


def read_bounding_boxes(bbox_path: Path) -> dict:
    mapping = {}
    with bbox_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            image_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            mapping[image_id] = (x, y, w, h)
    return mapping


def read_parts(parts_txt_path: Path) -> dict:
    parts = {}
    with parts_txt_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            part_id_str, part_name = line.split(" ", 1)
            parts[int(part_id_str)] = part_name
    return parts


def read_part_locs(part_locs_path: Path) -> list:
    rows = []
    with part_locs_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_id_str, part_id_str, x_str, y_str, vis_str = line.split()
            rows.append(
                {
                    "image_id": int(image_id_str),
                    "part_id": int(part_id_str),
                    "x": float(x_str),
                    "y": float(y_str),
                    "visible": int(vis_str),
                }
            )
    return rows


def write_csv(path: Path, fieldnames: list, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[prepare_cub_metadata] wrote {len(rows)} rows to {path}")


def build_image_and_class_metadata(cub_root: Path, out_dir: Path):
    classes = read_classes(cub_root / "classes.txt")
    images = read_images(cub_root / "images.txt")
    img_to_class = read_image_class_labels(cub_root / "image_class_labels.txt")
    split = read_train_test_split(cub_root / "train_test_split.txt")
    bboxes = read_bounding_boxes(cub_root / "bounding_boxes.txt")

    # images.csv
    image_rows = []
    for img_id, rel_path in images.items():
        class_id = img_to_class[img_id]
        class_name = classes[class_id]
        is_train = int(split[img_id])
        x, y, w, h = bboxes.get(img_id, (None, None, None, None))
        image_rows.append(
            {
                "image_id": img_id,
                "file_path": rel_path,
                "class_id": class_id,
                "class_name": class_name,
                "is_train": is_train,
                "bbox_x": x,
                "bbox_y": y,
                "bbox_width": w,
                "bbox_height": h,
            }
        )
    write_csv(
        out_dir / "images.csv",
        ["image_id", "file_path", "class_id", "class_name", "is_train",
         "bbox_x", "bbox_y", "bbox_width", "bbox_height"],
        image_rows,
    )

    # classes.csv
    class_rows = [{"class_id": cid, "class_name": cname} for cid, cname in sorted(classes.items())]
    write_csv(out_dir / "classes.csv", ["class_id", "class_name"], class_rows)


def build_part_metadata(cub_root: Path, out_dir: Path):
    parts_txt = cub_root / "parts" / "parts.txt"
    part_locs_txt = cub_root / "parts" / "part_locs.txt"

    parts = read_parts(parts_txt)
    part_rows = [{"part_id": pid, "part_name": pname} for pid, pname in sorted(parts.items())]
    write_csv(out_dir / "parts.csv", ["part_id", "part_name"], part_rows)

    part_loc_rows = read_part_locs(part_locs_txt)
    write_csv(
        out_dir / "part_locs.csv",
        ["image_id", "part_id", "x", "y", "visible"],
        part_loc_rows,
    )

    # Build image-level binary presence for each part_id (if any visible annotation)
    img_to_parts = defaultdict(set)
    for row in part_loc_rows:
        if row["visible"]:
            img_to_parts[row["image_id"]].add(row["part_id"])

    part_ids_sorted = sorted(parts.keys())
    img_part_rows = []
    for img_id, part_ids in img_to_parts.items():
        row = {"image_id": img_id}
        for pid in part_ids_sorted:
            row[f"part_{pid}_present"] = int(pid in part_ids)
        img_part_rows.append(row)

    if img_part_rows:
        fieldnames = ["image_id"] + [f"part_{pid}_present" for pid in part_ids_sorted]
        write_csv(out_dir / "image_parts_binary.csv", fieldnames, img_part_rows)
    else:
        print("[prepare_cub_metadata] Warning: no visible parts found; image_parts_binary.csv will be empty.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cub_root", type=str, required=True,
                        help="Path to CUB_200_2011 root directory (the folder with images.txt, etc.).")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for metadata CSVs (default: <cub_root>/metadata)")
    args = parser.parse_args()

    cub_root = Path(args.cub_root)
    out_dir = Path(args.out_dir) if args.out_dir is not None else cub_root / "metadata"

    print(f"[prepare_cub_metadata] CUB root: {cub_root}")
    print(f"[prepare_cub_metadata] Output dir: {out_dir}")

    build_image_and_class_metadata(cub_root, out_dir)
    build_part_metadata(cub_root, out_dir)


if __name__ == "__main__":
    main()
