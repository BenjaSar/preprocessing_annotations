#!/usr/bin/env python3
"""Validation harness: count coco annotations and list per-page rooms.

Usage: python check_coco.py <output_dir>
       python check_coco.py  # scans common output dirs
"""
import json, sys, os
from pathlib import Path

def check(coco_path: Path):
    try:
        d = json.loads(coco_path.read_text())
    except Exception as e:
        print(f"  ERROR reading {coco_path}: {e}")
        return 0

    annotations = d.get("annotations", [])
    images = {img["id"]: img["file_name"] for img in d.get("images", [])}
    categories = {c["id"]: c["name"] for c in d.get("categories", [])}

    # per-image count
    per_img: dict = {}
    for ann in annotations:
        iid = ann["image_id"]
        per_img.setdefault(iid, []).append(ann)

    total = len(annotations)
    print(f"\n{'='*60}")
    print(f"  {coco_path}")
    print(f"  images: {len(images)}  annotations: {total}")
    print(f"{'='*60}")
    for iid, fname in sorted(images.items(), key=lambda x: x[1]):
        anns = per_img.get(iid, [])
        names = [categories.get(a["category_id"], "?") for a in anns]
        print(f"  {Path(fname).name[:60]:60s}  {len(anns):3d}  {', '.join(names[:8])}")
    if total == 0:
        print("  *** 0 annotations — no SFT data exported ***")
    return total

if __name__ == "__main__":
    if len(sys.argv) > 1:
        targets = [Path(sys.argv[1])]
    else:
        # scan recent output dirs
        base = Path("/home/ubuntu/floorplan_classifier/VLM")
        targets = sorted(base.glob("*/coco"), key=os.path.getmtime, reverse=True)[:5]

    grand = 0
    found = 0
    for t in targets:
        for split in ["train.json", "val.json", "test.json"]:
            p = t / split if t.suffix != ".json" else t
            if t.suffix == ".json":
                p = t
            else:
                p = t / split
            if p.exists():
                grand += check(p)
                found += 1

    print(f"\nTOTAL annotations across {found} files: {grand}")
