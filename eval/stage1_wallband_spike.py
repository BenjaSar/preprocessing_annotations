"""Stage 1 spike: does find_wall_bands() align with real GT door/window
boxes, across all available real commercial GT images (21 total,
test_pcs/commercial 11 + test_pcs/coco_w_images 10)?

Pass criterion (pre-registered): >=70% of GT door/window boxes lie
on/adjacent to a detected wall band.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/ubuntu/floorplan_classifier/VLM/preprocessing_annotations/src")
from preprocessing_annotations.detection.window_strip_detector import (
    find_wall_bands, WindowStripDetectorConfig,
)

DATASETS = [
    (
        "commercial",
        Path("/home/ubuntu/floorplan_classifier/VLM/test_pcs/commercial/"
             "commercial_building_obj_detection_coco_w_images/result.json"),
        Path("/home/ubuntu/floorplan_classifier/VLM/test_pcs/commercial/"
             "commercial_building_obj_detection_coco_w_images/images"),
        {"door", "double_door", "bifold_door", "bypass_sliding_door", "window", "sliding_window"},
    ),
    (
        "coco_w_images",
        Path("/home/ubuntu/floorplan_classifier/VLM/test_pcs/coco_w_images/result.json"),
        Path("/home/ubuntu/floorplan_classifier/VLM/test_pcs/coco_w_images/images"),
        {"door", "door2", "sliding door", "window1"},
    ),
]

TOLERANCE_PX = 5  # matches the module's own measured "±2px boundary error", padded


def box_near_band(gt_box, bands, tol):
    gx1, gy1, gx2, gy2 = gt_box
    for (bx, by, bw, bh) in bands:
        bx1, by1, bx2, by2 = bx - tol, by - tol, bx + bw + tol, by + bh + tol
        if gx1 < bx2 and gx2 > bx1 and gy1 < by2 and gy2 > by1:
            return True
    return False


def main():
    total, aligned = 0, 0
    per_image = []

    for label, coco_path, images_dir, target_cats in DATASETS:
        data = json.loads(coco_path.read_text())
        cats = {c["id"]: c["name"] for c in data["categories"]}
        id_to_file = {}
        for img in data["images"]:
            fname = img["file_name"].replace("\\", "/").split("/")[-1]
            id_to_file[img["id"]] = fname

        by_image = {}
        for a in data["annotations"]:
            if cats.get(a["category_id"]) in target_cats:
                by_image.setdefault(a["image_id"], []).append(a["bbox"])

        for img_id, boxes in by_image.items():
            fname = id_to_file.get(img_id)
            img_path = images_dir / fname
            if not img_path.exists():
                print(f"  SKIP missing image: {img_path}")
                continue
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                print(f"  SKIP unreadable: {img_path}")
                continue
            bands = find_wall_bands(gray, WindowStripDetectorConfig())

            n_aligned = 0
            for (x, y, w, h) in boxes:
                gt_box = (x, y, x + w, y + h)
                total += 1
                if box_near_band(gt_box, bands, TOLERANCE_PX):
                    aligned += 1
                    n_aligned += 1
            per_image.append((label, fname, len(boxes), len(bands), n_aligned))

    print(f"{'dataset':12} {'image':45} {'GT boxes':9} {'bands':6} {'aligned':8} {'rate'}")
    for label, fname, n_gt, n_bands, n_aligned in per_image:
        rate = 100 * n_aligned / n_gt if n_gt else 0
        print(f"{label:12} {fname[:45]:45} {n_gt:9} {n_bands:6} {n_aligned:8} {rate:.0f}%")

    print(f"\nTOTAL: {aligned}/{total} GT door/window boxes aligned with a wall band ({100*aligned/total:.1f}%)")
    print(f"Pass criterion: >=70%. Result: {'PASS' if aligned/total >= 0.70 else 'FAIL'}")


if __name__ == "__main__":
    main()
