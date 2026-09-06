"""Stage 2 spike: does gap-closing at GT door locations + wall-band
barriers let flood-fill from real OCR label seeds produce BOUNDED
regions, instead of the refuted classical attempt's 77-82% page flood?

Pass criterion (pre-registered): >=50% of seeds yield a region <30% of
page area.
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

IMG_PATH = Path(
    "/home/ubuntu/floorplan_classifier/VLM/test_pcs/commercial/"
    "commercial_building_obj_detection_coco_w_images/images/"
    "aceb87d1-326_ROCKAWAY_-_AVI-ON_LAYOUT_page001.png"
)
GT_COCO = Path(
    "/home/ubuntu/floorplan_classifier/VLM/test_pcs/commercial/"
    "commercial_building_obj_detection_coco_w_images/result.json"
)
SEEDS_PATH = Path("/tmp/aceb87d1_ocr_seeds.json")
_DOOR_CATS = {"door", "double_door", "bifold_door", "bypass_sliding_door"}


def load_gt_door_boxes():
    data = json.loads(GT_COCO.read_text())
    cats = {c["id"]: c["name"] for c in data["categories"]}
    id_map = {img["id"]: img["file_name"].replace("\\", "/").split("/")[-1] for img in data["images"]}
    img_id = [k for k, v in id_map.items() if v == IMG_PATH.name][0]
    boxes = []
    for a in data["annotations"]:
        if a["image_id"] == img_id and cats.get(a["category_id"]) in _DOOR_CATS:
            x, y, w, h = a["bbox"]
            boxes.append((int(x), int(y), int(w), int(h)))
    return boxes


def build_barrier_mask(gray, bands, door_boxes):
    h, w = gray.shape
    barrier = np.zeros((h, w), dtype=np.uint8)
    # Walls: rasterize band boxes as filled barriers (first cut -- bounding
    # box, not traced wall shape; good enough to test containment).
    for (bx, by, bw, bh) in bands:
        barrier[by:by + bh, bx:bx + bw] = 255
    # Close gaps: GT door locations sealed as barriers too, so a doorway
    # does not leak two rooms into one region for segmentation purposes.
    for (dx, dy, dw, dh) in door_boxes:
        barrier[max(0, dy):dy + dh, max(0, dx):dx + dw] = 255
    return barrier


def flood_from_seed(barrier, seed_xy, page_area):
    """Flood-fill free space (non-barrier) from seed. Returns filled area."""
    h, w = barrier.shape
    free = (barrier == 0).astype(np.uint8) * 255
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    sx, sy = int(seed_xy[0]), int(seed_xy[1])
    sx = min(max(sx, 0), w - 1)
    sy = min(max(sy, 0), h - 1)
    if free[sy, sx] == 0:
        # seed itself sits on a barrier pixel (e.g. inside detected band) --
        # nudge to nearest free pixel in a small radius, else report 0.
        found = False
        for r in range(1, 15):
            for ddx in range(-r, r + 1):
                for ddy in range(-r, r + 1):
                    ny, nx = sy + ddy, sx + ddx
                    if 0 <= ny < h and 0 <= nx < w and free[ny, nx] != 0:
                        sy, sx = ny, nx
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            return 0
    _, filled, _, _ = cv2.floodFill(free.copy(), mask, (sx, sy), 128)
    area = int((filled == 128).sum())
    return area


def main():
    gray = cv2.imread(str(IMG_PATH), cv2.IMREAD_GRAYSCALE)
    h, w = gray.shape
    page_area = h * w

    bands = find_wall_bands(gray, WindowStripDetectorConfig())
    door_boxes = load_gt_door_boxes()
    seeds = json.loads(SEEDS_PATH.read_text())

    print(f"page: {w}x{h}, bands: {len(bands)}, GT doors: {len(door_boxes)}, seeds: {len(seeds)}")

    barrier_walls_only = build_barrier_mask(gray, bands, [])
    barrier_walls_doors = build_barrier_mask(gray, bands, door_boxes)

    results = []
    for s in seeds:
        x1, y1, x2, y2 = s["bbox"]
        seed_xy = ((x1 + x2) / 2, (y1 + y2) / 2)
        area_walls_only = flood_from_seed(barrier_walls_only, seed_xy, page_area)
        area_with_doors = flood_from_seed(barrier_walls_doors, seed_xy, page_area)
        results.append((s["name"], area_walls_only / page_area, area_with_doors / page_area))

    print(f"\n{'label':30} {'walls-only frac':16} {'walls+doors frac':16}")
    for name, f1, f2 in results:
        print(f"{name[:30]:30} {f1*100:15.1f}% {f2*100:15.1f}%")

    for label, idx in [("walls-only", 1), ("walls+doors", 2)]:
        bounded = sum(1 for r in results if r[idx] < 0.30)
        print(f"\n{label}: {bounded}/{len(results)} seeds bounded (<30% page) = {100*bounded/len(results):.0f}%")
    print("Pass criterion: >=50% bounded. ")


if __name__ == "__main__":
    main()
