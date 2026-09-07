"""Wall-band-derived box prompt spike (2026-09-03).

Gap: sam_segmenter.py's box-prompt path (F-B) exists but is deliberately
DISABLED for label-scale rooms (should_use_box_prompt) because the only
box ever available was the label's own tiny box, which would confine SAM
to the label instead of the room. Wall bands (validated 88% this session,
21 real images) can supply the missing input: a room-scale hypothesis box.

Technique: for each label seed, ray-cast to the nearest wall band in each
of 4 directions (up/down/left/right); the box bounded by those 4 walls is
the room hypothesis. Feed as label_bbox to the EXISTING, unmodified
segment_from_point(..., label_bbox=box) call.

Baselines already measured this session:
  point-prompt (current production path): 5.4% room-scale (47/877)
  flood-fill + door-gap-closing: 0/17 genuine (0%)

Pass: >=25% of seeds yield a room-scale result (not label_scale_noop,
not over-seg-guardrail-rejected). Fail: stop the segmentation axis.
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
from preprocessing_annotations.detection.sam_segmenter import (
    RoomSegmenter, is_label_scale_result, is_collapse,
)
from preprocessing_annotations.config import SAMConfig

MIN_EXPANSION_AREA_PX = 10_000  # SAMConfig default, matches production


def separate_bands(bands):
    """Split into horizontal (wide) vs vertical (tall) wall bands."""
    horiz, vert = [], []
    for (x, y, w, h) in bands:
        if w >= h:
            horiz.append((x, y, w, h))
        else:
            vert.append((x, y, w, h))
    return horiz, vert


def box_from_walls(seed_xy, horiz, vert, page_w, page_h, max_search=None):
    """Ray-cast from seed to nearest wall band in each of 4 directions.
    Falls back to page edge if no wall found in that direction."""
    sx, sy = seed_xy
    top, bottom, left, right = 0, page_h, 0, page_w

    # UP: horizontal bands above seed, x-range overlapping seed.x
    best = None
    for (x, y, w, h) in horiz:
        if x <= sx <= x + w and y + h <= sy:
            if best is None or (y + h) > best:
                best = y + h
    if best is not None:
        top = best

    # DOWN
    best = None
    for (x, y, w, h) in horiz:
        if x <= sx <= x + w and y >= sy:
            if best is None or y < best:
                best = y
    if best is not None:
        bottom = best

    # LEFT: vertical bands left of seed, y-range overlapping seed.y
    best = None
    for (x, y, w, h) in vert:
        if y <= sy <= y + h and x + w <= sx:
            if best is None or (x + w) > best:
                best = x + w
    if best is not None:
        left = best

    # RIGHT
    best = None
    for (x, y, w, h) in vert:
        if y <= sy <= y + h and x >= sx:
            if best is None or x < best:
                best = x
    if best is not None:
        right = best

    if right <= left or bottom <= top:
        return None
    return [int(left), int(top), int(right - left), int(bottom - top)]


def main():
    pages = [
        ("9d732b67-326_ROCKAWAY_-_AVI-ON_LAYOUT_page005.png", "easy (100% stage1)"),
        ("aceb87d1-326_ROCKAWAY_-_AVI-ON_LAYOUT_page001.png", "hard (74% stage1)"),
        ("980bd425-BP-1_Kennedy_Middle_School__-_AVI-ON_LAYOUT_FLAT_page002.png", "kennedy (89% stage1)"),
    ]
    base_dir = Path(
        "/home/ubuntu/floorplan_classifier/VLM/test_pcs/commercial/"
        "commercial_building_obj_detection_coco_w_images/images"
    )

    segmenter = RoomSegmenter(config=SAMConfig(device="cuda"))

    overall_total = 0
    overall_room_scale = 0
    overall_noop = 0
    overall_no_box = 0
    overall_error = 0

    for fname, tag in pages:
        img_path = base_dir / fname
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        h, w = gray.shape
        bands = find_wall_bands(gray, WindowStripDetectorConfig())
        horiz, vert = separate_bands(bands)

        # Real OCR seeds for THIS page (fast, cheap, no VLM).
        from preprocessing_annotations.config import OCRConfig
        from preprocessing_annotations.ingestion.two_pass_ocr_extractor import TwoPassOCRExtractor
        extractor = TwoPassOCRExtractor(OCRConfig(backend="paddleocr"))
        rooms, _ = extractor.find_rooms_pass1(img_path)

        n_total = n_room_scale = n_noop = n_no_box = n_error = 0
        for r in rooms:
            x1, y1, x2, y2 = r.bbox
            seed = ((x1 + x2) / 2, (y1 + y2) / 2)
            box = box_from_walls(seed, horiz, vert, w, h)
            n_total += 1
            if box is None:
                n_no_box += 1
                continue
            try:
                result = segmenter.segment_from_point(
                    img_path, (int(seed[0]), int(seed[1])),
                    include_mask=False, label_bbox=box,
                )
                sam_area = result.bbox[2] * result.bbox[3]
                if is_label_scale_result(sam_area, MIN_EXPANSION_AREA_PX):
                    n_noop += 1
                else:
                    n_room_scale += 1
            except Exception as e:
                n_error += 1

        print(f"{fname} ({tag}): {n_total} seeds, bands={len(bands)} "
              f"(h={len(horiz)},v={len(vert)})")
        print(f"  room_scale={n_room_scale} label_noop={n_noop} "
              f"no_box_derivable={n_no_box} error={n_error}  "
              f"rate={100*n_room_scale/n_total:.1f}%")

        overall_total += n_total
        overall_room_scale += n_room_scale
        overall_noop += n_noop
        overall_no_box += n_no_box
        overall_error += n_error

    print(f"\n=== OVERALL: {overall_room_scale}/{overall_total} room-scale "
          f"({100*overall_room_scale/overall_total:.1f}%) ===")
    print(f"  label_noop={overall_noop} no_box={overall_no_box} error={overall_error}")
    print(f"Baselines: point-prompt=5.4%, flood-fill=0%")
    print(f"Pass criterion: >=25%. Result: "
          f"{'PASS' if overall_room_scale/overall_total >= 0.25 else 'FAIL'}")


if __name__ == "__main__":
    main()
