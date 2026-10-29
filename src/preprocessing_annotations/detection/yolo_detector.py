"""T-I3: pipeline-integrated YOLO door/window detector.

Mirrors WindowDetector's shape (config-driven constructor, lazy model
load, detect_* method returning WindowDetection) so it slots into the
existing room-mapping machinery without a new architecture.

COORDINATE-SPACE NOTE (audited before writing this file): pipeline.py's
room bboxes (data["rooms"]) are in ORIGINAL full-image pixel space (VLM/
OCR detections, tile-merged back to full-image coords -- see
tile_splitter.py::rescale_rooms). WindowDetector's CubiCasa5K tier runs
on pipeline.py::_load_image_for_detection's ≤1024px-resized array, which
would put THIS detector's boxes in a different, smaller coordinate space
than the rooms they need to intersect -- that tier has never fired in
production (config comment: "Tier 2, not yet active"), so the mismatch
has never mattered. This detector must NOT repeat it: it runs directly
on img_path (the original, full-resolution image), the same call
pattern already verified correct by the T-S2 eval harness
(kaggle_door_window_eval.py measured its door/window numbers this exact
way). Ultralytics resizes internally for the network and rescales
predictions back to the SOURCE image's own dimensions -- so boxes come
back in original-image pixel space, matching data["rooms"] directly.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from ..config import YoloObjectDetectorConfig
from ..ingestion.tile_splitter import TileSplitter
from .window_detector import WindowDetection, WindowDetectionTier
from .yolo_infer import detect_objects as _yolo_infer_detect
from .yolo_infer import load_model

logger = logging.getLogger(__name__)


def _passes_shape_floor(bbox, min_area_px: int) -> bool:
    """D1 (2026-09-05): reject literally-degenerate (near-zero-area) boxes.
    NOT a precision filter -- an aspect-ratio cap was tried and rejected
    (see YoloObjectDetectorConfig.min_area_px docstring: windows are
    structurally elongated, a shared aspect cap dropped 97% of real
    windows on the measured run). The one confirmed punctuation-glyph FP
    this was originally meant to catch is NOT resolved by this floor --
    its shape overlaps real doors' own distribution too closely. bbox is
    (x1, y1, x2, y2) in full-image pixel space (post tile-rescale).
    """
    if not bbox or len(bbox) != 4:
        return True
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if w <= 0 or h <= 0:
        return True
    return (w * h) >= min_area_px


class YoloObjectDetector:
    """Fine-tuned YOLO door/window detector, config-driven, lazy-loaded."""

    def __init__(self, config: Optional[YoloObjectDetectorConfig] = None):
        self.config = config or YoloObjectDetectorConfig()
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = load_model(self.config.checkpoint_path)
        return self._model

    def detect_objects(self, img_path: Path) -> List[WindowDetection]:
        """Run the fine-tuned checkpoint on the original image, return
        WindowDetection objects tagged with metadata["type"] = the
        predicted class name ("door"/"window") -- this is exactly the
        field map_detections_to_rooms (window_detector.py) already
        dispatches on, so no new mapping logic is needed.
        """
        try:
            model = self._get_model()
        except Exception as e:
            logger.warning(f"YOLO object detector: model load failed: {e}")
            return []

        try:
            from ..orchestration.mlflow_tracking import traced
            with traced("yolo.detect_objects", "TOOL"):
                boxes = _yolo_infer_detect(
                    model, img_path, self.config.keep_categories,
                    self.config.confidence_threshold,
                )
        except Exception as e:
            logger.warning(f"YOLO object detector: inference failed for {img_path}: {e}")
            return []

        _n_pre = len(boxes)
        boxes = [
            (class_name, bbox, confidence) for class_name, bbox, confidence in boxes
            if _passes_shape_floor(bbox, self.config.min_area_px)
        ]
        if len(boxes) < _n_pre:
            logger.info(f"D1 shape floor: dropped {_n_pre - len(boxes)} of {_n_pre} boxes for {img_path}")

        return [
            WindowDetection(
                bbox=bbox,
                confidence=confidence,
                source_tier=WindowDetectionTier.YOLO,
                metadata={"type": class_name},
            )
            for class_name, bbox, confidence in boxes
        ]

    def detect_objects_tiled(self, img_path: Path) -> List[WindowDetection]:
        """R2: tiled detection -- recovers door/window symbols that
        full-page downscale destroys. Decisive test (2026-07-29): a
        native-resolution crop of a test_pcs page recovered 4/5 known GT
        doors (best IoU .53-.88) that were pure misses (best IoU 0) at
        full-page scale -- downscale, not domain gap, is the dominant
        recall killer. Falls back to detect_objects (full-image, no
        tiling) when tiling is off or the image is below the trigger
        size -- identical fallback shape to pipeline.py's VLM room
        tiling (_detect_rooms_maybe_tiled).
        """
        if not self.config.use_tiling:
            return self.detect_objects(img_path)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"YOLO tiled detector: could not open {img_path}: {e}")
            return []

        full_w, full_h = image.size
        if max(full_w, full_h) <= self.config.tile_trigger_px:
            return self.detect_objects(img_path)

        try:
            model = self._get_model()
        except Exception as e:
            logger.warning(f"YOLO object detector: model load failed: {e}")
            return []

        cols, rows = self._adaptive_grid(full_w, full_h)
        splitter = TileSplitter(
            cols=cols, rows=rows, overlap_pct=self.config.tile_overlap_pct
        )

        from ..orchestration.mlflow_tracking import log_stage_metric, traced
        log_stage_metric("yolo_tiled", "tile_count", cols * rows)

        all_dicts: List[Dict[str, Any]] = []
        with traced("yolo.detect_objects_tiled", "TOOL"):
            for tile_img, meta in splitter.split(image):
                try:
                    boxes = _yolo_infer_detect(
                        model, tile_img, self.config.keep_categories,
                        self.config.confidence_threshold,
                    )
                except Exception as e:
                    logger.warning(
                        f"YOLO tiled detector: tile ({meta.col},{meta.row}) "
                        f"inference failed: {e}"
                    )
                    continue
                tile_dicts = [
                    {"category": category, "bbox": list(bbox), "confidence": confidence}
                    for category, bbox, confidence in boxes
                ]
                all_dicts.extend(splitter.rescale_rooms(tile_dicts, meta))

            merged = _merge_per_category(
                splitter,
                all_dicts,
                self.config.tile_merge_iou,
                self.config.tile_max_detections_per_category,
            )

        _n_pre = len(merged)
        merged = [
            d for d in merged
            if _passes_shape_floor(d["bbox"], self.config.min_area_px)
        ]
        if len(merged) < _n_pre:
            logger.info(f"D1 shape floor: dropped {_n_pre - len(merged)} of {_n_pre} tiled boxes for {img_path}")

        return [
            WindowDetection(
                bbox=tuple(d["bbox"]),
                confidence=d["confidence"],
                source_tier=WindowDetectionTier.YOLO,
                metadata={"type": d["category"]},
            )
            for d in merged
        ]

    def _adaptive_grid(self, full_w: int, full_h: int) -> "tuple[int, int]":
        """Same adaptive-grid formula as pipeline.py's
        _detect_rooms_maybe_tiled (VLM room tiling) -- mirrored, not
        reinvented, so both tiling paths scale identically with image
        size."""
        cfg = self.config
        if cfg.tile_target_px > 0:
            cols = max(2, min(cfg.tile_cols, round(full_w / cfg.tile_target_px)))
            rows = max(2, min(cfg.tile_rows, round(full_h / cfg.tile_target_px)))
        else:
            cols, rows = cfg.tile_cols, cfg.tile_rows
        return cols, rows


def _drop_containment_unions(boxes: List[Dict[str, Any]], min_contained: int = 2,
                              coverage_threshold: float = 0.8) -> List[Dict[str, Any]]:
    """D2 (2026-09-07): drop a box that fully swallows >=2 sibling boxes.

    TileSplitter.merge's NMS only drops a candidate whose IoU with an
    already-kept box exceeds iou_threshold. IoU is symmetric-overlap-
    aware, not containment-aware: a wide box that CONTAINS several small
    real detections has a LOW IoU with each of them individually (the
    union is dominated by the big box's own area), so none of the
    IoU>=0.30 checks fire and all boxes survive side by side. Visually
    this renders as one "expanded" door -- confirmed on a real page
    (Kennedy AVI-ON FLAT p002): a 559x333px box whose corners matched
    the bounding union of 3 separately-kept ~100-300px door boxes to
    within a few px. A real door is one symbol; it should never contain
    multiple sibling detections of the same category.
    """
    def _area(b):
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    def _contains_frac(outer, inner):
        ix1, iy1 = max(outer[0], inner[0]), max(outer[1], inner[1])
        ix2, iy2 = min(outer[2], inner[2]), min(outer[3], inner[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        inner_area = _area(inner)
        return inter / inner_area if inner_area > 0 else 0.0

    kept = []
    for i, cand in enumerate(boxes):
        cb = cand.get("bbox")
        if not cb or len(cb) != 4:
            kept.append(cand)
            continue
        n_contained = sum(
            1 for j, other in enumerate(boxes)
            if j != i and other.get("bbox") and len(other["bbox"]) == 4
            and _area(other["bbox"]) < _area(cb)
            and _contains_frac(cb, other["bbox"]) >= coverage_threshold
        )
        if n_contained >= min_contained:
            logger.info(
                f"D2 containment filter: dropped box {cb} (contains "
                f"{n_contained} sibling detections)"
            )
            continue
        kept.append(cand)
    return kept


def _drop_area_outliers(boxes: List[Dict[str, Any]], category: str,
                         min_sample: int = 5, ratio: float = 4.0) -> List[Dict[str, Any]]:
    """D3 (2026-09-07): drop a door box whose area is an outlier vs the
    page's OWN other doors -- catches individually-oversized (not
    union/containment, see D2) model over-predictions.

    Confirmed real case (326 ROCKAWAY p000, "PLUMBING ROOM" FP pair):
    two boxes 23,956px^2 and 15,876px^2, each NORMAL aspect ratio (0.47,
    0.74 -- inside the measured p95=2.41 for real doors, so an aspect
    filter can't catch them, and aspect filters were already rejected
    once this session for cutting real windows). The page's other 6
    doors had a median area of 1,798px^2 -- both outliers are 6-10x that,
    while every real door on the page sits within 1.5x. Same-page
    relative comparison avoids a fixed px^2 constant that would break
    across different page resolutions/DPI.

    Door-only: NOT applied to window (only 15 window detections
    project-wide after the E4 regression fix -- too small a sample for
    a per-page median to be safe, and generalizing an unvalidated filter
    to the already-fragile category is exactly the mistake E4 made
    once). min_sample guards against an unstable median on pages with
    few doors -- skip the filter there rather than risk a wrong call.
    """
    if category != "door":
        return boxes
    valid = [b for b in boxes if b.get("bbox") and len(b["bbox"]) == 4]
    if len(valid) < min_sample:
        return boxes

    def _area(b):
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    areas = sorted(_area(b["bbox"]) for b in valid)
    n = len(areas)
    median = areas[n // 2] if n % 2 else (areas[n // 2 - 1] + areas[n // 2]) / 2
    if median <= 0:
        return boxes

    kept = []
    for b in boxes:
        bbox = b.get("bbox")
        if bbox and len(bbox) == 4 and _area(bbox) > ratio * median:
            logger.info(
                f"D3 area-outlier filter: dropped door box {bbox} "
                f"(area {_area(bbox):.0f} > {ratio}x page median {median:.0f})"
            )
            continue
        kept.append(b)
    return kept


def _merge_per_category(
    splitter: TileSplitter,
    detections: List[Dict[str, Any]],
    iou_threshold: float,
    max_per_category: int,
) -> List[Dict[str, Any]]:
    """Dedup tile-overlap detections per category.

    TileSplitter.merge is category-blind (no class field consulted) --
    merging all categories together would let a door box suppress an
    overlapping window box in the same NMS pass. Groups by category
    first, merges each group independently. After NMS, also drops
    containment-union artifacts (see _drop_containment_unions) and
    same-page area outliers (see _drop_area_outliers, door-only) --
    IoU-based NMS alone cannot catch either (see those functions'
    docstrings).
    """
    by_category: Dict[str, List[Dict[str, Any]]] = {}
    for d in detections:
        by_category.setdefault(d["category"], []).append(d)
    merged: List[Dict[str, Any]] = []
    for category, group in by_category.items():
        category_kept = splitter.merge(group, iou_threshold=iou_threshold, max_rooms=max_per_category)
        category_kept = _drop_containment_unions(category_kept)
        category_kept = _drop_area_outliers(category_kept, category)
        merged.extend(category_kept)
    return merged
