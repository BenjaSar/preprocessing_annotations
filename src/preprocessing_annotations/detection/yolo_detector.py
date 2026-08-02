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
            boxes = _yolo_infer_detect(
                model, img_path, self.config.keep_categories,
                self.config.confidence_threshold,
            )
        except Exception as e:
            logger.warning(f"YOLO object detector: inference failed for {img_path}: {e}")
            return []

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

        all_dicts: List[Dict[str, Any]] = []
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
    first, merges each group independently.
    """
    by_category: Dict[str, List[Dict[str, Any]]] = {}
    for d in detections:
        by_category.setdefault(d["category"], []).append(d)
    merged: List[Dict[str, Any]] = []
    for group in by_category.values():
        merged.extend(
            splitter.merge(group, iou_threshold=iou_threshold, max_rooms=max_per_category)
        )
    return merged
