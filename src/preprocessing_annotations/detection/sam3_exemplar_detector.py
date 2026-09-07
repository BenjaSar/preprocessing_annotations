"""Tech-eval plan P2: tiled SAM3 box-exemplar door detector, seeded by
YoloObjectDetector's own high-confidence detections.

Mirrors YoloObjectDetector's shape (config-driven constructor, lazy
model load, detect_* returning WindowDetection) with one deliberate
difference: this detector's whole design is "find more of what another
detector already found," so its detect method takes that detector's
output as an input, not just an image path. It returns ONLY its own
net-new detections (deduped against the seeds) -- callers union the
result with the seed list themselves, the same way yolo_detector.py's
own output already gets appended to data["objectDetections"]
(orchestration/pipeline.py::_detect_object_mappings). No new merge
logic: the returned detections flow through the exact same
map_detections_to_rooms path YOLO's own detections already use.

Native-resolution tiling is not optional, it is the mechanism: a
full-page resize puts a ~44px door under one 14px ViT patch token
(measured: imgsz 644 -> 0.45 tokens, max confidence .32 on a real
project page); a 1008px native tile centered on each seed puts it at
~3 tokens (measured: max confidence .898, 9/9 doors recovered on the
same image/exemplar). See Sam3ExemplarDetectorConfig's docstring for
the current cross-dataset numbers. History worth knowing before citing
anything: the "union beats YOLO-alone on both precision and recall"
claim once stated here was measured against a duplicate implementation
in the eval script that had diverged from this file (withdrawn at P6);
re-measured through THIS class it was false through P9; it became true
at P13 after fixing a dedup defect (additions were deduped only
against the high-confidence exemplar subset, so duplicates of
lower-confidence YOLO doors survived). Current: test_pcs P.423/R.242
vs baseline P.420/R.223; FloorPlanCAD P.506/R.580 vs P.489/R.500.
Read that docstring's caveat list before quoting these -- test_pcs GT
is known incomplete and the threshold is unswept. ONE MORE: those
numbers are at yolo_conf=0.5. P13's fix was a no-op when production's
default (YoloObjectDetectorConfig.confidence_threshold) equalled this
class's seed_confidence_threshold (both 0.75) -- the two door lists
P13 split apart collapsed back into one when YOLO never emitted below
the exemplar floor. STALE as of I-2 (2026-09-01): that default is now
0.25 < seed_confidence_threshold (0.75), so P13's fix is live in
production, not a no-op -- and these P13 numbers (measured at
yolo_conf=0.5, still above 0.25) describe neither regime exactly. Do
not cite these numbers as production's current behavior.

UPDATE (2026-08-12): the tile grid is now SYSTEMATIC (TileSplitter, same
shape as YoloObjectDetector.detect_objects_tiled), not seed-centred
crops. The seed-centred design measured 30% page coverage on a real
page (7 seeds -> 7 tiles vs YOLO's 8-tile grid covering 100%) -- a
structural recall ceiling, and the exemplar-producing detector (YOLO,
full coverage) and this exemplar-consuming detector were running under
different search conditions. See Sam3ExemplarDetectorConfig's docstring
for the full incident (false-positive cascade this also surfaced, and
why multi-exemplar was tried and rejected as a fix).

UPDATE (2026-08-16, Audit #16 P0/P2): _adaptive_grid's round()->ceil()
fix and the scale-consistency gate (_apply_scale_consistency_gate,
config.min_seed_area_ratio) both landed here. The 1->4 real-door
recovery measured on one held-out page after this update is NOT P0's
isolated effect -- it sits downstream of BOTH the 2026-08-12 systematic-
grid rewrite AND this update, and the two were never measured apart.
P2 alone kills whole-page uniform cascades (verified: 63->0 on one
cascade page) but was found NOT to catch cascades mixed with real doors
on the same page (held-out batch2: 8 of 23 surviving additions across
4 pages were the same confused-symbol false positive, on BOTH
architectural and MEP-style pages -- not MEP-scoped as first guessed).
P6's full dataset re-measurement (see config docstring) is the actual
verdict on net effect; the spot-checks above explain the mechanism,
they are not the validation.

UPDATE (2026-08-18, P9): ceil() alone did not fix the downscale it was
built to fix -- tile_cols/tile_rows (both 4) capped the grid BELOW what
ceil(full_w/tile_target_px) asked for on real pages (4500x3375 needs
5 cols; round() gave 4, ceil() also gave... 4, because min(4, 5)=4).
The cap, not the rounding function, was the actual binding constraint
on every project page on hand. _adaptive_grid's own docstring below
no longer claims a guarantee it can't back -- read it, not this
paragraph, for current behavior. tile_cols/tile_rows bumped 4->5 in
Sam3ExemplarDetectorConfig (see that dataclass for the page-size
verification). P6's numbers PREDATE this fix -- they still reflect
the capped grid, not re-measured since.
"""

import logging
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..bbox.bbox_metrics import BBox, iou
from ..config import Sam3ExemplarDetectorConfig, resolve_sam3_checkpoint
from ..ingestion.tile_splitter import TileSplitter
from .window_detector import WindowDetection, WindowDetectionTier

logger = logging.getLogger(__name__)

_CATEGORY_DOOR = "door"


class Sam3ExemplarDetector:
    """Tiled SAM3 box-exemplar door detector, config-driven, lazy-loaded."""

    def __init__(
        self,
        config: Optional[Sam3ExemplarDetectorConfig] = None,
        predictor: Optional[Any] = None,
    ):
        """``predictor``, when given, is used as-is (no ultralytics import,
        no checkpoint resolution) -- same injection shape as
        DoorDetector(detector=...) in door_detector.py, for testing
        without a GPU/model load.
        """
        self.config = config or Sam3ExemplarDetectorConfig()
        self._predictor = predictor

    def _get_predictor(self):
        if self._predictor is None:
            from ultralytics.models.sam import SAM3SemanticPredictor

            self._predictor = SAM3SemanticPredictor(
                overrides={
                    "conf": self.config.sam3_confidence_threshold,
                    "model": resolve_sam3_checkpoint(),
                    "imgsz": self.config.tile_px,
                    "save": False,
                }
            )
        return self._predictor

    def _adaptive_grid(self, full_w: int, full_h: int) -> Tuple[int, int]:
        """Parameterized by THIS config's own tile_target_px (1008, SAM3's
        native-resolution ceiling), not YOLO's (1200) -- so this uses
        ceil(), not YOLO's round(). round() undershoots tile density
        (round(4500/1008)=4 cols -> 1238px tiles, 0.81x downscale of the
        native-res target; a real door on a real project page was lost
        under exactly this confound -- Audit #16).

        ceil() does NOT by itself guarantee tile_target_px -- min(cfg.
        tile_cols, ...) below can still cap the grid smaller than what
        ceil() asks for (found: min(4, ceil(4500/1008)=5) = 4, same
        1238px tile ceil() was supposed to fix -- P9). The real
        guarantee is: tiles never exceed tile_target_px PROVIDED
        tile_cols/tile_rows are large enough for the page's real
        dimensions -- config.py's Sam3ExemplarDetectorConfig.tile_cols
        docstring carries the current checked page-size range (P9:
        bumped 4->5). A page wider than tile_cols*tile_target_px will
        still downscale silently; there is no runtime check for this.
        "Checked" above means geometric coverage only, not detection
        quality -- that same docstring also carries a CONTESTED note
        (C2): the 4->5 bump measurably lost real doors on 2/2 real
        pages tested. Read both notes together, not this one alone.
        """
        cfg = self.config
        if cfg.tile_target_px > 0:
            cols = max(2, min(cfg.tile_cols, math.ceil(full_w / cfg.tile_target_px)))
            rows = max(2, min(cfg.tile_rows, math.ceil(full_h / cfg.tile_target_px)))
        else:
            cols, rows = cfg.tile_cols, cfg.tile_rows
        return cols, rows

    def _apply_scale_consistency_gate(
        self,
        additions: List[Tuple[float, float, float, float, float]],
        seed_boxes: List[WindowDetection],
        img_path: Path,
    ) -> List[Tuple[float, float, float, float, float]]:
        """5th guardrail, same all-or-nothing shape as the existing
        over_segmentation/flood_swallowed_seed/collapse/label_scale_noop
        family (see pipeline.py). SAM3's same-image concept matching can
        lock onto a repeated non-door symbol elsewhere on the page
        (measured: light fixture + switch-leg wire, same "rectangle +
        arc" composition as a door swing) -- these cascade instances run
        systematically smaller than the real door seed that triggered
        them. Compare median area(additions) vs median area(seeds); a
        page whose ratio falls below min_seed_area_ratio has its
        additions discarded entirely, not filtered one-by-one (per-box
        confidence was tested and refuted as a discriminator -- FP and
        real confidence ranges fully overlap).
        """
        if not additions:
            return additions
        seed_areas = [
            (s.bbox[2] - s.bbox[0]) * (s.bbox[3] - s.bbox[1]) for s in seed_boxes
        ]
        addition_areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in additions]
        seed_median = statistics.median(seed_areas)
        if seed_median <= 0:
            return additions
        ratio = statistics.median(addition_areas) / seed_median
        if ratio < self.config.min_seed_area_ratio:
            logger.info(
                "SAM3 exemplar: scale-consistency gate discarded %d addition(s) "
                "for %s (area ratio %.3f < %.3f)",
                len(additions), img_path, ratio, self.config.min_seed_area_ratio,
            )
            return []
        return additions

    def detect_additional(
        self, img_path: Path, seeds: List[WindowDetection]
    ) -> List[WindowDetection]:
        """Return net-new door detections found by seeding SAM3's
        concept mode with ``seeds``' high-confidence door boxes.

        ``seeds`` is expected to be YoloObjectDetector's own output
        (any category) -- this method filters to door and to
        ``seed_confidence_threshold`` itself, so callers don't need to
        pre-filter.

        TWO distinct door lists, deliberately (P13):
        * ``seed_boxes`` -- door AND confidence >= seed_confidence_
          threshold. What SAM3 gets as exemplars. High-confidence only:
          a weak exemplar makes the concept worse, and the false-
          positive cascade this detector is prone to is seeded exactly
          that way.
        * ``dedup_boxes`` -- EVERY door in ``seeds``, no confidence
          floor. What "is this addition already found?" is checked
          against. Using seed_boxes for this instead was a real defect
          (found P13, 2026-08-18): a door YOLO detected at confidence
          0.5-0.75 was invisible to the dedup but still present in the
          caller's union, so SAM3 re-finding it produced a duplicate
          box scoring as a false positive. Measured on test_pcs: 4 of
          13 marginal false positives were duplicates of doors YOLO had
          already reported, at IoU 0.67-0.95 vs the YOLO box -- all
          well above seed_overlap_iou, i.e. all should have been
          dropped and none were. NO-OP only when the caller's own doors
          never fall below seed_confidence_threshold -- true when
          YoloObjectDetectorConfig.confidence_threshold ==
          seed_confidence_threshold (both were 0.75 until I-2,
          2026-09-01). Production's default is now 0.25 <
          seed_confidence_threshold (0.75), so door_boxes and
          seed_boxes are NOT the same list in production anymore -- this
          split is live there, not just in the GT eval harness
          (yolo_conf=0.5, where it always mattered).

        Systematic grid (TileSplitter), same shape as
        YoloObjectDetector.detect_objects_tiled: every tile with >=1
        seed inside it gets those seeds as exemplars; tiles with no
        seed are skipped (no exemplar to seed a concept with -- this is
        additive recall on top of YOLO, not a from-scratch search).
        Replaces the old seed-centred crop (measured: 30% page coverage
        on a real page vs YOLO's 100%) -- see module docstring.
        """
        door_boxes = [
            s for s in seeds if (s.metadata or {}).get("type") == _CATEGORY_DOOR
        ]
        seed_boxes = [
            s
            for s in door_boxes
            if s.confidence >= self.config.seed_confidence_threshold
        ]
        if not seed_boxes:
            return []

        try:
            predictor = self._get_predictor()
        except Exception as e:
            logger.warning(f"SAM3 exemplar detector: model load failed: {e}")
            return []

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(
                f"SAM3 exemplar detector: could not open {img_path}: {e}"
            )
            return []

        full_w, full_h = image.size
        cols, rows = self._adaptive_grid(full_w, full_h)
        splitter = TileSplitter(
            cols=cols, rows=rows, overlap_pct=self.config.tile_overlap_pct
        )

        raw_boxes: List[Tuple[float, float, float, float, float]] = []
        for tile_img, meta in splitter.split(image):
            # Seeds fully contained in this tile become its exemplars.
            # A seed straddling a tile boundary is skipped for that tile
            # (it's still available from whichever tile(s) fully contain
            # it -- the 10% overlap exists exactly to make that likely).
            tile_seeds = [
                [
                    s.bbox[0] - meta.x_offset, s.bbox[1] - meta.y_offset,
                    s.bbox[2] - meta.x_offset, s.bbox[3] - meta.y_offset,
                ]
                for s in seed_boxes
                if meta.x_offset <= s.bbox[0] and s.bbox[2] <= meta.x_offset + meta.tile_w
                and meta.y_offset <= s.bbox[1] and s.bbox[3] <= meta.y_offset + meta.tile_h
            ]
            if not tile_seeds:
                continue
            tile_array = np.array(tile_img)
            predictor.set_image(tile_array)
            # ONE exemplar per predictor call, not all of a tile's seeds
            # at once -- multi-exemplar was tested and REJECTED as a fix
            # for the false-positive cascade (measured: K=1 -> 1 instance,
            # K=2 -> 23 instances, same real tile; see config docstring).
            # A tile with multiple seeds calls the predictor once per
            # seed, matching the validated K=1 regime exactly; only the
            # SEARCH AREA (this tile) is now systematic, not the
            # exemplar count.
            for seed_box in tile_seeds:
                try:
                    result = predictor(bboxes=[seed_box])[0]
                except Exception as e:
                    logger.warning(
                        f"SAM3 exemplar detector: tile ({meta.col},{meta.row}) "
                        f"inference failed for {img_path}: {e}"
                    )
                    continue
                if result.boxes is None:
                    continue
                for xyxy, conf in zip(
                    result.boxes.xyxy.tolist(), result.boxes.conf.tolist()
                ):
                    raw_boxes.append((
                        xyxy[0] + meta.x_offset, xyxy[1] + meta.y_offset,
                        xyxy[2] + meta.x_offset, xyxy[3] + meta.y_offset, conf,
                    ))

        # Dedup SAM3's own boxes against each other first (overlapping
        # tiles, or multiple seeds in one tile, can rediscover the same
        # instance), highest confidence wins.
        deduped: List[Tuple[float, float, float, float, float]] = []
        for box in sorted(raw_boxes, key=lambda b: b[4], reverse=True):
            if not any(
                iou(BBox(*box[:4]), BBox(*kept[:4])) > self.config.dedup_iou
                for kept in deduped
            ):
                deduped.append(box)

        # Dedup against EVERY door the caller already has (door_boxes),
        # not just the high-confidence exemplar subset -- see this
        # method's docstring (P13). seed_boxes is the exemplar source;
        # door_boxes is what the caller will union these additions with,
        # so it is what "already found" has to mean.
        additions = [
            box
            for box in deduped
            if not any(
                iou(BBox(*box[:4]), BBox(*door.bbox)) > self.config.seed_overlap_iou
                for door in door_boxes
            )
        ]

        additions = self._apply_scale_consistency_gate(additions, seed_boxes, img_path)

        return [
            WindowDetection(
                bbox=tuple(box[:4]),
                confidence=box[4],
                source_tier=WindowDetectionTier.SAM3_EXEMPLAR,
                metadata={"type": _CATEGORY_DOOR},
            )
            for box in additions
        ]
