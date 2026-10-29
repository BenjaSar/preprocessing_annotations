#!/usr/bin/env python3
"""
Sam3ExemplarDetector Tests — validate tile-mapping, dedup, and seed-overlap
logic without loading SAM3 (a fake predictor is injected via the same
constructor-injection shape as DoorDetector(detector=...)).

Systematic-grid tiling (2026-08-12): tile geometry now comes from
TileSplitter/_adaptive_grid (image-dimension-derived), replacing the old
seed-centred crop (measured: 30% page coverage vs YOLO's 100%). Tests
use tile_target_px=0 with explicit tile_cols/tile_rows so grid geometry
is exact and deterministic in a small test image, not adaptive-rounded.

Covers:
  - seeds below seed_confidence_threshold are never sent to the predictor
  - non-door seeds are ignored
  - a SAM3 box overlapping an existing seed is dropped (not double-counted)
  - two tiles rediscovering the same instance dedup to one, higher conf wins
  - tile-local box coordinates map back to page coordinates correctly
  - tile origin is image-derived (same tile -> same origin regardless of
    which seed inside it triggered the call) -- NOT seed-derived like the
    old design
  - a seed straddling a tile boundary (not fully contained in any tile)
    is skipped for scoring, no crash
  - multiple seeds in ONE tile trigger ONE predictor call EACH, never a
    single call with all of them -- multi-exemplar was tested against
    the real cascade and made it worse (K=1->1, K=2->23 instances on
    the same real tile); this guards against silently reintroducing it
  - a predictor exception is caught, logged, and returns [] (no propagation)
  - returned detections carry SAM3_EXEMPLAR provenance + door type tag
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pytest
from PIL import Image

# Add the package dir (parent of test/) so `preprocessing_annotations` resolves.
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing_annotations.config import Sam3ExemplarDetectorConfig
from preprocessing_annotations.detection.sam3_exemplar_detector import Sam3ExemplarDetector
from preprocessing_annotations.detection.window_detector import (
    WindowDetection,
    WindowDetectionTier,
)


def _seed(bbox: Tuple[float, float, float, float], confidence: float, category: str = "door") -> WindowDetection:
    return WindowDetection(
        bbox=bbox,
        confidence=confidence,
        source_tier=WindowDetectionTier.YOLO,
        metadata={"type": category},
    )


class _FakeArray:
    """Mimics ultralytics' Results.boxes: .xyxy/.conf each expose .tolist()."""

    def __init__(self, data: List[List[float]]):
        self._data = data

    def tolist(self):
        return self._data


class _FakeBoxes:
    def __init__(self, xyxy: List[List[float]], conf: List[float]):
        self.xyxy = _FakeArray(xyxy)
        self.conf = _FakeArray(conf)


class _FakeResult:
    def __init__(self, boxes: Optional[_FakeBoxes]):
        self.boxes = boxes


class _FakeSam3Predictor:
    """Queue of per-call (xyxy, conf) responses, consumed in call order.
    ``None`` means "no boxes" (mirrors real Results.boxes is None)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.set_image_calls: List[np.ndarray] = []
        self.call_bboxes: List[list] = []

    def set_image(self, tile):
        self.set_image_calls.append(tile)

    def __call__(self, bboxes):
        self.call_bboxes.append(bboxes)
        payload = self._responses.pop(0)
        if payload is None:
            return [_FakeResult(None)]
        xyxy, conf = payload
        return [_FakeResult(_FakeBoxes(xyxy, conf))]


class _RaisingPredictor:
    def set_image(self, tile):
        pass

    def __call__(self, bboxes):
        raise RuntimeError("simulated SAM3 failure")


def _make_image(tmp_path: Path, w: int, h: int) -> Path:
    path = tmp_path / "page.png"
    Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(path)
    return path


def _detector(predictor, **config_overrides) -> Sam3ExemplarDetector:
    """Default grid: tile_target_px=0 (disables adaptive sizing) with
    tile_cols=2/tile_rows=1 -- on a 200x100 test image this gives exactly
    two 100x100 tiles, no overlap, deterministic and easy to reason about
    (see TileSplitter._adaptive_grid's own `if tile_target_px > 0` branch)."""
    cfg = Sam3ExemplarDetectorConfig(
        seed_confidence_threshold=config_overrides.pop("seed_confidence_threshold", 0.75),
        sam3_confidence_threshold=config_overrides.pop("sam3_confidence_threshold", 0.70),
        tile_px=config_overrides.pop("tile_px", 100),
        tile_cols=config_overrides.pop("tile_cols", 2),
        tile_rows=config_overrides.pop("tile_rows", 1),
        tile_overlap_pct=config_overrides.pop("tile_overlap_pct", 0.0),
        tile_target_px=config_overrides.pop("tile_target_px", 0),
        dedup_iou=config_overrides.pop("dedup_iou", 0.6),
        seed_overlap_iou=config_overrides.pop("seed_overlap_iou", 0.5),
        min_seed_area_ratio=config_overrides.pop("min_seed_area_ratio", 0.5),
    )
    assert not config_overrides, f"unknown overrides: {config_overrides}"
    return Sam3ExemplarDetector(config=cfg, predictor=predictor)


def test_seeds_below_threshold_never_call_predictor(tmp_path):
    img = _make_image(tmp_path, 300, 300)
    fake = _FakeSam3Predictor(responses=[])
    det = _detector(fake, seed_confidence_threshold=0.75)

    seeds = [_seed((10, 10, 30, 30), confidence=0.5)]  # below threshold
    result = det.detect_additional(img, seeds)

    assert result == []
    assert fake.call_bboxes == [], "predictor must not be invoked with no seeds"


def test_non_door_seeds_ignored(tmp_path):
    img = _make_image(tmp_path, 300, 300)
    fake = _FakeSam3Predictor(responses=[])
    det = _detector(fake)

    seeds = [_seed((10, 10, 30, 30), confidence=0.95, category="window")]
    result = det.detect_additional(img, seeds)

    assert result == []
    assert fake.call_bboxes == []


def test_sam3_box_overlapping_seed_is_dropped(tmp_path):
    img = _make_image(tmp_path, 200, 100)
    # Grid (2x1, no overlap): tile(0,0)=[0,100)x[0,100), tile(1,0)=[100,200)x[0,100)
    seed_bbox = (10.0, 10.0, 30.0, 30.0)  # fully inside tile(0,0), origin (0,0)
    local_seed = (10.0, 10.0, 30.0, 30.0)  # same box, tile origin (0,0) -> unchanged
    fake = _FakeSam3Predictor(responses=[([list(local_seed)], [0.9])])
    det = _detector(fake, seed_overlap_iou=0.5)

    seeds = [_seed(seed_bbox, confidence=0.9)]
    result = det.detect_additional(img, seeds)

    assert result == [], "a box re-finding the seed itself is not a new detection"


def test_dedup_across_tiles_keeps_higher_confidence(tmp_path):
    img = _make_image(tmp_path, 200, 100)
    # Seed A fully inside tile(0,0) [0,100)x[0,100), origin (0,0)
    # Seed B fully inside tile(1,0) [100,200)x[0,100), origin (100,0)
    seeds = [_seed((10, 10, 30, 30), confidence=0.9), _seed((110, 10, 130, 30), confidence=0.85)]
    # Both tiles' local boxes map to the SAME page box P=(140,40,160,60):
    #   tile A origin (0,0)   -> local (140,40,160,60)
    #   tile B origin (100,0) -> local (40,40,60,60)
    fake = _FakeSam3Predictor(
        responses=[
            ([[140.0, 40.0, 160.0, 60.0]], [0.6]),
            ([[40.0, 40.0, 60.0, 60.0]], [0.9]),
        ]
    )
    det = _detector(fake, dedup_iou=0.6, seed_overlap_iou=0.5)

    result = det.detect_additional(img, seeds)

    assert len(result) == 1, "identical page-space box from two tiles must dedup to one"
    assert result[0].confidence == pytest.approx(0.9), "higher-confidence duplicate must win"
    assert result[0].bbox == pytest.approx((140.0, 40.0, 160.0, 60.0))


def test_tile_local_to_page_coordinate_mapping(tmp_path):
    img = _make_image(tmp_path, 200, 100)
    # Seed fully inside tile(1,0) [100,200)x[0,100), origin (100,0)
    seeds = [_seed((110.0, 10.0, 130.0, 30.0), confidence=0.9)]
    # Fake returns a novel local box (5,5,25,25) inside the tile.
    fake = _FakeSam3Predictor(responses=[([[5.0, 5.0, 25.0, 25.0]], [0.8])])
    det = _detector(fake)

    result = det.detect_additional(img, seeds)

    assert len(result) == 1
    # page box = local + tile origin (100, 0)
    assert result[0].bbox == pytest.approx((105.0, 5.0, 125.0, 25.0))


def test_tile_origin_is_image_derived_not_seed_derived(tmp_path):
    """Two different seeds, both fully inside the SAME tile, must resolve
    through the SAME tile origin. This is the property the systematic-grid
    rewrite exists to guarantee -- the old seed-centred design computed a
    fresh origin per seed (cx - tile_px/2), so two seeds in the same
    visual area could get different, seed-position-dependent crops."""
    img = _make_image(tmp_path, 200, 100)
    seed_a = _seed((105.0, 5.0, 115.0, 15.0), confidence=0.9)   # top-left of tile(1,0)
    seed_b = _seed((180.0, 80.0, 195.0, 95.0), confidence=0.85)  # bottom-right of tile(1,0)
    # Same fake local box for both calls, mapping to a page box DISTINCT
    # from either seed's own bbox (so it isn't dropped as "rediscovered
    # the seed itself") -> if origin were seed-derived, the two calls
    # would land on different page boxes; both landing on the SAME one
    # proves the origin is the tile's, not the seed's.
    fake = _FakeSam3Predictor(
        responses=[([[50.0, 40.0, 60.0, 50.0]], [0.8]), ([[50.0, 40.0, 60.0, 50.0]], [0.79])]
    )
    det = _detector(fake, dedup_iou=0.99, seed_overlap_iou=0.5)

    result = det.detect_additional(img, [seed_a, seed_b])

    # Both calls used tile(1,0)'s origin (100, 0) -> both map to the
    # identical page box (150,40,160,50) -> dedup collapses them to one.
    assert len(result) == 1
    assert result[0].bbox == pytest.approx((150.0, 40.0, 160.0, 50.0))


def test_seed_straddling_tile_boundary_is_skipped_no_crash(tmp_path):
    """A seed not fully contained in ANY tile (straddles the boundary at
    x=100) gets no tile to seed from -- skipped, not a crash. Real
    limitation of full-containment matching; the 10% overlap used in
    production (not this 0%-overlap test) makes this rare in practice."""
    img = _make_image(tmp_path, 200, 100)
    seeds = [_seed((90.0, 10.0, 110.0, 30.0), confidence=0.9)]  # spans x=90..110, crosses x=100
    fake = _FakeSam3Predictor(responses=[])
    det = _detector(fake)

    result = det.detect_additional(img, seeds)

    assert result == []
    assert fake.call_bboxes == [], "a straddling seed must not reach the predictor at all"


def test_multiple_seeds_in_one_tile_call_predictor_separately(tmp_path):
    """Regression guard: multi-exemplar (multiple boxes in one predictor
    call) was tested against the real false-positive cascade and made it
    WORSE (K=1 -> 1 instance, K=2 -> 23 instances, same real tile). Two
    seeds in the same tile must produce TWO separate single-exemplar
    calls, never one call with both boxes."""
    img = _make_image(tmp_path, 200, 100)
    seed_a = _seed((10.0, 10.0, 30.0, 30.0), confidence=0.9)
    seed_b = _seed((50.0, 10.0, 70.0, 30.0), confidence=0.85)  # different spot, same tile(0,0)
    fake = _FakeSam3Predictor(responses=[None, None])  # no boxes back, only call shape matters here

    det = _detector(fake)
    det.detect_additional(img, [seed_a, seed_b])

    assert len(fake.call_bboxes) == 2, "one predictor call per seed, not one call for the tile"
    for call in fake.call_bboxes:
        assert len(call) == 1, f"each call must carry exactly one exemplar box, got {len(call)}"


def test_predictor_exception_is_caught_and_returns_empty(tmp_path):
    img = _make_image(tmp_path, 200, 100)
    seeds = [_seed((10.0, 10.0, 30.0, 30.0), confidence=0.9)]  # fully inside tile(0,0)
    det = _detector(_RaisingPredictor())

    result = det.detect_additional(img, seeds)  # must not raise

    assert result == []


def test_addition_overlapping_low_confidence_door_is_dropped(tmp_path):
    """P13 regression guard: dedup must run against EVERY door the
    caller passed in, not just the >= seed_confidence_threshold subset
    used as exemplars. A door YOLO found at confidence 0.5-0.75 is not
    an exemplar but IS in the caller's union, so an addition re-finding
    it is a duplicate, not a new detection. Measured on real GT before
    this fix: 4 of 13 marginal false positives on test_pcs were exactly
    this (IoU 0.67-0.95 against a YOLO door box, all above
    seed_overlap_iou, none dropped)."""
    img = _make_image(tmp_path, 200, 100)
    # Exemplar-grade seed in tile(0,0) -- drives the predictor call.
    seed = _seed((10.0, 10.0, 30.0, 30.0), confidence=0.9)
    # Low-confidence door: below seed_confidence_threshold, so NOT an
    # exemplar, but still a real detection the caller will union.
    low_conf_door = _seed((50.0, 10.0, 70.0, 30.0), confidence=0.6)
    # SAM3 "finds" that same low-confidence door's box again.
    fake = _FakeSam3Predictor(responses=[([[50.0, 10.0, 70.0, 30.0]], [0.8])])
    det = _detector(fake, seed_confidence_threshold=0.75, seed_overlap_iou=0.5)

    result = det.detect_additional(img, [seed, low_conf_door])

    assert result == [], (
        "an addition duplicating an already-detected low-confidence door "
        "must be dropped, not returned as a new detection"
    )


def test_low_confidence_door_is_not_used_as_exemplar(tmp_path):
    """Companion to the above: widening the DEDUP reference to all
    doors must NOT widen the EXEMPLAR set. A sub-threshold door still
    never reaches the predictor -- weak exemplars are what seed the
    false-positive cascade."""
    img = _make_image(tmp_path, 200, 100)
    low_conf_door = _seed((110.0, 10.0, 130.0, 30.0), confidence=0.6)
    fake = _FakeSam3Predictor(responses=[])

    det = _detector(fake, seed_confidence_threshold=0.75)
    result = det.detect_additional(img, [low_conf_door])

    assert result == []
    assert fake.call_bboxes == [], "sub-threshold door must not become an exemplar"


def test_scale_consistency_gate_discards_undersized_additions(tmp_path):
    """5th guardrail (Audit #16 P2): additions systematically smaller
    than the seeds that triggered them are a signature of the
    real-world false-positive cascade (SAM3 confusing an MEP symbol for
    a door swing) -- discard ALL additions for the page, not just the
    small ones, matching the existing all-or-nothing guardrail family."""
    img = _make_image(tmp_path, 200, 100)
    seed = _seed((10.0, 10.0, 30.0, 30.0), confidence=0.9)  # 20x20 = 400 area
    # local box (5,5,15,15) = 10x10 = 100 area -> ratio 100/400 = 0.25 < 0.5
    fake = _FakeSam3Predictor(responses=[([[5.0, 5.0, 15.0, 15.0]], [0.8])])
    det = _detector(fake, min_seed_area_ratio=0.5)

    result = det.detect_additional(img, [seed])

    assert result == [], "undersized addition must be discarded by the scale gate"


def test_scale_consistency_gate_keeps_comparable_additions(tmp_path):
    """Additions comparable in size to their seeds must survive the gate
    -- this is the ROCKAWAY p001 real-page shape (ratio 0.774, 4 real
    doors kept), not just the Lake Shore cascade shape (ratio 0.376,
    discarded, see test above)."""
    img = _make_image(tmp_path, 200, 100)
    seed = _seed((10.0, 10.0, 30.0, 30.0), confidence=0.9)  # 20x20 = 400 area
    # local box (105,5,125,25) in tile(1,0) -> page (205... out of range;
    # use a same-tile-relative box far enough from the seed to not be
    # dropped as "rediscovered the seed itself".
    fake = _FakeSam3Predictor(responses=[([[50.0, 5.0, 70.0, 25.0]], [0.8])])
    det = _detector(fake, min_seed_area_ratio=0.5)

    result = det.detect_additional(img, [seed])

    assert len(result) == 1, "comparably-sized addition must survive the scale gate"


def test_output_provenance_and_category_tag(tmp_path):
    img = _make_image(tmp_path, 200, 100)
    seeds = [_seed((110.0, 10.0, 130.0, 30.0), confidence=0.9)]  # fully inside tile(1,0)
    fake = _FakeSam3Predictor(responses=[([[5.0, 5.0, 25.0, 25.0]], [0.8])])
    det = _detector(fake)

    result = det.detect_additional(img, seeds)

    assert len(result) == 1
    detection = result[0]
    assert detection.source_tier == WindowDetectionTier.SAM3_EXEMPLAR
    assert detection.metadata == {"type": "door"}
