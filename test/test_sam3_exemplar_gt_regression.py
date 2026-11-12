#!/usr/bin/env python3
"""
T5 (+ Layer 4): pin the GT-scored numbers for the PRODUCTION exemplar
class (Sam3ExemplarDetector), via the eval tier that now calls it
directly (eval/kaggle_door_window_eval.py --tier sam3_exemplar_tiled,
P5) -- so a future dependency bump or unrelated refactor that silently
changes these numbers gets caught here, not rediscovered by hand.

Marked `slow`: runs real GT scoring (YOLO + SAM3 inference).
--yolo-conf 0.5 is passed explicitly and is NOT optional --
YoloObjectDetectorConfig.confidence_threshold's class default has since
moved to 0.75 (config.py); without pinning it here, this test would
silently start measuring a different baseline.

REPINNED (2026-08-18, P6): the numbers this file pinned before today
came from a duplicate exemplar implementation that had lived in the
eval script itself and silently diverged from Sam3ExemplarDetector
(Audit #16, P5 deleted the duplicate). They were never a measurement of
what ships.

REPINNED AGAIN (2026-08-18, P9, same day): P6's own numbers were taken
under a still-broken grid -- _adaptive_grid's tile_cols/tile_rows cap
(4) sat below what its own ceil() fix asked for on real pages, so P6
measured a 1238px-tile (0.81x downscale) grid despite the P0 fix having
already landed (Audit #16's second-order finding, caught auditing this
very file's numbers). Cap bumped 4->5 (config.py). That still left
precision BELOW baseline on both datasets.

REPINNED A THIRD TIME (2026-08-18, P13): auditing P9's own explanation
reconciled this file's per-box census (20 additions, 11 on GT) against
the scorer's marginal counts (+7 TP) for the first time -- they did not
match. The 4-box gap was a real defect: detect_additional deduped
additions against the >= seed_confidence_threshold EXEMPLAR subset
instead of against every door the caller passed in, so a door YOLO
found at confidence 0.5-0.75 was invisible to the dedup while still
present in the union. SAM3 re-finding it produced a duplicate box that
scored as a false positive (measured: IoU 0.67-0.95 against the YOLO
box, i.e. far above seed_overlap_iou). Fixed in P13; that defect, not
concept-matching error, was a third of test_pcs's marginal FP.

REPINNED A FOURTH TIME (2026-11-04, D3): test_pcs door false positives fell
112 -> 104 (YOLO) and 120 -> 111 (union) with true positives unchanged
(81 / 88). Not a regression and not noise: commit ac9990d (2026-10-10) added
_drop_area_outliers to yolo_detector._merge_per_category, and the tiled
detector this file scores now runs it. It drops a door box whose area is an
outlier against the page's own other doors (D3, 2026-09-07).

Attributed, not guessed. Bisecting the package source across the 12 commits
since these numbers were taken put the change at ac9990d (1271a0f, the first
commit after the 08-18 pins, reproduces FP=112 exactly; so does e966ad0).
Then, in-process, with the filters patched out one at a time:
  both ON                    YOLO FP=104   union FP=111  (union repeated: 111)
  _drop_area_outliers OFF    YOLO FP=112
  _drop_containment_unions OFF only   YOLO FP=104   (removes 1 of 415 boxes)
  BOTH OFF                   YOLO FP=112   union FP=120
_drop_area_outliers removes 8 of 193 door boxes on this set, none of them a
true positive. The other filter added in the same commit,
_drop_containment_unions, is irrelevant here. Three earlier explanations were
tested and refuted before this one: the door-layer filter (this eval path
never applies it), min_area_px / the shape floor (FP stays 104 at
min_area_px=0), and a library or checkpoint change (neither moved since July).
FloorPlanCAD is unaffected: its pins below still hold.

Note the pins were never failing because the model got worse: precision rose
(.4197 -> .4378 YOLO, .4231 -> .4422 union) at identical recall.

CURRENT (post-D3):

test_pcs (10 commercial images, exact reproduction):
  YOLO   P=.437838 R=.223140 TP=81 FP=104
  union  P=.442211 R=.242424 TP=88 FP=111

FloorPlanCAD (40-image sample, residential+commercial -- independent
of test_pcs, where sam3_confidence_threshold=0.70 was picked):
  YOLO   P=.489362 R=.500000 TP=69 FP= 72
  union  P=.506329 R=.579710 TP=80 FP= 78

The union now beats YOLO-alone on BOTH precision AND recall on BOTH
datasets AT THIS TEST'S CONFIGURATION (yolo_conf=0.5) -- the claim
this file originally asserted, then had to withdraw at P6 as an
artifact of a duplicate implementation, is now true of the class that
actually ships, in that configuration. It became true by fixing the
dedup defect, not by re-tuning a threshold.

Caveats that ride with these numbers, do not drop them:
* test_pcs GT is documented KNOWN INCOMPLETE (EvalConfig docstring):
  raw precision understated, recall overstated. Affects baseline and
  union alike, so the COMPARISON holds, but the absolute values do not.
* P13 costs real recall on FloorPlanCAD vs P9 (.609 -> .580): dropping
  duplicates also drops 4 boxes that had been scoring as TP. Still
  above baseline .500.
* sam3_confidence_threshold=0.70 was picked against the withdrawn P6
  numbers and has never been swept against these.
* Small n (10 / 40 images), door only.
* P13's dedup fix is a NO-OP when the seed detector's own confidence
  floor >= Sam3ExemplarDetectorConfig.seed_confidence_threshold
  (0.75) -- true when this was written (YoloObjectDetectorConfig.
  confidence_threshold was also 0.75). STALE as of I-2 (2026-09-01):
  that default is now 0.25, so the fix is NOT a no-op in production
  anymore. Either way, these numbers describe only the yolo_conf=0.5
  regime pinned above (_args()), never production's default at the
  time they were measured or since. Found auditing this file's own C2
  follow-up (2026-08-18), same day as the repin.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))

import argparse

from kaggle_door_window_eval import (
    _GT_SOURCE_BUILDERS,
    _DATASET_TEST_PCS,
    _DATASET_FLOORPLANCAD,
    build_sam3_exemplar_tiled_predict_fn,
    build_yolo_finetuned_tiled_predict_fn,
)
from gt_evaluator import score_against_gt_images

pytestmark = pytest.mark.slow

_YOLO_CHECKPOINT = (
    "/home/ubuntu/floorplan_classifier/VLM/yolo_checkpoints/runs/"
    "kaggle_door_window-6/weights/best.pt"
)


def _args(**overrides):
    base = dict(
        sample_size=None, yolo_checkpoint=_YOLO_CHECKPOINT, yolo_conf=0.5,
        seed_conf=0.75, sam3_conf=0.70, sam3_tile_px=1008,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _skip_if_gt_missing(dataset: str, args: argparse.Namespace):
    try:
        _GT_SOURCE_BUILDERS[dataset](args)
    except Exception as e:
        pytest.skip(f"{dataset} GT not available: {e}")


def _door_scores(dataset: str, args: argparse.Namespace, predict_fn_builder):
    source = _GT_SOURCE_BUILDERS[dataset](args)
    predict_fn = predict_fn_builder(args)
    scores = score_against_gt_images(source.gt_images, source.iou_threshold, predict_fn)
    return scores["door"]


def test_test_pcs_yolo_baseline_pinned():
    args = _args()
    _skip_if_gt_missing(_DATASET_TEST_PCS, args)
    door = _door_scores(_DATASET_TEST_PCS, args, build_yolo_finetuned_tiled_predict_fn)

    assert door.true_positives == 81
    assert door.false_positives == 104
    assert door.precision == pytest.approx(0.43783783783783786, abs=1e-6)
    assert door.recall == pytest.approx(0.2231404958677686, abs=1e-6)


def test_test_pcs_sam3_union_beats_baseline_on_both_axes():
    """P13: union beats YOLO-alone on BOTH precision and recall. This
    assertion was withdrawn at P6 (the number backing it came from a
    duplicate implementation) and stayed false through P9; it became
    true only after the P13 dedup fix. Read this file's header before
    citing it -- the caveats are not optional."""
    args = _args()
    _skip_if_gt_missing(_DATASET_TEST_PCS, args)
    door = _door_scores(_DATASET_TEST_PCS, args, build_sam3_exemplar_tiled_predict_fn)

    assert door.true_positives == 88
    assert door.false_positives == 111
    assert door.precision == pytest.approx(0.44221105527638194, abs=1e-6)
    assert door.recall == pytest.approx(0.24242424242424243, abs=1e-6)

    # Against the CURRENT YOLO-alone baseline (test_test_pcs_yolo_baseline_pinned),
    # not the pre-D3 one: comparing to a stale, weaker baseline would let the
    # claim hold even if the union had stopped helping.
    assert door.precision > 0.43783783783783786
    assert door.recall > 0.2231404958677686


def test_floorplancad_yolo_baseline_pinned():
    """Layer 4: independent-dataset gate, YOLO half. 40-image sample --
    same sample size the tech-eval plan's P1 run used."""
    args = _args(sample_size=40)
    _skip_if_gt_missing(_DATASET_FLOORPLANCAD, args)
    door = _door_scores(_DATASET_FLOORPLANCAD, args, build_yolo_finetuned_tiled_predict_fn)

    assert door.true_positives == 69
    assert door.false_positives == 72
    assert door.precision == pytest.approx(0.48936170212765956, abs=1e-6)
    assert door.recall == pytest.approx(0.5, abs=1e-6)


def test_floorplancad_sam3_union_beats_baseline_on_both_axes():
    """Layer 4: the independent-dataset gate. sam3_confidence_threshold
    =0.70 was picked on test_pcs and never re-swept, so this dataset is
    the check that P13's both-axes result is not a fit to the dataset
    the threshold came from. Note recall here is LOWER than pre-P13
    (.580 vs .609) -- dropping duplicate boxes costs some that had been
    scoring as TP -- but still above baseline on both axes."""
    args = _args(sample_size=40)
    _skip_if_gt_missing(_DATASET_FLOORPLANCAD, args)
    door = _door_scores(_DATASET_FLOORPLANCAD, args, build_sam3_exemplar_tiled_predict_fn)

    assert door.true_positives == 80
    assert door.false_positives == 78
    assert door.precision == pytest.approx(0.506329, abs=1e-6)
    assert door.recall == pytest.approx(0.579710, abs=1e-6)

    assert door.precision > 0.48936170212765956
    assert door.recall > 0.5
