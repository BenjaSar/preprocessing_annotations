"""One vote per document, beside the pooled figure.

Why this exists. The headline recall pools every GT room, so a document
contributes in proportion to its size. Measured 2026-11-03: 326_rockaway holds
559 of the 1497 GT rooms (37%) and scores .249 -- and it scores that because
the detector returns 1 room on a tile holding 66, a limit that four separate
experiments (token budget, truncation handling, tile granularity, input
resolution) all failed to move. Pooled, that one drawing sets the ceiling for
every comparison run against this GT.

The macro figure is reported beside the pooled one, never instead of it.
Dropping a hard document to make a number rise would be gaming; counting each
document once answers a different and also-legitimate question -- how the
pipeline does on a typical document -- and is the figure a per-document
regression actually shows up in.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))

from room_gt_eval import LocalizationScore, macro_average


def _score(located, gt_rooms, predictions=None):
    score = LocalizationScore()
    score.gt_rooms = gt_rooms
    score.gt_rooms_located = located
    score.predictions = gt_rooms if predictions is None else predictions
    score.predictions_center_in_gt = located
    return score


def test_every_document_counts_once_regardless_of_size():
    """The defect this addresses: a 559-room document outvoting nine others."""
    per_document = {"huge": _score(139, 559), "small": _score(9, 10)}

    macro = macro_average(per_document)

    pooled = (139 + 9) / (559 + 10)
    assert macro["label_recall"] == pytest.approx((139 / 559 + 9 / 10) / 2)
    assert macro["label_recall"] > pooled, "pooling is what the huge document dominates"


def test_macro_matches_pooled_when_documents_are_the_same_size():
    per_document = {"a": _score(5, 10), "b": _score(7, 10)}

    assert macro_average(per_document)["label_recall"] == pytest.approx(0.6)


def test_one_document_gives_its_own_score():
    assert macro_average({"only": _score(3, 4)})["label_recall"] == pytest.approx(0.75)


def test_no_documents_is_empty_not_a_crash():
    """A run that scored nothing must not divide by zero."""
    assert macro_average({}) == {}


def test_the_reported_metrics_are_the_comparable_ones():
    """Only rates average meaningfully. Counts (gt_rooms, predictions) would
    produce a nonsense 'average document', so they are deliberately absent."""
    macro = macro_average({"a": _score(5, 10)})

    assert set(macro) == {"label_precision", "label_recall", "label_f1"}


def test_a_document_the_pipeline_fails_still_counts():
    """Not an exclusion mechanism: a zero-recall document drags macro down,
    which is the point -- it must stay visible, just not dominant."""
    with_failure = macro_average({"good": _score(9, 10), "failed": _score(0, 100)})
    without = macro_average({"good": _score(9, 10)})

    assert with_failure["label_recall"] < without["label_recall"]
    assert with_failure["label_recall"] == pytest.approx(0.45)
