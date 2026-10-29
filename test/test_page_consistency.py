"""Phase 1 (2026-10-14): per-page box-convention consistency, measured
without GT.

Guards the properties that make this harness worth having:
  - it works on pages with no room GT, which is where the defect lives
    (sprint1_verify55: 0/107 pages have GT, 18.2% of rooms are vlm_only)
  - a page mixing label boxes and room-extent boxes is reported as mixed,
    and its size spread is what makes the mix visible
  - rooms predating box_convention are reported as untagged, never folded
    into "single convention"
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))

from preprocessing_annotations.automation.annotation_schema import (  # noqa: E402
    LABEL_BOX,
    ROOM_EXTENT_BOX,
)
from page_consistency import (  # noqa: E402
    area_spread,
    evaluate,
    page_consistency,
    scan_run,
    summarize,
)


def _room(convention, w, h):
    return {"bbox": [0, 0, w, h], "box_convention": convention}


def _annotation(rooms):
    return {"image_size": {"width": 1000, "height": 1000}, "rooms": rooms}


class TestAreaSpread:
    def test_ratio_of_largest_to_median(self):
        assert area_spread([1.0, 2.0, 10.0]) == pytest.approx(5.0)

    def test_fewer_than_two_rooms_has_no_spread(self):
        # None, not 1.0: a one-room page must stay out of the distribution
        # rather than enter it as an artificially consistent page.
        assert area_spread([5.0]) is None
        assert area_spread([]) is None

    def test_zero_median_has_no_spread(self):
        assert area_spread([0.0, 0.0, 7.0]) is None


class TestPageConsistency:
    def test_single_convention_page_is_not_mixed(self):
        page = page_consistency("p", _annotation([
            _room(LABEL_BOX, 10, 10), _room(LABEL_BOX, 20, 20),
        ]))
        assert page.conventions == [LABEL_BOX]
        assert page.is_mixed is False

    def test_two_conventions_on_one_page_is_mixed(self):
        page = page_consistency("p", _annotation([
            _room(LABEL_BOX, 10, 10), _room(ROOM_EXTENT_BOX, 100, 100),
        ]))
        assert sorted(page.conventions) == sorted([LABEL_BOX, ROOM_EXTENT_BOX])
        assert page.is_mixed is True

    def test_mixing_conventions_is_what_widens_the_spread(self):
        """The defect in one assertion: same room count, same label boxes,
        one room swapped to a room-extent box."""
        label_only = page_consistency("p", _annotation([
            _room(LABEL_BOX, 10, 10), _room(LABEL_BOX, 12, 12),
            _room(LABEL_BOX, 14, 14),
        ]))
        mixed = page_consistency("p", _annotation([
            _room(LABEL_BOX, 10, 10), _room(LABEL_BOX, 12, 12),
            _room(ROOM_EXTENT_BOX, 300, 300),
        ]))
        assert mixed.area_spread > label_only.area_spread

    def test_untagged_rooms_are_counted_not_treated_as_a_convention(self):
        page = page_consistency("p", _annotation([
            _room(LABEL_BOX, 10, 10), _room(None, 20, 20),
        ]))
        assert page.conventions == [LABEL_BOX]
        assert page.untagged_rooms == 1

    def test_rooms_with_malformed_bbox_are_skipped(self):
        page = page_consistency("p", _annotation([
            _room(LABEL_BOX, 10, 10), {"bbox": [1, 2], "box_convention": LABEL_BOX},
        ]))
        assert page.rooms == 1


class TestSummarize:
    def test_separates_mixed_from_single_convention_pages(self):
        pages = [
            page_consistency("mixed", _annotation([
                _room(LABEL_BOX, 10, 10), _room(ROOM_EXTENT_BOX, 300, 300),
            ])),
            page_consistency("single", _annotation([
                _room(LABEL_BOX, 10, 10), _room(LABEL_BOX, 12, 12),
            ])),
        ]
        summary = summarize(pages)
        assert summary["pages_mixed"] == 1
        assert summary["mixed_share_of_known"] == pytest.approx(0.5)
        spread = summary["area_spread"]
        assert spread["mixed_convention"]["median"] > spread["single_convention"]["median"]

    def test_untagged_pages_are_not_counted_as_single_convention(self):
        """A run predating box_convention must not read as clean: its pages
        state no convention, so their mix is unknown, not single."""
        pages = [page_consistency("legacy", _annotation([
            _room(None, 10, 10), _room(None, 300, 300),
        ]))]
        summary = summarize(pages)
        assert summary["pages_with_known_convention"] == 0
        assert summary["mixed_share_of_known"] == 0.0
        assert summary["area_spread"]["single_convention"] == {"pages": 0}
        assert summary["area_spread"]["unknown_convention"]["pages"] == 1

    def test_pages_without_measurable_spread_are_excluded_not_zeroed(self):
        pages = [page_consistency("one_room", _annotation([_room(LABEL_BOX, 10, 10)]))]
        summary = summarize(pages)
        assert summary["pages_scanned"] == 1
        assert summary["pages_with_measurable_spread"] == 0
        assert summary["area_spread"]["all"] == {"pages": 0}


class TestEvaluateOnRunDir:
    def test_reads_a_run_and_ranks_widest_spread_first(self, tmp_path):
        processed = tmp_path / "processed_annotations"
        processed.mkdir()
        (processed / "narrow.json").write_text(json.dumps(_annotation([
            _room(LABEL_BOX, 10, 10), _room(LABEL_BOX, 12, 12),
        ])))
        (processed / "wide.json").write_text(json.dumps(_annotation([
            _room(LABEL_BOX, 10, 10), _room(ROOM_EXTENT_BOX, 400, 400),
        ])))

        assert len(scan_run(processed)) == 2
        report = evaluate(tmp_path, worst=10)
        assert report["pages_scanned"] == 2
        assert report["pages_mixed"] == 1
        assert [p["page"] for p in report["widest_spread_pages"]] == ["wide", "narrow"]

    def test_worst_limits_the_listing(self, tmp_path):
        processed = tmp_path / "processed_annotations"
        processed.mkdir()
        for i in range(3):
            (processed / f"p{i}.json").write_text(json.dumps(_annotation([
                _room(LABEL_BOX, 10, 10), _room(ROOM_EXTENT_BOX, 100 + i, 100 + i),
            ])))
        assert len(evaluate(tmp_path, worst=2)["widest_spread_pages"]) == 2
