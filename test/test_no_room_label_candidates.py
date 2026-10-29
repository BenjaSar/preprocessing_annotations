"""P1 (2026-10-11): reason code for pages with no vocabulary room-name candidate.

Census basis: plan/technology-evaluation-2026-10-11-no-label-fallback.md section 2.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing_annotations.ingestion.ocr_extractor import _is_excluded_token  # noqa: E402
from preprocessing_annotations.automation.sft_validator import (
    NO_ROOM_NAME_CANDIDATES,
    dropped_page_record,
    has_room_name_candidate,
)


def _ocr(name):
    return {"name": name, "bbox": [100, 100, 80, 30], "source": "ocr_only", "confidence": 0.9}


def _annotation(**keys):
    return {"image_file": "p.png", **keys}


@pytest.mark.parametrize("names", [
    [],                                   # PLAN_6_page003: nothing detected
    ["LP#15", "EM", "23"],                # APEX: fixture / circuit tags
    ["LT-01", "C1-37", "C1-39"],          # Buffalo Run: fixture + circuit tags
    ["6a", "9h", "NL/EM"],                # Prager U: fixture types
    ["RI-A", "P-01", "EFAN"],             # Lesisure Town RCP: fixture + finish tags
])
def test_mep_tags_only_has_no_candidate(names):
    ann = _annotation(ocr_rooms=[_ocr(n) for n in names])
    assert has_room_name_candidate(ann) is False


@pytest.mark.parametrize("name", ["STORAGE ROOM", "ENTRY LOBBY", "STAIR #2"])
def test_vocabulary_room_name_is_candidate(name):
    ann = _annotation(ocr_rooms=[_ocr("LP#15"), _ocr(name)])
    assert has_room_name_candidate(ann) is True


def test_candidate_in_any_raw_key_counts():
    for key in ("rooms", "ocr_rooms", "roomsRecognized"):
        entry = {"room_name": "STORAGE ROOM", "bbox": [10, 10, 80, 30]}
        assert has_room_name_candidate(_annotation(**{key: [entry]})) is True


def test_record_none_when_candidate_and_no_drops():
    assert dropped_page_record(_annotation(), has_name_candidate=True) is None


def test_record_unchanged_shape_when_filter_drops_and_candidate():
    ann = _annotation(drop_attribution={"low_ink": 3}, drop_detail={"low_ink": [{"room_name": "X"}]})
    assert dropped_page_record(ann, has_name_candidate=True) == {
        "image_file": "p.png",
        "drop_attribution": {"low_ink": 3},
        "drop_detail": {"low_ink": [{"room_name": "X"}]},
    }


def test_record_written_for_no_candidate_page_without_drops():
    assert dropped_page_record(_annotation(), has_name_candidate=False) == {
        "image_file": "p.png",
        "drop_attribution": {},
        "page_reason": NO_ROOM_NAME_CANDIDATES,
    }


def test_record_adds_reason_alongside_drops():
    ann = _annotation(drop_attribution={"semantic": 2})
    record = dropped_page_record(ann, has_name_candidate=False)
    assert record["drop_attribution"] == {"semantic": 2}
    assert record["page_reason"] == NO_ROOM_NAME_CANDIDATES


class TestLegendHeaderExclusion:
    """F3 (2026-10-16): a lighting-legend header row seeds an exclusion zone.

    sprint1_verify55 "Lesisure Town Clubhouse Lighting_page005": the left
    strip is a legend (MARK / VOLT / MOUNT / WALL-CEILING over rows of
    120 / SURFACE / HANGING / FAN). LEGEND was the only excluded token there
    -- one token, under compute_exclusion_zones' 3-token minimum -- so no
    zone formed and all 5 of that page's zones sat at x >= 1304.

    Scoped to header words, never row contents: a zone still needs several of
    these clustered, so a stray MOUNT in a drawing cannot create one alone.
    """

    @pytest.mark.parametrize("header", ["MARK", "VOLT", "MOUNT", "WALL/CEILING"])
    def test_legend_headers_are_excluded_tokens(self, header):
        assert _is_excluded_token(header) is True

    @pytest.mark.parametrize("room", ["BREAK ROOM", "STAIR 2", "KITCHEN", "OFFICE"])
    def test_real_room_names_are_not_excluded(self, room):
        assert _is_excluded_token(room) is False

    def test_header_match_is_whole_token_not_substring(self):
        # "MOUNTING ROOM" must not be excluded just for starting with MOUNT.
        assert _is_excluded_token("MOUNTING ROOM") is False
