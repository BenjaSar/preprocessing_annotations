"""ingestion/crop_reread.py + its reader port/adapter + candidate_replay's
--rereads path. No model is loaded: readers and the backend are faked."""

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from candidate_replay import (
    BASELINE,
    PatternVariant,
    evaluate_variants,
    extractor_with_patterns,
    reread_detections,
)
from preprocessing_annotations.config import CropRereadConfig, PipelineConfig, RoomGtEvalConfig
from preprocessing_annotations.ingestion.crop_reread import (
    CropRereader,
    Reread,
    RereadSelector,
    apply_rereads,
    crop_box,
    is_consistent,
)
from preprocessing_annotations.ingestion.ocr_extractor import MEPTextExtractor, TextDetection
from preprocessing_annotations.orchestration.crop_reread_step import reread_selector
from preprocessing_annotations.vlm.read import UnslothCropTextReader
from preprocessing_annotations.vlm.vlm_backend import UnslothQwenBackend

CONFIG = CropRereadConfig()
KEYWORDS = {"STAIRWELL", "CLASSROOM", "OFFICE"}


def _detection(text, x=100, y=100, w=60, h=20):
    return TextDetection(bbox=[[x, y], [x + w, y], [x + w, y + h], [x, y + h]], text=text, confidence=0.9)


def _selector(labels=("OFFICE",)):
    return RereadSelector(lambda text: text in labels, KEYWORDS, CONFIG)


class _FakeReader:
    def __init__(self, answers):
        self._answers = answers
        self.sizes = []

    def read(self, image):
        self.sizes.append(image.size)
        return self._answers.pop(0)


# --- selector ---------------------------------------------------------------

@pytest.mark.parametrize("text", ["TAIR", "CLASSRODM", "LASSROOM- 129"])
def test_selector_picks_fragments_and_misreads(text):
    assert _selector().selects(text)


@pytest.mark.parametrize("text", ["OFFICE", "AB", "LSPC1", "17", "$"])
def test_selector_skips_labels_short_and_unrelated_text(text):
    assert not _selector().selects(text)


def test_select_returns_detection_indices():
    detections = [_detection("OFFICE"), _detection("TAIR"), _detection("LSPC1")]
    assert _selector().select(detections) == [1]


def test_selector_skips_tokens_inside_an_existing_room_label():
    label = (90, 90, 100, 40)   # x, y, w, h around both detections below
    selector = RereadSelector(lambda text: False, KEYWORDS, CONFIG, lambda detections: [label])
    detections = [_detection("TAIR"), _detection("TAIR", x=500)]
    assert selector.select(detections) == [1]


# Real OCR from sprint1_verify54, BP-1 Kennedy Middle School Electrical
# Pages p002: the extractor merges these into "SPECIAL EDUCATION" #128.
# Re-reading each line separately read the other line too and broke that
# merge -- the in-sample recall loss skip_tokens_in_room_labels fixes.
KENNEDY_STACK = [
    TextDetection([[2029, 802], [2078, 802], [2078, 817], [2029, 817]], "SPECIAL", 0.998),
    TextDetection([[2014, 822], [2085, 822], [2085, 836], [2014, 836]], "EDUCATION", 0.999),
    TextDetection([[2042, 841], [2066, 841], [2066, 857], [2042, 857]], "128", 0.999),
]


def test_reread_selector_leaves_a_merged_stacked_label_alone():
    extractor = MEPTextExtractor(PipelineConfig().ocr)
    unguarded = CropRereadConfig(skip_tokens_in_room_labels=False)
    assert reread_selector(extractor, unguarded).select(KENNEDY_STACK) == [0, 1]
    assert reread_selector(extractor, CONFIG).select(KENNEDY_STACK) == []


# --- geometry, guard, apply -------------------------------------------------

def test_crop_box_pads_by_box_height_and_clamps():
    quad = [[10, 10], [70, 10], [70, 30], [10, 30]]   # height 20
    assert crop_box(quad, (1000, 1000), 1.0) == (0, 0, 90, 50)
    assert crop_box(quad, (80, 40), 1.0) == (0, 0, 80, 40)


@pytest.mark.parametrize("original,reread,expected", [
    ("TAIR_B", "STAIR B", True),        # extension of the fragment
    ("CLASSRODM", "CLASSROOM", True),   # correction of a misread
    ("TYP.", "OFFICE", False),          # unrelated -> refused
    ("TAIR", "", False),                # nothing read
])
def test_consistency_guard(original, reread, expected):
    assert is_consistent(original, reread, CONFIG.consistency_min) is expected


def test_apply_rereads_swaps_only_accepted_text_and_keeps_boxes():
    detections = [_detection("TAIR"), _detection("TYP.", x=500)]
    rereads = [Reread(0, "TAIR", "STAIR"), Reread(1, "TYP.", "OFFICE")]
    result = apply_rereads(detections, rereads, CONFIG.consistency_min)
    assert [d.text for d in result] == ["STAIR", "TYP."]
    assert [d.bbox for d in result] == [d.bbox for d in detections]


def test_apply_rereads_refuses_rereads_of_other_ocr_text():
    with pytest.raises(ValueError, match="another OCR pass"):
        apply_rereads([_detection("TAIR")], [Reread(0, "AIR", "STAIR")], CONFIG.consistency_min)


# --- rereader + reader port -------------------------------------------------

def test_crop_rereader_crops_selected_detections_only():
    reader = _FakeReader(["STAIR"])
    rereader = CropRereader(_selector(), reader, CONFIG)
    image = Image.new("RGB", (1000, 1000), "white")
    rereads = rereader.reread(image, [_detection("OFFICE"), _detection("TAIR")])
    assert rereads == [Reread(1, "TAIR", "STAIR")]
    assert reader.sizes == [(100, 60)]   # 60x20 box padded by 20 on each side


@pytest.mark.parametrize("response,expected", [
    ('["STAIR B"]', "STAIR B"),
    ('Here: ["  CLASSROOM 117 "]', "CLASSROOM 117"),
    ("[]", ""),
    ("[42]", ""),
    ("no array", ""),
])
def test_parse_crop_text_response(response, expected):
    assert UnslothQwenBackend._parse_crop_text_response(response) == expected


def test_unsloth_reader_delegates_to_backend():
    class _FakeBackend:
        def read_crop_text(self, image):
            return "STAIR"

    assert UnslothCropTextReader(_FakeBackend()).read(Image.new("RGB", (4, 4))) == "STAIR"


# --- extractor predicate (DRY refactor of the inline pattern checks) ---------

def test_matches_room_label_is_number_or_name():
    extractor = MEPTextExtractor(PipelineConfig().ocr)
    assert extractor.is_room_name("OFFICE")
    assert extractor.matches_room_label("OFFICE")
    assert not extractor.is_room_name("TAIR")
    assert not extractor.matches_room_label("TAIR")


# --- candidate_replay --rereads ------------------------------------------------

def test_reread_detections_applies_probe_reads_to_selected():
    builder = extractor_with_patterns(PipelineConfig(), PatternVariant())
    detections = [_detection("TAIR"), _detection("OFFICE", x=500)]
    result = reread_detections(detections, [Reread(0, "TAIR", "STAIR")], builder, CONFIG)
    assert [d.text for d in result] == ["STAIR", "OFFICE"]


def test_reread_detections_refuses_selection_wider_than_probe():
    builder = extractor_with_patterns(PipelineConfig(), PatternVariant())
    with pytest.raises(ValueError, match="probe did not read"):
        reread_detections([_detection("TAIR")], [], builder, CONFIG)


def test_evaluate_variants_scores_a_reread_variant(tmp_path):
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "docA_page000.png"}],
        "categories": [{"id": 1, "name": "stairwell"}],
        "annotations": [{"image_id": 1, "category_id": 1, "segmentation": [[90, 90, 320, 90, 320, 140, 90, 140]]}],
    }))
    run = tmp_path / "run"
    (run / "annotations").mkdir(parents=True)
    fragment = {"text": "TAIR", "bbox": [[100, 100], [160, 100], [160, 120], [100, 120]], "confidence": 0.9}
    (run / "annotations" / "docA_page000.json").write_text(json.dumps({
        "image_file": "docA_page000.png", "image_size": {"width": 2000, "height": 2000},
        "rooms": [], "ocr_rooms": [], "ocrTextDetections": [fragment],
    }))

    report = evaluate_variants(
        RoomGtEvalConfig(gt_coco_path=gt_path), run, PipelineConfig(),
        {"reread": PatternVariant(reread=CONFIG)},
        {"docA_page000": [Reread(0, "TAIR", "STAIR")]},
    )

    assert report["candidates"][BASELINE]["localization"]["gt_rooms_located"] == 0
    assert report["candidates"]["reread"]["localization"]["gt_rooms_located"] == 1
    assert report["candidates"]["reread"]["reread"]["consistency_min"] == CONFIG.consistency_min
