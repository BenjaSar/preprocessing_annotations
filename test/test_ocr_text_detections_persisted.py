"""Raw Pass-1 OCR detections are written to the raw annotation as
`ocrTextDetections` (replay input for candidate generation) and removed by
prepare_sft_annotation, so processed annotations and exports are unchanged."""

import json

from PIL import Image

from preprocessing_annotations.automation.sft_validator import prepare_sft_annotation
from preprocessing_annotations.config import PipelineConfig
from preprocessing_annotations.ingestion.ocr_extractor import TextDetection
from preprocessing_annotations.orchestration.pipeline import AnnotationPipeline
from preprocessing_annotations.vlm.vlm_annotator import VLMAnnotationResult

DETECTION = TextDetection(
    bbox=[[10.0, 20.0], [50.0, 20.0], [50.0, 30.0], [10.0, 30.0]], text="MAIN OFFICE", confidence=0.97,
)
SERIALIZED = [{"text": "MAIN OFFICE", "bbox": [[10.0, 20.0], [50.0, 20.0], [50.0, 30.0], [10.0, 30.0]],
               "confidence": 0.97}]


def test_save_annotation_writes_ocr_text_detections(tmp_path):
    out = tmp_path / "p.json"
    result = VLMAnnotationResult(
        image_file="p.png", image_size={"width": 100, "height": 100},
        rooms=[], panels=[], electrical_counts={},
    )
    AnnotationPipeline(PipelineConfig())._save_annotation(
        result, out, ocr_rooms=[], img_path=None, ocr_text_detections=[DETECTION],
    )
    assert json.loads(out.read_text())["ocrTextDetections"] == SERIALIZED


def test_save_ocr_annotation_writes_ocr_text_detections(tmp_path):
    img = tmp_path / "p.png"
    Image.new("RGB", (100, 100), "white").save(img)
    AnnotationPipeline(PipelineConfig())._save_ocr_annotation(
        img, [], tmp_path, ocr_text_detections=[DETECTION],
    )
    assert json.loads((tmp_path / "p.json").read_text())["ocrTextDetections"] == SERIALIZED


def test_key_absent_when_not_passed(tmp_path):
    out = tmp_path / "p.json"
    result = VLMAnnotationResult(
        image_file="p.png", image_size={"width": 100, "height": 100},
        rooms=[], panels=[], electrical_counts={},
    )
    AnnotationPipeline(PipelineConfig())._save_annotation(result, out, ocr_rooms=[], img_path=None)
    assert "ocrTextDetections" not in json.loads(out.read_text())


def test_prepare_sft_annotation_drops_ocr_text_detections():
    annotation = {
        "image_file": "p.png", "image_size": {"width": 100, "height": 100},
        "rooms": [], "ocr_rooms": [], "ocrTextDetections": SERIALIZED,
    }
    assert "ocrTextDetections" not in prepare_sft_annotation(annotation)
