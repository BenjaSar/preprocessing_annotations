"""N2 (2026-10-11): detections the attachment policy attached to no room are
persisted under `unattachedObjectDetections`; `objectDetections` is unchanged
(post-processing re-attaches from it, so widening it would alter room attributes).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing_annotations.config import AttachmentConfig, PipelineConfig
from preprocessing_annotations.detection.attachment import AttachmentPolicyFactory
from preprocessing_annotations.detection.window_detector import (
    WindowDetection,
    WindowDetectionTier,
)
from PIL import Image

from preprocessing_annotations.ingestion.ocr_extractor import RoomCandidate
from preprocessing_annotations.orchestration.pipeline import AnnotationPipeline
from preprocessing_annotations.vlm.vlm_annotator import RoomAnnotation, VLMAnnotationResult


def _det(bbox, kind="door"):
    return WindowDetection(
        bbox=bbox, confidence=0.9, source_tier=WindowDetectionTier.YOLO,
        metadata={"type": kind},
    )


ON_ROOM = _det((45, 45, 55, 55))
OFF_PAGE = _det((900, 900, 910, 910), "window")
ROOMS = [{"id": 0, "bbox": [0, 0, 100, 100]}]


class _StubYolo:
    def __init__(self, detections):
        self._detections = detections

    def detect_objects_tiled(self, img_path):
        return list(self._detections)


def _pipeline(detections):
    config = PipelineConfig()
    config.use_yolo_objects = True
    pipeline = AnnotationPipeline(config)
    pipeline._yolo_object_detector = _StubYolo(detections)
    return pipeline


def test_intersection_policy_leaves_off_room_detection_unattached():
    policy = AttachmentPolicyFactory.create(AttachmentConfig())
    mappings = policy.attach([ON_ROOM, OFF_PAGE], ROOMS)
    attached = {id(w) for m in mappings for w in m.intersecting_windows}
    assert id(ON_ROOM) in attached and id(OFF_PAGE) not in attached


def test_detect_object_mappings_reports_unattached():
    detection_pass = _pipeline([ON_ROOM, OFF_PAGE])._detect_object_mappings(
        Path("p.png"), [{"id": 0, "bbox": [0, 0, 100, 100]}]
    )
    assert detection_pass.unattached == [OFF_PAGE]
    assert detection_pass.mappings[0].intersecting_windows == [ON_ROOM]


def test_serialize_detection_shape():
    assert AnnotationPipeline._serialize_detection(ON_ROOM) == {
        "category": "door", "bbox": [45, 45, 55, 55],
        "confidence": 0.9, "source_tier": "yolo",
    }


def test_save_annotation_writes_both_keys_without_widening_object_detections(
    tmp_path
):
    pipeline = _pipeline([ON_ROOM, OFF_PAGE])
    result = VLMAnnotationResult(
        image_file="p.png", image_size={"width": 1000, "height": 1000},
        rooms=[RoomAnnotation(room_number="1", room_name="OFFICE", category="office", bbox=[0, 0, 100, 100])],
        panels=[], electrical_counts={},
    )
    out = tmp_path / "p.json"

    pipeline._save_annotation(result, out, ocr_rooms=[], img_path=tmp_path / "p.png")

    data = json.loads(out.read_text())
    assert [d["bbox"] for d in data["objectDetections"]] == [[45, 45, 55, 55]]
    assert [d["bbox"] for d in data["unattachedObjectDetections"]] == [[900, 900, 910, 910]]


def test_save_ocr_annotation_writes_both_keys(tmp_path):
    img_path = tmp_path / "p.png"
    Image.new("RGB", (1000, 1000), "white").save(img_path)
    room = RoomCandidate(
        bbox=(0, 0, 100, 100), room_number="1", room_name="OFFICE", confidence=0.9
    )

    _pipeline([ON_ROOM, OFF_PAGE])._save_ocr_annotation(img_path, [room], tmp_path)

    data = json.loads((tmp_path / "p.json").read_text())
    assert [d["bbox"] for d in data["objectDetections"]] == [[45, 45, 55, 55]]
    assert [d["bbox"] for d in data["unattachedObjectDetections"]] == [[900, 900, 910, 910]]
