#!/usr/bin/env python3
"""
AV-1 ObjectDetectionCOCOExporter Tests.

Covers:
  - basic export: xyxy bbox -> COCO xywh, category id space built from
    used categories only
  - own category id space, independent of any room-category numbering
  - min_confidence filters low-confidence detections
  - invalid bbox / missing category / degenerate bbox are skipped, not
    exported
  - a processed annotation with no objectDetections at all contributes
    no image entry (matches COCOExporter's own min-annotation-per-image
    spirit: no empty images in the export)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing_annotations.export.object_exporter import ObjectDetectionCOCOExporter


def _write_annotation(processed_dir: Path, name: str, image_file: str,
                       width: int, height: int, detections: list) -> None:
    (processed_dir / f"{name}.json").write_text(json.dumps({
        "image_file": image_file,
        "image_size": {"width": width, "height": height},
        "objectDetections": detections,
    }))


def test_basic_export_xyxy_to_xywh(tmp_path):
    processed_dir = tmp_path / "processed_annotations"
    processed_dir.mkdir()
    _write_annotation(
        processed_dir, "page000", "page000.png", 4500, 3000,
        [{"category": "door", "bbox": [100, 200, 150, 260],
          "confidence": 0.82, "source_tier": "yolo"}],
    )

    exporter = ObjectDetectionCOCOExporter()
    n_images = exporter.export(processed_dir, tmp_path / "coco_objects.json")

    assert n_images == 1
    coco = json.loads((tmp_path / "coco_objects.json").read_text())
    assert coco["categories"] == [{"id": 1, "name": "door", "supercategory": "object"}]
    assert len(coco["annotations"]) == 1
    ann = coco["annotations"][0]
    assert ann["bbox"] == [100.0, 200.0, 50.0, 60.0]  # xyxy -> xywh
    assert ann["area"] == 50.0 * 60.0
    assert ann["category_id"] == 1
    assert ann["attributes"]["confidence"] == 0.82
    assert ann["attributes"]["source_tier"] == "yolo"


def test_own_category_id_space(tmp_path):
    """Category ids come only from objectDetections' own vocabulary,
    sorted alphabetically -- independent of any room-type numbering
    (this exporter takes no room-category argument at all)."""
    processed_dir = tmp_path / "processed_annotations"
    processed_dir.mkdir()
    _write_annotation(
        processed_dir, "page000", "page000.png", 1000, 1000,
        [
            {"category": "window", "bbox": [0, 0, 10, 10], "confidence": 0.9, "source_tier": "yolo"},
            {"category": "door", "bbox": [20, 20, 30, 30], "confidence": 0.9, "source_tier": "yolo"},
        ],
    )

    exporter = ObjectDetectionCOCOExporter()
    exporter.export(processed_dir, tmp_path / "coco_objects.json")

    coco = json.loads((tmp_path / "coco_objects.json").read_text())
    assert coco["categories"] == [
        {"id": 1, "name": "door", "supercategory": "object"},
        {"id": 2, "name": "window", "supercategory": "object"},
    ]


def test_min_confidence_filters_low_confidence_detections(tmp_path):
    processed_dir = tmp_path / "processed_annotations"
    processed_dir.mkdir()
    _write_annotation(
        processed_dir, "page000", "page000.png", 1000, 1000,
        [
            {"category": "door", "bbox": [0, 0, 10, 10], "confidence": 0.9, "source_tier": "yolo"},
            {"category": "window", "bbox": [20, 20, 30, 30], "confidence": 0.1, "source_tier": "yolo"},
        ],
    )

    exporter = ObjectDetectionCOCOExporter()
    exporter.export(processed_dir, tmp_path / "coco_objects.json", min_confidence=0.5)

    coco = json.loads((tmp_path / "coco_objects.json").read_text())
    assert len(coco["annotations"]) == 1
    assert coco["categories"] == [{"id": 1, "name": "door", "supercategory": "object"}]


def test_invalid_detections_skipped(tmp_path):
    processed_dir = tmp_path / "processed_annotations"
    processed_dir.mkdir()
    _write_annotation(
        processed_dir, "page000", "page000.png", 1000, 1000,
        [
            {"category": "door", "bbox": [0, 0, 10], "confidence": 0.9, "source_tier": "yolo"},  # bad bbox length
            {"category": None, "bbox": [0, 0, 10, 10], "confidence": 0.9, "source_tier": "yolo"},  # no category
            {"category": "door", "bbox": [10, 10, 10, 10], "confidence": 0.9, "source_tier": "yolo"},  # degenerate
            {"category": "door", "bbox": [0, 0, 10, 10], "confidence": 0.9, "source_tier": "yolo"},  # valid
        ],
    )

    exporter = ObjectDetectionCOCOExporter()
    n_images = exporter.export(processed_dir, tmp_path / "coco_objects.json")

    assert n_images == 1
    coco = json.loads((tmp_path / "coco_objects.json").read_text())
    assert len(coco["annotations"]) == 1


def test_annotation_with_no_detections_contributes_no_image(tmp_path):
    processed_dir = tmp_path / "processed_annotations"
    processed_dir.mkdir()
    _write_annotation(processed_dir, "page000", "page000.png", 1000, 1000, [])

    exporter = ObjectDetectionCOCOExporter()
    n_images = exporter.export(processed_dir, tmp_path / "coco_objects.json")

    assert n_images == 0
    coco = json.loads((tmp_path / "coco_objects.json").read_text())
    assert coco["images"] == []
    assert coco["annotations"] == []
