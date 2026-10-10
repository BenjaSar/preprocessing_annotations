"""AV-1 (2026-09-23): exports objectDetections (door/window/fixture
geometry) to COCO format.

Confirmed gap: COCOExporter reads ann["rooms"] only (exporters.py) --
every persisted objectDetections entry (door/window boxes from YOLO/
SAM3-exemplar) is invisible to every existing training-data consumer,
so improving the detector today has zero downstream effect.

This is a new, separate exporter -- not an edit to COCOExporter.
COCOExporter's category id space is built from whatever room types it
discovers per export (exporters.py, "Phase 1: Collect all rooms to
determine which categories are used"); injecting object categories into
that same id space would renumber ids out from under any existing
consumer of coco/*.json. ObjectDetectionCOCOExporter owns its own
category id space in its own output file instead.

Source/shape confirmed on-disk (sprint1_verify52/processed_annotations,
2026-09-23): objectDetections entries are
{"category": "door", "bbox": [x1, y1, x2, y2], "confidence": float,
"source_tier": str} -- xyxy, the same convention split
sft_validator.py documents ("rooms are [x, y, w, h]; objectDetections
are [x1, y1, x2, y2]").
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)


class ObjectDetectionCOCOExporter:
    """Exports objectDetections to a standalone COCO JSON file.

    Attributes:
        description: COCO "info.description" field for the output file.
    """

    def __init__(self, description: str = "MEP Floor Plan Object Detections"):
        self.description = description

    def export(
        self,
        processed_dir: Path | str,
        output_file: Path | str,
        min_confidence: float = 0.0,
    ) -> int:
        """Export every processed annotation's objectDetections to COCO.

        Args:
            processed_dir: Directory containing post-processed JSON
                annotations (the same directory COCOExporter reads room
                geometry from).
            output_file: Output path for the COCO JSON file.
            min_confidence: Drop detections below this confidence.
                Default 0.0 keeps every persisted detection -- no
                filtering beyond what the detector/pipeline already
                applied before persisting objectDetections.

        Returns:
            Number of images that contributed at least one object
            annotation.
        """
        processed_dir = Path(processed_dir)
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        all_file_data: List[Dict[str, Any]] = []
        used_categories: Set[str] = set()
        skipped = 0

        for json_file in sorted(processed_dir.glob("*.json")):
            try:
                with open(json_file) as f:
                    ann = json.load(f)
            except Exception as e:
                logger.error(
                    f"ObjectDetectionCOCOExporter: failed to read {json_file}: {e}"
                )
                continue

            detections = ann.get("objectDetections", [])
            if not detections:
                continue

            image_file = ann.get("image_file", json_file.stem + ".png")
            img_size = ann.get("image_size", {})
            w = img_size.get("width", 0)
            h = img_size.get("height", 0)

            valid_detections = []
            for det in detections:
                bbox = det.get("bbox", [])
                category = det.get("category")
                confidence = det.get("confidence", 0.0)
                if len(bbox) != 4 or not category:
                    skipped += 1
                    continue
                if confidence < min_confidence:
                    skipped += 1
                    continue
                x1, y1, x2, y2 = [float(v) for v in bbox]
                if x2 <= x1 or y2 <= y1:
                    skipped += 1
                    continue
                valid_detections.append(det)
                used_categories.add(category)

            if valid_detections:
                all_file_data.append({
                    "image_file": image_file,
                    "w": w,
                    "h": h,
                    "detections": valid_detections,
                })

        if skipped:
            logger.info(
                f"ObjectDetectionCOCOExporter: skipped {skipped} invalid/"
                f"low-confidence detection(s)"
            )

        sorted_categories = sorted(used_categories)
        cat_id_map = {c: i + 1 for i, c in enumerate(sorted_categories)}
        coco: Dict[str, Any] = {
            "info": {"description": self.description, "version": "1.0"},
            "categories": [
                {"id": cat_id_map[c], "name": c, "supercategory": "object"}
                for c in sorted_categories
            ],
            "images": [],
            "annotations": [],
        }

        ann_id = 1
        img_id = 0
        for file_data in all_file_data:
            img_id += 1
            coco["images"].append({
                "id": img_id,
                "file_name": file_data["image_file"],
                "width": file_data["w"],
                "height": file_data["h"],
            })
            for det in file_data["detections"]:
                x1, y1, x2, y2 = [float(v) for v in det["bbox"]]
                bw, bh = x2 - x1, y2 - y1
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id_map[det["category"]],
                    "bbox": [x1, y1, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                    "attributes": {
                        "confidence": det.get("confidence", 0.0),
                        "source_tier": det.get("source_tier", ""),
                    },
                })
                ann_id += 1

        with open(output_file, "w") as f:
            json.dump(coco, f, indent=2)

        logger.info(
            f"ObjectDetectionCOCOExporter: exported {img_id} image(s), "
            f"{ann_id - 1} object annotation(s) -> {output_file}"
        )
        return img_id
