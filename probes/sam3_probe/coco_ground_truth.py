"""
Reads room ground truth for one image out of an existing COCO annotation file.

Isolated so the application layer doesn't know COCO's on-disk JSON shape.
"""

import json
from pathlib import Path
from typing import List

from .domain import GroundTruthRoom


def load_rooms_for_image(coco_json_path: str | Path, image_file_name: str) -> List[GroundTruthRoom]:
    """Return ground-truth rooms for `image_file_name` from a COCO json file.

    Returns an empty list if the image or annotations aren't found — callers
    treat that as "no ground truth available" (openings have none anywhere).
    """
    with open(coco_json_path) as f:
        coco = json.load(f)

    categories = {c["id"]: c["name"] for c in coco.get("categories", [])}
    matches = [im for im in coco.get("images", []) if im["file_name"] == image_file_name]
    if not matches:
        return []
    image_id = matches[0]["id"]

    rooms = []
    for ann in coco.get("annotations", []):
        if ann["image_id"] != image_id:
            continue
        x, y, w, h = ann["bbox"]
        category = categories.get(ann["category_id"], "unknown")
        rooms.append(GroundTruthRoom(bbox_xywh=(x, y, w, h), category=category))
    return rooms
