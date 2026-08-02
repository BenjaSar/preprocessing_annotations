"""
Domain types for the SAM3 bake-off spike.

Framework-agnostic: no SAM3, torch, or COCO import here. Adapters translate
into/out of this shape so the application layer never touches vendor types.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class Detection:
    """A single promptable-segmentation result, in xyxy pixel coordinates."""

    bbox_xyxy: tuple[float, float, float, float]
    label: str
    score: float
    mask: Optional[np.ndarray] = None  # binary (H, W), same size as source image

    def to_bbox_metrics_dict(self) -> dict:
        """Shape expected by bbox_metrics.BBox.from_dict / evaluate_bboxes."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return {"bbox": [x1, y1, x2, y2], "room_name": self.label, "confidence": self.score}


@dataclass(frozen=True)
class GroundTruthRoom:
    """One annotated room, in xywh pixel coordinates (native COCO convention)."""

    bbox_xywh: tuple[float, float, float, float]
    category: str

    def to_bbox_metrics_dict(self) -> dict:
        x, y, w, h = self.bbox_xywh
        return {"bbox": [x, y, w, h], "room_name": self.category}


@dataclass(frozen=True)
class BakeOffCase:
    """One image to run through the bake-off: source image + optional ground truth."""

    name: str
    image_path: str
    image_width: int
    image_height: int
    ground_truth: List[GroundTruthRoom]
