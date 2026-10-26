"""Door detections that sit on wiring, not on a door (DoorLayerFilterConfig).

CAD-exported lighting plans keep doors and wiring runs on named drawing
layers. YOLO reads the curved wiring arcs between fixtures as door swings;
on a page that has door-layer geometry, a door detection whose box touches
wiring-layer geometry and no door-layer geometry is dropped. Pages without a
door layer ("FLAT" or scanned PDFs) keep every detection.
"""

import re
from pathlib import Path
from typing import List, Tuple

import fitz

from ..config import DoorLayerFilterConfig
from .window_detector import WindowDetection


class DoorLayerError(Exception):
    """The page's drawings could not be read (e.g. PyMuPDF's
    "too many nested graphics states")."""


class DoorLayerFilter:
    """One reason to change: which door detections the PDF layers contradict."""

    def __init__(self, config: DoorLayerFilterConfig):
        self._door = re.compile(config.door_layer_pattern, re.IGNORECASE)
        self._wiring = re.compile(config.wiring_layer_pattern, re.IGNORECASE)

    def _layer_boxes(
        self, pdf_path: Path, page_index: int, image_size: Tuple[int, int],
    ) -> Tuple[List[fitz.Rect], List[fitz.Rect]]:
        """Door and wiring drawing boxes in the pixel space of an image of
        `image_size` (width, height) rendered from this page, rotation included."""
        door, wiring = [], []
        try:
            with fitz.open(pdf_path) as doc:
                page = doc[page_index]
                width, height = image_size
                to_pixels = fitz.Matrix(width / page.rect.width, height / page.rect.height)
                rotation = page.rotation_matrix
                for drawing in page.get_drawings():
                    layer = drawing.get("layer") or ""
                    if self._door.search(layer):
                        door.append(drawing["rect"] * rotation * to_pixels)
                    elif self._wiring.search(layer):
                        wiring.append(drawing["rect"] * rotation * to_pixels)
        except RuntimeError as error:
            raise DoorLayerError(f"{pdf_path.name} page {page_index}: {error}") from error
        return door, wiring

    def split(
        self,
        detections: List[WindowDetection],
        pdf_path: Path,
        page_index: int,
        image_size: Tuple[int, int],
    ) -> Tuple[List[WindowDetection], List[WindowDetection]]:
        """(kept, dropped) detections of one page."""
        door, wiring = self._layer_boxes(pdf_path, page_index, image_size)
        if not door:
            return detections, []
        kept, dropped = [], []
        for detection in detections:
            (dropped if _is_wiring_door(detection, door, wiring) else kept).append(detection)
        return kept, dropped


def _is_wiring_door(
    detection: WindowDetection, door: List[fitz.Rect], wiring: List[fitz.Rect],
) -> bool:
    if (detection.metadata or {}).get("type") != "door":
        return False
    box = fitz.Rect(detection.bbox)
    return not any(box.intersects(r) for r in door) and any(box.intersects(r) for r in wiring)
