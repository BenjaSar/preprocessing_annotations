"""
Bounding box visualization for debugging VLM output.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Q4: object-detection overlay colors (door/window/etc). Categories match
# window_detector.py::map_detections_to_rooms' own type dispatch -- not
# invented here, just given a color. "other" covers any unrecognized
# category rather than silently dropping it.
OBJECT_COLOR_MAP = {
    "door": (220, 20, 60),        # Crimson
    "window": (30, 144, 255),     # Dodger blue
    "skylight": (255, 215, 0),    # Gold
    "opening": (255, 140, 0),     # Dark orange
    "toilet": (147, 112, 219),    # Medium purple
    "bathtub": (0, 206, 209),     # Dark turquoise
    "sink": (60, 179, 113),       # Medium sea green
    "other": (128, 128, 128),     # Gray
}


def _attr(obj, name: str, default=None):
    """Get attribute from dataclass or key from dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


class BboxVisualizer:
    """Draw bounding boxes on images for visual inspection."""

    def __init__(self, output_dir: str = None):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save overlay images. If None, no output.
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def draw_bboxes(
        self,
        image_path: str,
        rooms: List[Dict[str, Any]],
        image_name: str = None,
        stage: str = "raw",
        detections: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        Draw bounding boxes on image and save overlay.

        Args:
            image_path: Path to source image
            rooms: List of room dicts with 'bbox' and 'room_type' keys
            image_name: Stem of image for output filename. If None, derived from image_path.
            stage: Stage label for output filename (e.g., 'raw_vlm', 'post_hallucination')
            detections: Optional object-detection list (Q4, pipeline.py's
                data["objectDetections"] -- {category, bbox, ...} with bbox
                in [x1,y1,x2,y2] pixel format. NOT the same convention as
                rooms' bbox (xywh) -- drawn in a separate loop below, never
                interpreted through the room path. Default None/empty ->
                identical output to before this parameter existed.

        Returns:
            True if visualization saved, False if skipped or error
        """
        if not self.output_dir:
            logger.debug("BboxVisualizer.draw_bboxes: no output_dir configured, skipping")
            return False

        try:
            image = Image.open(image_path).convert("RGB")
            img_width, img_height = image.size
            draw = ImageDraw.Draw(image)

            # Try to use a small font; fall back to default if unavailable
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            except (IOError, OSError):
                font = ImageFont.load_default()

            color_map = {
                "PRIVATE OFFICE": (255, 0, 0),      # Red
                "OPEN OFFICE": (0, 190, 210),       # Cyan
                "CONFERENCE": (0, 255, 0),          # Green
                "CORRIDOR": (0, 0, 255),            # Blue
                "LOBBY": (255, 127, 0),             # Orange
                "RESTROOM": (255, 255, 0),          # Yellow
                "STAIRWELL": (255, 0, 255),         # Magenta
                "MECHANICAL": (0, 255, 255),        # Cyan
                "ELECTRICAL ROOM": (128, 0, 0),     # Dark red
                "STORAGE ROOM": (166, 86, 40),      # Brown
                "RESIDENTIAL UNIT": (55, 126, 184), # Steel blue
                "other": (128, 128, 128),           # Gray
            }

            for i, room in enumerate(rooms):
                bbox = _attr(room, "bbox")
                if not bbox or len(bbox) < 4:
                    continue

                room_type = (
                    _attr(room, "type")
                    or _attr(room, "category")
                    or _attr(room, "room_type")
                    or "other"
                )
                color = color_map.get(room_type, color_map["other"])

                try:
                    # Interpret bbox as [x, y, w, h] (xywh format)
                    x, y, w, h = bbox[:4]
                    x1, y1 = int(x), int(y)
                    x2, y2 = int(x + w), int(y + h)

                    # Clamp to image bounds
                    x1 = max(0, min(x1, img_width))
                    y1 = max(0, min(y1, img_height))
                    x2 = max(0, min(x2, img_width))
                    y2 = max(0, min(y2, img_height))

                    # Draw rectangle
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

                    # Draw label
                    label = f"{i}:{room_type[:4]}"
                    draw.text((x1 + 2, y1 + 2), label, fill=color, font=font)

                except (ValueError, TypeError):
                    logger.debug(f"Skipping room {i}: invalid bbox {bbox}")
                    continue

            # Q4: object detections (door/window/etc) -- separate loop,
            # separate bbox convention (xyxy, not rooms' xywh above).
            for i, det in enumerate(detections or []):
                bbox = det.get("bbox")
                if not bbox or len(bbox) < 4:
                    continue

                category = det.get("category") or "other"
                color = OBJECT_COLOR_MAP.get(category, OBJECT_COLOR_MAP["other"])
                # tech-eval plan: sam3_exemplar additions were reaching this
                # overlay already (source_tier is on every detection dict,
                # _collect_object_detections) but rendered identically to
                # every other tier -- a human reviewing real production
                # pages had no way to tell a recovered door from a YOLO
                # door. Marked distinctly, same category color, so
                # door/window identity is still readable.
                #
                # CORRECTION (found via a real review this rendering
                # itself caused): a first version used width=4 + an
                # inside-the-box label. On a ~41x46px door symbol that
                # outline and label occluded the door-swing arc itself --
                # produced a real false negative (reviewer, not model:
                # reported "no arc visible" on a detection that does have
                # one, confirmed by cropping the source image directly).
                # width stays 2 for every tier; sam3_exemplar is marked by
                # an OUTSET rectangle (drawn a few px outside the bbox, so
                # it brackets the symbol instead of covering it) and the
                # label is placed ABOVE the box instead of inside it.
                is_sam3_exemplar = det.get("source_tier") == "sam3_exemplar"
                _OUTSET_PX = 3

                try:
                    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]

                    # Clamp to image bounds
                    x1 = max(0, min(x1, img_width))
                    y1 = max(0, min(y1, img_height))
                    x2 = max(0, min(x2, img_width))
                    y2 = max(0, min(y2, img_height))

                    if is_sam3_exemplar:
                        rx1 = max(0, x1 - _OUTSET_PX)
                        ry1 = max(0, y1 - _OUTSET_PX)
                        rx2 = min(img_width, x2 + _OUTSET_PX)
                        ry2 = min(img_height, y2 + _OUTSET_PX)
                    else:
                        rx1, ry1, rx2, ry2 = x1, y1, x2, y2
                    draw.rectangle([rx1, ry1, rx2, ry2], outline=color, width=2)

                    label = f"{category[:6]}" + ("+S3" if is_sam3_exemplar else "")
                    if is_sam3_exemplar:
                        # Label above the (outset) box, not inside it --
                        # inside-the-box placement is what occluded the
                        # symbol before this fix. Every other tier keeps
                        # its original inside-top placement unchanged
                        # (see integration contract: non-sam3_exemplar
                        # rendering must not deviate from pre-fix output).
                        text_bbox = draw.textbbox((0, 0), label, font=font)
                        text_h = text_bbox[3] - text_bbox[1]
                        label_y = ry1 - text_h - 2
                        if label_y < 0:
                            # Box near the page top -- no room above it;
                            # fall back to inside-top (rare, still better
                            # than drawing off-canvas).
                            label_y = ry1 + 2
                    else:
                        label_y = ry1 + 2
                    draw.text((rx1 + 2, label_y), label, fill=color, font=font)

                except (ValueError, TypeError):
                    logger.debug(f"Skipping detection {i}: invalid bbox {bbox}")
                    continue

            # Save overlay
            if image_name is None:
                image_name = Path(image_path).stem

            output_path = self.output_dir / f"{image_name}_overlay_{stage}.png"
            image.save(output_path)
            logger.info(f"Saved overlay: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to draw bboxes for {image_path}: {e}")
            return False
