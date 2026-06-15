"""
Bounding box visualization for debugging VLM output.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


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
        stage: str = "raw"
    ) -> bool:
        """
        Draw bounding boxes on image and save overlay.

        Args:
            image_path: Path to source image
            rooms: List of room dicts with 'bbox' and 'room_type' keys
            image_name: Stem of image for output filename. If None, derived from image_path.
            stage: Stage label for output filename (e.g., 'raw_vlm', 'post_hallucination')

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
                bbox = room.get("bbox")
                if not bbox or len(bbox) < 4:
                    continue

                # Room dicts produced by the VLM use "type" or "category"
                # (and sometimes both); "room_type" is not a field anywhere
                # in the pipeline schema, so the previous lookup always
                # fell through to the "other" gray.
                room_type = (
                    room.get("type")
                    or room.get("category")
                    or room.get("room_type")
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
