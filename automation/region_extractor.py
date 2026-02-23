"""
Region extraction from annotated floor plans.

Extracts individual room regions as separate images for training.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class RegionExtractor:
    """Extract individual room regions from annotated floor plans."""

    def __init__(self, padding_pct: float = 0.1):
        """
        Initialize region extractor.

        Args:
            padding_pct: Percentage of bbox dimensions to add as padding (0.0-1.0)
        """
        self.padding_pct = max(0.0, min(1.0, padding_pct))

    def extract_regions(self,
                       image_path: str,
                       annotation: Dict,
                       output_dir: str,
                       prefix: str = "") -> List[str]:
        """
        Extract room regions from a floor plan image.

        Args:
            image_path: Path to floor plan image
            annotation: Annotation dictionary with rooms and bboxes
            output_dir: Directory to save extracted regions
            prefix: Optional prefix for output filenames

        Returns:
            List of paths to extracted region images
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return []

        img_height, img_width = image.shape[:2]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        extracted = []
        image_stem = Path(image_path).stem

        # Extract each room
        rooms = annotation.get("rooms", [])
        for i, room in enumerate(rooms):

            bbox = room.get("bbox")
            if not bbox or len(bbox) != 4:
                logger.warning(f"Skipping room {i}: invalid bbox")
                continue

            try:
                # FIX: Bbox is stored as [x, y, width, height].
                # Convert to corner coordinates before cropping.
                x, y, w, h = [int(v) for v in bbox]
                x1, y1, x2, y2 = x, y, x + w, y + h
            except (ValueError, TypeError):
                logger.warning(f"Skipping room {i}: non-numeric bbox")
                continue

            # Add padding
            width = x2 - x1
            height = y2 - y1
            pad_x = int(width * self.padding_pct)
            pad_y = int(height * self.padding_pct)

            x1_padded = max(0, x1 - pad_x)
            y1_padded = max(0, y1 - pad_y)
            x2_padded = min(img_width, x2 + pad_x)
            y2_padded = min(img_height, y2 + pad_y)

            # Extract region
            try:
                region = image[y1_padded:y2_padded, x1_padded:x2_padded]

                if region.size == 0:
                    logger.warning(f"Skipping room {i}: empty region after extraction")
                    continue

                # Save region
                # FIX: Read room type from canonical field chain (type > category > room_type).
                # Previously used only room.get("room_type") which never existed,
                # causing every extracted patch to be named "..._unknown.png".
                room_type = (
                    room.get("type")
                    or room.get("category")
                    or room.get("room_type")
                    or "unknown"
                )
                filename = f"{image_stem}_{i:03d}_{room_type}.png"
                if prefix:
                    filename = f"{prefix}_{filename}"

                output_path = output_dir / filename
                cv2.imwrite(str(output_path), region)

                extracted.append(str(output_path))
                logger.debug(f"Extracted room {i}: {output_path}")

            except Exception as e:
                logger.error(f"Error extracting room {i}: {e}")
                continue

        logger.info(f"Extracted {len(extracted)} room regions from {image_path}")
        return extracted

    def extract_batch(self,
                     image_dir: str,
                     annotation_dir: str,
                     output_dir: str) -> Dict[str, List[str]]:
        """
        Extract regions from multiple floor plans.

        Args:
            image_dir: Directory containing floor plan images
            annotation_dir: Directory containing annotation JSON files
            output_dir: Directory to save extracted regions

        Returns:
            Dictionary mapping image name to list of extracted regions
        """
        image_dir = Path(image_dir)
        annotation_dir = Path(annotation_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        import json

        # Process each image
        for image_path in sorted(image_dir.glob("*.png")):
            image_stem = image_path.stem
            ann_path = annotation_dir / f"{image_stem}.json"

            if not ann_path.exists():
                logger.warning(f"No annotation found for {image_path}")
                continue

            try:
                with open(ann_path) as f:
                    annotation = json.load(f)

                extracted = self.extract_regions(str(image_path), annotation, str(output_dir))
                results[image_stem] = extracted

            except Exception as e:
                logger.error(f"Error processing {image_stem}: {e}")
                results[image_stem] = []

        return results

    def extract_region_statistics(self,
                                  image_path: str,
                                  annotation: Dict) -> Dict:
        """
        Calculate statistics about extracted regions.

        Args:
            image_path: Path to floor plan image
            annotation: Annotation dictionary

        Returns:
            Dictionary with region statistics
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return {}

        img_height, img_width = image.shape[:2]
        stats = {
            "image_size": (img_width, img_height),
            "total_rooms": 0,
            "regions": []
        }

        rooms = annotation.get("rooms", [])
        for i, room in enumerate(rooms):
            bbox = room.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            try:
                x1, y1, x2, y2 = [float(v) for v in bbox]
            except (ValueError, TypeError):
                continue

            # Add padding
            width = x2 - x1
            height = y2 - y1
            pad_x = width * self.padding_pct
            pad_y = height * self.padding_pct

            x1_padded = max(0, x1 - pad_x)
            y1_padded = max(0, y1 - pad_y)
            x2_padded = min(img_width, x2 + pad_x)
            y2_padded = min(img_height, y2 + pad_y)

            region_width = x2_padded - x1_padded
            region_height = y2_padded - y1_padded
            region_area = region_width * region_height

            stats["regions"].append({
                "index": i,
                "original_bbox": [x1, y1, x2, y2],
                "padded_bbox": [x1_padded, y1_padded, x2_padded, y2_padded],
                "size": (int(region_width), int(region_height)),
                "area": int(region_area),
                "room_type": room.get("room_type", "unknown")
            })

            stats["total_rooms"] += 1

        return stats

    @staticmethod
    def get_region_info(region_path: str) -> Optional[Dict]:
        """
        Extract information from region filename.

        Expects format: {image_stem}_{room_id:03d}_{room_type}.png

        Args:
            region_path: Path to extracted region file

        Returns:
            Dictionary with extracted info, or None if format invalid
        """
        filename = Path(region_path).stem
        parts = filename.rsplit("_", 2)

        if len(parts) < 3:
            return None

        try:
            image_stem = "_".join(parts[:-2])
            room_id = int(parts[-2])
            room_type = parts[-1]

            return {
                "image_stem": image_stem,
                "room_id": room_id,
                "room_type": room_type,
                "path": str(region_path)
            }
        except (ValueError, IndexError):
            return None
