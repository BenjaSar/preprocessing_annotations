"""
Template-based symbol detection for MEP floor plans.

This module detects electrical symbols (receptacles, switches, fixtures)
using multi-scale, rotation-invariant template matching.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

try:
    from .config import TemplateConfig
except ImportError:
    from config import TemplateConfig

logger = logging.getLogger(__name__)


class SymbolDetectionError(Exception):
    """Raised when symbol detection fails."""

    pass


@dataclass
class SymbolDetection:
    """
    Detected symbol instance.

    Attributes:
        category: Symbol category (e.g., "receptacle", "switch").
        bbox: Bounding box as (x, y, width, height).
        confidence: Match confidence score.
        rotation: Detected rotation angle in degrees.
        scale: Detected scale factor.
    """

    category: str
    bbox: tuple[int, int, int, int]
    confidence: float
    rotation: int = 0
    scale: float = 1.0


class SymbolTemplateDetector:
    """
    Detects electrical symbols using template matching.

    Supports multi-scale and rotation-invariant matching for
    robust detection of MEP symbols at various orientations.

    Template Directory Structure:
        template_dir/
            receptacle/
                template1.png
                template2.png
            switch/
                template1.png
            fixture/
                ...

    Attributes:
        config: TemplateConfig with detection parameters.
        templates: Dictionary mapping categories to template variants.

    Example:
        detector = SymbolTemplateDetector("./templates", config)
        detections = detector.detect_symbols("floorplan.png")
    """

    def __init__(
        self, template_dir: str | Path, config: Optional[TemplateConfig] = None
    ):
        self.config = config or TemplateConfig()
        self.templates: Dict[str, List[dict]] = {}
        self._load_templates(template_dir)

    def _load_templates(self, template_dir: str | Path) -> None:
        """
        Load and preprocess all templates from directory.

        Creates scaled and rotated variants for each template.
        """
        template_path = Path(template_dir)

        if not template_path.exists():
            raise SymbolDetectionError(f"Template directory not found: {template_path}")

        for category_dir in template_path.iterdir():
            if not category_dir.is_dir():
                continue

            category = category_dir.name
            self.templates[category] = []

            for template_file in category_dir.glob("*.png"):
                template = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)

                if template is None:
                    logger.warning(f"Failed to load template: {template_file}")
                    continue

                # Generate variants for each scale and rotation
                for scale in self.config.scales:
                    for rotation in self.config.rotations:
                        variant = self._create_variant(template, scale, rotation)
                        self.templates[category].append(
                            {
                                "image": variant,
                                "scale": scale,
                                "rotation": rotation,
                                "source": template_file.name,
                            }
                        )

            logger.info(
                f"Loaded {len(self.templates[category])} template variants "
                f"for category '{category}'"
            )

    def _create_variant(
        self, template: np.ndarray, scale: float, rotation: int
    ) -> np.ndarray:
        """
        Create a scaled and rotated variant of a template.

        Args:
            template: Original template image.
            scale: Scale factor.
            rotation: Rotation angle in degrees.

        Returns:
            Transformed template image.
        """
        # Scale
        if scale != 1.0:
            new_size = (
                int(template.shape[1] * scale),
                int(template.shape[0] * scale),
            )
            template = cv2.resize(template, new_size, interpolation=cv2.INTER_LINEAR)

        # Rotate
        if rotation != 0:
            center = (template.shape[1] // 2, template.shape[0] // 2)
            matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)

            # Calculate new bounding box size
            cos = abs(matrix[0, 0])
            sin = abs(matrix[0, 1])
            new_w = int(template.shape[0] * sin + template.shape[1] * cos)
            new_h = int(template.shape[0] * cos + template.shape[1] * sin)

            # Adjust transformation matrix
            matrix[0, 2] += (new_w - template.shape[1]) / 2
            matrix[1, 2] += (new_h - template.shape[0]) / 2

            template = cv2.warpAffine(
                template,
                matrix,
                (new_w, new_h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )

        return template

    def detect_symbols(
        self, image_path: str | Path, threshold: Optional[float] = None
    ) -> List[SymbolDetection]:
        """
        Detect symbols in an image using template matching.

        Args:
            image_path: Path to the image file.
            threshold: Detection threshold (overrides config if provided).

        Returns:
            List of SymbolDetection objects after NMS.

        Raises:
            SymbolDetectionError: If image cannot be loaded.
        """
        image_path = Path(image_path)
        threshold = threshold or self.config.threshold

        # Load image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise SymbolDetectionError(f"Failed to load image: {image_path}")

        detections = []

        for category, variants in self.templates.items():
            for variant in variants:
                template = variant["image"]
                h, w = template.shape

                # Skip if template is larger than image
                if h > image.shape[0] or w > image.shape[1]:
                    continue

                # Template matching
                result = cv2.matchTemplate(
                    image, template, self.config.match_method
                )

                # Find locations above threshold
                locations = np.where(result >= threshold)

                for pt in zip(*locations[::-1]):
                    confidence = float(result[pt[1], pt[0]])
                    detections.append(
                        SymbolDetection(
                            category=category,
                            bbox=(pt[0], pt[1], w, h),
                            confidence=confidence,
                            rotation=variant["rotation"],
                            scale=variant["scale"],
                        )
                    )

        # Apply Non-Maximum Suppression
        filtered = self._apply_nms(detections)

        logger.debug(
            f"Detected {len(filtered)} symbols in {image_path.name} "
            f"(before NMS: {len(detections)})"
        )

        return filtered

    def _apply_nms(self, detections: List[SymbolDetection]) -> List[SymbolDetection]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections.

        Args:
            detections: List of raw detections.

        Returns:
            Filtered list of detections.
        """
        if not detections:
            return []

        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)

        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)

            # Remove detections with high IoU
            detections = [
                d
                for d in detections
                if self._compute_iou(best.bbox, d.bbox) < self.config.nms_iou_threshold
            ]

        return keep

    def _compute_iou(
        self, box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]
    ) -> float:
        """
        Compute Intersection over Union between two bounding boxes.

        Args:
            box1: First box as (x, y, width, height).
            box2: Second box as (x, y, width, height).

        Returns:
            IoU value between 0.0 and 1.0.
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Compute intersection
        xi = max(x1, x2)
        yi = max(y1, y2)
        wi = min(x1 + w1, x2 + w2) - xi
        hi = min(y1 + h1, y2 + h2) - yi

        if wi <= 0 or hi <= 0:
            return 0.0

        intersection = wi * hi
        union = w1 * h1 + w2 * h2 - intersection

        return intersection / union if union > 0 else 0.0

    def get_detection_summary(
        self, detections: List[SymbolDetection]
    ) -> Dict[str, int]:
        """
        Get count of detections by category.

        Args:
            detections: List of SymbolDetection objects.

        Returns:
            Dictionary mapping category names to counts.
        """
        summary = {}
        for detection in detections:
            summary[detection.category] = summary.get(detection.category, 0) + 1
        return summary
