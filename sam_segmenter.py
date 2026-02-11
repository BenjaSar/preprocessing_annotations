"""
SAM (Segment Anything Model) segmentation module for room boundary refinement.

This module uses SAM to refine room boundaries based on room label
locations detected by OCR or VLM annotation.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from .config import SAMConfig
except ImportError:
    from config import SAMConfig

logger = logging.getLogger(__name__)


class SAMSegmentationError(Exception):
    """Raised when SAM segmentation fails."""

    pass


@dataclass
class SegmentationResult:
    """
    Result of SAM segmentation for a single room.

    Attributes:
        prompt_point: The (x, y) point used to prompt SAM.
        bbox: Refined bounding box as [x, y, width, height].
        confidence: SAM confidence score.
        mask: Binary segmentation mask (numpy array).
        contour: Contour points of the segmented region.
    """

    prompt_point: Tuple[int, int]
    bbox: List[int]
    confidence: float
    mask: Optional[np.ndarray] = None
    contour: Optional[np.ndarray] = None


class RoomSegmenter:
    """
    Segments room boundaries using SAM with point prompts.

    Uses room label locations (from OCR or VLM) as prompts to
    SAM for generating precise room boundary masks.

    Attributes:
        config: SAMConfig with model parameters.

    Example:
        segmenter = RoomSegmenter(config)
        results = segmenter.segment_from_labels("floorplan.png", [(100, 200), (300, 400)])
    """

    def __init__(self, config: Optional[SAMConfig] = None):
        self.config = config or SAMConfig()
        self._predictor = None
        self._current_image_path = None

    @property
    def predictor(self):
        """Lazy initialization of SAM predictor."""
        if self._predictor is None:
            try:
                from segment_anything import SamPredictor, sam_model_registry
            except ImportError:
                raise SAMSegmentationError(
                    "segment_anything package not installed. "
                    "Install with: pip install segment-anything"
                )

            checkpoint_path = Path(self.config.checkpoint)
            if not checkpoint_path.exists():
                raise SAMSegmentationError(
                    f"SAM checkpoint not found: {checkpoint_path}. "
                    f"Download from https://github.com/facebookresearch/segment-anything"
                )

            try:
                sam = sam_model_registry[self.config.model_type](
                    checkpoint=str(checkpoint_path)
                )

                # Move to appropriate device
                if self.config.device == "cuda":
                    import torch

                    if torch.cuda.is_available():
                        sam.to("cuda")
                        logger.info("SAM loaded on CUDA")
                    else:
                        logger.warning("CUDA requested but not available, using CPU")
                elif self.config.device == "mps":
                    import torch

                    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        sam.to("mps")
                        logger.info("SAM loaded on MPS")
                    else:
                        logger.warning("MPS requested but not available, using CPU")
                else:
                    logger.info("SAM loaded on CPU")

                self._predictor = SamPredictor(sam)

            except Exception as e:
                raise SAMSegmentationError(f"Failed to load SAM model: {e}") from e

        return self._predictor

    def _set_image(self, image_path: str | Path) -> np.ndarray:
        """
        Load and set image for SAM predictor.

        Caches the current image to avoid reloading for multiple prompts.
        """
        image_path = Path(image_path)

        if str(image_path) == self._current_image_path:
            return None  # Image already set

        image = cv2.imread(str(image_path))
        if image is None:
            raise SAMSegmentationError(f"Failed to load image: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)
        self._current_image_path = str(image_path)

        logger.debug(f"Set image for SAM: {image_path.name}")
        return image

    def segment_from_point(
        self,
        image_path: str | Path,
        point: Tuple[int, int],
        include_mask: bool = True,
    ) -> SegmentationResult:
        """
        Segment a room from a single point prompt.

        Args:
            image_path: Path to the floor plan image.
            point: (x, y) point inside the room (typically label center).
            include_mask: Whether to include the full mask in result.

        Returns:
            SegmentationResult with refined bbox and optional mask.
        """
        self._set_image(image_path)

        input_point = np.array([[point[0], point[1]]])
        input_label = np.array([1])  # 1 = foreground

        masks, scores, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=self.config.multimask_output,
        )

        # Select best mask
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        confidence = float(scores[best_idx])

        # Extract bounding box and contour from mask
        bbox, contour = self._mask_to_bbox(mask)

        result = SegmentationResult(
            prompt_point=point,
            bbox=bbox,
            confidence=confidence,
            mask=mask if include_mask else None,
            contour=contour if include_mask else None,
        )

        return result

    def segment_from_labels(
        self,
        image_path: str | Path,
        label_points: List[Tuple[int, int]],
        include_masks: bool = False,
    ) -> List[SegmentationResult]:
        """
        Segment multiple rooms using label locations as prompts.

        Args:
            image_path: Path to the floor plan image.
            label_points: List of (x, y) center points of room labels.
            include_masks: Whether to include masks in results.

        Returns:
            List of SegmentationResult objects.
        """
        if not label_points:
            logger.warning("No label points provided for segmentation")
            return []

        results = []
        for point in label_points:
            try:
                result = self.segment_from_point(
                    image_path, point, include_mask=include_masks
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Segmentation failed for point {point}: {e}")
                continue

        logger.info(
            f"Segmented {len(results)}/{len(label_points)} rooms from {Path(image_path).name}"
        )

        return results

    def _mask_to_bbox(
        self, mask: np.ndarray
    ) -> Tuple[List[int], Optional[np.ndarray]]:
        """
        Convert binary mask to bounding box and contour.

        Args:
            mask: Binary segmentation mask.

        Returns:
            Tuple of (bbox as [x, y, w, h], largest contour).
        """
        mask_uint8 = mask.astype(np.uint8)

        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return [0, 0, 0, 0], None

        # Get largest contour
        largest = max(contours, key=cv2.contourArea)

        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest)

        return [x, y, w, h], largest

    def refine_annotations(
        self,
        image_path: str | Path,
        annotations: List[dict],
    ) -> List[dict]:
        """
        Refine VLM annotations with SAM segmentation.

        Takes VLM-generated annotations and replaces bboxes with
        SAM-refined boundaries.

        Args:
            image_path: Path to the floor plan image.
            annotations: List of annotation dicts with 'bbox' keys.

        Returns:
            List of annotations with refined bboxes.
        """
        refined = []

        for ann in annotations:
            bbox = ann.get("bbox", [0, 0, 100, 100])

            # Use bbox center as prompt point
            center_x = bbox[0] + bbox[2] // 2
            center_y = bbox[1] + bbox[3] // 2

            try:
                result = self.segment_from_point(
                    image_path, (center_x, center_y), include_mask=False
                )

                # Update annotation with refined bbox
                refined_ann = ann.copy()
                refined_ann["bbox"] = result.bbox
                refined_ann["sam_confidence"] = result.confidence
                refined_ann["original_bbox"] = bbox
                refined.append(refined_ann)

            except Exception as e:
                logger.warning(
                    f"SAM refinement failed for annotation, keeping original: {e}"
                )
                refined.append(ann)

        return refined

    def visualize_segmentation(
        self,
        image_path: str | Path,
        results: List[SegmentationResult],
        output_path: Optional[str | Path] = None,
    ) -> np.ndarray:
        """
        Visualize segmentation results on the image.

        Args:
            image_path: Path to the original image.
            results: List of SegmentationResult objects.
            output_path: Optional path to save visualization.

        Returns:
            Visualization image as numpy array.
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise SAMSegmentationError(f"Failed to load image: {image_path}")

        overlay = image.copy()

        # Draw each segmentation
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]

        for i, result in enumerate(results):
            color = colors[i % len(colors)]

            # Draw bounding box
            x, y, w, h = result.bbox
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

            # Draw prompt point
            cv2.circle(overlay, result.prompt_point, 5, color, -1)

            # Draw mask if available
            if result.mask is not None:
                mask_color = np.zeros_like(image)
                mask_color[result.mask] = color
                overlay = cv2.addWeighted(overlay, 0.7, mask_color, 0.3, 0)

        if output_path:
            cv2.imwrite(str(output_path), overlay)
            logger.info(f"Saved visualization to {output_path}")

        return overlay
