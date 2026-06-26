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

            # Resolve checkpoint path: if relative, resolve relative to module directory
            checkpoint_path = Path(self.config.checkpoint)
            if not checkpoint_path.is_absolute():
                # Resolve relative to this module's directory
                module_dir = Path(__file__).parent
                checkpoint_path = module_dir / checkpoint_path
            
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
        label_bbox: Optional[List[int]] = None,
    ) -> SegmentationResult:
        """
        Segment a room from a label centroid, optionally constrained by a box prompt.

        Args:
            image_path: Path to the floor plan image.
            point: (x, y) point inside the room (typically label center).
            include_mask: Whether to include the full mask in result.
            label_bbox: Optional [x, y, w, h] label box. When provided, used as a
                SAM box prompt (F-B) to confine segmentation to the room around
                the label rather than the whole floor plate.

        Returns:
            SegmentationResult with refined bbox and optional mask.
        """
        self._set_image(image_path)

        input_point = np.array([[point[0], point[1]]])
        input_label = np.array([1])  # 1 = foreground

        # F-B: box prompt from label bbox (xywh -> xyxy) constrains SAM region
        box = None
        if label_bbox is not None and len(label_bbox) == 4:
            bx, by, bw, bh = label_bbox
            box = np.array([bx, by, bx + bw, by + bh])

        # Always request 3 candidates so mask selection can avoid the floor-plate.
        masks, scores, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=box,
            multimask_output=True,
        )

        # F-A: pick a room-scale mask, NOT argmax(score) (which favours the
        # largest floor-plate region on floor plans).
        mask, confidence = self._select_room_mask(masks, scores)

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

    def _select_room_mask(
        self, masks: np.ndarray, scores: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Select a room-scale mask from SAM multimask candidates (F-A).

        SAM scores favour the largest (whole-floor) mask on floor plans.
        Instead: among masks whose area is below max_expand_frac of the image,
        pick the highest-scoring. If none qualify (all are floor-plate-sized),
        fall back to the smallest mask available.

        Args:
            masks: (N, H, W) boolean masks from predictor.predict.
            scores: (N,) confidence scores.

        Returns:
            (selected_mask, confidence).
        """
        img_area = masks.shape[1] * masks.shape[2]
        max_area = img_area * self.config.max_expand_frac

        areas = [int(m.sum()) for m in masks]

        # Candidates below the over-segmentation ceiling
        valid = [i for i, a in enumerate(areas) if 0 < a <= max_area]
        if valid:
            best_idx = max(valid, key=lambda i: scores[i])
        else:
            # All masks too large -> take the smallest non-empty one
            non_empty = [i for i, a in enumerate(areas) if a > 0]
            if not non_empty:
                return masks[0], float(scores[0])
            best_idx = min(non_empty, key=lambda i: areas[i])

        return masks[best_idx], float(scores[best_idx])

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
        img_width: int = 0,
        img_height: int = 0,
        max_expand_frac: float = 0.25,
    ) -> List[dict]:
        """
        Expand label-sized bboxes to room-boundary bboxes using SAM.

        For each annotation, seeds SAM with the label centroid, segments the
        enclosing room, and replaces the label bbox with the room mask bbox.

        Guardrails (keep original on fail):
          - SAM bbox area > max_expand_frac * image_area → over-segmentation
            (SAM grabbed the full floor or a multi-room region)
          - SAM bbox area < original label area → collapse (SAM shrunk)

        Args:
            image_path:      Path to the floor plan image.
            annotations:     List of annotation dicts with 'bbox' [x,y,w,h].
            img_width:       Image width in px (auto-read from file if 0).
            img_height:      Image height in px (auto-read from file if 0).
            max_expand_frac: Max fraction of image area any single SAM box may occupy.

        Returns:
            List of annotations with expanded bboxes (original preserved as
            'original_bbox'; sam_expanded=True marks successful expansions).
        """
        # Resolve image dimensions for guardrail checks
        if img_width == 0 or img_height == 0:
            try:
                from PIL import Image as _PIL
                with _PIL.open(image_path) as _img:
                    img_width, img_height = _img.size
            except Exception:
                img_width, img_height = 4500, 3375  # safe fallback

        # Grayscale array for per-room content-density (FIX-1). Boxes over blank
        # space (hallucinations) have near-zero non-white density; real rooms
        # contain walls/fixtures and score higher.
        try:
            from PIL import Image as _PIL
            _gray = np.asarray(_PIL.open(image_path).convert("L"))
        except Exception:
            _gray = None

        img_area = img_width * img_height
        max_sam_area = img_area * max_expand_frac

        refined = []
        expanded = 0
        kept_original = 0

        for ann in annotations:
            bbox = ann.get("bbox", [0, 0, 100, 100])

            # Centroid of label box → SAM prompt point
            if len(bbox) == 4:
                bx, by, bw, bh = bbox
                center_x = bx + bw // 2
                center_y = by + bh // 2
                original_area = max(1, bw * bh)
            else:
                center_x, center_y, original_area = 50, 50, 1

            try:
                result = self.segment_from_point(
                    image_path, (center_x, center_y), include_mask=False,
                    label_bbox=bbox if len(bbox) == 4 else None,
                )

                rx, ry, rw, rh = result.bbox
                sam_area = rw * rh

                # Guardrail 1: over-segmentation
                if sam_area > max_sam_area:
                    logger.debug(
                        f"SAM over-seg guardrail: {sam_area:.0f} > {max_sam_area:.0f} "
                        f"({max_expand_frac:.0%} of image). Keeping original."
                    )
                    refined_ann = ann.copy()
                    refined_ann["sam_expanded"] = False
                    refined_ann["sam_skip_reason"] = "over_segmentation"
                    refined.append(refined_ann)
                    kept_original += 1
                    continue

                # Guardrail 2: collapse
                if sam_area < original_area:
                    logger.debug(
                        f"SAM collapse guardrail: sam_area={sam_area} < "
                        f"original={original_area}. Keeping original."
                    )
                    refined_ann = ann.copy()
                    refined_ann["sam_expanded"] = False
                    refined_ann["sam_skip_reason"] = "collapse"
                    refined.append(refined_ann)
                    kept_original += 1
                    continue

                refined_ann = ann.copy()
                refined_ann["bbox"] = result.bbox
                refined_ann["original_bbox"] = bbox
                refined_ann["sam_confidence"] = result.confidence
                refined_ann["sam_expanded"] = True
                refined.append(refined_ann)
                expanded += 1

            except Exception as e:
                logger.warning(f"SAM segment failed, keeping original: {e}")
                refined_ann = ann.copy()
                refined_ann["sam_expanded"] = False
                refined_ann["sam_skip_reason"] = "error"
                refined.append(refined_ann)
                kept_original += 1

        # FIX-1: attach content-density (non-white fraction) on each room's final
        # bbox. Consumed by the SFT room-scale gate to drop blank-space boxes.
        if _gray is not None:
            for ann in refined:
                b = ann.get("bbox", [])
                if len(b) == 4:
                    x, y, w, h = [int(v) for v in b]
                    x2, y2 = min(x + w, _gray.shape[1]), min(y + h, _gray.shape[0])
                    x, y = max(0, x), max(0, y)
                    if x2 > x and y2 > y:
                        sub = _gray[y:y2, x:x2]
                        ann["content_density"] = float((sub < 200).sum()) / sub.size
                    else:
                        ann["content_density"] = 0.0

        logger.info(
            f"SAM refine_annotations: {expanded} expanded, "
            f"{kept_original} kept original (of {len(annotations)} total)"
        )
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
