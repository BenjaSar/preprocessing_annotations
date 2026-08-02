"""
Ports (interfaces) the application layer depends on.

Dependency-inversion boundary: bakeoff.py imports only this Protocol, never
sam3_adapter.py or the sam3 package directly. Swapping in a different
promptable segmenter (or a fake for testing) means writing a new adapter,
zero changes to the application layer.
"""

from typing import List, Protocol, Tuple

import numpy as np

from .domain import Detection


class PromptableSegmenter(Protocol):
    """A model that segments/detects objects from text or exemplar prompts."""

    def segment(
        self, image: np.ndarray, prompt: str, score_threshold: float
    ) -> List[Detection]:
        """Return detections matching `prompt` in `image` (RGB uint8, HxWx3)."""
        ...

    def segment_by_exemplars(
        self,
        image: np.ndarray,
        exemplar_boxes_cxcywh: List[Tuple[float, float, float, float]],
        score_threshold: float,
    ) -> List[Detection]:
        """Return all instances matching the given positive box exemplars.

        Args:
            image: RGB uint8, HxWx3.
            exemplar_boxes_cxcywh: Positive exemplar boxes as
                [center_x, center_y, width, height], normalized to [0, 1].
            score_threshold: Minimum detection confidence.

        Returns:
            Detections in xyxy pixel coordinates.
        """
        ...
