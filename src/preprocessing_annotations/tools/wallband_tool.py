"""AV-5 (2026-09-23): wall-band query + debug-artifact persistence.

Confirmed gap: walls are in this project's stated detection objective
(doors, walls, windows) but have no detector, no output field, and no
artifact of any kind. find_wall_bands (window_strip_detector.py) already
computes wall-band geometry -- Stage 1 of a two-stage window detector --
but its only caller was find_window_openings; the wall geometry itself
was never exposed on its own or persisted anywhere observable.

Scope, deliberately narrow: this tool exposes find_wall_bands' output
and can draw it to a debug PNG. It does NOT derive room extent from
wall bands -- that specific technique was tried and its validation
retracted (plan/technology-evaluation-2026-09-10.md Item 8, audited
2026-09-11: multi-room merges + non-floorplan-page contamination in the
target population). Nothing here reopens that.

Status of the underlying geometry: PROTOTYPE
(window_strip_detector.py's own module docstring) -- Stage-1 recall/
precision not scored at page level. This tool changes nothing about
that measurement status; it only makes the already-computed geometry
observable, which is the precondition the technique-evaluation plan's
Item 8 audit says any wall claim needs before being made at all.

Not wired into AnnotationPipeline.run() -- adding wall-band morphology
to every production page's hot path has a real runtime cost with no
validated consumer yet (YAGNI). Used via this tool directly, or via
scripts/persist_wall_bands.py for spot-checking a sample of pages.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from ..detection.window_strip_detector import (
    WindowStripDetectorConfig,
    find_wall_bands,
)

logger = logging.getLogger(__name__)

# Wall-band overlay color -- distinct from OBJECT_COLOR_MAP's existing
# door/window/etc palette (bbox_visualizer.py) so a combined overlay
# never collides with an object-detection box's color.
_WALL_BAND_COLOR = (46, 204, 113)  # Emerald green


def _to_grayscale(image_array: np.ndarray) -> np.ndarray:
    """find_wall_bands expects a single-channel grayscale array. Accepts
    an RGB array (as _load_image_for_detection loads it) and converts;
    passes an already-grayscale array through unchanged."""
    if image_array.ndim == 2:
        return image_array
    # Standard luminance weights -- matches cv2.cvtColor(..., COLOR_RGB2GRAY)
    # without adding an OpenCV call here for a 3-line conversion.
    return np.dot(image_array[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


class WallBandTool:
    """Query and persist wall-band geometry for one image."""

    def __init__(self, config: Optional[WindowStripDetectorConfig] = None):
        self._config = config or WindowStripDetectorConfig()

    def bands(self, image_array: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Returns (x, y, w, h) wall-band boxes in image pixel space."""
        gray = _to_grayscale(image_array)
        return find_wall_bands(gray, self._config)

    def persist(
        self,
        image_array: np.ndarray,
        output_path: Path,
        bands: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> Path:
        """Draws wall bands over image_array and saves a debug PNG.

        Args:
            image_array: Source image (RGB or grayscale).
            output_path: Where to save the overlay PNG.
            bands: Precomputed bands to draw. Computed via self.bands()
                if not given -- an explicit list lets a caller persist
                the same geometry it already queried without recomputing.

        Returns:
            output_path, for chaining.
        """
        if bands is None:
            bands = self.bands(image_array)

        base = image_array if image_array.ndim == 3 else np.stack([image_array] * 3, axis=-1)
        img = Image.fromarray(base.astype(np.uint8)).convert("RGB")
        draw = ImageDraw.Draw(img)
        for x, y, w, h in bands:
            draw.rectangle([x, y, x + w, y + h], outline=_WALL_BAND_COLOR, width=3)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        logger.info(f"WallBandTool: persisted {len(bands)} band(s) -> {output_path}")
        return output_path
