"""
Renders detections + ground truth on top of a source image for visual QA.

Pure rendering: takes domain objects, returns a PIL Image. No file I/O, no
metrics — callers decide where to save it.
"""

from typing import List

import numpy as np
from PIL import Image, ImageDraw

from .domain import Detection, GroundTruthRoom

_PRED_COLOR = (255, 0, 0)      # red
_GT_COLOR = (0, 170, 0)        # green


def render(
    image: np.ndarray,
    predictions: List[Detection],
    ground_truth: List[GroundTruthRoom],
) -> Image.Image:
    canvas = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)

    for gt in ground_truth:
        x, y, w, h = gt.bbox_xywh
        draw.rectangle([x, y, x + w, y + h], outline=_GT_COLOR, width=3)

    for det in predictions:
        x1, y1, x2, y2 = det.bbox_xyxy
        draw.rectangle([x1, y1, x2, y2], outline=_PRED_COLOR, width=3)
        draw.text((x1, max(0, y1 - 12)), f"{det.label} {det.score:.2f}", fill=_PRED_COLOR)

    return canvas
