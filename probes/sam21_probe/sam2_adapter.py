"""Adapter: SAM2.1 as a PointPromptExpander.

Reuses the SAME mask-selection algorithm as production SAM1
(sam_segmenter.select_room_mask / mask_to_bbox) so only the underlying model
differs between the two bake-off arms — a fair comparison.
"""

import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from preprocessing_annotations.detection.sam_segmenter import mask_to_bbox, select_room_mask, should_use_box_prompt

from .domain import Expansion


class Sam2Adapter:
    def __init__(self, config_file: str, checkpoint_path: str, device: str, max_expand_frac: float):
        model = build_sam2(config_file, checkpoint_path, device=device)
        self._predictor = SAM2ImagePredictor(model)
        self._max_expand_frac = max_expand_frac
        self._current_image_path: str | None = None

    def _set_image(self, image_path: str) -> None:
        if image_path == self._current_image_path:
            return
        image_rgb = np.array(Image.open(image_path).convert("RGB"))
        self._predictor.set_image(image_rgb)
        self._current_image_path = image_path

    def expand(self, image_path: str, point: tuple[int, int], label_bbox: tuple[int, int, int, int]) -> Expansion:
        self._set_image(image_path)

        # P-2B parity: same box-vs-point decision as production SAM1 (see
        # sam_segmenter.should_use_box_prompt) — only the model differs.
        bx, by, bw, bh = label_bbox
        original_area = max(1, bw * bh)
        box = np.array([bx, by, bx + bw, by + bh]) if should_use_box_prompt(original_area) else None

        masks, scores, _ = self._predictor.predict(
            point_coords=np.array([[point[0], point[1]]]),
            point_labels=np.array([1]),
            box=box,
            multimask_output=True,
        )
        mask, confidence = select_room_mask(masks.astype(bool), scores, self._max_expand_frac)
        bbox, _ = mask_to_bbox(mask)
        return Expansion(bbox_xywh=tuple(bbox), confidence=confidence, mask=mask)
