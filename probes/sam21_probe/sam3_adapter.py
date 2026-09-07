"""Adapter: interactive SAM3 (ultralytics) as a PointPromptExpander.

Reuses the SAME mask-selection algorithm as production SAM1
(sam_segmenter.select_room_mask / mask_to_bbox), same P-2B box-vs-point
parity (should_use_box_prompt) -- mirrors sam2_adapter.py exactly, only
the underlying model differs, for a fair comparison.

CONFOUND, stated up front per the tech-eval plan's P-B design (do not
discover this after the fact): this uses ultralytics' SAM3Predictor
(interactive box/point->mask), NOT the SAM3SemanticPredictor used
elsewhere in this project for door/window concept segmentation -- that
is a different capability (exemplar matching, not single-instance
point-prompt). Two measured platform differences from SAM1/SAM2.1:

1. Resolution: SAM3Predictor OOMs at imgsz=1008 on this project's T4
   (14.6GB) in isolation; imgsz=644 is the largest that fits. SAM1
   (vit_h) and SAM2.1 (hiera_large) both run at their native ~1024.
   Any SAM3 result here could reflect this gap, not the model itself.
2. Candidate count: ultralytics' predictor returns exactly ONE mask per
   prompt call (no multimask_output toggle). SAM1/SAM2.1 return 3
   candidates and select_room_mask picks among them specifically to
   avoid the floor-plate-sized mask SAM's own top score tends to favor.
   SAM3 gets no such selection -- select_room_mask still runs (N=1 is
   a safe degenerate case, no crash), but it cannot correct a bad top
   choice the way it can for the other two arms.
3. Confidence floor: usable only at conf ~0.01 (measured: 0 masks
   returned at the ultralytics default 0.25 on a real room box).
"""

import numpy as np
from PIL import Image
from preprocessing_annotations.detection.sam_segmenter import mask_to_bbox, select_room_mask, should_use_box_prompt

from .domain import Expansion

_SAM3_IMGSZ = 644  # measured ceiling on this project's GPU, see module docstring
_SAM3_CONF = 0.01  # measured: 0.25 (ultralytics default) returns 0 masks


class Sam3Adapter:
    def __init__(self, checkpoint_path: str, max_expand_frac: float, device: str = "cuda"):
        from ultralytics.models.sam import SAM3Predictor

        self._predictor = SAM3Predictor(
            overrides={"conf": _SAM3_CONF, "model": checkpoint_path, "imgsz": _SAM3_IMGSZ, "save": False}
        )
        self._max_expand_frac = max_expand_frac
        self._current_image_path: str | None = None

    def _set_image(self, image_path: str) -> None:
        if image_path == self._current_image_path:
            return
        self._predictor.set_image(image_path)
        self._current_image_path = image_path

    def expand(self, image_path: str, point: tuple[int, int], label_bbox: tuple[int, int, int, int]) -> Expansion:
        self._set_image(image_path)

        # P-2B parity: same box-vs-point decision as production SAM1.
        bx, by, bw, bh = label_bbox
        original_area = max(1, bw * bh)
        use_box = should_use_box_prompt(original_area)

        call_kwargs = {"points": [[point[0], point[1]]], "labels": [1]}
        if use_box:
            call_kwargs["bboxes"] = [[bx, by, bx + bw, by + bh]]

        result = self._predictor(**call_kwargs)[0]
        if result.masks is None or len(result.masks.data) == 0:
            # Measured failure mode: below-threshold confidence returns no
            # mask at all. Treat as a zero-area expansion -- production's
            # is_collapse will correctly classify this as "collapse"
            # (sam_area=0 < original_area), the honest outcome, not a crash.
            return Expansion(bbox_xywh=(bx, by, 0, 0), confidence=0.0, mask=None)

        masks = result.masks.data.cpu().numpy().astype(bool)
        # ultralytics gives exactly one mask (see module docstring, point 2) --
        # select_room_mask degrades safely to a no-op selection over N=1.
        conf = float(result.boxes.conf[0]) if result.boxes is not None and len(result.boxes.conf) else 1.0
        scores = np.array([conf] * masks.shape[0])
        mask, confidence = select_room_mask(masks, scores, self._max_expand_frac)
        bbox, _ = mask_to_bbox(mask)
        return Expansion(bbox_xywh=tuple(bbox), confidence=confidence, mask=mask)
