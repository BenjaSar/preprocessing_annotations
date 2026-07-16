"""Adapter: production SAM1 (RoomSegmenter) as a PointPromptExpander.

Thin wrapper — delegates to the existing production sam_segmenter.py, so this
spike measures the actual model already in the pipeline. No duplicated logic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SAMConfig  # noqa: E402
from sam_segmenter import RoomSegmenter, should_use_box_prompt  # noqa: E402

from .domain import Expansion


class Sam1Adapter:
    def __init__(self, config: SAMConfig | None = None):
        self._segmenter = RoomSegmenter(config or SAMConfig())

    def expand(self, image_path: str, point: tuple[int, int], label_bbox: tuple[int, int, int, int]) -> Expansion:
        # P-2B parity: same box-vs-point decision production's refine_annotations
        # makes (see sam_segmenter.should_use_box_prompt), so this adapter
        # exercises the identical prompt behavior as the pipeline.
        original_area = max(1, label_bbox[2] * label_bbox[3])
        box = list(label_bbox) if should_use_box_prompt(original_area) else None
        result = self._segmenter.segment_from_point(
            image_path, point, include_mask=True, label_bbox=box
        )
        return Expansion(bbox_xywh=tuple(result.bbox), confidence=result.confidence, mask=result.mask)
