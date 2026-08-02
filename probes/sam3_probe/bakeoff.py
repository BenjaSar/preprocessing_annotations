"""
Application layer: the Phase 0 bake-off use case.

Depends only on the PromptableSegmenter port (not on sam3_adapter or the sam3
package) and on the project's existing bbox_metrics module. Swap the segmenter
argument to compare any promptable-segmentation backend under the same harness.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from preprocessing_annotations.bbox.bbox_metrics import evaluate_bboxes

from . import overlay
from .domain import BakeOffCase, Detection
from .ports import PromptableSegmenter

logger = logging.getLogger(__name__)


@dataclass
class CasePromptResult:
    case_name: str
    prompt: str
    detections: List[Detection]
    metrics: Optional[Dict[str, Any]] = None  # None when no ground truth exists for this prompt


@dataclass
class BakeOffReport:
    results: List[CasePromptResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            f"{r.case_name}::{r.prompt}": {
                "num_detections": len(r.detections),
                "metrics": r.metrics,
            }
            for r in self.results
        }


def run_case(
    segmenter: PromptableSegmenter,
    case: BakeOffCase,
    prompts: List[str],
    score_threshold: float,
    overlay_dir: Optional[Path] = None,
) -> List[CasePromptResult]:
    """Run every prompt against one case, scoring against ground truth when available."""
    image = np.array(Image.open(case.image_path).convert("RGB"))
    results = []

    for prompt in prompts:
        detections = segmenter.segment(image, prompt, score_threshold)
        logger.info("%s / %r: %d detection(s)", case.name, prompt, len(detections))

        matching_gt = [gt for gt in case.ground_truth if _category_matches(gt.category, prompt)]
        metrics = None
        if matching_gt:
            predictions = [d.to_bbox_metrics_dict() for d in detections]
            ground_truth = [gt.to_bbox_metrics_dict() for gt in matching_gt]
            metrics = evaluate_bboxes(
                predictions, ground_truth, case.image_width, case.image_height
            ).to_dict()

        if overlay_dir is not None:
            rendered = overlay.render(image, detections, matching_gt)
            out_path = Path(overlay_dir) / f"{case.name}__{_slug(prompt)}.png"
            rendered.save(out_path)

        results.append(
            CasePromptResult(case_name=case.name, prompt=prompt, detections=detections, metrics=metrics)
        )
    return results


def run_bakeoff(
    segmenter: PromptableSegmenter,
    cases: List[BakeOffCase],
    prompts: List[str],
    score_threshold: float = 0.3,
    overlay_dir: Optional[Path] = None,
) -> BakeOffReport:
    report = BakeOffReport()
    for case in cases:
        report.results.extend(run_case(segmenter, case, prompts, score_threshold, overlay_dir))
    return report


_GENERIC_ROOM_PROMPTS = {"room", "space", "rooms"}


def _category_matches(category: str, prompt: str) -> bool:
    """Loose match: GT category names are uppercase multi-word (e.g. 'PRIVATE OFFICE').

    A generic prompt ("room"/"space") is scored against every GT category —
    it's asking "find all rooms," not one specific type.
    """
    prompt = prompt.strip().lower()
    if prompt in _GENERIC_ROOM_PROMPTS:
        return True
    return prompt in category.strip().lower()


def _slug(text: str) -> str:
    return text.strip().lower().replace(" ", "_")
