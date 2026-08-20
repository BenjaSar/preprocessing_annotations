"""Application layer: SAM1 vs SAM2.1 flood bake-off.

Depends only on the PointPromptExpander port, never on a concrete adapter.
Seeds are real production OCR Pass-1 candidates (not invented) from the two
pages where SAM1 was observed to flood in production
(sprint1_verify39, 326 ROCKAWAY page002 / Bradley Fair page000).
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from preprocessing_annotations.config import OCRConfig, SAMConfig
from preprocessing_annotations.ingestion.ocr_extractor import MEPTextExtractor
from PIL import Image
from preprocessing_annotations.detection.sam_segmenter import is_collapse, is_label_scale_result

from .domain import Expansion, Seed
from .ports import PointPromptExpander

# Same priority order as sam_segmenter.py's own refine_annotations loop:
# over_segmentation -> flood_swallowed_seed -> collapse -> label_scale_noop
# -> applied. Mutually exclusive, exactly like production's sam_skip_reason
# (each result gets ONE bucket, not independent booleans) -- P-B (tech-eval
# plan) needs the full distribution, not just flood/collapse, to compare
# against the production rate this bake-off was extended to measure.
_MIN_EXPANSION_AREA_PX = SAMConfig().min_expansion_area_px


def _bucket(
    area_frac: float, max_expand_frac: float, swallowed: bool,
    sam_area: int, original_area: int,
) -> str:
    if area_frac > max_expand_frac:
        return "over_segmentation"
    if swallowed:
        return "flood_swallowed_seed"
    if is_collapse(sam_area, original_area):
        return "collapse"
    if is_label_scale_result(sam_area, _MIN_EXPANSION_AREA_PX):
        return "label_scale_noop"
    return "applied"


def load_real_seeds(image_path: str) -> List[Seed]:
    """Real OCR Pass-1 candidates for one image — the actual seeds production
    SAM would receive. Not synthetic/invented.
    """
    extractor = MEPTextExtractor(OCRConfig())
    candidates, _ = extractor.extract_and_find_rooms(image_path)
    seeds = []
    for c in candidates:
        x, y, w, h = c.bbox
        label = c.room_name or c.room_number or "?"
        seeds.append(Seed(label=label, center=(x + w // 2, y + h // 2), label_bbox=(x, y, w, h)))
    return seeds


@dataclass
class SeedResult:
    seed: Seed
    expansion: Expansion
    image_area_frac: float
    swallowed_other_seed: bool  # flood signal: expansion contains another seed's center
    collapsed: bool  # production's own collapse guardrail (sam_segmenter.is_collapse):
    # SAM shrank below a label-scale seed, i.e. failed to find the room at all
    bucket: str  # mutually-exclusive production classification, see _bucket()


@dataclass
class CaseReport:
    case_name: str
    results: List[SeedResult] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        n = len(self.results)
        if n == 0:
            return {"n_seeds": 0}
        flooded = sum(1 for r in self.results if r.swallowed_other_seed)
        collapsed = sum(1 for r in self.results if r.collapsed)
        bucket_counts = {
            b: sum(1 for r in self.results if r.bucket == b)
            for b in ("over_segmentation", "flood_swallowed_seed", "collapse", "label_scale_noop", "applied")
        }
        undershoot = bucket_counts["label_scale_noop"] + bucket_counts["collapse"]
        overshoot = bucket_counts["flood_swallowed_seed"] + bucket_counts["over_segmentation"]
        return {
            "n_seeds": n,
            "flooded_count": flooded,
            "flooded_frac": flooded / n,
            "collapsed_count": collapsed,
            "collapsed_frac": collapsed / n,
            "mean_area_frac": sum(r.image_area_frac for r in self.results) / n,
            "max_area_frac": max(r.image_area_frac for r in self.results),
            "buckets": bucket_counts,
            "applied_frac": bucket_counts["applied"] / n,
            "undershoot_frac": undershoot / n,
            "overshoot_frac": overshoot / n,
        }


def run_case(
    expander: PointPromptExpander, image_path: str, case_name: str, max_expand_frac: float
) -> CaseReport:
    seeds = load_real_seeds(image_path)
    with Image.open(image_path) as im:
        img_w, img_h = im.size
    img_area = img_w * img_h

    report = CaseReport(case_name=case_name)
    for seed in seeds:
        expansion = expander.expand(image_path, seed.center, seed.label_bbox)
        ex, ey, ew, eh = expansion.bbox_xywh
        sam_area = ew * eh
        area_frac = sam_area / img_area if img_area > 0 else 0.0
        swallowed = any(
            other is not seed and ex <= other.center[0] <= ex + ew and ey <= other.center[1] <= ey + eh
            for other in seeds
        )
        _, _, sw, sh = seed.label_bbox
        original_area = max(1, sw * sh)
        collapsed = is_collapse(sam_area, original_area)
        bucket = _bucket(area_frac, max_expand_frac, swallowed, sam_area, original_area)
        report.results.append(
            SeedResult(
                seed=seed,
                expansion=expansion,
                image_area_frac=area_frac,
                swallowed_other_seed=swallowed,
                collapsed=collapsed,
                bucket=bucket,
            )
        )
    return report
