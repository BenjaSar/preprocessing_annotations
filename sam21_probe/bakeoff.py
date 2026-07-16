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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import OCRConfig  # noqa: E402
from ocr_extractor import MEPTextExtractor  # noqa: E402
from PIL import Image  # noqa: E402
from sam_segmenter import is_collapse  # noqa: E402

from .domain import Expansion, Seed
from .ports import PointPromptExpander


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
        return {
            "n_seeds": n,
            "flooded_count": flooded,
            "flooded_frac": flooded / n,
            "collapsed_count": collapsed,
            "collapsed_frac": collapsed / n,
            "mean_area_frac": sum(r.image_area_frac for r in self.results) / n,
            "max_area_frac": max(r.image_area_frac for r in self.results),
        }


def run_case(expander: PointPromptExpander, image_path: str, case_name: str) -> CaseReport:
    seeds = load_real_seeds(image_path)
    with Image.open(image_path) as im:
        img_w, img_h = im.size
    img_area = img_w * img_h

    report = CaseReport(case_name=case_name)
    for seed in seeds:
        expansion = expander.expand(image_path, seed.center, seed.label_bbox)
        ex, ey, ew, eh = expansion.bbox_xywh
        area_frac = (ew * eh) / img_area if img_area > 0 else 0.0
        swallowed = any(
            other is not seed and ex <= other.center[0] <= ex + ew and ey <= other.center[1] <= ey + eh
            for other in seeds
        )
        _, _, sw, sh = seed.label_bbox
        original_area = max(1, sw * sh)
        collapsed = is_collapse(ew * eh, original_area)
        report.results.append(
            SeedResult(
                seed=seed,
                expansion=expansion,
                image_area_frac=area_frac,
                swallowed_other_seed=swallowed,
                collapsed=collapsed,
            )
        )
    return report
