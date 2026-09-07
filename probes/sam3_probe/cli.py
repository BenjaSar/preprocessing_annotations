#!/usr/bin/env python3
"""
Phase 0 gate check: SAM3 vs. the project's current baselines.

Composition root — the only place that decides which concrete
PromptableSegmenter implementation backs the bake-off. Everything else
(bakeoff.py, overlay.py, domain.py) is wired against the PromptableSegmenter
port and doesn't know SAM3 exists.

Usage (run with the isolated sam3_venv interpreter — see sam3_venv/bin/python):
    python3 -m sam3_probe.cli --output-dir ./_sam3_probe_out
"""

import argparse
import json
import logging

from .bakeoff import run_bakeoff
from .domain import BakeOffCase
from .coco_ground_truth import load_rooms_for_image
from .sam3_adapter import Sam3Adapter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("sam3_probe.cli")

# Two fixed cases from the earlier CubiCasa5K smoke test, so results are
# directly comparable to that baseline (see conversation history /
# _cubicasa_smoke_overlay.png).
ROCKAWAY_IMAGE = (
    "/home/ubuntu/floorplan_classifier/VLM/marginfix_004634/images/"
    "326 ROCKAWAY - AVI-ON LAYOUT_page002.png"
)
ROCKAWAY_GT_COCO = "/home/ubuntu/floorplan_classifier/VLM/marginfix_004634/coco/train.json"
ROCKAWAY_GT_FILENAME = "326 ROCKAWAY - AVI-ON LAYOUT_page002.png"

MADISON_IMAGE = (
    "/home/ubuntu/floorplan_classifier/VLM/preprocessing_annotations/_layer_probe_out/"
    "540 Madison Avenue Divcowest Suite 29A - AVI-ON LAYOUT_page000_layers.png"
)

ROOM_PROMPTS = ["room", "office", "corridor"]
OPENING_PROMPTS = ["window", "door"]


def build_cases() -> list[BakeOffCase]:
    rockaway_gt = load_rooms_for_image(ROCKAWAY_GT_COCO, ROCKAWAY_GT_FILENAME)
    logger.info("ROCKAWAY page002: %d ground-truth rooms loaded", len(rockaway_gt))

    return [
        BakeOffCase(
            name="rockaway_p002",
            image_path=ROCKAWAY_IMAGE,
            image_width=4500,
            image_height=3375,
            ground_truth=rockaway_gt,
        ),
        BakeOffCase(
            name="madison_p000",
            image_path=MADISON_IMAGE,
            image_width=4500,
            image_height=3000,
            ground_truth=[],  # no room/opening ground truth exists for this plan
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="./_sam3_probe_out")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segmenter = Sam3Adapter(device=args.device)
    cases = build_cases()

    report = run_bakeoff(
        segmenter=segmenter,
        cases=cases,
        prompts=ROOM_PROMPTS + OPENING_PROMPTS,
        score_threshold=args.score_threshold,
        overlay_dir=output_dir,
    )

    report_path = output_dir / "bakeoff_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2))
    logger.info("Report written: %s", report_path)

    for result in report.results:
        if result.metrics is not None:
            logger.info(
                "%s / %r -> mAP@0.5=%.3f mean_IoU=%.3f (%d det, %d gt-matched)",
                result.case_name,
                result.prompt,
                result.metrics["mAP@0.5"],
                result.metrics["mean_IoU"],
                len(result.detections),
                sum(1 for _ in result.detections),
            )
        else:
            logger.info(
                "%s / %r -> %d detections (no ground truth to score against)",
                result.case_name,
                result.prompt,
                len(result.detections),
            )


if __name__ == "__main__":
    main()
