"""CLI entrypoint for the SAM1-vs-SAM2.1 flood bake-off.

Usage (from preprocessing_annotations/probes/):
    python -m sam21_probe.cli
"""

import json
from pathlib import Path

from preprocessing_annotations.config import SAMConfig

from . import bakeoff
from .sam1_adapter import Sam1Adapter
from .sam2_adapter import Sam2Adapter

CASES = [
    ("326_ROCKAWAY_page002", "sprint1_verify39/images/326 ROCKAWAY - AVI-ON LAYOUT_page002.png"),
    ("Bradley_page000", "sprint1_verify39/images/Bradley Fair_Avi-on Submittal_page000.png"),
]

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT = str(Path(__file__).resolve().parents[3] / "sam2_checkpoints" / "sam2.1_hiera_large.pt")


def main() -> None:
    sam_config = SAMConfig()
    adapters = {
        "sam1": Sam1Adapter(sam_config),
        "sam2.1": Sam2Adapter(
            SAM2_CONFIG, SAM2_CHECKPOINT, device="cuda", max_expand_frac=sam_config.max_expand_frac
        ),
    }

    report: dict = {}
    for case_name, image_path in CASES:
        report[case_name] = {}
        for model_name, expander in adapters.items():
            case_report = bakeoff.run_case(expander, image_path, f"{case_name}::{model_name}")
            report[case_name][model_name] = case_report.summary()
            print(f"{case_name} / {model_name}: {case_report.summary()}", flush=True)

    out_path = Path(__file__).resolve().parents[3] / "sam21_probe_out" / "report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"written: {out_path}", flush=True)


if __name__ == "__main__":
    main()
