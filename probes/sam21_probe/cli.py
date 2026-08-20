"""CLI entrypoint for the SAM1-vs-SAM2.1(-vs-SAM3) flood bake-off.

Usage (from the VLM/ project root -- CASES paths are relative to it):
    python -m sam21_probe.cli

P-B (tech-eval plan): added the sam3 arm on top of the existing sam1/
sam2.1 comparison, same real pages/seeds, same shared bucket
classification (bakeoff.py::_bucket, mirrors sam_segmenter.py's own
priority order) -- answers whether SAM3 changes the bimodal
undershoot/overshoot pattern measured in production (sprint1_verify46:
~53% label_scale_noop+collapse, ~42% flood+over_seg, ~5% applied) or
just inherits it. See sam3_adapter.py's docstring for the two stated
platform confounds (imgsz cap, single-candidate mask) -- read before
interpreting any SAM3-favorable result here.

CASES are the two pages already used for the SAM1-vs-SAM2.1 comparison
(chosen because SAM1 was observed to flood on them) -- kept as-is
rather than expanding the sample, so this reuses an established,
unbiased-by-me case set. That selection criterion means these two
pages are flood-biased, not a random draw from the full population;
don't read their overshoot_frac as the whole-population rate (use the
sprint1_verify46 numbers above for that).
"""

import gc
import json
from pathlib import Path

import torch

from preprocessing_annotations.config import SAMConfig, resolve_sam3_checkpoint

from . import bakeoff
from .sam1_adapter import Sam1Adapter
from .sam2_adapter import Sam2Adapter
from .sam3_adapter import Sam3Adapter

CASES = [
    ("326_ROCKAWAY_page002", "sprint1_verify39/images/326 ROCKAWAY - AVI-ON LAYOUT_page002.png"),
    ("Bradley_page000", "sprint1_verify39/images/Bradley Fair_Avi-on Submittal_page000.png"),
]

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT = str(Path(__file__).resolve().parents[3] / "sam2_checkpoints" / "sam2.1_hiera_large.pt")


def _build_adapters(sam_config: SAMConfig):
    """Factories, not instances -- SAM1+SAM2.1+SAM3 do not fit in this
    project's GPU (14.6GB) simultaneously (measured: OOM building SAM3
    with SAM1+SAM2.1 still resident from the case-first loop this
    replaced). One model loaded/torn down at a time instead."""
    return {
        "sam1": lambda: Sam1Adapter(sam_config),
        "sam2.1": lambda: Sam2Adapter(
            SAM2_CONFIG, SAM2_CHECKPOINT, device="cuda", max_expand_frac=sam_config.max_expand_frac
        ),
        "sam3": lambda: Sam3Adapter(
            resolve_sam3_checkpoint(), max_expand_frac=sam_config.max_expand_frac, device="cuda"
        ),
    }


def main() -> None:
    sam_config = SAMConfig()
    adapter_factories = _build_adapters(sam_config)

    report: dict = {name: {} for name, _ in CASES}
    for model_name, build in adapter_factories.items():
        expander = build()
        for case_name, image_path in CASES:
            case_report = bakeoff.run_case(
                expander, image_path, f"{case_name}::{model_name}", sam_config.max_expand_frac
            )
            report[case_name][model_name] = case_report.summary()
            print(f"{case_name} / {model_name}: {case_report.summary()}", flush=True)
        del expander
        gc.collect()
        torch.cuda.empty_cache()

    # Filename carries the arm set, and existing files are never clobbered.
    # A bare "report.json" here already cost this project one artifact: the
    # original 2-arm (sam1/sam2.1) bake-off report was overwritten by the
    # first 3-arm run, destroying the provenance of the numbers recorded in
    # the sam21-zeroshot-bakeoff notes. Same failure class this project has
    # now hit three times (see the run-artifact fragility notes) -- fix it
    # at the write, not by remembering to be careful.
    out_dir = Path(__file__).resolve().parents[3] / "sam21_probe_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "report_" + "_".join(sorted(_build_adapters(sam_config)))
    out_path = out_dir / f"{stem}.json"
    suffix = 0
    while out_path.exists():
        suffix += 1
        out_path = out_dir / f"{stem}.{suffix}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"written: {out_path}", flush=True)


if __name__ == "__main__":
    main()
