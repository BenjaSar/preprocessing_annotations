"""Crop re-read probe: CropRereader over a finished run's saved OCR.

For each raw annotation carrying ocrTextDetections, selects fragments with
the CURRENT code's room-label patterns, crops them from the run's page
image and transcribes each with the Unsloth VLM. Writes every raw
transcription (accepted or not) and per-crop seconds; the run is untouched.

Scoring is separate: candidate_replay.py --rereads applies the transcripts
under the consistency guard, so selector bounds and the guard are re-picked
offline without the GPU. Probe at the broadest selector bounds to be swept
-- a narrower selection is a subset, a wider one is refused by the replay.

Holds the pipeline's single-instance lock while reading: the VLM fits the
GPU only once, so a concurrent pipeline run or probe waits instead of
failing the 4-bit model load.
"""

import argparse
import dataclasses
import json
import logging
import time
from pathlib import Path
from statistics import median
from typing import Any, Dict, List

from PIL import Image

from preprocessing_annotations.config import CropRereadConfig, PipelineConfig
from preprocessing_annotations.ingestion.crop_reread import CropRereader, Reread
from preprocessing_annotations.ingestion.ocr_extractor import MEPTextExtractor
from preprocessing_annotations.orchestration.run_lock import single_instance
from preprocessing_annotations.vlm.read import CropTextReader, UnslothCropTextReader
from preprocessing_annotations.vlm.vlm_backend import UnslothQwenBackend

from candidate_replay import load_text_detections
from crop_reread_records import reread_selector, rereads_payload
from report_io import emit_report
from room_replay import parse_overrides
from run_images import find_page_image

logger = logging.getLogger(__name__)


class TimedReader:
    """CropTextReader decorator recording seconds per read."""

    def __init__(self, reader: CropTextReader):
        self._reader = reader
        self.seconds: List[float] = []

    def read(self, image: Any) -> str:
        start = time.perf_counter()
        text = self._reader.read(image)
        self.seconds.append(time.perf_counter() - start)
        return text


def probe_run(run_dir: Path, page_glob: str, rereader: CropRereader) -> Dict[str, List[Reread]]:
    pages: Dict[str, List[Reread]] = {}
    for path in sorted((run_dir / "annotations").glob(f"{page_glob}.json")):
        annotation = json.loads(path.read_text())
        detections = load_text_detections(annotation)
        with Image.open(find_page_image(run_dir, annotation)) as image:
            pages[path.stem] = rereader.reread(image.convert("RGB"), detections)
        logger.info("%s: %d crops", path.stem, len(pages[path.stem]))
    return pages


def probe_report(
    run_dir: Path, model: str, config: CropRereadConfig,
    pages: Dict[str, List[Reread]], seconds: List[float],
) -> Dict[str, Any]:
    rereads = [r for page in pages.values() for r in page]
    return {
        "run_dir": str(run_dir),
        "model": model,
        "selector": dataclasses.asdict(config),
        "crops": len(rereads),
        "empty_reads": sum(not r.reread for r in rereads),
        "unchanged_reads": sum(r.reread == r.original for r in rereads),
        "seconds_per_crop": {
            "median": median(seconds) if seconds else None,
            "max": max(seconds, default=None),
            "total": sum(seconds),
        },
        "pages": rereads_payload(pages),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM re-read of OCR fragments in a finished run")
    parser.add_argument("--run", type=Path, required=True, help="Run dir whose raw annotations carry ocrTextDetections")
    parser.add_argument("--out", type=Path, required=True, help="Re-read file (input of candidate_replay --rereads)")
    parser.add_argument(
        "--selector", nargs="*", default=[], metavar="KEY=VALUE",
        help="CropRereadConfig overrides; probe at the broadest bounds to be swept",
    )
    parser.add_argument("--pages", default="*", help="Annotation file-stem glob (default: all pages)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("preprocessing_annotations").setLevel(logging.WARNING)
    pipeline_config = PipelineConfig()
    reread_config = parse_overrides(CropRereadConfig(), args.selector)
    backend = UnslothQwenBackend(pipeline_config.vlm)
    reader = TimedReader(UnslothCropTextReader(backend))
    rereader = CropRereader(
        reread_selector(MEPTextExtractor(pipeline_config.ocr), reread_config), reader, reread_config
    )
    with single_instance():
        pages = probe_run(args.run, args.pages, rereader)
    emit_report(
        probe_report(args.run, pipeline_config.vlm.unsloth_model, reread_config, pages, reader.seconds),
        args.out,
    )
