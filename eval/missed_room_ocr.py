"""Why did a GT room never get a room candidate?

Of the room-GT pages' rooms, those whose polygon contains no candidate centre
(VLM `rooms` + `ocr_rooms` of the run's raw annotations) are "missed" before
any filter runs. Each missed room is sorted by OCR evidence, without the VLM:

  text_at_run_resolution          OCR on the run's own page image finds text
                                  inside the room: the text exists but never
                                  became a candidate (vocabulary/pattern gap, or
                                  a non-name token such as a room number).
  text_only_at_higher_resolution  no text at run resolution; OCR on a crop
                                  re-rendered from the PDF at a higher raster
                                  scale finds some: resolution.
  no_text                         neither pass finds text inside the room.

OCR is the pipeline's own MEPTextExtractor.extract_text, the call that
produced the run's raw detections. Page images are <= its resize cap, so run
coordinates equal OCR coordinates. Limits: an OCR confidence filter can drop
real text at run resolution that the higher-resolution pass then recovers --
that lands in the resolution bucket. no_text does not prove no label is drawn.
"""

import argparse
import json
import logging
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import fitz
from PIL import Image
from shapely.geometry import Point, Polygon

from preprocessing_annotations.config import (
    MissedRoomOcrConfig,
    PipelineConfig,
    RoomGtEvalConfig,
)
from preprocessing_annotations.ingestion.ocr_extractor import (
    MEPTextExtractor,
    TextDetection,
    _centroid,
)

from report_io import emit_report
from room_gt_eval import (
    BoxXYXY,
    load_gt_categories,
    load_gt_polygons,
    page_key,
    room_gt_index,
    xywh_to_xyxy,
)
from run_images import find_page_image

logger = logging.getLogger(__name__)

TEXT_AT_RUN_RESOLUTION = "text_at_run_resolution"
TEXT_ONLY_AT_HIGHER_RESOLUTION = "text_only_at_higher_resolution"
NO_TEXT = "no_text"

OcrFn = Callable[[Path], List[TextDetection]]

_PAGE_STEM = re.compile(r"^(?P<document>.+)_page(?P<index>\d+)$")


def candidate_boxes(annotation: Dict) -> List[BoxXYXY]:
    """Every room candidate of a raw annotation (VLM rooms + OCR rooms), xyxy."""
    return [
        xywh_to_xyxy(room["bbox"])
        for room in annotation.get("rooms", []) + annotation.get("ocr_rooms", [])
        if len(room.get("bbox", [])) == 4
    ]


def missed_gt_indices(gts: List[Polygon], candidates: List[BoxXYXY]) -> List[int]:
    """GT rooms whose polygon contains no candidate centre."""
    located = {i for i in room_gt_index(candidates, gts) if i is not None}
    return [i for i in range(len(gts)) if i not in located]


def texts_inside(
    polygon: Polygon, detections: Sequence[TextDetection],
    offset: Tuple[float, float] = (0.0, 0.0), scale: float = 1.0,
) -> List[str]:
    """Texts whose centre, mapped to run pixels as offset + centre / scale,
    lies inside the polygon."""
    texts = []
    for detection in detections:
        cx, cy = _centroid(detection.bbox)
        if polygon.contains(Point(offset[0] + cx / scale, offset[1] + cy / scale)):
            texts.append(detection.text)
    return texts


class PdfPageRenderer:
    """Re-renders a run page from its source PDF at a multiple of the run's
    raster scale. The run image and the render are both rasters of the same
    (rotated) page, so run pixel x maps to render pixel x * ratio."""

    def __init__(self, pdf_dir: Path, factor: float):
        self._pdf_dir = pdf_dir
        self._factor = factor

    def render(self, page_stem: str, run_size: Tuple[int, int]) -> Tuple[Image.Image, float]:
        match = _PAGE_STEM.match(page_stem)
        if match is None:
            raise ValueError(f"{page_stem}: not a <document>_pageNNN stem")
        with fitz.open(self._pdf_dir / f"{match['document']}.pdf") as document:
            page = document[int(match["index"])]
            run_scale = run_size[0] / page.rect.width
            pixmap = page.get_pixmap(matrix=fitz.Matrix(run_scale * self._factor, run_scale * self._factor))
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        ratio = image.width / run_size[0]
        if abs(image.height - run_size[1] * ratio) > ratio:
            raise ValueError(
                f"{page_stem}: render {image.size} is not a uniform scale of run {run_size}"
            )
        return image, ratio


def _ocr_crop(ocr: OcrFn, image: Image.Image, box: Tuple[int, int, int, int]) -> List[TextDetection]:
    with tempfile.NamedTemporaryFile(suffix=".png") as handle:
        image.crop(box).save(handle.name)
        return ocr(Path(handle.name))


@dataclass
class MissedRoom:
    page: str
    gt_index: int
    gt_category: str
    cause: str
    texts: List[str] = field(default_factory=list)


def classify_page(
    page_stem: str, run_image: Path, gts: List[Polygon], categories: List[str],
    candidates: List[BoxXYXY], ocr: OcrFn, renderer: PdfPageRenderer,
) -> List[MissedRoom]:
    """Sort one page's missed GT rooms by OCR evidence."""
    missed = missed_gt_indices(gts, candidates)
    if not missed:
        return []
    detections = ocr(run_image)
    rooms: List[MissedRoom] = []
    pending: List[int] = []
    for index in missed:
        texts = texts_inside(gts[index], detections)
        if texts:
            rooms.append(MissedRoom(page_stem, index, categories[index], TEXT_AT_RUN_RESOLUTION, texts))
        else:
            pending.append(index)
    if pending:
        with Image.open(run_image) as image:
            run_size = image.size
        render, ratio = renderer.render(page_stem, run_size)
        for index in pending:
            x1, y1, x2, y2 = gts[index].bounds
            crop_box = (int(x1 * ratio), int(y1 * ratio), int(x2 * ratio) + 1, int(y2 * ratio) + 1)
            texts = texts_inside(
                gts[index], _ocr_crop(ocr, render, crop_box),
                offset=(crop_box[0] / ratio, crop_box[1] / ratio), scale=ratio,
            )
            cause = TEXT_ONLY_AT_HIGHER_RESOLUTION if texts else NO_TEXT
            rooms.append(MissedRoom(page_stem, index, categories[index], cause, texts))
    return rooms


def evaluate(
    eval_config: RoomGtEvalConfig, run_dir: Path, pdf_dir: Path, ocr: OcrFn,
    renderer: PdfPageRenderer,
) -> Dict:
    gt = load_gt_polygons(eval_config.gt_coco_path)
    categories = load_gt_categories(eval_config.gt_coco_path)
    rooms: List[MissedRoom] = []
    pages = gt_rooms = 0
    for path in sorted((run_dir / "annotations").glob("*.json")):
        key = page_key(path.name)
        if key not in gt:
            continue
        pages += 1
        gt_rooms += len(gt[key])
        logger.info("missed-room OCR: %s", path.stem)
        annotation = json.loads(path.read_text())
        rooms += classify_page(
            path.stem, find_page_image(run_dir, annotation), gt[key], categories[key],
            candidate_boxes(annotation), ocr, renderer,
        )
    return {
        "run_dir": str(run_dir),
        "pdf_dir": str(pdf_dir),
        "gt_coco_path": str(eval_config.gt_coco_path),
        "pages_scored": pages,
        "gt_rooms": gt_rooms,
        "missed_gt_rooms": len(rooms),
        "by_cause": dict(Counter(room.cause for room in rooms)),
        "by_cause_and_category": {
            cause: dict(Counter(r.gt_category for r in rooms if r.cause == cause).most_common())
            for cause in (TEXT_AT_RUN_RESOLUTION, TEXT_ONLY_AT_HIGHER_RESOLUTION, NO_TEXT)
        },
        "rooms": [room.__dict__ for room in rooms],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort GT rooms that got no candidate by OCR evidence")
    parser.add_argument("--run", type=Path, required=True, help="Pipeline output dir (raw annotations + images)")
    parser.add_argument("--pdf-dir", type=Path, required=True, help="Source PDFs of that run")
    parser.add_argument("--rerender-scale-factor", type=float, default=None,
                        help="Override MissedRoomOcrConfig.rerender_scale_factor")
    parser.add_argument("--out", type=Path, default=None, help="Also write report JSON here")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    # PaddleOCR resets the root logger to WARNING after its first call; keep
    # this module's per-page progress visible.
    logger.setLevel(logging.INFO)
    missed_config = MissedRoomOcrConfig()
    if args.rerender_scale_factor is not None:
        missed_config.rerender_scale_factor = args.rerender_scale_factor
    extractor = MEPTextExtractor(PipelineConfig().ocr)
    emit_report(
        evaluate(
            RoomGtEvalConfig(), args.run, args.pdf_dir, extractor.extract_text,
            PdfPageRenderer(args.pdf_dir, missed_config.rerender_scale_factor),
        ),
        args.out,
    )
