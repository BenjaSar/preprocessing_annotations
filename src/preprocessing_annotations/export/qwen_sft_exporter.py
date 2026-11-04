"""Qwen3-VL SFT dataset: the pipeline's rooms as chat examples, one per tile.

Supersedes scripts/finetune_qwen.py, which targets Qwen2.5-VL, guesses each
bbox's convention (and gets it wrong for every room the pipeline emits:
413/413 on corpus_batch2_output came out with x2 < x1), and shuffles pages
into splits with no group awareness and no seed.

Four decisions this encodes, measured before the code existed:

D1  Targets are label boxes only (SFTDatasetConfig.box_convention). Rooms
    with another convention are skipped rather than mixed into one target.
D2  An example is the view inference gives the model. Pages above
    PipelineConfig.tile_trigger_px reach the VLM as tiles, so an example is
    one tile and its boxes are in that tile's pixels. The grid comes from
    TileSplitter.for_page, shared with the pipeline, so the two cannot drift.
D5  Tiles with no room are kept as negatives but capped at
    SFTDatasetConfig.max_empty_tile_fraction, exclusion-zone tiles first.
D6  A room is a target only in tiles that contain its whole box. Every room
    fits whole in at least one tile, so nothing is lost, and a box clipped
    at a tile edge would teach a truncated target.

Two things this export refuses to pass on silently: a room type that is only
taxonomy's unplaceable-name default is re-read from the room's own label and
dropped when nothing places it (SFTDatasetConfig.resolve_default_types_from_name),
and every export writes a seeded, magnified sample of its own targets
(export/sft_sample_audit.py) because these corpora have no room GT to score.

Splits come from the run's own coco/{train,val,test}.json, so the group-aware
assignment (export/group_split.py, GT documents pinned to test) is reused
rather than reimplemented. Geometry comes from `roomsRecognized` (xyxy, what
eval/room_gt_eval.py scores), not the legacy `rooms` key.

Usage:
    python -m preprocessing_annotations.export.qwen_sft_exporter \\
        --run <run_dir> --out <dataset_dir>
"""

import json
import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from ..automation.taxonomy import DEFAULT_MANDATORY_TYPE, resolve_mandatory
from ..config.config import PipelineConfig, SFTDatasetConfig
from ..export.group_split import SPLIT_NAMES, document_key
from ..export.sft_sample_audit import write_sample_audit
from ..ingestion.tile_splitter import TileMeta, TileSplitter
from ..vlm import prompt_templates

logger = logging.getLogger(__name__)

Box = Tuple[float, float, float, float]

# Tile images live beside the JSONL splits, under the dataset directory.
IMAGES_SUBDIR = "images"


@dataclass(frozen=True)
class Room:
    """One pipeline room in page pixels."""

    name: str
    room_type: str
    box: Box
    confidence: float


@dataclass
class ExportStats:
    """What the export kept and dropped, so a dataset can state its own shape."""

    pages: int = 0
    pages_tiled: int = 0
    rooms: int = 0
    rooms_skipped: Counter = field(default_factory=Counter)
    rooms_without_whole_tile: int = 0
    examples: int = 0
    targets: int = 0
    empty_tiles_available: int = 0
    empty_tiles_kept: int = 0
    empty_tiles_in_exclusion_zone: int = 0
    targets_dropped_by_document_cap: Counter = field(default_factory=Counter)
    per_split: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            **self.__dict__,
            "rooms_skipped": dict(sorted(self.rooms_skipped.items())),
            "targets_dropped_by_document_cap": dict(
                sorted(self.targets_dropped_by_document_cap.items())
            ),
            "per_split": dict(sorted(self.per_split.items())),
        }


def _tile_boxes(page: Tuple[int, int], config: PipelineConfig) -> List[Box]:
    """The tile rectangles inference would use on a page of this size."""
    splitter = TileSplitter.for_page(page[0], page[1], config)
    if splitter is None:
        return [(0.0, 0.0, float(page[0]), float(page[1]))]
    return [
        (float(m.x_offset), float(m.y_offset),
         float(m.x_offset + m.tile_w), float(m.y_offset + m.tile_h))
        for m in _tile_metas(page, splitter)
    ]


def _tile_metas(page: Tuple[int, int], splitter: TileSplitter) -> List[TileMeta]:
    """Tile geometry without rendering: split() needs only the page size."""
    return [meta for _, meta in splitter.split(Image.new("1", page))]


def _contains(outer: Box, inner: Box) -> bool:
    return (outer[0] <= inner[0] and outer[1] <= inner[1]
            and inner[2] <= outer[2] and inner[3] <= outer[3])


def _overlaps_fraction(tile: Box, zone: Sequence[float]) -> float:
    """Fraction of `tile` covered by an exclusion zone, both xyxy."""
    x1, y1 = max(tile[0], zone[0]), max(tile[1], zone[1])
    x2, y2 = min(tile[2], zone[2]), min(tile[3], zone[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    tile_area = (tile[2] - tile[0]) * (tile[3] - tile[1])
    return ((x2 - x1) * (y2 - y1) / tile_area) if tile_area else 0.0


def _to_tile(room: Room, tile: Box) -> Dict[str, Any]:
    """A room as the model should answer it: integer pixels inside the tile."""
    return {
        "room_name": room.name,
        "room_type": room.room_type,
        "bbox": [
            int(room.box[0] - tile[0]), int(room.box[1] - tile[1]),
            int(room.box[2] - tile[0]), int(room.box[3] - tile[1]),
        ],
        # The pipeline's own confidence, not a constant: TileSplitter.merge
        # resolves overlapping tiles by confidence, so a model trained on all
        # -1.0 answers would hand it ties to break arbitrarily.
        "confidence": round(room.confidence, 3),
    }


def rooms_of(
    annotation: Dict[str, Any],
    convention: str,
    min_name_ocr_confidence: float = 0.0,
    *,
    resolve_default_from_name: bool = False,
) -> Tuple[List[Room], Counter]:
    """Rooms carrying the trained box convention, and why the rest were skipped.

    Geometry comes from roomsRecognized, whose bbox is xyxy. A room whose
    label was read below `min_name_ocr_confidence` is skipped too (D9): the
    target string is trained verbatim, so a misread name teaches a misreading.
    The default keeps every room, which is the behaviour measured so far.

    With `resolve_default_from_name` (SFTDatasetConfig.
    resolve_default_types_from_name) a room whose stored type is the
    taxonomy's unplaceable-name default is re-resolved from its own label and
    dropped when that resolves to nothing, so the dataset never teaches a
    guessed class. Every other stored type is kept as the run recorded it.

    Skips are counted by reason, not totalled: a dataset that silently shrank
    should say which rule shrank it.
    """
    rooms: List[Room] = []
    skipped: Counter = Counter()
    for room in annotation.get("roomsRecognized") or []:
        coordinates = room.get("coordinates") or {}
        box = coordinates.get("bbox")
        if coordinates.get("box_convention") != convention:
            skipped["wrong_convention"] += 1
            continue
        if not box or len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            skipped["unusable_box"] += 1
            continue
        if float((room.get("confidence_detail") or {}).get("ocr", 1.0)) < min_name_ocr_confidence:
            skipped["low_name_ocr_confidence"] += 1
            continue
        name = (room.get("name") or "").strip()
        if not name:
            skipped["no_name"] += 1
            continue
        room_type = (room.get("type") or "OTHER").strip()
        if resolve_default_from_name and room_type == DEFAULT_MANDATORY_TYPE:
            resolved = resolve_mandatory(name)
            if resolved is None:
                skipped["unresolved_type"] += 1
                continue
            room_type = resolved
        rooms.append(Room(
            name=name,
            room_type=room_type,
            box=tuple(float(v) for v in box),
            confidence=float(room.get("confidence", 0.0)),
        ))
    return rooms, skipped


def page_examples(
    annotation: Dict[str, Any],
    page_image: Path,
    pipeline_config: PipelineConfig,
    sft_config: SFTDatasetConfig,
    stats: ExportStats,
) -> Tuple[List[dict], List[dict]]:
    """One page's examples, split into those with rooms and the empty ones.

    Empty tiles are returned separately because keeping them is a corpus-wide
    budget (D5), not a per-page choice.
    """
    size = annotation.get("image_size") or {}
    page = (int(size.get("width", 0)), int(size.get("height", 0)))
    if page[0] <= 0 or page[1] <= 0:
        logger.warning("skipping %s: no image size", page_image.name)
        return [], []

    rooms, skipped = rooms_of(
        annotation, sft_config.box_convention, sft_config.min_name_ocr_confidence,
        resolve_default_from_name=sft_config.resolve_default_types_from_name,
    )
    stats.pages += 1
    stats.rooms += len(rooms)
    stats.rooms_skipped.update(skipped)

    tiles = _tile_boxes(page, pipeline_config)
    if len(tiles) > 1:
        stats.pages_tiled += 1
    zones = annotation.get("exclusion_zones") or []

    placed = set()
    with_rooms, empty = [], []
    for tile in tiles:
        targets = [
            _to_tile(room, tile) for room in rooms
            if _contains(tile, room.box)  # D6: whole box only
        ]
        for index, room in enumerate(rooms):
            if _contains(tile, room.box):
                placed.add(index)
        example = _example(page_image, tile, targets, pipeline_config)
        if targets:
            with_rooms.append(example)
        else:
            example["metadata"]["in_exclusion_zone"] = any(
                _overlaps_fraction(tile, zone) > 0.5 for zone in zones
            )
            empty.append(example)

    stats.rooms_without_whole_tile += len(rooms) - len(placed)
    return with_rooms, empty


def _crop_name(page_image: Path, tile: Box) -> str:
    """Published path of a tile's own image file, inside the dataset dir.

    The crop is written rather than referenced as a region of the page,
    because a trainer reads `messages[].image` and nothing else: pointing it
    at the full page while the prompt states the tile's size and the boxes
    are tile-local would train mismatched pairs, silently.

    Relative so the dataset directory can be moved or copied whole; the
    trainer resolves it against the JSONL's own directory.
    """
    x1, y1, x2, y2 = (int(v) for v in tile)
    return f"{IMAGES_SUBDIR}/{page_image.stem}_{x1}_{y1}_{x2}_{y2}.png"


def _example(
    page_image: Path, tile: Box, targets: List[dict], pipeline_config: PipelineConfig
) -> dict:
    """One chat example: the tile the model sees, and the answer it should give."""
    width, height = int(tile[2] - tile[0]), int(tile[3] - tile[1])
    prompt = prompt_templates.build_room_prompt_sft_label(
        width, height, pipeline_config.vlm.max_rooms
    )
    crop_name = _crop_name(page_image, tile)
    return {
        "messages": [
            {"role": "system", "content": [
                {"type": "text", "text": prompt_templates.build_json_array_system_prompt()},
            ]},
            {"role": "user", "content": [
                {"type": "image", "image": crop_name},
                {"type": "text", "text": prompt},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": json.dumps(targets)},
            ]},
        ],
        "metadata": {
            "image": crop_name,
            "source_page": str(page_image),
            "crop": [int(v) for v in tile],
            "tile_size": [width, height],
            "rooms": len(targets),
        },
    }


def _write_crops(examples: Sequence[dict], output_dir: Path) -> None:
    """Write each kept example's tile image, one page load per page."""
    (output_dir / IMAGES_SUBDIR).mkdir(parents=True, exist_ok=True)
    by_page: Dict[str, List[dict]] = {}
    for example in examples:
        by_page.setdefault(example["metadata"]["source_page"], []).append(example)
    for source_page, page_examples_ in sorted(by_page.items()):
        with Image.open(source_page) as page:
            page = page.convert("RGB")
            for example in page_examples_:
                crop_path = output_dir / example["metadata"]["image"]
                page.crop(tuple(example["metadata"]["crop"])).save(crop_path)


def _split_pages(run_dir: Path) -> Dict[str, List[str]]:
    """Image file name -> split, from the run's own group-aware COCO export."""
    pages: Dict[str, List[str]] = {}
    for split in SPLIT_NAMES:
        path = run_dir / "coco" / f"{split}.json"
        if not path.exists():
            logger.warning("no %s split in %s", split, path.parent)
            continue
        pages[split] = [image["file_name"] for image in json.loads(path.read_text())["images"]]
    return pages


def _capped_empty(
    empty: List[dict], kept_with_rooms: int, sft_config: SFTDatasetConfig
) -> List[dict]:
    """Empty tiles up to the cap: exclusion-zone ones first, then sampled (D5)."""
    fraction = sft_config.max_empty_tile_fraction
    budget = int(round(kept_with_rooms * fraction / (1.0 - fraction))) if fraction < 1.0 else len(empty)
    if len(empty) <= budget:
        return empty
    in_zone = [e for e in empty if e["metadata"].get("in_exclusion_zone")]
    rest = [e for e in empty if not e["metadata"].get("in_exclusion_zone")]
    random.Random(sft_config.empty_tile_sample_seed).shuffle(rest)
    return (in_zone + rest)[:budget]


def _capped_by_document(
    with_rooms: List[dict], sft_config: SFTDatasetConfig
) -> Tuple[List[dict], Dict[str, int]]:
    """Tiles kept after no document may exceed max_targets_per_document.

    Whole tiles are dropped, never individual targets: the answer has to list
    every room visible in the tile, so removing one target from a tile that
    still shows its label trains a false negative. See the config field for
    the measurement that rules the per-label alternative out.

    Within an over-budget document, tiles are taken densest-first after a
    seeded shuffle, so the cap keeps the most targets per tile (and per crop
    written) while staying reproducible.
    """
    cap = sft_config.max_targets_per_document
    if cap is None:
        return with_rooms, {}

    by_document: Dict[str, List[dict]] = {}
    for example in with_rooms:
        key = document_key(Path(example["metadata"]["source_page"]).name)
        by_document.setdefault(key, []).append(example)

    kept, dropped = [], {}
    shuffler = random.Random(sft_config.document_cap_sample_seed)
    for key, examples in sorted(by_document.items()):
        total = sum(e["metadata"]["rooms"] for e in examples)
        if total <= cap:
            kept.extend(examples)
            continue
        shuffler.shuffle(examples)
        examples.sort(key=lambda e: e["metadata"]["rooms"], reverse=True)
        running, taken = 0, []
        for example in examples:
            if running + example["metadata"]["rooms"] > cap:
                continue
            taken.append(example)
            running += example["metadata"]["rooms"]
        kept.extend(taken)
        dropped[key] = total - running
        logger.info(
            "document cap: %s contributed %d targets, kept %d", key, total, running
        )
    return kept, dropped


def export(
    run_dir: Path,
    output_dir: Path,
    pipeline_config: Optional[PipelineConfig] = None,
    sft_config: Optional[SFTDatasetConfig] = None,
) -> Dict[str, Any]:
    """Write train/val/test JSONL for the run, and a stats report beside them."""
    pipeline_config = pipeline_config or PipelineConfig()
    sft_config = sft_config or SFTDatasetConfig()
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = ExportStats()
    audited: List[Tuple[str, dict]] = []
    for split, file_names in _split_pages(run_dir).items():
        with_rooms, empty = [], []
        for file_name in sorted(file_names):
            annotation_path = run_dir / "processed_annotations" / f"{Path(file_name).stem}.json"
            if not annotation_path.exists():
                logger.warning("no annotation for %s", file_name)
                continue
            page_with_rooms, page_empty = page_examples(
                json.loads(annotation_path.read_text()),
                (run_dir / "images" / file_name).resolve(),
                pipeline_config, sft_config, stats,
            )
            with_rooms.extend(page_with_rooms)
            empty.extend(page_empty)

        # Before the empty-tile cap, so the empty budget is computed against
        # the tiles that actually survive (the two caps compose in one order
        # only: emptiness is a share of what is kept).
        with_rooms, dropped_by_cap = _capped_by_document(with_rooms, sft_config)
        for key, count in dropped_by_cap.items():
            stats.targets_dropped_by_document_cap[key] += count

        stats.empty_tiles_available += len(empty)
        kept_empty = _capped_empty(empty, len(with_rooms), sft_config)
        stats.empty_tiles_kept += len(kept_empty)
        stats.empty_tiles_in_exclusion_zone += sum(
            1 for e in kept_empty if e["metadata"].get("in_exclusion_zone")
        )

        examples = with_rooms + kept_empty
        stats.examples += len(examples)
        stats.targets += sum(e["metadata"]["rooms"] for e in examples)
        stats.per_split[split] = len(examples)
        # Crops only for the examples that survived the empty-tile cap.
        _write_crops(examples, output_dir)
        _write_jsonl(output_dir / f"{split}.jsonl", examples)
        audited.extend((split, example) for example in with_rooms)

    report = {
        "run": str(run_dir),
        "box_convention": sft_config.box_convention,
        "max_empty_tile_fraction": sft_config.max_empty_tile_fraction,
        "max_targets_per_document": sft_config.max_targets_per_document,
        "resolve_default_types_from_name": sft_config.resolve_default_types_from_name,
        "prompt_version": prompt_templates.PROMPT_VERSIONS["build_room_prompt_sft_label"],
        **stats.as_dict(),
        # Written last: the audit reads the tile crops the splits above wrote.
        "sample_audit": write_sample_audit(audited, output_dir, sft_config),
    }
    (output_dir / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _write_jsonl(path: Path, examples: Sequence[dict]) -> None:
    with path.open("w") as handle:
        for example in examples:
            handle.write(json.dumps(example) + "\n")
    logger.info("wrote %d examples to %s", len(examples), path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the Qwen3-VL SFT dataset from a run")
    parser.add_argument("--run", type=Path, required=True, help="Run dir with coco/ and processed_annotations/")
    parser.add_argument("--out", type=Path, required=True, help="Directory for the JSONL splits")
    parser.add_argument(
        "--max-targets-per-document", type=int, default=None,
        help="SFTDatasetConfig.max_targets_per_document (default: off)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sft_config = SFTDatasetConfig(max_targets_per_document=args.max_targets_per_document)
    print(json.dumps(export(args.run, args.out, sft_config=sft_config), indent=2))
