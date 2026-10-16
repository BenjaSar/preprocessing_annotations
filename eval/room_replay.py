"""H2 (2026-10-11): replay the SFT filter stage of a finished run under
alternative RoomFilterConfig values and score it against room-polygon GT.

Why: after N1/N1b every drop point records per-room detail, so filter
tuning (N4/N5b) can be judged offline from a run's raw annotations in
minutes instead of re-running the 7 h pipeline. Read-only on the run dir.

Scope: replays prepare_sft_annotation only. Anything upstream (OCR, VLM,
SAM) is whatever the run produced; a variant that changes upstream
behavior cannot be evaluated here.

Reports label-localization precision/recall overall and per document, and
lists documents whose GT-room recall fell under the variant (a tuning must
not regress any document). `as_run` scores the run's own recorded final
rooms: the baseline replay equals it only if the filter code on disk is the
code the run used, so any difference is code drift, not a tuning effect.
"""

import argparse
import dataclasses
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Sequence, Tuple, TypeVar

from preprocessing_annotations.automation import prepare_sft_annotation
from preprocessing_annotations.config import (
    PipelineConfig,
    RoomFilterConfig,
    RoomGtEvalConfig,
)

from held_out import leave_one_document_out
from report_io import emit_report
from room_gt_eval import (
    BoxXYXY,
    CategoryScore,
    ExtentScore,
    LocalizationScore,
    PredictedRoom,
    boxes_of,
    load_gt_categories,
    load_gt_category_map,
    load_gt_polygons,
    load_run_rooms,
    page_key,
    score_category,
    score_extent,
    score_localization,
    xywh_to_xyxy,
)

_PAGE_SUFFIX = re.compile(r"_page\d+$")


def document_key(key: str) -> str:
    """Page key -> document key (page index stripped)."""
    return _PAGE_SUFFIX.sub("", key)


ConfigT = TypeVar("ConfigT")


def parse_overrides(base: ConfigT, overrides: Sequence[str]) -> ConfigT:
    """Apply `KEY=VALUE` overrides to a config dataclass, typed after the field's default."""
    values: Dict[str, Any] = {}
    fields = {f.name for f in dataclasses.fields(base)}
    for override in overrides:
        name, _, raw = override.partition("=")
        if name not in fields or not raw:
            raise ValueError(
                f"bad override '{override}': expected KEY=VALUE, KEY in {sorted(fields)}"
            )
        values[name] = _parse_value(type(getattr(base, name)), raw)
    return dataclasses.replace(base, **values)


def _parse_value(kind: type, raw: str) -> Any:
    if kind is not bool:
        return kind(raw)
    if raw.lower() not in ("true", "false"):
        raise ValueError(f"bad bool value '{raw}': expected true or false")
    return raw.lower() == "true"


def replay_annotation(
    annotation: Dict[str, Any], image_dir: Path, config: PipelineConfig,
    room_filter: RoomFilterConfig,
) -> Tuple[List[PredictedRoom], Dict[str, int]]:
    """Final rooms (box xyxy + class) and drop_attribution for one raw annotation."""
    processed = prepare_sft_annotation(
        annotation,
        min_rooms_for_sft=config.min_rooms_for_sft,
        image_dir=image_dir,
        use_window_elongation_filter=config.use_window_elongation_filter,
        exempt_sam_expanded_from_ink_filter=config.exempt_sam_expanded_from_ink_filter,
        block_single_letter_abbreviations=config.block_single_letter_abbreviations,
        room_filter=room_filter,
    )
    rooms = [
        PredictedRoom(xywh_to_xyxy(room["bbox"]), room.get("type"))
        for room in processed.get("rooms", [])
        if len(room.get("bbox", [])) == 4
    ]
    return rooms, processed.get("drop_attribution", {})


def replay_page(
    annotation_path: Path, image_dir: Path, config: PipelineConfig,
    room_filter: RoomFilterConfig,
) -> Tuple[List[PredictedRoom], Dict[str, int]]:
    """replay_annotation on a raw annotation file."""
    return replay_annotation(
        json.loads(annotation_path.read_text()), image_dir, config, room_filter
    )


def replay_run(
    run_dir: Path, config: PipelineConfig, room_filter: RoomFilterConfig,
) -> Tuple[Dict[str, List[PredictedRoom]], Dict[str, int]]:
    """Replay every page in run_dir/annotations (GT or not)."""
    rooms: Dict[str, List[PredictedRoom]] = {}
    drops: Dict[str, int] = {}
    for path in sorted((run_dir / "annotations").glob("*.json")):
        key = page_key(path.name)
        rooms[key], page_drops = replay_page(path, run_dir / "images", config, room_filter)
        for reason, count in page_drops.items():
            drops[reason] = drops.get(reason, 0) + count
    return rooms, drops


def score_replay(
    rooms: Dict[str, List[PredictedRoom]], gt: Dict[str, List[Any]], iou_threshold: float,
) -> Tuple[LocalizationScore, Dict[str, LocalizationScore]]:
    """Overall and per-document localization scores for the replayed rooms."""
    overall = LocalizationScore()
    per_document: Dict[str, LocalizationScore] = {}
    for key, page_rooms in rooms.items():
        boxes = boxes_of(page_rooms)
        score_localization(boxes, gt[key], iou_threshold, overall)
        score_localization(
            boxes, gt[key], iou_threshold,
            per_document.setdefault(document_key(key), LocalizationScore()),
        )
    return overall, per_document


def score_replay_category(
    rooms: Dict[str, List[PredictedRoom]], gt: Dict[str, List[Any]],
    gt_categories: Mapping[str, Sequence[str]],
    class_of_category: Mapping[str, Any],
) -> CategoryScore:
    """Category score for the replayed rooms, re-derived by current code."""
    score = CategoryScore()
    for key, page_rooms in rooms.items():
        score_category(page_rooms, gt[key], gt_categories.get(key, []), class_of_category, score)
    return score


def score_replay_extent(
    rooms: Dict[str, List[PredictedRoom]], gt: Dict[str, List[Any]],
    max_area_ratio: float,
) -> ExtentScore:
    """Box-size score for the replayed rooms: oversize is the one failure
    bbox IoU cannot show, so every variant must report it."""
    score = ExtentScore()
    for key, page_rooms in rooms.items():
        score_extent(boxes_of(page_rooms), gt[key], max_area_ratio, score)
    return score


class GtReference(NamedTuple):
    """The GT a replay is scored against, as one collaborator."""

    polygons: Dict[str, List[Any]]
    categories: Mapping[str, Sequence[str]]
    class_of_category: Mapping[str, Any]

    @classmethod
    def load(cls, eval_config: RoomGtEvalConfig) -> "GtReference":
        return cls(
            load_gt_polygons(eval_config.gt_coco_path),
            load_gt_categories(eval_config.gt_coco_path),
            load_gt_category_map(eval_config.gt_category_map_path)["mapping"],
        )


def _summary(
    rooms: Dict[str, List[PredictedRoom]], drops: Dict[str, int],
    reference: GtReference, eval_config: RoomGtEvalConfig,
) -> Tuple[Dict[str, Any], Dict[str, LocalizationScore]]:
    overall, per_document = score_replay(
        rooms, reference.polygons, eval_config.gt_iou_threshold
    )
    category = score_replay_category(
        rooms, reference.polygons, reference.categories, reference.class_of_category
    )
    extent = score_replay_extent(
        rooms, reference.polygons, eval_config.max_box_to_gt_area_ratio
    )
    return {
        "localization": overall.to_dict(),
        "extent_size": extent.to_dict(),
        "category": category.to_dict(),
        "drop_attribution": dict(sorted(drops.items(), key=lambda kv: -kv[1])),
        "per_document": {doc: s.to_dict() for doc, s in sorted(per_document.items())},
    }, per_document


def regressed_documents(
    baseline: Dict[str, LocalizationScore], variant: Dict[str, LocalizationScore],
) -> List[str]:
    """Documents whose located GT rooms fell under the variant."""
    return sorted(
        doc for doc, score in baseline.items()
        if variant[doc].gt_rooms_located < score.gt_rooms_located
    )


def room_count_changes(
    baseline: Dict[str, List[BoxXYXY]], variant: Dict[str, List[BoxXYXY]],
) -> Dict[str, List[int]]:
    """[baseline, variant] final-room counts for every page where they differ,
    GT page or not: the variant's blast radius."""
    return {
        page: [len(baseline[page]), len(variant[page])]
        for page in sorted(baseline) if len(baseline[page]) != len(variant[page])
    }


def evaluate_replay(
    eval_config: RoomGtEvalConfig, run_dir: Path, pipeline_config: PipelineConfig,
    variant_filter: RoomFilterConfig,
) -> Dict[str, Any]:
    reference = GtReference.load(eval_config)
    gt = reference.polygons
    runs = {
        name: replay_run(run_dir, pipeline_config, room_filter)
        for name, room_filter in (
            ("baseline", pipeline_config.room_filter), ("variant", variant_filter),
        )
    }
    scored = {
        name: {page: boxes for page, boxes in rooms.items() if page in gt}
        for name, (rooms, _) in runs.items()
    }
    summaries = {
        name: _summary(scored[name], runs[name][1], reference, eval_config)
        for name in runs
    }
    as_run_rooms = {
        key: rooms for key, rooms in load_run_rooms(run_dir).items() if key in scored["variant"]
    }
    return {
        "run_dir": str(run_dir),
        "gt_coco_path": str(eval_config.gt_coco_path),
        "pages_replayed": len(runs["variant"][0]),
        "pages_scored": len(scored["variant"]),
        "gt_pages_missing_from_run": sorted(set(gt) - set(scored["variant"])),
        "baseline_room_filter": dataclasses.asdict(pipeline_config.room_filter),
        "variant_room_filter": dataclasses.asdict(variant_filter),
        "as_run": _summary(as_run_rooms, {}, reference, eval_config)[0],
        "baseline": summaries["baseline"][0],
        "variant": summaries["variant"][0],
        "documents_with_recall_drop": regressed_documents(
            summaries["baseline"][1], summaries["variant"][1]
        ),
        "room_count_changes": room_count_changes(runs["baseline"][0], runs["variant"][0]),
    }


def evaluate_sweep(
    eval_config: RoomGtEvalConfig, run_dir: Path, pipeline_config: PipelineConfig,
    setting: str,
) -> Dict[str, Any]:
    """Replay one RoomFilterConfig field over `KEY=v1,v2,...` (the current
    value is always candidate 0) and judge the choice leave-one-document-out."""
    name, _, raw_values = setting.partition("=")
    base = pipeline_config.room_filter
    variants = [parse_overrides(base, [f"{name}={raw}"]) for raw in raw_values.split(",")]
    default = getattr(base, name)
    candidates = {default: base}
    for variant in variants:
        candidates.setdefault(getattr(variant, name), variant)
    reference = GtReference.load(eval_config)
    gt = reference.polygons
    scores: Dict[Any, Dict[str, Any]] = {}
    report: Dict[str, Any] = {}
    for value, room_filter in candidates.items():
        rooms, drops = replay_run(run_dir, pipeline_config, room_filter)
        summary, per_document = _summary(
            {page: boxes for page, boxes in rooms.items() if page in gt},
            drops, reference, eval_config,
        )
        scores[value] = per_document
        # Same scoring path as evaluate_replay; per-document detail is left
        # out because a sweep reports it once, as the recall-drop list below.
        report[str(value)] = {
            key: summary[key] for key in ("localization", "extent_size", "category")
        }
    for value in scores:
        report[str(value)]["documents_with_recall_drop"] = regressed_documents(
            scores[default], scores[value]
        )
    return {
        "run_dir": str(run_dir),
        "field": name,
        "default": default,
        "candidates": report,
        "leave_one_document_out": leave_one_document_out(scores, default).to_dict(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="H2: replay the SFT filter stage under RoomFilterConfig overrides"
    )
    parser.add_argument("--run", type=Path, required=True, help="Pipeline output dir")
    parser.add_argument(
        "--room-filter", action="append", default=[], metavar="KEY=VALUE",
        help="RoomFilterConfig override, repeatable (variant run only)",
    )
    parser.add_argument(
        "--sweep", default=None, metavar="KEY=V1,V2,...",
        help="Replay one RoomFilterConfig field over these values (current value "
             "included) and judge the choice leave-one-document-out; replaces --room-filter",
    )
    parser.add_argument("--out", type=Path, default=None, help="Also write report JSON here")
    args = parser.parse_args()

    logging.getLogger("preprocessing_annotations").setLevel(logging.ERROR)
    pipeline_config = PipelineConfig()
    if args.sweep is not None:
        report = evaluate_sweep(RoomGtEvalConfig(), args.run, pipeline_config, args.sweep)
    else:
        report = evaluate_replay(
            RoomGtEvalConfig(), args.run, pipeline_config,
            parse_overrides(pipeline_config.room_filter, args.room_filter),
        )
    emit_report(report, args.out)
