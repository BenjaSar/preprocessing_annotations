"""Replay OCR candidate generation from a run's saved raw OCR text under
alternative room-name patterns, then the SFT filter stage, scored vs room GT.

Needs a run whose raw annotations carry `ocrTextDetections`. Candidate
generation (MEPTextExtractor.find_room_candidates) is a pure function of those
detections and OCRConfig, so the baseline variant reproduces the run's own
Pass-1 candidates exactly (checked per run: baseline == as_run).

Per page, the run's `ocr_rooms` split into Pass-1 candidates (those the
baseline replay regenerates) and the rest (Pass-2 VLM fallback candidates).
A variant swaps only the Pass-1 part; the rest, and the VLM `rooms`, stay as
the run recorded them. Not recomputed: VLM-room filtering/reconciliation
against OCR candidates (baked into `rooms` at run time) and residential
abbreviation recovery.

Guard: if the run's config snapshot records ocr.room_name_patterns they must
equal the code's -- exact and authoritative. Only when a run predates that
field does the baseline instead have to regenerate a subset of the candidates
the run recorded; that fallback catches patterns added since the run but not
removed ones, and it cannot distinguish a pattern change from a candidate the
run's own post-processing (VLM refine, abbreviation recovery) later dropped.
Either way a mismatch refuses the replay rather than silently skewing it.

A variant adds regexes to and/or removes existing ones from
OCRConfig.room_name_patterns (removal = "what did these patterns contribute"
on a run made with them), and/or applies crop re-reads from a
crop_reread_probe.py file (--rereads): the fragments its selector picks get
the VLM transcription when the consistency guard accepts it. The probe must
have selected at least what the variant selects; otherwise the replay refuses.

--run-lacks names patterns the code has but the run predates. The run's own
patterns (code minus those) then split its ocr_rooms and feed the guard,
while the baseline variant builds candidates with the code's: replaying an
older run on current code, without mistaking a candidate the new patterns
merge differently for a Pass-2 one. Per variant: GT-page localization and per-page
room-count changes on every page (the blast radius on corpora without GT). The choice among variants is judged
leave-one-document-out, as in room_replay --sweep.
"""

import argparse
import dataclasses
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from preprocessing_annotations.config import CropRereadConfig, PipelineConfig, RoomGtEvalConfig
from preprocessing_annotations.ingestion.crop_reread import Reread, apply_rereads
from preprocessing_annotations.ingestion.ocr_extractor import (
    MEPTextExtractor,
    RoomCandidate,
    TextDetection,
)

from crop_reread_records import load_rereads, reread_selector
from held_out import leave_one_document_out
from report_io import emit_report
from room_gt_eval import BoxXYXY, load_gt_polygons, page_key
from room_replay import (
    parse_overrides,
    regressed_documents,
    replay_annotation,
    room_count_changes,
    score_replay,
)

BASELINE = "baseline"


def load_text_detections(annotation: Mapping[str, Any]) -> List[TextDetection]:
    if "ocrTextDetections" not in annotation:
        raise ValueError(
            f"{annotation.get('image_file')}: no ocrTextDetections -- the run predates raw-OCR persistence"
        )
    return [
        TextDetection(bbox=d["bbox"], text=d["text"], confidence=d["confidence"])
        for d in annotation["ocrTextDetections"]
    ]


def candidate_dicts(candidates: Sequence[RoomCandidate]) -> List[Dict[str, Any]]:
    """RoomCandidates in the shape pipeline._save_annotation writes to ocr_rooms."""
    return [
        {
            "room_number": c.room_number,
            "room_name": c.room_name,
            "name_expanded": c.name_expanded,
            "bbox": list(c.bbox),
            "confidence": c.confidence,
        }
        for c in candidates
    ]


def _key(room: Mapping[str, Any]) -> Tuple[str, Tuple[Any, ...]]:
    return room["room_name"], tuple(room["bbox"])


def with_variant_candidates(
    annotation: Dict[str, Any], baseline: MEPTextExtractor, variant: MEPTextExtractor,
    verify_against_recorded: bool = True,
    variant_detections: Optional[Sequence[TextDetection]] = None,
) -> Dict[str, Any]:
    """Copy of the annotation whose Pass-1 ocr_rooms come from `variant`, built
    from `variant_detections` (default: the run's recorded OCR detections)."""
    detections = load_text_detections(annotation)
    baseline_keys = {_key(c) for c in candidate_dicts(baseline.find_room_candidates(detections))}
    recorded_keys = {_key(room) for room in annotation.get("ocr_rooms", [])}
    if verify_against_recorded and not baseline_keys <= recorded_keys:
        raise ValueError(
            f"{annotation.get('image_file')}: baseline patterns generate candidates the run "
            f"did not record -- the code's OCR patterns differ from the run's"
        )
    pass2 = [room for room in annotation.get("ocr_rooms", []) if _key(room) not in baseline_keys]
    rebuilt = dict(annotation)
    built_from = detections if variant_detections is None else variant_detections
    rebuilt["ocr_rooms"] = candidate_dicts(variant.find_room_candidates(built_from)) + pass2
    return rebuilt


@dataclasses.dataclass(frozen=True)
class PatternVariant:
    """Room-name patterns added to, and removed from, OCRConfig.room_name_patterns;
    `reread`: crop re-reads applied under these selector bounds and guard."""

    added: Tuple[str, ...] = ()
    removed: Tuple[str, ...] = ()
    reread: Optional[CropRereadConfig] = None


def reread_detections(
    detections: Sequence[TextDetection], probed: Sequence[Reread],
    builder: MEPTextExtractor, config: CropRereadConfig,
) -> List[TextDetection]:
    """`detections` with the probe's re-reads applied to what `config`'s
    selector picks; refuses when the probe did not read one of them."""
    by_index = {r.index: r for r in probed}
    selected = reread_selector(builder, config).select(detections)
    missing = [i for i in selected if i not in by_index]
    if missing:
        raise ValueError(
            f"selector picks {len(missing)} detections the probe did not read -- "
            f"probe with bounds at least as broad as the variant's"
        )
    return apply_rereads(detections, [by_index[i] for i in selected], config.consistency_min)


def extractor_with_patterns(
    pipeline_config: PipelineConfig, variant: PatternVariant,
) -> MEPTextExtractor:
    base = list(pipeline_config.ocr.room_name_patterns)
    unknown = [p for p in variant.removed if p not in base]
    if unknown:
        raise ValueError(f"cannot remove patterns not in OCRConfig.room_name_patterns: {unknown}")
    patterns = [p for p in base if p not in variant.removed] + list(variant.added)
    return MEPTextExtractor(dataclasses.replace(pipeline_config.ocr, room_name_patterns=patterns))


def check_recorded_patterns(
    run_dir: Path, pipeline_config: PipelineConfig, run_lacks: Sequence[str] = (),
) -> bool:
    """Compare the run's recorded ocr.room_name_patterns with the code's.

    Returns True when the run records them and they match: an exact,
    authoritative check (runs from 2026-10-13 on). Returns False when the run
    predates that snapshot field, leaving the caller to fall back on the
    per-page subset guard, which catches patterns added since the run but not
    removed ones. Raises when the recorded patterns differ."""
    snapshot = run_dir / "pipeline_config.json"
    if not snapshot.exists():
        return False
    recorded = json.loads(snapshot.read_text()).get("ocr", {}).get("room_name_patterns")
    if recorded is None:
        return False
    expected = [p for p in pipeline_config.ocr.room_name_patterns if p not in run_lacks]
    if recorded != expected:
        raise ValueError(
            f"{run_dir}: the run's recorded room_name_patterns differ from the code's"
        )
    return True


def replay_variant(
    run_dir: Path, pipeline_config: PipelineConfig, variant: PatternVariant,
    verify_against_recorded: bool = True,
    rereads: Mapping[str, Sequence[Reread]] = {},
    run_lacks: Sequence[str] = (),
) -> Dict[str, List[BoxXYXY]]:
    # Splits the run's ocr_rooms: built with the run's own patterns.
    baseline = extractor_with_patterns(pipeline_config, PatternVariant(removed=tuple(run_lacks)))
    candidate_builder = extractor_with_patterns(pipeline_config, variant)
    rooms: Dict[str, List[BoxXYXY]] = {}
    for path in sorted((run_dir / "annotations").glob("*.json")):
        recorded = json.loads(path.read_text())
        variant_detections = None
        if variant.reread is not None:
            variant_detections = reread_detections(
                load_text_detections(recorded), rereads.get(path.stem, ()),
                candidate_builder, variant.reread,
            )
        annotation = with_variant_candidates(
            recorded, baseline, candidate_builder,
            verify_against_recorded=verify_against_recorded,
            variant_detections=variant_detections,
        )
        rooms[page_key(path.name)], _ = replay_annotation(
            annotation, run_dir / "images", pipeline_config, pipeline_config.room_filter
        )
    return rooms


def evaluate_variants(
    eval_config: RoomGtEvalConfig, run_dir: Path, pipeline_config: PipelineConfig,
    variants: Mapping[str, PatternVariant],
    rereads: Mapping[str, Sequence[Reread]] = {},
    run_lacks: Sequence[str] = (),
) -> Dict[str, Any]:
    patterns_verified = check_recorded_patterns(run_dir, pipeline_config, run_lacks)
    gt = load_gt_polygons(eval_config.gt_coco_path)
    candidates = {BASELINE: PatternVariant(), **variants}
    rooms_by_variant = {
        name: replay_variant(
            run_dir, pipeline_config, variant,
            verify_against_recorded=not patterns_verified,
            rereads=rereads,
            run_lacks=run_lacks,
        )
        for name, variant in candidates.items()
    }
    scores: Dict[str, Dict[str, Any]] = {}
    report: Dict[str, Any] = {}
    for name, rooms in rooms_by_variant.items():
        overall, per_document = score_replay(
            {page: boxes for page, boxes in rooms.items() if page in gt},
            gt, eval_config.gt_iou_threshold,
        )
        scores[name] = per_document
        report[name] = {
            "added_name_patterns": list(candidates[name].added),
            "removed_name_patterns": list(candidates[name].removed),
            "reread": (
                dataclasses.asdict(candidates[name].reread)
                if candidates[name].reread is not None else None
            ),
            "localization": overall.to_dict(),
            "documents_with_recall_drop": regressed_documents(scores[BASELINE], per_document),
            "room_count_changes": room_count_changes(rooms_by_variant[BASELINE], rooms),
        }
    return {
        "run_dir": str(run_dir),
        "run_lacks_patterns": list(run_lacks),
        "pages_replayed": len(rooms_by_variant[BASELINE]),
        "pages_scored": sum(page in gt for page in rooms_by_variant[BASELINE]),
        "candidates": report,
        "leave_one_document_out": leave_one_document_out(scores, BASELINE).to_dict(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay OCR candidate generation under changed room-name patterns")
    parser.add_argument("--run", type=Path, required=True, help="Run dir whose raw annotations carry ocrTextDetections")
    parser.add_argument(
        "--variant", nargs=2, action="append", default=[], metavar=("NAME", "REGEX"),
        help="Room-name regex ADDED in variant NAME; repeat to add several",
    )
    parser.add_argument(
        "--without", nargs=2, action="append", default=[], metavar=("NAME", "REGEX"),
        help="Existing room-name regex REMOVED in variant NAME; repeat to remove several",
    )
    parser.add_argument(
        "--rereads", type=Path, default=None, help="crop_reread_probe.py output for --reread variants",
    )
    parser.add_argument(
        "--reread", nargs=2, action="append", default=[], metavar=("NAME", "KEY=VALUE,..."),
        help="Variant NAME applies the re-reads under CropRereadConfig overrides ('-' = defaults)",
    )
    parser.add_argument(
        "--run-lacks", action="append", default=[], metavar="REGEX",
        help="Code room-name regex the run predates; repeat for several",
    )
    parser.add_argument("--out", type=Path, default=None, help="Also write report JSON here")
    args = parser.parse_args()
    if args.reread and args.rereads is None:
        parser.error("--reread needs --rereads")

    logging.getLogger("preprocessing_annotations").setLevel(logging.ERROR)
    added: Dict[str, List[str]] = {}
    removed: Dict[str, List[str]] = {}
    for name, pattern in args.variant:
        added.setdefault(name, []).append(pattern)
    for name, pattern in args.without:
        removed.setdefault(name, []).append(pattern)
    reread: Dict[str, CropRereadConfig] = {
        name: parse_overrides(CropRereadConfig(), [] if raw == "-" else raw.split(","))
        for name, raw in args.reread
    }
    variants = {
        name: PatternVariant(
            tuple(added.get(name, ())), tuple(removed.get(name, ())), reread.get(name),
        )
        for name in {**added, **removed, **reread}
    }
    emit_report(
        evaluate_variants(
            RoomGtEvalConfig(), args.run, PipelineConfig(), variants,
            load_rereads(args.rereads) if args.rereads else {},
            args.run_lacks,
        ),
        args.out,
    )
