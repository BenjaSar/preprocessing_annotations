"""R1 (tech-eval plan): does SAM3 produce room boxes that are actually
correct, or just plausibly-sized -- the question P-B could not answer
(no GT existed on its 2 pages). GT IoU against CubiCasa5K room bboxes.

Reuses the existing PointPromptExpander port and all three adapters
(sam1/sam2.1/sam3) UNCHANGED -- this file is the application layer only,
same shape as bakeoff.py, different metric (paired GT IoU instead of
production's guardrail buckets, which don't transfer: CubiCasa's median
room is 14,274px^2, only 1.4x the 10,000px^2 min_expansion_area_px floor
tuned for ~4500x3375 production pages -- see this session's audit).

Two seed modes, same measurement code (run_model + _summarize), differing
only in how a room gets its seed point/box:

  --seed-mode gt_point (C1, default): GT room bbox CENTROID, box prompt
    DELIBERATELY OMITTED via a degenerate 1x1 label_bbox (forces
    should_use_box_prompt()=False on every call without touching any of
    the three adapters, which each hardcode the same P-2B decision by
    design). Optimistic seed -- the room's true center, not where OCR
    text actually sits. First run of this mode ([:40] slice) was
    STYLE-BIASED: CubiCasa's COCO order puts all 40 "high_quality_
    architectural" images first, none of "colorful"/"high_quality" (270/
    67/63 in the full 400). Fixed here with per-bucket stratification.

  --seed-mode ocr (C2, gated on C1 passing the stratified check): REAL
    OCR Pass-1 label centroids (bakeoff.load_real_seeds -- the same
    extractor production's Pass-1 uses), matched to whichever GT room
    bbox contains the seed's center. Uses the seed's REAL label_bbox
    (not degenerate), so should_use_box_prompt() makes the SAME box-vs-
    point decision production would make. Rooms with no OCR label
    falling inside them are unscored (reported as "unmatched", not
    silently dropped) -- this is real seed yield, not invented.

Both modes: residential-only GT, room-scale ~10x smaller than production
(CubiCasa median 14,274px^2). CubiCasa's images (median 753x627) also
make this a cleaner comparison than P-B on the resolution confound P-B
flagged: SAM3 at imgsz=644 runs near-native here, while SAM1/SAM2.1
(~1024 native) are upscaling rather than downscaling.

Neither mode can license production adoption on its own: gt_point is
optimistic-seed capability ceiling; ocr is realistic-seed but still
residential-only, still room-scale-mismatched to production, and R1's
first (biased) run already showed the regime gap directly -- SAM1
scored 43.5% Acc@0.5 here vs 5.3% applied in real production
(sprint1_verify46), same model. Zero commercial room GT exists anywhere
in this project (standing G14 blocker) -- neither mode touches that.
"""

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch

from preprocessing_annotations.bbox.bbox_metrics import BBox, iou
from preprocessing_annotations.config import CubicasaEvalConfig, SAMConfig

from . import cli
from .bakeoff import load_real_seeds

# eval/ holds cubicasa_gt.py (not an installed package) -- same sys.path
# convention sam3_probe/exemplar_spike.py already uses for this exact problem.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from cubicasa_gt import load_cubicasa_ground_truth  # noqa: E402

_PER_BUCKET_CAP = 40  # matches the FloorPlanCAD validation sample size (tech-eval plan)


def _style_bucket(image_path: Path) -> str:
    """CubiCasa5K's export path is .../<style_bucket>/<per-image-folder>/
    F1_original.png -- the style bucket is 3 components up from the file."""
    parts = str(image_path).split("/")
    return parts[-3] if len(parts) >= 3 else "?"


def _stratified_sample(gt_images: list, per_bucket_cap: int) -> list:
    """Up to `per_bucket_cap` images per style bucket, deterministic (sorted
    path order within each bucket -- no RNG, reproducible across runs).

    Fixes C1: the original R1 run used gt_images[:40], and CubiCasa5K's COCO
    export orders all "high_quality_architectural" images first -- that
    slice was 100% one of three style buckets (270/67/63 in the full 400),
    not a representative sample.
    """
    by_bucket: Dict[str, list] = {}
    for gi in gt_images:
        by_bucket.setdefault(_style_bucket(gi.image_path), []).append(gi)
    sampled = []
    for bucket in sorted(by_bucket):
        images = sorted(by_bucket[bucket], key=lambda g: str(g.image_path))
        sampled.extend(images[:per_bucket_cap])
    return sampled


SeedResolver = Callable[[object, list], Dict[int, Tuple[Tuple[int, int], Tuple[int, int, int, int]]]]


def _gt_point_seed_resolver(gt_image, room_anns) -> Dict[int, Tuple]:
    """C1: GT centroid, point-only (degenerate box forces should_use_box_prompt()=False)."""
    seeds = {}
    for ann in room_anns:
        cx = int((ann.bbox.x1 + ann.bbox.x2) / 2)
        cy = int((ann.bbox.y1 + ann.bbox.y2) / 2)
        seeds[id(ann)] = ((cx, cy), (cx, cy, 1, 1))
    return seeds


def _make_ocr_seed_resolver() -> SeedResolver:
    """C2: real OCR Pass-1 label centroids, matched to the GT room whose
    bbox contains the seed's center. Real label_bbox preserved (not
    degenerate) -- production's own box-vs-point decision applies.

    A room with no OCR label inside it gets no entry (unmatched, reported
    as yield loss, not scored as a failure -- there was no seed to fail).
    Multiple OCR seeds landing in one room: smallest label_bbox wins (most
    likely that room's own label, not a stray larger text block spanning
    into it).

    OCR output is CACHED per image path across the whole run. Two reasons,
    the first being correctness not speed: run_model is called once per
    model arm, so without a cache PaddleOCR re-runs per arm and any
    run-to-run variation would hand the three models DIFFERENT seeds,
    silently breaking the same-seed fairness this comparison depends on.
    (Second reason: OCR here falls back to CPU -- cuDNN unavailable in
    this venv -- at ~4s/image, so caching turns 3x120 calls into 120.)
    """
    ocr_cache: Dict[str, list] = {}

    def resolve(gt_image, room_anns) -> Dict[int, Tuple]:
        key = str(gt_image.image_path)
        if key not in ocr_cache:
            try:
                ocr_cache[key] = load_real_seeds(key)
            except Exception:
                ocr_cache[key] = []
        ocr_seeds = ocr_cache[key]
        seeds = {}
        for ann in room_anns:
            candidates = [
                s for s in ocr_seeds
                if ann.bbox.x1 <= s.center[0] <= ann.bbox.x2
                and ann.bbox.y1 <= s.center[1] <= ann.bbox.y2
            ]
            if candidates:
                best = min(candidates, key=lambda s: s.label_bbox[2] * s.label_bbox[3])
                seeds[id(ann)] = (best.center, best.label_bbox)
        return seeds
    return resolve


def run_model(expander, gt_images, seed_resolver: SeedResolver) -> List[dict]:
    """Returns one record per SCORED room (unmatched rooms under seed_mode=ocr
    are counted separately, not included here) -- bucket/iou/area_ratio,
    so callers can slice by style bucket without re-running inference."""
    records = []
    for gt_image in gt_images:
        bucket = _style_bucket(gt_image.image_path)
        room_anns = [a for a in gt_image.annotations if a.category == "room"]
        seeds = seed_resolver(gt_image, room_anns)
        for ann in room_anns:
            seed = seeds.get(id(ann))
            if seed is None:
                continue
            point, label_bbox = seed
            expansion = expander.expand(str(gt_image.image_path), point, label_bbox)
            px, py, pw, ph = expansion.bbox_xywh
            pred_bbox = BBox(px, py, px + pw, py + ph)
            gt_area = max(1, (ann.bbox.x2 - ann.bbox.x1) * (ann.bbox.y2 - ann.bbox.y1))
            records.append({
                "bucket": bucket,
                "iou": iou(pred_bbox, ann.bbox),
                "area_ratio": max(1, pw * ph) / gt_area,
            })
    return records


def _summarize(records: List[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"n_rooms": 0}
    ious = [r["iou"] for r in records]
    ratios = [r["area_ratio"] for r in records]
    sorted_ious, sorted_ratios = sorted(ious), sorted(ratios)
    return {
        "n_rooms": n,
        "mean_iou": sum(ious) / n,
        "median_iou": sorted_ious[n // 2],
        "accuracy_at_0.5": sum(1 for v in ious if v >= 0.5) / n,
        "accuracy_at_0.7": sum(1 for v in ious if v >= 0.7) / n,
        "mean_area_ratio": sum(ratios) / n,  # 1.0 = right-sized; <1 undershoot; >1 flood
        "median_area_ratio": sorted_ratios[n // 2],
        "undershoot_frac": sum(1 for v in ratios if v < 0.5) / n,
        "flood_frac": sum(1 for v in ratios if v > 2.0) / n,
    }


def _summarize_by_bucket(records: List[dict]) -> Dict[str, dict]:
    buckets = sorted(set(r["bucket"] for r in records))
    return {b: _summarize([r for r in records if r["bucket"] == b]) for b in buckets}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-mode", choices=("gt_point", "ocr"), default="gt_point")
    parser.add_argument("--per-bucket-cap", type=int, default=_PER_BUCKET_CAP)
    args = parser.parse_args()

    sam_config = SAMConfig()
    cc_config = CubicasaEvalConfig()
    all_gt_images = load_cubicasa_ground_truth(cc_config, cc_config.coco_test_path)
    gt_images = _stratified_sample(all_gt_images, args.per_bucket_cap)
    n_gt_rooms = sum(1 for gi in gt_images for a in gi.annotations if a.category == "room")
    bucket_counts = {b: len([g for g in gt_images if _style_bucket(g.image_path) == b])
                      for b in sorted(set(_style_bucket(g.image_path) for g in gt_images))}
    print(f"stratified sample: {len(gt_images)} images {bucket_counts}, {n_gt_rooms} room GT annotations", flush=True)

    seed_resolver = _gt_point_seed_resolver if args.seed_mode == "gt_point" else _make_ocr_seed_resolver()

    adapter_factories = cli._build_adapters(sam_config)
    report: dict = {}
    for model_name, build in adapter_factories.items():
        expander = build()
        records = run_model(expander, gt_images, seed_resolver)
        n_scored = len(records)
        result = {
            "overall": _summarize(records),
            "by_bucket": _summarize_by_bucket(records),
        }
        if args.seed_mode == "ocr":
            result["seed_yield"] = {"n_gt_rooms": n_gt_rooms, "n_scored": n_scored,
                                     "match_rate": n_scored / n_gt_rooms if n_gt_rooms else 0.0}
        report[model_name] = result
        print(f"{model_name}: overall={result['overall']}", flush=True)
        for b, s in result["by_bucket"].items():
            print(f"  {b}: {s}", flush=True)
        del expander
        gc.collect()
        torch.cuda.empty_cache()

    out_dir = Path(__file__).resolve().parents[3] / "sam21_probe_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"gt_bakeoff_{args.seed_mode}_" + "_".join(sorted(adapter_factories))
    out_path = out_dir / f"{stem}.json"
    suffix = 0
    while out_path.exists():
        suffix += 1
        out_path = out_dir / f"{stem}.{suffix}.json"
    out_path.write_text(json.dumps(
        {
            "_provenance": {
                "measured": "2026-08-11",
                "dataset": "CubiCasa5K COCO test split, room category, bbox IoU (no mask GT exists)",
                "seed_mode": args.seed_mode,
                "per_bucket_cap": args.per_bucket_cap,
                "bucket_counts": bucket_counts,
                "n_gt_rooms": n_gt_rooms,
                "seed": (
                    "GT room bbox centroid, point-only (degenerate 1x1 label_bbox forces "
                    "should_use_box_prompt()=False)" if args.seed_mode == "gt_point" else
                    "real OCR Pass-1 label centroid matched to containing GT room bbox, "
                    "real label_bbox preserved (production's own box-vs-point decision applies)"
                ),
            },
            "results": report,
        },
        indent=2,
    ))
    print(f"written: {out_path}", flush=True)


if __name__ == "__main__":
    main()
