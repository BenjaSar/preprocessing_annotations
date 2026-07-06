"""
T0: Baseline metrics script.

Reads sprint1_verify17 output and writes baseline_metrics.json with:
  - sft_ready_pct, sft_ready_count, total_rooms
  - skipped_count, needs_review_count
  - drop_attribution (geometric, semantic, not_room_scale, oob)
  - bbox_overlap_count (pairwise IoU > 0.1 across processed_annotations/)

Usage:
    python compute_baseline_metrics.py [verify_dir] [--iou-threshold 0.1]
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple


def _iou_xywh(a: List[float], b: List[float]) -> float:
    """Compute IoU of two [x, y, w, h] bboxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _count_bbox_overlaps(processed_dir: Path, iou_threshold: float) -> int:
    """Count pairwise (i, j) room pairs with IoU > iou_threshold across all annotations."""
    count = 0
    for ann_path in sorted(processed_dir.glob("*.json")):
        try:
            with open(ann_path) as f:
                data = json.load(f)
        except Exception:
            continue

        rooms = data.get("rooms", [])
        bboxes = [r["bbox"] for r in rooms if len(r.get("bbox", [])) == 4]

        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                if _iou_xywh(bboxes[i], bboxes[j]) > iou_threshold:
                    count += 1

    return count


def main(verify_dir: Path, iou_threshold: float) -> None:
    metrics: dict = {}

    # --- sft_ready / total_rooms from coverage_report.json ---
    coverage_path = verify_dir / "coverage_report.json"
    if not coverage_path.exists():
        print(f"ERROR: {coverage_path} not found", file=sys.stderr)
        sys.exit(1)
    with open(coverage_path) as f:
        coverage = json.load(f)

    total_images = coverage.get("total_images", 0)
    sft_ready_count = coverage.get("sft_ready_images", 0)
    metrics["sft_ready_count"] = sft_ready_count
    metrics["total_images"] = total_images
    metrics["sft_ready_pct"] = round(sft_ready_count / total_images, 4) if total_images else 0.0
    metrics["total_rooms"] = coverage.get("total_sft_ready_rooms", 0)

    # --- drop_attribution from run_metrics.jsonl ---
    metrics_path = verify_dir / "run_metrics.jsonl"
    drop_attribution: dict = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("stage") == "drop_attribution_totals":
                    drop_attribution = rec.get("drop_attribution", {})
                    break
    metrics["drop_attribution"] = drop_attribution

    # --- skipped_count ---
    skipped_dir = verify_dir / "skipped_pages"
    metrics["skipped_count"] = len(list(skipped_dir.glob("*"))) if skipped_dir.exists() else 0

    # --- needs_review_count ---
    needs_review_path = verify_dir / "needs_review.json"
    if needs_review_path.exists():
        with open(needs_review_path) as f:
            try:
                nr = json.load(f)
                metrics["needs_review_count"] = len(nr) if isinstance(nr, list) else len(nr.get("images", []))
            except Exception:
                metrics["needs_review_count"] = 0
    else:
        metrics["needs_review_count"] = 0

    # --- BBOX_OVERLAP count ---
    processed_dir = verify_dir / "processed_annotations"
    if processed_dir.exists():
        bbox_overlap_count = _count_bbox_overlaps(processed_dir, iou_threshold)
    else:
        bbox_overlap_count = 0
        print(f"WARNING: {processed_dir} not found — bbox_overlap_count set to 0", file=sys.stderr)

    metrics["bbox_overlap_count"] = bbox_overlap_count
    metrics["bbox_overlap_iou_threshold"] = iou_threshold

    if bbox_overlap_count == 0:
        print(
            "WARNING: BBOX_OVERLAP baseline = 0; T3 acceptance criterion unmeasurable "
            "— T3 spike cannot demonstrate gain.",
            file=sys.stderr,
        )

    # --- Write output ---
    out_path = verify_dir / "baseline_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Baseline metrics written to: {out_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute baseline metrics for a verify run.")
    parser.add_argument(
        "verify_dir",
        nargs="?",
        default=str(Path(__file__).parent.parent / "sprint1_verify17"),
        help="Path to verify output directory (default: ../sprint1_verify17)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.1,
        help="IoU threshold for counting overlapping bbox pairs (default: 0.1)",
    )
    args = parser.parse_args()
    main(Path(args.verify_dir), args.iou_threshold)
