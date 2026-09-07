"""
Export modules for annotation formats and review prioritization.

This module provides exporters for Label Studio format and utilities
for prioritizing images that need human review.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:
    from ..config import ExportConfig
except ImportError:
    from config import ExportConfig

logger = logging.getLogger(__name__)


class ExportError(Exception):
    """Raised when export operations fail."""

    pass


@dataclass
class ReviewItem:
    """
    An image flagged for human review.

    Attributes:
        filename: Name of the annotation file.
        confidence: Computed confidence score.
        reason: Reason for flagging.
    """

    filename: str
    confidence: float
    reason: str


class LabelStudioExporter:
    """
    Exports annotations to Label Studio format with pre-labels.

    Converts VLM/OCR annotations to Label Studio's JSON format,
    allowing import as pre-labeled data for human review.

    Attributes:
        config: ExportConfig with export parameters.

    Example:
        exporter = LabelStudioExporter()
        exporter.export(
            annotations_dir="./annotations",
            images_dir="./images",
            output_file="label_studio_import.json"
        )
    """

    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig()

    def export(
        self,
        annotations_dir: str | Path,
        images_dir: str | Path,
        output_file: str | Path,
        image_url_prefix: str = "/data/local-files/?d=",
    ) -> int:
        """
        Export annotations to Label Studio format.

        Args:
            annotations_dir: Directory containing JSON annotation files.
            images_dir: Directory containing source images.
            output_file: Output JSON file path.
            image_url_prefix: URL prefix for image paths in Label Studio.

        Returns:
            Number of tasks exported.

        Raises:
            ExportError: If export fails.
        """
        annotations_dir = Path(annotations_dir)
        images_dir = Path(images_dir)
        output_file = Path(output_file)

        if not annotations_dir.exists():
            raise ExportError(f"Annotations directory not found: {annotations_dir}")

        tasks = []
        json_files = sorted(annotations_dir.glob("*.json"))

        if not json_files:
            logger.warning(f"No annotation files found in {annotations_dir}")
            return 0

        for json_file in json_files:
            try:
                with open(json_file) as f:
                    ann = json.load(f)

                task = self._create_task(ann, images_dir, image_url_prefix)
                if task:
                    tasks.append(task)

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in {json_file}: {e}")
            except Exception as e:
                logger.error(f"Failed to process {json_file}: {e}")

        # Write output
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(tasks, f, indent=2)

        logger.info(f"Exported {len(tasks)} tasks to {output_file}")
        return len(tasks)

    def _create_task(
        self, annotation: dict, images_dir: Path, image_url_prefix: str
    ) -> Optional[dict]:
        """Create a single Label Studio task from annotation."""
        image_file = annotation.get("image_file")
        if not image_file:
            return None

        # Hard gate 1: only export SFT-ready annotations.
        # sft_ready is set by prepare_sft_annotation(); False means zero valid
        # rooms remained after semantic filtering.  Exporting them would
        # populate Label Studio with empty or noise-only tasks.
        if not annotation.get("sft_ready", True):
            logger.warning(f"Skipping {image_file}: sft_ready=False")
            return None

        # Hard gate 2: reject unprocessed annotations (still contain panels or raw OCR)
        if annotation.get("panels"):
            logger.warning(
                f"Skipping {image_file}: Contains {len(annotation.get('panels', []))} panels "
                "(run through post-processing pipeline before export)"
            )
            return None

        if "ocr_rooms" in annotation:
            logger.warning(
                f"Skipping {image_file}: Contains raw ocr_rooms key "
                "(must pass through SFT validation first)"
            )
            return None

        # Get image dimensions
        img_w = annotation.get("image_size", {}).get(
            "width", self.config.default_width
        )
        img_h = annotation.get("image_size", {}).get(
            "height", self.config.default_height
        )

        # Build task structure
        # CRITICAL FIX (pipeline-output-analysis §Category C):
        # Label Studio expects room annotations in the "annotations" key,
        # not "predictions".  Using "predictions" causes Label Studio to
        # show 0 annotations and display them only as pre-annotations,
        # breaking the human review workflow.
        task = {
            "data": {"image": f"{image_url_prefix}{images_dir}/{image_file}"},
            "annotations": [{"result": []}],
        }

        result = task["annotations"][0]["result"]

        # Add room annotations
        for i, room in enumerate(annotation.get("rooms", [])):
            bbox = room.get("bbox", [0, 0, 100, 100])

            # Convert to Label Studio percentage format
            x_pct = (bbox[0] / img_w) * 100
            y_pct = (bbox[1] / img_h) * 100
            w_pct = (bbox[2] / img_w) * 100
            h_pct = (bbox[3] / img_h) * 100

            # CRITICAL FIX (pipeline-output-analysis §Category C):
            # Drop annotations whose percentage coordinates exceed 100%.
            # These are produced when the VLM returns pixel coordinates
            # relative to full-resolution images while the pipeline
            # expects fractional coordinates.  Without this guard, Label
            # Studio renders them off-canvas, invisible to reviewers.
            if x_pct > 100 or y_pct > 100 or x_pct + w_pct > 100 or y_pct + h_pct > 100:
                room_name = room.get("room_name") or room.get("name", "unknown")
                logger.warning(
                    f"LabelStudioExporter: dropping out-of-bounds annotation "
                    f"'{room_name}' (x={x_pct:.1f}%, y={y_pct:.1f}%, "
                    f"w={w_pct:.1f}%, h={h_pct:.1f}%) in {image_file}"
                )
                continue

            result.append(
                {
                    "id": f"room_{i}",
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image",
                    "original_width": img_w,
                    "original_height": img_h,
                    "value": {
                        "x": x_pct,
                        "y": y_pct,
                        "width": w_pct,
                        "height": h_pct,
                        # BUG-2 fix: use same type→category→"other" resolution
                        # chain as COCOExporter to guarantee identical category
                        # assignment across both export formats.
                        "rectanglelabels": [
                            room.get("type") or room.get("category") or "other"
                        ],
                    },
                }
            )

            # Add text annotation for room label
            # CRITICAL FIX: Prioritize semantic room_name over numeric room_number
            # Before: room_number → number always wins if present
            # After: room_name → semantic label preferred, number is fallback
            room_label = room.get("room_name", "") or room.get("room_number", "")
            if room_label:
                result.append(
                    {
                        "id": f"room_{i}_text",
                        "type": "textarea",
                        "from_name": "room_label",
                        "to_name": "image",
                        "value": {
                            "x": x_pct,
                            "y": y_pct,
                            "width": w_pct,
                            "height": h_pct,
                            "text": [room_label],
                        },
                    }
                )

        # CRITICAL FIX: Removed panel export section
        # Hard constraint: Equipment, panels, and symbols MUST NOT appear in final annotations
        # Only rooms/spaces should be exported for SFT dataset
        # Panels are electrical equipment, not spatial rooms - they contaminate VLM training

        return task


class ReviewPrioritizer:
    """
    Prioritizes annotations for human review based on confidence.

    Analyzes annotation quality signals to identify images that
    need human verification.

    Attributes:
        config: ExportConfig with review threshold.

    Example:
        prioritizer = ReviewPrioritizer()
        needs_review = prioritizer.prioritize("./annotations")
    """

    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig()

    def prioritize(
        self, annotations_dir: str | Path, top_percent: Optional[float] = None
    ) -> List[ReviewItem]:
        """
        Select lowest-confidence annotations for human review.

        Args:
            annotations_dir: Directory containing JSON annotation files.
            top_percent: Percentage of images to flag (overrides config).

        Returns:
            List of ReviewItem objects, sorted by confidence (lowest first).
        """
        annotations_dir = Path(annotations_dir)
        top_percent = top_percent or self.config.review_threshold

        if not annotations_dir.exists():
            logger.warning(f"Annotations directory not found: {annotations_dir}")
            return []

        items = []
        json_files = list(annotations_dir.glob("*.json"))

        for json_file in json_files:
            try:
                with open(json_file) as f:
                    ann = json.load(f)

                confidence, reason = self._compute_confidence(ann)
                items.append(
                    ReviewItem(
                        filename=json_file.name, confidence=confidence, reason=reason
                    )
                )

            except Exception as e:
                # Files that fail to load should definitely be reviewed
                items.append(
                    ReviewItem(
                        filename=json_file.name,
                        confidence=0.0,
                        reason=f"Failed to load: {e}",
                    )
                )

        # Sort by confidence (lowest first)
        items.sort(key=lambda x: x.confidence)

        # Select top percentage
        n_review = max(1, int(len(items) * top_percent))
        to_review = items[:n_review]

        logger.info(
            f"Flagged {len(to_review)}/{len(items)} annotations for review "
            f"({top_percent*100:.0f}%)"
        )

        return to_review

    def _compute_confidence(self, annotation: dict) -> tuple[float, str]:
        """
        Compute a multi-factor confidence score for review prioritisation.

        Replaces the naive base(0.5)+has_rooms(+0.2)+completeness(+0.3) formula
        which ignored OCR confidence, room density, and SFT readiness flags.

        Factors:
          - sft_ready=False           → 0.10 immediately
          - VLM/OCR error             → 0.00 immediately
          - No rooms                  → 0.20
          - Average OCR confidence    → 40% weight
          - Room density vs image size→ 30% weight
          - Name+number completeness  → 30% weight
        """
        if annotation.get("error"):
            return 0.0, f"VLM error: {annotation['error']}"

        if not annotation.get("sft_ready", True):
            return 0.10, "sft_ready=False"

        rooms = annotation.get("rooms", [])
        if not rooms:
            return 0.20, "No rooms detected"

        # Average OCR confidence (VLM rooms without confidence field assumed 0.92)
        confidences = [float(r.get("confidence", 0.92)) for r in rooms]
        avg_conf = sum(confidences) / len(confidences)

        # Room density: expect ≥0.5 rooms per megapixel
        img = annotation.get("image_size", {})
        mp = (img.get("width", 1000) * img.get("height", 1000)) / 1_000_000
        expected = max(1.0, mp * 0.5)
        density = min(1.0, len(rooms) / expected)

        # Completeness: rooms with both room_name and room_number
        complete = sum(
            1 for r in rooms
            if (r.get("room_name") or r.get("name")) and r.get("room_number")
        )
        completeness = complete / len(rooms)

        score = 0.40 * avg_conf + 0.30 * density + 0.30 * completeness
        reason = (
            f"{len(rooms)} rooms | avg_conf={avg_conf:.2f} | "
            f"density={density:.2f} | complete={complete}/{len(rooms)}"
        )
        return round(min(1.0, score), 4), reason

    def save_review_list(
        self, items: List[ReviewItem], output_file: str | Path
    ) -> None:
        """
        Save review list to a file.

        Args:
            items: List of ReviewItem objects.
            output_file: Output file path (.txt or .json).
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if output_file.suffix == ".json":
            data = [
                {
                    "filename": item.filename,
                    "confidence": item.confidence,
                    "reason": item.reason,
                }
                for item in items
            ]
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
        else:
            # Text format
            with open(output_file, "w") as f:
                f.write("# Files flagged for review (lowest confidence first)\n")
                f.write("# Format: filename | confidence | reason\n\n")
                for item in items:
                    f.write(f"{item.filename} | {item.confidence:.2f} | {item.reason}\n")

        logger.info(f"Saved review list to {output_file}")

    def print_summary(self, items: List[ReviewItem], limit: int = 10) -> None:
        """Print a summary of items needing review."""
        print(f"\n{'='*60}")
        print(f"REVIEW PRIORITY (showing top {min(limit, len(items))})")
        print(f"{'='*60}")

        for item in items[:limit]:
            print(f"  {item.filename}")
            print(f"    Confidence: {item.confidence:.2f}")
            print(f"    Reason: {item.reason}")
            print()

        if len(items) > limit:
            print(f"  ... and {len(items) - limit} more")


# ---------------------------------------------------------------------------
# COCO JSON Exporter
# ---------------------------------------------------------------------------

def _flatten_polygon(points) -> Optional[List[float]]:
    """Flatten a [[x, y], ...] polygon to COCO's [x1, y1, x2, y2, ...] form.

    Returns None when `points` is empty or not a usable polygon (< 3 vertices),
    which is the caller's signal to fall back to a bbox rectangle.
    """
    if not points:
        return None
    try:
        flat = [float(v) for point in points for v in point]
    except (TypeError, ValueError):
        return None
    if len(flat) < 6 or len(flat) % 2 != 0:  # < 3 vertices
        return None
    return flat


def _polygon_area(flat: List[float]) -> float:
    """Shoelace area of a flat [x1, y1, x2, y2, ...] polygon.

    Plain-Python on purpose: this module serializes annotations and imports
    nothing heavier than json/pathlib. cv2.contourArea would be equivalent but
    would pull OpenCV into the export layer for one arithmetic loop.
    """
    n = len(flat) // 2
    total = 0.0
    for i in range(n):
        x1, y1 = flat[2 * i], flat[2 * i + 1]
        j = (i + 1) % n
        x2, y2 = flat[2 * j], flat[2 * j + 1]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _polygon_bounds(flat: List[float]) -> List[float]:
    """Return [x, y, w, h] of a flat polygon's axis-aligned bounding box."""
    xs = flat[0::2]
    ys = flat[1::2]
    x, y = min(xs), min(ys)
    return [x, y, max(xs) - x, max(ys) - y]


class COCOExporter:
    """
    Export SFT-ready annotations in COCO object detection format.

    COCO format is the native input for most VLM fine-tuning frameworks
    (LLaVA, InternVL, PaliGemma, Qwen-VL, Idefics2).  Without this exporter
    the pipeline required a manual conversion step before every training run.

    Bbox convention: COCO uses [x, y, width, height] natively — matching the
    pipeline's internal storage format, so no conversion is needed.
    """

    def __init__(self, description: str = "MEP Floor Plan Room Annotations"):
        self.description = description

    def export(
        self,
        processed_dir: Path | str,
        images_dir: Path | str,
        output_file: Path | str,
        sft_only: bool = True,
        master_categories: list = None,
        min_annotations_per_image: int = 3,
    ) -> int:
        """
        Export all SFT-ready annotations to a single COCO JSON file.

        Args:
            processed_dir: Directory containing post-processed JSON annotations.
            images_dir:    Directory containing the source PNG images.
            output_file:   Output path for the COCO JSON file.
            sft_only:      If True (default), skip annotations with sft_ready=False.
            master_categories: If provided, use this global category list for all splits.
            min_annotations_per_image: Minimum rooms per image to include (Fix 8).

        Returns:
            Number of images exported.
        """
        processed_dir = Path(processed_dir)
        images_dir = Path(images_dir)
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Fix 6: Track all exclusion reasons for logging
        exclusion_log = []

        # Remediation Fix #3: Load post_processing_summary to check for unresolved overlaps
        summary_file = processed_dir.parent / "post_processing_summary.json"
        bbox_overlap_issues = {}
        if summary_file.exists():
            try:
                with open(summary_file) as f:
                    summary = json.load(f)
                for issue in summary.get("issue_details", []):
                    if issue.get("issue_type") == "ValidationIssue" and "BBOX_OVERLAP" in str(issue.get("description", "")):
                        image_name = issue.get("image", "")
                        bbox_overlap_issues[image_name] = issue
            except Exception as e:
                logger.debug(f"COCOExporter: could not load post_processing_summary: {e}")

        # ── Phase 1: Collect all rooms to determine which categories are used ──
        all_file_data = []
        used_types: set = set()

        for json_file in sorted(processed_dir.glob("*.json")):
            try:
                with open(json_file) as f:
                    ann = json.load(f)
            except Exception as e:
                logger.error(f"COCOExporter: failed to read {json_file}: {e}")
                continue

            if sft_only and not ann.get("sft_ready", False):
                exclusion_log.append({
                    "file": json_file.name,
                    "reason": "sft_ready=False",
                    "rooms_dropped": len(ann.get("rooms", [])),
                })
                continue

            # Remediation Fix #3: Skip images with unresolved BBOX_OVERLAP issues
            if json_file.name in bbox_overlap_issues:
                exclusion_log.append({
                    "file": json_file.name,
                    "reason": "unresolved BBOX_OVERLAP (geometry quality gate)",
                    "rooms_dropped": len(ann.get("rooms", [])),
                })
                logger.warning(
                    f"COCOExporter: skipping {json_file.name} due to unresolved bbox overlaps "
                    f"(Remediation Fix #3 quality gate)"
                )
                continue

            image_file = ann.get("image_file", json_file.stem + ".png")
            img_size = ann.get("image_size", {})
            w = img_size.get("width", 0)
            h = img_size.get("height", 0)

            # Pre-filter rooms and collect used types
            valid_rooms = []
            for room in ann.get("rooms", []):
                bbox = room.get("bbox", [])
                rname = room.get("room_name") or room.get("name", "?")
                if len(bbox) != 4:
                    exclusion_log.append({
                        "file": json_file.name, "room": rname,
                        "reason": f"invalid bbox (len={len(bbox)})",
                    })
                    continue
                bx, by, bw, bh = [float(v) for v in bbox]
                if bw <= 0 or bh <= 0:
                    exclusion_log.append({
                        "file": json_file.name, "room": rname,
                        "reason": f"zero/negative dimension ({bw}×{bh})",
                    })
                    continue
                if w > 0 and h > 0:
                    if bx < 0 or by < 0 or bx + bw > w or by + bh > h:
                        exclusion_log.append({
                            "file": json_file.name, "room": rname,
                            "reason": f"OOB bbox [{bx:.0f},{by:.0f},{bw:.0f},{bh:.0f}] "
                                      f"in {w}×{h} image",
                        })
                        logger.warning(
                            f"COCOExporter: dropping out-of-bounds bbox "
                            f"[{bx:.0f},{by:.0f},{bw:.0f},{bh:.0f}] "
                            f"for '{rname}' (image {w}×{h})"
                        )
                        continue
                raw_type = room.get("type") or room.get("category") or "other"

                # Fix 5: Reject "other" category annotations (checklist #14)
                if raw_type == "other":
                    exclusion_log.append({
                        "file": json_file.name, "room": rname,
                        "reason": f"category='other' (unmapped)",
                    })
                    logger.warning(
                        f"COCOExporter: dropping '{rname}' with category='other'"
                    )
                    continue

                used_types.add(raw_type)
                valid_rooms.append(room)

            # Fix 8: Minimum annotation count per image
            if len(valid_rooms) < min_annotations_per_image:
                exclusion_log.append({
                    "file": json_file.name,
                    "reason": f"below min annotations ({len(valid_rooms)} < {min_annotations_per_image})",
                    "rooms_dropped": len(valid_rooms),
                })
                logger.warning(
                    f"COCOExporter: excluding {image_file} "
                    f"({len(valid_rooms)} annotations < {min_annotations_per_image} minimum)"
                )
                continue

            if valid_rooms:
                all_file_data.append({
                    "image_file": image_file,
                    "w": w,
                    "h": h,
                    "rooms": valid_rooms,
                    "source_file": json_file.name,
                })

        # Fix 6: Log all exclusions
        if exclusion_log:
            logger.info(
                f"COCOExporter: {len(exclusion_log)} exclusion(s) logged"
            )
            for ex in exclusion_log:
                logger.debug(f"  Excluded: {ex}")

        # ── Phase 2: Build COCO structure ──
        # Fix 7: Use master categories if provided (global across splits)
        if master_categories:
            cat_id_map = {t: i + 1 for i, t in enumerate(master_categories)}
            coco_categories = [
                {"id": cat_id_map[t], "name": t, "supercategory": "room"}
                for t in master_categories
            ]
        else:
            sorted_used = sorted(used_types)
            cat_id_map = {t: i + 1 for i, t in enumerate(sorted_used)}
            coco_categories = [
                {"id": cat_id_map[t], "name": t, "supercategory": "room"}
                for t in sorted_used
            ]

        coco: dict = {
            "info": {
                "description": self.description,
                "version": "1.0",
            },
            "categories": coco_categories,
            "images": [],
            "annotations": [],
        }

        ann_id = 1
        img_id = 0

        for file_data in all_file_data:
            img_id += 1
            coco["images"].append({
                "id": img_id,
                "file_name": file_data["image_file"],
                "width": file_data["w"],
                "height": file_data["h"],
            })

            for room in file_data["rooms"]:
                bbox = room.get("bbox", [])
                bx, by, bw, bh = [float(v) for v in bbox]

                raw_type = room.get("type") or room.get("category") or "other"
                cat_id = cat_id_map.get(raw_type, cat_id_map.get("other", 1))

                # Prefer the real room outline when SAM produced one
                # (room["segmentation"], written by
                # sam_segmenter.refine_annotations as page-pixel [[x, y], ...]).
                # Falls back to a bbox rectangle for rooms that took a guardrail
                # path (over_segmentation / flood / collapse / error), the
                # cc_split path, or any run with use_sam off -- so output shape
                # is unchanged whenever no polygon exists.
                polygon = _flatten_polygon(room.get("segmentation"))
                if polygon is not None and _polygon_area(polygon) > 0:
                    segmentation = [polygon]
                    area = _polygon_area(polygon)
                else:
                    segmentation = [[
                        bx,      by,
                        bx + bw, by,
                        bx + bw, by + bh,
                        bx,      by + bh,
                    ]]
                    area = bw * bh

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "segmentation": segmentation,
                    "bbox": [bx, by, bw, bh],
                    "area": area,
                    "iscrowd": 0,
                    "attributes": {
                        "room_name": room.get("room_name") or room.get("name", ""),
                        "room_number": room.get("room_number", ""),
                        "confidence": room.get("confidence", 1.0),
                    },
                })
                ann_id += 1

        # Fix 5: Post-export validation — verify bbox integrity
        # Build source lookup: image_file → list of (room_name, bbox)
        source_by_image = {}
        for file_data in all_file_data:
            img_file = file_data["image_file"]
            source_by_image[img_file] = []
            for room in file_data["rooms"]:
                rname = room.get("room_name") or room.get("name", "?")
                src_bbox = [float(v) for v in room.get("bbox", [])]
                source_by_image[img_file].append((rname, src_bbox))

        # Build image_id → file_name map from COCO
        img_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

        integrity_ok = True
        for coco_ann in coco["annotations"]:
            coco_bbox = coco_ann["bbox"]
            rname = coco_ann["attributes"]["room_name"]
            img_file = img_id_to_file.get(coco_ann["image_id"], "")

            # Find matching source bbox for this image + room name
            # Handle duplicate names by checking bbox proximity
            src_rooms = source_by_image.get(img_file, [])
            matched = False
            for src_name, src_bbox in src_rooms:
                if src_name == rname and len(src_bbox) == 4:
                    if all(abs(coco_bbox[i] - src_bbox[i]) <= 1.0 for i in range(4)):
                        matched = True
                        break
            if not matched:
                # Check if there's a name match with bbox mismatch
                name_matches = [(sn, sb) for sn, sb in src_rooms if sn == rname]
                if name_matches:
                    for sn, sb in name_matches:
                        logger.error(
                            f"Fix5 INTEGRITY FAILURE: '{rname}' in {img_file}: "
                            f"COCO={coco_bbox} != source={sb}"
                        )
                        integrity_ok = False
                        break

            # Polygon/bbox agreement: a real outline must be bounded by the
            # bbox it ships with (same ±1px tolerance as above). Catches a
            # stale polygon surviving next to a fresh bbox -- the exact
            # mismatch refine_annotations now strips up front.
            seg = coco_ann.get("segmentation") or []
            if seg and len(seg[0]) > 8:  # 8 = the 4-vertex rect fallback
                poly_bounds = _polygon_bounds(seg[0])
                if not all(
                    abs(poly_bounds[i] - coco_bbox[i]) <= 1.0 for i in range(4)
                ):
                    logger.error(
                        f"Fix5 POLYGON INTEGRITY FAILURE: '{rname}' in "
                        f"{img_file}: polygon bounds={poly_bounds} != "
                        f"bbox={coco_bbox}"
                    )
                    integrity_ok = False

        if integrity_ok:
            logger.info(
                "Fix5: bbox integrity check PASSED — all bboxes match source "
                "±1px; all polygons bounded by their bbox ±1px"
            )

        with open(output_file, "w") as f:
            json.dump(coco, f, indent=2)

        logger.info(
            f"COCOExporter: exported {img_id} images, "
            f"{ann_id - 1} annotations → {output_file}"
        )
        return img_id

    def export_splits(
        self,
        processed_dir: Path | str,
        images_dir: Path | str,
        output_dir: Path | str,
        splits: tuple = (0.70, 0.15, 0.15),
        seed: int = 42,
        min_annotations_per_image: int = 3,
    ) -> dict:
        """
        Export train / val / test COCO files with stratified splitting.

        Fix 7: Master category array shared across all splits.
        Fix 8: Minimum annotation count per image.
        Fix 9: Proper 70/15/15 split balance.

        Args:
            processed_dir: Directory of processed annotation JSON files.
            images_dir:    Directory of source images.
            output_dir:    Directory to write train.json, val.json, test.json.
            splits:        Fractions for (train, val, test). Must sum to 1.0.
            seed:          Random seed for reproducibility.
            min_annotations_per_image: Minimum rooms per image for inclusion.

        Returns:
            Dict with split names as keys and image counts as values.
        """
        import random
        from collections import defaultdict

        assert abs(sum(splits) - 1.0) < 1e-6, "splits must sum to 1.0"

        processed_dir = Path(processed_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load all SFT-ready annotations
        all_anns = []
        for json_file in sorted(processed_dir.glob("*.json")):
            try:
                with open(json_file) as f:
                    ann = json.load(f)
                if ann.get("sft_ready", False):
                    all_anns.append((json_file.name, ann))
            except Exception as e:
                logger.error(f"Split: failed to read {json_file}: {e}")

        if not all_anns:
            logger.warning("No SFT-ready annotations found for splitting")
            return {}

        # Fix 7: Compute master category list (union of all categories across
        # ALL annotations) before splitting.  This ensures identical category
        # arrays and IDs in train/val/test.
        all_used_types: set = set()
        for _, ann in all_anns:
            for room in ann.get("rooms", []):
                raw_type = room.get("type") or room.get("category") or "other"
                if raw_type != "other":  # Checklist #14: no "other" category
                    all_used_types.add(raw_type)
        master_categories = sorted(all_used_types)
        logger.info(
            f"Fix7: master category list ({len(master_categories)}): "
            f"{master_categories}"
        )

        # Group by dominant room type for stratification
        def dominant_type(ann: dict) -> str:
            from collections import Counter
            rooms = ann.get("rooms", [])
            if not rooms:
                return "other"
            types = [r.get("type") or r.get("category") or "other" for r in rooms]
            return Counter(types).most_common(1)[0][0]

        by_type: dict = defaultdict(list)
        for fname, ann in all_anns:
            by_type[dominant_type(ann)].append((fname, ann))

        # Fix 9: Stratified split with proper 70/15/15 balance
        rng = random.Random(seed)
        train_anns, val_anns, test_anns = [], [], []

        for group in by_type.values():
            rng.shuffle(group)
            n = len(group)
            n_train = max(1, int(n * splits[0])) if n >= 2 else n
            n_val = int(n * splits[1])
            train_anns.extend(group[:n_train])
            val_anns.extend(group[n_train:n_train + n_val])
            test_anns.extend(group[n_train + n_val:])

        # Fix 9: Post-hoc redistribution for small datasets.
        # Guarantee: train ≥ 60% of images, no empty splits when total ≥ 3.
        split_lists = [train_anns, val_anns, test_anns]
        total = len(all_anns)

        if total >= 3:
            # Ensure no empty splits
            for i, slist in enumerate(split_lists):
                if len(slist) == 0:
                    donor_idx = max(range(3), key=lambda j: len(split_lists[j]))
                    if len(split_lists[donor_idx]) > 1:
                        slist.append(split_lists[donor_idx].pop())
                        split_names = ["train", "val", "test"]
                        logger.info(
                            f"Fix9: moved 1 image from "
                            f"'{split_names[donor_idx]}' to '{split_names[i]}' "
                            f"to prevent empty split"
                        )

            # Ensure train has ≥60% of images (checklist #10)
            train_pct = len(train_anns) / total
            while train_pct < 0.60 and total >= 3:
                # Move from largest non-train split
                non_train = [(1, val_anns), (2, test_anns)]
                donor_idx, donor_list = max(non_train, key=lambda x: len(x[1]))
                if len(donor_list) > 1:
                    train_anns.append(donor_list.pop())
                    split_names = ["train", "val", "test"]
                    logger.info(
                        f"Fix9: moved 1 image from '{split_names[donor_idx]}' "
                        f"to 'train' (train was {train_pct:.0%})"
                    )
                    train_pct = len(train_anns) / total
                else:
                    break

        # Log final split distribution
        logger.info(
            f"Fix9: final splits — train={len(train_anns)}, "
            f"val={len(val_anns)}, test={len(test_anns)} "
            f"(total={total})"
        )

        # Write each split
        counts = {}
        for split_name, split_data in [("train", train_anns), ("val", val_anns), ("test", test_anns)]:
            # Build a temporary processed_dir containing only this split
            split_dir = output_dir / f"_{split_name}_tmp"
            split_dir.mkdir(exist_ok=True)
            for fname, ann in split_data:
                import shutil
                shutil.copy2(processed_dir / fname, split_dir / fname)

            # Fix 7: Pass master_categories to export for consistent IDs
            # Fix 8: Pass min_annotations_per_image
            n = self.export(
                split_dir, images_dir, output_dir / f"{split_name}.json",
                master_categories=master_categories,
                min_annotations_per_image=min_annotations_per_image,
            )
            counts[split_name] = n

            # Clean up temp dir
            import shutil
            shutil.rmtree(split_dir)
            logger.info(f"Split '{split_name}': {n} images → {output_dir}/{split_name}.json")

        return counts


# ---------------------------------------------------------------------------
# Coverage Report Generator
# ---------------------------------------------------------------------------

class CoverageReporter:
    """
    Generate a room-type coverage report across all SFT-ready annotations.

    Without this report there is no visibility into class imbalance before
    fine-tuning.  A plan set with 200 'office' annotations and 2 'janitor'
    annotations will produce a biased model with no warning.
    """

    # Types below this count are flagged as low-coverage
    LOW_COVERAGE_THRESHOLD: int = 10

    def generate(
        self,
        processed_dir: Path | str,
        output_file: Path | str,
    ) -> dict:
        """
        Scan all SFT-ready annotations and produce a coverage report JSON.

        Args:
            processed_dir: Directory of processed annotation JSON files.
            output_file:   Path to write the report JSON.

        Returns:
            Coverage dict (also written to output_file).
        """
        from collections import Counter

        processed_dir = Path(processed_dir)
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        type_counts: Counter = Counter()
        total_images = 0
        sft_ready_images = 0

        for json_file in sorted(processed_dir.glob("*.json")):
            try:
                with open(json_file) as f:
                    ann = json.load(f)
            except Exception:
                continue

            total_images += 1
            if not ann.get("sft_ready", False):
                continue
            sft_ready_images += 1

            for room in ann.get("rooms", []):
                t = room.get("type") or room.get("category") or "other"
                type_counts[t] += 1

        low_coverage = [t for t, c in type_counts.items() if c < self.LOW_COVERAGE_THRESHOLD]

        report = {
            "total_images": total_images,
            "sft_ready_images": sft_ready_images,
            "total_sft_ready_rooms": sum(type_counts.values()),
            "by_type": dict(type_counts.most_common()),
            "low_coverage_types": low_coverage,
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        # Console output
        print(f"\n{'='*60}")
        print(f"COVERAGE REPORT  ({sft_ready_images}/{total_images} images SFT-ready)")
        print(f"{'='*60}")
        for t, count in type_counts.most_common():
            bar = "█" * min(40, count // max(1, sum(type_counts.values()) // 40))
            flag = "  ⚠  LOW" if count < self.LOW_COVERAGE_THRESHOLD else ""
            print(f"  {t:30s} {count:5d}  {bar}{flag}")
        if low_coverage:
            print(f"\n⚠  Low-coverage types (<{self.LOW_COVERAGE_THRESHOLD}): {', '.join(low_coverage)}")
        print(f"{'='*60}\n")

        return report

    def print_summary(self, report: dict) -> None:
        """Print a summary from a previously generated report dict."""
        print(f"SFT-ready rooms: {report.get('total_sft_ready_rooms', 0)}")
        low = report.get("low_coverage_types", [])
        if low:
            print(f"⚠  Low-coverage types: {', '.join(low)}")
