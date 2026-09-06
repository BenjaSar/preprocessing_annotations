#!/usr/bin/env python3
"""Visualize bounding box annotations on floor plan images.

Reads annotation JSON files produced by the pipeline and draws color-coded
bounding boxes with labels on the source images. Useful for visually verifying
VLM room detection quality.

Usage:
    # Visualize all annotations from a pipeline run
    python visualize_bbox.py --input /path/to/pipeline_output

    # Visualize a single annotation
    python visualize_bbox.py --input /path/to/pipeline_output --single page000.json

    # Custom output directory and opacity
    python visualize_bbox.py --input /path/to/output --output /tmp/viz --opacity 0.3
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from preprocessing_annotations.bbox.bbox_visualizer import OBJECT_COLOR_MAP

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette: room type -> (R, G, B)
# ---------------------------------------------------------------------------
# Semantic-group palette covering all 22 mandatory taxonomy types (see
# automation/taxonomy.py VALID_TYPES). Each group shares a hue family (derived
# from the dataviz skill's validated 8-slot categorical ramp); members within a
# group are distinguished by shade — safe because every box always carries a
# text label banner (secondary encoding), per the skill's floor-band exception
# for >12 categories. Validated: node validate_palette.js on the 7 group anchors
# + fallback, worst adjacent CVD ΔE 11.2 (floor band, legal with labels).
ROOM_COLORS = {
    # Office — blue family
    "PRIVATE OFFICE":     (42,  120, 214),
    "OPEN OFFICE":        (30,  86,  155),
    "CONFERENCE":         (93,  153, 224),
    "MEETING":            (20,  58,  103),
    # Circulation — gold family
    "CORRIDOR":           (237, 161, 0),
    "LOBBY":              (166, 112, 0),
    "STAIRWELL":          (255, 187, 43),
    # Service / Utility — violet family
    "ELECTRICAL ROOM":    (74,  58,  167),
    "STORAGE ROOM":       (51,  40,  114),
    "JANITOR CLOSET":     (105, 89,  197),
    "PARKING GARAGE":     (37,  29,  83),
    "WAREHOUSE":          (139, 127, 210),
    # Residential — green family
    "RESIDENTIAL UNIT":   (0,   131, 0),
    # Education — orange family
    "CLASSROOM":          (235, 104, 52),
    "LECTURE HALL":       (196, 70,  19),
    "TRAINING ROOM":      (240, 145, 108),
    "MULTIPURPOSE ROOM":  (141, 50,  14),
    "GYMNASIUM":          (242, 158, 125),
    # Retail / Food — magenta family
    "RETAIL":             (232, 123, 164),
    "RESTAURANT":         (221, 62,  122),
    "CAFETERIA":          (234, 133, 171),
    # Sanitary — aqua family
    "RESTROOM":           (27,  175, 122),
    # Legacy / extended types seen in some data (not in mandatory taxonomy)
    "ELEVATOR":           (200, 100, 100),
    "KITCHEN":            (245, 130, 48),
    "RECEPTION":          (70,  130, 180),
}
# Pure magenta: not used by any group above, so an unmapped type is unmistakable
# rather than silently rendered near-invisible (the bug this replaces).
DEFAULT_COLOR = (255, 0, 255)


def _get_color(room_type: str) -> tuple:
    """Look up color for a room type (case-insensitive)."""
    return ROOM_COLORS.get(room_type.upper(), DEFAULT_COLOR)


def _dashed_rectangle(draw: ImageDraw.ImageDraw, box, color, width: int = 1, dash: int = 10):
    """Draw a dashed rectangle outline — visually distinct from a solid one.

    Used for the T-5 enlarged visibility marker so it's never mistaken for a
    real room-sized box (both previously rendered as identical solid rects).
    """
    x1, y1, x2, y2 = box
    edges = [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
             ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]
    for (sx, sy), (ex, ey) in edges:
        length = max(abs(ex - sx), abs(ey - sy))
        steps = max(1, length // dash)
        for i in range(0, int(steps), 2):
            t0, t1 = i / steps, min(1.0, (i + 1) / steps)
            draw.line([
                (sx + (ex - sx) * t0, sy + (ey - sy) * t0),
                (sx + (ex - sx) * t1, sy + (ey - sy) * t1),
            ], fill=color, width=width)


def _rects_overlap(a: tuple, b: tuple) -> bool:
    """AABB overlap test for (x1, y1, x2, y2) rects."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2


def _load_font(image_width: int):
    """Load a scaled font based on image dimensions."""
    # Target: ~0.4% of image width, clamped to 14-48px
    size = max(14, min(48, int(image_width * 0.004)))
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in font_paths:
        try:
            return ImageFont.truetype(fp, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def draw_annotations(image_path: Path, annotation: dict, output_path: Path,
                     opacity: float = 0.25, min_area: int = 0) -> int:
    """Draw bounding boxes from roomsRecognized onto the source image, plus
    any objectDetections (door/window/etc, F8) in a separate pass.

    Args:
        image_path:  Path to source PNG image.
        annotation:  Parsed annotation dict (must contain 'roomsRecognized';
                     'objectDetections' is optional -- absent/empty draws
                     nothing extra, identical to before this existed).
        output_path: Where to save the visualization PNG.
        opacity:     Fill opacity for bbox rectangles (0.0-1.0).
        min_area:    Skip boxes whose pixel area (w*h) < min_area (P1).

    Returns:
        Number of rooms drawn (objectDetections are not counted here --
        same "drawn" contract as before this feature existed).
    """
    # Load base image and convert to RGBA for compositing
    base = Image.open(image_path).convert("RGBA")

    # Coord-space guard: bboxes are in annotation["image_size"] space. If that
    # doesn't match the loaded image (e.g. a re-extraction changed resolution),
    # blindly drawing raw coords silently mislocates every box off the real
    # geometry ("outside floorplan"). Scale instead of trusting them as-is.
    ann_size = annotation.get("image_size") or {}
    ann_w, ann_h = ann_size.get("width"), ann_size.get("height")
    scale_x = scale_y = 1.0
    if ann_w and ann_h and (ann_w != base.width or ann_h != base.height):
        scale_x, scale_y = base.width / ann_w, base.height / ann_h
        logger.warning(
            f"{image_path.name}: annotation image_size {ann_w}x{ann_h} != "
            f"loaded image {base.width}x{base.height} — scaling bboxes "
            f"(x*{scale_x:.3f}, y*{scale_y:.3f})"
        )

    # Create two overlay layers: one for semi-transparent fills, one for strokes/text
    fill_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    stroke_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_fill = ImageDraw.Draw(fill_layer)
    draw_stroke = ImageDraw.Draw(stroke_layer)

    font = _load_font(base.width)
    rooms = annotation.get("roomsRecognized", [])
    stroke_width = max(2, int(base.width * 0.0006))
    drawn = 0

    for room in rooms:
        bbox = room.get("coordinates", {}).get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        if scale_x != 1.0 or scale_y != 1.0:
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
        if min_area > 0 and (x2 - x1) * (y2 - y1) < min_area:
            continue
        drawn += 1

        room_type = room.get("type", "UNKNOWN")
        room_name = room.get("name", "")
        color = _get_color(room_type)
        fill_alpha = int(255 * opacity)

        # T-5: label-scale rooms (e.g. TELECOM ROOM 112x20px) are present in the
        # data but render as an invisible sliver at sheet scale, so they get
        # mis-read as "omitted". Draw the outline around a minimum-visible box
        # centered on the true bbox so every kept room is findable. Fill stays
        # on the true bbox — only the visibility marker is enlarged.
        MIN_VIS = max(24, int(base.width * 0.006))  # ~27px at 4500px width
        vx1, vy1, vx2, vy2 = x1, y1, x2, y2
        is_enlarged = (x2 - x1) < MIN_VIS or (y2 - y1) < MIN_VIS
        if is_enlarged:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            half = MIN_VIS // 2
            vx1, vy1, vx2, vy2 = cx - half, cy - half, cx + half, cy + half

        # Semi-transparent fill on fill layer (true bbox)
        draw_fill.rectangle([x1, y1, x2, y2], fill=(*color, fill_alpha))

        # Solid outline for real-size boxes; dashed + thinner outline for the
        # enlarged visibility marker so it's never mistaken for an actual
        # room-sized detection (a real bug this session mistook for one).
        if is_enlarged:
            _dashed_rectangle(draw_stroke, [vx1, vy1, vx2, vy2], (*color, 255),
                               width=max(1, stroke_width // 2))
        else:
            draw_stroke.rectangle([vx1, vy1, vx2, vy2], outline=(*color, 255), width=stroke_width)

        # Label text
        label = f"{room_type}"
        if room_name:
            label += f" - {room_name}"

        # Measure text for background box
        text_bbox = draw_stroke.textbbox((x1, y1), label, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        pad = 3

        # Position label above the bbox; if too close to top edge, put it inside
        label_y = y1 - th - pad * 2
        if label_y < 0:
            label_y = y1 + pad

        # Background rectangle for label readability (opaque, on stroke layer)
        draw_stroke.rectangle(
            [x1, label_y, x1 + tw + pad * 2, label_y + th + pad * 2],
            fill=(*color, 255),
        )
        # White text
        draw_stroke.text(
            (x1 + pad, label_y + pad), label, fill=(255, 255, 255, 255), font=font
        )

    # F8: objectDetections (door/window/etc) -- same coord-space scaling
    # guard as rooms above, own color map (OBJECT_COLOR_MAP, shared with
    # bbox_visualizer.py so the two tools agree on colors), no min_area
    # filter (these boxes are inherently small, that filter is room-only)
    # and no enlarged-visibility marker (that's the room-specific T-5 fix
    # for rare tiny label-scale rooms, not applicable here).
    #
    # T4: adjacent/stacked doors (e.g. a double-door drawn as 2 abutting
    # single-leaf detections) previously got labels placed independently,
    # so both "door" banners land on the same pixels and blend into one
    # smear -- a reviewer scanning the overview reads that as "one leaf
    # missing" even though both were detected (confirmed on a real page:
    # 326 ROCKAWAY page004, BR132/BR118 double doors). Track placed label
    # rects and stack straight down on collision so every label stays
    # individually readable.
    placed_label_rects: list = []

    def _place_label_y(x1: int, y1: int, w: int, h: int, img_h: int) -> int:
        y = y1 - h - pad * 2
        if y < 0:
            y = y1 + pad
        while True:
            cand = (x1, y, x1 + w, y + h)
            collision = next(
                (r for r in placed_label_rects if _rects_overlap(cand, r)), None
            )
            if collision is None:
                break
            y = collision[3]  # stack below the label it collided with
            if y + h > img_h:
                break
        placed_label_rects.append((x1, y, x1 + w, y + h))
        return y

    for det in annotation.get("objectDetections", []):
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        try:
            dx1, dy1, dx2, dy2 = [int(v) for v in bbox]
        except (ValueError, TypeError):
            continue
        if scale_x != 1.0 or scale_y != 1.0:
            dx1, dx2 = int(dx1 * scale_x), int(dx2 * scale_x)
            dy1, dy2 = int(dy1 * scale_y), int(dy2 * scale_y)

        category = det.get("category") or "other"
        color = OBJECT_COLOR_MAP.get(category, OBJECT_COLOR_MAP["other"])
        fill_alpha = int(255 * opacity)

        draw_fill.rectangle([dx1, dy1, dx2, dy2], fill=(*color, fill_alpha))
        draw_stroke.rectangle([dx1, dy1, dx2, dy2], outline=(*color, 255), width=stroke_width)

        label = category
        text_bbox = draw_stroke.textbbox((dx1, dy1), label, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        pad = 3
        label_y = _place_label_y(dx1, dy1, tw + pad * 2, th + pad * 2, base.height)
        draw_stroke.rectangle(
            [dx1, label_y, dx1 + tw + pad * 2, label_y + th + pad * 2],
            fill=(*color, 255),
        )
        draw_stroke.text(
            (dx1 + pad, label_y + pad), label, fill=(255, 255, 255, 255), font=font
        )

    # Composite: base + fill_layer + stroke_layer
    result = Image.alpha_composite(base, fill_layer)
    result = Image.alpha_composite(result, stroke_layer)

    # Save as RGB PNG (drop alpha)
    result.convert("RGB").save(output_path, "PNG")
    return drawn


def remove_stale_visualizations(stem: str, overview_dir: Path, regions_dir: Path) -> int:
    """Delete a page's previously-written overview and region crops.

    Called when a page now has zero rooms: without this, an earlier run's
    visualizations linger on disk and misrepresent the current annotation
    (a page filtered down to no rooms would still show the old boxes). Keeps
    the visualization directory a faithful mirror of the annotation data.

    Args:
        stem:         Annotation filename stem (json_path.stem), used to match
                      both "<stem>_overview.png" and "<stem>_<id>_<type>.png".
        overview_dir: Directory holding overview images.
        regions_dir:  Directory holding per-room crop images.

    Returns:
        Number of files removed.
    """
    removed = 0
    overview = overview_dir / f"{stem}_overview.png"
    if overview.exists():
        overview.unlink()
        removed += 1
    for crop in regions_dir.glob(f"{stem}_*.png"):
        crop.unlink()
        removed += 1
    return removed


def draw_room_crops(image_path: Path, annotation: dict, output_dir: Path,
                    padding_pct: float = 0.1, opacity: float = 0.25,
                    min_area: int = 0) -> int:
    """Draw cropped room regions with bboxes annotated.

    Extracts individual rooms from the source image with padding, then draws
    the bbox rectangle on each crop. Uses the same cropping strategy as
    region_extractor.py but with visual annotations.

    Args:
        image_path:  Path to source PNG image.
        annotation:  Parsed annotation dict (must contain 'roomsRecognized').
        output_dir:  Directory to save cropped region images.
        padding_pct: Percentage of bbox dimension to add as padding (default 0.1 = 10%).
        opacity:     Fill opacity for bbox rectangle (0.0-1.0).
        min_area:    Skip boxes whose pixel area (w*h) < min_area (P1).

    Returns:
        Number of room crops drawn.
    """
    # Load source image
    base = Image.open(image_path).convert("RGB")
    img_width, img_height = base.size
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rooms = annotation.get("roomsRecognized", [])
    if not rooms:
        return 0

    image_stem = image_path.stem
    drawn_count = 0
    font = _load_font(base.width)
    stroke_width = max(1, int(base.width * 0.0003))

    for room in rooms:
        bbox = room.get("coordinates", {}).get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
        except (ValueError, TypeError):
            continue

        if min_area > 0 and (x2 - x1) * (y2 - y1) < min_area:
            continue

        # Compute padding (as percentage of bbox dimensions)
        width = x2 - x1
        height = y2 - y1
        pad_x = int(width * padding_pct)
        pad_y = int(height * padding_pct)

        # Apply padding with bounds clipping
        x1_padded = max(0, x1 - pad_x)
        y1_padded = max(0, y1 - pad_y)
        x2_padded = min(img_width, x2 + pad_x)
        y2_padded = min(img_height, y2 + pad_y)

        # Skip if crop would be empty or tiny
        crop_width = x2_padded - x1_padded
        crop_height = y2_padded - y1_padded
        if crop_width < 10 or crop_height < 10:
            continue

        # Crop the region
        crop = base.crop((x1_padded, y1_padded, x2_padded, y2_padded))

        # Draw bbox on the crop (offset by padding)
        draw_crop = ImageDraw.Draw(crop)
        bbox_x1 = x1 - x1_padded
        bbox_y1 = y1 - y1_padded
        bbox_x2 = x2 - x1_padded
        bbox_y2 = y2 - y1_padded

        color = _get_color(room.get("type", "UNKNOWN"))

        # Semi-transparent fill (draw on a separate layer)
        crop_rgba = crop.convert("RGBA")
        fill_layer = Image.new("RGBA", crop_rgba.size, (0, 0, 0, 0))
        draw_fill = ImageDraw.Draw(fill_layer)
        fill_alpha = int(255 * opacity)
        draw_fill.rectangle([bbox_x1, bbox_y1, bbox_x2, bbox_y2],
                           fill=(*color, fill_alpha))
        crop_rgba = Image.alpha_composite(crop_rgba, fill_layer)
        crop = crop_rgba.convert("RGB")
        draw_crop = ImageDraw.Draw(crop)

        # Solid outline
        draw_crop.rectangle([bbox_x1, bbox_y1, bbox_x2, bbox_y2],
                           outline=color, width=stroke_width)

        # Label
        room_type = room.get("type", "UNKNOWN")
        room_name = room.get("name", "")
        label = f"{room_type}"
        if room_name:
            label += f" - {room_name}"

        # Measure text
        text_bbox = draw_crop.textbbox((bbox_x1, bbox_y1), label, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        pad = 3

        # Position label above bbox; if too close to top, place inside
        label_y = bbox_y1 - th - pad * 2
        if label_y < 0:
            label_y = bbox_y1 + pad

        # Background rectangle
        draw_crop.rectangle(
            [bbox_x1, label_y, bbox_x1 + tw + pad * 2, label_y + th + pad * 2],
            fill=color,
        )
        # White text
        draw_crop.text((bbox_x1 + pad, label_y + pad), label,
                      fill=(255, 255, 255), font=font)

        # Save crop
        room_id = room.get("id", -1)
        out_name = f"{image_stem}_{room_id:03d}_{room_type}.png"
        out_path = output_dir / out_name
        crop.save(out_path, "PNG")

        drawn_count += 1
        logger.debug(f"Saved room crop: {out_name}")

    return drawn_count


def main():
    parser = argparse.ArgumentParser(
        description="Visualize bbox annotations on floor plan images."
    )
    parser.add_argument(
        "--input", required=True,
        help="Pipeline output directory (must contain annotations/ and images/ subdirs).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory for visualizations (default: <input>/visualizations/).",
    )
    parser.add_argument(
        "--single", default=None,
        help="Process only this annotation file (filename, not full path).",
    )
    parser.add_argument(
        "--opacity", type=float, default=0.25,
        help="Fill opacity for bbox rectangles, 0.0-1.0 (default: 0.25).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging.",
    )
    # P0-1: source selection
    parser.add_argument(
        "--source", default="processed_annotations",
        choices=["annotations", "processed_annotations"],
        help="Annotation set to draw. 'processed_annotations' (default) = final "
             "SFT spaces after SAM expansion and filtering. "
             "'annotations' = raw VLM/OCR output for debugging.",
    )
    # P1: area filter
    parser.add_argument(
        "--min-area", type=int, default=0,
        dest="min_area",
        help="Skip boxes whose pixel area (w*h) is below this value "
             "(e.g. 40000 drops label-sized boxes). Default: 0 (no filter).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    input_dir = Path(args.input)
    images_dir = input_dir / "images"

    # P0-1 + P0-2: resolve annotation source with fallback
    requested_dir = input_dir / args.source
    if requested_dir.is_dir() and any(requested_dir.glob("*.json")):
        annotations_dir = requested_dir
        logger.info(f"Source: {args.source}/")
    elif args.source == "processed_annotations":
        fallback = input_dir / "annotations"
        if fallback.is_dir():
            logger.warning(
                f"processed_annotations/ absent or empty — falling back to annotations/"
            )
            annotations_dir = fallback
        else:
            logger.error(f"Neither processed_annotations/ nor annotations/ found in {input_dir}")
            sys.exit(1)
    else:
        logger.error(f"annotations/ not found in {input_dir}")
        sys.exit(1)

    if not images_dir.is_dir():
        logger.error(f"images/ not found in {input_dir}")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else input_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for overview and room crops
    overview_dir = output_dir / "overview"
    regions_dir = output_dir / "regions"
    overview_dir.mkdir(parents=True, exist_ok=True)
    regions_dir.mkdir(parents=True, exist_ok=True)

    # Collect annotation files
    if args.single:
        json_files = [annotations_dir / args.single]
        if not json_files[0].exists():
            logger.error(f"Annotation file not found: {json_files[0]}")
            sys.exit(1)
    else:
        json_files = sorted(annotations_dir.glob("*.json"))

    if not json_files:
        logger.error(f"No .json files found in {annotations_dir}")
        sys.exit(1)

    logger.info(f"Processing {len(json_files)} annotation(s) from {annotations_dir}")

    processed = 0
    skipped = 0
    total_rooms = 0
    total_crops = 0

    for json_path in json_files:
        with open(json_path) as f:
            annotation = json.load(f)

        image_file = annotation.get("image_file", "")
        image_path = images_dir / image_file

        if not image_path.exists():
            logger.warning(f"Image not found, skipping: {image_path}")
            skipped += 1
            continue

        n_rooms = len(annotation.get("roomsRecognized", []))
        if n_rooms == 0:
            removed = remove_stale_visualizations(
                json_path.stem, overview_dir, regions_dir
            )
            if removed:
                logger.info(
                    f"{json_path.name}: 0 rooms — removed {removed} stale visualization(s)"
                )
            else:
                logger.debug(f"No roomsRecognized in {json_path.name}, nothing to draw")
            skipped += 1
            continue

        # Generate overview (full image with all bboxes)
        out_name = json_path.stem + "_overview.png"
        out_path = overview_dir / out_name
        drawn = draw_annotations(image_path, annotation, out_path, args.opacity,
                                 min_area=args.min_area)
        total_rooms += drawn
        logger.debug(f"  Overview: {out_name} ({drawn} rooms)")

        # Generate room crops (individual rooms with bbox annotated)
        crops_drawn = draw_room_crops(image_path, annotation, regions_dir,
                                      padding_pct=0.1, opacity=args.opacity,
                                      min_area=args.min_area)
        total_crops += crops_drawn
        logger.debug(f"  Room crops: {crops_drawn} regions saved")

        processed += 1
        logger.info(f"  {json_path.name}: {drawn} rooms (overview) + {crops_drawn} crops (regions)")

    logger.info(
        f"Done: {processed} images processed, {total_rooms} total rooms (overview), "
        f"{total_crops} total crops (regions), {skipped} skipped"
    )


if __name__ == "__main__":
    main()
