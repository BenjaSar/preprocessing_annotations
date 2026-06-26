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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette: room type -> (R, G, B)
# ---------------------------------------------------------------------------
ROOM_COLORS = {
    "PRIVATE OFFICE":    (55,  126, 184),   # steel blue
    "OPEN OFFICE":       (0,   190, 210),   # cyan
    "CONFERENCE":        (77,  175, 74),    # green
    "MEETING":           (0,   166, 153),   # teal
    "LOBBY":             (255, 127, 0),     # orange
    "CORRIDOR":          (255, 215, 0),     # gold
    "RESTROOM":          (152, 78,  163),   # purple
    "STAIRWELL":         (228, 26,  28),    # red
    "ELECTRICAL ROOM":   (150, 150, 150),   # gray
    "STORAGE ROOM":      (166, 86,  40),    # brown
    "ELEVATOR":          (200, 100, 100),   # dusty rose
    "KITCHEN":           (245, 130, 48),    # tangerine
    "RECEPTION":         (70,  130, 180),   # light steel blue
}
DEFAULT_COLOR = (200, 200, 200)  # light gray for unknown types


def _get_color(room_type: str) -> tuple:
    """Look up color for a room type (case-insensitive)."""
    return ROOM_COLORS.get(room_type.upper(), DEFAULT_COLOR)


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
    """Draw bounding boxes from roomsRecognized onto the source image.

    Args:
        image_path:  Path to source PNG image.
        annotation:  Parsed annotation dict (must contain 'roomsRecognized').
        output_path: Where to save the visualization PNG.
        opacity:     Fill opacity for bbox rectangles (0.0-1.0).
        min_area:    Skip boxes whose pixel area (w*h) < min_area (P1).

    Returns:
        Number of rooms drawn.
    """
    # Load base image and convert to RGBA for compositing
    base = Image.open(image_path).convert("RGBA")
    
    # Create two overlay layers: one for semi-transparent fills, one for strokes/text
    fill_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    stroke_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_fill = ImageDraw.Draw(fill_layer)
    draw_stroke = ImageDraw.Draw(stroke_layer)

    font = _load_font(base.width)
    rooms = annotation.get("roomsRecognized", [])
    stroke_width = max(2, int(base.width * 0.0006))

    for room in rooms:
        bbox = room.get("coordinates", {}).get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        if min_area > 0 and (x2 - x1) * (y2 - y1) < min_area:
            continue

        room_type = room.get("type", "UNKNOWN")
        room_name = room.get("name", "")
        color = _get_color(room_type)
        fill_alpha = int(255 * opacity)

        # Semi-transparent fill on fill layer
        draw_fill.rectangle([x1, y1, x2, y2], fill=(*color, fill_alpha))

        # Solid outline on stroke layer
        draw_stroke.rectangle([x1, y1, x2, y2], outline=(*color, 255), width=stroke_width)

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

    # Composite: base + fill_layer + stroke_layer
    result = Image.alpha_composite(base, fill_layer)
    result = Image.alpha_composite(result, stroke_layer)

    # Save as RGB PNG (drop alpha)
    result.convert("RGB").save(output_path, "PNG")
    return len(rooms)


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
            logger.warning(f"No roomsRecognized in {json_path.name}, skipping")
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
