"""
F9-F12: Bounding box validation, persistence, and schema management.

Implements:
- F9: Bbox validation with OOB/NaN/invalid checks
- F10: Coordinate normalization with reference dimensions
- F11: Atomic writes with schema versioning
- F12: OCR bbox filtering before persistence
"""

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BBoxValidationResult:
    """Result of bbox validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    fixed: bool = False
    fixed_bbox: Optional[List[float]] = None


def validate_bbox(
    bbox: List[float],
    image_width: int,
    image_height: int,
    allow_partial_oob: bool = False,
    fix_issues: bool = True,
) -> BBoxValidationResult:
    """
    F9: Comprehensive bbox validation with OOB/NaN/invalid checks.
    
    Args:
        bbox: Bounding box [x, y, w, h] or [x1, y1, x2, y2]
        image_width: Image width for boundary checking
        image_height: Image height for boundary checking
        allow_partial_oob: Allow bboxes partially outside image bounds
        fix_issues: Attempt to fix fixable issues (clamping, etc.)
    
    Returns:
        BBoxValidationResult with validation status and any fixes applied
    """
    result = BBoxValidationResult(is_valid=True, errors=[], warnings=[])
    
    # Check basic structure
    if not bbox or len(bbox) != 4:
        result.is_valid = False
        result.errors.append(f"Invalid bbox length: {len(bbox)} (expected 4)")
        return result
    
    try:
        bbox_floats = [float(v) for v in bbox]
    except (ValueError, TypeError) as e:
        result.is_valid = False
        result.errors.append(f"Non-numeric bbox values: {bbox}, error: {e}")
        return result
    
    x, y, w, h = bbox_floats
    
    # Check for NaN/Inf
    if not all(isinstance(v, float) or isinstance(v, int) for v in bbox_floats):
        result.is_valid = False
        result.errors.append("Bbox contains non-numeric values")
        return result
    
    import math
    if any(math.isnan(v) or math.isinf(v) for v in bbox_floats):
        result.is_valid = False
        result.errors.append(f"Bbox contains NaN or Inf: {bbox_floats}")
        return result
    
    # Check for negative values
    if any(v < 0 for v in [w, h]):
        result.is_valid = False
        result.errors.append(f"Bbox has negative width/height: w={w}, h={h}")
        if fix_issues:
            result.warnings.append("Cannot fix negative dimensions - bbox is invalid")
        return result
    
    # Check for zero area
    if w == 0 or h == 0:
        result.is_valid = False
        result.errors.append(f"Bbox has zero area: w={w}, h={h}")
        return result
    
    # Assume xywh format for now
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    
    # Check bounds
    oob_x1 = x1 < 0
    oob_y1 = y1 < 0
    oob_x2 = x2 > image_width
    oob_y2 = y2 > image_height
    
    oob_issues = [oob_x1, oob_y1, oob_x2, oob_y2]
    
    if any(oob_issues):
        if all(oob_issues) or (oob_x1 and oob_x2) or (oob_y1 and oob_y2):
            # Completely outside or inverted
            result.is_valid = False
            result.errors.append(
                f"Bbox completely out of bounds or inverted: "
                f"({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) vs image({image_width}x{image_height})"
            )
            return result
        
        if allow_partial_oob:
            result.warnings.append(
                f"Bbox partially out of bounds: "
                f"({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) vs image({image_width}x{image_height})"
            )
        else:
            result.is_valid = False
            result.errors.append(
                f"Bbox out of bounds: "
                f"({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) vs image({image_width}x{image_height})"
            )
            
            if fix_issues:
                # Try to clamp
                x1_clamp = max(0, min(image_width, x1))
                y1_clamp = max(0, min(image_height, y1))
                x2_clamp = max(0, min(image_width, x2))
                y2_clamp = max(0, min(image_height, y2))
                
                if x2_clamp > x1_clamp and y2_clamp > y1_clamp:
                    # Convert back to xywh
                    w_new = x2_clamp - x1_clamp
                    h_new = y2_clamp - y1_clamp
                    result.fixed_bbox = [x1_clamp, y1_clamp, w_new, h_new]
                    result.fixed = True
                    result.warnings.append(f"Clamped bbox to: {result.fixed_bbox}")
    
    if not result.errors:
        result.is_valid = True
    
    return result


def normalize_bboxes(
    bboxes: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    F10: Normalize bboxes to [0, 1] range with reference dimensions.
    
    Args:
        bboxes: List of bbox dicts with 'bbox' field in [x, y, w, h] format
        image_width: Original image width
        image_height: Original image height
    
    Returns:
        (normalized_bboxes, reference_dimensions)
    """
    normalized = []
    reference_dims = {
        "width": image_width,
        "height": image_height,
        "bbox_format": "xywh_normalized"  # [x, y, w, h] all in [0, 1]
    }
    
    for bbox_dict in bboxes:
        normalized_dict = bbox_dict.copy()
        bbox = bbox_dict.get("bbox", [])
        
        if bbox and len(bbox) == 4:
            x, y, w, h = bbox
            # Normalize to [0, 1]
            normalized_bbox = [
                x / image_width,
                y / image_height,
                w / image_width,
                h / image_height
            ]
            normalized_dict["bbox"] = normalized_bbox
        
        normalized.append(normalized_dict)
    
    return normalized, reference_dims


def denormalize_bboxes(
    normalized_bboxes: List[Dict[str, Any]],
    reference_dims: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    F10: Convert normalized bboxes back to pixel coordinates.
    
    Args:
        normalized_bboxes: List of bbox dicts with normalized [0, 1] coordinates
        reference_dims: Reference dimensions dict with width/height
    
    Returns:
        Denormalized bboxes in pixel coordinates
    """
    denormalized = []
    width = reference_dims.get("width", 4500)
    height = reference_dims.get("height", 3375)
    
    for bbox_dict in normalized_bboxes:
        denorm_dict = bbox_dict.copy()
        bbox = bbox_dict.get("bbox", [])
        
        if bbox and len(bbox) == 4:
            x_norm, y_norm, w_norm, h_norm = bbox
            # Denormalize back to pixels
            denorm_bbox = [
                x_norm * width,
                y_norm * height,
                w_norm * width,
                h_norm * height
            ]
            denorm_dict["bbox"] = denorm_bbox
        
        denormalized.append(denorm_dict)
    
    return denormalized


def write_annotation_atomic(
    data: Dict[str, Any],
    output_path: Path,
    schema_version: str = "1.0",
) -> bool:
    """
    F11: Write annotation with atomic writes (tmp-then-rename) and schema versioning.
    
    Args:
        data: Annotation data to write
        output_path: Target output path
        schema_version: Schema version string
    
    Returns:
        Success status
    """
    try:
        # Add schema version
        data = data.copy()
        data["schema_version"] = schema_version
        
        # Write to temporary file first
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=output_path.parent,
            prefix='.tmp_',
            suffix='.json',
            delete=False
        ) as tmp_file:
            json.dump(data, tmp_file, indent=2)
            tmp_path = Path(tmp_file.name)
        
        # Atomic rename
        tmp_path.replace(output_path)
        logger.debug(f"Wrote annotation to {output_path} (schema v{schema_version})")
        return True
        
    except Exception as e:
        logger.error(f"Failed to write annotation to {output_path}: {e}")
        if tmp_path.exists():
            tmp_path.unlink()
        return False


def filter_oob_bboxes(
    bboxes: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    F12: Filter out OCR/detection bboxes that are out-of-bounds.
    
    Args:
        bboxes: List of bbox dicts to filter
        image_width: Image width for validation
        image_height: Image height for validation
    
    Returns:
        (filtered_bboxes, num_filtered)
    """
    filtered = []
    filtered_count = 0
    
    for bbox_dict in bboxes:
        bbox = bbox_dict.get("bbox", [])
        
        if not bbox or len(bbox) != 4:
            filtered.append(bbox_dict)
            continue
        
        x, y, w, h = bbox
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        
        # Check if bbox is within image bounds
        if 0 <= x1 < image_width and 0 <= y1 < image_height and \
           0 < x2 <= image_width and 0 < y2 <= image_height:
            filtered.append(bbox_dict)
        else:
            logger.debug(
                f"Filtering OOB bbox: ({x1},{y1},{x2},{y2}) vs "
                f"image({image_width}x{image_height})"
            )
            filtered_count += 1
    
    return filtered, filtered_count


class AnnotationValidator:
    """F9-F12: Comprehensive annotation validation and persistence."""
    
    def __init__(self, image_width: int = 4500, image_height: int = 3375):
        """Initialize validator with image dimensions."""
        self.image_width = image_width
        self.image_height = image_height
    
    def validate_annotation(self, annotation: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate entire annotation structure.
        
        Args:
            annotation: Annotation dict with 'rooms', 'image_size', etc.
        
        Returns:
            (is_valid, errors)
        """
        errors = []
        
        # Check image size
        image_size = annotation.get("image_size", {})
        if not image_size:
            errors.append("Missing image_size field")
        
        # Validate all room bboxes (F9)
        rooms = annotation.get("rooms", [])
        for i, room in enumerate(rooms):
            bbox = room.get("bbox", [])
            if bbox:
                result = validate_bbox(
                    bbox,
                    image_size.get("width", self.image_width),
                    image_size.get("height", self.image_height),
                    allow_partial_oob=True,  # Warnings only for rooms
                )
                if not result.is_valid:
                    errors.append(f"Room {i} bbox invalid: {result.errors[0]}")
        
        return len(errors) == 0, errors
    
    def process_and_save(
        self,
        annotation: Dict[str, Any],
        output_path: Path,
        normalize: bool = False,
        filter_oob: bool = True,
        schema_version: str = "1.0"
    ) -> bool:
        """
        Process annotation (filter OOB, optionally normalize) and save with atomic writes.
        
        Implements F9-F12:
        - F9: Validates bboxes
        - F10: Optionally normalizes to [0, 1]
        - F11: Writes atomically with schema version
        - F12: Filters OOB bboxes
        
        Args:
            annotation: Raw annotation data
            output_path: Target output path
            normalize: Whether to normalize to [0, 1] range
            filter_oob: Whether to filter OOB bboxes (F12)
            schema_version: Schema version string
        
        Returns:
            Success status
        """
        output_annotation = annotation.copy()
        image_size = annotation.get("image_size", {})
        image_width = image_size.get("width", self.image_width)
        image_height = image_size.get("height", self.image_height)
        
        # F12: Filter OOB bboxes
        if filter_oob:
            rooms = annotation.get("rooms", [])
            filtered_rooms, num_filtered = filter_oob_bboxes(rooms, image_width, image_height)
            if num_filtered > 0:
                logger.info(f"Filtered {num_filtered} OOB bboxes")
            output_annotation["rooms"] = filtered_rooms
        
        # F10: Normalize if requested
        if normalize:
            rooms = output_annotation.get("rooms", [])
            normalized_rooms, ref_dims = normalize_bboxes(rooms, image_width, image_height)
            output_annotation["rooms"] = normalized_rooms
            output_annotation["reference_dimensions"] = ref_dims
        
        # F11: Write atomically with schema version
        return write_annotation_atomic(output_annotation, output_path, schema_version)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate and process annotations")
    parser.add_argument("--input", type=Path, required=True, help="Input annotation file")
    parser.add_argument("--output", type=Path, required=True, help="Output annotation file")
    parser.add_argument("--normalize", action="store_true", help="Normalize bboxes to [0,1]")
    parser.add_argument("--filter-oob", action="store_true", help="Filter OOB bboxes")
    
    args = parser.parse_args()
    
    with open(args.input, 'r') as f:
        annotation = json.load(f)
    
    validator = AnnotationValidator()
    success = validator.process_and_save(
        annotation,
        args.output,
        normalize=args.normalize,
        filter_oob=args.filter_oob
    )
    
    if success:
        print(f"✓ Successfully processed {args.input} → {args.output}")
    else:
        print(f"✗ Failed to process {args.input}")
