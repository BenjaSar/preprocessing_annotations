"""
Quality validation for room annotations.

Fixes applied vs previous version:
  1. Bbox format: readers now convert [x, y, w, h] → [x1, y1, x2, y2] before
     unpacking, matching the format written by all annotation writers.
  2. Field name: reads room type from "type" | "category" | "room_type"
     (in priority order) instead of the non-existent "room_type"-only lookup.
  3. Taxonomy: VALID_ROOM_TYPES imported from canonical taxonomy module so
     the whitelist is always in sync with TaxonomyNormalizer.
  4. Overlap detection: circulation types (hallway, corridor, lobby, elevator,
     stairwell) are exempt — adjacency overlaps are expected in floor plans.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

try:
    from .taxonomy import VALID_TYPES
except ImportError:
    from taxonomy import VALID_TYPES

logger = logging.getLogger(__name__)


def _xywh_to_xyxy(bbox: List) -> Tuple[float, float, float, float]:
    """Convert [x, y, width, height] → (x1, y1, x2, y2)."""
    x, y, w, h = [float(v) for v in bbox]
    return x, y, x + w, y + h


def _get_room_type(room: Dict) -> Optional[str]:
    """Read room type using canonical field priority: type > category > room_type."""
    return room.get("type") or room.get("category") or room.get("room_type")


@dataclass
class ValidationIssue:
    severity: str
    code: str
    message: str
    room_index: Optional[int] = None
    affected_rooms: Optional[List[int]] = None


class QualityChecker:
    """Validate annotation quality and consistency against mandatory SFT taxonomy."""

    MIN_BBOX_AREA: int = 2500
    VALID_ROOM_TYPES: Set[str] = VALID_TYPES  # Mandatory classes only (21 types)
    # Circulation types exempt from overlap checks (adjacency overlaps are expected)
    OVERLAP_EXEMPT_CATEGORIES: Set[str] = {
        "CORRIDOR", "LOBBY", "STAIRWELL",  # Mandatory class names
    }

    def __init__(self, min_bbox_area: int = 2500, allow_overlaps: bool = False):
        self.MIN_BBOX_AREA = min_bbox_area
        self.allow_overlaps = allow_overlaps

    def check_annotation(self, annotation: Dict, image_path: Optional[str] = None) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        image_size = annotation.get("image_size", {})
        width = image_size.get("width")
        height = image_size.get("height")

        if not width or not height:
            issues.append(ValidationIssue("error", "MISSING_IMAGE_SIZE", "Image dimensions not specified"))
            return issues

        rooms = annotation.get("rooms", [])
        if not rooms:
            issues.append(ValidationIssue("warning", "NO_ROOMS_DETECTED", "No rooms detected in image"))
            return issues

        for i, room in enumerate(rooms):
            issues.extend(self._check_room(room, i, width, height))

        if not self.allow_overlaps:
            issues.extend(self._check_overlaps(rooms))

        return issues

    def _check_room(self, room: Dict, index: int, img_width: int, img_height: int) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        bbox = room.get("bbox")

        if not bbox:
            issues.append(ValidationIssue("error", "MISSING_BBOX", "Bounding box missing", room_index=index))
            return issues

        if len(bbox) != 4:
            issues.append(ValidationIssue("error", "INVALID_BBOX_FORMAT", f"Expected 4 values, got {len(bbox)}", room_index=index))
            return issues

        try:
            bbox_floats = [float(v) for v in bbox]
        except (ValueError, TypeError):
            issues.append(ValidationIssue("error", "BBOX_NOT_NUMERIC", "Bbox values not numeric", room_index=index))
            return issues

        # FIX 1: Convert [x,y,w,h] → [x1,y1,x2,y2]
        x1, y1, x2, y2 = _xywh_to_xyxy(bbox_floats)

        if x1 >= x2 or y1 >= y2:
            issues.append(ValidationIssue("error", "INVALID_BBOX_COORDINATES",
                f"Degenerate bbox after xywh→xyxy: ({x1:.0f},{y1:.0f})→({x2:.0f},{y2:.0f})", room_index=index))
            return issues

        if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
            issues.append(ValidationIssue("warning", "BBOX_OUT_OF_BOUNDS",
                f"Bbox extends outside image ({img_width}×{img_height})", room_index=index))

        area = (x2 - x1) * (y2 - y1)
        if area < self.MIN_BBOX_AREA:
            issues.append(ValidationIssue("warning", "BBOX_TOO_SMALL",
                f"Bbox area {int(area)} < {self.MIN_BBOX_AREA}", room_index=index))

        # FIX 2: Use canonical field chain; FIX 3: validate against full taxonomy
        room_type = _get_room_type(room)
        if not room_type:
            issues.append(ValidationIssue("error", "MISSING_ROOM_TYPE",
                "Room type missing (checked: type, category, room_type)", room_index=index))
        elif room_type not in self.VALID_ROOM_TYPES:
            issues.append(ValidationIssue("error", "INVALID_ROOM_TYPE",
                f"Unknown room type: '{room_type}'", room_index=index))

        confidence = room.get("confidence")
        if confidence is not None:
            if not isinstance(confidence, (int, float)):
                issues.append(ValidationIssue("warning", "INVALID_CONFIDENCE_TYPE",
                    "Confidence not numeric", room_index=index))
            elif not (0.0 <= float(confidence) <= 1.0):
                issues.append(ValidationIssue("warning", "CONFIDENCE_OUT_OF_RANGE",
                    f"Confidence {confidence} out of [0,1]", room_index=index))

        return issues

    def _check_overlaps(self, rooms: List[Dict]) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        for i, room_i in enumerate(rooms):
            bbox_i = room_i.get("bbox")
            if not bbox_i or len(bbox_i) != 4:
                continue
            # FIX 4: exempt circulation types (mandatory class names are uppercase)
            type_i = _get_room_type(room_i) or ""
            if type_i in self.OVERLAP_EXEMPT_CATEGORIES:
                continue
            try:
                box_i = _xywh_to_xyxy(bbox_i)
            except (ValueError, TypeError):
                continue

            for j, room_j in enumerate(rooms):
                if i >= j:
                    continue
                bbox_j = room_j.get("bbox")
                if not bbox_j or len(bbox_j) != 4:
                    continue
                type_j = _get_room_type(room_j) or ""
                if type_j in self.OVERLAP_EXEMPT_CATEGORIES:
                    continue
                try:
                    box_j = _xywh_to_xyxy(bbox_j)
                except (ValueError, TypeError):
                    continue

                if self._boxes_overlap(box_i, box_j):
                    if self._containment_ratio(box_i, box_j) > 0.7:
                        continue  # label inside room polygon — expected
                    issues.append(ValidationIssue("warning", "BBOX_OVERLAP",
                        "Overlapping bboxes (non-circulation types)", affected_rooms=[i, j]))
        return issues

    def check_batch(self, annotations: List[Dict]) -> Dict[str, List[ValidationIssue]]:
        return {str(i): iss for i, ann in enumerate(annotations) if (iss := self.check_annotation(ann))}

    def get_summary(self, issues: List[ValidationIssue]) -> Dict:
        s: Dict = {"total": len(issues), "by_severity": {}, "by_code": {}}
        for iss in issues:
            s["by_severity"][iss.severity] = s["by_severity"].get(iss.severity, 0) + 1
            s["by_code"][iss.code] = s["by_code"].get(iss.code, 0) + 1
        return s

    def report(self, issues: List[ValidationIssue]) -> str:
        if not issues:
            return "✅ No issues found"
        lines = [f"Found {len(issues)} issues:\n"]
        by_sev: Dict = {}
        for iss in issues:
            by_sev.setdefault(iss.severity, []).append(iss)
        for sev in ("error", "warning", "info"):
            if sev not in by_sev:
                continue
            lines.append(f"{sev.upper()} ({len(by_sev[sev])}):")
            for iss in by_sev[sev]:
                loc = ""
                if iss.room_index is not None:
                    loc = f" [room {iss.room_index}]"
                elif iss.affected_rooms:
                    loc = f" [rooms {iss.affected_rooms}]"
                lines.append(f"  - {iss.code}{loc}: {iss.message}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _boxes_overlap(b1, b2) -> bool:
        x1_1, y1_1, x2_1, y2_1 = b1
        x1_2, y1_2, x2_2, y2_2 = b2
        return not (x2_1 <= x1_2 or x2_2 <= x1_1 or y2_1 <= y1_2 or y2_2 <= y1_1)

    @staticmethod
    def _containment_ratio(b1, b2) -> float:
        ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        smaller = min(a1, a2)
        return inter / smaller if smaller > 0 else 0.0
