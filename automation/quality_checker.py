"""
Quality validation for room annotations.

Checks annotation consistency, validates bounding boxes, and identifies issues.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Represents a single annotation quality issue."""

    severity: str  # "error", "warning", "info"
    code: str  # Issue code (e.g., "BBOX_OUT_OF_BOUNDS")
    message: str  # Human-readable description
    room_index: Optional[int] = None  # Which room (if applicable)
    affected_rooms: Optional[List[int]] = None  # Multiple rooms (e.g., overlaps)


class QualityChecker:
    """Validate annotation quality and consistency."""

    # Minimum bounding box area (in pixels)
    MIN_BBOX_AREA = 50 * 50

    # Valid room types (standardized)
    VALID_ROOM_TYPES = {
        "office",
        "conference_room",
        "meeting_room",
        "restroom",
        "kitchen",
        "lobby",
        "hallway",
        "storage",
        "mechanical_room",
        "elevator",
        "stairwell",
        "other"
    }

    def __init__(self, min_bbox_area: int = 2500, allow_overlaps: bool = False):
        """
        Initialize quality checker.

        Args:
            min_bbox_area: Minimum bounding box area in pixels
            allow_overlaps: Whether to allow overlapping bounding boxes
        """
        self.MIN_BBOX_AREA = min_bbox_area
        self.allow_overlaps = allow_overlaps

    def check_annotation(self, annotation: Dict, image_path: Optional[str] = None) -> List[ValidationIssue]:
        """
        Check a single annotation for quality issues.

        Args:
            annotation: Annotation dictionary with "rooms" and "image_size"
            image_path: Optional path to image file (for verification)

        Returns:
            List of ValidationIssue objects
        """
        issues = []

        # Check image metadata
        image_size = annotation.get("image_size", {})
        width = image_size.get("width")
        height = image_size.get("height")

        if not width or not height:
            issues.append(ValidationIssue(
                severity="error",
                code="MISSING_IMAGE_SIZE",
                message="Image dimensions not specified"
            ))
            # Can't validate bboxes without image size
            return issues

        # Check room annotations
        rooms = annotation.get("rooms", [])

        if not rooms:
            issues.append(ValidationIssue(
                severity="warning",
                code="NO_ROOMS_DETECTED",
                message="No rooms detected in image"
            ))
            return issues

        # Check each room
        for i, room in enumerate(rooms):
            room_issues = self._check_room(room, i, width, height)
            issues.extend(room_issues)

        # Check for overlaps
        if not self.allow_overlaps:
            overlap_issues = self._check_overlaps(rooms)
            issues.extend(overlap_issues)

        return issues

    def _check_room(self, room: Dict, index: int, img_width: int, img_height: int) -> List[ValidationIssue]:
        """Check a single room annotation."""
        issues = []

        # Check bbox existence
        bbox = room.get("bbox")
        if not bbox:
            issues.append(ValidationIssue(
                severity="error",
                code="MISSING_BBOX",
                message="Bounding box missing",
                room_index=index
            ))
            return issues

        # Check bbox format
        if len(bbox) != 4:
            issues.append(ValidationIssue(
                severity="error",
                code="INVALID_BBOX_FORMAT",
                message=f"Bbox format invalid: expected 4 values, got {len(bbox)}",
                room_index=index
            ))
            return issues

        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except (ValueError, TypeError):
            issues.append(ValidationIssue(
                severity="error",
                code="BBOX_NOT_NUMERIC",
                message="Bbox values are not numeric",
                room_index=index
            ))
            return issues

        # Check bbox ordering
        if x1 >= x2 or y1 >= y2:
            issues.append(ValidationIssue(
                severity="error",
                code="INVALID_BBOX_COORDINATES",
                message=f"Invalid bbox: x1 >= x2 or y1 >= y2",
                room_index=index
            ))
            return issues

        # Check bounds
        if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
            issues.append(ValidationIssue(
                severity="warning",
                code="BBOX_OUT_OF_BOUNDS",
                message=f"Bbox extends outside image bounds ({img_width}x{img_height})",
                room_index=index
            ))

        # Check minimum area
        area = (x2 - x1) * (y2 - y1)
        if area < self.MIN_BBOX_AREA:
            issues.append(ValidationIssue(
                severity="warning",
                code="BBOX_TOO_SMALL",
                message=f"Bbox area {int(area)} < {self.MIN_BBOX_AREA}",
                room_index=index
            ))

        # Check room type
        room_type = room.get("room_type")
        if not room_type:
            issues.append(ValidationIssue(
                severity="error",
                code="MISSING_ROOM_TYPE",
                message="Room type label missing",
                room_index=index
            ))
        elif room_type not in self.VALID_ROOM_TYPES:
            issues.append(ValidationIssue(
                severity="error",
                code="INVALID_ROOM_TYPE",
                message=f"Unknown room type: {room_type}",
                room_index=index
            ))

        # Optional: Check confidence score
        confidence = room.get("confidence")
        if confidence is not None:
            if not isinstance(confidence, (int, float)):
                issues.append(ValidationIssue(
                    severity="warning",
                    code="INVALID_CONFIDENCE_TYPE",
                    message="Confidence score is not numeric",
                    room_index=index
                ))
            elif not (0.0 <= confidence <= 1.0):
                issues.append(ValidationIssue(
                    severity="warning",
                    code="CONFIDENCE_OUT_OF_RANGE",
                    message=f"Confidence {confidence} not in [0.0, 1.0]",
                    room_index=index
                ))

        return issues

    def _check_overlaps(self, rooms: List[Dict]) -> List[ValidationIssue]:
        """Check for overlapping bounding boxes."""
        issues = []

        for i, room_i in enumerate(rooms):
            bbox_i = room_i.get("bbox")
            if not bbox_i or len(bbox_i) != 4:
                continue

            for j, room_j in enumerate(rooms):
                if i >= j:
                    continue

                bbox_j = room_j.get("bbox")
                if not bbox_j or len(bbox_j) != 4:
                    continue

                if self._boxes_overlap(bbox_i, bbox_j):
                    issues.append(ValidationIssue(
                        severity="warning",
                        code="BBOX_OVERLAP",
                        message=f"Overlapping bboxes detected",
                        affected_rooms=[i, j]
                    ))

        return issues

    @staticmethod
    def _boxes_overlap(box1: List, box2: List) -> bool:
        """Check if two bounding boxes overlap."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Boxes don't overlap if separated on either axis
        if x2_1 <= x1_2 or x2_2 <= x1_1:
            return False
        if y2_1 <= y1_2 or y2_2 <= y1_1:
            return False

        return True

    def check_batch(self, annotations: List[Dict]) -> Dict[str, List[ValidationIssue]]:
        """
        Check multiple annotations.

        Returns:
            Dictionary mapping annotation index to list of issues
        """
        results = {}
        for i, annotation in enumerate(annotations):
            issues = self.check_annotation(annotation)
            if issues:
                results[str(i)] = issues

        return results

    def get_summary(self, issues: List[ValidationIssue]) -> Dict[str, int]:
        """
        Get summary counts of issue types.

        Args:
            issues: List of validation issues

        Returns:
            Dictionary with counts by severity and code
        """
        summary = {
            "total": len(issues),
            "by_severity": {},
            "by_code": {}
        }

        for issue in issues:
            # Count by severity
            summary["by_severity"][issue.severity] = \
                summary["by_severity"].get(issue.severity, 0) + 1

            # Count by code
            summary["by_code"][issue.code] = \
                summary["by_code"].get(issue.code, 0) + 1

        return summary

    def report(self, issues: List[ValidationIssue]) -> str:
        """
        Generate a human-readable report of issues.

        Args:
            issues: List of validation issues

        Returns:
            Formatted report string
        """
        if not issues:
            return "✅ No issues found"

        report_lines = [f"Found {len(issues)} issues:\n"]

        # Group by severity
        by_severity = {}
        for issue in issues:
            if issue.severity not in by_severity:
                by_severity[issue.severity] = []
            by_severity[issue.severity].append(issue)

        # Print by severity (errors first, then warnings)
        for severity in ["error", "warning", "info"]:
            if severity not in by_severity:
                continue

            issues_for_severity = by_severity[severity]
            report_lines.append(f"{severity.upper()} ({len(issues_for_severity)}):")

            for issue in issues_for_severity:
                location = ""
                if issue.room_index is not None:
                    location = f" [room {issue.room_index}]"
                elif issue.affected_rooms:
                    location = f" [rooms {issue.affected_rooms}]"

                report_lines.append(f"  - {issue.code}{location}: {issue.message}")

            report_lines.append("")

        return "\n".join(report_lines)
