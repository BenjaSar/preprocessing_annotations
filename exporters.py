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
    from .config import ExportConfig
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

        # Get image dimensions
        img_w = annotation.get("image_size", {}).get(
            "width", self.config.default_width
        )
        img_h = annotation.get("image_size", {}).get(
            "height", self.config.default_height
        )

        # Build task structure
        task = {
            "data": {"image": f"{image_url_prefix}{images_dir}/{image_file}"},
            "predictions": [{"result": []}],
        }

        result = task["predictions"][0]["result"]

        # Add room annotations
        for i, room in enumerate(annotation.get("rooms", [])):
            bbox = room.get("bbox", [0, 0, 100, 100])

            # Convert to Label Studio percentage format
            result.append(
                {
                    "id": f"room_{i}",
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image",
                    "original_width": img_w,
                    "original_height": img_h,
                    "value": {
                        "x": (bbox[0] / img_w) * 100,
                        "y": (bbox[1] / img_h) * 100,
                        "width": (bbox[2] / img_w) * 100,
                        "height": (bbox[3] / img_h) * 100,
                        "rectanglelabels": [room.get("category", "unknown")],
                    },
                }
            )

            # Add text annotation for room label
            room_label = room.get("room_number", "") or room.get("room_name", "")
            if room_label:
                result.append(
                    {
                        "id": f"room_{i}_text",
                        "type": "textarea",
                        "from_name": "room_label",
                        "to_name": "image",
                        "value": {
                            "x": (bbox[0] / img_w) * 100,
                            "y": (bbox[1] / img_h) * 100,
                            "width": (bbox[2] / img_w) * 100,
                            "height": (bbox[3] / img_h) * 100,
                            "text": [room_label],
                        },
                    }
                )

        # Add panel annotations
        for i, panel in enumerate(annotation.get("panels", [])):
            bbox = panel.get("bbox", [0, 0, 50, 50])

            result.append(
                {
                    "id": f"panel_{i}",
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image",
                    "original_width": img_w,
                    "original_height": img_h,
                    "value": {
                        "x": (bbox[0] / img_w) * 100,
                        "y": (bbox[1] / img_h) * 100,
                        "width": (bbox[2] / img_w) * 100,
                        "height": (bbox[3] / img_h) * 100,
                        "rectanglelabels": ["electrical_panel"],
                    },
                }
            )

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
        Compute confidence score for an annotation.

        Scoring factors:
        - Base score: 0.5
        - Has rooms: +0.2
        - Room has number: +0.1 per room
        - Room has name: +0.1 per room
        - Has error: -0.5

        Returns:
            Tuple of (confidence score, reason string).
        """
        score = 0.5
        reasons = []

        # Check for errors
        if annotation.get("error"):
            return 0.0, f"Error: {annotation['error']}"

        rooms = annotation.get("rooms", [])

        # No rooms detected
        if not rooms:
            return 0.2, "No rooms detected"

        # Has rooms
        score += 0.2
        reasons.append(f"{len(rooms)} rooms")

        # Evaluate room quality
        complete_rooms = 0
        for room in rooms:
            room_score = 0.7

            if room.get("room_number"):
                room_score += 0.15

            if room.get("room_name"):
                room_score += 0.15

            if room_score >= 0.9:
                complete_rooms += 1

        # Bonus for complete rooms
        completeness = complete_rooms / len(rooms) if rooms else 0
        score += completeness * 0.3
        reasons.append(f"{complete_rooms}/{len(rooms)} complete")

        # Cap at 1.0
        score = min(1.0, score)

        return score, ", ".join(reasons)

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
