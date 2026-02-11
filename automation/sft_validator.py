"""
SFT-Grade Data Validation for Room/Space Annotations.

Ensures annotations are production-ready for supervised fine-tuning of VLMs.
Filters non-spatial text, validates taxonomy, enforces confidence thresholds.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)


class ImageResizer:
    """Resize images to fit Claude API 5MB limit."""

    @staticmethod
    def resize_for_vlm(image_path: str | Path, max_kb: int = 3500) -> Image.Image:
        """
        Resize image to fit Claude API 5MB constraint (aim for 3.5MB to be safe).

        Args:
            image_path: Path to image file
            max_kb: Maximum file size in KB (default: 3500 for safety margin)

        Returns:
            Resized PIL Image object
        """
        img = Image.open(image_path)
        original_size = (img.width, img.height)
        max_dimension = max(original_size)  # Use max, not min, for better resizing

        # Aggressive binary search for target file size
        while max_dimension > 512:
            img_resized = img.copy()
            img_resized.thumbnail(
                (max_dimension, max_dimension),
                Image.Resampling.LANCZOS
            )

            buffer = io.BytesIO()
            img_resized.save(buffer, format="PNG", optimize=True, quality=85)
            kb = len(buffer.getvalue()) / 1024

            if kb <= max_kb:
                return img_resized

            # Reduce more aggressively (50% reduction per iteration)
            max_dimension = int(max_dimension * 0.7)

        return img_resized

    @staticmethod
    def resize_in_place(image_path: str | Path, max_kb: int = 4500) -> None:
        """Resize image file in place."""
        resized = ImageResizer.resize_for_vlm(image_path, max_kb)
        resized.save(image_path, "PNG", optimize=True)
        logger.info(f"Resized {Path(image_path).name} to fit API constraints")


class SemanticRoomValidator:
    """Filter non-spatial text from room annotations."""

    NON_ROOM_PATTERNS = [
        # Documentation/Compliance
        r"(DOCUMENTATION|REQUIREMENTS|RECOMMENDED|APPROVAL|PERMIT)",
        r"(ENERGY CODE|COMPLIANCE|STANDARD|SPECIFICATION)",
        r"(DISCLAIMER|NOTES|LEGEND|SYMBOL|ABBREVIATION)",
        # Sheet/Drawing metadata
        r"(SHEET|DRAWING|PLAN|REVISIONS|TITLE BLOCK|SCALE)",
        r"(ELECTRICAL|MECHANICAL|PLUMBING).*(PLAN|FIRST FLOOR|SECOND|BASEMENT|PAGE)",
        # Administrative text
        r"(DOCUMENT|STATEMENT|OUTLINE|MOVEMENT|OVERRIDE|PROVIDE)",
        r"(SCHEDULE|INDEX|KEY|REFERENCE|ENDORSEMENT)",
        r"^(OUTLINED|THE OUTLINED)",
    ]

    VALID_ROOM_KEYWORDS = {
        "OFFICE", "CONFERENCE", "MEETING", "LOBBY", "RESTROOM",
        "BATHROOM", "KITCHEN", "STORAGE", "ELEVATOR", "STAIRWELL",
        "HALLWAY", "CORRIDOR", "VESTIBULE", "FOYER", "RECEPTION",
        "LOUNGE", "BREAKROOM", "CAFE", "AUDITORIUM", "CLASSROOM",
        "LAB", "MECHANICAL", "ELECTRICAL", "DATA CENTER", "SERVER",
        "PROGRAM SUPPORT", "STUDENT SERVICES", "CARPENTRY", "WORKSHOP",
        "FACULTY", "ENTRANCE", "ACEMENT"  # OCR errors
    }

    def filter_rooms(self, rooms: List[Dict]) -> List[Dict]:
        """
        Filter rooms, keeping only valid spatial annotations.

        Args:
            rooms: List of room dictionaries with 'name' and 'bbox' keys

        Returns:
            Filtered list of valid rooms
        """
        valid = []

        for room in rooms:
            name = room.get("name", "").upper().strip()

            # Skip if matches non-room pattern
            if self._matches_non_room_pattern(name):
                logger.debug(f"Filtered (non-room pattern): {name}")
                continue

            # Skip if too small (header/footer text)
            bbox = room.get("bbox", [])
            if len(bbox) >= 4 and bbox[3] < 20:
                logger.debug(f"Filtered (too small): {name}")
                continue

            # Skip if too wide/short (documentation blocks)
            if len(bbox) >= 4 and bbox[2] > bbox[3] * 8:
                logger.debug(f"Filtered (too wide): {name}")
                continue

            # Keep if contains valid room keywords OR has good confidence
            confidence = room.get("confidence", 0)
            if any(kw in name for kw in self.VALID_ROOM_KEYWORDS) or confidence > 0.95:
                valid.append(room)

        return valid

    def _matches_non_room_pattern(self, name: str) -> bool:
        """Check if name matches non-room patterns."""
        for pattern in self.NON_ROOM_PATTERNS:
            if re.search(pattern, name):
                return True
        return False


class TaxonomyNormalizer:
    """Normalize room names to standard taxonomy."""

    STANDARD_TAXONOMY = {
        "office": [
            "OFFICE", "EXECUTIVE OFFICE", "INDIVIDUAL OFFICE", "OFFICE SUITE"
        ],
        "conference_room": [
            "CONFERENCE ROOM", "CONFERENCE", "MEETING ROOM", "MEETING"
        ],
        "restroom": [
            "BATHROOM", "RESTROOM", "MEN'S BATHROOM", "WOMEN'S BATHROOM",
            "MEN'S FACULTY", "WOMEN'S FACULTY", "TOILET", "WC", "MEN'S BATHROO"
        ],
        "storage": ["STORAGE", "STORAGE ROOM", "STORE"],
        "lobby": [
            "LOBBY", "RECEPTION", "ENTRANCE", "FOYER", "RECEPTION AREA",
            "ENTRANCE VESTIBULE"
        ],
        "hallway": [
            "HALLWAY", "CORRIDOR", "PASSAGE", "WALK", "THE OUTLINED CORRIDOR"
        ],
        "elevator": ["ELEVATOR", "LIFT"],
        "stairwell": ["STAIRWELL", "STAIRS", "STAIR", "STAIRCASE"],
        "mechanical": ["MECHANICAL ROOM", "MECHANICAL", "MECH ROOM"],
        "electrical": ["ELECTRICAL ROOM", "ELECTRICAL"],
        "carpentry": ["CARPENTRY", "CARPENTRY SHOP", "WOOD SHOP"],
        "other": ["ROOM", "SPACE", "AREA", "ACEMENT"]
    }

    def normalize(self, room_name: str) -> str:
        """
        Map extracted room name to standard taxonomy.

        Args:
            room_name: Raw room name from extraction

        Returns:
            Standardized room type (lowercase)
        """
        name_upper = room_name.upper().strip()

        # Exact match first
        for std_type, variants in self.STANDARD_TAXONOMY.items():
            if name_upper in variants:
                return std_type

        # Fuzzy match for OCR errors
        best_match = None
        best_ratio = 0.0

        all_variants = [
            v for variants in self.STANDARD_TAXONOMY.values() for v in variants
        ]

        for variant in all_variants:
            ratio = SequenceMatcher(None, name_upper, variant).ratio()
            if ratio > best_ratio and ratio > 0.80:
                best_ratio = ratio
                best_match = variant

        if best_match:
            for std_type, variants in self.STANDARD_TAXONOMY.items():
                if best_match in variants:
                    return std_type

        return "other"


def filter_by_confidence(rooms: List[Dict], min_confidence: float = 0.85) -> List[Dict]:
    """Filter rooms by confidence threshold."""
    return [r for r in rooms if r.get("confidence", 0) >= min_confidence]


def validate_for_sft(room: Dict) -> Tuple[bool, Dict[str, bool]]:
    """
    Strict validation for SFT ground truth data.

    Args:
        room: Single room annotation dictionary

    Returns:
        Tuple of (is_valid, checks_dict)
    """
    checks = {
        "has_name": bool(room.get("name", "").strip()),
        "has_bbox": len(room.get("bbox", [])) == 4,
        "bbox_valid": all(isinstance(x, (int, float)) for x in room.get("bbox", [])),
        "confidence_high": room.get("confidence", 0) >= 0.85,
        "name_not_generic": room.get("name", "").upper() not in ["ROOM", "SPACE"],
        "type_in_taxonomy": room.get("type", "other") in [
            "office", "conference_room", "restroom", "storage", "lobby",
            "hallway", "elevator", "stairwell", "mechanical", "electrical",
            "carpentry", "other"
        ],
    }

    is_valid = all(checks.values())
    return is_valid, checks


def prepare_sft_annotation(annotation: Dict) -> Dict:
    """
    Convert annotation to SFT-ready format.

    Args:
        annotation: Raw annotation dictionary

    Returns:
        SFT-ready annotation with filtered, normalized rooms
    """
    # Initialize validator and normalizer
    validator = SemanticRoomValidator()
    normalizer = TaxonomyNormalizer()

    # Get rooms (from either 'rooms' or 'ocr_rooms')
    rooms = annotation.get("rooms", []) or annotation.get("ocr_rooms", [])

    # Step 1: Semantic filtering
    rooms = validator.filter_rooms(rooms)
    logger.debug(f"After semantic filter: {len(rooms)} rooms")

    # Step 2: Confidence filtering
    rooms = filter_by_confidence(rooms, min_confidence=0.85)
    logger.debug(f"After confidence filter: {len(rooms)} rooms")

    # Step 3: Normalize room types
    for room in rooms:
        room_type = normalizer.normalize(room.get("name", ""))
        room["type"] = room_type
        logger.debug(f"Normalized '{room.get('name')}' → {room_type}")

    # Step 4: SFT validation
    sft_ready = []
    for room in rooms:
        is_valid, checks = validate_for_sft(room)
        if not is_valid:
            failed_checks = [k for k, v in checks.items() if not v]
            logger.debug(f"Room rejected: {room.get('name')} - {failed_checks}")
            continue
        sft_ready.append(room)

    # Return updated annotation
    annotation["rooms"] = sft_ready
    annotation["sft_ready"] = True
    return annotation
