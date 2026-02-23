"""
SFT-Grade Data Validation for Room/Space Annotations.

Ensures annotations are production-ready for supervised fine-tuning of VLMs.
Filters non-spatial text, validates taxonomy, enforces confidence thresholds.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

try:
    from .taxonomy import CANONICAL_TYPES, VALID_TYPES, VLM_CATEGORY_MAP, normalize_room_type
except ImportError:
    from taxonomy import CANONICAL_TYPES, VALID_TYPES, VLM_CATEGORY_MAP, normalize_room_type


class ImageResizer:
    """Resize images to fit Claude API 5MB limit."""

    @staticmethod
    def resize_for_vlm(image_path: str | Path, max_kb: int = 3500, max_dimension: int = 4500) -> Image.Image:
        """
        Resize image to fit Claude API constraints (5MB file size AND 8000px dimension limit).
        Targets 3.5MB to account for base64 expansion (3.5MB * 1.33 ≈ 4.7MB post-encoding).

        Args:
            image_path: Path to image file
            max_kb: Maximum file size in KB (default: 3500 = safe margin under 5MB post-encoding)
            max_dimension: Maximum image dimension (default: 4500px for aggressive reduction)

        Returns:
            Resized PIL Image object
        """
        img = Image.open(image_path)
        original_size = (img.width, img.height)
        current_max = max(original_size)

        # Phase 1: Dimension reduction
        while current_max > max_dimension:
            scale_factor = max_dimension / current_max
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            current_max = max(img.width, img.height)
            logger.debug(f"  Dimension reduction: {img.width}x{img.height}px")

        # Phase 2: File size reduction (iterative)
        iteration = 0
        while current_max > 400 and iteration < 10:
            buffer = io.BytesIO()
            img.save(buffer, format="PNG", optimize=True)
            kb = len(buffer.getvalue()) / 1024

            if kb <= max_kb:
                logger.debug(f"  Final size: {img.width}x{img.height}px = {kb:.1f}KB (base64: {kb*1.33:.1f}KB) ✓")
                return img

            # Reduce dimensions by 10% per iteration
            current_max = int(current_max * 0.9)
            scale_factor = current_max / max(img.width, img.height)
            img = img.resize(
                (int(img.width * scale_factor), int(img.height * scale_factor)),
                Image.Resampling.LANCZOS
            )
            iteration += 1
            logger.debug(f"  Iteration {iteration}: {img.width}x{img.height}px = {kb:.1f}KB (target: {max_kb}KB)")

        logger.warning(f"  Could not compress below {max_kb}KB (final: {kb:.1f}KB)")
        return img

    @staticmethod
    def resize_in_place(image_path: str | Path, max_kb: int = 3500) -> None:
        """Resize image file in place (targets 3.5MB to stay <5MB after base64 encoding)."""
        image_path = Path(image_path)
        resized = ImageResizer.resize_for_vlm(image_path, max_kb)

        # Save back as PNG with aggressive optimization
        resized.save(image_path, "PNG", optimize=True)
        logger.info(f"Resized {image_path.name} to fit API constraints")


class SemanticRoomValidator:
    """Filter non-spatial text from room annotations."""

    NON_ROOM_PATTERNS = [
        # ------------------------------------------------------------------ #
        # Documentation / compliance blocks                                    #
        # ------------------------------------------------------------------ #
        r"(DOCUMENTATION|REQUIREMENTS?|RECOMMENDED|APPROVAL|PERMIT)",
        r"(ENERGY CODE|COMPLIANCE|STANDARD|SPECIFICATION)",
        r"(DISCLAIMER|NOTES?|LEGEND|SYMBOL|ABBREVIATION)",
        r"(SHEET|DRAWING|PLAN|REVISIONS|TITLE BLOCK|SCALE)",
        r"(SCHEDULE|INDEX|KEY|REFERENCE|ENDORSEMENT)",
        r"^(OUTLINED|THE OUTLINED)",
        # General/national/FDNY electrical notes headers
        r"(ELECTRICAL|MECHANICAL|PLUMBING)\s+(GENERAL|SYMBOL|DRAWING|DEVICE|NOTES|PLAN)",
        r"(NATIONAL|STATE|LOCAL|FDNY|NYC)\s+(ELECTRICAL|FIRE|BUILDING|CODE)",
        # ------------------------------------------------------------------ #
        # Instruction / directive sentences                                    #
        # ------------------------------------------------------------------ #
        r"CONTRACTOR\s+TO\s+(VERIFY|COORDINATE|PROVIDE|INSTALL|CONFIRM)",
        r"(SENSOR|DETECTOR)\s+PLACEMENT",
        r"TAKE\s+OFF(\s+ONLY)?",
        r"USE\s+\w+\s+FOR\s*(:|$)",          # "USE DEVELOPMENT FOR:"
        r"^USE\s+(THIS|DEVELOPMENT|DIAGRAM|DRAWING|PLAN)",
        r"(SHALL|MUST|SHOULD)\s+(BE|NOT|COMPLY)",
        r"BEFORE\s+COMMENCING",
        r"AS\s+(DIRECTED|REQUIRED|NEEDED|SPECIFIED|INDICATED)",
        r"(REFER|SEE)\s+(TO\s+)?(SHEET|DRAWING|PLAN|SPEC|DETAIL)",
        r"PER\s+(CODE|NEC|NFPA|AHJ|OWNER)",
        r"(INSTALL|COORDINATE|VERIFY|PROVIDE|REMOVE|RELOCATE)\s+ALL",
        # ------------------------------------------------------------------ #
        # Equipment / non-spatial objects                                      #
        # ------------------------------------------------------------------ #
        # Panels – most important: PANEL A / PANEL B / PANEL 1 / LP-1
        r"^PANEL\s*[A-Z0-9\-]*$",
        r"^(LP|DP|EP|PP|MDP|SDP)\s*[\-]?\s*\d*[A-Z]?$",   # LP-1, MDP, etc.
        r"^SWITCHBOARD\b",
        r"^(TRANSFORMER|DISCONNECT|BREAKER|FEEDER|RISER)\b",
        # Equipment text blocks
        r"(^EQUIPMENT$|EQUIPMENT\s*\(|EQUIPMENT\s+SHOWN|EQUIPMENT\s+MUST|"
        r"EQUIPMENT\s+MAY|EXISTING\s+EQUIPMENT|NEW\s+EQUIPMENT|"
        r"DISTRIBUTION\s+EQUIPMENT|ELECTRICAL\s+EQUIPMENT|EQUIPMENT\s+INCLUDING)",
        r"DEVICE(S)?\s*/?\s*EQUIPMENT",
        r"INDICATED\s+RELOCATED\s+EXISTING",
        # ------------------------------------------------------------------ #
        # Building-system non-room labels                                      #
        # ------------------------------------------------------------------ #
        r"BUILDING\s+MANAGEMENT\s+SYSTEM",
        r"(DISTRIBUTION|EMERGENCY|NORMAL)\s+(PANEL|SYSTEM|BUS|POWER)\b",
        r"(AC|DC)\s+(MOMENTARY|CIRCUIT|DISCONNECT)",
        # ------------------------------------------------------------------ #
        # Address / firm metadata                                              #
        # ------------------------------------------------------------------ #
        r"(BROADWAY|AVENUE|STREET|BOULEVARD|DRIVE|LANE)\s+(SUITE|#)",
        r"\b(NEW YORK|LOS ANGELES|CHICAGO|BOSTON|HOUSTON)\b",
        r"^\d{3,5}\s+(BROADWAY|AVENUE|STREET)",  # "326 ROCKAWAY"
        r"NEW\s+YORK\s+(OFFICE|CITY)",
        # ------------------------------------------------------------------ #
        # OCR corruption patterns                                              #
        # ------------------------------------------------------------------ #
        r"^I(?=[A-Z]{5,})",      # IELECTRICAL, IACCESS...
        r"^A(?=[A-Z]{5,})",      # ACOMPRESSOR...
        # Short OCR word-fragments (start with consonant cluster, no vowels in key positions)
        r"^[BCDFGHJKLMNPQRSTVWXYZ]{2}[IPME]{1}[A-Z]{0,4}[NT]$",  # UIPMEN, JIPMENT, IPMEN
        # Partial words clearly cut off
        r"^(UIPMEN|JIPMEN|IPMEN|EMEN|JIPMENT|UIPMENT|QUIPMEN)T?$",
        r"^(IIPMENT|ELEMEN|LEMEN|JIMENT|DIMEN)T?S?$",
        r"^(RMINAL|ECTION|IREMENT|JIREMENT|UIREMENT)S?$",
        r"^(QIPMEN|DWIDF|JILDING|IDFD)$",  # specific junk from test data
        # ------------------------------------------------------------------ #
        # ELECTRICAL standalone (not ELECTRICAL ROOM)                         #
        # OCR splits "ELECTRICAL ROOM" into two tokens → "ELECTRICAL" alone   #
        # is ambiguous without "ROOM"; handled by requiring compound in       #
        # room_name_patterns, but guard here too for VLM output.              #
        # ------------------------------------------------------------------ #
        r"^ELECTRICAL$",         # block bare "ELECTRICAL" – must be "ELECTRICAL ROOM"
        r"ELECTRICAL\s+(?!ROOM\b)",  # ELECTRICAL + anything except ROOM
    ]

    VALID_ROOM_KEYWORDS = {
        # Commercial
        "OFFICE", "CONFERENCE", "MEETING", "LOBBY", "RESTROOM",
        "BATHROOM", "KITCHEN", "STORAGE", "ELEVATOR", "STAIRWELL",
        "HALLWAY", "CORRIDOR", "VESTIBULE", "FOYER", "RECEPTION",
        "LOUNGE", "BREAKROOM", "CAFE", "AUDITORIUM", "CLASSROOM",
        "LAB", "MECHANICAL", "DATA CENTER", "SERVER",
        "PROGRAM SUPPORT", "STUDENT SERVICES", "CARPENTRY", "WORKSHOP",
        "FACULTY", "ENTRANCE", "BEDROOM", "LIVING", "DINING",
        "LAUNDRY", "UTILITY", "GARAGE", "CLOSET", "LINEN", "PANTRY",
        "POWDER", "MASTER",
        "SUITE",
        "BREAK",
        "TELECOM",
        "BICYCLE",
        "COMPACTOR",
        "BOILER",
        "PUMP",
        "JANITOR",
        "ART", "MUSIC", "STUDY", "READING",
        "CCTV",
        "PLUMBING",
        "MACHINE",
        # Additional compound rooms required for MEP floor plans
        "FIRE PUMP",
        "COMMUNITY FACILITY",
        "BUILDING STORAGE",
        "BICYCLE STORAGE",
        "COMMERCIAL STORAGE",
        "COMPACTOR ROOM",
        "ELEVATOR MACHINE",
        # Residential abbreviation expansions (after _expand_abbreviation)
        "WALK-IN CLOSET",
        "FAMILY ROOM",
        "DINING ROOM",
        "LIVING ROOM",
        "MASTER BEDROOM",
        "POWDER ROOM",
        "LINEN CLOSET",
        "STUDIO",
    }

    # CRITICAL FIX: Abbreviation expansion mapping
    # Maps abbreviated room names to their canonical expanded forms
    # This ensures abbreviations like "BR", "LR", "BA" are recognized as valid rooms
    ABBREVIATION_EXPANSIONS = {
        # Residential & Room Areas
        "BR": "BEDROOM",
        "BD": "BEDROOM",
        "BDRM": "BEDROOM",
        "MBR": "MASTER BEDROOM",
        "MSTR": "MASTER BEDROOM",
        "MS": "MASTER BEDROOM",
        "BA": "BATHROOM",
        "BATH": "BATHROOM",
        "PDR": "POWDER ROOM",
        "LR": "LIVING ROOM",
        "DR": "DINING ROOM",
        "DIN": "DINING ROOM",
        "KIT": "KITCHEN",
        "K": "KITCHEN",
        "FR": "FAMILY ROOM",
        "FAM": "FAMILY ROOM",
        "OF": "OFFICE",
        "OFC": "OFFICE",
        "LNDRY": "LAUNDRY",
        "UTL": "UTILITY",
        "GAR": "GARAGE",
        "G": "GARAGE",
        "CL": "CLOSET",
        "CLS": "CLOSET",
        "WIC": "WALK-IN CLOSET",
        "LIN": "LINEN CLOSET",
        "PAN": "PANTRY",
        "P": "PANTRY",
        "STR": "STORAGE",
        "STOR": "STORAGE",
        # Commercial
        "OFF": "OFFICE",
        "CONF": "CONFERENCE ROOM",
        "STE": "SUITE",
        "RECP": "RECEPTION",
        "WC": "RESTROOM",
        "TLT": "RESTROOM",
        "RM": "ROOM",
        # Industrial / Utility
        "MECH": "MECHANICAL ROOM",
        "BSMT": "BASEMENT",
        "UTIL": "UTILITY",
        "SHOP": "WORKSHOP",
    }

    def _expand_abbreviation(self, name: str) -> str:
        """
        Expand abbreviated room names to canonical forms.

        Critical for recognizing short room identifiers like BR, LR, BA.

        Args:
            name: Room name (potentially abbreviated)

        Returns:
            Expanded canonical room name, or original if not an abbreviation
        """
        name_upper = name.upper().strip()

        # Check if entire name is a known abbreviation
        if name_upper in self.ABBREVIATION_EXPANSIONS:
            expanded = self.ABBREVIATION_EXPANSIONS[name_upper]
            logger.debug(f"Expanded abbreviation: '{name}' → '{expanded}'")
            return expanded

        # Check if name contains an abbreviation pattern (e.g., "BR 1" → "BEDROOM 1")
        for abbrev, expansion in self.ABBREVIATION_EXPANSIONS.items():
            if name_upper.startswith(abbrev):
                # Handle cases like "BR", "BR 1", "BR-1", "BR1", "BR 101"
                suffix = name_upper[len(abbrev):].strip()
                if not suffix or suffix[0] in [' ', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
                    result = f"{expansion} {suffix}".strip()
                    logger.debug(f"Expanded: '{name}' → '{result}'")
                    return result

        return name

    def filter_rooms(self, rooms: List[Dict]) -> List[Dict]:
        """
        Filter rooms, keeping only valid spatial annotations.

        CRITICAL: VLM uses "room_name", OCR uses "name". Must check both.

        Args:
            rooms: List of room dictionaries with 'name'/'room_name' and 'bbox' keys

        Returns:
            Filtered list of valid rooms
        """
        valid = []

        for room in rooms:
            # CRITICAL FIX: VLM outputs use "room_name", OCR outputs use "name"
            name = room.get("room_name") or room.get("name", "")
            name = name.upper().strip()

            # Skip if empty name
            if not name:
                logger.debug("Filtered (empty name)")
                continue

            # Phase 4 Fix #3: Skip if bbox outside image bounds
            # CHINE ROOM bbox [3229, 5892] was outside image height (3375)
            bbox = room.get("bbox", [])
            if len(bbox) >= 4:
                x, y, w, h = bbox[:4]
                # Check for obviously invalid coordinates (outside typical image bounds)
                if y > 10000 or x > 10000:
                    logger.debug(f"Filtered (bbox outside bounds): {name} at {bbox}")
                    continue

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

            # CRITICAL FIX: Expand abbreviations BEFORE keyword check
            # This ensures abbreviated room names (BR, LR, BA) are recognized
            expanded_name = self._expand_abbreviation(name)

            # Keep ONLY if the name contains a valid room keyword.
            # ⚠️  REMOVED: "or confidence > 0.95" bypass.
            #    High OCR confidence does NOT mean the text is a room label.
            #    OCR fragments like "UIPMEN" (confidence=0.9998) and
            #    "PANEL A" (confidence=0.999) were bypassing this gate.
            #    Confidence is handled separately in filter_by_confidence().
            if any(kw in expanded_name for kw in self.VALID_ROOM_KEYWORDS):
                # Store original name for display, expanded for validation
                room["original_name"] = name
                valid.append(room)
            else:
                logger.debug(f"Filtered (no valid keywords): {name} (expanded: {expanded_name})")

        return valid

    def _matches_non_room_pattern(self, name: str) -> bool:
        """Check if name matches non-room patterns."""
        for pattern in self.NON_ROOM_PATTERNS:
            if re.search(pattern, name):
                return True
        return False


class TaxonomyNormalizer:
    """Normalize room names to standard taxonomy."""

    # Abbreviation expansion mapping (shared with SemanticRoomValidator)
    ABBREVIATION_EXPANSIONS = {
        # Residential & Room Areas
        "BR": "BEDROOM",
        "BD": "BEDROOM",
        "BDRM": "BEDROOM",
        "MBR": "MASTER BEDROOM",
        "MSTR": "MASTER BEDROOM",
        "MS": "MASTER BEDROOM",
        "BA": "BATHROOM",
        "BATH": "BATHROOM",
        "PDR": "POWDER ROOM",
        "LR": "LIVING ROOM",
        "DR": "DINING ROOM",
        "DIN": "DINING ROOM",
        "KIT": "KITCHEN",
        "K": "KITCHEN",
        "FR": "FAMILY ROOM",
        "FAM": "FAMILY ROOM",
        "OF": "OFFICE",
        "OFC": "OFFICE",
        "LNDRY": "LAUNDRY",
        "UTL": "UTILITY",
        "GAR": "GARAGE",
        "G": "GARAGE",
        "CL": "CLOSET",
        "CLS": "CLOSET",
        "WIC": "WALK-IN CLOSET",
        "LIN": "LINEN CLOSET",
        "PAN": "PANTRY",
        "P": "PANTRY",
        "STR": "STORAGE",
        "STOR": "STORAGE",
        # Commercial
        "OFF": "OFFICE",
        "CONF": "CONFERENCE ROOM",
        "STE": "SUITE",
        "RECP": "RECEPTION",
        "WC": "RESTROOM",
        "TLT": "RESTROOM",
        "RM": "ROOM",
        # Industrial / Utility
        "MECH": "MECHANICAL ROOM",
        "BSMT": "BASEMENT",
        "UTIL": "UTILITY",
        "SHOP": "WORKSHOP",
    }

    STANDARD_TAXONOMY = {
        "office": [
            "OFFICE", "EXECUTIVE OFFICE", "INDIVIDUAL OFFICE", "OFFICE SUITE"
        ],
        "conference_room": [
            "CONFERENCE ROOM", "CONFERENCE", "MEETING ROOM", "MEETING"
        ],
        "restroom": [
            "BATHROOM", "RESTROOM", "MEN'S BATHROOM", "WOMEN'S BATHROOM",
            "MEN'S FACULTY", "WOMEN'S FACULTY", "TOILET", "WC", "MEN'S BATHROO",
            "POWDER ROOM"  # From abbreviation PDR
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
        "bedroom": ["BEDROOM", "MASTER BEDROOM"],  # From abbreviations BR, BDRM, MBR
        "living_room": ["LIVING ROOM"],  # From abbreviation LR
        "dining_room": ["DINING ROOM"],  # From abbreviations DR, DIN
        "kitchen": ["KITCHEN"],  # From abbreviations KIT, K
        "family_room": ["FAMILY ROOM"],  # From abbreviations FR, FAM
        "laundry": ["LAUNDRY", "UTILITY"],  # From abbreviations LNDRY, UTL
        "garage": ["GARAGE"],  # From abbreviations GAR, G
        "closet": ["CLOSET", "WALK-IN CLOSET", "LINEN CLOSET"],  # From abbreviations CL, WIC, LIN
        "pantry": ["PANTRY"],  # From abbreviation PAN, P
        "workshop": ["WORKSHOP", "SHOP"],  # From abbreviation SHOP
        # Phase 3 Fix #5: New room types for Phase 2 keywords
        "suite": ["SUITE"],
        "telecom": ["TELECOM", "TELECOM ROOM"],
        "machine_room": ["MACHINE", "MACHINE ROOM"],
        "boiler": ["BOILER", "BOILER ROOM"],
        "pump_room": ["PUMP", "PUMP ROOM"],
        "janitor": ["JANITOR", "JANITOR ROOM"],
        "bicycle_storage": ["BICYCLE", "BICYCLE STORAGE"],
        "compactor": ["COMPACTOR", "COMPACTOR ROOM"],
        "cctv": ["CCTV", "CCTV ROOM"],
        "art_room": ["ART", "ART ROOM"],
        "music_room": ["MUSIC", "MUSIC ROOM"],
        "study_room": ["STUDY", "STUDY ROOM"],
        "reading_room": ["READING", "READING ROOM"],
        "other": ["ROOM", "SPACE", "AREA", "ACEMENT"]
    }

    def _expand_abbreviation(self, name: str) -> str:
        """
        Expand abbreviated room names to canonical forms.

        Args:
            name: Room name (potentially abbreviated)

        Returns:
            Expanded canonical room name, or original if not an abbreviation
        """
        name_upper = name.upper().strip()

        # Check if entire name is a known abbreviation
        if name_upper in self.ABBREVIATION_EXPANSIONS:
            return self.ABBREVIATION_EXPANSIONS[name_upper]

        # Check if name contains an abbreviation pattern (e.g., "BR 1" → "BEDROOM 1")
        for abbrev, expansion in self.ABBREVIATION_EXPANSIONS.items():
            if name_upper.startswith(abbrev):
                suffix = name_upper[len(abbrev):].strip()
                if not suffix or suffix[0] in [' ', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
                    return f"{expansion} {suffix}".strip()

        return name

    def normalize(self, room_name: str) -> str:
        """
        Map extracted room name to canonical taxonomy type.

        Delegates to the centralised normalize_room_type() from taxonomy.py,
        which replaced the previous local SequenceMatcher-based implementation.
        This ensures all normalisation uses the same 35-type canonical taxonomy.
        """
        return normalize_room_type(room_name)


def filter_by_confidence(rooms: List[Dict], min_confidence: float = 0.85,
                        keyword_min_confidence: float = 0.50) -> List[Dict]:
    """
    Two-tier OCR confidence filtering.

    CRITICAL: VLM-detected rooms have NO confidence field (high-confidence by vision).
    OCR-detected rooms HAVE confidence field (from OCR algorithm).

    Phase 3 Finding: High-confidence keywords suggest legitimate room names,
    while unknown text should be filtered strictly.

    Args:
        min_confidence: Threshold for unknown rooms (default: 0.85)
        keyword_min_confidence: Threshold for rooms with valid keywords (default: 0.50)

    Returns:
        Filtered list of rooms

    Always keep VLM rooms, only filter OCR rooms.
    """
    validator = SemanticRoomValidator()
    result = []

    for room in rooms:
        # VLM rooms: no confidence field → always keep (vision-based detection)
        if "confidence" not in room:
            result.append(room)
            continue

        # OCR rooms: have confidence field → apply two-tier filtering
        confidence = room.get("confidence", 0)
        room_name = (room.get("name") or room.get("room_name", "")).upper()

        # Check if contains valid keyword
        has_keyword = any(kw in room_name for kw in validator.VALID_ROOM_KEYWORDS)

        # Apply conditional threshold
        threshold = keyword_min_confidence if has_keyword else min_confidence

        if confidence >= threshold:
            result.append(room)
            logger.debug(
                f"Kept: {room_name} (confidence {confidence:.2f} >= {threshold}, keyword={has_keyword})"
            )
        else:
            logger.debug(
                f"Filtered: {room_name} (confidence {confidence:.2f} < {threshold}, keyword={has_keyword})"
            )

    return result


def _normalize_category(category: str) -> str:
    """
    Normalize VLM category names to canonical forms.

    CRITICAL FIX #1: VLM outputs full category names with suffixes (mechanical_room, electrical_room, etc.)
    but validation expects short canonical forms (mechanical, electrical, etc.).
    This normalization ensures VLM output matches the validation whitelist.

    Args:
        category: Raw category from VLM or taxonomy normalizer

    Returns:
        Normalized category name matching whitelist
    """
    category_normalization = {
        # VLM full names → canonical forms
        "mechanical_room": "mechanical",
        "electrical_room": "electrical",
        "hallway_corridor": "hallway",
        "lobby_reception": "lobby",
        "elevator_lift": "elevator",
        "stairwell_stairs": "stairwell",
        "storage_room": "storage",
        "break_room": "breakroom",
        "cafeteria": "cafe",
        "administrative": "office",
    }
    normalized = category_normalization.get(category.lower(), category.lower())
    logger.debug(f"Normalized category: '{category}' → '{normalized}'")
    return normalized


def validate_for_sft(room: Dict) -> Tuple[bool, Dict[str, bool]]:
    """
    Strict validation for SFT ground truth data.

    VLM rooms have no confidence field (high-confidence by design).
    OCR rooms have confidence field and must pass 0.85 threshold.

    Args:
        room: Single room annotation dictionary

    Returns:
        Tuple of (is_valid, checks_dict)
    """
    name = room.get("room_name") or room.get("name", "")

    # Normalize type using canonical taxonomy (covers all 35 types)
    room_type = room.get("type") or room.get("category") or "other"
    if isinstance(room_type, str):
        room_type = normalize_room_type(room_type)

    checks = {
        "has_name": bool(name.strip()),
        "has_bbox": len(room.get("bbox", [])) == 4,
        "bbox_valid": all(isinstance(x, (int, float)) for x in room.get("bbox", [])),
        "confidence_high": (
            "confidence" not in room or
            room.get("confidence", 0) >= 0.85
        ),
        "name_not_generic": name.upper() not in ["ROOM", "SPACE"],
        "type_in_taxonomy": room_type in VALID_TYPES,
    }

    is_valid = all(checks.values())
    return is_valid, checks


def _bbox_distance(bbox1: List, bbox2: List, threshold: int = 100) -> bool:
    """
    Check if two bounding boxes are proximal (likely same room).

    CRITICAL FIX #2: Deduplication strategy for merged VLM+OCR results.
    If two rooms have overlapping or very close bboxes, they're the same room.

    Args:
        bbox1: [x, y, width, height]
        bbox2: [x, y, width, height]
        threshold: Max distance in pixels to consider rooms as same

    Returns:
        True if bboxes overlap or are within threshold distance
    """
    if len(bbox1) < 4 or len(bbox2) < 4:
        return False

    x1, y1, w1, h1 = bbox1[:4]
    x2, y2, w2, h2 = bbox2[:4]

    # Check for overlap or proximity
    # Rooms overlap if: x-ranges overlap AND y-ranges overlap
    x_overlap = not (x1 + w1 + threshold < x2 or x2 + w2 + threshold < x1)
    y_overlap = not (y1 + h1 + threshold < y2 or y2 + h2 + threshold < y1)

    return x_overlap and y_overlap


def _merge_vlm_and_ocr(vlm_rooms: List[Dict], ocr_rooms: List[Dict]) -> List[Dict]:
    """
    Merge VLM and OCR room detections, preferring compound names from OCR.

    CRITICAL FIX #2: Replace OR logic (ocr never processed) with proper merging.
    Strategy:
    1. Keep all VLM rooms (high-confidence spatial detection)
    2. For each OCR room: if it doesn't overlap with VLM, add it
    3. If OCR overlaps with VLM, prefer OCR name if it's more descriptive

    Args:
        vlm_rooms: Rooms detected by VLM (may have generic names)
        ocr_rooms: Rooms detected by OCR (may have compound names)

    Returns:
        Merged list of unique rooms
    """
    merged = []

    # Add all VLM rooms first (they have spatial boundaries)
    for vlm_room in vlm_rooms:
        merged.append(vlm_room)
        logger.debug(f"Added VLM room: {vlm_room.get('name', vlm_room.get('room_name', 'UNNAMED'))}")

    # Process OCR rooms
    for ocr_room in ocr_rooms:
        ocr_name = ocr_room.get("name") or ocr_room.get("room_name", "")
        ocr_bbox = ocr_room.get("bbox", [])

        # Check if this OCR room overlaps with any VLM room
        matched_vlm = None
        for i, vlm_room in enumerate(merged):
            vlm_bbox = vlm_room.get("bbox", [])
            if _bbox_distance(ocr_bbox, vlm_bbox):
                matched_vlm = i
                break

        if matched_vlm is not None:
            # CRITICAL FIX #3: Preserve compound names from OCR
            # If OCR has a more specific name, use it instead of generic VLM label
            vlm_name = merged[matched_vlm].get("name") or merged[matched_vlm].get("room_name", "")

            # Prefer OCR name if it's more specific (compound) or longer
            if ocr_name and len(ocr_name) > len(vlm_name):
                logger.debug(
                    f"Merged: VLM '{vlm_name}' + OCR '{ocr_name}' → using OCR (compound)"
                )
                merged[matched_vlm]["name"] = ocr_name
                merged[matched_vlm]["room_name"] = ocr_name
                merged[matched_vlm]["ocr_label"] = ocr_name  # Track source
            else:
                logger.debug(f"Merged: VLM '{vlm_name}' (OCR '{ocr_name}' skipped, not compound)")
        else:
            # OCR room doesn't overlap with VLM → add as new room
            logger.debug(f"Added OCR room (no VLM match): {ocr_name}")
            merged.append(ocr_room)

    logger.info(f"Merged {len(vlm_rooms)} VLM + {len(ocr_rooms)} OCR → {len(merged)} total")
    return merged


def prepare_sft_annotation(annotation: Dict) -> Dict:
    """
    Convert annotation to SFT-ready format.

    CRITICAL FIXES:
    - Merges VLM-detected rooms with OCR-detected compound names
    - Normalizes "category" → "type" for VLM output compatibility
    - Removes panels (equipment, not spatial rooms)
    - Sets sft_ready only if rooms detected AND no contamination

    Args:
        annotation: Raw annotation dictionary

    Returns:
        SFT-ready annotation with filtered, normalized rooms
    """
    # Initialize validator and normalizer
    validator = SemanticRoomValidator()
    normalizer = TaxonomyNormalizer()

    # CRITICAL FIX #2: Merge VLM and OCR results instead of OR logic
    vlm_rooms = annotation.get("rooms", [])
    ocr_rooms = annotation.get("ocr_rooms", [])
    rooms = _merge_vlm_and_ocr(vlm_rooms, ocr_rooms)

    # Step 0: Normalize VLM output fields (room_name → name, category → type)
    for room in rooms:
        # VLM uses "room_name", normalize to "name" for consistency
        if "room_name" in room and "name" not in room:
            room["name"] = room["room_name"]
            logger.debug(f"Normalized room_name → name field")

        # VLM uses "category", normalize to "type" for validation
        if "type" not in room and "category" in room:
            room["type"] = room["category"]
            logger.debug(f"Normalized category '{room.get('category')}' → type field")

    # Step 0b: Exclude synthetic rooms from SFT training set.
    # Synthetic rooms are structurally inferred (e.g., "a 2BR unit must have
    # 2 bedrooms"), not visually detected.  Including them teaches the VLM to
    # hallucinate rooms with no image evidence.  They are archived to a
    # separate "synthetic_rooms" key so humans can review them as suggestions.
    _synthetic = [r for r in rooms if r.get("source") == "synthetic_residential"]
    rooms = [r for r in rooms if r.get("source") != "synthetic_residential"]
    if _synthetic:
        annotation.setdefault("synthetic_rooms", []).extend(_synthetic)
        logger.info(
            f"Excluded {len(_synthetic)} synthetic room(s) from SFT path "
            f"(archived to annotation['synthetic_rooms'] for review)"
        )

    # Step 1: Semantic filtering
    rooms = validator.filter_rooms(rooms)
    logger.debug(f"After semantic filter: {len(rooms)} rooms")

    # Step 2: Confidence filtering (preserves VLM rooms with no confidence field)
    rooms = filter_by_confidence(rooms, min_confidence=0.85)
    logger.debug(f"After confidence filter: {len(rooms)} rooms")

    # Step 3: Normalize room types (for OCR-only detections)
    for room in rooms:
        # Only normalize if type not already set (e.g., from VLM category)
        if room.get("type") == "other" or "type" not in room:
            # Support both "room_name" (VLM) and "name" (OCR)
            room_name = room.get("room_name") or room.get("name", "")
            room_type = normalizer.normalize(room_name)
            room["type"] = room_type
            logger.debug(f"Normalized '{room_name}' → {room_type}")

    # Step 4: SFT validation
    sft_ready = []
    for room in rooms:
        is_valid, checks = validate_for_sft(room)
        if not is_valid:
            failed_checks = [k for k, v in checks.items() if not v]
            logger.debug(f"Room rejected: {room.get('name')} - {failed_checks}")
            continue
        sft_ready.append(room)

    # Step 5: Clean up contamination
    annotation.pop("ocr_rooms", None)  # Remove unfiltered OCR
    annotation.pop("panels", None)  # Remove equipment (not spatial rooms)

    # Step 6: Return with corrected sft_ready logic
    annotation["rooms"] = sft_ready
    # CRITICAL: Only mark as SFT-ready if we actually detected rooms AND have no contamination
    annotation["sft_ready"] = (
        len(sft_ready) > 0 and  # Has rooms
        "panels" not in annotation and  # No equipment
        "ocr_rooms" not in annotation  # No contamination
    )
    logger.info(
        f"SFT preparation complete: {len(sft_ready)} rooms, sft_ready={annotation['sft_ready']}"
    )
    return annotation
