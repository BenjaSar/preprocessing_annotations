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

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from .taxonomy import (
        MANDATORY_CLASSES, VALID_TYPES, normalize_to_mandatory,
        get_extended_type, strip_window_suffix,
    )
    from .abbreviations import ABBREVIATION_MAP as _ABBREVIATION_MAP
except ImportError:
    from taxonomy import (
        MANDATORY_CLASSES, VALID_TYPES, normalize_to_mandatory,
        get_extended_type, strip_window_suffix,
    )
    from abbreviations import ABBREVIATION_MAP as _ABBREVIATION_MAP


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
        """Resize image to fit API size constraints, writing atomically via a temp file.

        Uses a temp file + rename so the original is never partially overwritten,
        and the source image is not destroyed if a PermissionError or OOM occurs.
        """
        import tempfile
        import os
        image_path = Path(image_path)
        resized = ImageResizer.resize_for_vlm(image_path, max_kb)

        # Write to a temp file in the same directory, then rename atomically.
        # Same-directory placement ensures the rename is on the same filesystem
        # (cross-device renames would fail with OSError on some platforms).
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".png", dir=image_path.parent, prefix=".resize_tmp_"
        )
        try:
            os.close(tmp_fd)
            resized.save(tmp_path, "PNG", optimize=True)
            os.replace(tmp_path, image_path)
            logger.info(f"Resized {image_path.name} to fit API constraints")
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


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
        # Institutional/school rooms — previously dropped by keyword gate
        # (verified: CUSTODIAL x4 reached gate and were filtered). Each maps
        # to a canonical type via taxonomy (TOILET→TOILET, GYM→GYMNASIUM,
        # CUSTODIAL/LOCKER→OTHER, SPECIAL EDUCATION→CLASSROOM, COACH→OFFICE).
        "TOILET", "CUSTODIAL", "LOCKER", "LOCKERS", "GYM", "GYMNASIUM",
        "SPECIAL EDUCATION", "SPECIAL ED", "COACH",
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
        # Residential unit-type expansions (0BR/1BR/2BR → STUDIO / N BEDROOM)
        "1 BEDROOM",
        "2 BEDROOM",
        "3 BEDROOM",
        "4 BEDROOM",
        # Pipeline-output-analysis §4/§5: Keywords present in CANONICAL_TYPES
        # surface forms but previously absent from this whitelist, causing
        # the keyword gate to silently drop valid room labels before
        # taxonomy normalization could map them to canonical types.
        "COMMUNITY",        # §4 Case 1: "COMMUNITY ROOM" → community_facility
        "COMMUNITY ROOM",   # §5 table: direct compound match
        "WAITING",          # §4 Case 3: "WAITING" / "WAITING ROOM" → lobby
        "WAITING ROOM",     # §4 Case 3: compound form
        "REFUSE",           # §4 Case 4: "REFUSE ROOM" → compactor
        "TRASH",            # §5 table:  "TRASH ROOM"  → compactor
        # Residential unit keywords — covers multi-family / mixed-use floor plans.
        # "TYPE" matches architectural unit-type codes like "TYPE-A1 OBR".
        # The word-boundary regex (\bTYPE\b) ensures it doesn't match e.g. "PROTOTYPE".
        "RESIDENTIAL", "APARTMENT", "UNIT", "DWELLING",
        "TYPE",         # architectural unit-type prefix: TYPE-A1, TYPE-B3, etc.
        "STUDIO",
        "OBR",          # 0BR misread by OCR (O/0 confusion); maps to RESIDENTIAL UNIT
        "1BR", "2BR", "3BR", "4BR",
    }

    # Class-level compiled word-boundary pattern built from VALID_ROOM_KEYWORDS.
    # Replaces the previous `any(kw in name ...)` substring check which allowed
    # labels like "BUILDING MANAGEMENT SYSTEM" to pass because they contained
    # "BUILDING" as a substring.  Word-boundary matching ensures keywords only
    # match when they appear as complete words.
    #
    # Keywords are sorted longest-first so multi-word phrases ("FIRE PUMP",
    # "ELEVATOR MACHINE") are tried before their component words ("PUMP",
    # "MACHINE"), preventing partial matches that shadow the full phrase.
    #
    # The pattern is compiled once at class definition time (not per-instance)
    # to avoid repeated re.compile() overhead in tight filter loops.
    _KW_PATTERN: "re.Pattern" = None  # populated after class definition

    # CRITICAL FIX: Abbreviation expansion mapping — single source of truth.
    # Previously this was a separate dict that diverged from ResidentialAbbreviationRecovery
    # and TaxonomyNormalizer.  All three now import from abbreviations.py.
    ABBREVIATION_EXPANSIONS = _ABBREVIATION_MAP

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

            # T-2: OCR-anchored rooms (ocr_only / vlm_ocr_merged) are exempt
            # from the size/aspect checks below. These checks target header/
            # footer text and documentation blocks — real located room labels
            # (e.g. ELEVATOR bbox height=14px) are legitimately small/wide and
            # would otherwise be dropped here before Option A's area-gate
            # exemption ever applies downstream.
            is_ocr_anchored = room.get("source") in ("ocr_only", "vlm_ocr_merged")

            # Skip if too small (header/footer text)
            bbox = room.get("bbox", [])
            if not is_ocr_anchored and len(bbox) >= 4 and bbox[3] < 20:
                logger.debug(f"Filtered (too small): {name}")
                continue

            # Skip if too wide/short (documentation blocks)
            if not is_ocr_anchored and len(bbox) >= 4 and bbox[2] > bbox[3] * 8:
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
            #
            # Word-boundary regex replaces the previous substring check
            # `any(kw in expanded_name ...)` which allowed labels like
            # "BUILDING MANAGEMENT SYSTEM" to pass because they contained
            # "BUILDING" as a substring.
            if SemanticRoomValidator._KW_PATTERN.search(expanded_name):
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


# Build the class-level keyword pattern now that VALID_ROOM_KEYWORDS is defined.
# Sorted longest-first so multi-word phrases ("FIRE PUMP") match before their
# constituent words ("PUMP") when both could apply to the same string.
_sorted_kws = sorted(SemanticRoomValidator.VALID_ROOM_KEYWORDS, key=len, reverse=True)
SemanticRoomValidator._KW_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(kw) for kw in _sorted_kws) + r")\b",
    re.IGNORECASE,
)


class TaxonomyNormalizer:
    """Normalize room names to SFT mandatory taxonomy."""

    # Single source of truth — imported from abbreviations.py.
    # Previously a separate dict that diverged from SemanticRoomValidator
    # and ResidentialAbbreviationRecovery.
    ABBREVIATION_EXPANSIONS = _ABBREVIATION_MAP

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
        Map extracted room name to mandatory SFT taxonomy type.

        Remediation Fix #4: Flag slash-separated compound names for review.
        If a room name contains '/', check if both parts are valid room types.
        If so, flag for human review (may indicate two distinct rooms).

        Delegates to the centralised normalize_to_mandatory() from taxonomy.py,
        which ensures all normalisation uses the mandatory SFT taxonomy.
        """
        # Remediation Fix #4: Detect slash-compound labels
        if "/" in room_name:
            parts = room_name.split("/")
            logger.warning(
                f"Remediation Fix #4: slash-compound label detected: '{room_name}' "
                f"— verify it represents a single room (e.g., IT/STORAGE) and not two rooms"
            )

        return normalize_to_mandatory(room_name)


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

        # Abbreviation-recovered rooms have already been validated by dictionary
        # lookup in ResidentialAbbreviationRecovery.  Applying a confidence
        # threshold here is redundant and risks dropping legitimate rooms whose
        # raw OCR confidence is low (common for small abbreviated text).
        if room.get("source") == "ocr_abbreviation_recovered":
            result.append(room)
            logger.debug(f"Kept (abbreviation exemption): {room_name}")
            continue

        # Word-boundary check replaces the previous `any(kw in room_name ...)`
        # substring test.  See SemanticRoomValidator._KW_PATTERN for rationale.
        has_keyword = bool(SemanticRoomValidator._KW_PATTERN.search(room_name))

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

    # Validate against the base mandatory type.
    # strip_window_suffix() removes any "w/ windows" / "w/ skylights" / "w/ side
    # openings" suffix first — those are visual attributes, not taxonomy entries.
    # normalize_to_mandatory() then resolves the base string to VALID_TYPES.
    # This makes the two-step logic explicit rather than relying on substring
    # matching inside normalize_to_mandatory() to accidentally strip the suffix.
    room_type = room.get("type") or room.get("category") or "STORAGE ROOM"
    if isinstance(room_type, str):
        base_type = strip_window_suffix(room_type)
        room_type = normalize_to_mandatory(base_type)

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


def _bbox_in_bounds(bbox: List, img_w: int, img_h: int) -> bool:
    """
    Check if a bounding box lies entirely within the image bounds.

    (pipeline-revalidation-analysis §3 Fix B)

    Rooms with coordinates exceeding image dimensions are produced when
    OCR or VLM returns pixel coordinates relative to the full-resolution
    PDF page rather than the cropped/resized image used by the pipeline.
    These must be filtered BEFORE sft_ready is computed, otherwise images
    with only OOB rooms are marked sft_ready=True and appear in COCO
    splits as zero-annotation ghost images.

    Args:
        bbox: [x, y, width, height]
        img_w: Image width in pixels
        img_h: Image height in pixels

    Returns:
        True if bbox is entirely within image bounds
    """
    if len(bbox) < 4:
        return False
    x, y, w, h = bbox[:4]
    return x >= 0 and y >= 0 and x + w <= img_w and y + h <= img_h


def _bbox_iou(bbox1: List, bbox2: List) -> float:
    """
    Compute Intersection over Union (IoU) for two bboxes.

    Remediation Fix #2: Detect overlapping room bboxes for resolution.

    Args:
        bbox1: [x, y, width, height]
        bbox2: [x, y, width, height]

    Returns:
        IoU in [0, 1], or 0 if bboxes don't overlap or are malformed.
    """
    if len(bbox1) < 4 or len(bbox2) < 4:
        return 0.0

    x1, y1, w1, h1 = bbox1[:4]
    x2, y2, w2, h2 = bbox2[:4]

    # Convert to (x1, y1, x2, y2) format
    x1_1, y1_1, x2_1, y2_1 = x1, y1, x1 + w1, y1 + h1
    x1_2, y1_2, x2_2, y2_2 = x2, y2, x2 + w2, y2 + h2

    # Compute intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    intersection = (xi2 - xi1) * (yi2 - yi1)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def _resolve_bbox_overlaps(rooms: List[Dict]) -> List[Dict]:
    """
    Resolve overlapping bounding boxes in non-circulation room types.

    Remediation Fix #2: Post-VLM overlap resolution.
    - Circulation types (hallway, elevator, stairwell, lobby, corridor) are exempt
    - For overlapping non-circulation pairs:
      * If IoU > 0.5: merge as duplicate (keep higher confidence, larger bbox)
      * If IoU <= 0.5: shrink smaller bbox to remove overlap (clip boundary)
    - Log all resolutions for human review

    Args:
        rooms: List of room annotations

    Returns:
        List with overlaps resolved
    """
    if not rooms:
        return rooms

    circulation_types = {'hallway', 'corridor', 'lobby', 'elevator', 'stairwell', 'riser'}
    processed = []
    skip_indices = set()

    for i, room_i in enumerate(rooms):
        if i in skip_indices:
            continue

        bbox_i = room_i.get("bbox", [])
        type_i = (room_i.get("type") or room_i.get("category") or "").lower()

        # Skip circulation types
        if type_i in circulation_types or len(bbox_i) != 4:
            processed.append(room_i)
            continue

        # Check for overlaps with subsequent rooms
        merged = False
        for j in range(i + 1, len(rooms)):
            if j in skip_indices:
                continue

            room_j = rooms[j]
            bbox_j = room_j.get("bbox", [])
            type_j = (room_j.get("type") or room_j.get("category") or "").lower()

            # Skip circulation types
            if type_j in circulation_types or len(bbox_j) != 4:
                continue

            iou = _bbox_iou(bbox_i, bbox_j)
            if iou == 0.0:
                continue

            # Overlap detected
            name_i = room_i.get("room_name") or room_i.get("name", "?")
            name_j = room_j.get("room_name") or room_j.get("name", "?")

            if iou > 0.5:
                # Merge: keep higher-confidence room; tie-break on room_i (first seen).
                # Area is not used here — we are not doing spatial measurement at this
                # stage.  The only signal available is the VLM/OCR confidence score.
                conf_i = room_i.get("confidence", 0.5)
                conf_j = room_j.get("confidence", 0.5)

                if conf_i >= conf_j:
                    logger.warning(
                        f"Fix2: merged overlapping rooms (IoU={iou:.2f}): "
                        f"'{name_i}' (conf={conf_i:.2f}) kept, '{name_j}' (conf={conf_j:.2f}) removed"
                    )
                    skip_indices.add(j)
                else:
                    logger.warning(
                        f"Fix2: merged overlapping rooms (IoU={iou:.2f}): "
                        f"'{name_j}' (conf={conf_j:.2f}) kept, '{name_i}' (conf={conf_i:.2f}) removed"
                    )
                    skip_indices.add(i)
                    merged = True
                    break
            else:
                # IoU <= 0.5: partial overlap between distinct adjacent rooms.
                # Previous behaviour clipped the smaller bbox iteratively, which
                # cascaded — each clip made the bbox smaller, triggering more
                # overlaps with neighbours, ultimately collapsing many valid
                # room bboxes to 1×1 px and then eliminating them in the
                # min-area filter.
                #
                # Fix: keep BOTH rooms unchanged.  Partial overlap is expected
                # in dense floorplans where VLM bboxes are not pixel-perfect.
                # Flag for human review but do not mutate any geometry.
                logger.debug(
                    f"Fix2: partial overlap (IoU={iou:.2f}) between "
                    f"'{name_i}' and '{name_j}' — keeping both unchanged "
                    f"(review recommended)"
                )

        if not merged:
            processed.append(room_i)

    return processed


def _dedup_nested_same_type(rooms: List[Dict], contain_frac: float = 0.7) -> List[Dict]:
    """Drop the LARGER of two same-canonical-type boxes when one nearly contains
    the other — a duplicate detection of the same space at two scales.

    Motivation (Phase 2, verified): tiled VLM inference emits the same room at
    drifting sizes. When the pair's IoU is just under 0.5 (e.g. a wide LOBBY
    strip [162,57,2012,625] nesting [162,57,2012,323], IoU 0.47) _resolve_bbox_
    overlaps' 0.5 merge misses it, and its circulation-type exemption skips it
    entirely — so an oversized duplicate survives past the 8% ceiling. Unlike
    adjacency overlap (expected for corridors), CONTAINMENT of a same-type box
    is a true duplicate, so this runs for ALL types incl. circulation. Keeps the
    tighter (smaller) box, which hugs the actual label/room better.

    contain_frac: min fraction of the SMALLER box's area that must lie inside
    the larger to count as "contained" (0.7 = mostly nested, not mere touching).
    """
    def _type(r):
        # Key on the CANONICAL type derived from the immutable room_name, not the
        # raw "category" field — tiled VLM duplicates of the same label often
        # carry different garbage categories (e.g. two "VESTIBULE" boxes tagged
        # "STORAGE" and "GYM OFFICE"), which would otherwise defeat same-type
        # matching. normalize_to_mandatory maps both to LOBBY.
        name = r.get("room_name") or r.get("name") or r.get("type") or r.get("category") or ""
        try:
            return normalize_to_mandatory(name)
        except Exception:
            return name.lower()

    drop = set()
    for i in range(len(rooms)):
        if i in drop:
            continue
        bi = rooms[i].get("bbox", [])
        if len(bi) != 4:
            continue
        for j in range(len(rooms)):
            if j == i or j in drop:
                continue
            bj = rooms[j].get("bbox", [])
            if len(bj) != 4 or _type(rooms[i]) != _type(rooms[j]):
                continue
            ai = (bi[2] - bi[0]) * (bi[3] - bi[1])
            aj = (bj[2] - bj[0]) * (bj[3] - bj[1])
            if ai <= 0 or aj <= 0:
                continue
            # intersection (boxes are xywh here: [x, y, w, h])
            ix1, iy1 = max(bi[0], bj[0]), max(bi[1], bj[1])
            ix2 = min(bi[0] + bi[2], bj[0] + bj[2])
            iy2 = min(bi[1] + bi[3], bj[1] + bj[3])
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            smaller = min(ai, aj)
            if inter / smaller >= contain_frac:
                # Drop the larger of the pair (keep tighter box).
                larger_idx = i if ai >= aj else j
                drop.add(larger_idx)
                rn = rooms[larger_idx].get("room_name") or rooms[larger_idx].get("name", "?")
                logger.warning(
                    f"Fix2b: dropped nested same-type duplicate (larger) '{rn}' "
                    f"(kept tighter box of type '{_type(rooms[larger_idx])}')"
                )
                if larger_idx == i:
                    break
    return [r for k, r in enumerate(rooms) if k not in drop]





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
        r = dict(vlm_room)
        r.setdefault("source", "vlm_only")
        merged.append(r)
        logger.debug(f"Added VLM room: {r.get('name', r.get('room_name', 'UNNAMED'))}")

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
            vlm_name = merged[matched_vlm].get("name") or merged[matched_vlm].get("room_name", "")

            already_claimed = merged[matched_vlm].get("source") == "vlm_ocr_merged"

            if ocr_name and already_claimed:
                # T-1: this VLM box already absorbed an earlier compound OCR
                # label (rename-chain). Overwriting again would silently erase
                # the first winner (measured: TELECOM ROOM claimed a box, then
                # BUILDING STORAGE matched the same box and overwrote it —
                # TELECOM ROOM vanished from the final annotation entirely).
                # Give the second (and any further) claimant its own ocr_only
                # room instead of overwriting the first.
                logger.debug(
                    f"T-1: VLM box already claimed by '{vlm_name}' → OCR "
                    f"'{ocr_name}' kept as separate ocr_only room (no overwrite)"
                )
                r = dict(ocr_room)
                r["source"] = "ocr_only"
                merged.append(r)
            elif ocr_name and len(ocr_name) > len(vlm_name):
                logger.debug(
                    f"Merged: VLM '{vlm_name}' + OCR '{ocr_name}' → using OCR (compound)"
                )
                # FP-1: use OCR bbox as geometry, not VLM's. VLM boxes on this
                # pipeline are frequently oversized/mislocated (measured: single
                # VLM box matching 5-21 distinct OCR labels). OCR bbox is the
                # precise label location; SAM (Step 3.5) expands it to room scale.
                merged[matched_vlm]["vlm_bbox"] = merged[matched_vlm].get("bbox")
                merged[matched_vlm]["bbox"] = ocr_bbox
                merged[matched_vlm]["name"] = ocr_name
                merged[matched_vlm]["room_name"] = ocr_name
                merged[matched_vlm]["ocr_label"] = ocr_name
                merged[matched_vlm]["source"] = "vlm_ocr_merged"
                # Geometry provenance must follow the geometry: bbox now comes
                # from ocr_room, so original_bbox/sam_expanded/sam_skip_reason
                # must too. Without this, these fields keep the VLM room's own
                # (unrelated, also-stale) pre-SAM values — wrong provenance for
                # any downstream check on the OCR-sourced box's SAM history.
                merged[matched_vlm]["original_bbox"] = ocr_room.get("original_bbox")
                merged[matched_vlm]["sam_expanded"] = ocr_room.get("sam_expanded")
                merged[matched_vlm]["sam_skip_reason"] = ocr_room.get("sam_skip_reason")
                merged[matched_vlm]["sam_confidence"] = ocr_room.get("sam_confidence")
            else:
                # FM-1 (D-M2): a matched-but-not-compound OCR label was previously
                # DROPPED here. On dense floors one oversized VLM box overlaps many
                # distinct OCR labels (measured: 13 labels into 1 box on page000);
                # keeping only the winner silently lost real rooms (STAIR, ELEV,
                # MECHANICAL, distinct units). Instead, add the OCR label as its own
                # ocr_only room (own precise bbox), UNLESS it is empty or a substring
                # of the matched VLM name (true fragment/duplicate of the same label,
                # e.g. 'BR' inside 'TYPE-D2 3BR', ''). Downstream overlap-resolution
                # and IoU dedup collapse any genuine duplicate geometry.
                if ocr_name and ocr_name not in vlm_name:
                    logger.debug(
                        f"FM-1: VLM '{vlm_name}' overlaps OCR '{ocr_name}' → "
                        f"kept as separate ocr_only room (distinct label)"
                    )
                    r = dict(ocr_room)
                    r["source"] = "ocr_only"
                    merged.append(r)
                else:
                    logger.debug(f"Merged: VLM '{vlm_name}' (OCR '{ocr_name}' skipped, fragment/empty)")
        else:
            # OCR room doesn't overlap with VLM → add as new room
            logger.debug(f"Added OCR room (no VLM match): {ocr_name}")
            r = dict(ocr_room)
            r.setdefault("source", "ocr_only")
            merged.append(r)

    logger.info(f"Merged {len(vlm_rooms)} VLM + {len(ocr_rooms)} OCR → {len(merged)} total")
    return merged


# ── FIX-5: non-drawing-region drop (ink-fraction under box) ──────────────────
# Verified defect (18-file visual audit, sprint1_verify35): the VLM/OCR places a
# room label in a region that carries no floor-plan linework — blank margin,
# BOM/notes/schedule table, or title block. Two size-distinct symptoms share one
# root: oversized vlm_only floor-plate blobs AND small schedule-table label grids
# both sit off the drawing. Existing exclusion zones (FIX-4) miss them because no
# excluded-token cluster forms in the blank gaps between tables, so a text-driven
# zone never covers the box.
#
# Signal: fraction of ink (dark) pixels under the box footprint. A box over the
# drawing covers walls/fixtures (high ink); a box in a blank margin covers
# near-white paper (near-zero ink). Size-, source-, and grid-agnostic — measured
# to separate the two defect classes where those weaker signals do not.
#
# Constants are data-derived, not chosen — measured over 473 boxes across 58
# sheets against the 18-file visual ground truth:
#   INK_BINARISE_MAX  grayscale value below which a pixel counts as ink. A fixed
#     near-white cut. Per-page Otsu was tested and rejected: it collapses the
#     wrong/right gap (on-drawing boxes fall to ~0 ink on some sheets).
#   INK_MIN_FRAC  at that binarise level the wrong-box ink maxed at 0.077 and the
#     right-box ink bottomed at 0.102; the cut sits in that empty gap.
# Known residual (1 of 112): a margin giant overlapping an inset detail-plan
# reaches 0.111 ink and survives — a false negative, not a false drop.
INK_BINARISE_MAX = 200
INK_MIN_FRAC = 0.09


def _bbox_ink_fraction(gray: Image.Image, bbox: List, binarise_max: int) -> Optional[float]:
    """Fraction of ink (dark) pixels inside an XYWH bbox.

    Returns None when the bbox is malformed or has no area, so the caller can
    distinguish "could not measure" from "measured zero ink".
    """
    if not bbox or len(bbox) < 4:
        return None
    x, y, w, h = (int(round(v)) for v in bbox[:4])
    if w <= 0 or h <= 0:
        return None
    crop = gray.crop((x, y, x + w, y + h))
    total = crop.width * crop.height
    if total == 0:
        return None
    ink_mask = crop.point(lambda p: 255 if p < binarise_max else 0)
    ink_pixels = ink_mask.histogram()[255]
    return ink_pixels / total


def _filter_low_ink_rooms(
    rooms: List[Dict], gray: Image.Image, min_frac: float, binarise_max: int
) -> Tuple[List[Dict], int]:
    """Drop rooms whose box lies over a non-drawing region (ink below min_frac).

    Boxes that cannot be measured (malformed bbox, out of image) are kept — this
    filter only removes positively-confirmed low-ink placements.
    """
    kept: List[Dict] = []
    dropped = 0
    for room in rooms:
        frac = _bbox_ink_fraction(gray, room.get("bbox", []), binarise_max)
        if frac is not None and frac < min_frac:
            rn = room.get("room_name") or room.get("name", "?")
            logger.warning(
                f"FIX-5: dropped non-drawing bbox '{rn}' "
                f"(ink={frac:.3f} < {min_frac} min)"
            )
            dropped += 1
            continue
        kept.append(room)
    return kept, dropped


# ── FIX-6: ruled-table detection drop ────────────────────────────────────────
# Verified defect (Rockaway p000): the OCR reads a BOM/schedule row as a room and
# places a box on the table. Its footprint has ink (table rules + text), so FIX-5
# (blank-ink) cannot catch it, and no excluded-token cluster covers the body so
# FIX-4 (zones) misses it too. A ruled table is visually distinct from the floor
# plan: it is a stack of evenly-spaced full-width horizontal rules. This detects
# that structure directly from the image and drops rooms whose centroid lands in
# a detected table — independent of OCR, so no re-OCR is needed.
#
# Constants are data-derived, measured over table + drawing regions on all 58
# sheets (see /tmp validation), not chosen:
#   TABLE_ROW_REGULARITY_MAX: row-gap std/mean. Real tables measured 0.000-0.005;
#     floor-plan line groups measured >= 1.2. Cut placed in that wide empty gap.
#   TABLE_MIN_ROWS: smallest real table sampled had 6 rows; 4 = conservative floor.
#   *_FRAC: geometric fractions of image width (scale-invariant) — a rule spans
#     >= 3% width, a >90%-width line is a sheet border (not a table row), and one
#     table's rows share left/right edges within 2% of width.
# Validated: fires on 7 BOM tables across 58 sheets, catches the bug box, drops
# zero legit on-drawing rooms.
TABLE_ROW_REGULARITY_MAX = 0.10
TABLE_MIN_ROWS = 4
TABLE_RULE_MIN_WIDTH_FRAC = 0.03
TABLE_BORDER_MAX_WIDTH_FRAC = 0.90
TABLE_ROW_EDGE_TOL_FRAC = 0.02


def _detect_ruled_tables(gray: Image.Image, binarise_max: int) -> List[Tuple[int, int, int, int]]:
    """Detect ruled tables (BOM/schedule blocks) as (x1, y1, x2, y2) rectangles.

    A table is a group of >= TABLE_MIN_ROWS full-width horizontal rules that share
    left/right edges and are evenly spaced (row-gap regularity below the cut).
    Isolates tables from the floor plan, whose line groups are irregular.
    """
    arr = np.asarray(gray)
    H, W = arr.shape
    _, ink = cv2.threshold(arr, binarise_max, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(W * TABLE_RULE_MIN_WIDTH_FRAC)), 1))
    horiz = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    n, _lbl, stats, _cent = cv2.connectedComponentsWithStats(horiz, 8)

    rules = []  # (cy, x1, x2)
    for i in range(1, n):
        x, y, w, h, _area = stats[i]
        if W * TABLE_RULE_MIN_WIDTH_FRAC <= w <= W * TABLE_BORDER_MAX_WIDTH_FRAC:
            rules.append((y + h / 2.0, x, x + w))
    rules.sort()

    edge_tol = W * TABLE_ROW_EDGE_TOL_FRAC
    tables: List[Tuple[int, int, int, int]] = []
    used = [False] * len(rules)
    for i in range(len(rules)):
        if used[i]:
            continue
        band = [rules[i]]
        used[i] = True
        for j in range(i + 1, len(rules)):
            if used[j]:
                continue
            if (abs(rules[j][1] - band[0][1]) <= edge_tol
                    and abs(rules[j][2] - band[0][2]) <= edge_tol):
                band.append(rules[j])
                used[j] = True
        if len(band) < TABLE_MIN_ROWS:
            continue
        ys = sorted(b[0] for b in band)
        gaps = np.diff(ys)
        regularity = float(np.std(gaps) / np.mean(gaps)) if len(gaps) and np.mean(gaps) else 9.9
        if regularity <= TABLE_ROW_REGULARITY_MAX:
            x1 = min(b[1] for b in band)
            x2 = max(b[2] for b in band)
            tables.append((int(x1), int(ys[0]), int(x2), int(ys[-1])))
    return tables


def _filter_rooms_in_tables(
    rooms: List[Dict], tables: List[Tuple[int, int, int, int]]
) -> Tuple[List[Dict], int]:
    """Drop rooms whose bbox centroid falls inside a detected ruled table."""
    if not tables:
        return rooms, 0
    kept: List[Dict] = []
    dropped = 0
    for room in rooms:
        bbox = room.get("bbox") or []
        if len(bbox) >= 4:
            bx, by, bw, bh = bbox[:4]
            cx, cy = bx + bw / 2, by + bh / 2
            if any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in tables):
                rn = room.get("room_name") or room.get("name", "?")
                logger.warning(f"FIX-6: dropped bbox '{rn}' — centroid inside ruled table")
                dropped += 1
                continue
        kept.append(room)
    return kept, dropped


def prepare_sft_annotation(
    annotation: Dict,
    min_rooms_for_sft: int = 1,
    image_dir: Optional[Path] = None,
) -> Dict:
    """
    Convert annotation to SFT-ready format.

    Args:
        annotation: Raw annotation dictionary.
        min_rooms_for_sft: Minimum rooms required for sft_ready=True (default 1).
            Images with ≥3 rooms also receive sft_recommended=True.
        image_dir: Directory holding the source page image (named by
            annotation["image_file"]). When provided, FIX-5 drops rooms placed
            over non-drawing regions. When None, FIX-5 is skipped.

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

    # Per-step rejection counter — attached to annotation for run-level aggregation.
    drop_attribution: Dict[str, int] = {}

    # CRITICAL FIX #2: Merge VLM and OCR results instead of OR logic
    vlm_rooms = annotation.get("rooms", [])
    ocr_rooms = annotation.get("ocr_rooms", [])
    rooms = _merge_vlm_and_ocr(vlm_rooms, ocr_rooms)

    # Step 0-pre: FIX-4 — drop rooms whose centroid falls inside a BOM/title-block exclusion zone.
    # Exclusion zones are computed from OCR excluded-token clusters (BOM tables, panel schedules,
    # title blocks). VLM/OCR sometimes localise a label into these margin regions and SAM then
    # expands it to a giant floor-plate blob. Centroid-in-zone is the most conservative check:
    # a mislocalized detection has its label IN the margin; a real room only touches the margin.
    exclusion_zones = annotation.get("exclusion_zones", [])
    if exclusion_zones and rooms:
        pre_excl = len(rooms)
        def _in_any_zone(bbox, zones):
            if len(bbox) < 4:
                return False
            bx, by, bw, bh = bbox[0], bbox[1], bbox[2], bbox[3]
            if bw <= 0 or bh <= 0:
                return False
            cx, cy = bx + bw // 2, by + bh // 2
            room_area = bw * bh
            for x1, y1, x2, y2 in zones:
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    return True
                # T-OVERLAP: centroid-only misses large mislocalized boxes that
                # overlap a zone (e.g. BOM table) without being centered on it.
                # Two directions matter, checked independently — a huge box can
                # swallow a small marker zone (verified: CORRIDOR bbox 1037x587
                # fully contains a 290x127 AVI-ON BOM zone, but that's only 6%
                # of the room's own area — overlap/room_area alone misses it),
                # and a marker zone can be large relative to a smaller room.
                ix1, iy1 = max(bx, x1), max(by, y1)
                ix2, iy2 = min(bx + bw, x2), min(by + bh, y2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    zone_area = (x2 - x1) * (y2 - y1)
                    if inter / room_area > 0.3 or (zone_area > 0 and inter / zone_area > 0.4):
                        return True
            return False
        rooms = [r for r in rooms if not _in_any_zone(r.get("bbox", []), exclusion_zones)]
        n_excl = pre_excl - len(rooms)
        if n_excl:
            drop_attribution["bom_zone"] = n_excl
            logger.info(f"FIX-4: dropped {n_excl} room(s) with centroid in BOM/title-block zone")

    # Step 0-pre: FIX-5 — drop rooms placed over non-drawing regions (blank
    # margin, BOM/notes table, title block) where no linework lies under the
    # box. Complements FIX-4: catches placements in the blank gaps that no
    # excluded-token cluster covers. Requires the source image; skipped without.
    if image_dir is not None and rooms:
        img_name = annotation.get("image_file")
        img_path = Path(image_dir) / img_name if img_name else None
        if img_path and img_path.exists():
            try:
                with Image.open(img_path) as im:
                    gray = im.convert("L")
                rooms, n_ink = _filter_low_ink_rooms(
                    rooms, gray, INK_MIN_FRAC, INK_BINARISE_MAX
                )
                if n_ink:
                    drop_attribution["low_ink"] = n_ink
                    logger.info(
                        f"FIX-5: dropped {n_ink} room(s) placed over non-drawing region"
                    )

                # FIX-6: drop rooms whose centroid lands in a ruled table (BOM/
                # schedule). Reuses the image already loaded for FIX-5.
                tables = _detect_ruled_tables(gray, INK_BINARISE_MAX)
                rooms, n_table = _filter_rooms_in_tables(rooms, tables)
                if n_table:
                    drop_attribution["ruled_table"] = n_table
                    logger.info(
                        f"FIX-6: dropped {n_table} room(s) placed over a ruled table"
                    )
            except Exception as exc:  # noqa: BLE001 — never fail the page on FIX-5/6
                logger.warning(f"FIX-5/6 skipped ({img_path.name}): {exc}")
        else:
            logger.debug("FIX-5/6 skipped: image not found for %s", img_name)

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
        drop_attribution["synthetic"] = len(_synthetic)
        annotation.setdefault("synthetic_rooms", []).extend(_synthetic)
        logger.info(
            f"Excluded {len(_synthetic)} synthetic room(s) from SFT path "
            f"(archived to annotation['synthetic_rooms'] for review)"
        )

    # Step 0c-ii: Fix 2 — Room number concatenated into room_name.
    # VLM reads "CONFERENCE 2902" as a single label and puts the entire
    # string in room_name.  Strip trailing 3+ digit numeric identifiers
    # and move them to room_number if room_number is empty or auto-incremented.
    _ROOM_NUM_SUFFIX = re.compile(r'^([A-Z][A-Z /]+?)\s+#?(\d{3,})$')
    for room in rooms:
        rn = room.get("room_name") or room.get("name", "")
        m = _ROOM_NUM_SUFFIX.match(rn.strip())
        if m:
            clean_name = m.group(1).strip()
            extracted_num = m.group(2)
            existing_num = room.get("room_number", "")
            # Only move if room_number is empty or looks auto-incremented
            if not existing_num or existing_num != extracted_num:
                room["room_number"] = extracted_num
            room["room_name"] = clean_name
            room["name"] = clean_name
            logger.info(
                f"Fix2: cleaned room_name '{rn}' → "
                f"name='{clean_name}', room_number='{extracted_num}'"
            )

    # Step 0c-iii: Fix 2b — Also clean "OFFICE 2905" style OCR artifacts
    _ROOM_NUM_SUFFIX_SHORT = re.compile(r'^([A-Z][A-Z /]+?)\s+(\d{2,})$')
    for room in rooms:
        rn = room.get("room_name") or room.get("name", "")
        if _ROOM_NUM_SUFFIX.match(rn.strip()):
            continue  # Already cleaned above
        m = _ROOM_NUM_SUFFIX_SHORT.match(rn.strip())
        if m and len(m.group(2)) >= 2:
            clean_name = m.group(1).strip()
            extracted_num = m.group(2)
            existing_num = room.get("room_number", "")
            if not existing_num:
                room["room_number"] = extracted_num
            room["room_name"] = clean_name
            room["name"] = clean_name
            logger.info(f"Fix2b: cleaned room_name '{rn}' → name='{clean_name}'")

    # Step 0c-iv: Geometric filter — single location for ALL bbox-shape rejection.
    #
    # This is the ONLY place in the pipeline that inspects bbox geometry.
    # Four conditions, each with a single auditable cause:
    #
    #   (a) Extreme aspect ratio + small minimum dimension — text labels
    #   (b) Bbox smaller than 2.5% of image dimension — sub-room fragments
    #   (c) Absolute area below MIN_ROOM_AREA_PX — degenerate tiny boxes
    #   (d) Area exceeds MAX_ROOM_FRAC of image — mislocalized giant blobs
    #       (e.g. SAM grabs a 43% floor plate from a single label centroid).
    #       A single labelled room cannot physically occupy >30% of a floor.
    MIN_ROOM_AREA_PX = 10_000  # ~0.5"×0.5" at 200 DPI; excludes all text labels
    MAX_ROOM_FRAC    = 0.30    # >30% of image = mislocalized blob, not one room

    # DV-3 / Option A: OCR-anchored rooms are exempt from the text-label-shape
    # checks (a)/(b) and use a much lower absolute-area floor for (c).
    # Rationale (measured on page002): SAM either over-expands a label centroid
    # into a giant floor-plate blob or barely grows it past label size — only
    # ~11% of located OCR labels land in the intended room-scale window. The
    # (a)/(b)/(c) checks below assume SAM reliably expands every box to room
    # scale; when it doesn't, they discard the correctly-located label instead
    # of the (already-guarded-elsewhere) giant. Located label boxes are kept
    # as-is rather than thrown away for being label-shaped — that IS their
    # source geometry now. (d) MAX_ROOM_FRAC still applies to every source, so
    # a giant that slips past SAM is still rejected here regardless of origin.
    _OCR_ANCHORED_SOURCES = {"ocr_only", "vlm_ocr_merged"}
    OCR_MIN_AREA_PX = 150  # excludes near-zero-area / degenerate boxes only

    img_w_pre = annotation.get("image_size", {}).get("width", 0)
    img_h_pre = annotation.get("image_size", {}).get("height", 0)
    img_area_pre = img_w_pre * img_h_pre if img_w_pre and img_h_pre else 0
    pre_geom = len(rooms)
    geom_filtered = []
    for room in rooms:
        bbox = room.get("bbox", [])
        rn = room.get("room_name") or room.get("name", "?")
        is_ocr_anchored = room.get("source") in _OCR_ANCHORED_SOURCES

        if len(bbox) == 4:
            bx, by, bw, bh = bbox
            area = bw * bh

            min_dim = min(bw, bh)
            max_dim = max(bw, bh)
            aspect = max_dim / min_dim if min_dim > 0 else 999

            if is_ocr_anchored:
                # (c') OCR-anchored: only reject near-degenerate boxes.
                if area < OCR_MIN_AREA_PX:
                    logger.warning(
                        f"Fix1: dropped degenerate OCR bbox '{rn}' "
                        f"(area={area:.0f} < {OCR_MIN_AREA_PX} px²)"
                    )
                    continue

                # T-C1 (revised): SAM sometimes over-expands an OCR-anchored
                # label past a single-unit's plausible size (measured: OCR
                # label 0.02% of image -> SAM box 8.8%). The located label
                # itself (original_bbox, pre-SAM) is correct; the expansion is
                # not. Revert to the located label rather than drop the room
                # or keep the drifted giant. Reuses the existing single-unit
                # ceiling (MAX_VLM_UNIT_FRAC, defined below) — no new threshold.
                if (room.get("sam_expanded") and room.get("original_bbox")
                        and img_area_pre > 0
                        and area > img_area_pre * 0.08):
                    ob = room["original_bbox"]
                    if len(ob) == 4 and ob[2] * ob[3] >= OCR_MIN_AREA_PX:
                        logger.warning(
                            f"T-C1: reverted over-expanded OCR bbox '{rn}' "
                            f"(sam_area={area:.0f} = {100*area/img_area_pre:.1f}% "
                            f"> 8%) to located label {ob}"
                        )
                        room["bbox"] = ob
                        bbox = ob
                        bx, by, bw, bh = ob
                        area = bw * bh
                        room["sam_expanded"] = False
                        room["sam_skip_reason"] = "reverted_overexpansion"
            else:
                # (a) Text-label bbox: extreme aspect + small minimum dimension
                if aspect > 4.0 and min_dim < 100:
                    logger.warning(
                        f"Fix1: dropped text-label bbox '{rn}' "
                        f"(aspect={aspect:.1f}, min_dim={min_dim:.0f}px)"
                    )
                    continue

                # (b) Bbox narrower/shorter than 2.5% of image dimension
                if img_w_pre > 0 and img_h_pre > 0:
                    if bw < img_w_pre * 0.025 or bh < img_h_pre * 0.025:
                        logger.warning(
                            f"Fix1: dropped sub-2.5%% bbox '{rn}' "
                            f"(w={bw:.0f}<{img_w_pre*0.025:.0f}, "
                            f"h={bh:.0f}<{img_h_pre*0.025:.0f})"
                        )
                        continue

                # (c) Absolute area below minimum room size
                if area < MIN_ROOM_AREA_PX:
                    logger.warning(
                        f"Fix1: dropped sub-minimum-area bbox '{rn}' "
                        f"(area={area:.0f} < {MIN_ROOM_AREA_PX} px²)"
                    )
                    continue

            # (d) Area exceeds 30% of image — mislocalized giant blob
            if img_area_pre > 0 and area > img_area_pre * MAX_ROOM_FRAC:
                logger.warning(
                    f"Fix1: dropped oversized bbox '{rn}' "
                    f"(area={area:.0f} = {100*area/img_area_pre:.0f}% of image "
                    f"> {MAX_ROOM_FRAC:.0%} max)"
                )
                continue

            # (e) P-1: VLM-only rooms exceeding single-unit ceiling.
            # A single placed room cannot span >8% of a floor plan. vlm_only rooms
            # above this are mislocalized over multi-room regions or notes.
            # OCR-sourced and merged rooms are exempt — OCR labels are precise.
            MAX_VLM_UNIT_FRAC = 0.08
            if (img_area_pre > 0
                    and room.get("source") == "vlm_only"
                    and area > img_area_pre * MAX_VLM_UNIT_FRAC):
                logger.warning(
                    f"P-1: dropped vlm_only giant bbox '{rn}' "
                    f"(area={area:.0f} = {100*area/img_area_pre:.1f}% > "
                    f"{MAX_VLM_UNIT_FRAC:.0%} single-unit ceiling)"
                )
                continue

            # (f) T-ASPECT: type-aware aspect cap for degenerate thin strips.
            # Verified failure mode (Kennedy/Lake Shore school sheets): the VLM
            # emits thin-tall/thin-wide boxes (e.g. STORAGE aspect 12 spanning
            # 74% of image height) that evade the area caps (d)/(e) — a thin box
            # keeps total area moderate despite spanning most of a dimension —
            # and evade the (a) text-label check (min_dim > 100). A non-
            # circulation room physically cannot be both very elongated AND span
            # a large fraction of the sheet; that shape is a mislocalized strip.
            # Circulation types (corridor/hallway/lobby/stairwell/elevator/riser/
            # vestibule) are EXEMPT — they are legitimately long-and-thin.
            # Conjunction (aspect AND span) is deliberate: spares normal-aspect
            # rooms and small thin closets (low span).
            _CIRC = {"corridor", "hallway", "lobby", "elevator", "stairwell",
                     "riser", "vestibule"}
            _type = (room.get("type") or room.get("category") or "").lower()
            _name = (rn or "").lower()
            _is_circ = any(c in _type or c in _name for c in _CIRC)
            if (not _is_circ and img_w_pre > 0 and img_h_pre > 0
                    and aspect > 4.0
                    and max(bw / img_w_pre, bh / img_h_pre) > 0.25):
                logger.warning(
                    f"T-ASPECT: dropped degenerate strip '{rn}' "
                    f"(type={_type or 'n/a'}, aspect={aspect:.1f}, "
                    f"span={max(bw/img_w_pre, bh/img_h_pre):.0%} of a dimension)"
                )
                continue

        geom_filtered.append(room)
    rooms = geom_filtered
    n_geom = pre_geom - len(rooms)
    drop_attribution["geometric"] = n_geom
    if n_geom > 0:
        logger.info(f"Fix1: geometric filter removed {n_geom} bbox(es)")

    # Remediation Fix #2: Resolve bbox overlaps in non-circulation room types.
    # This must happen AFTER geometric filter but BEFORE semantic filtering
    # to ensure clean spatial geometry for SFT.
    pre_overlap = len(rooms)
    rooms = _resolve_bbox_overlaps(rooms)
    # Fix2b (Phase 2): same-type containment dedup — catches nested duplicate
    # boxes (esp. circulation types skipped by _resolve_bbox_overlaps) whose
    # IoU falls just under the 0.5 merge threshold, leaving an oversized copy
    # that survives past the 8% single-unit ceiling.
    rooms = _dedup_nested_same_type(rooms)
    n_resolved = pre_overlap - len(rooms)
    drop_attribution["overlap"] = n_resolved
    if n_resolved > 0:
        logger.info(f"Remediation Fix #2: overlap resolution merged/removed {n_resolved} room(s)")

    # Step 1: Semantic filtering
    pre_semantic = len(rooms)
    rooms = validator.filter_rooms(rooms)
    n_semantic = pre_semantic - len(rooms)
    drop_attribution["semantic"] = n_semantic
    if n_semantic > 0:
        logger.info(f"Step 1: semantic filter removed {n_semantic} room(s) ({len(rooms)} remaining)")
    logger.debug(f"After semantic filter: {len(rooms)} rooms")

    # Step 2: Confidence filtering (preserves VLM rooms with no confidence field)
    pre_confidence = len(rooms)
    rooms = filter_by_confidence(rooms, min_confidence=0.85)
    n_confidence = pre_confidence - len(rooms)
    drop_attribution["confidence"] = n_confidence
    if n_confidence > 0:
        logger.info(f"Step 2: confidence filter removed {n_confidence} room(s) ({len(rooms)} remaining)")
    else:
        logger.debug(f"After confidence filter: {len(rooms)} rooms")

    # Step 2b: Filter out-of-bounds rooms
    # (pipeline-revalidation-analysis §3 Fix B)
    #
    # Rooms with bboxes exceeding the image dimensions are produced when
    # OCR returns pixel coordinates from the full-resolution PDF rather
    # than the cropped image.  Previously these were only filtered in the
    # exporters, AFTER sft_ready was set.  This caused pages with only
    # OOB rooms (e.g., page000 with ELEVATOR at y=5892 on a 3375-tall
    # image) to get sft_ready=True and appear in COCO splits as
    # zero-annotation ghost images.  Filtering here ensures sft_ready
    # accurately reflects exportable room count.
    img_w = annotation.get("image_size", {}).get("width", 0)
    img_h = annotation.get("image_size", {}).get("height", 0)
    if img_w > 0 and img_h > 0:
        pre_oob = len(rooms)
        rooms = [r for r in rooms if _bbox_in_bounds(r.get("bbox", []), img_w, img_h)]
        n_dropped = pre_oob - len(rooms)
        drop_attribution["oob"] = n_dropped
        if n_dropped > 0:
            logger.warning(
                f"Dropped {n_dropped} out-of-bounds room(s) "
                f"(image={img_w}x{img_h})"
            )
    logger.debug(f"After OOB filter: {len(rooms)} rooms")

    # Step 2c: removed.
    # Area-based bbox rejection was consolidated into Step 0c-iv (geometric
    # filter) where all spatial/shape concerns are handled in one place.
    # No area checks belong in the semantic/confidence/taxonomy stage.

    # Step 3: Normalize ALL room types through canonical taxonomy
    # (pipeline-revalidation-analysis §3 Fix A)
    #
    # CRITICAL FIX: The previous condition (type == "other" or type missing)
    # skipped rooms where the VLM assigned a plausible-but-wrong type.
    # Example: VLM assigns PANTRY → type="storage" (wrong; should be kitchen).
    # Because type≠"other", normalization was skipped and the error propagated
    # to COCO output.  Unconditional normalization through the canonical
    # taxonomy guarantees deterministic, consistent category assignment
    # regardless of VLM output variance.  This is idempotent: rooms already
    # correctly typed (KITCHEN→kitchen) are unchanged.
    for room in rooms:
        room_name = room.get("room_name") or room.get("name", "")
        if room_name:
            normalized = normalizer.normalize(room_name)
            room["type"] = normalized
            room["category"] = normalized  # Sync category field for COCO exporter
            logger.debug(f"Normalized '{room_name}' → {normalized}")

    # Step 4: SFT validation
    pre_sft = len(rooms)
    sft_ready = []
    for room in rooms:
        is_valid, checks = validate_for_sft(room)
        if not is_valid:
            failed_checks = [k for k, v in checks.items() if not v]
            logger.debug(f"Room rejected: {room.get('name')} - {failed_checks}")
            continue
        sft_ready.append(room)
    drop_attribution["sft_validation"] = pre_sft - len(sft_ready)

    # Step 4b: Zero-dimension bbox guard.
    # Stripe detection is now handled upstream in hallucination_detector.detect_hallucinations()
    # (Pattern 5, 60% threshold) so stripes rarely reach this stage.
    # Only check for zero-dimension bboxes that somehow survived geometric filter.
    degenerate_reasons: List[str] = []
    zero_dims = [
        i for i, r in enumerate(sft_ready)
        if len(r.get("bbox", [])) == 4 and (r["bbox"][2] == 0 or r["bbox"][3] == 0)
    ]
    if zero_dims:
        degenerate_reasons.append(
            f"{len(zero_dims)} room(s) have zero-width or zero-height bbox: indices {zero_dims}"
        )

    # Step 4c: Room-scale + content gate (F-D + FIX-1).
    # Two independent signals:
    #   (a) area >= ROOM_MIN_AREA — rejects label-sized boxes.
    #   (b) content_density >= MIN_CONTENT_DENSITY — rejects boxes over blank space
    #       (hallucinations on roof/site sheets and empty margins score ~0;
    #       real rooms with walls/fixtures score 0.05-0.19).
    # content_density is attached by SAM refine_annotations. When absent (SAM off),
    # the density check is skipped and area alone applies (backward compatible).
    ROOM_MIN_AREA = 40_000          # ~200x200px at 200 DPI; separates rooms from labels
    # Rejects boxes over genuinely blank space (density ~0). NOT raised higher:
    # valid spaces with giant mislocalized boxes (e.g. 2BR units, cd 0.036-0.045)
    # would be discarded — that is a LOCALIZATION problem, fixed upstream by the
    # SAM collapse-guardrail change, not by dropping the valid room here.
    MIN_CONTENT_DENSITY = 0.03      # below this the box is over blank space, not a room
    for room in sft_ready:
        bbox = room.get("bbox", [])
        area = bbox[2] * bbox[3] if len(bbox) == 4 else 0
        # DV-3 / Option A: OCR-anchored rooms bypass the room-scale area floor
        # (already passed the OCR_MIN_AREA_PX check upstream in the geometric
        # filter) but still must pass the content-density check — a blank-space
        # hallucination anchored to real OCR text is still not a real room.
        if room.get("source") in _OCR_ANCHORED_SOURCES:
            area_ok = True
        else:
            area_ok = area >= ROOM_MIN_AREA
        density = room.get("content_density")
        density_ok = density is None or density >= MIN_CONTENT_DENSITY
        room["is_room_scale"] = area_ok and density_ok

    pre_strip = len(sft_ready)
    sft_ready = [r for r in sft_ready if r.get("is_room_scale")]
    n_stripped = pre_strip - len(sft_ready)
    if n_stripped:
        drop_attribution["not_room_scale"] = n_stripped
        logger.info(
            f"F-D: stripped {n_stripped} non-room box(es) "
            f"(area<{ROOM_MIN_AREA} or density<{MIN_CONTENT_DENSITY}); "
            f"{len(sft_ready)} kept"
        )

    # FIX-B: distinguish "rooms detected but all mislocalized" from "no rooms".
    # When the VLM produced detections but every one failed the content gate
    # (boxes over blank space — localization failure on dense floors), the page
    # has real rooms that need manual annotation, NOT a blank/non-floor page.
    annotation["localization_failed"] = (pre_strip > 0 and len(sft_ready) == 0)

    # Step 5: Clean up contamination
    annotation.pop("ocr_rooms", None)  # Remove unfiltered OCR
    annotation.pop("panels", None)  # Remove equipment (not spatial rooms)

    # Step 6: Return with corrected sft_ready logic
    annotation["rooms"] = sft_ready

    clean = (
        not degenerate_reasons and
        "panels" not in annotation and
        "ocr_rooms" not in annotation
    )
    annotation["sft_ready"] = len(sft_ready) >= min_rooms_for_sft and clean
    # sft_recommended: higher-quality tier requiring ≥3 rooms, regardless of min_rooms_for_sft.
    SFT_RECOMMENDED_MIN = 3
    annotation["sft_recommended"] = len(sft_ready) >= SFT_RECOMMENDED_MIN and clean

    if 0 < len(sft_ready) < min_rooms_for_sft:
        logger.warning(
            f"Fix8: {len(sft_ready)} rooms below configured minimum "
            f"({min_rooms_for_sft}) — sft_ready=False"
        )
        drop_attribution["min_rooms_gate"] = min_rooms_for_sft - len(sft_ready)

    if degenerate_reasons:
        drop_attribution["degenerate_bbox"] = len(degenerate_reasons)
        annotation["degenerate_bbox_reasons"] = degenerate_reasons
        for reason in degenerate_reasons:
            logger.warning(f"Degenerate-bbox: {reason}")

    # Attach per-step drop counts (omit zero-valued entries for readability).
    annotation["drop_attribution"] = {k: v for k, v in drop_attribution.items() if v}

    logger.info(
        f"SFT preparation complete: {len(sft_ready)} rooms, sft_ready={annotation['sft_ready']}"
    )
    return annotation
