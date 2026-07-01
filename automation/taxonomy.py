"""
Mandatory SFT Taxonomy — Single Source of Truth.

Implements the standardized 31-class room taxonomy required for Vision-Language
Model fine-tuning, while preserving fine-grained residential/utility classification
via extended_type for non-SFT use cases.

ALL room type mappings throughout the pipeline import from here.

Window/skylight/opening suffixes (e.g., "CONFERENCE w/ windows") are NOT
determined by room type — they are visual attributes computed by spatial
intersection of window detector output with room bboxes (window_detector.py).
Any base type can receive a window suffix; eligibility is not restricted by
room category.  Use add_window_suffix() to compose, strip_window_suffix() to
decompose back to the base mandatory type.

Usage
-----
    from automation.taxonomy import MANDATORY_CLASSES, VALID_TYPES, EXTENDED_TYPES
    from automation.taxonomy import normalize_to_mandatory, get_extended_type

    # For SFT output (mandatory taxonomy only)
    mandatory_type = normalize_to_mandatory("CONFERENCE ROOM")  # → "CONFERENCE"

    # For extended classification (backward compatibility)
    extended_type = get_extended_type("CONFERENCE ROOM")  # → "conference_room"
"""

from typing import Dict, List, Set, Optional, NamedTuple
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mandatory Taxonomy: Base Classes for SFT Training
# ---------------------------------------------------------------------------
# These are the ONLY types allowed in the "type" field of SFT-ready output.
# Window/skylight/opening variants are applied as post-processing suffixes.
# ---------------------------------------------------------------------------

MANDATORY_CLASSES: Dict[str, List[str]] = {
    # ── Residential Units ────────────────────────────────────────────────────
    # Covers apartment/condo unit layouts on residential and mixed-use plans.
    # Architectural unit-type codes (TYPE-A1, TYPE-B3, etc.) are common in
    # multi-family building plans; they all map here regardless of bedroom count.
    "RESIDENTIAL UNIT": [
        "RESIDENTIAL UNIT", "RESIDENTIAL", "APARTMENT", "DWELLING",
        "UNIT", "APARTMENT UNIT",
        # Bedroom-count unit codes (n-bedroom apartment)
        "STUDIO", "STUDIO APARTMENT",
        "0BR", "0 BR",
        "1BR", "1 BR", "1 BEDROOM",
        "2BR", "2 BR", "2 BEDROOM",
        "3BR", "3 BR", "3 BEDROOM",
        "4BR", "4 BR", "4 BEDROOM",
        # OBR is a common OCR misread of 0BR (digit-zero ↔ letter-O)
        "OBR",
        # Architectural unit-type code prefixes (TYPE-A1 OBR, TYPE-B3 1BR …)
        "TYPE-A", "TYPE-B", "TYPE-C", "TYPE-D",
        # Short residential abbreviations — listed here so _SURFACE_TO_MANDATORY
        # resolves them before normalize_to_mandatory's short-token guard (len≤4)
        # fires and falls back to STORAGE ROOM.
        "BR", "BD", "BDRM", "MBR", "MSTR",
        "LR", "LV",
        "DR", "DIN",
        "BEDROOM", "MASTER BEDROOM",
        "LIVING ROOM", "DINING ROOM",
        "FAMILY ROOM",
    ],

    # ── Office ──────────────────────────────────────────────────────────────
    "PRIVATE OFFICE": [
        "PRIVATE OFFICE", "PRIVATE", "INDIVIDUAL OFFICE",
        "EXECUTIVE OFFICE", "FACULTY OFFICE", "PERSONAL OFFICE",
        "OFFICE", "OFFICE ROOM", "SINGLE OFFICE",
    ],
    "OPEN OFFICE": [
        "OPEN OFFICE", "OPEN PLAN", "OPEN PLAN WORKSPACE",
        "OPEN WORKSPACE", "BULLPEN", "OPEN FLOOR",
    ],

    # ── Meeting/Conference ──────────────────────────────────────────────────
    "CONFERENCE": [
        "CONFERENCE ROOM", "CONFERENCE", "BOARDROOM", "BOARD ROOM",
        "CONF RM", "CONF ROOM", "CONF",
    ],
    "MEETING": [
        "MEETING ROOM", "MEETING", "BREAKOUT ROOM", "BREAKOUT SPACE",
    ],
    "MULTIPURPOSE ROOM": [
        "MULTIPURPOSE ROOM", "MULTI-PURPOSE", "COMMUNITY ROOM",
        "AMENITY ROOM", "FLEXIBLE SPACE", "WORKSHOP",
    ],

    # ── Education ───────────────────────────────────────────────────────────
    "CLASSROOM": [
        "CLASSROOM", "CLASS ROOM", "LAB", "LABORATORY",
        "COMPUTER LAB", "SCIENCE LAB",
    ],
    "LECTURE HALL": [
        "LECTURE HALL", "AUDITORIUM", "THEATER", "THEATRE",
        "ASSEMBLY ROOM",
    ],
    "TRAINING ROOM": [
        "TRAINING ROOM", "TRAINING", "INSTRUCTION ROOM",
    ],

    # ── Food Service ────────────────────────────────────────────────────────
    "RESTAURANT": [
        "RESTAURANT", "DINING HALL", "DINING ROOM", "DINING",
        "RESTAURANT DINING",
    ],
    "CAFETERIA": [
        "CAFETERIA", "CAFE", "BREAK ROOM", "BREAKROOM",
        "KITCHEN", "KITCHENETTE", "PANTRY", "BREAK AREA",
    ],

    # ── Retail ──────────────────────────────────────────────────────────────
    "RETAIL": [
        "RETAIL", "RETAIL SPACE", "SHOP", "STORE",
        "COMMERCIAL SPACE", "SHOWROOM",
    ],

    # ── Circulation ─────────────────────────────────────────────────────────
    "LOBBY": [
        "LOBBY", "RECEPTION", "FOYER", "VESTIBULE", "ENTRANCE",
        "ENTRY", "ENTRY AREA", "ENTRANCE VESTIBULE",
        "WAITING ROOM", "WAITING AREA",
    ],
    "CORRIDOR": [
        "CORRIDOR", "HALLWAY", "PASSAGE", "WALKWAY", "HALL",
        "CIRCULATION", "HALLWAY/CORRIDOR", "CORR",
    ],

    # ── Sanitary ────────────────────────────────────────────────────────────
    "RESTROOM": [
        "RESTROOM", "BATHROOM", "TOILET", "WC", "LAVATORY",
        "POWDER ROOM", "MEN'S RESTROOM", "WOMEN'S RESTROOM",
    ],

    # ── Vertical Circulation ────────────────────────────────────────────────
    "STAIRWELL": [
        "STAIRWELL", "STAIR", "STAIRS", "STAIRCASE",
        "EMERGENCY STAIRS", "EXIT STAIRS", "FIRE STAIR",
    ],

    # ── Storage ─────────────────────────────────────────────────────────────
    "STORAGE ROOM": [
        "STORAGE", "STORAGE ROOM", "STOREROOM", "STORE ROOM",
        "SUPPLY ROOM", "ARCHIVES", "FILE ROOM",
    ],
    "JANITOR CLOSET": [
        "JANITOR", "JANITOR CLOSET", "JANITOR ROOM",
        "CUSTODIAL CLOSET", "CLEANING CLOSET", "CUSTODIAN",
    ],

    # ── MEP ─────────────────────────────────────────────────────────────────
    "ELECTRICAL ROOM": [
        "ELECTRICAL ROOM", "ELECTRICAL", "ELECTRIC ROOM",
        "ELECTRICAL/MECHANICAL ROOM", "TELECOM ROOM", "RISER",
        "SERVER ROOM", "DATA CENTER", "NETWORK ROOM",
    ],

    # ── Athletic ────────────────────────────────────────────────────────────
    "GYMNASIUM": [
        "GYMNASIUM", "GYM", "FITNESS CENTER", "EXERCISE ROOM",
    ],

    # ── Parking ─────────────────────────────────────────────────────────────
    "PARKING GARAGE": [
        "PARKING GARAGE", "PARKING", "GARAGE", "PARKING STRUCTURE",
    ],

    # ── Industrial ──────────────────────────────────────────────────────────
    "WAREHOUSE": [
        "WAREHOUSE", "LOADING DOCK", "RECEIVING", "WAREHOUSE SPACE",
    ],
}

# ---------------------------------------------------------------------------
# Extended Taxonomy: Fine-grained room types for non-SFT output
# ---------------------------------------------------------------------------
# These types are used when preserving backward compatibility or detailed
# classification beyond the SFT mandatory set (e.g., for internal analytics).
# These are NOT allowed in the "type" field of SFT-ready output.
# ---------------------------------------------------------------------------

EXTENDED_TYPES: Dict[str, List[str]] = {
    # ── MEP (not in mandatory taxonomy) ──────────────────────────────────────
    "mechanical": [
        "MECHANICAL ROOM", "MECHANICAL", "MECH ROOM",
        "MECHANICAL/ELECTRICAL ROOM", "HVAC ROOM", "BUILDING SYSTEMS",
    ],
    "machine_room": [
        "MACHINE ROOM", "MACHINE", "ELEVATOR MACHINE ROOM",
    ],
    "boiler": [
        "BOILER ROOM", "BOILER",
    ],
    "pump_room": [
        "PUMP ROOM", "PUMP", "FIRE PUMP ROOM", "FIRE PUMP",
    ],
    "compactor": [
        "COMPACTOR ROOM", "COMPACTOR", "TRASH ROOM", "REFUSE ROOM",
    ],
    "riser": [
        "RISER ROOM", "PIPE CHASE",
    ],

    # ── Circulation (not in mandatory taxonomy) ──────────────────────────────
    "elevator": [
        "ELEVATOR", "ELEVATOR CORE", "ELEVATOR AREA",
        "LIFT", "ELEVATOR MACHINE ROOM",
    ],

    # ── Storage variants (beyond mandatory) ──────────────────────────────────
    "bicycle_storage": [
        "BICYCLE STORAGE", "BICYCLE ROOM", "BICYCLE", "BIKE STORAGE",
        "BIKE ROOM",
    ],

    # ── Residential (not in mandatory taxonomy) ─────────────────────────────
    "bedroom": [
        "BEDROOM", "MASTER BEDROOM", "BEDROOM 1", "BEDROOM 2", "BEDROOM 3",
        "1 BEDROOM", "2 BEDROOM", "3 BEDROOM",
        # compact no-space (OCR sometimes omits the space)
        "BR", "BR1", "BR2", "BR3", "BR4", "BR5",
        # spaced BR-N form — canonical floor-plan label (OCR output)
        "BR 1", "BR 2", "BR 3", "BR 4", "BR 5",
        # BD abbreviation (common in residential plans)
        "BD", "BD1", "BD2", "BD3", "BD4",
        "BD 1", "BD 2", "BD 3", "BD 4",
        # master bedroom variants
        "MBR", "MBR1", "MBR2", "MSTR BR", "MSTR", "BDRM",
        # digit-first forms
        "1BR", "2BR", "3BR", "4BR",
        "1 BR", "2 BR", "3 BR", "4 BR",
        "MASTER BR",
    ],
    "living_room": [
        "LIVING ROOM", "LIVING", "GREAT ROOM", "FAMILY ROOM",
        "LR", "LV", "LVG",
        "LIVING RM", "LIV ROOM",
    ],
    "laundry": [
        "LAUNDRY", "LAUNDRY ROOM", "UTILITY ROOM", "UTILITY",
    ],
    "garage": [
        "GARAGE",
    ],
    "studio": [
        "STUDIO", "STUDIO APARTMENT",
        "0BR", "0 BR",
    ],

    # ── Specialized (not in mandatory taxonomy) ──────────────────────────────
    "carpentry": [
        "CARPENTRY", "CARPENTRY SHOP", "WOOD SHOP", "SHOP",
    ],
    "community_facility": [
        "COMMUNITY FACILITY", "COMMUNITY CENTER",
    ],
    "cctv": [
        "CCTV ROOM", "CCTV", "SECURITY ROOM", "SECURITY",
    ],

    # ── Fallback ────────────────────────────────────────────────────────────
    "other": [
        "ROOM", "SPACE", "AREA", "MISC", "OTHER",
        "UNKNOWN", "UNDEFINED",
    ],
}

# ---------------------------------------------------------------------------
# Lookup structures (computed at import time)
# ---------------------------------------------------------------------------

# Flat set of valid mandatory class names (for SFT output validation)
VALID_TYPES: Set[str] = set(MANDATORY_CLASSES.keys())

# Flat set of valid extended class names
EXTENDED_ONLY_TYPES: Set[str] = set(EXTENDED_TYPES.keys())

# Reverse map: surface form (uppercase) → mandatory class
_SURFACE_TO_MANDATORY: Dict[str, str] = {}
for _mandatory, _variants in MANDATORY_CLASSES.items():
    for _v in _variants:
        _SURFACE_TO_MANDATORY[_v.upper()] = _mandatory

# Reverse map: surface form (uppercase) → extended class
_SURFACE_TO_EXTENDED: Dict[str, str] = {}
for _extended, _variants in EXTENDED_TYPES.items():
    for _v in _variants:
        _SURFACE_TO_EXTENDED[_v.upper()] = _extended

# ---------------------------------------------------------------------------
# VLM output category → mandatory type mapping
# ---------------------------------------------------------------------------
# Translates VLM outputs (which may use architectural convenience names)
# to mandatory SFT types.
# ---------------------------------------------------------------------------

VLM_CATEGORY_MAP: Dict[str, str] = {
    # Residential units
    "residential_unit": "RESIDENTIAL UNIT",
    "apartment": "RESIDENTIAL UNIT",
    "dwelling": "RESIDENTIAL UNIT",
    "studio": "RESIDENTIAL UNIT",
    "unit": "RESIDENTIAL UNIT",
    # Offices
    "office": "PRIVATE OFFICE",
    "open_plan_workspace": "OPEN OFFICE",
    "executive_office": "PRIVATE OFFICE",
    "cubicle_workstation": "OPEN OFFICE",
    "private_office": "PRIVATE OFFICE",

    # Conference/Meeting
    "conference_room": "CONFERENCE",
    "meeting_room": "MEETING",
    "training_room": "TRAINING ROOM",
    "breakout_space": "MEETING",

    # Circulation
    "lobby_reception": "LOBBY",
    "lobby": "LOBBY",
    "hallway_corridor": "CORRIDOR",
    "hallway": "CORRIDOR",
    "corridor": "CORRIDOR",

    # Sanitary
    "restroom": "RESTROOM",
    "bathroom": "RESTROOM",

    # Food Service
    "kitchen_break_room": "CAFETERIA",
    "kitchen": "CAFETERIA",
    "break_room": "CAFETERIA",
    "dining_room": "RESTAURANT",
    "cafeteria": "CAFETERIA",

    # Vertical circulation
    "stairwell": "STAIRWELL",
    "stairwell_stairs": "STAIRWELL",
    "elevator": "CORRIDOR",  # Elevator shafts are circulation, not a mandatory class

    # Storage
    "storage": "STORAGE ROOM",
    "storage_room": "STORAGE ROOM",
    "closet": "STORAGE ROOM",

    # Janitor
    "janitor": "JANITOR CLOSET",
    "custodial": "JANITOR CLOSET",

    # Education
    "classroom": "CLASSROOM",
    "auditorium": "LECTURE HALL",
    "lecture_hall": "LECTURE HALL",

    # Sports
    "gymnasium": "GYMNASIUM",
    "gym": "GYMNASIUM",

    # Parking
    "parking_garage": "PARKING GARAGE",
    "parking": "PARKING GARAGE",
    "garage": "PARKING GARAGE",

    # Warehouse
    "warehouse": "WAREHOUSE",

    # Retail
    "retail": "RETAIL",

    # Multipurpose
    "multipurpose_room": "MULTIPURPOSE ROOM",
    "community_room": "MULTIPURPOSE ROOM",

    # Fallback
    "other": "STORAGE ROOM",  # Default to least-intrusive class
    "unknown": "STORAGE ROOM",
}

# ---------------------------------------------------------------------------
# VLM prompt category list (mandatory classes for prompt)
# ---------------------------------------------------------------------------
# Sent to VLM to constrain its output to mandatory taxonomy.
# Used in vlm_backend.py prompts.
# ---------------------------------------------------------------------------

VLM_PROMPT_CATEGORIES: List[str] = [
    "RESIDENTIAL UNIT",
    "PRIVATE OFFICE",
    "OPEN OFFICE",
    "CONFERENCE",
    "MEETING",
    "MULTIPURPOSE ROOM",
    "CLASSROOM",
    "LECTURE HALL",
    "TRAINING ROOM",
    "RESTAURANT",
    "CAFETERIA",
    "RETAIL",
    "LOBBY",
    "CORRIDOR",
    "RESTROOM",
    "STAIRWELL",
    "STORAGE ROOM",
    "JANITOR CLOSET",
    "ELECTRICAL ROOM",
    "GYMNASIUM",
    "PARKING GARAGE",
    "WAREHOUSE",
]

# ---------------------------------------------------------------------------
# Window-suffix constants and eligibility sets
# ---------------------------------------------------------------------------
# The three suffix strings that add_window_suffix() can append.
WINDOW_SUFFIX = "w/ windows"
SKYLIGHT_SUFFIX = "w/ skylights"
OPENING_SUFFIX = "w/ side openings"

# Ordered most-specific first — used by strip_window_suffix().
_ALL_SUFFIXES = (SKYLIGHT_SUFFIX, OPENING_SUFFIX, WINDOW_SUFFIX)

# ---------------------------------------------------------------------------
# Eligibility sets — CLOSED LISTS from the mandatory SFT taxonomy.
#
# These sets answer the question: "does a suffixed class exist in the
# mandatory taxonomy for this base type?"  They are NOT a detection filter
# (the window detector runs on all rooms regardless).  They are a taxonomy
# constraint: only these base types have named suffixed variants in the SFT
# target class list.
#
# For example:
#   CONFERENCE + has_windows → "CONFERENCE w/ windows"  (eligible ✓)
#   STORAGE ROOM + has_windows → "STORAGE ROOM"          (not eligible,
#                                                          no such SFT class)
#
# The raw detection result (has_windows=True) is still stored in the
# room's attributes block regardless of eligibility, preserving the full
# detector output for non-SFT consumers.
# ---------------------------------------------------------------------------

# Base types that have a "w/ windows" SFT target class.
# Exactly the 9 types listed in the mandatory taxonomy specification.
WINDOW_ELIGIBLE: Set[str] = {
    "PRIVATE OFFICE",
    "OPEN OFFICE",
    "CONFERENCE",
    "MEETING",
    "MULTIPURPOSE ROOM",
    "CLASSROOM",
    "LECTURE HALL",
    "TRAINING ROOM",
    "LOBBY",
}

# Base types that have a "w/ skylights" SFT target class.
SKYLIGHT_ELIGIBLE: Set[str] = {
    "GYMNASIUM",
    "WAREHOUSE",
}

# Base types that have a "w/ side openings" SFT target class.
OPENING_ELIGIBLE: Set[str] = {
    "PARKING GARAGE",
}


# ---------------------------------------------------------------------------
# Normalization functions
# ---------------------------------------------------------------------------

class NormalizationResult(NamedTuple):
    """Result of room type normalization."""
    mandatory_type: str  # Always one of VALID_TYPES (for SFT output)
    extended_type: Optional[str] = None  # One of EXTENDED_ONLY_TYPES if applicable


# Single source of truth: extended class → nearest mandatory class.
# Used by both the exact short-token lookup and the step-5 substring resolution.
_EXTENDED_TO_MANDATORY = {
    "mechanical": "ELECTRICAL ROOM",
    "machine_room": "ELECTRICAL ROOM",
    "boiler": "ELECTRICAL ROOM",
    "pump_room": "ELECTRICAL ROOM",
    "compactor": "STORAGE ROOM",
    "riser": "ELECTRICAL ROOM",
    "elevator": "CORRIDOR",
    "bicycle_storage": "STORAGE ROOM",
    "bedroom": "RESIDENTIAL UNIT",
    "living_room": "RESIDENTIAL UNIT",
    "laundry": "RESIDENTIAL UNIT",
    "garage": "PARKING GARAGE",
    "studio": "RESIDENTIAL UNIT",
    "carpentry": "MULTIPURPOSE ROOM",
    "community_facility": "MULTIPURPOSE ROOM",
    "cctv": "ELECTRICAL ROOM",
    "other": "STORAGE ROOM",
}

# Exact variant string → mandatory class, built once at import.
# Lets short tokens (≤4 chars) that are EXACT members of a mandatory/extended
# variant list (e.g. "BR1", "LR", "0BR") resolve correctly instead of being
# blanket-dumped to STORAGE ROOM by the short-token guard. Exact match only —
# no substring — so noise tokens ("RM", "A1") are unaffected.
_EXACT_VARIANT_TO_MANDATORY: dict = {}
for _mandatory, _variants in MANDATORY_CLASSES.items():
    for _v in _variants:
        _EXACT_VARIANT_TO_MANDATORY.setdefault(_v.upper(), _mandatory)
for _extended, _variants in EXTENDED_TYPES.items():
    _mapped = _EXTENDED_TO_MANDATORY.get(_extended, "STORAGE ROOM")
    for _v in _variants:
        _EXACT_VARIANT_TO_MANDATORY.setdefault(_v.upper(), _mapped)


def normalize_to_mandatory(raw: str) -> str:
    """
    Normalize any raw room type string to a mandatory SFT class.

    Resolution order:
    1. Already a mandatory type → return as-is.
    2. VLM category map → translate directly.
    3. Surface-form exact match (case-insensitive) → map to mandatory.
    4. Substring match in mandatory classes.
    5. Substring match in extended classes → map to nearest mandatory.
    6. Fallback → "STORAGE ROOM" (least-intrusive default).

    Args:
        raw: Raw room type string from any source (OCR, VLM, abbreviation, etc.)

    Returns:
        Mandatory class name from VALID_TYPES, guaranteed to be SFT-compliant.
    """
    if not raw:
        return "STORAGE ROOM"

    raw_stripped = raw.strip()
    raw_lower = raw_stripped.lower()
    raw_upper = raw_stripped.upper()

    # 1. Already a mandatory type
    if raw_upper in VALID_TYPES:
        return raw_upper

    # 2. VLM category map (exact, case-insensitive)
    if raw_lower in VLM_CATEGORY_MAP:
        return VLM_CATEGORY_MAP[raw_lower]

    # 3. Surface-form exact match (mandatory)
    if raw_upper in _SURFACE_TO_MANDATORY:
        return _SURFACE_TO_MANDATORY[raw_upper]

    # 3b. Short-token guard. Before defaulting to STORAGE, try an EXACT lookup
    # against mandatory/extended variant lists so valid short codes (BR1, LR,
    # 0BR, BD) resolve correctly. Exact match only — noise tokens still fall
    # through to STORAGE ROOM.
    if len(raw_upper) <= 4:
        if raw_upper in _EXACT_VARIANT_TO_MANDATORY:
            return _EXACT_VARIANT_TO_MANDATORY[raw_upper]
        logger.debug(f"Short token '{raw}' no exact match → STORAGE ROOM")
        return "STORAGE ROOM"

    # 4. Substring containment in mandatory classes
    for mandatory, variants in MANDATORY_CLASSES.items():
        for variant in variants:
            if variant in raw_upper or raw_upper in variant:
                logger.debug(f"Substring match (mandatory): '{raw}' → '{mandatory}'")
                return mandatory

    # 5. Check extended classes for fallback mapping
    for extended, variants in EXTENDED_TYPES.items():
        for variant in variants:
            if variant in raw_upper or raw_upper in variant:
                logger.debug(
                    f"Substring match (extended): '{raw}' → '{extended}', "
                    f"mapping to nearest mandatory"
                )
                return _EXTENDED_TO_MANDATORY.get(extended, "STORAGE ROOM")

    logger.debug(f"No match for room type: '{raw}' → STORAGE ROOM (fallback)")
    return "STORAGE ROOM"


def get_extended_type(raw: str) -> Optional[str]:
    """
    Get the fine-grained extended type classification.

    This is used for backward compatibility and non-SFT output only.
    For SFT, always use normalize_to_mandatory().

    Args:
        raw: Raw room type string.

    Returns:
        Extended class name from EXTENDED_ONLY_TYPES, or None if not found.
    """
    if not raw:
        return None

    raw_stripped = raw.strip()
    raw_lower = raw_stripped.lower()
    raw_upper = raw_stripped.upper()

    # 1. Already an extended type
    if raw_lower in EXTENDED_ONLY_TYPES:
        return raw_lower

    # 2. Surface-form exact match (extended)
    if raw_upper in _SURFACE_TO_EXTENDED:
        return _SURFACE_TO_EXTENDED[raw_upper]

    # 3. Short-token guard
    if len(raw_upper) <= 4:
        return None

    # 4. Substring match in extended
    for extended, variants in EXTENDED_TYPES.items():
        for variant in variants:
            if variant in raw_upper or raw_upper in variant:
                logger.debug(f"Extended substring match: '{raw}' → '{extended}'")
                return extended

    return None


def get_vlm_categories_string() -> str:
    """Return comma-separated VLM prompt categories."""
    return ", ".join(VLM_PROMPT_CATEGORIES)


def strip_window_suffix(suffixed_type: str) -> str:
    """
    Remove a window/skylight/opening suffix and return the base mandatory type.

    Inverse of add_window_suffix().  Safe to call on types that have no suffix.

    Examples:
        "CONFERENCE w/ windows" → "CONFERENCE"
        "GYMNASIUM w/ skylights" → "GYMNASIUM"
        "CONFERENCE" → "CONFERENCE"

    Args:
        suffixed_type: Type string, optionally containing a window suffix.

    Returns:
        Base mandatory class name.
    """
    if not suffixed_type:
        return suffixed_type
    for suffix in _ALL_SUFFIXES:
        candidate = f" {suffix}"
        if suffixed_type.endswith(candidate):
            return suffixed_type[: -len(candidate)]
    return suffixed_type


def add_window_suffix(
    base_type: str,
    has_windows: bool = False,
    has_skylights: bool = False,
    has_openings: bool = False,
) -> str:
    """
    Compose the final SFT type string from a base mandatory type and window
    attribute flags produced by the window detector.

    Two independent axes are combined here:
      - Room type (semantic): base_type — what the space IS.
      - Window attribute (visual): has_windows / has_skylights / has_openings
        — what the detector found in the space.

    Not every base type has a suffixed variant in the mandatory SFT taxonomy.
    The eligibility sets (WINDOW_ELIGIBLE, SKYLIGHT_ELIGIBLE, OPENING_ELIGIBLE)
    define which base types do.  If a room is not eligible, this function
    returns base_type unchanged even when detection flags are True — meaning
    the detector found windows in a CORRIDOR, but "CORRIDOR w/ windows" is not
    a defined SFT class, so the composed type is still "CORRIDOR".

    The raw detection result is preserved in the room's attributes block
    regardless of eligibility; only the SFT type string is gated here.

    Priority when multiple flags are True: skylights > openings > windows.

    Args:
        base_type:     Mandatory class name (must be in VALID_TYPES).
        has_windows:   True if the window detector found windows in this room.
        has_skylights: True if skylights were detected.
        has_openings:  True if side openings were detected.

    Returns:
        Suffixed string (e.g., "CONFERENCE w/ windows") when base_type is
        eligible and the corresponding flag is True; base_type unchanged
        otherwise.
    """
    if not base_type or base_type not in VALID_TYPES:
        return base_type

    # Priority: skylights > openings > windows > base
    if has_skylights and base_type in SKYLIGHT_ELIGIBLE:
        return f"{base_type} {SKYLIGHT_SUFFIX}"
    if has_openings and base_type in OPENING_ELIGIBLE:
        return f"{base_type} {OPENING_SUFFIX}"
    if has_windows and base_type in WINDOW_ELIGIBLE:
        return f"{base_type} {WINDOW_SUFFIX}"

    return base_type
