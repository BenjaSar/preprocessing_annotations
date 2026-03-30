"""
Canonical Room Taxonomy — Single Source of Truth.

ALL room type lists, normalisation maps, and category whitelists in the
pipeline import from here.  This eliminates the four previously diverging
independent lists that caused:
  - 25 types produced by TaxonomyNormalizer not recognised by QualityChecker
  - 13 VLM output categories silently collapsing to "other"
  - LabelNormalizer using "mechanical_room" while TaxonomyNormalizer used "mechanical"

Usage
-----
    from automation.taxonomy import CANONICAL_TYPES, VALID_TYPES, VLM_CATEGORY_MAP, normalize_room_type
"""

from typing import Dict, List, Set, Optional
import re
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Master taxonomy
# ---------------------------------------------------------------------------
# Key   = canonical type name (used in all annotation JSON files as "type")
# Value = list of surface-form strings that map to this canonical type.
#         Matching is case-insensitive exact match first, then fuzzy fallback.
# ---------------------------------------------------------------------------
CANONICAL_TYPES: Dict[str, List[str]] = {
    # ── Commercial / office ─────────────────────────────────────────────────
    "office": [
        "OFFICE", "EXECUTIVE OFFICE", "INDIVIDUAL OFFICE", "PRIVATE OFFICE",
        "OFFICE SUITE", "OFFICE SPACE", "WORK SPACE", "WORKSTATION",
        "OPEN PLAN WORKSPACE", "OPEN OFFICE", "CUBICLE", "CUBICLE WORKSTATION",
        "ADMINISTRATIVE", "FACULTY OFFICE",
    ],
    "conference_room": [
        "CONFERENCE ROOM", "CONFERENCE", "MEETING ROOM", "MEETING",
        "BOARDROOM", "BOARD ROOM", "TRAINING ROOM", "BREAKOUT SPACE",
        "BREAKOUT ROOM", "CONF ROOM",
    ],
    "suite": [
        "SUITE", "OFFICE SUITE",
    ],

    # ── Circulation ──────────────────────────────────────────────────────────
    "lobby": [
        "LOBBY", "RECEPTION", "RECEPTION AREA", "ENTRANCE", "ENTRY",
        "ENTRANCE VESTIBULE", "ENTRY AREA", "FOYER", "VESTIBULE",
        "LOBBY RECEPTION",
        # Elevator lobby is a waiting area adjacent to elevator shafts,
        # not an elevator shaft itself. Moved from "elevator" per BUG-4.
        "ELEVATOR LOBBY",
        # Waiting areas are functionally reception/lobby spaces
        # (common in supportive housing, medical offices, government buildings)
        "WAITING ROOM", "WAITING", "WAITING AREA",
    ],
    "hallway": [
        "HALLWAY", "CORRIDOR", "PASSAGE", "WALKWAY", "CIRCULATION",
        "HALLWAY/CORRIDOR", "HALL",
    ],
    "elevator": [
        "ELEVATOR", "ELEVATOR CORE", "ELEVATOR AREA",
        "LIFT", "ELEVATOR MACHINE ROOM",
    ],
    "stairwell": [
        "STAIRWELL", "STAIR", "STAIRS", "STAIRCASE",
        "EMERGENCY STAIRS", "EXIT STAIRS", "FIRE STAIR",
    ],

    # ── Sanitary ─────────────────────────────────────────────────────────────
    "restroom": [
        "RESTROOM", "BATHROOM", "MEN'S RESTROOM", "WOMEN'S RESTROOM",
        "MEN'S BATHROOM", "WOMEN'S BATHROOM", "MEN'S LOCKER",
        "WOMEN'S LOCKER", "TOILET", "WC", "POWDER ROOM", "LAVATORY",
        "MEN'S BATHROO",  # common OCR truncation
    ],

    # ── Food / break ─────────────────────────────────────────────────────────
    "kitchen": [
        "KITCHEN", "BREAK ROOM", "BREAKROOM", "KITCHEN BREAK ROOM",
        "KITCHENETTE", "PANTRY", "CAFE", "CAFETERIA", "DINING",
        "DINING ROOM",
    ],

    # ── Utility / MEP rooms ──────────────────────────────────────────────────
    "mechanical": [
        "MECHANICAL ROOM", "MECHANICAL", "MECH ROOM", "MECHANICAL/ELECTRICAL ROOM",
        "HVAC ROOM", "BUILDING SYSTEMS",
    ],
    "electrical": [
        "ELECTRICAL ROOM", "ELECTRICAL", "ELECTRICAL/MECHANICAL ROOM",
        "ELECTRIC ROOM",
    ],
    "server_room": [
        "SERVER ROOM", "SERVER", "DATA CENTER", "IDF", "MDF",
        "TELECOM ROOM", "TELECOM", "NETWORK ROOM", "IT ROOM",
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
    "custodial": [
        "CUSTODIAL", "CUSTODIAL CLOSET", "JANITOR", "JANITOR ROOM",
        "JANITOR CLOSET", "CLEANING CLOSET",
    ],
    "riser": [
        "RISER ROOM", "RISER", "PIPE CHASE",
    ],

    # ── Storage ──────────────────────────────────────────────────────────────
    "storage": [
        "STORAGE", "STORAGE ROOM", "STORE ROOM", "STOREROOM",
        "BUILDING STORAGE", "COMMERCIAL STORAGE", "GENERAL STORAGE",
        "SUPPLY ROOM", "ARCHIVES", "FILE ROOM", "CLOSET",
        "LINEN CLOSET", "WALK-IN CLOSET",
        # Fix 3: Compound slash-delimited names preserved intact
        "IT/STORAGE/CONFERENCE", "IT/STORAGE",
    ],
    "bicycle_storage": [
        "BICYCLE STORAGE", "BICYCLE ROOM", "BICYCLE", "BIKE STORAGE",
        "BIKE ROOM",
    ],

    # ── Specialised commercial ────────────────────────────────────────────────
    "auditorium": [
        "AUDITORIUM", "LECTURE HALL", "THEATER", "THEATRE",
        "ASSEMBLY ROOM",
    ],
    "classroom": [
        "CLASSROOM", "CLASS ROOM", "LAB", "LABORATORY",
        "COMPUTER LAB", "SCIENCE LAB",
    ],
    "carpentry": [
        "CARPENTRY", "CARPENTRY SHOP", "WORKSHOP", "WOOD SHOP", "SHOP",
    ],
    "community_facility": [
        "COMMUNITY FACILITY", "COMMUNITY ROOM", "AMENITY",
    ],
    "cctv": [
        "CCTV ROOM", "CCTV", "SECURITY ROOM", "SECURITY",
    ],

    # ── Residential ──────────────────────────────────────────────────────────
    "bedroom": [
        "BEDROOM", "MASTER BEDROOM", "BEDROOM 1", "BEDROOM 2", "BEDROOM 3",
        "1 BEDROOM", "2 BEDROOM", "3 BEDROOM",
        # Abbreviations — declared here so _SURFACE_TO_CANONICAL catches them
        # at Step 3 (exact match) BEFORE the substring scan reaches
        # conference_room, whose surface form "BREAKOUT SPACE" contains "BR".
        "BR", "BR1", "BR2", "BR3",
        "MBR", "MSTR BR", "MSTR", "BDRM",
        "1BR", "2BR", "3BR", "4BR",
        "1 BR", "2 BR", "3 BR", "4 BR",
        "MASTER BR",
    ],
    "living_room": [
        "LIVING ROOM", "LIVING", "GREAT ROOM", "FAMILY ROOM",
        # Abbreviations
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
        # 0-bedroom unit = studio
        "0BR", "0 BR",
    ],

    # ── Catch-all ─────────────────────────────────────────────────────────────
    "other": [
        "ROOM", "SPACE", "AREA", "MISC", "OTHER",
        "UNKNOWN", "UNDEFINED",
    ],
}

# ---------------------------------------------------------------------------
# Derived lookup structures (computed once at import time)
# ---------------------------------------------------------------------------

# Flat set of all valid canonical type names — used by QualityChecker whitelist
VALID_TYPES: Set[str] = set(CANONICAL_TYPES.keys())

# Reverse map: surface form (uppercase) → canonical type
_SURFACE_TO_CANONICAL: Dict[str, str] = {}
for _canonical, _variants in CANONICAL_TYPES.items():
    for _v in _variants:
        _SURFACE_TO_CANONICAL[_v.upper()] = _canonical

# ---------------------------------------------------------------------------
# VLM output category → canonical type
# ---------------------------------------------------------------------------
# Claude's VLM prompt uses its own category vocabulary.  This map translates
# VLM outputs to canonical types so they survive normalisation unchanged.
VLM_CATEGORY_MAP: Dict[str, str] = {
    # VLM name                  → canonical
    "office":                     "office",
    "open_plan_workspace":        "office",
    "executive_office":           "office",
    "cubicle_workstation":        "office",
    "conference_room":            "conference_room",
    "meeting_room":               "conference_room",
    "training_room":              "conference_room",
    "breakout_space":             "conference_room",
    "lobby_reception":            "lobby",
    "hallway_corridor":           "hallway",
    "restroom":                   "restroom",
    "kitchen_break_room":         "kitchen",
    "dining_room":                "kitchen",
    "family_room":                "living_room",
    "janitor":                    "custodial",
    "closet":                     "storage",
    # Residential unit-type abbreviations used in residential floor plans
    "0br":                        "studio",
    "1br":                        "bedroom",
    "2br":                        "bedroom",
    "3br":                        "bedroom",
    "4br":                        "bedroom",
    "br":                         "bedroom",
    "lr":                         "living_room",
    "lv":                         "living_room",
    "mbr":                        "bedroom",
    "storage":                    "storage",
    "mechanical_room":            "mechanical",
    "mechanical":                 "mechanical",
    "electrical_room":            "electrical",
    "electrical":                 "electrical",
    "elevator":                   "elevator",
    "elevator_lift":              "elevator",
    "stairwell":                  "stairwell",
    "stairwell_stairs":           "stairwell",
    "auditorium":                 "auditorium",
    "data_center":                "server_room",
    "server_room":                "server_room",
    "carpentry":                  "carpentry",
    "storage_room":               "storage",
    "break_room":                 "kitchen",
    "cafeteria":                  "kitchen",
    "administrative":             "office",
    "hallway":                    "hallway",
    "lobby":                      "lobby",
    "kitchen":                    "kitchen",
    "other":                      "other",
}

# ---------------------------------------------------------------------------
# VLM prompt category list
# ---------------------------------------------------------------------------
# Flat list of categories sent to Claude in the VLM prompt.
# Using canonical names so VLM output directly maps to canonical types
# without an intermediate translation step.
VLM_PROMPT_CATEGORIES: List[str] = [
    "office",
    "conference_room",
    "lobby",
    "hallway",
    "restroom",
    "kitchen",
    "storage",
    "mechanical",
    "electrical",
    "server_room",
    "elevator",
    "stairwell",
    "auditorium",
    "classroom",
    "carpentry",
    "bedroom",
    "living_room",
    "bicycle_storage",
    "compactor",
    "pump_room",
    "machine_room",
    "custodial",
    "community_facility",
    "other",
]

# ---------------------------------------------------------------------------
# Normalisation function
# ---------------------------------------------------------------------------

def normalize_room_type(raw: str) -> str:
    """
    Normalise any raw room type string to a canonical type.

    Resolution order:
    1. Already a canonical type → return as-is.
    2. VLM category map → translate directly.
    3. Surface-form exact match (case-insensitive) → map to canonical.
    4. Substring match → return first canonical type whose surface forms
       contain the raw string as a substring.
    5. Fallback → "other".

    Args:
        raw: Raw room type string from any source.

    Returns:
        Canonical type name from VALID_TYPES.
    """
    if not raw:
        return "other"

    raw_stripped = raw.strip()
    raw_lower = raw_stripped.lower()
    raw_upper = raw_stripped.upper()

    # 1. Already canonical
    if raw_lower in VALID_TYPES:
        return raw_lower

    # 2. VLM category map (exact, case-insensitive)
    if raw_lower in VLM_CATEGORY_MAP:
        return VLM_CATEGORY_MAP[raw_lower]

    # 3. Surface-form exact match
    if raw_upper in _SURFACE_TO_CANONICAL:
        return _SURFACE_TO_CANONICAL[raw_upper]

    # 3b. Short-token guard: tokens of ≤4 characters that are not in the
    #     exact map cannot be reliably resolved by substring containment.
    #     The scan was designed for partial-word fuzzy matching of full phrases
    #     ("MECH ROOM" vs "MECHANICAL"), not two-letter abbreviations.
    #     Without this guard, "BR" matches "BREAKOUT SPACE" (conference_room)
    #     because Python's `"BR" in "BREAKOUT SPACE"` is True.
    #     Any abbreviation that should be recognized MUST be added to
    #     CANONICAL_TYPES surface forms above so it is caught at Step 3.
    if len(raw_upper) <= 4:
        logger.debug(f"Short token '{raw}' not in exact map → 'other' (add to CANONICAL_TYPES if needed)")
        return "other"

    # 4. Substring containment: check if any canonical surface form is
    #    contained in raw_upper OR raw_upper is contained in a surface form.
    for canonical, variants in CANONICAL_TYPES.items():
        for variant in variants:
            if variant in raw_upper or raw_upper in variant:
                logger.debug(f"Substring match: '{raw}' → '{canonical}' via '{variant}'")
                return canonical

    logger.debug(f"No match for room type: '{raw}' → 'other'")
    return "other"


def get_vlm_categories_string() -> str:
    """Return comma-separated VLM prompt categories."""
    return ", ".join(VLM_PROMPT_CATEGORIES)
