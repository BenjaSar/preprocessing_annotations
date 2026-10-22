"""
SFT-Grade Data Validation for Room/Space Annotations.

Ensures annotations are production-ready for supervised fine-tuning of VLMs.
Filters non-spatial text, validates taxonomy, enforces confidence thresholds.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from PIL import Image
import io
import logging

from .duplicate_boxes import split_redundant_room_extents

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from .taxonomy import (
        MANDATORY_CLASSES, VALID_TYPES, normalize_to_mandatory,
        get_extended_type, strip_window_suffix, resolve_exact_short_variant,
    )
    from .abbreviations import ABBREVIATION_MAP as _ABBREVIATION_MAP
    from .abbreviations import expand_abbreviation as _expand_abbreviation_shared
    from .annotation_schema import box_convention_of
    from ..detection.window_evidence import is_window_shaped
    from ..config import WindowEvidenceConfig, RoomFilterConfig
except ImportError:
    from taxonomy import (
        MANDATORY_CLASSES, VALID_TYPES, normalize_to_mandatory,
        get_extended_type, strip_window_suffix, resolve_exact_short_variant,
    )
    from abbreviations import ABBREVIATION_MAP as _ABBREVIATION_MAP
    from abbreviations import expand_abbreviation as _expand_abbreviation_shared
    from annotation_schema import box_convention_of
    from detection.window_evidence import is_window_shaped
    from config import WindowEvidenceConfig, RoomFilterConfig


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
        # Bare "GARAGE" alone is a fixture/zone legend token (VLM misread on
        # office/school/residential floors), not a room; it was mapping to
        # PARKING GARAGE. Real parking spaces use "PARKING"/"PARKING GARAGE",
        # which are unaffected. Anchored so only the standalone token is dropped.
        r"^GARAGE$",
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
        # School-sheet keywords — verified via live OCR re-run (Masonic Heights,
        # Violet Elementary): reached the keyword gate and were filtered
        # because the underlying word was absent from this whitelist, even
        # though taxonomy.py already had a mapping for the full label.
        "SUPPLY",       # "SUPPLY" (room label) → STORAGE ROOM
        "PRINCIPAL",    # "PRINCIPAL" (office) → PRIVATE OFFICE
        "TEACHER",      # "TEACHER LOUNGE" / "TEACHER WORK ROOM" → CAFETERIA
        "COMPUTER",     # "COMPUTER LAB" → CLASSROOM
        # User-confirmed omitted room labels (sprint38 review) — real labels
        # reaching the keyword gate and filtered because the underlying word
        # was absent from this whitelist.
        "WORK",         # "WORK RM" → STORAGE ROOM
        "SECRETARY",    # "SECRETARY" → STORAGE ROOM
        "MEN'S", "WOMEN'S",  # → RESTROOM
        "VENDING",      # "VENDING" → STORAGE ROOM
        "VAULT",        # "VAULT" → STORAGE ROOM
        "ELECTRICAL",   # "ELECTRICAL ROOM" → ELECTRICAL ROOM (was missing entirely)
        "FITNESS",      # "FITNESS ROOM" → STORAGE ROOM
        "YOGA", "MEDITATION",  # "YOGA/MEDITATION ROOM" → STORAGE ROOM
        "MAIL",         # "MAIL ROOM" → STORAGE ROOM
        # User-confirmed room labels absent from this whitelist, so the keyword
        # gate dropped them even though taxonomy now maps each one. Compound
        # forms are used where the single word is too generic to gate on
        # (e.g. "PLAY ROOM", not "PLAY").
        "SOCIAL SERVICE", "SOCIAL SERVICES",  # → MULTIPURPOSE ROOM
        "RECREATION",                          # → MULTIPURPOSE ROOM
        "PLAY ROOM",                           # "CHILDREN'S PLAY ROOM" → MULTIPURPOSE ROOM
        # R3 (2026-09-14): bare "STAIR" / "STAIR #N" reached this gate and was
        # filtered even though taxonomy.py's STAIRWELL synonym list already
        # includes "STAIR" (line ~169) — this stage was simply missing the
        # word the pipeline's own downstream normalizer already recognizes.
        # Verified on 4 real-gap pages (220301_Elect_plans_05 pages 3/5/6/7):
        # "STAIR #3", "STAIR #1", bare "STAIR" — all real stairwell labels.
        "STAIR",                               # → STAIRWELL
        # R3 continued (2026-09-14): short abbreviations on the same
        # 220301_Elect_plans_05 sheet set, each paired with a taxonomy.py
        # variant-list addition (see that file) so they resolve to the
        # correct type, not just pass this gate. Corpus-scanned for false
        # positives first: REST/ELEV/UTIL hits are ~all on this one document
        # family, each followed by a room number or "#N" (ELEV #1, UTIL 301,
        # REST 4-2) — real MEP-labeled rooms, not stray abbreviation noise.
        "REST",         # "REST 4-2" → RESTROOM
        "ELEV",         # "ELEV #1" → CORRIDOR (elevator shaft, per existing convention)
        "UTIL",         # "UTIL 301" → RESIDENTIAL UNIT (per existing UTILITY→laundry mapping)
        # Track 2 pilot (2026-09-15): first cannabis-cultivation-facility
        # drawings seen in this corpus ("Crooked Creek Bloom/Veg Room" PDFs).
        # "BLOOM ROOM"/"VEG ROOM" are standard grow-facility room types
        # (flowering room / vegetative room), same shape as any other
        # too-generic-alone compound (cf. "PLAY ROOM"). "BLOOM RM" is the
        # abbreviated form seen on the same page -- _expand_abbreviation only
        # expands from the START of a string, so "BLOOM RM 1" never reaches
        # "BLOOM ROOM" and needs its own entry.
        "BLOOM ROOM", "BLOOM RM",   # → OTHER (grow-facility flowering room)
        "VEG ROOM",                 # → OTHER (grow-facility vegetative room)
        # R3 side-finding, resolved (2026-09-14): "R.R." abbreviation for
        # restroom ("Men's R.R.", "224 R.R.", "STAFF RR", "RR 204") was
        # falling through taxonomy's least-intrusive default to STORAGE ROOM
        # (or, for the bare-"RR" forms below, failing this gate outright).
        # Corpus-scanned exhaustively (every "R.R"/"RR" occurrence in all 141
        # pages, not just the drop population): 42 hits, 100% restroom
        # context (gendered/staff/patient/unisex/room-number/vestibule
        # prefix) — zero railroad/rural-route/address ambiguity anywhere in
        # this corpus. Dotted "R.R." handled by a VALID_ROOM_REGEXES entry
        # below (bare "RR" as a keyword hits a word-boundary regex dead end
        # on trailing punctuation — verified before shipping, not assumed).
        "RR",           # "MEN RR" / "STAFF RR" / "RR 204" → RESTROOM
        # Second-gate sync (2026-09-17): this keyword set gates BOTH OCR- and
        # VLM-sourced room candidates (VLM proposes names directly, bypassing
        # config.py's room_name_patterns entirely), so it's the more
        # consequential of the two vocabulary gates. Confirmed via real pilot
        # run (track2_pilot_output_v2): "CAR WASH"/"DETAIL AREA"/
        # "TRASH ENCLOSURE" (VLM-sourced, E-10.0 REFLECTED CEILING PLANS - CAR
        # WASH & DETAIL AREA) all reached this gate and were dropped, taking
        # the whole page down to 0 rooms and misclassifying it
        # sft_ready=false "non-floorplan page" even though it's a real
        # floorplan. Same session's earlier config.py room_name_patterns
        # additions (bank/dealership/clinic vocabulary — see
        # showroom-triage-vocabulary-gap and Rally Palmhurst bank-vocabulary
        # fixes) never synced here; CAFE/LAB/RESIDENTIAL happened to already
        # be present, the rest below were not.
        "SHOWROOM",
        "ZOOM",
        "CAR WASH",
        "TRASH ENCLOSURE",
        "DETAIL AREA",
        "EQUIPMENT ROOM",   # bare "EQUIPMENT" stays banned via NON_ROOM_PATTERNS;
                            # confirmed "EQUIPMENT ROOM" itself matches none of those
        "TRIAGE",
        "DRAW",
        "NURSE STATION",
        "EXAM RM",
        "PROCEDURE",
        "INTAKE",
        "CASHIER",
        "WORKROOM",         # bare "WORK" (line 287) doesn't cover this: \bWORK\b
                            # needs a boundary after K, absent in "WORKROOM"
        "TELLER LINE",
        "MOTHER'S ROOM",
        "NIGHT DROP/ITM",
        # N3 (2026-10-10): GT-confirmed vocabulary gaps on the 48-page
        # target-domain eval (room_gt_eval_sprint1_verify51.json).
        # CAFETERIA is already in taxonomy.py's CAFETERIA class but the
        # keyword gate pattern \bCAFE\b does NOT match CAFETERIA — verified
        # via re.search(r"\bCAFE\b", "CAFETERIA") → no match. Add the
        # full word so "CAFETERIA" passes this gate and reaches taxonomy.
        "CAFETERIA",
        # MEDIA CENTER: 8/10 candidates inside GT media_center polygons.
        # Maps to CLASSROOM via taxonomy.py (added same session).
        "MEDIA CENTER",
        # MDF/IDF: 3/4 and 2/3 respectively inside GT polygons.
        # Maps to ELECTRICAL ROOM via taxonomy.py (added same session).
        "MDF", "IDF",
    }

    # Generalized label forms a fixed keyword cannot express because they carry
    # a variable part (a room number). Kept separate from VALID_ROOM_KEYWORDS,
    # which is joined into a word-alternation group, because these supply their
    # own quantifiers.
    #
    # Deliberately NOT the bare word "ROOM": that would let any text containing
    # "ROOM" satisfy the gate, reopening exactly the substring leak the
    # word-boundary pattern below was introduced to close. Fully anchored for
    # the same reason — an unanchored form admitted note text such as
    # "SEE ROOM 2 SCHEDULE" (verified).
    VALID_ROOM_REGEXES = (
        r"^ROOM\s*\d+$",
        # R3 (2026-09-14): apartment unit-type-code + unit-number labels
        # ("1.1R 229", "2.1R 227", "1.1 217", "2.1 214") — the box literally
        # stacks a unit-type code over a unit number on the source drawing
        # (verified visually, 220301_Elect_plans_05_page003, corridor of
        # repeating residential units with bathroom fixtures). Corpus-scanned:
        # all 72 matches confined to this one document's 5 pages, all paired
        # with a plausible 2-4 digit room number; zero matches anywhere else
        # in the 141-page corpus (structural grid-line callouts like "A-6"
        # use a different letter-first format, don't collide).
        r"^\d\.\d[A-Z]?\s*\d{2,4}$",
        # R3 side-finding, resolved (2026-09-14): dotted "R.R." restroom
        # abbreviation ("Men's R.R.", "224 R.R.", "NON-COMPLIANT R.R."). Not
        # placed in VALID_ROOM_KEYWORDS: a trailing-punctuation token wrapped
        # in \b...\b never matches (verified: \bR\.R\.\b fails on every real
        # sample — both sides of the boundary after a trailing "." followed
        # by whitespace/end-of-string are non-word, so no transition exists).
        # Leading boundary only, deliberately not anchored full-string
        # (prefix text varies: "224 ", "231 Staff Men ", "NON-COMPLIANT ") —
        # corpus-scanned exhaustively (all 141 pages, every "R.R"/"RR"
        # occurrence, not just the drop population): 42 hits, 100% restroom
        # context, zero railroad/rural-route ambiguity.
        r"\bR\.R\.?",
        # Second-gate sync (2026-09-17), same reasoning as room_name_patterns'
        # matching entries added earlier this session: real OCR produced more
        # than one surface form for each, so a plain VALID_ROOM_KEYWORDS
        # literal isn't enough.
        r"\bCUSTOMER\s*R\.?\s*R\.?",   # "CUSTOMER R.R.", "CUSTOMER RR", OCR-merged "CUSTOMERRR"
        r"\bMECH\s*/\s*ELEC\b",        # "MECH/ELEC", "MECH /ELEC"
    )

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

    def _expand_abbreviation(self, name: str, block_single_letter: bool = True) -> str:
        """
        Expand abbreviated room names to canonical forms.

        Critical for recognizing short room identifiers like BR, LR, BA.

        Args:
            name: Room name (potentially abbreviated)
            block_single_letter: E2 (2026-09-09). When True (default), a
                single-letter abbreviation ("K", "P", "G" in
                ABBREVIATION_MAP) is never expanded here. Measured on
                corpus_batch2 (88 pages): every one of 30 bare "K" room
                candidates was a false positive on visual inspection
                (fixture-type tag on a wire, a structural grid-line label,
                or a schedule-table column-header letter) -- zero were a
                real kitchen. "K"'s expansion to "KITCHEN" is the only
                reason a bare single letter passes the room-keyword gate at
                all (a bare "A"/"B"/etc. matches nothing and is dropped
                harmlessly elsewhere). Multi-character abbreviations (BR,
                WC, 1BR, ...) are unaffected -- those were checked
                individually against real crops and are unambiguous.
                Set False only for a corpus confirmed to use bare
                single-letter room labels (see plan/single-letter-label-gate-plan-2026-09-09.md).

        Returns:
            Expanded canonical room name, or original if not an abbreviation
            (including a blocked single-letter abbreviation).
        """
        expanded = _expand_abbreviation_shared(name, block_single_letter=block_single_letter)
        if expanded != name:
            logger.debug(f"Expanded abbreviation: '{name}' → '{expanded}'")
        return expanded

    def filter_rooms(self, rooms: List[Dict], block_single_letter_abbreviations: bool = True) -> List[Dict]:
        """
        Filter rooms, keeping only valid spatial annotations.

        CRITICAL: VLM uses "room_name", OCR uses "name". Must check both.

        Args:
            rooms: List of room dictionaries with 'name'/'room_name' and 'bbox' keys
            block_single_letter_abbreviations: E2 (2026-09-09). Passed through
                to _expand_abbreviation -- see that method's docstring.

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
            expanded_name = self._expand_abbreviation(
                name, block_single_letter=block_single_letter_abbreviations
            )

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
#
# CLASSROOM alone gets a widened trailing boundary (plain \b OR immediate
# digit) to match the concatenated room-number form school sheets emit with
# no separator ("CLASSROOM5", "CLASSROOM17") — verified via live OCR. This is
# scoped to CLASSROOM only, not all keywords: applying it globally let
# "SUITE" + digit match "SUITE301", reviving a known VLM-hallucinated giant
# box (verified: spans an entire floor, was previously — coincidentally —
# blocked by this same gate). Every other keyword keeps the strict \b.
_sorted_kws = sorted(SemanticRoomValidator.VALID_ROOM_KEYWORDS - {"CLASSROOM"}, key=len, reverse=True)
_kw_alternation = "|".join(re.escape(kw) for kw in _sorted_kws)
# VALID_ROOM_REGEXES contribute their own anchors/quantifiers, so they are
# appended as separate alternatives rather than wrapped in the \b(?:...)\b group.
SemanticRoomValidator._KW_PATTERN = re.compile(
    "|".join(
        (
            r"\bCLASSROOM(?:\b|(?=\d))",
            r"\b(?:" + _kw_alternation + r")\b",
            *SemanticRoomValidator.VALID_ROOM_REGEXES,
        )
    ),
    re.IGNORECASE,
)


class TaxonomyNormalizer:
    """Normalize room names to SFT mandatory taxonomy."""

    # Single source of truth — imported from abbreviations.py.
    # Previously a separate dict that diverged from SemanticRoomValidator
    # and ResidentialAbbreviationRecovery.
    ABBREVIATION_EXPANSIONS = _ABBREVIATION_MAP

    def _expand_abbreviation(self, name: str, block_single_letter: bool = True) -> str:
        """
        Expand abbreviated room names to canonical forms.

        Args:
            name: Room name (potentially abbreviated)
            block_single_letter: E2 (2026-09-09) -- see SemanticRoomValidator's
                sibling method for the full measured rationale. Default True.

        Returns:
            Expanded canonical room name, or original if not an abbreviation
            (including a blocked single-letter abbreviation).
        """
        return _expand_abbreviation_shared(name, block_single_letter=block_single_letter)

    def normalize(self, room_name: str, block_single_letter_abbreviations: bool = True) -> str:
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

        # Try the raw token against taxonomy.py's own exact short-code table
        # FIRST, before abbreviation expansion. Some short codes (e.g. "DR")
        # are deliberately placed under one mandatory class directly in
        # taxonomy.py, but abbreviations.py expands them to a phrase that
        # resolves under a DIFFERENT class ("DR" -> "DINING ROOM" -> RESTAURANT,
        # vs. taxonomy.py's own "DR" -> RESIDENTIAL UNIT). Letting the raw
        # exact match win here honors taxonomy.py's explicit placement.
        exact = resolve_exact_short_variant(room_name)
        if exact is not None:
            return exact

        # _expand_abbreviation existed on this class but was never called here,
        # so a bare abbreviation (e.g. "CL") reached normalize_to_mandatory()
        # unexpanded, hit the short-token guard, and fell back to STORAGE ROOM
        # instead of resolving through its real expansion ("CLOSET").
        return normalize_to_mandatory(
            self._expand_abbreviation(room_name, block_single_letter=block_single_letter_abbreviations)
        )


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

    # 'vestibule' listed since VESTIBULE became its own class (2026-10-13):
    # these rooms were exempt here as LOBBY, and splitting the class must not
    # silently expose them to overlap resolution. T-ASPECT's _CIRC already
    # treats vestibule as circulation.
    circulation_types = {'hallway', 'corridor', 'lobby', 'elevator', 'stairwell',
                         'riser', 'vestibule'}
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





def _room_desc(room: Dict) -> Dict[str, Any]:
    """Minimal per-room record stored in drop_detail."""
    return {
        "room_name": room.get("room_name") or room.get("name"),
        "bbox": room.get("bbox"),
        "source": room.get("source"),
    }


@dataclass
class MergeResult:
    """Merged rooms plus the candidates the merge discarded, keyed by reason."""

    rooms: List[Dict]
    discards: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def record(self, reason: str, room: Dict) -> None:
        self.discards.setdefault(reason, []).append(_room_desc(room))


def _is_vlm_echo(ocr_room: Dict, vlm_rooms: List[Dict], tolerance_px: int) -> bool:
    """True when an OCR candidate's box equals a VLM room's box within
    tolerance. A text-OCR box cannot coincide with a VLM room extent, so such
    a candidate is that VLM detection seen twice (its label may differ)."""
    ocr_bbox = ocr_room.get("bbox", [])
    return bool(ocr_bbox) and any(
        len(vlm.get("bbox", [])) == len(ocr_bbox)
        and all(abs(a - b) <= tolerance_px for a, b in zip(ocr_bbox, vlm["bbox"]))
        for vlm in vlm_rooms
    )


def _same_label(ocr_name: str, vlm_name: str) -> bool:
    """True when both detectors produced the same room label.

    Compared case-insensitively on stripped text: the two sources normalise
    whitespace and case differently, so a raw `==` would miss real agreement.
    Empty strings never count as agreement.
    """
    if not ocr_name or not vlm_name:
        return False
    return ocr_name.strip().upper() == vlm_name.strip().upper()


def _adopt_ocr_geometry(target: Dict, ocr_room: Dict, ocr_name: str,
                        ocr_bbox: List) -> None:
    """Promote `target` to a VLM+OCR corroborated room carrying the OCR geometry.

    FP-1: the OCR bbox — not the VLM one — becomes the room geometry. VLM boxes
    on this pipeline are frequently oversized/mislocated (measured: a single VLM
    box matching 5-21 distinct OCR labels); the OCR bbox is the precise label
    location, which SAM (Step 3.5) later expands to room scale.

    Geometry provenance must follow the geometry: bbox now comes from ocr_room,
    so original_bbox/sam_* must too. Without this the fields keep the VLM room's
    own (unrelated, also-stale) pre-SAM values — wrong provenance for any
    downstream check on the OCR-sourced box's SAM history.
    """
    target["vlm_bbox"] = target.get("bbox")
    target["bbox"] = ocr_bbox
    target["name"] = ocr_name
    target["room_name"] = ocr_name
    target["ocr_label"] = ocr_name
    target["source"] = "vlm_ocr_merged"
    target["original_bbox"] = ocr_room.get("original_bbox")
    target["sam_expanded"] = ocr_room.get("sam_expanded")
    target["sam_skip_reason"] = ocr_room.get("sam_skip_reason")
    target["sam_confidence"] = ocr_room.get("sam_confidence")
    # B1 (2026-08-21 audit): content_density is attached per-room by SAM
    # BEFORE this merge (pipeline.py runs refine_annotations separately on
    # vlm_rooms and ocr_rooms), so target's own value describes the VLM
    # box being discarded above, not ocr_room's box now becoming the
    # geometry. Measured 16/16 (100%) of vlm_ocr_merged rooms on a real
    # 3-page run carried a stale density (max drift 0.2848) before this
    # fix -- same provenance-follows-geometry rule as sam_* below.
    target["content_density"] = ocr_room.get("content_density")
    # Same provenance rule as the sam_* fields above: the polygon is derived
    # from ocr_room's bbox (SAM segments the OCR label, not the VLM box), so
    # it must travel with the geometry. Missing this silently demoted every
    # vlm_ocr_merged room to the bbox-rectangle export fallback even when SAM
    # succeeded -- found by comparing a real pipeline run's output against an
    # isolated refine_annotations() call on the same image (T2, 2026-08-06).
    if "segmentation" in ocr_room:
        target["segmentation"] = ocr_room["segmentation"]
    else:
        target.pop("segmentation", None)


def _merge_vlm_and_ocr(
    vlm_rooms: List[Dict], ocr_rooms: List[Dict], proximity_px: int
) -> MergeResult:
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
        MergeResult: merged unique rooms, plus candidates discarded by the
        merge (OCR labels skipped as fragments; VLM boxes whose geometry was
        replaced by an OCR box) so every merge loss is attributable.
    """
    merged = []
    result = MergeResult(rooms=merged)

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
            if _bbox_distance(ocr_bbox, vlm_bbox, threshold=proximity_px):
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
                result.record("merge_vlm_geometry_replaced", merged[matched_vlm])
                _adopt_ocr_geometry(merged[matched_vlm], ocr_room, ocr_name, ocr_bbox)
            elif _same_label(ocr_name, vlm_name):
                # FM-2: VLM and OCR independently produced the SAME label for the
                # same box. That is the strongest corroboration available, but it
                # was previously discarded by the fragment branch below (an equal
                # string is trivially a substring), leaving the room `vlm_only`
                # and so unconditionally dropped by FIX-7 despite two agreeing
                # detectors. No second room is added — the duplicate is still
                # collapsed — only the corroboration is now recorded.
                logger.debug(
                    f"FM-2: VLM '{vlm_name}' corroborated by identical OCR label "
                    f"→ vlm_ocr_merged"
                )
                result.record("merge_vlm_geometry_replaced", merged[matched_vlm])
                _adopt_ocr_geometry(merged[matched_vlm], ocr_room, ocr_name, ocr_bbox)
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
                    result.record("merge_fragment_skip", ocr_room)
                    logger.debug(f"Merged: VLM '{vlm_name}' (OCR '{ocr_name}' skipped, fragment/empty)")
        else:
            # OCR room doesn't overlap with VLM → add as new room
            logger.debug(f"Added OCR room (no VLM match): {ocr_name}")
            r = dict(ocr_room)
            r.setdefault("source", "ocr_only")
            merged.append(r)

    logger.info(f"Merged {len(vlm_rooms)} VLM + {len(ocr_rooms)} OCR → {len(merged)} total")
    return result


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
# The two thresholds (RoomFilterConfig.ink_binarise_max / min_ink_frac) and the
# 18-file audit they were derived from are documented on that config.


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
    rooms: List[Dict],
    gray: Image.Image,
    min_frac: float,
    binarise_max: int,
    exempt_sam_expanded: bool = False,
) -> Tuple[List[Dict], int]:
    """Drop rooms whose box lies over a non-drawing region (ink below min_frac).

    Boxes that cannot be measured (malformed bbox, out of image) are kept — this
    filter only removes positively-confirmed low-ink placements.

    exempt_sam_expanded (A1, 2026-08-21): when True, rooms with
    room["sam_expanded"] truthy skip this check entirely -- measured to have
    no discriminative power for that population (see
    PipelineConfig.exempt_sam_expanded_from_ink_filter for the numbers).
    Non-expanded rooms are unaffected either way.
    """
    kept: List[Dict] = []
    dropped = 0
    for room in rooms:
        if exempt_sam_expanded and room.get("sam_expanded"):
            kept.append(room)
            continue
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

# T-C1 SAM-over-expansion revert ceiling. SAM sometimes expands a correctly-
# placed tiny OCR label into a giant multi-room/floor-plate blob; when the
# expanded box exceeds this fraction of the image it is reverted to the pre-SAM
# located label (original_bbox). Data-derived (measured over 200+ sam_expanded
# rooms, sprint36+38): normal expansions top out at 0.91% of image; every
# over-expansion giant was >= 1.63%. Cut placed in that empty gap. Replaces the
# previous inline 0.08 ceiling, which was ~6x too loose and let 9 giants
# (1.63-7.72%) survive.
SAM_OVEREXPAND_MAX_FRAC = 0.012

# Exclusion-zone overlap cuts (FIX-4 rooms, FIX-9 objectDetections). A box is
# treated as mislocalized into a BOM/title-block zone when either direction of
# overlap is heavy enough: the zone covers this fraction of the BOX's own area,
# or the box covers this fraction of the ZONE's area. Both directions matter —
# see the T-OVERLAP note in FIX-4. Values are the ones FIX-4 has used inline
# since T-OVERLAP; named here so rooms and object detections share one source of
# truth instead of two drifting copies.
ZONE_OVERLAP_BOX_FRAC = 0.3
ZONE_OVERLAP_ZONE_FRAC = 0.4


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
    use_window_elongation_filter: bool = False,
    exempt_sam_expanded_from_ink_filter: bool = False,
    block_single_letter_abbreviations: bool = True,
    room_filter: Optional["RoomFilterConfig"] = None,
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
        use_window_elongation_filter: BA3 (2026-08-03). When True, drops
            objectDetections tagged "window" whose bbox fails the elongation
            predicate (window_evidence.is_window_shaped) -- catches door-swing
            symbols mislabeled window (near-square boxes; real windows are
            elongated, portrait or landscape). Default False: off, same
            precedent as every other detector-behavior flag in this pipeline.
        exempt_sam_expanded_from_ink_filter: A1 (2026-08-21). When True,
            FIX-5's ink-density check (_filter_low_ink_rooms) skips rooms
            with sam_expanded truthy -- measured to have no discriminative
            power for that population (see PipelineConfig's docstring for
            the numbers). Default False pending validation.
        block_single_letter_abbreviations: E2 (2026-09-09). When True
            (default), a single-letter abbreviation ("K" -> KITCHEN, "P" ->
            PANTRY, "G" -> GARAGE in ABBREVIATION_MAP) is never expanded, so
            it cannot pass the room-keyword gate. Measured on corpus_batch2:
            30/30 bare "K" room candidates in 88 pages were false positives
            on visual inspection (fixture-type tags, structural grid-line
            labels, schedule-table column headers) -- zero were a real
            kitchen. Set False only for a corpus confirmed to use bare
            single-letter room labels genuinely (see
            plan/single-letter-label-gate-plan-2026-09-09.md).

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

    # N5a (2026-10-10): resolve tunable thresholds from config or defaults.
    _rf = room_filter if room_filter is not None else RoomFilterConfig()

    # Per-step rejection counter — attached to annotation for run-level aggregation.
    drop_attribution: Dict[str, int] = {}
    # N1 (2026-10-10): per-room detail for EVERY filter step, same minimal
    # descriptor shape, so every loss is re-derivable from run artifacts
    # (previously only vlm_uncorroborated and ocr_name_unresolved had detail).
    drop_detail: Dict[str, List[Dict[str, Any]]] = {}

    # CRITICAL FIX #2: Merge VLM and OCR results instead of OR logic
    vlm_rooms = annotation.get("rooms", [])
    ocr_rooms = annotation.get("ocr_rooms", [])
    if _rf.skip_vlm_echo_candidates:
        echoes: List[Dict] = []
        independent: List[Dict] = []
        for ocr_room in ocr_rooms:
            is_echo = _is_vlm_echo(ocr_room, vlm_rooms, _rf.vlm_echo_bbox_tolerance_px)
            (echoes if is_echo else independent).append(ocr_room)
        if echoes:
            ocr_rooms = independent
            drop_attribution["vlm_echo_skipped"] = len(echoes)
            drop_detail["vlm_echo_skipped"] = [_room_desc(o) for o in echoes]
    merge = _merge_vlm_and_ocr(vlm_rooms, ocr_rooms, _rf.merge_proximity_px)
    rooms = merge.rooms
    for reason, discarded in merge.discards.items():
        drop_attribution[reason] = len(discarded)
        drop_detail[reason] = discarded

    # FIX-7: drop *large* VLM detections with no OCR corroboration (source ==
    # vlm_only). By merge semantics a real *labeled* room's VLM box matches an
    # OCR label and becomes vlm_ocr_merged; a surviving vlm_only box is a
    # pure-visual detection no independent OCR pass confirms. Qwen3-VL visual
    # grounding on CAD line-art is unreliable for large boxes specifically
    # (verified: 11/11 vlm_only rooms >=1.67% area across 116 files were
    # oversized/mislocated giants — logo/margin hallucinations and
    # on-drawing swallowers; zero legit large vlm_only found).
    #
    # R3 (2026-09-14): that 1.67% threshold was measured but never applied —
    # the original code dropped every vlm_only room regardless of size, not
    # just the validated giants. Cross-checked against
    # plan/zero-room-page-recovery-plan-2026-09-14.md's R2 filter ranking:
    # this was the single largest cause of the 40/53 real-floor-plan pages
    # losing every room (886 of 1437 dropped rooms across those 40 pages,
    # 61.7%, dominant filter on 34/40) — small vlm_only rooms have a VLM-read
    # name (room numbers, OFFICE/CORRIDOR/RESTROOM/...) that simply failed to
    # spatially merge with an independent OCR box, not a hallucination.
    # Blast-radius checked on the 88 pages that already survive: area-gating
    # this filter recovers 71 more real rooms across 26 pages (spot-checked,
    # all plausible named rooms) with zero rooms lost anywhere in that set —
    # narrowing the drop condition can only ever admit rooms this stage used
    # to remove, never remove ones it used to keep.
    _img_w0 = annotation.get("image_size", {}).get("width", 0)
    _img_h0 = annotation.get("image_size", {}).get("height", 0)
    _img_area0 = _img_w0 * _img_h0 if _img_w0 and _img_h0 else 0
    _VLM_UNCORROBORATED_AREA_THRESHOLD = _rf.vlm_uncorroborated_area_frac
    def _vlm_only_area_frac(r):
        bbox = r.get("bbox")
        if bbox and len(bbox) == 4 and _img_area0:
            return (bbox[2] * bbox[3]) / _img_area0
        return None
    def _drop_vlm_only(r):
        if r.get("source") != "vlm_only":
            return False
        area_frac = _vlm_only_area_frac(r)
        return area_frac is None or area_frac >= _VLM_UNCORROBORATED_AREA_THRESHOLD
    pre_vlm = len(rooms)
    _dropped_vlm_only = [r for r in rooms if _drop_vlm_only(r)]
    rooms = [r for r in rooms if not _drop_vlm_only(r)]
    n_vlm = pre_vlm - len(rooms)
    if n_vlm:
        drop_attribution["vlm_uncorroborated"] = n_vlm
        _detail = []
        for r in _dropped_vlm_only:
            bbox = r.get("bbox")
            area_frac = None
            if bbox and len(bbox) == 4 and _img_area0:
                area_frac = round((bbox[2] * bbox[3]) / _img_area0, 4)
            _detail.append({
                "room_name": r.get("room_name") or r.get("name"),
                "room_type": r.get("room_type") or r.get("type"),
                "bbox": bbox,
                "area_frac": area_frac,
            })
        drop_detail["vlm_uncorroborated"] = _detail
        logger.info(f"FIX-7: dropped {n_vlm} uncorroborated vlm_only room(s)")

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
                    if (inter / room_area > ZONE_OVERLAP_BOX_FRAC
                            or (zone_area > 0 and inter / zone_area > ZONE_OVERLAP_ZONE_FRAC)):
                        return True
            return False
        _pre_excl_rooms = list(rooms)
        rooms = [r for r in rooms if not _in_any_zone(r.get("bbox", []), exclusion_zones)]
        n_excl = pre_excl - len(rooms)
        if n_excl:
            drop_attribution["bom_zone"] = n_excl
            _kept_ids = {id(r) for r in rooms}
            drop_detail["bom_zone"] = [_room_desc(r) for r in _pre_excl_rooms if id(r) not in _kept_ids]
            logger.info(f"FIX-4: dropped {n_excl} room(s) with centroid in BOM/title-block zone")

    # Step 0-pre: FIX-9 — same BOM/title-block protection for objectDetections
    # (door/window boxes from the YOLO detector). FIX-4 above only ever filtered
    # rooms, so detections landing in a title block/BOM survived into
    # processed_annotations and got drawn on overlays (verified: 2 of 11 window
    # detections on a real page had their centroid inside a real zone).
    #
    # Separate from _in_any_zone above because the bbox CONVENTIONS DIFFER:
    # rooms are [x, y, w, h]; objectDetections are [x1, y1, x2, y2] (written by
    # pipeline.py::_collect_object_detections). Reusing the xywh closure here
    # would silently misread every coordinate. Same centroid + dual-direction
    # overlap rule, same shared cuts.
    #
    # Gated on its own key, not on `rooms`: a page can have zero surviving rooms
    # and still carry detections that must be filtered.
    object_detections = annotation.get("objectDetections", [])
    if exclusion_zones and object_detections:
        def _in_any_zone_xyxy(bbox, zones):
            if len(bbox) < 4:
                return False
            bx1, by1, bx2, by2 = bbox[0], bbox[1], bbox[2], bbox[3]
            bw, bh = bx2 - bx1, by2 - by1
            if bw <= 0 or bh <= 0:
                return False
            cx, cy = bx1 + bw / 2, by1 + bh / 2
            box_area = bw * bh
            for x1, y1, x2, y2 in zones:
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    return True
                ix1, iy1 = max(bx1, x1), max(by1, y1)
                ix2, iy2 = min(bx2, x2), min(by2, y2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    zone_area = (x2 - x1) * (y2 - y1)
                    if (inter / box_area > ZONE_OVERLAP_BOX_FRAC
                            or (zone_area > 0 and inter / zone_area > ZONE_OVERLAP_ZONE_FRAC)):
                        return True
            return False

        pre_obj = len(object_detections)
        object_detections = [
            d for d in object_detections
            if not _in_any_zone_xyxy(d.get("bbox", []), exclusion_zones)
        ]
        n_obj_excl = pre_obj - len(object_detections)
        if n_obj_excl:
            drop_attribution["bom_zone_objects"] = n_obj_excl
            logger.info(
                f"FIX-9: dropped {n_obj_excl} objectDetection(s) in BOM/title-block zone"
            )
        annotation["objectDetections"] = object_detections

    # Step 0-pre: FIX-9b — BA3 (2026-08-03): drop "window" objectDetections
    # whose bbox is not elongated (window_evidence.is_window_shaped). Root
    # cause verified this session: the YOLO detector mislabels door-swing
    # symbols (near-square boxes) as "window" -- confirmed on 10 pixel-
    # inspected real production boxes (10/10 correctly separated) and on
    # test_pcs/kaggle GT (precision up on both, ≤1 real window TP lost per
    # set). Sibling of FIX-9, not a merge: independent gate
    # (use_window_elongation_filter), independent drop_attribution key, and
    # runs regardless of exclusion_zones -- a page with no BOM zones at all
    # can still carry mislabeled windows.
    #
    # Bbox-only predicate (no image read) -- cheap, no I/O added to this path.
    object_detections = annotation.get("objectDetections", [])
    if use_window_elongation_filter and object_detections:
        window_evidence_config = WindowEvidenceConfig()
        pre_shape = len(object_detections)
        object_detections = [
            d for d in object_detections
            if d.get("category") != "window"
            # `is not False`, not `is True`: is_window_shaped returns None for
            # a degenerate bbox (zero width/height) -- fail open, keep it,
            # rather than drop on a box we can't actually evaluate.
            or is_window_shaped(d.get("bbox", []), window_evidence_config) is not False
        ]
        n_shape_excl = pre_shape - len(object_detections)
        if n_shape_excl:
            drop_attribution["window_shape_mismatch"] = n_shape_excl

            logger.info(
                f"FIX-9b: dropped {n_shape_excl} objectDetection(s) tagged "
                f"window with a non-elongated (door-like) bbox"
            )
        annotation["objectDetections"] = object_detections

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
                _pre_ink = list(rooms)
                rooms, n_ink = _filter_low_ink_rooms(
                    rooms,
                    gray,
                    _rf.min_ink_frac,
                    _rf.ink_binarise_max,
                    exempt_sam_expanded=exempt_sam_expanded_from_ink_filter,
                )
                if n_ink:
                    drop_attribution["low_ink"] = n_ink
                    _kept_ids = {id(r) for r in rooms}
                    drop_detail["low_ink"] = [_room_desc(r) for r in _pre_ink if id(r) not in _kept_ids]
                    logger.info(
                        f"FIX-5: dropped {n_ink} room(s) placed over non-drawing region"
                    )

                # FIX-6: drop rooms whose centroid lands in a ruled table (BOM/
                # schedule). Reuses the image already loaded for FIX-5.
                tables = _detect_ruled_tables(gray, _rf.ink_binarise_max)
                _pre_table = list(rooms)
                rooms, n_table = _filter_rooms_in_tables(rooms, tables)
                if n_table:
                    drop_attribution["ruled_table"] = n_table
                    _kept_ids = {id(r) for r in rooms}
                    drop_detail["ruled_table"] = [_room_desc(r) for r in _pre_table if id(r) not in _kept_ids]
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
    MIN_ROOM_AREA_PX = _rf.min_room_area_px
    MAX_ROOM_FRAC    = _rf.max_room_frac

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
    geom_pre_snap = list(rooms)
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
                # or keep the drifted giant. Ceiling is SAM_OVEREXPAND_MAX_FRAC
                # (data-derived gap 0.91%-1.63%); the old 8% inline ceiling let
                # 9 giants (1.63-7.72%) survive.
                if (room.get("sam_expanded") and room.get("original_bbox")
                        and img_area_pre > 0
                        and area > img_area_pre * SAM_OVEREXPAND_MAX_FRAC):
                    ob = room["original_bbox"]
                    if len(ob) == 4 and ob[2] * ob[3] >= OCR_MIN_AREA_PX:
                        logger.warning(
                            f"T-C1: reverted over-expanded OCR bbox '{rn}' "
                            f"(sam_area={area:.0f} = {100*area/img_area_pre:.1f}% "
                            f"> {SAM_OVEREXPAND_MAX_FRAC:.1%}) to located label {ob}"
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
            MAX_VLM_UNIT_FRAC = _rf.max_vlm_unit_frac
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
        _kept_ids = {id(r) for r in rooms}
        drop_detail["geometric"] = [_room_desc(r) for r in geom_pre_snap if id(r) not in _kept_ids]
        logger.info(f"Fix1: geometric filter removed {n_geom} bbox(es)")

    # Remediation Fix #2: Resolve bbox overlaps in non-circulation room types.
    # This must happen AFTER geometric filter but BEFORE semantic filtering
    # to ensure clean spatial geometry for SFT.
    pre_overlap = len(rooms)
    _pre_overlap_snap = list(rooms)
    rooms = _resolve_bbox_overlaps(rooms)
    # Fix2b (Phase 2): same-type containment dedup — catches nested duplicate
    # boxes (esp. circulation types skipped by _resolve_bbox_overlaps) whose
    # IoU falls just under the 0.5 merge threshold, leaving an oversized copy
    # that survives past the 8% single-unit ceiling.
    rooms = _dedup_nested_same_type(rooms)
    n_resolved = pre_overlap - len(rooms)
    drop_attribution["overlap"] = n_resolved
    if n_resolved > 0:
        _kept_ids = {id(r) for r in rooms}
        drop_detail["overlap"] = [_room_desc(r) for r in _pre_overlap_snap if id(r) not in _kept_ids]
        logger.info(f"Remediation Fix #2: overlap resolution merged/removed {n_resolved} room(s)")

    # Step 0d: Split out OCR candidates where a room-number tag was found but
    # the descriptive room-name text was never resolved — see
    # quality_checker.py's OCR_NAME_EMPTY check. Root cause is case-by-case:
    # confirmed 2026-09-15 to most often be a room_name_patterns vocabulary
    # gap (the name OCR'd fine but matched no known pattern, so it never
    # became a name candidate to link the number to) rather than an
    # image-quality issue — do not assume hatch/grid interference without
    # checking the specific page. These previously fell into the generic
    # "semantic" bucket below, hiding the cause entirely. Attribute them
    # separately here so semantic doesn't double-count them.
    def _name_unresolved(r):
        return bool(r.get("room_number")) and not (r.get("room_name") or r.get("name") or "").strip()

    pre_unresolved = len(rooms)
    unresolved_name_rooms = [r for r in rooms if _name_unresolved(r)]
    if unresolved_name_rooms:
        rooms = [r for r in rooms if not _name_unresolved(r)]
        n_unresolved = pre_unresolved - len(rooms)
        drop_attribution["ocr_name_unresolved"] = n_unresolved
        drop_detail["ocr_name_unresolved"] = [
            {"bbox": r.get("bbox"), "room_number": r.get("room_number")}
            for r in unresolved_name_rooms
        ]
        logger.info(
            f"Step 0d: {n_unresolved} room(s) dropped — room-number found but "
            f"name never resolved to a known room_name pattern"
        )

    # Step 1: Semantic filtering
    pre_semantic = len(rooms)
    _pre_semantic_snap = list(rooms)
    rooms = validator.filter_rooms(
        rooms, block_single_letter_abbreviations=block_single_letter_abbreviations
    )
    n_semantic = pre_semantic - len(rooms)
    drop_attribution["semantic"] = n_semantic
    if n_semantic > 0:
        _kept_ids = {id(r) for r in rooms}
        drop_detail["semantic"] = [_room_desc(r) for r in _pre_semantic_snap if id(r) not in _kept_ids]
        logger.info(f"Step 1: semantic filter removed {n_semantic} room(s) ({len(rooms)} remaining)")
    logger.debug(f"After semantic filter: {len(rooms)} rooms")

    # Step 2: Confidence filtering (preserves VLM rooms with no confidence field)
    pre_confidence = len(rooms)
    _pre_confidence_snap = list(rooms)
    rooms = filter_by_confidence(rooms, min_confidence=0.85)
    n_confidence = pre_confidence - len(rooms)
    drop_attribution["confidence"] = n_confidence
    if n_confidence > 0:
        _kept_ids = {id(r) for r in rooms}
        drop_detail["confidence"] = [_room_desc(r) for r in _pre_confidence_snap if id(r) not in _kept_ids]
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
        _pre_oob_snap = list(rooms)
        rooms = [r for r in rooms if _bbox_in_bounds(r.get("bbox", []), img_w, img_h)]
        n_dropped = pre_oob - len(rooms)
        drop_attribution["oob"] = n_dropped
        if n_dropped > 0:
            _kept_ids = {id(r) for r in rooms}
            drop_detail["oob"] = [_room_desc(r) for r in _pre_oob_snap if id(r) not in _kept_ids]
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
            normalized = normalizer.normalize(
                room_name, block_single_letter_abbreviations=block_single_letter_abbreviations
            )
            room["type"] = normalized
            room["category"] = normalized  # Sync category field for COCO exporter
            logger.debug(f"Normalized '{room_name}' → {normalized}")

    # One room, two boxes: see automation/duplicate_boxes.py. Runs HERE, not
    # earlier in the chain: the rule pairs a room-extent box with a label box
    # of the same room type, and a room only has its type once the loop above
    # has normalised it (measured: placed before this, it silently missed
    # duplicates whose rooms were not yet typed).
    if _rf.drop_redundant_room_extent_boxes:
        rooms, redundant = split_redundant_room_extents(rooms)
        if redundant:
            drop_attribution["redundant_room_extent_box"] = len(redundant)
            drop_detail["redundant_room_extent_box"] = [_room_desc(r) for r in redundant]
            logger.info(
                f"dropped {len(redundant)} room-extent box(es) already holding a label box"
            )

    # Step 4: SFT validation
    pre_sft = len(rooms)
    sft_ready = []
    _sft_validation_dropped: List[Dict[str, Any]] = []
    for room in rooms:
        is_valid, checks = validate_for_sft(room)
        if not is_valid:
            failed_checks = [k for k, v in checks.items() if not v]
            logger.debug(f"Room rejected: {room.get('name')} - {failed_checks}")
            desc = _room_desc(room)
            desc["failed_checks"] = failed_checks
            _sft_validation_dropped.append(desc)
            continue
        sft_ready.append(room)
    n_sft_dropped = pre_sft - len(sft_ready)
    drop_attribution["sft_validation"] = n_sft_dropped
    if _sft_validation_dropped:
        drop_detail["sft_validation"] = _sft_validation_dropped

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
    ROOM_MIN_AREA = _rf.room_min_area
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
    _pre_strip_snap = list(sft_ready)
    sft_ready = [r for r in sft_ready if r.get("is_room_scale")]
    n_stripped = pre_strip - len(sft_ready)
    if n_stripped:
        drop_attribution["not_room_scale"] = n_stripped
        _kept_ids = {id(r) for r in sft_ready}
        drop_detail["not_room_scale"] = [_room_desc(r) for r in _pre_strip_snap if id(r) not in _kept_ids]
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
    annotation.pop("ocrTextDetections", None)  # raw-annotation replay input only
    annotation.pop("panels", None)  # Remove equipment (not spatial rooms)

    # Step 6: Return with corrected sft_ready logic
    # Stamp what each surviving box encloses. The list mixes two conventions
    # -- an OCR-anchored room's box is its label text, a vlm_only room's box
    # is the whole room (the VLM prompt demands wall-to-wall) -- and every
    # consumer so far re-derived that from `source` or, worse, ignored it and
    # trained on both as one target. Written once, here, so exporters and
    # converters read a field instead of each repeating the mapping.
    for room in sft_ready:
        room["box_convention"] = box_convention_of(room.get("source"))
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
    if drop_detail:
        annotation["drop_detail"] = drop_detail

    logger.info(
        f"SFT preparation complete: {len(sft_ready)} rooms, sft_ready={annotation['sft_ready']}"
    )
    return annotation


# Raw annotation keys that carry room-name candidates before SFT filtering.
_RAW_CANDIDATE_KEYS = ("rooms", "ocr_rooms", "roomsRecognized")

NO_ROOM_NAME_CANDIDATES = "NO_ROOM_LABEL_CANDIDATES"


def has_room_name_candidate(
    raw_annotation: Dict, block_single_letter_abbreviations: bool = True
) -> bool:
    """True if any raw candidate passes SemanticRoomValidator's room-name gate.

    Call on the annotation BEFORE prepare_sft_annotation (which pops
    ocr_rooms and shrinks rooms). False means the page offered no vocabulary
    room name -- MEP tags (LP#15, LT-01, C1-37) alone do not count. It does not
    mean "label-free floor plan": labelled pages whose labels the OCR missed or
    the vocabulary lacks also return False (see
    plan/technology-evaluation-2026-10-11-no-label-fallback.md section 2).
    """
    candidates = [
        room for key in _RAW_CANDIDATE_KEYS for room in raw_annotation.get(key, [])
    ]
    return bool(SemanticRoomValidator().filter_rooms(
        candidates, block_single_letter_abbreviations
    ))


def dropped_page_record(
    annotation: Dict, has_name_candidate: bool
) -> Optional[Dict]:
    """Attribution record for a page that ended with zero rooms, or None.

    Extends the existing drop_attribution/drop_detail record with page_reason
    when the page had no room-name candidate. Pages with neither filter drops
    nor that reason produce no record, as before.
    """
    drop_attr = annotation.get("drop_attribution")
    if has_name_candidate and not drop_attr:
        return None
    record: Dict[str, Any] = {
        "image_file": annotation.get("image_file"),
        "drop_attribution": drop_attr or {},
    }
    if annotation.get("drop_detail"):
        record["drop_detail"] = annotation["drop_detail"]
    if not has_name_candidate:
        record["page_reason"] = NO_ROOM_NAME_CANDIDATES
    return record
