"""
Canonical abbreviation map for MEP and residential floor plan room labels.

Single source of truth — imported by:
  - abbreviation_ocr_recovery.py  (Step 2 OCR recovery gate)
  - sft_validator.py              (SemanticRoomValidator + TaxonomyNormalizer)

Keys:   uppercase stripped tokens exactly as OCR produces them.
Values: canonical expanded forms used for taxonomy normalisation.

Previously these mappings existed in three separate dictionaries that had
diverged — ResidentialAbbreviationRecovery, SemanticRoomValidator, and
TaxonomyNormalizer each maintained their own copy.  Divergence caused
abbreviations like FR/FAM (Family Room), OF/OFC (Office), CL/CLS (Closet),
and PDR (Powder Room) to be silently dropped at the OCR gate because they
were absent from ResidentialAbbreviationRecovery and from config.py
room_name_patterns, even though they existed in the SFT-layer dicts.
"""

ABBREVIATION_MAP: dict[str, str] = {
    # ── Residential bedroom ──────────────────────────────────────────────────
    "BR":    "BEDROOM",
    "BD":    "BEDROOM",
    "BDRM":  "BEDROOM",
    "MBR":   "MASTER BEDROOM",
    "MSTR":  "MASTER BEDROOM",
    "MS":    "MASTER BEDROOM",
    "BR 1":  "BEDROOM 1",
    "BR 2":  "BEDROOM 2",
    "BR 3":  "BEDROOM 3",
    "BR1":   "BEDROOM 1",
    "BR2":   "BEDROOM 2",
    "BR3":   "BEDROOM 3",
    # ── Unit types (apartment floor plans) ───────────────────────────────────
    "0BR":   "STUDIO",
    "0 BR":  "STUDIO",
    # OBR: common PaddleOCR misread of 0BR (digit-zero read as letter-O)
    "OBR":   "STUDIO",
    "1BR":   "1 BEDROOM",
    "1 BR":  "1 BEDROOM",
    "2BR":   "2 BEDROOM",
    "2 BR":  "2 BEDROOM",
    "3BR":   "3 BEDROOM",
    "3 BR":  "3 BEDROOM",
    "4BR":   "4 BEDROOM",
    "4 BR":  "4 BEDROOM",
    # ── Janitor / custodial ──────────────────────────────────────────────────
    "JC":    "JANITOR",
    "JAN":   "JANITOR",
    # ── Bathrooms ────────────────────────────────────────────────────────────
    "BA":    "BATHROOM",
    "BATH":  "BATHROOM",
    "MB":    "MASTER BATHROOM",
    "PB":    "POWDER BATHROOM",
    "PDR":   "POWDER ROOM",
    # ── Living / common areas ────────────────────────────────────────────────
    "LR":    "LIVING ROOM",
    "LV":    "LIVING ROOM",
    "DR":    "DINING ROOM",
    "DIN":   "DINING ROOM",
    # FR and FAM were missing from ResidentialAbbreviationRecovery, causing
    # family rooms in residential plans to be dropped at the OCR gate.
    "FR":    "FAMILY ROOM",
    "FAM":   "FAMILY ROOM",
    # ── Kitchen / pantry ─────────────────────────────────────────────────────
    "KIT":   "KITCHEN",
    "K":     "KITCHEN",
    "PAN":   "PANTRY",
    "P":     "PANTRY",
    # ── Closet / storage ─────────────────────────────────────────────────────
    "WIC":   "WALK-IN CLOSET",
    # CL and CLS were missing from all three dictionaries.
    "CL":    "CLOSET",
    "CLS":   "CLOSET",
    "LIN":   "LINEN CLOSET",
    "STR":   "STORAGE",
    "STOR":  "STORAGE",
    # ── Office / commercial ───────────────────────────────────────────────────
    # OF and OFC were missing from ResidentialAbbreviationRecovery.
    "OF":    "OFFICE",
    "OFC":   "OFFICE",
    "OFF":   "OFFICE",
    "CONF":  "CONFERENCE ROOM",
    "STE":   "SUITE",
    "RECP":  "RECEPTION",
    "WC":    "RESTROOM",
    "TLT":   "RESTROOM",
    "RM":    "ROOM",
    # ── Building services / MEP ───────────────────────────────────────────────
    "MECH":  "MECHANICAL ROOM",
    "UTIL":  "UTILITY",
    "UTL":   "UTILITY",
    "LNDRY": "LAUNDRY",
    "BSMT":  "BASEMENT",
    "GAR":   "GARAGE",
    "G":     "GARAGE",
    "SHOP":  "WORKSHOP",
}


# Characters that may follow an abbreviation for it to still be the whole
# token: a separator, or the start of a room number. A letter means the
# abbreviation is just the first letters of a longer word ("BR" in "BRIDGE"),
# which must not expand.
_ABBREVIATION_BOUNDARY: tuple[str, ...] = (" ", "-")


def expand_abbreviation(name: str, block_single_letter: bool = True) -> str:
    """Expand a leading room-label abbreviation ("BR 1" -> "BEDROOM 1").

    Shared by SemanticRoomValidator and TaxonomyNormalizer, which each used
    to carry their own copy of this logic.

    block_single_letter (E2, 2026-09-09): when True a single-letter
    abbreviation ("K", "P", "G") never expands -- measured on corpus_batch2,
    30/30 bare "K" room candidates were fixture tags or grid labels, not
    kitchens. See SemanticRoomValidator._expand_abbreviation for the full
    rationale.

    The boundary is tested on the raw remainder, before stripping: the old
    code stripped first and then checked for a space, so a space could never
    match and "<ABBREV> <WORD>" never expanded ("JAN CLOSET" stayed as-is
    while "JAN" and "JAN 101" expanded).
    """
    name_upper = name.upper().strip()

    if name_upper in ABBREVIATION_MAP:
        if block_single_letter and len(name_upper) == 1:
            return name
        return ABBREVIATION_MAP[name_upper]

    for abbrev, expansion in ABBREVIATION_MAP.items():
        if block_single_letter and len(abbrev) == 1:
            continue
        if not name_upper.startswith(abbrev):
            continue
        remainder = name_upper[len(abbrev):]
        if remainder and not (remainder[0] in _ABBREVIATION_BOUNDARY or remainder[0].isdigit()):
            continue
        return f"{expansion} {remainder.strip()}".strip()

    return name
