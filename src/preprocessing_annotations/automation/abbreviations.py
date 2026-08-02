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
