"""
Label normalization — thin wrapper around the mandatory SFT taxonomy.

All normalisation logic has been moved to automation/taxonomy.py.
This module keeps the LabelNormalizer class API intact for backward
compatibility with pipeline.py Step 4b.

IMPORTANT: Now uses mandatory SFT taxonomy (MANDATORY_CLASSES) instead of
the old 31-type canonical taxonomy. This ensures SFT compliance.

The module provides two normalization paths:
  1. normalize() → normalize_to_mandatory() (SFT-compliant, primary)
  2. normalize_with_extended() → get_extended_type() (backward compatibility)
"""

from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple
import logging

try:
    from .taxonomy import MANDATORY_CLASSES, normalize_to_mandatory, get_extended_type
except ImportError:
    from taxonomy import MANDATORY_CLASSES, normalize_to_mandatory, get_extended_type

logger = logging.getLogger(__name__)


class LabelNormalizer:
    """Normalise room type labels to mandatory SFT taxonomy."""

    # Expose taxonomy map for compatibility with any code that reads it directly
    TAXONOMY_MAP: Dict[str, List[str]] = MANDATORY_CLASSES

    def __init__(self, fuzzy_threshold: float = 0.85):
        """
        Args:
            fuzzy_threshold: Similarity threshold for fuzzy matching (raised
                             from 0.75 to 0.85 to reduce false positives).
        """
        self.fuzzy_threshold = fuzzy_threshold

        # Build reverse map for exact matching (from mandatory taxonomy)
        self._exact_match_map: Dict[str, str] = {}
        for standard, variants in MANDATORY_CLASSES.items():
            for variant in variants:
                self._exact_match_map[variant.lower()] = standard

    def normalize(self, label: Optional[str], strict: bool = False) -> str:
        """
        Normalise a room label to mandatory SFT class.

        Resolution order:
        1. Empty → "STORAGE ROOM" (least-intrusive default)
        2. Already a mandatory class → return as-is
        3. Exact match in surface forms
        4. Fuzzy match (only for labels ≥ 5 chars, threshold 0.85)
        5. Fallback → "STORAGE ROOM" (or raise if strict=True)
        
        Returns:
            Mandatory class name from MANDATORY_CLASSES (always SFT-compliant).
        """
        if not label:
            return "STORAGE ROOM"
        label = label.strip()
        if not label:
            return "STORAGE ROOM"

        # Delegate to mandatory SFT normaliser (covers steps 1–4 efficiently)
        result = normalize_to_mandatory(label)
        if result != "STORAGE ROOM":
            return result

        # Fuzzy fallback (only for longer labels to avoid false matches)
        if len(label) >= 5:
            fuzzy = self._fuzzy_match(label.lower())
            if fuzzy:
                return fuzzy

        if strict:
            raise ValueError(f"Cannot normalize label: {label}")

        logger.debug(f"Using STORAGE ROOM for unknown label: {label}")
        return "STORAGE ROOM"

    def _fuzzy_match(self, label: str) -> Optional[str]:
        """
        Fuzzy match against all known surface forms in mandatory taxonomy.

        Guard: minimum label length of 5 chars prevents short strings like
        "lab", "WC", "BR" from producing spurious character-similarity matches
        (e.g., "lab" ↔ "LECTURE HALL" at 0.67, "WC" ↔ "RESTROOM" at 0.50).
        """
        if len(label) < 5:
            return None

        best_score = 0.0
        best_match = None

        for standard, variants in MANDATORY_CLASSES.items():
            for variant in variants:
                score = SequenceMatcher(None, label, variant.lower()).ratio()
                if score > best_score:
                    best_score = score
                    best_match = standard

        return best_match if best_score >= self.fuzzy_threshold else None

    def normalize_batch(self, labels: List[str], strict: bool = False) -> List[str]:
        return [self.normalize(l, strict=strict) for l in labels]

    def get_standard_labels(self) -> List[str]:
        """Return all mandatory class names."""
        return list(MANDATORY_CLASSES.keys())

    def is_standard(self, label: str) -> bool:
        """Check if label is a mandatory class."""
        return label in MANDATORY_CLASSES

    def add_variant(self, standard: str, variants: List[str]) -> None:
        """Add surface form variants to a mandatory class."""
        if standard not in MANDATORY_CLASSES:
            MANDATORY_CLASSES[standard] = []
        for v in variants:
            MANDATORY_CLASSES[standard].append(v)
            self._exact_match_map[v.lower()] = standard
        logger.info(f"Added {len(variants)} variants to '{standard}'")


def normalize_label(label: str, threshold: float = 0.85) -> str:
    """Convenience function."""
    return LabelNormalizer(fuzzy_threshold=threshold).normalize(label)
