"""
Label normalization — thin wrapper around the canonical taxonomy.

All normalisation logic has been moved to automation/taxonomy.py.
This module keeps the LabelNormalizer class API intact for backward
compatibility with pipeline.py Step 4b.

Fixes vs previous version:
  - TAXONOMY_MAP now reflects the 35-type canonical taxonomy instead of 12 types.
  - fuzzy_threshold raised from 0.75 → 0.85 to reduce false-positive matches.
  - Minimum label length guard: strings shorter than 5 chars skip fuzzy matching
    (short tokens like "lab", "WC", "BR" are handled by exact match only).
  - Single source of truth: canonical normalisation delegates to taxonomy.normalize_room_type().
"""

from difflib import SequenceMatcher
from typing import Dict, List, Optional
import logging

try:
    from .taxonomy import CANONICAL_TYPES, normalize_room_type
except ImportError:
    from taxonomy import CANONICAL_TYPES, normalize_room_type

logger = logging.getLogger(__name__)


class LabelNormalizer:
    """Normalise room type labels to canonical taxonomy."""

    # Expose taxonomy map for compatibility with any code that reads it directly
    TAXONOMY_MAP: Dict[str, List[str]] = CANONICAL_TYPES

    def __init__(self, fuzzy_threshold: float = 0.85):
        """
        Args:
            fuzzy_threshold: Similarity threshold for fuzzy matching (raised
                             from 0.75 to 0.85 to reduce false positives).
        """
        self.fuzzy_threshold = fuzzy_threshold

        # Build reverse map for exact matching
        self._exact_match_map: Dict[str, str] = {}
        for standard, variants in CANONICAL_TYPES.items():
            for variant in variants:
                self._exact_match_map[variant.lower()] = standard

    def normalize(self, label: Optional[str], strict: bool = False) -> str:
        """
        Normalise a room label to canonical form.

        Resolution order:
        1. Empty → "other"
        2. Already canonical → return as-is
        3. Exact match in reverse surface map
        4. Fuzzy match (only for labels ≥ 5 chars, threshold 0.85)
        5. Fallback → "other" (or raise if strict=True)
        """
        if not label:
            return "other"
        label = label.strip()
        if not label:
            return "other"

        # Delegate to canonical normaliser (covers steps 1–4 efficiently)
        result = normalize_room_type(label)
        if result != "other":
            return result

        # Fuzzy fallback (only for longer labels to avoid false matches)
        if len(label) >= 5:
            fuzzy = self._fuzzy_match(label.lower())
            if fuzzy:
                return fuzzy

        if strict:
            raise ValueError(f"Cannot normalize label: {label}")

        logger.debug(f"Using 'other' for unknown label: {label}")
        return "other"

    def _fuzzy_match(self, label: str) -> Optional[str]:
        """
        Fuzzy match against all known surface forms.

        Guard: minimum label length of 5 chars prevents short strings like
        "lab", "WC", "BR" from producing spurious character-similarity matches
        (e.g., "lab" ↔ "lobby" at 0.67, "WC" ↔ "office" at 0.50).
        """
        if len(label) < 5:
            return None

        best_score = 0.0
        best_match = None

        for standard, variants in CANONICAL_TYPES.items():
            for variant in variants:
                score = SequenceMatcher(None, label, variant.lower()).ratio()
                if score > best_score:
                    best_score = score
                    best_match = standard

        return best_match if best_score >= self.fuzzy_threshold else None

    def normalize_batch(self, labels: List[str], strict: bool = False) -> List[str]:
        return [self.normalize(l, strict=strict) for l in labels]

    def get_standard_labels(self) -> List[str]:
        return list(CANONICAL_TYPES.keys())

    def is_standard(self, label: str) -> bool:
        return label in CANONICAL_TYPES

    def add_variant(self, standard: str, variants: List[str]) -> None:
        if standard not in CANONICAL_TYPES:
            CANONICAL_TYPES[standard] = []
        for v in variants:
            CANONICAL_TYPES[standard].append(v)
            self._exact_match_map[v.lower()] = standard
        logger.info(f"Added {len(variants)} variants to '{standard}'")


def normalize_label(label: str, threshold: float = 0.85) -> str:
    """Convenience function."""
    return LabelNormalizer(fuzzy_threshold=threshold).normalize(label)
