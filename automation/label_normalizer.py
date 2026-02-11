"""
Label normalization for room type standardization.

Converts various room label formats to standard taxonomy used in the project.
"""

from difflib import SequenceMatcher
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class LabelNormalizer:
    """Normalize room type labels to standard taxonomy."""

    # Standard room taxonomy with known variants
    TAXONOMY_MAP: Dict[str, List[str]] = {
        "office": [
            "office", "office_suite", "exec_office", "executive_office",
            "individual office", "private office", "personal office",
            "office space", "work space", "desk"
        ],
        "conference_room": [
            "conference", "conference_room", "conference room",
            "meeting room", "boardroom", "board room", "meeting",
            "conf_room", "conf room", "boardroom"
        ],
        "meeting_room": [
            "meeting", "meeting_room", "meeting room",
            "breakout", "breakout_space", "breakout space"
        ],
        "restroom": [
            "restroom", "bathroom", "wc", "toilet", "toilets",
            "mens_room", "men's room", "womens_room", "women's room",
            "rest_room", "wash_room", "lavatory"
        ],
        "kitchen": [
            "kitchen", "break_room", "break room", "pantry",
            "kitchen_break", "break", "kitchenette", "dining"
        ],
        "lobby": [
            "lobby", "reception", "entrance", "entry",
            "entry_area", "entrance_area", "foyer", "vestibule",
            "reception_area"
        ],
        "hallway": [
            "hallway", "corridor", "hallway/corridor", "passage",
            "walkway", "circulation"
        ],
        "storage": [
            "storage", "closet", "store room", "storeroom",
            "supply", "supply_room", "archives", "file_room"
        ],
        "mechanical_room": [
            "mechanical", "mechanical_room", "mechanical room",
            "mech_room", "equipment room", "building_systems"
        ],
        "elevator": [
            "elevator", "lift", "elevator_core", "elevator_lobby",
            "elevator_area"
        ],
        "stairwell": [
            "stair", "stairwell", "stairs", "staircase",
            "emergency_stairs", "exit_stairs"
        ],
        "other": [
            "other", "unknown", "misc", "miscellaneous",
            "undefined", "unspecified"
        ]
    }

    def __init__(self, fuzzy_threshold: float = 0.75):
        """
        Initialize normalizer.

        Args:
            fuzzy_threshold: Confidence threshold for fuzzy matching (0.0-1.0).
                            Higher = stricter matching.
        """
        self.fuzzy_threshold = fuzzy_threshold

        # Build reverse map for exact matching
        self._exact_match_map: Dict[str, str] = {}
        for standard, variants in self.TAXONOMY_MAP.items():
            for variant in variants:
                self._exact_match_map[variant.lower()] = standard

    def normalize(self, label: Optional[str], strict: bool = False) -> str:
        """
        Normalize a room label to standard form.

        Args:
            label: Input label (any format)
            strict: If True, only return standard labels; raise on unknown

        Returns:
            Standardized room type label

        Raises:
            ValueError: If strict=True and label cannot be normalized
        """
        if not label:
            return "other"

        label = label.strip()
        if not label:
            return "other"

        # Try exact match first (fast path)
        label_lower = label.lower()
        if label_lower in self._exact_match_map:
            return self._exact_match_map[label_lower]

        # Try fuzzy match
        best_match = self._fuzzy_match(label_lower)
        if best_match:
            return best_match

        # Fallback
        if strict:
            raise ValueError(f"Cannot normalize label: {label}")
        else:
            logger.warning(f"Using 'other' for unknown label: {label}")
            return "other"

    def _fuzzy_match(self, label: str) -> Optional[str]:
        """
        Fuzzy match a label against known variants.

        Args:
            label: Lowercase input label

        Returns:
            Best matching standard label, or None if below threshold
        """
        best_score = 0.0
        best_match = None

        for standard, variants in self.TAXONOMY_MAP.items():
            for variant in variants:
                # Calculate similarity
                score = SequenceMatcher(None, label, variant.lower()).ratio()

                if score > best_score:
                    best_score = score
                    best_match = standard

        # Return only if above threshold
        if best_score >= self.fuzzy_threshold:
            return best_match
        else:
            return None

    def normalize_batch(self, labels: List[str], strict: bool = False) -> List[str]:
        """
        Normalize multiple labels.

        Args:
            labels: List of input labels
            strict: If True, raise on any unknown label

        Returns:
            List of normalized labels
        """
        return [self.normalize(label, strict=strict) for label in labels]

    def get_standard_labels(self) -> List[str]:
        """Get list of all standard room type labels."""
        return list(self.TAXONOMY_MAP.keys())

    def is_standard(self, label: str) -> bool:
        """Check if label is already in standard form."""
        return label in self.TAXONOMY_MAP

    def add_variant(self, standard: str, variants: List[str]) -> None:
        """
        Add new variants for a standard label.

        Args:
            standard: Standard label name
            variants: List of variant spellings
        """
        if standard not in self.TAXONOMY_MAP:
            self.TAXONOMY_MAP[standard] = []

        for variant in variants:
            self.TAXONOMY_MAP[standard].append(variant)
            self._exact_match_map[variant.lower()] = standard

        logger.info(f"Added {len(variants)} variants to '{standard}'")


# Convenience function
def normalize_label(label: str, threshold: float = 0.75) -> str:
    """Quick normalization without creating normalizer object."""
    normalizer = LabelNormalizer(fuzzy_threshold=threshold)
    return normalizer.normalize(label)
