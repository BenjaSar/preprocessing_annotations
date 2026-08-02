"""
Layer 3: Synthetic Residential Label Generation

Generates room labels for residential units when OCR/VLM detection is insufficient.
Creates standardized labels based on apartment structure (1BR, 2BR, etc.).

Confidence = 0.65 (lower to indicate synthetic/inferred generation)
Source = "synthetic_residential"
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ApartmentStructure:
    """Represents apartment structure for synthetic label generation."""
    bedrooms: int
    bathrooms: int
    has_kitchen: bool = True
    has_living_room: bool = True
    unit_id: str = "unit_unknown"
    unit_type: str = "unknown"


class SyntheticLabelGenerator:
    """Generate synthetic room labels for residential units."""

    # Synthetic label confidence (lower than detected, indicates inference)
    SYNTHETIC_CONFIDENCE = 0.65

    # Standard apartment configurations
    APARTMENT_CONFIGS = {
        "studio": {"bedrooms": 0, "bathrooms": 1, "has_living_room": False, "has_kitchen": True},
        "1br1ba": {"bedrooms": 1, "bathrooms": 1, "has_living_room": True, "has_kitchen": True},
        "1br2ba": {"bedrooms": 1, "bathrooms": 2, "has_living_room": True, "has_kitchen": True},
        "2br1ba": {"bedrooms": 2, "bathrooms": 1, "has_living_room": True, "has_kitchen": True},
        "2br2ba": {"bedrooms": 2, "bathrooms": 2, "has_living_room": True, "has_kitchen": True},
        "2br2ba_den": {"bedrooms": 2, "bathrooms": 2, "has_living_room": True, "has_kitchen": True},
        "3br1ba": {"bedrooms": 3, "bathrooms": 1, "has_living_room": True, "has_kitchen": True},
        "3br2ba": {"bedrooms": 3, "bathrooms": 2, "has_living_room": True, "has_kitchen": True},
    }

    @staticmethod
    def identify_apartment_type(structure: ApartmentStructure) -> str:
        """
        Identify apartment type from structure.

        Args:
            structure: ApartmentStructure with bedroom/bathroom info

        Returns:
            Apartment type string (e.g., "1br1ba", "2br2ba")
        """
        br = structure.bedrooms
        ba = structure.bathrooms

        if br == 0:
            return "studio"
        elif br == 1:
            if ba == 1:
                return "1br1ba"
            else:
                return "1br2ba"
        elif br == 2:
            if ba == 1:
                return "2br1ba"
            else:
                return "2br2ba"
        elif br == 3:
            if ba == 1:
                return "3br1ba"
            else:
                return "3br2ba"
        else:
            return f"{br}br{ba}ba"

    @staticmethod
    def generate_labels_for_unit(structure: ApartmentStructure) -> List[Dict]:
        """
        Generate room labels for an apartment unit.

        Args:
            structure: ApartmentStructure with unit info

        Returns:
            List of generated room label dictionaries
        """
        labels = []

        # Bedrooms
        if structure.bedrooms == 0:
            # Studio: one main room
            labels.append({
                "label": "STUDIO",
                "abbreviation": "STUDIO",
                "full_name": "STUDIO APARTMENT",
                "room_type": "studio",
                "category": "studio"
            })
        elif structure.bedrooms == 1:
            # Single bedroom
            labels.append({
                "label": "BR",
                "abbreviation": "BR",
                "full_name": "BEDROOM",
                "room_type": "bedroom",
                "category": "bedroom"
            })
        else:
            # Multiple bedrooms with numbers
            for i in range(1, structure.bedrooms + 1):
                label = f"BR {i}"
                labels.append({
                    "label": label,
                    "abbreviation": label,
                    "full_name": f"BEDROOM {i}",
                    "room_type": "bedroom",
                    "category": "bedroom"
                })

            # Master bedroom (first bedroom with alternate name)
            labels[0].update({
                "is_master": True,
                "alternate_label": "MBR"
            })

        # Living room (if exists and not studio)
        if structure.has_living_room and structure.bedrooms > 0:
            labels.append({
                "label": "LR",
                "abbreviation": "LR",
                "full_name": "LIVING ROOM",
                "room_type": "living_room",
                "category": "living_room"
            })

        # Kitchen (if exists)
        if structure.has_kitchen:
            labels.append({
                "label": "KIT",
                "abbreviation": "KIT",
                "full_name": "KITCHEN",
                "room_type": "kitchen",
                "category": "kitchen"
            })

        # Bathrooms
        if structure.bathrooms == 1:
            labels.append({
                "label": "BA",
                "abbreviation": "BA",
                "full_name": "BATHROOM",
                "room_type": "restroom",
                "category": "restroom"
            })
        elif structure.bathrooms == 2:
            # Primary and secondary bathrooms
            labels.append({
                "label": "MBA",
                "abbreviation": "MBA",
                "full_name": "MASTER BATHROOM",
                "room_type": "restroom",
                "category": "restroom"
            })
            labels.append({
                "label": "BA 2",
                "abbreviation": "BA2",
                "full_name": "BATHROOM 2",
                "room_type": "restroom",
                "category": "restroom"
            })
        else:
            # Multiple numbered bathrooms
            for i in range(1, structure.bathrooms + 1):
                label = f"BA {i}"
                labels.append({
                    "label": label,
                    "abbreviation": label.replace(" ", ""),
                    "full_name": f"BATHROOM {i}",
                    "room_type": "restroom",
                    "category": "restroom"
                })

        # Entry/hallway
        labels.append({
            "label": "ENTRY",
            "abbreviation": "ENTRY",
            "full_name": "ENTRY/HALLWAY",
            "room_type": "hallway",
            "category": "hallway"
        })

        logger.debug(f"Generated {len(labels)} labels for {structure.unit_type} unit")
        return labels

    @staticmethod
    def create_room_annotations(
        units: List[ApartmentStructure],
        base_confidence: float = SYNTHETIC_CONFIDENCE
    ) -> List[Dict]:
        """
        Create complete room annotations from units (synthetic generation).

        Args:
            units: List of ApartmentStructure objects
            base_confidence: Base confidence for synthetic labels

        Returns:
            List of room annotation dictionaries
        """
        annotations = []

        for unit in units:
            # Generate labels for unit
            labels = SyntheticLabelGenerator.generate_labels_for_unit(unit)

            # Create annotations
            for label_info in labels:
                annotation = {
                    "unit_id": unit.unit_id,
                    "room_name": label_info["full_name"],
                    "name": label_info["full_name"],
                    "abbreviation": label_info["abbreviation"],
                    "category": label_info["category"],
                    "type": label_info["room_type"],
                    "confidence": base_confidence,  # Synthetic marker
                    "source": "synthetic_residential",
                    "unit_type": unit.unit_type,
                    "inferred": True,  # Flag as inferred
                    "confidence_note": "Lower confidence indicates synthetic/inferred generation"
                }

                annotations.append(annotation)

        logger.info(f"Created {len(annotations)} synthetic annotations for {len(units)} units")
        return annotations

    @staticmethod
    def validate_label_count(unit_type: str, label_count: int) -> Tuple[bool, str]:
        """
        Validate that generated label count matches expected unit structure.

        Args:
            unit_type: Type like "1br1ba", "2br2ba"
            label_count: Number of labels generated

        Returns:
            (is_valid, message)
        """
        # Expected room counts by type
        expected_counts = {
            "studio": 3,      # Studio + KIT + BA + ENTRY
            "1br1ba": 5,      # BR + LR + KIT + BA + ENTRY
            "1br2ba": 6,      # BR + LR + KIT + BA + BA2 + ENTRY
            "2br1ba": 6,      # BR1 + BR2 + LR + KIT + BA + ENTRY
            "2br2ba": 7,      # BR1 + BR2 + LR + KIT + MBA + BA2 + ENTRY
            "3br1ba": 7,      # BR1 + BR2 + BR3 + LR + KIT + BA + ENTRY
            "3br2ba": 8,      # BR1 + BR2 + BR3 + LR + KIT + MBA + BA2 + ENTRY
        }

        expected = expected_counts.get(unit_type.lower())

        if expected is None:
            return True, f"Unknown unit type: {unit_type}"

        if label_count == expected:
            return True, f"✓ Label count {label_count} matches {unit_type}"
        else:
            return False, f"⚠ Label count {label_count} != expected {expected} for {unit_type}"


class SyntheticLabelValidator:
    """Validate synthetic labels against ground truth or heuristics."""

    @staticmethod
    def validate_bedroom_count(
        unit: ApartmentStructure,
        detected_rooms: List[Dict]
    ) -> Tuple[bool, str]:
        """
        Validate that generated bedroom count matches detected rooms.

        Args:
            unit: ApartmentStructure with expected bedroom count
            detected_rooms: List of detected room annotations

        Returns:
            (is_valid, message)
        """
        bedroom_count = sum(
            1 for room in detected_rooms
            if "BEDROOM" in room.get("room_name", "").upper()
        )

        if bedroom_count == unit.bedrooms:
            return True, f"✓ Bedroom count {bedroom_count} matches unit ({unit.unit_type})"
        else:
            return False, f"⚠ Bedroom count {bedroom_count} != unit {unit.bedrooms} ({unit.unit_type})"

    @staticmethod
    def validate_bathroom_count(
        unit: ApartmentStructure,
        detected_rooms: List[Dict]
    ) -> Tuple[bool, str]:
        """Validate bathroom count."""
        bathroom_count = sum(
            1 for room in detected_rooms
            if "BATHROOM" in room.get("room_name", "").upper()
            or "BATH" in room.get("room_name", "").upper()
        )

        if bathroom_count == unit.bathrooms:
            return True, f"✓ Bathroom count {bathroom_count} matches unit ({unit.unit_type})"
        else:
            return False, f"⚠ Bathroom count {bathroom_count} != unit {unit.bathrooms} ({unit.unit_type})"

    @staticmethod
    def validate_unit_labels(
        unit: ApartmentStructure,
        generated_annotations: List[Dict]
    ) -> Dict[str, Tuple[bool, str]]:
        """
        Comprehensive validation of synthetic labels.

        Args:
            unit: ApartmentStructure being validated
            generated_annotations: Annotations generated for this unit

        Returns:
            Dictionary of validation results
        """
        results = {}

        # Check label count
        label_count_valid, label_count_msg = SyntheticLabelGenerator.validate_label_count(
            unit.unit_type,
            len(generated_annotations)
        )
        results["label_count"] = (label_count_valid, label_count_msg)

        # Check bedroom count
        bedroom_valid, bedroom_msg = SyntheticLabelValidator.validate_bedroom_count(
            unit, generated_annotations
        )
        results["bedroom_count"] = (bedroom_valid, bedroom_msg)

        # Check bathroom count
        bathroom_valid, bathroom_msg = SyntheticLabelValidator.validate_bathroom_count(
            unit, generated_annotations
        )
        results["bathroom_count"] = (bathroom_valid, bathroom_msg)

        # Check for essential rooms
        room_names = [r.get("room_name", "").upper() for r in generated_annotations]

        # Kitchen check
        has_kitchen = any("KITCHEN" in name for name in room_names)
        results["has_kitchen"] = (
            has_kitchen == unit.has_kitchen,
            f"{'✓' if has_kitchen == unit.has_kitchen else '⚠'} Kitchen present: {has_kitchen} (expected: {unit.has_kitchen})"
        )

        # Entry check
        has_entry = any("ENTRY" in name for name in room_names)
        results["has_entry"] = (
            has_entry,
            f"{'✓' if has_entry else '⚠'} Entry/Hallway present: {has_entry}"
        )

        return results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.DEBUG)

    # Test different apartment types
    test_units = [
        ApartmentStructure(bedrooms=0, bathrooms=1, unit_id="unit_001", unit_type="studio"),
        ApartmentStructure(bedrooms=1, bathrooms=1, unit_id="unit_002", unit_type="1br1ba"),
        ApartmentStructure(bedrooms=2, bathrooms=2, unit_id="unit_003", unit_type="2br2ba"),
        ApartmentStructure(bedrooms=3, bathrooms=2, unit_id="unit_004", unit_type="3br2ba"),
    ]

    print("\nTesting synthetic label generation:")
    for unit in test_units:
        labels = SyntheticLabelGenerator.generate_labels_for_unit(unit)
        print(f"\n{unit.unit_type.upper()}:")
        for label in labels:
            print(f"  - {label['label']:10} → {label['full_name']}")

    print("\n\nTesting label count validation:")
    for unit in test_units:
        labels = SyntheticLabelGenerator.generate_labels_for_unit(unit)
        is_valid, msg = SyntheticLabelGenerator.validate_label_count(unit.unit_type, len(labels))
        print(f"  {msg}")

    print("\n\nCreating annotations for all units:")
    annotations = SyntheticLabelGenerator.create_room_annotations(test_units)
    print(f"Total annotations: {len(annotations)}")
    for unit in test_units:
        unit_annotations = [a for a in annotations if a["unit_id"] == unit.unit_id]
        print(f"  {unit.unit_type}: {len(unit_annotations)} rooms")
