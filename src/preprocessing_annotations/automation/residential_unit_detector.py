"""
Layer 1: VLM-Based Residential Unit Detection

Detects apartment units in residential floor plans and infers their structure
(bedroom count, bathroom count, etc.) using Vision Language Models.

Generates standardized room labels: BR, BR1, BR2, LR, KIT, BA, etc.
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class ResidentialUnit:
    """Represents a detected apartment unit."""
    unit_id: str
    bedrooms: int
    bathrooms: int
    has_kitchen: bool
    has_living_room: bool
    bbox: Tuple[int, int, int, int]  # [x, y, width, height]
    confidence: float
    source: str = "vlm_residential"


class ResidentialUnitDetector:
    """Detect residential apartment units using VLM."""

    # VLM Prompt for residential floor analysis
    RESIDENTIAL_DETECTION_PROMPT = """
    Analyze this floor plan image for RESIDENTIAL APARTMENT UNITS.

    This appears to be a residential building floor plan. Your task is to:

    1. Identify each separate apartment unit (each should be a distinct bounded area)
    2. For each unit, determine:
       - Number of bedrooms (count bed symbols or room labels)
       - Number of bathrooms (count toilet/sink symbols)
       - Has kitchen (yes/no)
       - Has living/main room area (yes/no)
       - Spatial bounds (approximate coordinates as [x, y, width, height])

    IMPORTANT:
    - Studio apartments are 0BR (one main room, no separate bedrooms)
    - 1BR apartments have 1 bedroom + living room
    - 2BR apartments have 2 bedrooms + living room
    - Always count fixtures (beds, toilets) to verify room types

    Return ONLY a valid JSON object (no markdown, no extra text):
    {
      "floor_type": "residential",
      "total_units": N,
      "units": [
        {
          "unit_id": "unit_001",
          "unit_type": "1BR1BA",
          "bedrooms": 1,
          "bathrooms": 1,
          "has_kitchen": true,
          "has_living_room": true,
          "bbox": [x, y, width, height],
          "confidence": 0.85,
          "notes": "Standard 1-bedroom apartment"
        }
      ]
    }
    """

    @staticmethod
    def extract_unit_structure(vlm_response: Dict) -> List[ResidentialUnit]:
        """
        Parse VLM response and extract unit structures.

        Args:
            vlm_response: Dict from VLM with units array

        Returns:
            List of ResidentialUnit objects
        """
        units = []

        try:
            floor_type = vlm_response.get("floor_type", "unknown").lower()
            if "residential" not in floor_type:
                logger.warning(f"Floor type '{floor_type}' not recognized as residential")
                return units

            unit_list = vlm_response.get("units", [])

            for i, unit_data in enumerate(unit_list):
                try:
                    unit_id = unit_data.get("unit_id", f"unit_{i+1:03d}")
                    bedrooms = unit_data.get("bedrooms", 0)
                    bathrooms = unit_data.get("bathrooms", 1)
                    has_kitchen = unit_data.get("has_kitchen", True)
                    has_living = unit_data.get("has_living_room", True)
                    bbox = unit_data.get("bbox", [0, 0, 100, 100])
                    confidence = unit_data.get("confidence", 0.75)

                    # Validate data
                    if not isinstance(bedrooms, int) or bedrooms < 0:
                        bedrooms = 0
                    if not isinstance(bathrooms, int) or bathrooms < 1:
                        bathrooms = 1
                    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                        logger.warning(f"Invalid bbox for {unit_id}: {bbox}")
                        continue
                    if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                        confidence = 0.75

                    unit = ResidentialUnit(
                        unit_id=unit_id,
                        bedrooms=bedrooms,
                        bathrooms=bathrooms,
                        has_kitchen=has_kitchen,
                        has_living_room=has_living,
                        bbox=tuple(bbox[:4]),
                        confidence=confidence
                    )

                    units.append(unit)
                    logger.debug(f"Extracted unit: {unit_id} ({bedrooms}BR{bathrooms}BA, confidence={confidence:.2f})")

                except Exception as e:
                    logger.warning(f"Error parsing unit {i}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error extracting unit structure: {e}")

        logger.info(f"Extracted {len(units)} residential units")
        return units

    @staticmethod
    def generate_room_labels(unit: ResidentialUnit) -> List[Dict]:
        """
        Generate room labels for a residential unit.

        Args:
            unit: ResidentialUnit object

        Returns:
            List of room dictionaries with labels
        """
        rooms = []

        # Bedroom labels
        if unit.bedrooms == 0:
            # Studio: one main room
            rooms.append({
                "label": "STUDIO",
                "abbreviation": "ST",
                "full_name": "STUDIO APARTMENT",
                "room_type": "studio",
                "source": "vlm_residential"
            })
        elif unit.bedrooms == 1:
            # 1-bedroom
            rooms.append({
                "label": "BR",
                "abbreviation": "BR",
                "full_name": "BEDROOM",
                "room_type": "bedroom",
                "source": "vlm_residential"
            })
        else:
            # Multi-bedroom (2BR, 3BR, etc.)
            for i in range(1, unit.bedrooms + 1):
                label = f"BR {i}"
                rooms.append({
                    "label": label,
                    "abbreviation": label,
                    "full_name": f"BEDROOM {i}",
                    "room_type": "bedroom",
                    "source": "vlm_residential"
                })

        # Living room (if exists and not studio)
        if unit.has_living_room and unit.bedrooms > 0:
            rooms.append({
                "label": "LR",
                "abbreviation": "LR",
                "full_name": "LIVING ROOM",
                "room_type": "living_room",
                "source": "vlm_residential"
            })

        # Kitchen (if exists)
        if unit.has_kitchen:
            rooms.append({
                "label": "KIT",
                "abbreviation": "KIT",
                "full_name": "KITCHEN",
                "room_type": "kitchen",
                "source": "vlm_residential"
            })

        # Bathrooms
        if unit.bathrooms == 1:
            rooms.append({
                "label": "BA",
                "abbreviation": "BA",
                "full_name": "BATHROOM",
                "room_type": "restroom",
                "source": "vlm_residential"
            })
        else:
            for i in range(1, unit.bathrooms + 1):
                label = f"BA {i}"
                rooms.append({
                    "label": label,
                    "abbreviation": label,
                    "full_name": f"BATHROOM {i}",
                    "room_type": "restroom",
                    "source": "vlm_residential"
                })

        # Entry/Hallway
        rooms.append({
            "label": "ENTRY",
            "abbreviation": "ENTRY",
            "full_name": "ENTRY/HALLWAY",
            "room_type": "hallway",
            "source": "vlm_residential"
        })

        return rooms

    @staticmethod
    def create_room_annotations(
        units: List[ResidentialUnit],
        base_confidence: float = 0.75
    ) -> List[Dict]:
        """
        Create complete room annotations from units.

        Args:
            units: List of ResidentialUnit objects
            base_confidence: Base confidence score for VLM rooms

        Returns:
            List of annotated rooms ready for output
        """
        annotations = []

        for unit in units:
            room_labels = ResidentialUnitDetector.generate_room_labels(unit)

            for i, room_label in enumerate(room_labels):
                annotation = {
                    "unit_id": unit.unit_id,
                    "room_name": room_label["full_name"],
                    "name": room_label["full_name"],
                    "abbreviation": room_label["abbreviation"],
                    "category": room_label["room_type"],
                    "type": room_label["room_type"],
                    "bbox": list(unit.bbox),
                    "confidence": base_confidence * (unit.confidence / 0.75),  # Scale by unit confidence
                    "source": "vlm_residential_detected",
                    "unit_type": f"{unit.bedrooms}BR{unit.bathrooms}BA"
                }
                annotations.append(annotation)

        logger.info(f"Created {len(annotations)} room annotations for {len(units)} units")
        return annotations

    @staticmethod
    def validate_unit_count(annotation_json: Dict) -> Optional[int]:
        """
        Estimate apartment unit count from floor layout metadata.

        Args:
            annotation_json: Raw annotation JSON

        Returns:
            Estimated unit count or None
        """
        rooms = annotation_json.get("rooms", [])
        electrical_counts = annotation_json.get("electrical_counts", {})

        # Heuristic: residential floors have fixtures distributed across units
        fixtures = electrical_counts.get("fixtures", 0)
        receptacles = electrical_counts.get("receptacles", 0)

        # Typical unit has 8-12 fixtures, 5-7 receptacles
        if fixtures > 0:
            estimated_units_from_fixtures = max(1, fixtures // 10)
            logger.debug(f"Estimated {estimated_units_from_fixtures} units from fixtures")
            return estimated_units_from_fixtures

        return None


def detect_residential_units(
    image_path: str,
    annotation_json: Dict,
    vlm_client=None,  # Would be passed from pipeline
) -> List[ResidentialUnit]:
    """
    Main function: Detect residential units in floor plan image.

    Args:
        image_path: Path to floor plan image
        annotation_json: Raw annotation data
        vlm_client: VLM client for analysis

    Returns:
        List of detected ResidentialUnit objects
    """

    # If VLM not available, use fallback heuristics
    if vlm_client is None:
        logger.warning("No VLM client provided; using heuristic estimation")
        estimated_count = ResidentialUnitDetector.validate_unit_count(annotation_json)
        logger.info(f"Estimated residential units: {estimated_count}")
        return []

    try:
        # Call VLM with residential detection prompt
        vlm_response = vlm_client.analyze_image(
            image_path=image_path,
            prompt=ResidentialUnitDetector.RESIDENTIAL_DETECTION_PROMPT,
            response_format="json"
        )

        # Parse response
        units = ResidentialUnitDetector.extract_unit_structure(vlm_response)
        return units

    except Exception as e:
        logger.error(f"Error detecting residential units: {e}")
        return []


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.DEBUG)

    # Mock unit for testing
    test_unit = ResidentialUnit(
        unit_id="apt_001",
        bedrooms=1,
        bathrooms=1,
        has_kitchen=True,
        has_living_room=True,
        bbox=(100, 200, 300, 400),
        confidence=0.85
    )

    # Generate labels
    labels = ResidentialUnitDetector.generate_room_labels(test_unit)
    print(f"\nGenerated labels for 1BR1BA unit:")
    for label in labels:
        print(f"  - {label['label']} ({label['full_name']})")

    # Create annotations
    annotations = ResidentialUnitDetector.create_room_annotations([test_unit])
    print(f"\nGenerated annotations:")
    for ann in annotations:
        print(f"  - {ann['room_name']} (confidence: {ann['confidence']:.2f})")
