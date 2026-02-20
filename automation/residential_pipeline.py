"""
Residential Abbreviation Recovery - Complete Pipeline Integration

Integrates all 3 layers:
- Layer 1: VLM residential unit detection
- Layer 2: OCR abbreviation recovery
- Layer 3: Synthetic label generation

Orchestrates the complete workflow for processing residential apartment floors.
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import asdict

from residential_unit_detector import (
    ResidentialUnitDetector,
    ResidentialUnit,
    detect_residential_units
)
from abbreviation_ocr_recovery import (
    AbbreviationOCRRecovery,
    ImagePreprocessor,
    ResidentialAbbreviationRecovery
)
from synthetic_label_generator import (
    SyntheticLabelGenerator,
    ApartmentStructure,
    SyntheticLabelValidator
)

logger = logging.getLogger(__name__)


class ResidentialAnnotationPipeline:
    """Master pipeline for residential apartment annotation."""

    def __init__(self, vlm_client=None, ocr_tool=None):
        """
        Initialize pipeline.

        Args:
            vlm_client: VLM client for residential detection
            ocr_tool: OCR tool for abbreviation recovery
        """
        self.vlm_client = vlm_client
        self.ocr_tool = ocr_tool
        self.detector = ResidentialUnitDetector()
        self.ocr_recovery = AbbreviationOCRRecovery(ocr_tool)
        self.synthetic_generator = SyntheticLabelGenerator()

    def is_residential_floor(self, annotation_json: Dict) -> bool:
        """
        Check if annotation is for residential apartment floor.

        Args:
            annotation_json: Raw annotation JSON

        Returns:
            True if residential
        """
        rooms = annotation_json.get("rooms", [])
        room_names = " ".join([r.get("room_name", "") for r in rooms]).upper()

        # Indicators
        residential_indicators = ["RESIDENTIAL", "APARTMENT", "BEDROOM", "STUDIO"]
        return any(ind in room_names for ind in residential_indicators)

    def detect_units(
        self,
        image_path: str,
        annotation_json: Dict
    ) -> List[ResidentialUnit]:
        """
        LAYER 1: Detect apartment units using VLM.

        Args:
            image_path: Path to floor plan image
            annotation_json: Raw annotation data

        Returns:
            List of detected ResidentialUnit objects
        """
        logger.info("=== LAYER 1: VLM Residential Unit Detection ===")

        try:
            units = detect_residential_units(
                image_path=image_path,
                annotation_json=annotation_json,
                vlm_client=self.vlm_client
            )

            logger.info(f"Layer 1 Result: Detected {len(units)} apartment units")
            for unit in units:
                logger.debug(f"  - {unit.unit_id}: {unit.bedrooms}BR{unit.bathrooms}BA (conf: {unit.confidence:.2f})")

            return units

        except Exception as e:
            logger.error(f"Layer 1 failed: {e}")
            return []

    def recover_abbreviations(
        self,
        image_path: str,
        units: List[ResidentialUnit]
    ) -> List[Dict]:
        """
        LAYER 2: Recover abbreviations using OCR.

        Args:
            image_path: Path to floor plan image
            units: List of detected units (from Layer 1)

        Returns:
            List of recovered abbreviation annotations
        """
        logger.info("=== LAYER 2: OCR Abbreviation Recovery ===")

        try:
            if not self.ocr_tool:
                logger.warning("No OCR tool available; skipping abbreviation recovery")
                return []

            # Convert units to dict format for batch recovery
            unit_dicts = [
                {
                    "unit_id": u.unit_id,
                    "bbox": u.bbox,
                    "bedrooms": u.bedrooms,
                    "bathrooms": u.bathrooms
                }
                for u in units
            ]

            # Batch recover abbreviations
            abbreviations = self.ocr_recovery.batch_recover_abbreviations(
                image_path=image_path,
                units=unit_dicts
            )

            # Create annotations
            annotations = self.ocr_recovery.create_annotations(abbreviations)

            logger.info(f"Layer 2 Result: Recovered {len(annotations)} abbreviation annotations")
            for ann in annotations[:5]:  # Log first 5
                logger.debug(f"  - {ann['room_name']} from {ann['abbreviation']} (conf: {ann['confidence']:.2f})")

            return annotations

        except Exception as e:
            logger.error(f"Layer 2 failed: {e}")
            return []

    def generate_synthetic_labels(
        self,
        units: List[ResidentialUnit]
    ) -> List[Dict]:
        """
        LAYER 3: Generate synthetic labels.

        Args:
            units: List of detected units (from Layer 1)

        Returns:
            List of synthetic label annotations
        """
        logger.info("=== LAYER 3: Synthetic Label Generation ===")

        try:
            # Convert to ApartmentStructure format
            structures = [
                ApartmentStructure(
                    bedrooms=u.bedrooms,
                    bathrooms=u.bathrooms,
                    has_kitchen=u.has_kitchen,
                    has_living_room=u.has_living_room,
                    unit_id=u.unit_id,
                    unit_type=f"{u.bedrooms}BR{u.bathrooms}BA"
                )
                for u in units
            ]

            # Generate synthetic labels
            annotations = self.synthetic_generator.create_room_annotations(structures)

            logger.info(f"Layer 3 Result: Generated {len(annotations)} synthetic label annotations")
            for ann in annotations[:5]:  # Log first 5
                logger.debug(f"  - {ann['room_name']} for {ann['unit_id']} (conf: {ann['confidence']:.2f})")

            return annotations

        except Exception as e:
            logger.error(f"Layer 3 failed: {e}")
            return []

    @staticmethod
    def deduplicate_rooms(all_rooms: List[Dict]) -> List[Dict]:
        """
        Remove duplicate room detections (same unit + room type).

        Priority: VLM > OCR > Synthetic (higher source priority first)

        Args:
            all_rooms: List of all room annotations from all sources

        Returns:
            Deduplicated list
        """
        # Sort by priority (VLM highest, synthetic lowest)
        source_priority = {
            "vlm_residential_detected": 3,
            "ocr_abbreviation_recovered": 2,
            "synthetic_residential": 1,
            "other": 0
        }

        all_rooms_sorted = sorted(
            all_rooms,
            key=lambda r: source_priority.get(r.get("source", "other"), 0),
            reverse=True
        )

        # Deduplicate by (unit_id, room_type)
        seen = set()
        deduplicated = []

        for room in all_rooms_sorted:
            unit_id = room.get("unit_id", "unknown")
            room_type = room.get("category", room.get("type", "unknown"))
            key = (unit_id, room_type)

            if key not in seen:
                seen.add(key)
                deduplicated.append(room)

        logger.info(f"Deduplication: {len(all_rooms)} → {len(deduplicated)} unique rooms")
        return deduplicated

    def process_residential_floor(
        self,
        image_path: str,
        annotation_json: Dict
    ) -> Dict:
        """
        Main pipeline: Process residential apartment floor through all 3 layers.

        Args:
            image_path: Path to floor plan image
            annotation_json: Raw annotation JSON

        Returns:
            Updated annotation JSON with residential labels
        """
        logger.info("\n" + "=" * 60)
        logger.info("STARTING RESIDENTIAL ABBREVIATION RECOVERY PIPELINE")
        logger.info("=" * 60)

        # Check if residential
        if not self.is_residential_floor(annotation_json):
            logger.info("Not a residential floor; returning original annotation")
            annotation_json["residential_processing"] = {
                "processed": False,
                "reason": "Not identified as residential"
            }
            return annotation_json

        logger.info("✓ Identified as residential floor; proceeding with processing")

        # LAYER 1: VLM Detection
        vlm_units = self.detect_units(image_path, annotation_json)

        if not vlm_units:
            logger.warning("No units detected by VLM; proceeding with fallback")
            vlm_units = []

        # LAYER 2: OCR Recovery
        ocr_annotations = self.recover_abbreviations(image_path, vlm_units)

        # LAYER 3: Synthetic Generation
        synthetic_annotations = self.generate_synthetic_labels(vlm_units)

        # Merge all sources
        existing_rooms = annotation_json.get("rooms", [])
        all_rooms = existing_rooms + ocr_annotations + synthetic_annotations

        logger.info(f"\nMerging all sources:")
        logger.info(f"  Existing VLM rooms: {len(existing_rooms)}")
        logger.info(f"  OCR recovered: {len(ocr_annotations)}")
        logger.info(f"  Synthetic generated: {len(synthetic_annotations)}")
        logger.info(f"  Total before dedup: {len(all_rooms)}")

        # Deduplicate
        final_rooms = self.deduplicate_rooms(all_rooms)

        # Update annotation
        annotation_json["rooms"] = final_rooms

        # Add metadata
        annotation_json["residential_processing"] = {
            "processed": True,
            "units_detected": len(vlm_units),
            "abbrevs_recovered": len(ocr_annotations),
            "labels_synthetic": len(synthetic_annotations),
            "total_rooms_after_dedup": len(final_rooms),
            "data_sources": {
                "vlm": len(existing_rooms),
                "ocr": len(ocr_annotations),
                "synthetic": len(synthetic_annotations)
            }
        }

        # Calculate SFT readiness
        if len(final_rooms) > 0:
            coverage = len(final_rooms) / max(1, len(vlm_units) * 4)  # Expect ~4 rooms per unit
            annotation_json["residential_sft_ready"] = coverage >= 0.75

        logger.info(f"\n{'=' * 60}")
        logger.info(f"PIPELINE COMPLETE")
        logger.info(f"  Final rooms: {len(final_rooms)}")
        logger.info(f"  SFT ready: {annotation_json.get('residential_sft_ready', False)}")
        logger.info(f"{'=' * 60}\n")

        return annotation_json


def process_batch_residential_floors(
    image_paths: List[str],
    annotation_jsons: List[Dict],
    vlm_client=None,
    ocr_tool=None
) -> List[Dict]:
    """
    Process multiple residential floors.

    Args:
        image_paths: List of floor plan image paths
        annotation_jsons: List of annotation JSONs
        vlm_client: VLM client
        ocr_tool: OCR tool

    Returns:
        List of processed annotation JSONs
    """
    pipeline = ResidentialAnnotationPipeline(vlm_client, ocr_tool)
    processed = []

    for image_path, annotation_json in zip(image_paths, annotation_jsons):
        try:
            processed_json = pipeline.process_residential_floor(image_path, annotation_json)
            processed.append(processed_json)
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            processed.append(annotation_json)  # Return original on error

    return processed


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Residential Abbreviation Recovery Pipeline - Demo Mode")
    logger.info("(Running without actual VLM/OCR tools - showing structure only)")

    # Create mock annotation
    mock_annotation = {
        "image_file": "page001.png",
        "rooms": [
            {"room_name": "RESIDENTIAL APARTMENTS", "category": "other"}
        ],
        "electrical_counts": {"fixtures": 150}
    }

    # Create pipeline (no tools)
    pipeline = ResidentialAnnotationPipeline()

    # Check if residential
    is_residential = pipeline.is_residential_floor(mock_annotation)
    print(f"\nIs residential: {is_residential}")
    print(f"Pipeline structure: Layer 1 → Layer 2 → Layer 3 → Integration")
    print(f"  Ready for implementation with VLM and OCR tools")
