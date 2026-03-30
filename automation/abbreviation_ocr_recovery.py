"""
Layer 2: OCR Abbreviation Recovery

Recovers bedroom and room abbreviations from small text in apartment units.
Uses image preprocessing (crop, enhance, upscale) to improve OCR accuracy.

Targets abbreviations: BR, BR1, BR2, LR, KIT, BA, WIC, FR, FAM, OF, CL, etc.

The abbreviation dictionary is sourced from automation/abbreviations.py —
the single source of truth shared with sft_validator.py.  Previously this
class maintained its own divergent copy that was missing FR, FAM, OF, OFC,
CL, CLS, and PDR, causing those room types to be silently dropped.
"""

import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np

logger = logging.getLogger(__name__)

try:
    from .abbreviations import ABBREVIATION_MAP
except ImportError:
    from abbreviations import ABBREVIATION_MAP


class ResidentialAbbreviationRecovery:
    """Recover residential abbreviations from floor plan images."""

    # Single source of truth — imported from automation/abbreviations.py.
    # All entries in ABBREVIATION_MAP are valid abbreviations; the name
    # RESIDENTIAL_ABBREVIATIONS is kept for backward compatibility with
    # callers that reference it directly (e.g., pipeline.py Step 2).
    RESIDENTIAL_ABBREVIATIONS = ABBREVIATION_MAP

    @staticmethod
    def is_residential_abbreviation(text: str) -> bool:
        """Check if text is a recognized residential abbreviation."""
        text_upper = text.upper().strip()
        return text_upper in ResidentialAbbreviationRecovery.RESIDENTIAL_ABBREVIATIONS

    @staticmethod
    def expand_abbreviation(abbrev: str) -> str:
        """Expand abbreviation to full room name."""
        abbrev_upper = abbrev.upper().strip()
        return ResidentialAbbreviationRecovery.RESIDENTIAL_ABBREVIATIONS.get(
            abbrev_upper, abbrev
        )


class ImagePreprocessor:
    """Preprocess floor plan images for abbreviation OCR detection."""

    @staticmethod
    def crop_unit_region(
        image: Image.Image,
        bbox: Tuple[int, int, int, int],
        padding_percent: float = 0.10
    ) -> Image.Image:
        """
        Crop apartment unit region from floor plan.

        Args:
            image: PIL Image of floor plan
            bbox: [x, y, width, height] coordinates
            padding_percent: Percentage padding around bbox

        Returns:
            Cropped PIL Image
        """
        if len(bbox) < 4:
            logger.warning(f"Invalid bbox: {bbox}")
            return image

        x, y, w, h = bbox[:4]

        # Add padding
        pad_x = int(w * padding_percent)
        pad_y = int(h * padding_percent)

        # Calculate crop box with bounds checking
        left = max(0, x - pad_x)
        top = max(0, y - pad_y)
        right = min(image.width, x + w + pad_x)
        bottom = min(image.height, y + h + pad_y)

        if left >= right or top >= bottom:
            logger.warning(f"Invalid crop region: ({left}, {top}, {right}, {bottom})")
            return image

        cropped = image.crop((left, top, right, bottom))
        logger.debug(f"Cropped unit region: {cropped.size} from bbox {bbox}")

        return cropped

    @staticmethod
    def enhance_contrast(image: Image.Image, factor: float = 1.5) -> Image.Image:
        """
        Enhance image contrast for better OCR.

        Args:
            image: PIL Image
            factor: Contrast enhancement factor (1.5 = 50% increase)

        Returns:
            Enhanced PIL Image
        """
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(factor)
        logger.debug(f"Enhanced contrast by {factor}x")
        return enhanced

    @staticmethod
    def enhance_sharpness(image: Image.Image, factor: float = 1.3) -> Image.Image:
        """
        Enhance image sharpness.

        Args:
            image: PIL Image
            factor: Sharpness enhancement factor

        Returns:
            Sharpened PIL Image
        """
        enhancer = ImageEnhance.Sharpness(image)
        enhanced = enhancer.enhance(factor)
        logger.debug(f"Enhanced sharpness by {factor}x")
        return enhanced

    @staticmethod
    def upscale_image(image: Image.Image, scale_factor: int = 2) -> Image.Image:
        """
        Upscale image for better OCR accuracy on small text.

        Args:
            image: PIL Image
            scale_factor: Scaling factor (2 = 2x size)

        Returns:
            Upscaled PIL Image
        """
        new_size = (
            image.width * scale_factor,
            image.height * scale_factor
        )
        upscaled = image.resize(new_size, Image.Resampling.LANCZOS)
        logger.debug(f"Upscaled image from {image.size} to {upscaled.size}")
        return upscaled

    @staticmethod
    def preprocess_for_ocr(
        image: Image.Image,
        contrast_factor: float = 1.5,
        sharpness_factor: float = 1.3,
        upscale_factor: int = 2
    ) -> Image.Image:
        """
        Complete preprocessing pipeline for OCR.

        Args:
            image: PIL Image
            contrast_factor: Contrast enhancement
            sharpness_factor: Sharpness enhancement
            upscale_factor: Upscaling factor

        Returns:
            Preprocessed PIL Image
        """
        # Step 1: Enhance contrast
        image = ImagePreprocessor.enhance_contrast(image, contrast_factor)

        # Step 2: Enhance sharpness
        image = ImagePreprocessor.enhance_sharpness(image, sharpness_factor)

        # Step 3: Upscale
        image = ImagePreprocessor.upscale_image(image, upscale_factor)

        logger.debug("Completed OCR preprocessing pipeline")
        return image


class AbbreviationOCRRecovery:
    """Recover abbreviations from preprocessed apartment unit images."""

    def __init__(self, ocr_tool=None):
        """
        Initialize abbreviation recovery.

        Args:
            ocr_tool: OCR tool (easyocr, paddleocr, etc.)
        """
        self.ocr_tool = ocr_tool
        self.preprocessor = ImagePreprocessor()
        self.abbreviations_found = []

    def recover_abbreviations_from_image(
        self,
        image_path: str,
        unit_bbox: Tuple[int, int, int, int],
        unit_id: str = "unknown"
    ) -> List[Dict]:
        """
        Recover abbreviations from a unit region in floor plan.

        Args:
            image_path: Path to floor plan image
            unit_bbox: [x, y, width, height] of unit
            unit_id: Unit identifier for tracking

        Returns:
            List of detected abbreviations with metadata
        """
        recovered = []

        try:
            # Load image
            image = Image.open(image_path)
            logger.debug(f"Loaded image: {image_path} ({image.size})")

            # Crop unit region
            unit_image = self.preprocessor.crop_unit_region(image, unit_bbox)

            # Preprocess for OCR
            processed_image = self.preprocessor.preprocess_for_ocr(unit_image)

            # Run OCR if available
            if self.ocr_tool is None:
                logger.warning("No OCR tool available; skipping abbreviation recovery")
                return recovered

            ocr_results = self.ocr_tool.ocr(processed_image)

            # Extract abbreviations from OCR results
            for detection in ocr_results:
                if not detection or len(detection) < 2:
                    continue

                bbox_info = detection[0]  # Bounding box
                text_confidence = detection[1]  # (text, confidence)

                if not isinstance(text_confidence, (tuple, list)):
                    continue

                text = str(text_confidence[0]).strip().upper()
                confidence = float(text_confidence[1]) if len(text_confidence) > 1 else 0.5

                # Check if abbreviation
                if ResidentialAbbreviationRecovery.is_residential_abbreviation(text):
                    expanded = ResidentialAbbreviationRecovery.expand_abbreviation(text)

                    abbrev_dict = {
                        "abbreviation": text,
                        "expanded": expanded,
                        "confidence": confidence,
                        "unit_id": unit_id,
                        "bbox": bbox_info if isinstance(bbox_info, list) else None,
                        "source": "ocr_preprocessed"
                    }

                    recovered.append(abbrev_dict)
                    logger.debug(f"Recovered abbrev: {text} → {expanded} (confidence: {confidence:.2f})")

        except Exception as e:
            logger.error(f"Error recovering abbreviations from {image_path}: {e}")

        self.abbreviations_found.extend(recovered)
        logger.info(f"Recovered {len(recovered)} abbreviations from unit {unit_id}")

        return recovered

    def batch_recover_abbreviations(
        self,
        image_path: str,
        units: List[Dict]
    ) -> List[Dict]:
        """
        Recover abbreviations from multiple units in one image.

        Args:
            image_path: Path to floor plan image
            units: List of unit dictionaries with bbox

        Returns:
            List of all recovered abbreviations
        """
        all_abbreviations = []

        for unit in units:
            unit_id = unit.get("unit_id", "unknown")
            bbox = unit.get("bbox", [0, 0, 100, 100])

            abbrevs = self.recover_abbreviations_from_image(
                image_path=image_path,
                unit_bbox=tuple(bbox),
                unit_id=unit_id
            )

            all_abbreviations.extend(abbrevs)

        logger.info(f"Batch recovery: {len(all_abbreviations)} abbreviations from {len(units)} units")
        return all_abbreviations

    def create_annotations(
        self,
        abbreviations: List[Dict],
        base_confidence_boost: float = 0.15
    ) -> List[Dict]:
        """
        Convert recovered abbreviations to room annotations.

        Args:
            abbreviations: List of recovered abbreviations
            base_confidence_boost: Confidence boost for OCR-recovered items

        Returns:
            List of room annotations ready for output
        """
        annotations = []

        for abbrev in abbreviations:
            annotation = {
                "room_name": abbrev["expanded"],
                "name": abbrev["expanded"],
                "abbreviation": abbrev["abbreviation"],
                "category": self._map_category(abbrev["expanded"]),
                "type": self._map_category(abbrev["expanded"]),
                "unit_id": abbrev.get("unit_id", "unknown"),
                "confidence": min(1.0, abbrev.get("confidence", 0.5) + base_confidence_boost),
                "source": "ocr_abbreviation_recovered",
                "original_abbrev": abbrev["abbreviation"]
            }

            annotations.append(annotation)

        logger.info(f"Created {len(annotations)} annotations from abbreviations")
        return annotations

    @staticmethod
    def _map_category(room_name: str) -> str:
        """Map expanded room name to standard category."""
        name_upper = room_name.upper()

        if "BEDROOM" in name_upper:
            return "bedroom"
        elif "LIVING" in name_upper:
            return "living_room"
        elif "DINING" in name_upper:
            return "dining_room"
        elif "KITCHEN" in name_upper:
            return "kitchen"
        elif "BATHROOM" in name_upper or "BATH" in name_upper:
            return "restroom"
        elif "CLOSET" in name_upper:
            return "closet"
        elif "PANTRY" in name_upper:
            return "pantry"
        elif "GARAGE" in name_upper:
            return "garage"
        elif "STUDIO" in name_upper:
            return "studio"
        else:
            return "other"


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.DEBUG)

    # Test abbreviation validation
    test_abbrevs = ["BR", "BR1", "1BR", "LR", "KIT", "BA", "INVALID"]

    print("\nTesting abbreviation validation:")
    for abbrev in test_abbrevs:
        is_valid = ResidentialAbbreviationRecovery.is_residential_abbreviation(abbrev)
        expanded = ResidentialAbbreviationRecovery.expand_abbreviation(abbrev)
        print(f"  {abbrev:10} → valid: {is_valid}, expanded: {expanded}")

    # Test image preprocessing
    print("\nImage preprocessing pipeline:")
    print("  1. Crop unit region (with 10% padding)")
    print("  2. Enhance contrast (1.5x)")
    print("  3. Enhance sharpness (1.3x)")
    print("  4. Upscale image (2x resolution)")
    print("  → Result: Improved OCR accuracy for small text")
