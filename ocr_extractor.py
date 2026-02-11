"""
OCR text extraction module for MEP floor plans.

This module extracts room labels, panel identifiers, and other text
from floor plan images using EasyOCR with optional preprocessing.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from .config import OCRConfig
except ImportError:
    from config import OCRConfig

logger = logging.getLogger(__name__)


class OCRError(Exception):
    """Raised when OCR processing fails."""

    pass


@dataclass
class RoomCandidate:
    """
    Represents a detected room label from OCR.

    Attributes:
        bbox: Bounding box as (x, y, width, height).
        room_number: Detected room number (e.g., "113", "S1.100").
        room_name: Detected room name (e.g., "MECHANICAL ROOM").
        confidence: OCR confidence score (0.0 to 1.0).
        raw_text: Original text before classification.
    """

    bbox: Tuple[int, int, int, int]
    room_number: str
    room_name: str
    confidence: float
    raw_text: str = ""


@dataclass
class TextDetection:
    """
    Raw text detection result.

    Attributes:
        bbox: Bounding box as list of 4 corner points.
        text: Detected text string.
        confidence: Detection confidence (0.0 to 1.0).
    """

    bbox: List[List[int]]
    text: str
    confidence: float


class MEPTextExtractor:
    """
    Extracts and classifies text from MEP floor plan images.

    Uses EasyOCR for text detection and applies domain-specific
    patterns to identify room labels and other relevant text.

    Attributes:
        config: OCRConfig with extraction parameters.

    Example:
        extractor = MEPTextExtractor()
        results = extractor.extract_text("floorplan.png")
        rooms = extractor.find_room_candidates(results)
    """

    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        self._reader = None

        # Compile patterns for efficiency
        self._room_number_pattern = re.compile(
            self.config.room_number_pattern, re.IGNORECASE
        )
        self._room_name_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.config.room_name_patterns
        ]

    @property
    def reader(self):
        """Lazy initialization of EasyOCR reader."""
        if self._reader is None:
            try:
                import easyocr

                use_gpu = self.config.device in ("cuda", "mps")
                self._reader = easyocr.Reader(
                    self.config.languages, gpu=use_gpu, verbose=False
                )
                logger.info(
                    f"Initialized EasyOCR with languages={self.config.languages}, "
                    f"device={self.config.device}"
                )
            except Exception as e:
                raise OCRError(f"Failed to initialize EasyOCR: {e}") from e
        return self._reader

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to improve OCR accuracy on floor plans.

        Preprocessing steps:
        1. Convert to grayscale
        2. Apply CLAHE for contrast enhancement
        3. Denoise with bilateral filter

        Args:
            image: Input image as BGR numpy array.

        Returns:
            Preprocessed grayscale image.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size,
        )
        enhanced = clahe.apply(gray)

        # Denoise while preserving edges
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

        return denoised

    def extract_text(self, image_path: str | Path) -> List[TextDetection]:
        """
        Extract all text with bounding boxes from an image.

        Args:
            image_path: Path to the image file.

        Returns:
            List of TextDetection objects.

        Raises:
            OCRError: If image cannot be loaded or OCR fails.
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise OCRError(f"Image file not found: {image_path}")

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise OCRError(f"Failed to load image: {image_path}")

        # Optional preprocessing
        if self.config.preprocess:
            processed = self.preprocess_image(image)
            # Convert back to BGR for EasyOCR
            image = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        # Run OCR
        try:
            raw_results = self.reader.readtext(image)
        except Exception as e:
            raise OCRError(f"OCR failed on {image_path}: {e}") from e

        # Convert to TextDetection objects
        detections = []
        for bbox, text, conf in raw_results:
            if conf >= self.config.confidence_threshold:
                detections.append(
                    TextDetection(
                        bbox=[[int(p[0]), int(p[1])] for p in bbox],
                        text=text,
                        confidence=conf,
                    )
                )

        logger.debug(f"Extracted {len(detections)} text regions from {image_path.name}")
        return detections

    def find_room_candidates(
        self, detections: List[TextDetection]
    ) -> List[RoomCandidate]:
        """
        Identify text detections that represent room labels.

        Args:
            detections: List of TextDetection objects from extract_text().

        Returns:
            List of RoomCandidate objects for detected room labels.
        """
        candidates = []

        for detection in detections:
            text = detection.text.strip()
            room_number = None
            room_name = None

            # Check for room number pattern
            if self._room_number_pattern.match(text):
                room_number = text

            # Check for room name patterns
            for pattern in self._room_name_patterns:
                if pattern.search(text):
                    room_name = text
                    break

            # Only create candidate if we found something
            if room_number or room_name:
                # Convert polygon bbox to (x, y, w, h)
                x_coords = [p[0] for p in detection.bbox]
                y_coords = [p[1] for p in detection.bbox]
                x = min(x_coords)
                y = min(y_coords)
                w = max(x_coords) - x
                h = max(y_coords) - y

                candidates.append(
                    RoomCandidate(
                        bbox=(x, y, w, h),
                        room_number=room_number or "",
                        room_name=room_name or "",
                        confidence=detection.confidence,
                        raw_text=text,
                    )
                )

        logger.debug(f"Found {len(candidates)} room candidates")
        return candidates

    def extract_and_find_rooms(self, image_path: str | Path) -> List[RoomCandidate]:
        """
        Convenience method to extract text and find rooms in one call.

        Args:
            image_path: Path to the image file.

        Returns:
            List of RoomCandidate objects.
        """
        detections = self.extract_text(image_path)
        return self.find_room_candidates(detections)

    def get_label_centers(self, candidates: List[RoomCandidate]) -> List[Tuple[int, int]]:
        """
        Get center points of room label bounding boxes.

        Useful for providing prompts to SAM segmentation.

        Args:
            candidates: List of RoomCandidate objects.

        Returns:
            List of (x, y) center points.
        """
        centers = []
        for candidate in candidates:
            x, y, w, h = candidate.bbox
            center_x = x + w // 2
            center_y = y + h // 2
            centers.append((center_x, center_y))
        return centers
