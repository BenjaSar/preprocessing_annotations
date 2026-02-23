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


# ---------------------------------------------------------------------------
# Early-exclusion patterns – applied BEFORE spatial merging.
# These reject tokens that can never be a room label regardless of context.
# ---------------------------------------------------------------------------
_PANEL_PATTERN = re.compile(
    r"^PANEL\s*[A-Z0-9]?$", re.IGNORECASE
)
_EQUIPMENT_ONLY_PATTERN = re.compile(
    r"^(PANEL|SWITCHBOARD|TRANSFORMER|DISCONNECT|BREAKER|CIRCUIT|FEEDER|CONDUIT|RACEWAY)(\s+[A-Z0-9]+)?$",
    re.IGNORECASE,
)
_INSTRUCTION_PATTERN = re.compile(
    r"(contractor\s+to\s+verify|sensor\s+placement|take\s+off|use\s+\w+\s+for|"
    r"for\s+\w+\s+only|shall\s+(be|not)|must\s+(be|not)|as\s+directed|"
    r"refer\s+to|see\s+sheet|per\s+(code|nec|nfpa)|"
    r"installation|coordination|approved\s+equal)",
    re.IGNORECASE,
)
_DOCUMENTATION_PATTERN = re.compile(
    r"(general\s+notes|symbol\s+list|legend|title\s+block|drawing\s+index|"
    r"abbreviation|schedule|requirements\s+of|fdny|nec\s+\d|nfpa\s+\d|"
    r"electrical\s+(general|symbol|drawing|device|equipment|circuit|notes)|"
    r"national\s+electrical|distribution\s+equipment|building\s+management\s+system)",
    re.IGNORECASE,
)


def _is_excluded_token(text: str) -> bool:
    """Return True if the raw OCR text should never become a room candidate."""
    return bool(
        _PANEL_PATTERN.match(text)
        or _EQUIPMENT_ONLY_PATTERN.match(text)
        or _INSTRUCTION_PATTERN.search(text)
        or _DOCUMENTATION_PATTERN.search(text)
    )


def _centroid(bbox_points: List[List[int]]) -> Tuple[float, float]:
    """Return (cx, cy) centroid of a 4-point polygon bbox."""
    xs = [p[0] for p in bbox_points]
    ys = [p[1] for p in bbox_points]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _bbox_distance(a_points: List[List[int]], b_points: List[List[int]]) -> float:
    """Euclidean distance between the centroids of two polygon bboxes."""
    ax, ay = _centroid(a_points)
    bx, by = _centroid(b_points)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _same_line(a_points: List[List[int]], b_points: List[List[int]],
               y_tolerance: int = 20) -> bool:
    """True if the two detections are on the same horizontal text line."""
    ay = _centroid(a_points)[1]
    by = _centroid(b_points)[1]
    return abs(ay - by) <= y_tolerance


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
        Apply adaptive preprocessing to improve OCR accuracy on floor plans.

        Changes vs previous version:
        - CLAHE clip limit is now adapted to image contrast (std-dev based).
          High-contrast images (std > 60) get a mild clip (1.5) to avoid
          over-sharpening that causes OCR hallucinations at symbol edges.
          Low-contrast images (faint blueprints) get aggressive clip (3.5)
          to make faint text legible.
        - Previously used a fixed clipLimit=2.0 for all images regardless
          of source contrast, producing over-processing or under-processing.

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

        # Adaptive CLAHE: choose clip limit based on image contrast.
        std = float(gray.std())
        if std > 60:
            clip_limit = 1.5   # high-contrast scan → mild enhancement
        elif std > 30:
            clip_limit = 2.0   # normal contrast → default
        else:
            clip_limit = 3.5   # low-contrast / faint blueprint → aggressive

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
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

        # Convert to TextDetection objects.
        # Length-aware confidence thresholds replace the flat 0.5 cutoff:
        # EasyOCR systematically underestimates confidence for short tokens
        # (fewer characters = less context for the model) so abbreviations
        # like BR/LR at 0.6 confidence are often correct, while a 20-char
        # token at 0.6 is more likely garbled.
        # Conversely, single characters need very high confidence to avoid
        # spurious symbol detections.
        MIN_CONF_BY_LEN = {1: 0.85, 2: 0.70, 3: 0.65, 4: 0.60}
        DEFAULT_MIN_CONF = self.config.confidence_threshold  # 0.50

        detections = []
        for bbox, text, conf in raw_results:
            min_conf = MIN_CONF_BY_LEN.get(len(text.strip()), DEFAULT_MIN_CONF)
            if conf >= min_conf:
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

        Pipeline:
        1. Exclude panel / equipment / documentation tokens early.
        2. Merge adjacent same-line tokens into compound labels
           (e.g. "BUILDING" + "STORAGE" → "BUILDING STORAGE").
        3. Classify merged tokens as room-number or room-name.
        4. Spatially link orphan room-numbers to their nearest room-name.

        Args:
            detections: List of TextDetection objects from extract_text().

        Returns:
            List of RoomCandidate objects for detected room labels.
        """
        # ------------------------------------------------------------------ #
        # Step 1: Early exclusion                                              #
        # ------------------------------------------------------------------ #
        surviving = [d for d in detections if not _is_excluded_token(d.text)]
        excluded = len(detections) - len(surviving)
        if excluded:
            logger.debug(f"Early exclusion removed {excluded} non-room tokens")

        # ------------------------------------------------------------------ #
        # Step 2: Compound-label merging                                       #
        # Tokens on the same text-line within MERGE_X_THRESHOLD pixels are    #
        # concatenated left-to-right to reconstruct multi-word labels.        #
        # ------------------------------------------------------------------ #
        # DPI-aware threshold: base 120px calibrated at 200 DPI.
        # At 300 DPI inter-word gap scales to ~180px; at 150 DPI to ~90px.
        _base_dpi = 200
        _current_dpi = getattr(getattr(self, "config", None), "dpi", _base_dpi) or _base_dpi
        MERGE_X_THRESHOLD = int(120 * _current_dpi / _base_dpi)
        Y_LINE_TOLERANCE = 20     # pixels; same-line check

        merged_detections = self._merge_compound_labels(
            surviving, MERGE_X_THRESHOLD, Y_LINE_TOLERANCE
        )

        # ------------------------------------------------------------------ #
        # Step 3: Classify as room-number or room-name                        #
        # ------------------------------------------------------------------ #
        number_candidates: List[RoomCandidate] = []
        name_candidates: List[RoomCandidate] = []

        for detection in merged_detections:
            text = detection.text.strip()

            is_number = bool(self._room_number_pattern.match(text))
            is_name = any(p.search(text) for p in self._room_name_patterns)

            if not (is_number or is_name):
                continue

            x_coords = [p[0] for p in detection.bbox]
            y_coords = [p[1] for p in detection.bbox]
            x = min(x_coords)
            y = min(y_coords)
            w = max(x_coords) - x
            h = max(y_coords) - y

            candidate = RoomCandidate(
                bbox=(x, y, w, h),
                room_number=text if is_number else "",
                room_name=text if is_name else "",
                confidence=detection.confidence,
                raw_text=text,
            )

            if is_number and not is_name:
                number_candidates.append(candidate)
            else:
                name_candidates.append(candidate)

        # ------------------------------------------------------------------ #
        # Step 4: Spatial linking – attach each orphan room-number to its     #
        # nearest room-name candidate within LINK_THRESHOLD pixels.           #
        # ------------------------------------------------------------------ #
        linked = self._link_room_numbers_to_names(number_candidates, name_candidates)
        candidates = name_candidates + linked

        logger.debug(
            f"find_room_candidates: {len(detections)} raw → "
            f"{len(surviving)} after exclusion → "
            f"{len(merged_detections)} after merge → "
            f"{len(candidates)} final candidates "
            f"({len(name_candidates)} names, {len(linked)} linked numbers)"
        )
        return candidates

    # ---------------------------------------------------------------------- #
    # Private helpers                                                          #
    # ---------------------------------------------------------------------- #

    def _merge_compound_labels(
        self,
        detections: List[TextDetection],
        x_threshold: int = 120,
        y_tolerance: int = 20,
    ) -> List[TextDetection]:
        """
        Merge adjacent same-line tokens into compound room labels.

        Algorithm — greedy window scan:
        1. Sort detections by (text-row, left-edge).
        2. Build adjacency chains: consecutive tokens on the same line within
           x_threshold pixels.
        3. Within each chain, scan left-to-right with windows of 2–4 tokens.
           For each position, try the longest window whose joined text matches
           a room pattern.  If a compound is found, emit it and advance past
           all constituent tokens.  If no compound is found, emit the single
           token and advance by 1.

        This correctly handles:
        - "BUILDING" + "STORAGE" → "BUILDING STORAGE"  (neither matches alone)
        - "FIRE" + "PUMP" + "ROOM" → "FIRE PUMP ROOM"  (none match alone)
        - "ELEVATOR" (self-matching) → emitted as-is, not absorbed
        - "BR" "LR" on same line → each emitted separately (each self-matches)
        """
        if not detections:
            return []

        def sort_key(d: TextDetection):
            _, cy = _centroid(d.bbox)
            return (round(cy / y_tolerance), min(p[0] for p in d.bbox))

        sorted_dets = sorted(detections, key=sort_key)

        def _self_matches(text: str) -> bool:
            return (
                bool(self._room_number_pattern.match(text))
                or any(p.search(text) for p in self._room_name_patterns)
            )

        def _make_merged(group: List[TextDetection]) -> TextDetection:
            """Produce a single TextDetection from a group of constituents."""
            g_sorted = sorted(group, key=lambda d: min(p[0] for p in d.bbox))
            combined_text = " ".join(d.text.strip() for d in g_sorted)
            all_x = [p[0] for d in group for p in d.bbox]
            all_y = [p[1] for d in group for p in d.bbox]
            combined_bbox = [
                [min(all_x), min(all_y)], [max(all_x), min(all_y)],
                [max(all_x), max(all_y)], [min(all_x), max(all_y)],
            ]
            total_len = sum(len(d.text) for d in group)
            avg_conf = (
                sum(d.confidence * len(d.text) for d in group) / total_len
                if total_len > 0 else group[0].confidence
            )
            return TextDetection(bbox=combined_bbox, text=combined_text, confidence=avg_conf)

        # ── Step 1: Build adjacency chains ─────────────────────────────────
        chains: List[List[TextDetection]] = []
        chain = [sorted_dets[0]]
        for det in sorted_dets[1:]:
            last = chain[-1]
            last_right = max(p[0] for p in last.bbox)
            next_left = min(p[0] for p in det.bbox)
            if (
                _same_line(last.bbox, det.bbox, y_tolerance)
                and (next_left - last_right) <= x_threshold
            ):
                chain.append(det)
            else:
                chains.append(chain)
                chain = [det]
        chains.append(chain)

        # ── Step 2: Greedy window scan within each chain ───────────────────
        result: List[TextDetection] = []
        MAX_WINDOW = 4  # max tokens to try combining

        for ch in chains:
            k = 0
            while k < len(ch):
                if len(ch) == 1 or k == len(ch) - 1:
                    result.append(ch[k])
                    k += 1
                    continue

                # Try longest window first, then shorter
                best_end: Optional[int] = None
                best_merged: Optional[TextDetection] = None

                for window in range(min(MAX_WINDOW, len(ch) - k), 1, -1):
                    group = ch[k: k + window]
                    merged = _make_merged(group)
                    if _self_matches(merged.text.strip()):
                        best_end = k + window
                        best_merged = merged
                        logger.debug(
                            f"Compound found: {[d.text for d in group]} → "
                            f"'{merged.text}'"
                        )
                        break

                if best_merged is not None:
                    result.append(best_merged)
                    k = best_end
                else:
                    # No compound match at this position; emit single token
                    result.append(ch[k])
                    k += 1

        return result

    def _link_room_numbers_to_names(
        self,
        number_candidates: List[RoomCandidate],
        name_candidates: List[RoomCandidate],
        link_threshold: int = 300,
    ) -> List[RoomCandidate]:
        """
        Attach orphan room-numbers to their nearest room-name candidate.

        For each number-only candidate, find the closest name candidate
        (by centroid distance).  If within link_threshold pixels, copy the
        room_number into the matched name candidate in-place and do NOT
        return the number as a standalone entry.

        Numbers that cannot be matched within the threshold are returned as
        standalone candidates so they are not silently dropped.

        Args:
            number_candidates: RoomCandidate objects with room_number set, room_name empty.
            name_candidates:   RoomCandidate objects with room_name set.
            link_threshold:    Max centroid distance (px) to consider a match.

        Returns:
            List of standalone (unmatched) number candidates.
        """
        unmatched: List[RoomCandidate] = []

        for num_cand in number_candidates:
            nx = num_cand.bbox[0] + num_cand.bbox[2] / 2
            ny = num_cand.bbox[1] + num_cand.bbox[3] / 2

            best_dist = float("inf")
            best_name_cand = None

            for name_cand in name_candidates:
                cnx = name_cand.bbox[0] + name_cand.bbox[2] / 2
                cny = name_cand.bbox[1] + name_cand.bbox[3] / 2
                dist = ((nx - cnx) ** 2 + (ny - cny) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_name_cand = name_cand

            if best_name_cand is not None and best_dist <= link_threshold:
                # Enrich the name candidate with the room number (in-place)
                if not best_name_cand.room_number:
                    best_name_cand.room_number = num_cand.room_number
                    logger.debug(
                        f"Linked room #{num_cand.room_number} → "
                        f"'{best_name_cand.room_name}' (dist={best_dist:.0f}px)"
                    )
            else:
                # Keep as standalone so it is not silently discarded
                unmatched.append(num_cand)
                logger.debug(
                    f"Unmatched room number #{num_cand.room_number} "
                    f"(nearest dist={best_dist:.0f}px > {link_threshold}px)"
                )

        return unmatched

    def extract_and_find_rooms(self, image_path: str | Path):
        """
        Extract text and find room candidates, returning BOTH results.

        Returns a tuple (candidates, raw_detections) so callers can pass
        raw_detections directly to AbbreviationOCRRecovery instead of
        calling extract_text() a second time (avoids double OCR per image).

        Args:
            image_path: Path to the image file.

        Returns:
            Tuple of (List[RoomCandidate], List[TextDetection]).
        """
        raw_detections = self.extract_text(image_path)
        candidates = self.find_room_candidates(raw_detections)
        return candidates, raw_detections

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
