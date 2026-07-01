"""
OCR text extraction module for MEP floor plans.

This module extracts room labels, panel identifiers, and other text
from floor plan images using a configurable OCR backend (PaddleOCR by default)
with optional preprocessing. Supports both PaddleOCR (default) and EasyOCR (legacy).
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
    from .ocr_adapter import OCRFactory, TextDetection as AdapterTextDetection
except ImportError:
    from config import OCRConfig
    from ocr_adapter import OCRFactory, TextDetection as AdapterTextDetection

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


def _normalized_bbox_to_quad(bbox: Tuple[float, float, float, float]) -> List[List[int]]:
    """Convert normalized bbox [x1, y1, x2, y2] to quadrilateral [[x,y], [x,y], [x,y], [x,y]].
    
    Args:
        bbox: (x1, y1, x2, y2) - top-left and bottom-right corners
        
    Returns:
        [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] - counter-clockwise from top-left
    """
    x1, y1, x2, y2 = bbox
    return [[int(x1), int(y1)], [int(x2), int(y1)], [int(x2), int(y2)], [int(x1), int(y2)]]


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

    IMMUTABILITY CONTRACT: room_name field is IMMUTABLE. It contains the original
    detected text from OCR and must NEVER be modified by downstream stages.
    
    If abbreviations are expanded (e.g., "BR" → "BEDROOM"), the expansion is
    tracked in the name_expanded field, not in room_name.

    Attributes:
        bbox: Bounding box as (x, y, width, height).
        room_number: Detected room number (e.g., "113", "S1.100").
        room_name: Detected room name (e.g., "BR", "MECH RM A") — IMMUTABLE.
        confidence: OCR confidence score (0.0 to 1.0).
        raw_text: Original text before classification.
        name_expanded: Abbreviation expansion if applicable (e.g., "BR" → "BEDROOM").
    """

    bbox: Tuple[int, int, int, int]
    room_number: str
    room_name: str  # ← IMMUTABLE — original OCR text only
    confidence: float
    raw_text: str = ""
    name_expanded: Optional[str] = None  # ← Expansion tracking only


@dataclass
@dataclass
class TextDetection:
    """
    Raw text detection result.

    Attributes:
        bbox: Bounding box as list of 4 corner points [[x,y], [x,y], [x,y], [x,y]].
        text: Detected text string.
        confidence: Detection confidence (0.0 to 1.0).
    """

    bbox: List[List[int]]
    text: str
    confidence: float


class MEPTextExtractor:
    """
    Extract text from floor plan images using pluggable OCR backends + domain-specific heuristics.

    This module handles all aspects of OCR-based text extraction:
      - image preprocessing (adaptive CLAHE)
      - raw text detection via configurable backend (EasyOCR or PaddleOCR)
      - room candidate identification (vertical stack merging, regex filters)
      - abbreviation recovery
      - room number ↔ room name linking

    Usage:
        config = OCRConfig(backend='paddleocr')  # or 'easyocr'
        extractor = MEPTextExtractor(config)
        candidates, raw_detections = extractor.extract_and_find_rooms("path/to/image.png")

    Attributes:
        config: OCRConfig instance for all settings.
        _ocr_backend: Lazy-initialized OCR backend (EasyOCR or PaddleOCR).
    """

    def __init__(self, config: OCRConfig):
        """Initialize the text extractor with OCR configuration.

        Args:
            config: OCRConfig instance containing model/preprocessing parameters.
                    The backend field determines which OCR engine to use ('easyocr' or 'paddleocr').
        """
        self.config = config
        self._ocr_backend = None

        # Compile patterns for efficiency
        self._room_number_pattern = re.compile(
            self.config.room_number_pattern, re.IGNORECASE
        )
        self._room_name_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.config.room_name_patterns
        ]

    @property
    def ocr_backend(self):
        """Lazy initialization of OCR backend (EasyOCR or PaddleOCR)."""
        if self._ocr_backend is None:
            try:
                self._ocr_backend = OCRFactory.create(self.config)
                logger.info(
                    f"Initialized OCR backend: {self.config.backend} "
                    f"with languages={self.config.languages}, device={self.config.device}"
                )
            except Exception as e:
                raise OCRError(f"Failed to initialize OCR backend '{self.config.backend}': {e}") from e
        return self._ocr_backend

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

        This method:
        1. Loads the image from disk
        2. Applies optional preprocessing (adaptive CLAHE)
        3. Runs the configured OCR backend (EasyOCR or PaddleOCR)
        4. Filters by length-aware confidence thresholds
        5. Returns normalized TextDetection objects

        Args:
            image_path: Path to the image file.

        Returns:
            List of TextDetection objects with bbox in [x1, y1, x2, y2] format.

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
        else:
            processed = image

        # Run OCR using the configured backend (EasyOCR or PaddleOCR)
        try:
            raw_detections = self.ocr_backend.extract_text(processed)
        except Exception as e:
            raise OCRError(f"OCR failed on {image_path}: {e}") from e

        # Apply length-aware confidence thresholds.
        # OCR models systematically underestimate confidence for short tokens
        # (fewer characters = less context) so abbreviations like BR/LR at 0.6
        # confidence are often correct, while a 20-char token at 0.6 is likely
        # garbled. Conversely, single characters need very high confidence to
        # avoid spurious symbol detections.
        MIN_CONF_BY_LEN = {1: 0.85, 2: 0.70, 3: 0.65, 4: 0.60}
        DEFAULT_MIN_CONF = self.config.confidence_threshold  # 0.50

        detections = []
        for detection in raw_detections:
            min_conf = MIN_CONF_BY_LEN.get(len(detection.text.strip()), DEFAULT_MIN_CONF)
            if detection.confidence >= min_conf:
                # Convert normalized bbox [x1,y1,x2,y2] to quadrilateral format
                # expected by downstream code
                quad_bbox = _normalized_bbox_to_quad(detection.bbox)
                detections.append(
                    TextDetection(
                        bbox=quad_bbox,
                        text=detection.text,
                        confidence=detection.confidence,
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
        #                                                                      #
        # Floor plan labels appear in TWO spatial arrangements:               #
        #   (a) Horizontal: "BUILDING STORAGE" side-by-side on one line       #
        #   (b) Vertical stack:  FIRE          ← separate OCR tokens          #
        #                        PUMP          ← on different text lines       #
        #                        ROOM          ← horizontal merge misses these #
        #                                                                      #
        # Root cause of missing compound rooms (confirmed 2026-02-23):        #
        # _merge_compound_labels only chains tokens whose centroids are within #
        # Y_LINE_TOLERANCE=20px — i.e., horizontally adjacent on the same     #
        # text row.  Vertically stacked labels span multiple rows so          #
        # _same_line() returns False and they never enter the same chain.     #
        # "FIRE"/"PUMP"/"ROOM" each fail every room pattern individually, so  #
        # all three are discarded in Step 3 classification.                   #
        #                                                                      #
        # Fix: run _merge_vertical_stacks FIRST (Step 2a), then the existing  #
        # horizontal merge (Step 2b) on the result.  Two-pass coverage        #
        # reconstructs all compound labels regardless of layout orientation.  #
        # ------------------------------------------------------------------ #
        _base_dpi = 200
        _current_dpi = getattr(getattr(self, "config", None), "dpi", _base_dpi) or _base_dpi
        MERGE_X_THRESHOLD = int(120 * _current_dpi / _base_dpi)
        Y_LINE_TOLERANCE = 20     # pixels; same-line check (horizontal pass)

        # Step 2a: Vertical-stack merging (new — handles FIRE/PUMP/ROOM etc.)
        after_vertical = self._merge_vertical_stacks(surviving, dpi=_current_dpi)

        # Step 2b: Horizontal merging (existing, now runs on vertically-merged output)
        merged_detections = self._merge_compound_labels(
            after_vertical, MERGE_X_THRESHOLD, Y_LINE_TOLERANCE
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

    def _merge_vertical_stacks(
        self,
        detections: List[TextDetection],
        dpi: int = 200,
    ) -> List[TextDetection]:
        """
        Merge vertically stacked tokens that form a single compound room label.

        Problem solved
        ──────────────
        In architectural floor plans, multi-word room labels are frequently
        stacked vertically to fit inside narrow room boundaries:

            FIRE          BUILDING          COMMUNITY          ELEVATOR
            PUMP          STORAGE           FACILITY            MACHINE
            ROOM                                                  ROOM

        Each individual word (FIRE, PUMP, ROOM) fails every room-name pattern,
        so all three tokens are silently discarded in Step 3 classification.
        The horizontal merger in _merge_compound_labels never sees them in the
        same chain because _same_line() returns False across rows.

        Algorithm
        ─────────
        1.  Sort tokens top-to-bottom by Y centroid.
        2.  For each unprocessed token T, collect candidate "stack members":
            tokens below T whose X range overlaps T's by ≥ X_OVERLAP_FRACTION
            and whose top edge is within MAX_Y_GAP pixels of T's bottom edge.
        3.  Try combining T plus the next 1–3 stack members (longest first).
            Accept the first combination that matches a room pattern.
        4.  Emit the merged token and mark all constituent tokens as consumed.
            If no combination matches, emit T unchanged.

        Parameters
        ──────────
        dpi : int
            Extraction DPI.  Used to scale the Y-gap threshold so the same
            code works at 150 DPI (default), 200 DPI, and 300 DPI.
        """
        if not detections:
            return []

        def _self_matches(text: str) -> bool:
            return (
                bool(self._room_number_pattern.match(text))
                or any(p.search(text) for p in self._room_name_patterns)
            )

        def _x_range(d: TextDetection):
            xs = [p[0] for p in d.bbox]
            return min(xs), max(xs)

        def _x_overlap_frac(a: TextDetection, b: TextDetection) -> float:
            """Fraction of the shorter token's X span that overlaps the other."""
            a_min, a_max = _x_range(a)
            b_min, b_max = _x_range(b)
            overlap = max(0, min(a_max, b_max) - max(a_min, b_min))
            shorter = min(a_max - a_min, b_max - b_min)
            return overlap / shorter if shorter > 0 else 0.0

        def _make_stack_merge(group: List[TextDetection]) -> TextDetection:
            """Merge a vertical stack top-to-bottom."""
            g_sorted = sorted(group, key=lambda d: _centroid(d.bbox)[1])
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

        # DPI-aware Y-gap threshold.
        # At 200 DPI: typical room-label text is ~20–25px tall; line spacing
        # ~1.2–1.5× that gives 24–38px.  Use 60px to be lenient.
        MAX_Y_GAP = int(60 * dpi / 200)
        X_OVERLAP_FRACTION = 0.30   # ≥30% X overlap → same column
        MAX_STACK_DEPTH = 4         # max tokens in one vertical stack

        # Sort top-to-bottom by Y centroid
        sorted_dets = sorted(detections, key=lambda d: _centroid(d.bbox)[1])
        consumed = [False] * len(sorted_dets)
        result: List[TextDetection] = []

        for i, anchor in enumerate(sorted_dets):
            if consumed[i]:
                continue

            # Collect tokens that could stack below anchor
            stack_candidates: List[tuple] = []   # (index, token)
            anchor_cx, anchor_cy = _centroid(anchor.bbox)
            anchor_bottom = max(p[1] for p in anchor.bbox)

            for j in range(i + 1, len(sorted_dets)):
                if consumed[j]:
                    continue
                below = sorted_dets[j]
                below_cx, below_cy = _centroid(below.bbox)
                below_top = min(p[1] for p in below.bbox)

                # Must be below anchor
                if below_cy <= anchor_cy:
                    continue
                # Top edge of below must be within MAX_Y_GAP of anchor's bottom
                if below_top - anchor_bottom > MAX_Y_GAP:
                    break  # sorted by Y — no further tokens can qualify
                # X ranges must overlap enough to be in the same column
                if _x_overlap_frac(anchor, below) < X_OVERLAP_FRACTION:
                    continue

                stack_candidates.append((j, below))
                if len(stack_candidates) >= MAX_STACK_DEPTH - 1:
                    break

            if not stack_candidates:
                # No vertical neighbours; emit as-is for horizontal pass
                result.append(anchor)
                consumed[i] = True
                continue

            # Try longest stack first, then shorter
            merged_token = None
            best_indices: List[int] = []

            for depth in range(min(MAX_STACK_DEPTH - 1, len(stack_candidates)), 0, -1):
                group_tokens = [anchor] + [t for _, t in stack_candidates[:depth]]
                group_indices = [i] + [idx for idx, _ in stack_candidates[:depth]]
                candidate_text = " ".join(d.text.strip() for d in
                                          sorted(group_tokens, key=lambda d: _centroid(d.bbox)[1]))
                if _self_matches(candidate_text):
                    merged_token = _make_stack_merge(group_tokens)
                    best_indices = group_indices
                    logger.debug(
                        f"Vertical stack merged: "
                        f"{[d.text for d in sorted(group_tokens, key=lambda d: _centroid(d.bbox)[1])]} "
                        f"→ '{merged_token.text}'"
                    )
                    break

            if merged_token is not None:
                result.append(merged_token)
                for idx in best_indices:
                    consumed[idx] = True
            else:
                # No compound match for any stack depth; emit anchor alone
                result.append(anchor)
                consumed[i] = True

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

    # Maximum pixel dimension for OCR input. OCR must run on the SAME resolution
    # as the annotation image_size field. PDF extraction produces 9600×7200 originals;
    # Step 3a VLM pre-resize targets 4500px. If OCR runs on 9600px, its bbox coords
    # are ≈2.13× larger than the annotation space → all OCR anchors mislocalized.
    # Running at ≤4500px keeps coord spaces aligned. 4608 det_limit is still used
    # (separately configured), giving full in-plan label recall at 4500px input.
    _OCR_MAX_DIM_PX: int = 4500

    def extract_and_find_rooms(self, image_path: str | Path):
        """
        Extract text and find room candidates, returning BOTH results.

        Resizes the image to ≤_OCR_MAX_DIM_PX before OCR so that OCR bbox
        coordinates match the annotation image_size (also ≤4500px after Step 3a
        VLM pre-resize). Without this, OCR on 9600px originals produces coords
        ≈2.13× off, causing all OCR anchors to be mislocalized.

        Args:
            image_path: Path to the image file.

        Returns:
            Tuple of (List[RoomCandidate], List[TextDetection]).
        """
        image_path = Path(image_path)
        ocr_path = image_path  # default: use original

        # Resize in-memory to ≤_OCR_MAX_DIM_PX if the image is larger
        import tempfile, os
        _tmp = None
        try:
            from PIL import Image as _PIL
            with _PIL.open(image_path) as _img:
                orig_w, orig_h = _img.size
            if max(orig_w, orig_h) > self._OCR_MAX_DIM_PX:
                scale = self._OCR_MAX_DIM_PX / max(orig_w, orig_h)
                new_w, new_h = int(orig_w * scale), int(orig_h * scale)
                logger.info(
                    f"OCR resize: {orig_w}x{orig_h} → {new_w}x{new_h} "
                    f"(scale={scale:.3f}) to match annotation coordinate space"
                )
                with _PIL.open(image_path) as _img:
                    resized = _img.resize((new_w, new_h), _PIL.Resampling.LANCZOS)
                _fd, _tmp = tempfile.mkstemp(suffix=".png")
                os.close(_fd)
                resized.save(_tmp, "PNG")
                ocr_path = Path(_tmp)
        except Exception as _e:
            logger.warning(f"OCR pre-resize failed ({_e}), using original")

        try:
            raw_detections = self.extract_text(ocr_path)
        finally:
            if _tmp and os.path.exists(_tmp):
                try:
                    os.unlink(_tmp)
                except OSError:
                    pass

        candidates = self.find_room_candidates(raw_detections)
        return candidates, raw_detections

    def compute_exclusion_zones(
        self, detections: List[TextDetection], min_cluster_tokens: int = 3
    ) -> List[Tuple[int, int, int, int]]:
        """Compute bounding-box exclusion zones from excluded-token clusters (FIX-4).

        Floor plans carry BOM tables, panel schedules, and title blocks whose text
        was detected by OCR but excluded from room candidates via _is_excluded_token().
        Rooms/spaces whose centroid lands inside one of these zones are mislocalized
        (VLM/OCR read a layout element label and placed a detection in the margin).

        Strategy:
          1. Collect all TextDetections where _is_excluded_token() is True.
          2. Cluster them spatially: any token within 300px of an existing cluster
             is joined to it.  Isolated excluded tokens (n < min_cluster_tokens)
             are ignored — they are rare equipment labels scattered in the drawing.
          3. Return the bounding rectangle of each qualifying cluster, expanded by
             50px on each side so a room centroid just outside the table still hits.

        Args:
            detections:          All raw TextDetection objects from extract_text().
            min_cluster_tokens:  Minimum tokens in a cluster to form a zone (default 3).

        Returns:
            List of (x1, y1, x2, y2) exclusion zone rectangles in image pixel coords.
        """
        excluded_dets = [d for d in detections if _is_excluded_token(d.text)]
        if not excluded_dets:
            return []

        # Simple greedy clustering by proximity (300px radius)
        CLUSTER_RADIUS = 300
        ZONE_PAD = 50
        clusters: list = []   # each entry = list of TextDetection

        for det in excluded_dets:
            bx, by, bw, bh = det.bbox
            cx, cy = bx + bw // 2, by + bh // 2
            placed = False
            for cluster in clusters:
                for member in cluster:
                    mx, my = member.bbox[0] + member.bbox[2] // 2, member.bbox[1] + member.bbox[3] // 2
                    if abs(cx - mx) <= CLUSTER_RADIUS and abs(cy - my) <= CLUSTER_RADIUS:
                        cluster.append(det)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                clusters.append([det])

        zones = []
        for cluster in clusters:
            if len(cluster) < min_cluster_tokens:
                continue
            xs = [d.bbox[0] for d in cluster]
            ys = [d.bbox[1] for d in cluster]
            x2s = [d.bbox[0] + d.bbox[2] for d in cluster]
            y2s = [d.bbox[1] + d.bbox[3] for d in cluster]
            zones.append((
                max(0, min(xs) - ZONE_PAD),
                max(0, min(ys) - ZONE_PAD),
                max(x2s) + ZONE_PAD,
                max(y2s) + ZONE_PAD,
            ))
            logger.debug(
                f"Exclusion zone: {len(cluster)} excluded tokens → "
                f"[{zones[-1][0]},{zones[-1][1]},{zones[-1][2]},{zones[-1][3]}]"
            )

        return zones

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
