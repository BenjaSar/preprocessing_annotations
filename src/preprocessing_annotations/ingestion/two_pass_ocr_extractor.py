"""
Two-Pass OCR Strategy: PaddleOCR (all regions) + VLM fallback (low-confidence < 0.7).

This module implements a robust OCR pipeline:
1. Pass 1: Run PaddleOCR on all regions (fast, baseline)
2. Pass 2 (VLM Fallback): For low-confidence results (< 0.7), use VLM as fallback
3. Merge: Return best results combining both passes

Architecture:
- TwoPassOCRExtractor wraps MEPTextExtractor for Pass 1
- Uses VLM backend's detect_rooms() for Pass 2 fallback
- Spatial matching merges OCR and VLM results by bbox proximity
- Confidence-based selection returns higher-confidence labels

Usage:
    config = PipelineConfig()
    extractor = TwoPassOCRExtractor(
        ocr_config=config.ocr_config,
        vlm_backend=vlm_backend,
        confidence_threshold=0.7
    )
    rooms, raw = extractor.extract_and_find_rooms_with_vlm_fallback("image.png")
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    from ..config import OCRConfig, PipelineConfig
    from .ocr_extractor import MEPTextExtractor, RoomCandidate, TextDetection
    from ..vlm.tiled_rooms import detect_rooms_in_tiles, splitter_for_image
    from ..vlm.vlm_backend import VLMBackend
except ImportError:
    from config import OCRConfig, PipelineConfig
    from ocr_extractor import MEPTextExtractor, RoomCandidate, TextDetection
    from tiled_rooms import detect_rooms_in_tiles, splitter_for_image
    from vlm_backend import VLMBackend

logger = logging.getLogger(__name__)

# Centroid distance below which a Pass 1 and a Pass 2 room are the same room.
# Caller-overridable argument default, kept here so the number has one home.
DEFAULT_BBOX_MATCH_DISTANCE_PX = 100.0


class TwoPassOCRExtractor:
    """
    Two-pass OCR extraction with VLM fallback for low-confidence results.
    
    Pass 1: PaddleOCR on all regions (baseline, fast)
    Pass 2: VLM as fallback for low-confidence results (< confidence_threshold)
    
    Attributes:
        ocr_extractor: MEPTextExtractor for Pass 1 (PaddleOCR)
        vlm_backend: VLMBackend for Pass 2 (VLM fallback)
        confidence_threshold: Confidence threshold below which to use VLM fallback (0.0-1.0)
    """
    
    def __init__(
        self,
        ocr_config: OCRConfig,
        vlm_backend: Optional[VLMBackend] = None,
        confidence_threshold: float = 0.92,
        vlm_fallback_enabled: bool = True,
        tiling_config: Optional[PipelineConfig] = None,
    ):
        """
        Initialize two-pass OCR extractor.

        Args:
            ocr_config: OCRConfig for Pass 1 (PaddleOCR)
            vlm_backend: VLMBackend for Pass 2 fallback (optional)
            vlm_fallback_enabled: False skips Pass 2 while leaving the backend
                (and therefore the local-VLM CPU forcing below) in place.
            tiling_config: the grid Pass 2 detects on, read by
                TileSplitter.for_page. None detects the page whole, which is
                what a caller without a PipelineConfig gets.
            confidence_threshold: Use VLM fallback if confidence < this value (default: 0.92).
                PaddleOCR typically returns >0.97 on legible floor plan text; 0.92 triggers
                the fallback only for tokens that are genuinely ambiguous.
        
        Raises:
            ValueError: If confidence_threshold not in [0.0, 1.0]
        """
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be in [0.0, 1.0], got {confidence_threshold}")
        
        # If using local VLM model (Qwen/Unsloth), force PaddleOCR to CPU to save VRAM
        if vlm_backend is not None:
            backend_class_name = vlm_backend.__class__.__name__
            if backend_class_name in ("Qwen2_5VLBackend", "UnslothQwenBackend"):
                logger.info(
                    f"Detected local VLM backend ({backend_class_name}). "
                    f"Forcing PaddleOCR to CPU to preserve VRAM for VLM model."
                )
                ocr_config.device = "cpu"
            else:
                logger.debug(f"Using remote VLM backend ({backend_class_name}). PaddleOCR device: {ocr_config.device}")
        
        self.ocr_extractor = MEPTextExtractor(ocr_config)
        self.vlm_backend = vlm_backend
        self.confidence_threshold = confidence_threshold
        self.vlm_fallback_enabled = vlm_fallback_enabled
        self.tiling_config = tiling_config
        
        logger.info(
            f"Initialized TwoPassOCRExtractor with confidence_threshold={confidence_threshold}"
        )
        if vlm_backend is None:
            logger.warning(
                "vlm_backend not provided; will use PaddleOCR only (no VLM fallback)"
            )
    
    def find_rooms_pass1(
        self,
        image_path: Union[str, Path],
        extra_detections: Optional[List[TextDetection]] = None,
    ) -> Tuple[List[RoomCandidate], List[TextDetection]]:
        """Pass 1 only: PaddleOCR extraction (no VLM).

        Separated from Pass 2 so the caller can run the memory-heavy PaddleOCR
        sweep for every page, release its native heap, and only then load the VLM
        for Pass 2 — keeping the two engines from being resident at the same time.

        Args:
            image_path: Path to floor plan image
            extra_detections: Passed straight through to
                MEPTextExtractor.extract_and_find_rooms -- see that
                method's docstring for the merge contract (candidate
                classification only; the returned raw_detections stays
                OCR-only). Default None -- byte-identical to prior
                behavior.

        Returns:
            Tuple of (pass1_room_candidates, raw_text_detections)

        Raises:
            OCRError: If OCR extraction fails
        """
        image_path = Path(image_path)
        logger.info(f"Pass 1 (PaddleOCR): Extracting rooms from {image_path.name}")
        try:
            pass1_rooms, raw_detections = self.ocr_extractor.extract_and_find_rooms(
                image_path, extra_detections=extra_detections
            )
        except Exception as e:
            logger.error(f"Pass 1 (PaddleOCR) failed: {e}")
            raise
        logger.info(f"Pass 1 complete: extracted {len(pass1_rooms)} room candidates")
        return pass1_rooms, raw_detections

    def refine_with_vlm(
        self,
        image_path: Union[str, Path],
        pass1_rooms: List[RoomCandidate],
    ) -> List[RoomCandidate]:
        """Pass 2: VLM fallback + merge for low-confidence Pass-1 rooms.

        No-op (returns pass1_rooms unchanged) when there is no VLM backend or no
        low-confidence room — identical gating to the original single-call path,
        so results match regardless of when this runs.

        Args:
            image_path: Path to floor plan image
            pass1_rooms: Room candidates from find_rooms_pass1

        Returns:
            Merged room candidates (or pass1_rooms unchanged when Pass 2 skipped)
        """
        image_path = Path(image_path)

        low_confidence_rooms = [
            r for r in pass1_rooms
            if r.confidence < self.confidence_threshold
        ]

        if not low_confidence_rooms or self.vlm_backend is None or not self.vlm_fallback_enabled:
            logger.info(
                f"Pass 2 skipped: {len(low_confidence_rooms)} low-confidence rooms, "
                f"VLM backend available: {self.vlm_backend is not None}, "
                f"fallback enabled: {self.vlm_fallback_enabled}"
            )
            return pass1_rooms

        logger.info(
            f"Pass 2 (VLM Fallback): Processing {len(low_confidence_rooms)} "
            f"low-confidence rooms (confidence < {self.confidence_threshold})"
        )
        try:
            pass2_rooms = self._extract_rooms_via_vlm(image_path)
            if not pass2_rooms:
                logger.warning("Pass 2 (VLM) returned no results, using Pass 1 only")
                return pass1_rooms
        except Exception as e:
            logger.warning(f"Pass 2 (VLM) failed, using Pass 1 results: {e}")
            return pass1_rooms

        merged_rooms = self._merge_room_candidates(pass1_rooms, pass2_rooms)
        logger.info(f"Merged {len(pass1_rooms)} OCR + {len(pass2_rooms)} VLM → {len(merged_rooms)} final candidates")
        return merged_rooms

    def release_ocr(self) -> None:
        """Release Pass-1 PaddleOCR native memory (safe to call once Pass 1 is done
        for all pages). Delegates to the underlying MEPTextExtractor."""
        self.ocr_extractor.release()

    def extract_and_find_rooms_with_vlm_fallback(
        self,
        image_path: Union[str, Path],
    ) -> Tuple[List[RoomCandidate], List[TextDetection]]:
        """
        Extract rooms using two-pass strategy: PaddleOCR + VLM fallback.

        Pass 1: Run PaddleOCR on all regions, returns baseline room candidates
        Pass 2: For low-confidence results (< threshold), use VLM as fallback
        Merge: Combine results, preferring higher confidence results

        Kept as the single-call composition of find_rooms_pass1 + refine_with_vlm
        for callers that do not need to separate the two passes in time.

        Args:
            image_path: Path to floor plan image

        Returns:
            Tuple of (room_candidates, raw_text_detections)

        Raises:
            OCRError: If OCR extraction fails
        """
        pass1_rooms, raw_detections = self.find_rooms_pass1(image_path)
        merged_rooms = self.refine_with_vlm(image_path, pass1_rooms)
        return merged_rooms, raw_detections
    
    def _detect_rooms(self, image_path: Path) -> List[Dict[str, Any]]:
        """Pass-2 detections for one page, in that page's coordinates.

        Tiled when a tiling_config says this page is tiled, because a
        whole-page call does not ground small room labels: measured on
        sprint1_verify55, 1 of 712 name-matched Pass-2 candidates had its own
        OCR token inside its box, against 1773 of 1773 for Pass 1. The tiling
        itself is the pipeline's (vlm.tiled_rooms), not a second copy of it.
        """
        tiling = (
            splitter_for_image(image_path, self.tiling_config)
            if self.tiling_config is not None
            else None
        )
        if tiling is None:
            return self.vlm_backend.detect_rooms(image_path)
        image, splitter = tiling
        return detect_rooms_in_tiles(self.vlm_backend, image, splitter, image_path.name)

    def _extract_rooms_via_vlm(self, image_path: Path) -> List[RoomCandidate]:
        """
        Extract rooms using VLM backend (Pass 2 fallback).

        Args:
            image_path: Path to floor plan image

        Returns:
            List of RoomCandidate objects from VLM detection

        Raises:
            Exception: If VLM detection fails
        """
        if self.vlm_backend is None:
            return []

        try:
            vlm_detections = self._detect_rooms(image_path)

            if not vlm_detections:
                logger.warning("VLM detect_rooms returned empty list")
                return []
            
            # Convert VLM detections to RoomCandidate format
            room_candidates = []
            for detection in vlm_detections:
                # VLM returns bbox as [x1, y1, x2, y2], convert to (x, y, w, h) to match Pass 1 format
                bbox = detection.get("bbox")
                if bbox and isinstance(bbox, (list, tuple)):
                    if len(bbox) >= 4:
                        # Convert [x1, y1, x2, y2] -> (x, y, w, h)
                        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                        bbox_tuple = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                    else:
                        logger.warning(f"Invalid bbox format: {bbox}, skipping")
                        continue
                else:
                    logger.warning(f"No bbox in VLM detection: {detection}")
                    continue
                
                room_name = detection.get("room_name", "UNKNOWN")
                confidence = float(detection.get("confidence", 0.5))
                room_number = detection.get("room_id", "")
                
                candidate = RoomCandidate(
                    bbox=bbox_tuple,
                    room_number=room_number,
                    room_name=room_name,
                    confidence=confidence,
                    raw_text=room_name,
                    name_expanded=None
                )
                room_candidates.append(candidate)
            
            logger.info(f"VLM extracted {len(room_candidates)} room candidates")
            return room_candidates
            
        except Exception as e:
            logger.error(f"VLM room extraction failed: {e}")
            raise
    
    @staticmethod
    def _closest_unclaimed(
        room: RoomCandidate,
        candidates: List[RoomCandidate],
        claimed: Set[int],
        distance_threshold: float,
    ) -> Optional[int]:
        """Index of the nearest still-unclaimed candidate within the threshold.

        One rule for "these two are the same room", so the high-confidence and
        the low-confidence branch cannot drift apart. A claimed candidate is
        skipped: a Pass 2 room describes one place, so it can stand in for one
        Pass 1 room, not for every Pass 1 room that happens to sit near it.
        """
        def centroid(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
            x, y, w, h = bbox
            return x + w / 2, y + h / 2

        def distance(first: Tuple[int, int, int, int], second: Tuple[int, int, int, int]) -> float:
            (cx1, cy1), (cx2, cy2) = centroid(first), centroid(second)
            return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

        best_index: Optional[int] = None
        best_distance = distance_threshold
        for index, candidate in enumerate(candidates):
            if index in claimed:
                continue
            candidate_distance = distance(room.bbox, candidate.bbox)
            if candidate_distance < best_distance:
                best_distance, best_index = candidate_distance, index
        return best_index

    def _merge_room_candidates(
        self,
        pass1_rooms: List[RoomCandidate],
        pass2_rooms: List[RoomCandidate],
        bbox_distance_threshold: float = DEFAULT_BBOX_MATCH_DISTANCE_PX,
    ) -> List[RoomCandidate]:
        """
        Merge room candidates from Pass 1 (OCR) and Pass 2 (VLM).

        Strategy:
        1. Every Pass 1 room claims its nearest unclaimed Pass 2 room within the
           threshold -- the two passes read the same place.
        2. A high-confidence Pass 1 room keeps its own label and consumes the
           claim, so the Pass 2 duplicate is not appended a second time.
        3. A low-confidence Pass 1 room is replaced by the Pass 2 room it
           claimed, and kept as-is when it claimed nothing.
        4. Pass 2 rooms nobody claimed are added as new candidates.

        Args:
            pass1_rooms: Room candidates from PaddleOCR
            pass2_rooms: Room candidates from VLM
            bbox_distance_threshold: Max centroid distance (px) to call two rooms the same

        Returns:
            Merged list of RoomCandidate objects
        """
        merged: List[RoomCandidate] = []
        claimed: Set[int] = set()

        for p1_room in pass1_rooms:
            match_index = self._closest_unclaimed(
                p1_room, pass2_rooms, claimed, bbox_distance_threshold
            )
            if match_index is not None:
                claimed.add(match_index)

            if p1_room.confidence >= self.confidence_threshold:
                if match_index is not None:
                    logger.debug(
                        f"High-confidence OCR ({p1_room.room_name}, conf={p1_room.confidence:.2f}) "
                        f"kept; VLM duplicate ({pass2_rooms[match_index].room_name}) dropped"
                    )
                merged.append(p1_room)
                continue

            if match_index is None:
                logger.debug(
                    f"No VLM match for low-confidence OCR ({p1_room.room_name}, "
                    f"conf={p1_room.confidence:.2f}), using OCR result"
                )
                merged.append(p1_room)
                continue

            p2_room = pass2_rooms[match_index]
            logger.debug(
                f"Matched low-confidence OCR ({p1_room.room_name}, conf={p1_room.confidence:.2f}) "
                f"→ VLM ({p2_room.room_name}, conf={p2_room.confidence:.2f})"
            )
            merged.append(p2_room)

        for index, p2_room in enumerate(pass2_rooms):
            if index not in claimed:
                logger.debug(f"Adding unmatched VLM result: {p2_room.room_name} (conf={p2_room.confidence:.2f})")
                merged.append(p2_room)

        return merged
