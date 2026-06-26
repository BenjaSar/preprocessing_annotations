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
from typing import Dict, List, Optional, Tuple, Union

try:
    from .config import OCRConfig, PipelineConfig
    from .ocr_extractor import MEPTextExtractor, RoomCandidate, TextDetection
    from .vlm_backend import VLMBackend
except ImportError:
    from config import OCRConfig, PipelineConfig
    from ocr_extractor import MEPTextExtractor, RoomCandidate, TextDetection
    from vlm_backend import VLMBackend

logger = logging.getLogger(__name__)


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
    ):
        """
        Initialize two-pass OCR extractor.

        Args:
            ocr_config: OCRConfig for Pass 1 (PaddleOCR)
            vlm_backend: VLMBackend for Pass 2 fallback (optional)
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
        
        logger.info(
            f"Initialized TwoPassOCRExtractor with confidence_threshold={confidence_threshold}"
        )
        if vlm_backend is None:
            logger.warning(
                "vlm_backend not provided; will use PaddleOCR only (no VLM fallback)"
            )
    
    def extract_and_find_rooms_with_vlm_fallback(
        self,
        image_path: Union[str, Path],
    ) -> Tuple[List[RoomCandidate], List[TextDetection]]:
        """
        Extract rooms using two-pass strategy: PaddleOCR + VLM fallback.
        
        Pass 1: Run PaddleOCR on all regions, returns baseline room candidates
        Pass 2: For low-confidence results (< threshold), use VLM as fallback
        Merge: Combine results, preferring higher confidence results
        
        Args:
            image_path: Path to floor plan image
        
        Returns:
            Tuple of (room_candidates, raw_text_detections)
            - room_candidates: List of RoomCandidate with merged results
            - raw_text_detections: List of all raw TextDetection objects from OCR
        
        Raises:
            OCRError: If OCR extraction fails
        """
        image_path = Path(image_path)
        
        # Pass 1: PaddleOCR extraction
        logger.info(f"Pass 1 (PaddleOCR): Extracting rooms from {image_path.name}")
        try:
            pass1_rooms, raw_detections = self.ocr_extractor.extract_and_find_rooms(image_path)
        except Exception as e:
            logger.error(f"Pass 1 (PaddleOCR) failed: {e}")
            raise
        
        logger.info(f"Pass 1 complete: extracted {len(pass1_rooms)} room candidates")
        
        # If no VLM backend or all results are high-confidence, return Pass 1 results
        low_confidence_rooms = [
            r for r in pass1_rooms
            if r.confidence < self.confidence_threshold
        ]
        
        if not low_confidence_rooms or self.vlm_backend is None:
            logger.info(
                f"Pass 2 skipped: {len(low_confidence_rooms)} low-confidence rooms, "
                f"VLM backend available: {self.vlm_backend is not None}"
            )
            return pass1_rooms, raw_detections
        
        # Pass 2: VLM fallback for low-confidence rooms
        logger.info(
            f"Pass 2 (VLM Fallback): Processing {len(low_confidence_rooms)} "
            f"low-confidence rooms (confidence < {self.confidence_threshold})"
        )
        try:
            pass2_rooms = self._extract_rooms_via_vlm(image_path)
            if not pass2_rooms:
                logger.warning("Pass 2 (VLM) returned no results, using Pass 1 only")
                return pass1_rooms, raw_detections
        except Exception as e:
            logger.warning(f"Pass 2 (VLM) failed, using Pass 1 results: {e}")
            return pass1_rooms, raw_detections
        
        # Merge results: prefer higher confidence
        merged_rooms = self._merge_room_candidates(pass1_rooms, pass2_rooms)
        logger.info(f"Merged {len(pass1_rooms)} OCR + {len(pass2_rooms)} VLM → {len(merged_rooms)} final candidates")
        
        return merged_rooms, raw_detections
    
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
            # Call VLM backend's detect_rooms method
            vlm_detections = self.vlm_backend.detect_rooms(image_path)
            
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
    
    def _merge_room_candidates(
        self,
        pass1_rooms: List[RoomCandidate],
        pass2_rooms: List[RoomCandidate],
        bbox_distance_threshold: float = 100.0,
    ) -> List[RoomCandidate]:
        """
        Merge room candidates from Pass 1 (OCR) and Pass 2 (VLM).
        
        Strategy:
        1. For each low-confidence Pass 1 room, find the closest Pass 2 room (by bbox centroid)
        2. If distance < threshold, replace Pass 1 with Pass 2 (if higher confidence)
        3. Otherwise, keep Pass 1 result
        4. Add unmatched Pass 2 rooms as new candidates
        
        Args:
            pass1_rooms: Room candidates from PaddleOCR
            pass2_rooms: Room candidates from VLM
            bbox_distance_threshold: Max pixel distance to match rooms (default: 100.0)
        
        Returns:
            Merged list of RoomCandidate objects
        """
        def centroid(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
            """Return (cx, cy) centroid of bbox (x, y, w, h)."""
            x, y, w, h = bbox
            return x + w / 2, y + h / 2
        
        def bbox_distance(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
            """Euclidean distance between bbox centroids."""
            cx1, cy1 = centroid(bbox1)
            cx2, cy2 = centroid(bbox2)
            return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
        
        merged = []
        matched_pass2_indices = set()
        
        # Process Pass 1 rooms
        for p1_room in pass1_rooms:
            # If high confidence, keep as-is
            if p1_room.confidence >= self.confidence_threshold:
                merged.append(p1_room)
                continue
            
            # Low confidence: find best matching Pass 2 room
            best_match = None
            best_distance = bbox_distance_threshold
            best_idx = -1
            
            for idx, p2_room in enumerate(pass2_rooms):
                dist = bbox_distance(p1_room.bbox, p2_room.bbox)
                if dist < best_distance:
                    best_distance = dist
                    best_match = p2_room
                    best_idx = idx
            
            # If good match found, use Pass 2 result
            if best_match is not None:
                logger.debug(
                    f"Matched low-confidence OCR ({p1_room.room_name}, conf={p1_room.confidence:.2f}) "
                    f"→ VLM ({best_match.room_name}, conf={best_match.confidence:.2f}), "
                    f"distance={best_distance:.1f}px"
                )
                merged.append(best_match)
                matched_pass2_indices.add(best_idx)
            else:
                # No good match, keep Pass 1 result
                logger.debug(
                    f"No VLM match for low-confidence OCR ({p1_room.room_name}, "
                    f"conf={p1_room.confidence:.2f}), using OCR result"
                )
                merged.append(p1_room)
        
        # Add unmatched Pass 2 rooms as new candidates
        for idx, p2_room in enumerate(pass2_rooms):
            if idx not in matched_pass2_indices:
                logger.debug(f"Adding unmatched VLM result: {p2_room.room_name} (conf={p2_room.confidence:.2f})")
                merged.append(p2_room)
        
        return merged
