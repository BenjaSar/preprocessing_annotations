"""
Semantic Reconciliation Module — Merge OCR text with VLM room detections.

Solution C (Hybrid) Architecture:
  1. PaddleOCR extracts raw text with bounding boxes
  2. Qwen2.5-VL detects room polygons with semantic classification
  3. SemanticReconciler matches OCR text to VLM room polygons by spatial containment
  4. Result: Enriched annotations with room name, number, type, and full text coverage

Core Logic:
  - For each VLM-detected room polygon, find all OCR-detected text centroids inside it
  - Extract room_name and room_number from matched OCR text using regex patterns
  - Use VLM-detected room type as authoritative classification
  - Combine confidence scores (min of OCR and VLM confidence)
  - Maintain provenance (track which model produced which annotation)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

try:
    from shapely.geometry import Point, Polygon
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

logger = logging.getLogger(__name__)


@dataclass
class OCRText:
    """Single OCR-detected text item."""
    text: str
    bbox: Tuple[float, float, float, float]  # [x1, y1, x2, y2]
    confidence: float
    
    def centroid(self) -> Tuple[float, float]:
        """Return (x, y) centroid of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class VLMRoom:
    """Single VLM-detected room with polygon and classification."""
    room_id: str  # Unique identifier
    polygon: List[Tuple[float, float]]  # List of (x, y) vertices defining the room
    room_type: str  # Classification: bedroom, bathroom, kitchen, etc.
    vlm_name: Optional[str] = None  # VLM's suggested room name (may be generic)
    vlm_confidence: float = 1.0  # VLM's confidence in this detection
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional VLM-provided data


@dataclass
class ReconciledRoom:
    """Final reconciled room annotation combining OCR and VLM outputs."""
    room_id: str
    polygon: List[Tuple[float, float]]
    bbox: Tuple[float, float, float, float]  # Derived from polygon bounds
    room_name: str  # Extracted from OCR text, fallback to VLM name
    room_number: Optional[str]  # Extracted from OCR text if present
    room_type: str  # From VLM classification
    confidence: float  # Min of VLM and OCR confidences
    text_labels: List[str]  # All OCR text found in this room
    ocr_confidences: List[float]  # Confidence for each OCR text label
    provenance: Dict[str, str] = field(default_factory=dict)  # Track sources


class SemanticReconciler:
    """
    Match OCR text to VLM room detections via spatial containment.
    
    Usage:
        reconciler = SemanticReconciler(config)
        ocr_results = [...]  # From MEPTextExtractor
        vlm_results = [...]  # From VLMAnnotator
        enriched = reconciler.reconcile(ocr_results, vlm_results)
    """
    
    def __init__(self, config=None):
        """Initialize reconciler with optional configuration.
        
        Args:
            config: Optional config object with room_name_pattern and room_number_pattern
        """
        self.config = config
        
        # Compile patterns for name/number extraction
        if config and hasattr(config, 'room_number_pattern'):
            self.room_number_pattern = re.compile(config.room_number_pattern, re.IGNORECASE)
        else:
            # Default pattern: 3-digit room number optionally with letter suffix
            self.room_number_pattern = re.compile(r'^(\d{3}[A-Z]?|\d{2,3})$', re.IGNORECASE)
        
        if config and hasattr(config, 'room_name_patterns'):
            self.room_name_patterns = [
                re.compile(pattern, re.IGNORECASE)
                for pattern in config.room_name_patterns
            ]
        else:
            # Minimal default patterns
            self.room_name_patterns = [
                re.compile(r'^(BEDROOM|BATHROOM|KITCHEN|LIVING|OFFICE|STORAGE).*$', re.IGNORECASE)
            ]
        
        if not HAS_SHAPELY:
            logger.warning(
                "Shapely not installed. SemanticReconciler requires spatial operations. "
                "Install via: pip install shapely"
            )
    
    def reconcile(
        self,
        ocr_results: List[Dict[str, Any]],
        vlm_results: List[Dict[str, Any]]
    ) -> List[ReconciledRoom]:
        """
        Match OCR text to VLM room polygons and produce enriched annotations.
        
        Args:
            ocr_results: List of OCR detections from MEPTextExtractor.
                Expected format: [{"text": str, "bbox": [x1,y1,x2,y2], "confidence": float}, ...]
                
            vlm_results: List of VLM room detections.
                Expected format: [{"polygon": [[x,y], [x,y], ...], "room_type": str, ...}, ...]
        
        Returns:
            List of ReconciledRoom objects with matched text and enriched metadata.
        """
        if not HAS_SHAPELY:
            logger.error("Shapely is required for SemanticReconciler. Returning empty results.")
            return []
        
        # Convert OCR results to OCRText objects
        ocr_texts = []
        for result in ocr_results:
            try:
                ocr_texts.append(OCRText(
                    text=result.get('text', ''),
                    bbox=tuple(result.get('bbox', [0, 0, 0, 0])),
                    confidence=float(result.get('confidence', 0.5))
                ))
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid OCR result format: {result}. Error: {e}")
                continue
        
        # Convert VLM results to VLMRoom objects
        vlm_rooms = []
        for idx, result in enumerate(vlm_results):
            try:
                room_id = result.get('room_id', f"room_{idx}")
                polygon = result.get('polygon', [])
                if not polygon:
                    logger.warning(f"VLM result {room_id} has no polygon. Skipping.")
                    continue
                
                vlm_rooms.append(VLMRoom(
                    room_id=room_id,
                    polygon=[(float(p[0]), float(p[1])) for p in polygon],
                    room_type=result.get('room_type', 'other'),
                    vlm_name=result.get('room_name', None),
                    vlm_confidence=float(result.get('confidence', 1.0)),
                    metadata=result.get('metadata', {})
                ))
            except (ValueError, TypeError, IndexError) as e:
                logger.warning(f"Invalid VLM result format: {result}. Error: {e}")
                continue
        
        logger.info(f"Reconciling {len(ocr_texts)} OCR texts with {len(vlm_rooms)} VLM rooms")
        
        # Match and enrich
        reconciled = []
        for room in vlm_rooms:
            reconciled.append(self._match_room_to_ocr(room, ocr_texts))
        
        return reconciled
    
    def _match_room_to_ocr(self, room: VLMRoom, all_ocr_texts: List[OCRText]) -> ReconciledRoom:
        """
        Match all OCR texts that fall inside a room polygon.
        
        Args:
            room: VLMRoom polygon and metadata
            all_ocr_texts: All OCR-detected texts
        
        Returns:
            ReconciledRoom with matched OCR text and extracted room name/number
        """
        # Create Shapely polygon for containment checks
        try:
            polygon = Polygon(room.polygon)
        except Exception as e:
            logger.warning(f"Failed to create polygon for room {room.room_id}: {e}")
            # Fallback: return room with no matched text
            return ReconciledRoom(
                room_id=room.room_id,
                polygon=room.polygon,
                bbox=self._polygon_bounds(room.polygon),
                room_name=room.vlm_name or "Unknown",
                room_number=None,
                room_type=room.room_type,
                confidence=room.vlm_confidence,
                text_labels=[],
                ocr_confidences=[],
                provenance={"vlm": "qwen2.5-vl-7b"}
            )
        
        # Find all OCR texts with centroids inside the polygon
        matched_texts = []
        matched_confidences = []
        for ocr_text in all_ocr_texts:
            cx, cy = ocr_text.centroid()
            if polygon.contains(Point(cx, cy)):
                matched_texts.append(ocr_text.text)
                matched_confidences.append(ocr_text.confidence)
        
        logger.debug(
            f"Room {room.room_id}: Found {len(matched_texts)} OCR texts inside polygon"
        )
        
        # Extract room name and number from matched texts
        room_name = self._extract_room_name(matched_texts, room.vlm_name)
        room_number = self._extract_room_number(matched_texts)
        
        # Combine confidence scores (take minimum to be conservative)
        ocr_avg_conf = sum(matched_confidences) / len(matched_confidences) if matched_confidences else 0.5
        combined_confidence = min(room.vlm_confidence, ocr_avg_conf)
        
        return ReconciledRoom(
            room_id=room.room_id,
            polygon=room.polygon,
            bbox=self._polygon_bounds(room.polygon),
            room_name=room_name,
            room_number=room_number,
            room_type=room.room_type,
            confidence=combined_confidence,
            text_labels=matched_texts,
            ocr_confidences=matched_confidences,
            provenance={
                "ocr": "paddleocr",
                "vlm": "qwen2.5-vl-7b",
                "reconciler": "spatial_containment"
            }
        )
    
    def _extract_room_name(self, texts: List[str], vlm_fallback: Optional[str]) -> str:
        """
        Extract room name from OCR texts using regex patterns.
        
        Strategy:
        1. Look for texts matching room_name_patterns
        2. Expand known abbreviations
        3. Fall back to VLM-provided name if no match
        4. Default to 'Room' if nothing found
        
        Args:
            texts: List of OCR text strings
            vlm_fallback: VLM-provided name for fallback
        
        Returns:
            Best-guess room name
        """
        for text in texts:
            for pattern in self.room_name_patterns:
                if pattern.match(text.strip()):
                    return text.strip()
        
        # Fallback to VLM name if available
        if vlm_fallback:
            return vlm_fallback
        
        # Last resort: use longest text as name
        if texts:
            longest = max(texts, key=len)
            if len(longest) > 2:
                return longest
        
        return "Room"
    
    def _extract_room_number(self, texts: List[str]) -> Optional[str]:
        """
        Extract room number from OCR texts using regex pattern.
        
        Looks for 2-3 digit numbers or 3 digits + letter (e.g., 301, 203A).
        
        Args:
            texts: List of OCR text strings
        
        Returns:
            Room number string or None if not found
        """
        for text in texts:
            stripped = text.strip()
            if self.room_number_pattern.match(stripped):
                # Extract just the number part if there's extra text
                match = re.search(r'\d{2,3}[A-Z]?', stripped)
                if match:
                    return match.group()
        
        return None
    
    @staticmethod
    def _polygon_bounds(polygon: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
        """
        Calculate bounding box from polygon vertices.
        
        Args:
            polygon: List of (x, y) vertices
        
        Returns:
            (x1, y1, x2, y2) bounding box
        """
        if not polygon:
            return (0, 0, 0, 0)
        
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        return (min(xs), min(ys), max(xs), max(ys))
    
    def reconcile_with_config(self, ocr_results, vlm_results, config) -> List[Dict[str, Any]]:
        """
        High-level reconciliation that returns plain dictionaries compatible with pipeline.
        
        Args:
            ocr_results: OCR detections
            vlm_results: VLM room detections
            config: Pipeline config (used to extract patterns)
        
        Returns:
            List of reconciled room dicts
        """
        # Refresh patterns from config
        self.config = config
        if hasattr(config, 'ocr'):
            ocr_config = config.ocr
            if hasattr(ocr_config, 'room_number_pattern'):
                self.room_number_pattern = re.compile(
                    ocr_config.room_number_pattern,
                    re.IGNORECASE
                )
            if hasattr(ocr_config, 'room_name_patterns'):
                self.room_name_patterns = [
                    re.compile(pattern, re.IGNORECASE)
                    for pattern in ocr_config.room_name_patterns
                ]
        
        # Perform reconciliation
        reconciled_rooms = self.reconcile(ocr_results, vlm_results)
        
        # Convert to dict format for pipeline compatibility
        return [
            {
                "room_id": room.room_id,
                "room_name": room.room_name,
                "room_number": room.room_number,
                "room_type": room.room_type,
                "polygon": room.polygon,
                "bbox": room.bbox,
                "confidence": room.confidence,
                "text_labels": room.text_labels,
                "provenance": room.provenance,
                "source": "reconciled"  # Mark as coming from semantic reconciliation
            }
            for room in reconciled_rooms
        ]
