"""
SFT-Ready Annotation Schema — Mandatory Output Format.

Implements the standardized JSON schema for floorplan annotations required
for Vision-Language Model fine-tuning (SFT).

    Schema structure:
    {
        "image_file": "...",
        "image_size": {"width": int, "height": int},
        "roomsRecognized": [
            {
                "id": int,
                "type": "<MANDATORY_CLASS or MANDATORY_CLASS w/ suffix>",
                "name": "<original_label>",
                "nameUnique": "<type - zone - seq>",
                "coordinates": {"bbox": [x1, y1, x2, y2], "polygon": [...]},
                "confidence": float,
                "coverage": {
                    "spatial_fraction": float,
                    "text_tokens_matched": int,
                    "text_tokens_total": int,
                    "source": "ocr|vlm|ocr+vlm|synthetic"
                },
                "attributes": {
                    "has_windows": bool,
                    "window_count": int,
                    "has_skylights": bool,
                    "has_openings": bool,
                    "window_instances": [
                        {"bbox": [...], "confidence": float, "detection_tier": str}
                    ]
                },
                "extended_type": "...",  # for backward compatibility
                "provenance": {...}  # detection/classification/ocr sources
            }
        ]
    }

    Design: room type (semantic) and window presence (visual attribute) are
    orthogonal axes.  "type" holds the base mandatory class or its suffixed
    form; "attributes" holds structured detection data.  The two are produced
    independently and composed at export time via add_window_suffix().
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class RoomCoordinates:
    """Spatial coordinates of a room."""
    bbox: List[float]  # [x1, y1, x2, y2] in pixels
    polygon: Optional[List[List[float]]] = None  # [[x, y], ...] optional


@dataclass
class RoomCoverage:
    """Coverage metrics for a room."""
    spatial_fraction: float = 0.0  # room_area / total_floorplan_area
    text_tokens_matched: int = 0  # OCR tokens assigned to this room
    text_tokens_total: int = 0  # OCR tokens inside room bbox
    source: str = "unknown"  # "ocr_only", "vlm_only", "ocr+vlm", "synthetic"


@dataclass
class ConfidenceDetail:
    """Breakdown of confidence score into component factors."""
    detection: float = 0.5  # Is there a room here? (0-1)
    classification: float = 0.5  # Is the type correct? (0-1)
    ocr: float = 0.5  # Was the text recognized? (0-1)
    # T-1 (2026-09-03): additive corroboration bonus from object-detection
    # evidence (has_door/has_windows/etc, see pipeline.py's
    # _apply_object_attributes). 0.0 = no evidence pass has run yet, or
    # none found nearby -- identical to pre-T-1 behavior either way.
    corroboration: float = 0.0


@dataclass
class RoomProvenanceRoomDetection:
    """Detection source and metadata."""
    method: str  # "ocr", "vlm", "synthetic"
    model: Optional[str] = None  # "PaddleOCR", "Claude", "Qwen", etc.
    score: float = 0.0  # Raw score from detection stage


@dataclass
class RoomProvenance:
    """Provenance: how this room was detected and classified."""
    detection: RoomProvenanceRoomDetection
    classification: Optional[str] = None  # "exact_match", "substring", "fuzzy", "vlm"
    ocr_backend: Optional[str] = None  # "PaddleOCR", "EasyOCR", etc.


@dataclass
class WindowInstance:
    """A single window detection spatially associated with a room."""
    bbox: List[float]           # [x1, y1, x2, y2] in image pixels
    confidence: float = 0.0
    detection_tier: str = "none"  # "pdf_layers" | "object_detector" | "vlm_prompt"


@dataclass
class RoomAttributes:
    """
    Visual attributes detected on a room — independent of room type.

    Populated by the window detector after spatial association with room
    regions.  Default is "no attributes detected", which is the correct
    representation when detection has not run or found nothing.
    """
    has_windows: bool = False
    window_count: int = 0
    has_skylights: bool = False
    has_openings: bool = False
    window_instances: List[WindowInstance] = field(default_factory=list)


@dataclass
class SFTRoom:
    """Mandatory SFT-ready room annotation."""
    id: int  # Sequential room ID (1-indexed)
    type: str  # Mandatory class or suffixed form (e.g., "CONFERENCE w/ windows")
    name: str  # Original label (immutable)
    name_unique: str  # Disambiguated name (e.g., "CORRIDOR - Art - 01")
    coordinates: RoomCoordinates
    confidence: float  # Combined confidence (0-1)
    coverage: RoomCoverage
    confidence_detail: ConfidenceDetail
    provenance: RoomProvenance
    attributes: RoomAttributes = field(default_factory=RoomAttributes)
    extended_type: Optional[str] = None  # For backward compatibility (e.g., "conference_room")
    name_expanded: Optional[str] = None  # Abbreviation expansion (e.g., "BR" → "BEDROOM")


class NameUniquifier:
    """
    Generate unique disambiguated names for duplicate room types.

    Strategy: "{Type} - {Zone} - {SeqNum:02d}"
    where Zone is extracted from room number or set to "General".
    """

    def __init__(self):
        self.type_zone_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def get_unique_name(self, base_type: str, room_number: Optional[str]) -> str:
        """
        Generate a unique name for a room.

        Args:
            base_type: Mandatory class name (e.g., "CORRIDOR").
            room_number: Room identifier, may contain zone prefix (e.g., "A-101").

        Returns:
            Unique name string (e.g., "CORRIDOR - A - 01").
        """
        # Extract zone from room number (e.g., "A-101" → "A", "3F-102" → "3F").
        # Contract declares Optional[str], but VLM-extracted room_number can
        # arrive as int (observed: mixed "room_id": "1" vs "room_id": 2 in
        # raw VLM JSON) -- enforce the contract here, the one place that
        # calls .split() on it, instead of trusting every caller.
        if room_number:
            if not isinstance(room_number, str):
                logger.debug(
                    f"get_unique_name: room_number {room_number!r} is "
                    f"{type(room_number).__name__}, not str -- coercing"
                )
                room_number = str(room_number)
            parts = room_number.split("-")
            zone = parts[0].upper() if parts[0] else "General"
        else:
            zone = "General"

        # Increment counter for this type-zone pair
        self.type_zone_counts[base_type][zone] += 1
        seq_num = self.type_zone_counts[base_type][zone]

        return f"{base_type} - {zone} - {seq_num:02d}"


class ConfidenceComputer:
    """
    Unified confidence computation: composite score from three factors,
    plus an optional additive corroboration bonus (T-1, G-1).

    Formula:
        confidence = w_det * C_detection + w_cls * C_classification
                     + w_ocr * C_ocr + w_corrob * C_corroboration

    w_corrob is NOT drawn from the same 1.0 pie as the other three --
    see ConfidenceWeightsConfig's docstring (config/config.py) for why
    this must be additive, not a reallocated weighted average. Every
    weight defaults to its pre-T-1 literal value (0.3/0.4/0.3/0.0), so
    calling this with no weight overrides reproduces the original
    3-factor formula exactly.
    """

    @staticmethod
    def compute(
        detection_score: float = 0.9,
        classification_score: float = 0.9,
        ocr_score: float = 0.7,
        corroboration_score: float = 0.0,
        w_detection: float = 0.3,
        w_classification: float = 0.4,
        w_ocr: float = 0.3,
        w_corroboration: float = 0.0,
    ) -> tuple[float, ConfidenceDetail]:
        """
        Compute combined confidence from three factors plus an optional
        additive corroboration bonus.

        Args:
            detection_score: Probability that a room exists here (0-1).
                Default 0.9 for VLM (high confidence), 0.95 for OCR bbox.
            classification_score: Probability that the type is correct (0-1).
                Depends on match quality: exact=1.0, substring=0.8, fuzzy=0.7, fallback=0.3.
            ocr_score: Text recognition confidence (0-1).
                From PaddleOCR confidence, or 1.0 if VLM-generated.
            corroboration_score: Object-detection evidence strength (0-1),
                e.g. the evidence_score fraction from
                pipeline.py's _apply_object_attributes. 0.0 (default) =
                no evidence pass has run, or none found nearby.
            w_detection, w_classification, w_ocr: Weights for the base
                3-factor average. Default to the original literals.
            w_corroboration: Weight for the additive corroboration bonus.
                Default 0.0 -- inert until a caller explicitly enables it
                (ConfidenceWeightsConfig.corroboration).

        Returns:
            Tuple of (combined_confidence, detail_breakdown).
        """
        # Clamp inputs to [0, 1]
        det = max(0.0, min(1.0, detection_score))
        cls = max(0.0, min(1.0, classification_score))
        ocr = max(0.0, min(1.0, ocr_score))
        corrob = max(0.0, min(1.0, corroboration_score))

        base = w_detection * det + w_classification * cls + w_ocr * ocr
        # Additive bonus, not folded into the weighted average above --
        # corrob=0 (no evidence) leaves `base` untouched, satisfying the
        # "absence of evidence never penalizes" requirement (T-1 risk table).
        combined = max(0.0, min(1.0, base + w_corroboration * corrob))

        detail = ConfidenceDetail(
            detection=round(det, 4),
            classification=round(cls, 4),
            ocr=round(ocr, 4),
            corroboration=round(corrob, 4),
        )

        return round(combined, 4), detail

    @staticmethod
    def apply_corroboration(
        room_dict: Dict[str, Any],
        corroboration_score: float,
        w_detection: float = 0.3,
        w_classification: float = 0.4,
        w_ocr: float = 0.3,
        w_corroboration: float = 0.0,
    ) -> Dict[str, Any]:
        """T-1 (G-1): recompute an already-serialized room dict's
        `confidence`/`confidence_detail` with a corroboration bonus.

        Object-detection evidence (has_door/has_windows/etc) is only
        known AFTER `build_room()` has already produced and serialized
        the room (pipeline.py's per-image flow runs room-building before
        object-detection mapping in the same pass) -- so this re-derives
        the combined score from the room's own already-clamped
        `confidence_detail` components rather than requiring the raw
        pre-clamp inputs to be threaded through a second call path.

        Mutates and returns `room_dict` in place, matching
        `_apply_object_attributes`'s own mutate-and-return style. With
        `w_corroboration=0.0` (default) this is a no-op: recomputing
        from the same three stored components with the same three base
        weights reproduces the existing `confidence` value exactly.
        """
        detail = room_dict.get("confidence_detail", {})
        combined, new_detail = ConfidenceComputer.compute(
            detection_score=detail.get("detection", 0.9),
            classification_score=detail.get("classification", 0.9),
            ocr_score=detail.get("ocr", 0.7),
            corroboration_score=corroboration_score,
            w_detection=w_detection,
            w_classification=w_classification,
            w_ocr=w_ocr,
            w_corroboration=w_corroboration,
        )
        room_dict["confidence"] = combined
        room_dict["confidence_detail"] = {
            "detection": new_detail.detection,
            "classification": new_detail.classification,
            "ocr": new_detail.ocr,
            "corroboration": new_detail.corroboration,
        }
        return room_dict

    @staticmethod
    def classification_score_for_match(match_type: str) -> float:
        """
        Get classification confidence based on match quality.

        Args:
            match_type: One of "exact", "substring", "fuzzy", "fallback".

        Returns:
            Confidence score (0-1).
        """
        scores = {
            "exact": 1.0,  # Exact surface-form match
            "substring": 0.8,  # Substring match within variant
            "fuzzy": 0.7,  # Fuzzy matching with high threshold
            "fallback": 0.3,  # Last-resort mapping
            "vlm": 0.9,  # VLM classification (inherent uncertainty)
        }
        return scores.get(match_type, 0.3)


class SFTAnnotationBuilder:
    """
    Builder for constructing SFT-ready annotations from raw data.
    """

    def __init__(self):
        self.uniquifier = NameUniquifier()

    def build_room(
        self,
        room_id: int,
        mandatory_type: str,
        original_name: str,
        room_number: Optional[str],
        bbox: List[float],  # [x1, y1, x2, y2]
        polygon: Optional[List[List[float]]] = None,
        detection_score: float = 0.9,
        classification_match_type: str = "exact",
        ocr_confidence: float = 0.7,
        spatial_fraction: float = 0.0,
        text_tokens_matched: int = 0,
        text_tokens_total: int = 0,
        source: str = "unknown",
        extended_type: Optional[str] = None,
        name_expanded: Optional[str] = None,
        detection_method: str = "unknown",
        detection_model: Optional[str] = None,
        ocr_backend: Optional[str] = None,
        window_mapping: Optional[Any] = None,  # RoomWindowMapping from window_detector
    ) -> SFTRoom:
        """
        Build a single SFT-ready room annotation.

        Args:
            room_id: Sequential ID.
            mandatory_type: Standardized class (must be in VALID_TYPES).
            original_name: Raw label from OCR or VLM (immutable).
            room_number: Room identifier for zone extraction.
            bbox: Bounding box [x1, y1, x2, y2] in pixels.
            polygon: Optional room polygon (room boundary from SAM).
            detection_score: Room existence confidence (0-1).
            classification_match_type: "exact", "substring", "fuzzy", "fallback", "vlm".
            ocr_confidence: Text recognition confidence (0-1).
            spatial_fraction: Room area / total floorplan area.
            text_tokens_matched: OCR tokens assigned to this room.
            text_tokens_total: OCR tokens inside room bbox.
            source: "ocr_only", "vlm_only", "ocr+vlm", "synthetic".
            extended_type: Fine-grained type for backward compatibility.
            name_expanded: Abbreviation expansion.
            detection_method: "ocr", "vlm", "synthetic".
            detection_model: Model name ("PaddleOCR", "Claude", "Qwen", etc.).
            ocr_backend: OCR engine name.
            window_mapping: Optional RoomWindowMapping from window_detector.
                            If provided, populates attributes and composes the
                            final type string via add_window_suffix().

        Returns:
            SFTRoom dataclass instance.
        """
        # Generate unique name
        name_unique = self.uniquifier.get_unique_name(mandatory_type, room_number)

        # Compute unified confidence
        cls_score = ConfidenceComputer.classification_score_for_match(
            classification_match_type
        )
        combined_conf, conf_detail = ConfidenceComputer.compute(
            detection_score=detection_score,
            classification_score=cls_score,
            ocr_score=ocr_confidence,
        )

        # Build coordinates
        coordinates = RoomCoordinates(bbox=bbox, polygon=polygon)

        # Build coverage
        coverage = RoomCoverage(
            spatial_fraction=spatial_fraction,
            text_tokens_matched=text_tokens_matched,
            text_tokens_total=text_tokens_total,
            source=source,
        )

        # Build provenance
        detection_prov = RoomProvenanceRoomDetection(
            method=detection_method,
            model=detection_model,
            score=detection_score,
        )
        provenance = RoomProvenance(
            detection=detection_prov,
            classification=classification_match_type,
            ocr_backend=ocr_backend,
        )

        # Build attributes from window mapping (orthogonal to room type)
        attributes = RoomAttributes()
        if window_mapping is not None:
            win_instances = [
                WindowInstance(
                    bbox=list(w.bbox),
                    confidence=w.confidence,
                    detection_tier=w.source_tier.value
                    if hasattr(w.source_tier, "value")
                    else str(w.source_tier),
                )
                for w in (window_mapping.intersecting_windows or [])
            ]
            attributes = RoomAttributes(
                has_windows=window_mapping.has_windows,
                window_count=window_mapping.window_count,
                has_skylights=window_mapping.has_skylights,
                has_openings=window_mapping.has_openings,
                window_instances=win_instances,
            )

        # Compose final type string: base type + window suffix (if any)
        # Import here to avoid circular import (taxonomy ← annotation_schema)
        try:
            from .taxonomy import add_window_suffix
        except ImportError:
            from taxonomy import add_window_suffix

        final_type = add_window_suffix(
            mandatory_type,
            has_windows=attributes.has_windows,
            has_skylights=attributes.has_skylights,
            has_openings=attributes.has_openings,
        )

        return SFTRoom(
            id=room_id,
            type=final_type,
            name=original_name,
            name_unique=name_unique,
            coordinates=coordinates,
            confidence=combined_conf,
            coverage=coverage,
            confidence_detail=conf_detail,
            provenance=provenance,
            attributes=attributes,
            extended_type=extended_type,
            name_expanded=name_expanded,
        )

    def to_dict(self, room: SFTRoom) -> Dict[str, Any]:
        """Convert SFTRoom to dictionary for JSON serialization."""
        result = {
            "id": room.id,
            "type": room.type,
            "name": room.name,
            "nameUnique": room.name_unique,
            "coordinates": {
                "bbox": room.coordinates.bbox,
            },
            "confidence": room.confidence,
            "coverage": {
                "spatial_fraction": room.coverage.spatial_fraction,
                "text_tokens_matched": room.coverage.text_tokens_matched,
                "text_tokens_total": room.coverage.text_tokens_total,
                "source": room.coverage.source,
            },
            # Structured visual attributes — always present, defaults to all-False
            # when window detection has not run or found nothing.
            "attributes": {
                "has_windows": room.attributes.has_windows,
                "window_count": room.attributes.window_count,
                "has_skylights": room.attributes.has_skylights,
                "has_openings": room.attributes.has_openings,
                "window_instances": [
                    {
                        "bbox": wi.bbox,
                        "confidence": wi.confidence,
                        "detection_tier": wi.detection_tier,
                    }
                    for wi in room.attributes.window_instances
                ],
            },
        }

        # Add optional fields if present
        if room.extended_type:
            result["extended_type"] = room.extended_type
        if room.name_expanded:
            result["name_expanded"] = room.name_expanded
        if room.coordinates.polygon:
            result["coordinates"]["polygon"] = room.coordinates.polygon

        # Add provenance
        result["provenance"] = {
            "detection": {
                "method": room.provenance.detection.method,
                "model": room.provenance.detection.model,
                "score": room.provenance.detection.score,
            },
            "classification": room.provenance.classification,
            "ocr_backend": room.provenance.ocr_backend,
        }

        # Add confidence detail (for debugging)
        result["confidence_detail"] = {
            "detection": room.confidence_detail.detection,
            "classification": room.confidence_detail.classification,
            "ocr": room.confidence_detail.ocr,
            "corroboration": room.confidence_detail.corroboration,
        }

        return result


def build_annotation_json(
    image_file: str,
    image_size: Dict[str, int],
    rooms: List[SFTRoom],
) -> Dict[str, Any]:
    """
    Build the complete SFT annotation JSON structure.

    Args:
        image_file: Filename of the annotated image.
        image_size: {"width": int, "height": int}.
        rooms: List of SFTRoom objects.

    Returns:
        Annotation JSON dict.
    """
    builder = SFTAnnotationBuilder()

    return {
        "image_file": image_file,
        "image_size": image_size,
        "roomsRecognized": [builder.to_dict(room) for room in rooms],
    }
