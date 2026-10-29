"""AV-3: nearest-room-with-cutoff attachment policy.

Status: Prototype First / Insufficient Evidence -- cutoff_px has no
measured optimum on this corpus. Not the pipeline default; must be
explicitly selected via AttachmentConfig.policy. Do not raise its
priority without the validation step described in the technique's own
evaluation (bit-identical default check + adjudicated-sample precision
comparison against IntersectionAttachmentPolicy).

Rationale: a bare nearest-room fallback with no cutoff was already
tried and rejected for this project (memory
nearest-room-attachment-spike-2026-09-03: plain nearest-room causes
"catch-all absorption" by whichever room is most central on the page).
This policy only differs from that rejected variant by adding a hard
distance cutoff, and only attaches a detection to its nearest room when
no room already intersects it geometrically -- intersection matches
still win first, so a room that already receives a detection under the
default policy behaves identically here.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from ...config import AttachmentConfig
from ..object_types import DEFAULT_OBJECT_TYPE_REGISTRY, ObjectTypeRegistry
from ..window_detector import RoomWindowMapping, WindowDetection, map_detections_to_rooms


def _room_center(room: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    bbox = room.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _detection_center(detection: WindowDetection) -> Tuple[float, float]:
    x1, y1, x2, y2 = detection.bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _nearest_room_id(
    detection_center: Tuple[float, float], rooms: List[Dict[str, Any]],
) -> Optional[Tuple[Any, float]]:
    """Returns (room_id, distance_px) of the nearest room by centroid
    distance, or None if no room in `rooms` has a usable bbox."""
    best_id = None
    best_dist = float("inf")
    for idx, room in enumerate(rooms):
        center = _room_center(room)
        if center is None:
            continue
        dist = math.hypot(
            center[0] - detection_center[0], center[1] - detection_center[1]
        )
        if dist < best_dist:
            best_dist = dist
            best_id = room.get("id", f"room_{idx}")
    return (best_id, best_dist) if best_id is not None else None


def _apply_detection(
    mapping: RoomWindowMapping,
    detection: WindowDetection,
    registry: ObjectTypeRegistry,
    strict_type_dispatch: bool,
) -> None:
    """Mutates mapping in place -- same flag-setting rule
    map_detections_to_rooms applies to an intersecting detection."""
    window_type = (detection.metadata or {}).get("type", "window")
    flag = registry.flag_for(window_type, strict=strict_type_dispatch)
    if flag is not None:
        setattr(mapping, flag, True)
    else:
        mapping.unknown_type_count += 1
    mapping.window_count += 1
    mapping.intersecting_windows.append(detection)


class NearestWithCutoffAttachmentPolicy:
    """Falls back to nearest-room-by-centroid, within cutoff_px, only
    for detections that no room's geometry already intersected."""

    def __init__(self, config: AttachmentConfig, type_registry: Optional[ObjectTypeRegistry] = None):
        if config.nearest_cutoff_px is None or config.nearest_cutoff_px <= 0:
            raise ValueError(
                "AttachmentConfig.nearest_cutoff_px must be a positive "
                "number when policy='nearest_with_cutoff' -- no default "
                "cutoff is assumed, it must be set explicitly."
            )
        self._cutoff_px = config.nearest_cutoff_px
        self._strict_type_dispatch = config.strict_type_dispatch
        self._registry = type_registry or DEFAULT_OBJECT_TYPE_REGISTRY

    def attach(
        self,
        detections: List[WindowDetection],
        rooms: List[Dict[str, Any]],
    ) -> List[RoomWindowMapping]:
        mappings = map_detections_to_rooms(
            detections, rooms, type_registry=self._registry,
            strict_type_dispatch=self._strict_type_dispatch,
        )
        attached_ids = {id(w) for m in mappings for w in m.intersecting_windows}
        unattached = [d for d in detections if id(d) not in attached_ids]
        if not unattached:
            return mappings

        mapping_by_id = {m.room_id: m for m in mappings}
        for detection in unattached:
            result = _nearest_room_id(_detection_center(detection), rooms)
            if result is None:
                continue
            room_id, dist = result
            if dist > self._cutoff_px:
                continue
            mapping = mapping_by_id.get(room_id)
            if mapping is None:
                continue
            _apply_detection(mapping, detection, self._registry, self._strict_type_dispatch)

        return list(mapping_by_id.values())
