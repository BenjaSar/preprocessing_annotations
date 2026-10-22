"""One room, two boxes: a room-extent box that already holds a label box.

`vlm_only` rooms carry a wall-to-wall box (the VLM prompt asks for one) while
OCR-anchored rooms carry the label's own box, so one room can appear twice --
once as a big box, once as a small one inside it (annotation_schema's
box_convention_of names the two conventions). This module drops the big box
only where the small one makes it redundant; a room-extent box with no label
box inside it is the only record of that room and is never touched here.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .annotation_schema import LABEL_BOX, ROOM_EXTENT_BOX, box_convention_of


def _room_type(room: Dict[str, Any]) -> Optional[str]:
    return room.get("type") or room.get("category")


def _centre(bbox: Sequence[float]) -> Tuple[float, float]:
    x, y, width, height = bbox
    return x + width / 2, y + height / 2


def _holds_centre(outer: Sequence[float], inner: Sequence[float]) -> bool:
    x, y, width, height = outer
    cx, cy = _centre(inner)
    return x <= cx <= x + width and y <= cy <= y + height


def _usable(room: Dict[str, Any]) -> bool:
    bbox = room.get("bbox")
    return isinstance(bbox, (list, tuple)) and len(bbox) == 4


def split_redundant_room_extents(
    rooms: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(kept, dropped): a room-extent room is dropped when a label-box room of
    the same type has its centre inside it -- the same room, said twice."""
    labels_by_type: Dict[Optional[str], List[Dict[str, Any]]] = {}
    for room in rooms:
        if _usable(room) and box_convention_of(room.get("source")) == LABEL_BOX:
            labels_by_type.setdefault(_room_type(room), []).append(room)

    kept, dropped = [], []
    for room in rooms:
        redundant = (
            _usable(room)
            and box_convention_of(room.get("source")) == ROOM_EXTENT_BOX
            and any(
                _holds_centre(room["bbox"], label["bbox"])
                for label in labels_by_type.get(_room_type(room), ())
            )
        )
        (dropped if redundant else kept).append(room)
    return kept, dropped
