"""Room extents from PDF vector walls, attached beside each room's label box.

A room takes the smallest wall face containing its label-box centre, and
only when no other room label of the page lands in that face: a face shared
by several labels is rooms merged through an opening the drawn walls leave
open, not one room. Rooms without such a face keep no extent. The label box
itself is never changed, so every label-location measurement is unaffected.
"""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree

from ..ingestion.vector_walls import VectorWallReader

EXTENT_SOURCE = "pdf_vector_walls"


def _centre(box: Sequence[float]) -> Point:
    x1, y1, x2, y2 = box
    return Point((x1 + x2) / 2, (y1 + y2) / 2)


def assign_extents(
    faces: Sequence[Polygon], label_boxes: Sequence[Sequence[float]],
) -> List[Optional[Polygon]]:
    """Per xyxy label box: its single-label face, or None."""
    tree = STRtree(faces) if faces else None
    chosen: List[Optional[int]] = []
    for box in label_boxes:
        centre = _centre(box)
        containing = (
            [i for i in tree.query(centre) if faces[i].contains(centre)]
            if tree is not None else []
        )
        chosen.append(min(containing, key=lambda i: faces[i].area) if containing else None)
    labels_per_face = Counter(i for i in chosen if i is not None)
    return [
        faces[i] if i is not None and labels_per_face[i] == 1 else None
        for i in chosen
    ]


def extent_record(face: Polygon) -> Dict[str, Any]:
    """The coordinates.extent entry: xyxy box, outline, provenance."""
    x1, y1, x2, y2 = face.bounds
    return {
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "polygon": [[int(x), int(y)] for x, y in face.exterior.coords],
        "source": EXTENT_SOURCE,
    }


class RoomExtentStep:
    """Attaches extents to one page's roomsRecognized entries."""

    def __init__(self, reader: VectorWallReader):
        self._reader = reader

    def attach(
        self, rooms: List[Dict[str, Any]], pdf_path: Path, page_index: int,
        image_size: Tuple[int, int],
    ) -> int:
        """Sets coordinates.extent on rooms that get one; returns how many.
        Raises VectorWallError when the page's drawings cannot be read."""
        faces = self._reader.faces(pdf_path, page_index, image_size)
        boxes = [room["coordinates"]["bbox"] for room in rooms]
        attached = 0
        for room, face in zip(rooms, assign_extents(faces, boxes)):
            if face is not None:
                room["coordinates"]["extent"] = extent_record(face)
                attached += 1
        return attached
