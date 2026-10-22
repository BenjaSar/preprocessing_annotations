"""Express a room-extent box in the label convention.

A `vlm_only` room carries a wall-to-wall box (build_room_prompt_unsloth asks
for one) while every OCR-anchored room carries its label's own box, so one
room list mixes two box conventions (annotation_schema.box_convention_of).
Here the room's own box is cropped and a RoomLabelLocator is asked where the
room's printed name sits inside it; the answer, mapped back to page pixels,
is that room's label box.

Geometry is pure and separate from the model call, so the mapping is testable
without a VLM. A room whose name the locator cannot find keeps its own box:
this narrows a box, it never drops a room.
"""

import dataclasses
from typing import Any, List, Optional, Sequence, Tuple

from ..config import RoomLabelAnchorConfig
from ..vlm.read import RoomLabelLocator

PixelBoxXYWH = Tuple[int, int, int, int]
PixelBoxXYXY = Tuple[int, int, int, int]


@dataclasses.dataclass(frozen=True)
class LabelAnchor:
    """One attempt: the room's own box and the label box found inside it."""

    room_index: int
    room_name: str
    room_bbox: PixelBoxXYWH
    label_bbox: Optional[PixelBoxXYWH]


def crop_box(room_bbox: Sequence[float], image_size: Tuple[int, int], padding_frac: float) -> PixelBoxXYXY:
    """(left, top, right, bottom) around a room's xywh box, padded by
    `padding_frac` of its longest side and clamped to the image."""
    x, y, width, height = room_bbox
    pad = max(width, height) * padding_frac
    image_width, image_height = image_size
    return (
        max(0, int(x - pad)),
        max(0, int(y - pad)),
        min(image_width, int(x + width + pad)),
        min(image_height, int(y + height + pad)),
    )


def to_page_box(label_box: Sequence[int], crop: PixelBoxXYXY) -> Optional[PixelBoxXYWH]:
    """A locator's crop-pixel [x1, y1, x2, y2] as a page-pixel xywh box, or
    None when it is degenerate or falls outside the crop."""
    x1, y1, x2, y2 = (int(v) for v in label_box)
    left, top, right, bottom = crop
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    page = (left + x1, top + y1, x2 - x1, y2 - y1)
    if page[0] + page[2] > right or page[1] + page[3] > bottom:
        return None
    return page


def _room_name(room: dict) -> str:
    return room.get("room_name") or room.get("name") or ""


class RoomLabelAnchorer:
    """Anchors room-extent boxes to their printed label.

    One reason to change: how a room's label box is obtained. Owns no model --
    the locator is supplied, as with every other adapter in this codebase.
    """

    def __init__(self, locator: RoomLabelLocator, config: RoomLabelAnchorConfig):
        self._locator = locator
        self._config = config

    def anchor(self, image: Any, rooms: Sequence[dict], indices: Sequence[int]) -> List[LabelAnchor]:
        """`image`: the RGB page the room boxes are in. `indices`: which rooms
        to anchor -- the caller decides (it owns the box convention)."""
        anchors = []
        for index in indices:
            room = rooms[index]
            bbox = tuple(int(v) for v in room["bbox"])
            crop = crop_box(bbox, image.size, self._config.crop_padding_frac)
            found = self._locator.locate(image.crop(crop), _room_name(room))
            label = to_page_box(found, crop) if found else None
            anchors.append(LabelAnchor(index, _room_name(room), bbox, label))
        return anchors
