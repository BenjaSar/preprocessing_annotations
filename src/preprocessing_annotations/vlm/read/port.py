"""Crop text reader port (ingestion/crop_reread.py's model dependency).

One method, deliberately not added to VLMBackend -- same reasoning as
vlm/classify/port.py: only UnslothQwenBackend implements it, and a
narrow task-specific interface keeps the other backends untouched (ISP).
"""

from typing import Any, List, Optional, Protocol


class CropTextReader(Protocol):
    """Any implementation must be swappable for another (LSP): same input,
    a plain string out, never raises ("" when nothing legible or on
    failure)."""

    def read(self, image: Any) -> str:
        """Transcribe the text label at the centre of an RGB PIL image."""
        ...


class RoomLabelLocator(Protocol):
    """Where a room's printed name sits inside a crop of that room.

    Separate from CropTextReader on purpose (ISP): that one returns text for
    a box already known, this one returns a box for text already known, and
    no caller needs both. Same contract otherwise: swappable (LSP), never
    raises, None when the name is not legibly printed in the crop."""

    def locate(self, image: Any, room_name: str) -> Optional[List[int]]:
        """[x1, y1, x2, y2] in crop pixels, or None."""
        ...
