"""Crop text reader port (ingestion/crop_reread.py's model dependency).

One method, deliberately not added to VLMBackend -- same reasoning as
vlm/classify/port.py: only UnslothQwenBackend implements it, and a
narrow task-specific interface keeps the other backends untouched (ISP).
"""

from typing import Any, Protocol


class CropTextReader(Protocol):
    """Any implementation must be swappable for another (LSP): same input,
    a plain string out, never raises ("" when nothing legible or on
    failure)."""

    def read(self, image: Any) -> str:
        """Transcribe the text label at the centre of an RGB PIL image."""
        ...
