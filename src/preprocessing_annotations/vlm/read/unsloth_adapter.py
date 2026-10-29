"""CropTextReader adapter over UnslothQwenBackend.

Thin translation only -- prompt, generation and parsing live on
UnslothQwenBackend.read_crop_text. Does not own or create the model: the
caller supplies an already-constructed backend, as with
vlm/classify/unsloth_adapter.py.
"""

from typing import Any

from ..vlm_backend import UnslothQwenBackend


class UnslothCropTextReader:
    """CropTextReader backed by an UnslothQwenBackend instance."""

    def __init__(self, backend: UnslothQwenBackend):
        self._backend = backend

    def read(self, image: Any) -> str:
        return self._backend.read_crop_text(image)
