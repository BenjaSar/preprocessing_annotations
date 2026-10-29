"""AV-6 (2026-09-23): PageClassifier adapter over UnslothQwenBackend.

Thin translation only -- all model/prompt/parsing logic lives on
UnslothQwenBackend.classify_page_type (vlm_backend.py), the same
concrete class STEP 3's room detection already uses. This adapter
exists so pipeline.py depends on the PageClassifier port (DIP) instead
of importing UnslothQwenBackend directly for this one purpose, and so a
future non-Unsloth adapter can be added (OCP) without touching
pipeline.py's call site.

Does not own or create the model: the caller supplies the same backend
instance already used for room detection (pipeline.py's vlm_annotator
property), so this never triggers a second model load and never risks
reopening the VLM/SAM residency conflict _release_vlm_before_sam exists
to prevent (pipeline.py).
"""

from pathlib import Path
from typing import Union

from ..vlm_backend import UnslothQwenBackend
from .port import PageTypeResult


class UnslothPageClassifier:
    """PageClassifier backed by an already-constructed
    UnslothQwenBackend instance."""

    def __init__(self, backend: UnslothQwenBackend):
        self._backend = backend

    def classify(self, image_path: Union[str, Path]) -> PageTypeResult:
        result = self._backend.classify_page_type(image_path)
        return PageTypeResult(
            page_type=result.get("page_type", "other"),
            rationale=result.get("rationale", ""),
        )
