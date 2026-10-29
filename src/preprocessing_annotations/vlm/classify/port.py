"""AV-6 (2026-09-23): page-type classifier port.

One method, deliberately not added to VLMBackend (the ABC every
ClaudeBackend/Qwen2_5VLBackend/UnslothQwenBackend implements) --
widening that shared contract would force every backend to implement
page-type classification even though only UnslothQwenBackend is
exercised in production (every run manifest on disk on this project
uses --vlm-backend unsloth). ISP: a narrow, task-specific interface for
this one caller's need, instead of a wider shared one.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Union


@dataclass(frozen=True)
class PageTypeResult:
    page_type: str
    rationale: str


class PageClassifier(Protocol):
    """Any implementation must be swappable for another (LSP): same
    input, same PageTypeResult shape, never raises (falls back to a
    result with page_type="other" instead)."""

    def classify(self, image_path: Union[str, Path]) -> PageTypeResult:
        ...
