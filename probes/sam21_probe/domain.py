"""Domain types for the SAM1-vs-SAM2.1 flood bake-off.

Framework-agnostic: no SAM1/SAM2, torch import here. Adapters translate into
this shape so the application layer never touches vendor types.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Seed:
    """A real, production-derived room-label seed (OCR Pass-1 candidate)."""

    label: str
    center: tuple[int, int]
    label_bbox: tuple[int, int, int, int]  # xywh, as produced by OCR


@dataclass(frozen=True)
class Expansion:
    """One model's point-prompt expansion of a single seed."""

    bbox_xywh: tuple[int, int, int, int]
    confidence: float
    mask: Optional[np.ndarray] = None
