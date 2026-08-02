"""Door detection — first-class architectural-object detector (T-B0/T-B1).

Tiered design mirroring ``window_detector.py``:
  - CubiCasa5K tier: reuse the pretrained model's door icon class.
  - VLM-prompt tier: bridge to a backend's ``detect_doors`` method.

CubiCasa5K already emits doors (``_DETECTABLE[2] == "door"``), but
``WindowDetector`` surfaces them only as a side effect and its window prompt
excludes doors, so no first-class door path existed. This module adds one
without loading a new model, reusing an injected/lazy ``CubiCasa5KDetector``
and optionally a pre-computed icon dict to avoid a second forward pass.

Nothing wires it into the default pipeline yet, so current behavior is
unchanged.
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Icon class name as exposed by CubiCasa5KDetector._DETECTABLE.
_CUBICASA_DOOR_CLASS = "door"

# Fixed prior confidence for a CubiCasa door: the segmentation model yields no
# per-instance score, matching the window detector's CubiCasa-tier constant.
_CUBICASA_DOOR_CONFIDENCE = 0.9

# Fallback confidence when a VLM detection omits one.
_VLM_DEFAULT_CONFIDENCE = 0.7

# Expected length of a [x1, y1, x2, y2] bbox.
_BBOX_LEN = 4


class DoorDetectionTier(Enum):
    """Detection tier identifier for provenance tracking."""

    CUBICASA5K = "cubicasa5k"
    VLM_PROMPT = "vlm_prompt"
    NONE = "none"


@dataclass
class DoorDetection:
    """A single detected door (bbox in absolute pixels)."""

    bbox: Tuple[float, float, float, float]
    confidence: float = _CUBICASA_DOOR_CONFIDENCE
    source_tier: DoorDetectionTier = DoorDetectionTier.NONE
    metadata: Optional[Dict[str, Any]] = None


def _import_cubicasa_cls() -> Optional[type]:
    """Import ``CubiCasa5KDetector`` from either package layout."""
    try:
        from cubicasa5k_detector import CubiCasa5KDetector
        return CubiCasa5KDetector
    except ImportError:
        pass
    try:
        from preprocessing_annotations.detection.cubicasa5k_detector import (
            CubiCasa5KDetector,
        )
        return CubiCasa5KDetector
    except ImportError:
        logger.warning("CubiCasa5K import failed; door detection unavailable")
        return None


def _resolve_device(device: Optional[str]) -> str:
    """Return an explicit device, auto-detecting when None."""
    if device is not None:
        return device
    try:
        from config import _detect_device
    except ImportError:
        from preprocessing_annotations.config import _detect_device
    return _detect_device()


def _cubicasa_doors(
    bboxes: List[Tuple[float, float, float, float]],
) -> List[DoorDetection]:
    """Wrap CubiCasa door bboxes as CubiCasa-tier detections."""
    return [
        DoorDetection(
            bbox=bbox,
            confidence=_CUBICASA_DOOR_CONFIDENCE,
            source_tier=DoorDetectionTier.CUBICASA5K,
            metadata={"type": _CUBICASA_DOOR_CLASS},
        )
        for bbox in bboxes
    ]


def _door_from_dict(entry: Dict[str, Any]) -> Optional[DoorDetection]:
    """Convert one backend detection dict to a DoorDetection.

    Returns None for malformed or degenerate (zero-area) boxes.
    """
    bbox = entry.get("bbox", [])
    if len(bbox) != _BBOX_LEN:
        return None
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    confidence = float(entry.get("confidence", _VLM_DEFAULT_CONFIDENCE))
    return DoorDetection(
        bbox=(float(x1), float(y1), float(x2), float(y2)),
        confidence=confidence,
        source_tier=DoorDetectionTier.VLM_PROMPT,
        metadata={"type": entry.get("type", _CUBICASA_DOOR_CLASS)},
    )


def _doors_from_backend_dicts(
    raw: List[Dict[str, Any]],
) -> List[DoorDetection]:
    """Convert a backend's door dicts to validated detections."""
    converted = (_door_from_dict(entry) for entry in raw)
    return [door for door in converted if door is not None]


def _write_temp_png(
    image_array: np.ndarray,
) -> Tuple[Optional[Path], Optional[str]]:
    """Write an array to a temp PNG; return (path, temp_name)."""
    try:
        from PIL import Image
        handle, tmp_name = tempfile.mkstemp(suffix=".png")
        os.close(handle)
        Image.fromarray(image_array).save(tmp_name)
        return Path(tmp_name), tmp_name
    except (OSError, ValueError) as error:
        logger.warning("VLM door tier: temp image write failed: %s", error)
        return None, None


def _resolve_image_source(
    img_path: Optional[Path],
    image_array: Optional[np.ndarray],
) -> Tuple[Optional[Path], Optional[str]]:
    """Resolve an image path for the backend, writing a temp file if needed.

    Returns:
        (path, temp_name); temp_name is set only when a temp file was created.
    """
    if img_path is not None and Path(img_path).is_file():
        return Path(img_path), None
    if image_array is None:
        return None, None
    return _write_temp_png(image_array)


def _cleanup_temp(temp_name: Optional[str]) -> None:
    """Delete a temp file if one was created; ignore if already gone."""
    if temp_name is None:
        return
    try:
        os.unlink(temp_name)
    except OSError:
        pass


class DoorDetector:
    """Orchestrator for tiered door detection.

    Args:
        detector: A ready CubiCasa5KDetector (dependency injection). If None,
            one is lazily built on first use.
        device: Device for a lazily-built detector; auto-detected if None.
    """

    def __init__(
        self,
        detector: Optional[Any] = None,
        device: Optional[str] = None,
    ) -> None:
        self._detector = detector
        self._device = device

    def _cubicasa(self) -> Optional[Any]:
        """Return a loaded CubiCasa5KDetector, or None if unavailable."""
        if self._detector is not None:
            return self._detector
        detector_cls = _import_cubicasa_cls()
        if detector_cls is None:
            return None
        detector = detector_cls(device=_resolve_device(self._device))
        if not detector.load_model():
            logger.debug("CubiCasa5K model unavailable; no doors")
            return None
        self._detector = detector
        return detector

    def _resolve_icons(
        self,
        image_array: Optional[np.ndarray],
        icons: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Return an icon dict: the given one, or a fresh CubiCasa run."""
        if icons is not None:
            return icons
        if image_array is None:
            return None
        detector = self._cubicasa()
        if detector is None:
            return None
        return detector.detect_icons(image_array)

    def detect_doors_from_cubicasa5k(
        self,
        image_array: Optional[np.ndarray] = None,
        icons: Optional[Dict[str, Any]] = None,
    ) -> List[DoorDetection]:
        """Extract doors from CubiCasa5K icon output.

        Args:
            image_array: RGB uint8 (H, W, 3); used only when ``icons`` is None.
            icons: Pre-computed ``detect_icons`` result; skips inference.

        Returns:
            One detection per door bbox; empty if none or model unavailable.
        """
        icons = self._resolve_icons(image_array, icons)
        if icons is None:
            return []
        door_mask = icons.get(_CUBICASA_DOOR_CLASS)
        if door_mask is None:
            return []
        doors = _cubicasa_doors(door_mask.bboxes)
        if doors:
            logger.info("CubiCasa5K door tier: %d door(s)", len(doors))
        return doors

    def detect_doors_from_vlm_prompt(
        self,
        vlm_backend: Any,
        img_path: Optional[Path] = None,
        image_array: Optional[np.ndarray] = None,
    ) -> List[DoorDetection]:
        """Detect doors via a VLM backend's ``detect_doors`` method.

        Prefers a file path; otherwise writes ``image_array`` to a temp PNG
        and removes it afterward.
        """
        if vlm_backend is None:
            return []
        effective_path, temp_name = _resolve_image_source(
            img_path, image_array
        )
        if effective_path is None:
            return []
        raw = self._call_backend_doors(vlm_backend, effective_path, temp_name)
        doors = _doors_from_backend_dicts(raw)
        logger.info("VLM door tier: %d door(s)", len(doors))
        return doors

    @staticmethod
    def _call_backend_doors(
        vlm_backend: Any,
        image_path: Path,
        temp_name: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Call the backend and always clean up the temp file."""
        try:
            return vlm_backend.detect_doors(image_path)
        except (OSError, ValueError, RuntimeError) as error:
            logger.warning("VLM door tier failed: %s", error)
            return []
        finally:
            _cleanup_temp(temp_name)

    def detect_doors(
        self,
        image_array: Optional[np.ndarray] = None,
        icons: Optional[Dict[str, Any]] = None,
        vlm_backend: Optional[Any] = None,
        img_path: Optional[Path] = None,
        use_vlm_tier: bool = False,
    ) -> List[DoorDetection]:
        """Run the tiered door-detection pipeline.

        Always runs the CubiCasa5K tier. When ``use_vlm_tier`` is True and a
        backend is given, the VLM tier runs too and its detections are unioned
        (each carries ``source_tier``). Default is CubiCasa-only, identical to
        prior behavior, so nothing regresses.
        """
        doors = self.detect_doors_from_cubicasa5k(
            image_array=image_array, icons=icons
        )
        if use_vlm_tier and vlm_backend is not None:
            doors = doors + self.detect_doors_from_vlm_prompt(
                vlm_backend, img_path=img_path, image_array=image_array
            )
        return doors
