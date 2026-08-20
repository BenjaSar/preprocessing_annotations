"""
CubiCasa5K icon + room detection — Tier 2 integration.

Output layout (44-channel tensor, verified from checkpoint):
  channels  0–20: junction heatmaps  (sigmoid applied in forward)
  channels 21–32: room segmentation  (12 classes)
  channels 33–43: icon segmentation  (11 classes, softmax over this slice)

Icon classes (index within the 11-class slice):
  0=Empty, 1=Window, 2=Door, 3=Closet, 4=ElectricalAppliance,
  5=Toilet, 6=Sink, 7=SaunaBench, 8=FirePlace, 9=Bathtub, 10=Chimney

Room classes (index within the 12-class slice) — channel map verified
empirically via test/_cubicasa_probe.py (2026-06-30 probe, commercial
images): argmax regions align to drawn wall boundaries, not noise (GO):
  0=Background, 1=Outdoor, 2=Wall, 3=Kitchen, 4=LivingRoom, 5=BedRoom,
  6=Bath, 7=Entry, 8=Railing, 9=Storage, 10=Garage, 11=Undefined
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# The checkpoint (models/cubicasa5k_model.pkl) lives at the repo root, not
# alongside this file. This module is at
# preprocessing_annotations/src/preprocessing_annotations/detection/cubicasa5k_detector.py,
# 4 levels below the repo root -- parents[3], not Path(__file__).parent
# (which was correct only when this file lived directly at the repo root).
MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "cubicasa5k_model.pkl"

_ICON_OFFSET = 33   # first icon channel in the 44-ch output
_ICON_N      = 11   # number of icon classes

# Icon slice indices → semantic name (only the ones we expose)
_DETECTABLE = {1: "window", 2: "door", 5: "toilet", 6: "sink", 9: "bathtub"}

_ROOM_OFFSET = 21   # first room channel in the 44-ch output
_ROOM_N      = 12   # number of room classes
# Background/Outdoor/Wall are not room instances — every other class
# (including Undefined) counts as "a room" for class-agnostic instance
# extraction, matching cubicasa_gt.py's generic "room" GT category.
_NON_ROOM_CLASSES = {0, 1, 2}

_MAX_SIDE = 1024    # resize to ≤ this before inference (T4 memory constraint)


@dataclass
class IconMask:
    mask: np.ndarray                              # binary (H, W) uint8 at original resolution
    bboxes: List[Tuple[float, float, float, float]]  # [(x1,y1,x2,y2), ...]


@dataclass
class RoomMask:
    mask: np.ndarray                              # binary (H, W) uint8, 1 = any room class
    bboxes: List[Tuple[float, float, float, float]]  # [(x1,y1,x2,y2), ...] per connected component


class CubiCasa5KDetector:
    def __init__(self, model_path: Optional[Path] = None, device: str = "cuda"):
        self.model_path = model_path or MODEL_PATH
        self.device = device
        self.model = None
        self.is_available = self.model_path.exists()
        if not self.is_available:
            logger.warning(f"CubiCasa5K model not found: {self.model_path}")

    def load_model(self) -> bool:
        if self.model is not None:
            return True
        if not self.is_available:
            return False
        try:
            import torch
            from .vendor.seg_model import hg_furukawa_original
            ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
            m = hg_furukawa_original(n_classes=44)
            m.load_state_dict(ckpt["model_state"], strict=True)
            m.eval()
            m.to(self.device)
            self.model = m
            logger.info("CubiCasa5K model loaded")
            return True
        except Exception as e:
            logger.warning(f"CubiCasa5K load failed: {e}")
            return False

    def detect_icons(
        self, image: np.ndarray, threshold: float = 0.3
    ) -> Dict[str, IconMask]:
        """
        Run inference and return per-icon binary masks + bboxes.

        Args:
            image: RGB uint8 (H, W, 3) at any resolution.
            threshold: Minimum softmax probability to count as detected.

        Returns:
            Dict keyed by icon name ("window","door","toilet","sink","bathtub").
            Missing key = no pixels above threshold for that class.
        """
        if self.model is None and not self.load_model():
            return {}

        try:
            import torch
            import torch.nn.functional as F

            orig_h, orig_w = image.shape[:2]
            tensor, scale = self._preprocess(image)

            with torch.no_grad():
                out = self.model(tensor)  # (1, 44, H', W')

            # Softmax over the 11 icon classes
            icon_logits = out[0, _ICON_OFFSET: _ICON_OFFSET + _ICON_N]  # (11, H', W')
            icon_probs = torch.softmax(icon_logits, dim=0).cpu().numpy()  # (11, H', W')

            results: Dict[str, IconMask] = {}
            for idx, name in _DETECTABLE.items():
                prob_map = icon_probs[idx]
                bin_mask = (prob_map > threshold).astype(np.uint8)
                if bin_mask.any():
                    # Upsample mask back to original resolution
                    if scale < 1.0:
                        import cv2
                        bin_mask = cv2.resize(
                            bin_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                        )
                    bboxes = self._extract_bboxes(bin_mask)
                    results[name] = IconMask(mask=bin_mask, bboxes=bboxes)

            return results

        except Exception as e:
            logger.warning(f"CubiCasa5K inference failed: {e}")
            return {}

    def detect_rooms(
        self,
        image: np.ndarray,
        per_class: bool = True,
        min_area_px: int = 0,
        barrier_dilate_px: int = 0,
    ) -> RoomMask:
        """
        Run inference and return class-agnostic room instance boxes.

        Room-TYPE labels (Kitchen/Bath/...) are not scored anywhere in this
        project — eval/cubicasa_gt.py's GT is a generic "room" category only
        (see its docstring). Boxes are therefore emitted without a type, but
        the type still matters for INSTANCE SEPARATION:

        per_class=True (default) runs connected components independently
        within each room-type channel, then pools the boxes. per_class=False
        unions all room classes into one binary mask first, which merges
        adjacent rooms of DIFFERENT types (a kitchen abutting a living room
        becomes one component) and massively undersegments — measured at
        89.3% of false negatives being localize-fail, 91.4% of those
        oversize, via eval/cubicasa_cnn_room_fn_diagnostic.py. Kept as a
        flag only so that diagnostic can reproduce the union behaviour.

        per_class alone does not fix SAME-type adjacent rooms (two bedrooms
        sharing a wall are still one component if the wall's argmax is thin
        or broken) -- measured at min_area_px=2500: 70.9% of remaining FN
        still localize-fail, 82.1% of those oversize, 554 predictions each
        swallowing >=2 GT rooms (cubicasa_cnn_room_fn_diagnostic.py). This is
        what barrier_dilate_px targets: > 0 builds a barrier mask from the
        model's OWN wall class (room slice, class 2) and door/window classes
        (icon slice, classes 1/2 -- both already in the `out` tensor this
        call already computes, no extra inference), dilates it by that many
        pixels, and subtracts it from each per-class room mask before
        connected-components -- so a thin/interrupted predicted wall line
        still fully separates two same-type neighbors. `[Inferred]`: general
        marker/barrier instance-separation applied to this checkpoint's own
        boundary channels; untested before eval/cubicasa_cnn_room_sweep.py
        --barrier-dilate-px sweeps confirm it helps net P/R.

        Args:
            image: RGB uint8 (H, W, 3) at any resolution.
            per_class: extract instances per room-type channel (see above).
            barrier_dilate_px: dilate the wall+opening barrier by this many
                pixels before subtracting from room masks. 0 = off (barrier
                pixels are still excluded via per-class extraction's own
                argmax boundary, but not dilated/reinforced).

        Returns:
            RoomMask with the binary room mask and per-instance bboxes at
            original resolution. Empty (zero-size mask, no bboxes) if the
            model is unavailable.
        """
        if self.model is None and not self.load_model():
            return RoomMask(mask=np.zeros((0, 0), dtype=np.uint8), bboxes=[])

        try:
            import torch

            orig_h, orig_w = image.shape[:2]
            tensor, scale = self._preprocess(image)

            with torch.no_grad():
                out = self.model(tensor)  # (1, 44, H', W')

            room_logits = out[0, _ROOM_OFFSET: _ROOM_OFFSET + _ROOM_N]  # (12, H', W')
            room_argmax = torch.argmax(room_logits, dim=0).cpu().numpy()  # (H', W')

            barrier = None
            if barrier_dilate_px > 0:
                icon_logits = out[0, _ICON_OFFSET: _ICON_OFFSET + _ICON_N]  # (11, H', W')
                icon_argmax = torch.argmax(icon_logits, dim=0).cpu().numpy()  # (H', W')
                wall = room_argmax == 2  # Wall (room slice)
                opening = np.isin(icon_argmax, [1, 2])  # Window, Door (icon slice)
                barrier = (wall | opening).astype(np.uint8)

            if scale < 1.0:
                import cv2
                room_argmax = cv2.resize(
                    room_argmax.astype(np.int32), (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                if barrier is not None:
                    barrier = cv2.resize(
                        barrier, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                    )

            if barrier is not None:
                import cv2
                k = 2 * barrier_dilate_px + 1
                kernel = np.ones((k, k), dtype=np.uint8)
                barrier = cv2.dilate(barrier, kernel, iterations=1)

            room_mask = np.isin(
                room_argmax, list(_NON_ROOM_CLASSES), invert=True
            ).astype(np.uint8)
            if barrier is not None:
                room_mask = room_mask & (1 - barrier)

            if per_class:
                bboxes: List[Tuple[float, float, float, float]] = []
                for class_idx in range(_ROOM_N):
                    if class_idx in _NON_ROOM_CLASSES:
                        continue
                    class_mask = (room_argmax == class_idx).astype(np.uint8)
                    if barrier is not None:
                        class_mask = class_mask & (1 - barrier)
                    if class_mask.any():
                        bboxes.extend(self._extract_bboxes(class_mask))
            else:
                bboxes = self._extract_bboxes(room_mask)

            if min_area_px > 0:
                bboxes = [
                    b for b in bboxes
                    if (b[2] - b[0]) * (b[3] - b[1]) >= min_area_px
                ]

            return RoomMask(mask=room_mask, bboxes=bboxes)

        except Exception as e:
            logger.warning(f"CubiCasa5K room inference failed: {e}")
            return RoomMask(mask=np.zeros((0, 0), dtype=np.uint8), bboxes=[])

    # ── helpers ──────────────────────────────────────────────────────────────

    def _preprocess(self, image: np.ndarray) -> Tuple["torch.Tensor", float]:
        import torch
        h, w = image.shape[:2]
        scale = min(1.0, _MAX_SIDE / max(h, w))
        if scale < 1.0:
            import cv2
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        img = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return tensor, scale

    @staticmethod
    def _extract_bboxes(mask: np.ndarray) -> List[Tuple[float, float, float, float]]:
        try:
            import cv2
            from scipy import ndimage
            labeled, n = ndimage.label(mask)
            bboxes = []
            for cid in range(1, n + 1):
                comp = (labeled == cid).astype(np.uint8)
                contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w > 10 and h > 10:
                        bboxes.append((float(x), float(y), float(x + w), float(y + h)))
            return bboxes
        except Exception as e:
            logger.warning(f"bbox extraction failed: {e}")
            return []
