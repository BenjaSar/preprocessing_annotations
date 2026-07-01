"""
CubiCasa5K icon detection — Tier 2 integration.

Output layout (44-channel tensor, verified from checkpoint):
  channels  0–20: junction heatmaps  (sigmoid applied in forward)
  channels 21–32: room segmentation  (12 classes)
  channels 33–43: icon segmentation  (11 classes, softmax over this slice)

Icon classes (index within the 11-class slice):
  0=Empty, 1=Window, 2=Door, 3=Closet, 4=ElectricalAppliance,
  5=Toilet, 6=Sink, 7=SaunaBench, 8=FirePlace, 9=Bathtub, 10=Chimney
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "models" / "cubicasa5k_model.pkl"

_ICON_OFFSET = 33   # first icon channel in the 44-ch output
_ICON_N      = 11   # number of icon classes

# Icon slice indices → semantic name (only the ones we expose)
_DETECTABLE = {1: "window", 2: "door", 5: "toilet", 6: "sink", 9: "bathtub"}

_MAX_SIDE = 1024    # resize to ≤ this before inference (T4 memory constraint)


@dataclass
class IconMask:
    mask: np.ndarray                              # binary (H, W) uint8 at original resolution
    bboxes: List[Tuple[float, float, float, float]]  # [(x1,y1,x2,y2), ...]


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
            from models.seg_model import hg_furukawa_original
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
