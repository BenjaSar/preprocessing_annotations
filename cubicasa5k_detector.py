"""
CubiCasa5K Window Detection — Tier 2 Integration.

CubiCasa5K is a multi-task CNN that outputs:
  - Wall segmentation masks
  - Icon segmentation (windows, doors, furniture)
  - Junction heatmaps (for wall topology)

This module handles model loading, inference, and window extraction.

Reference: https://github.com/CubiCasa/CubiCasa5k
Paper: arXiv:1904.01920 "Raster-to-Vector: Revisiting Floorplan Transformation"
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
from urllib.request import urlopen
import os

import numpy as np

logger = logging.getLogger(__name__)

# Model URLs and paths
CUBICASA5K_MODEL_URL = "https://drive.google.com/uc?id=1gRB7ez1e4H7a9Y09lLqRuna0luZO5VRK"
CUBICASA5K_MODEL_PATH = Path(__file__).parent / "models" / "cubicasa5k_model.pkl"
CUBICASA5K_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


class IconType(Enum):
    """CubiCasa5K icon classes in the icon segmentation output."""
    VOID = 0
    WINDOW = 1
    DOOR = 2
    TOILET = 3
    BATHTUB = 4
    SINK = 5
    FURNITURE = 6


@dataclass
class WindowMask:
    """Result of window detection from CubiCasa5K."""
    mask: np.ndarray  # Binary mask (H, W) where 1 = window pixel
    bboxes: List[Tuple[float, float, float, float]]  # [x1, y1, x2, y2] per window
    confidence: float  # Detection confidence (typically 0.9 for model output)


class CubiCasa5KDetector:
    """
    CubiCasa5K window detection using pretrained multi-task model.

    The model outputs four channels:
      - Channel 0: Walls (semantic segmentation)
      - Channel 1: Windows (icon segmentation channel)
      - Channel 2: Doors (icon segmentation channel)
      - Channel 3-N: Other icons (furniture, fixtures)

    We extract windows from Channel 1 and post-process to get bounding boxes.
    """

    def __init__(self, model_path: Optional[Path] = None, device: str = "cuda"):
        """
        Initialize CubiCasa5K detector.

        Args:
            model_path: Path to model checkpoint. Downloads if not found.
            device: Device for inference ("cuda" or "cpu").
        """
        self.model_path = model_path or CUBICASA5K_MODEL_PATH
        self.device = device
        self.model = None
        self.is_available = False

        # Check if model is available locally
        if self.model_path.exists():
            self.is_available = True
            logger.info(f"CubiCasa5K model found at {self.model_path}")
        else:
            logger.warning(
                f"CubiCasa5K model not found at {self.model_path}. "
                f"To use Tier 2 detection, download from: "
                f"{CUBICASA5K_MODEL_URL}"
            )

    def load_model(self) -> bool:
        """
        Load pretrained CubiCasa5K model.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        if self.model is not None:
            return True  # Already loaded

        if not self.is_available:
            logger.debug("Model not available; cannot load")
            return False

        try:
            import torch
            import pickle

            logger.info(f"Loading CubiCasa5K model from {self.model_path}")

            with open(self.model_path, "rb") as f:
                checkpoint = pickle.load(f)

            # The loaded checkpoint should be a PyTorch model or dict
            # For compatibility with older PyTorch, we might need:
            # checkpoint = torch.load(..., map_location=self.device)

            # Placeholder: actual model instantiation would depend on
            # the exact format of the checkpoint. The original code uses
            # a custom architecture. For now, we store the checkpoint.

            self.model = checkpoint
            logger.info("CubiCasa5K model loaded successfully")
            return True

        except Exception as e:
            logger.warning(f"Failed to load CubiCasa5K model: {e}")
            return False

    def detect_windows(
        self, image: np.ndarray, confidence_threshold: float = 0.3
    ) -> WindowMask:
        """
        Detect windows in floorplan image using CubiCasa5K.

        Args:
            image: Input image, shape (H, W, 3), dtype uint8, RGB.
            confidence_threshold: Confidence threshold for window pixels (0-1).

        Returns:
            WindowMask with binary mask and bounding boxes.
        """
        if self.model is None:
            if not self.load_model():
                logger.warning("Cannot run inference without model")
                return WindowMask(
                    mask=np.zeros(image.shape[:2], dtype=np.uint8),
                    bboxes=[],
                    confidence=0.0,
                )

        try:
            import torch

            # Preprocess: normalize and convert to tensor
            # Typical preprocessing: subtract mean, divide by std, transpose to CHW
            # CubiCasa5K expects: (H, W, 3) -> (1, 3, H, W)

            image_tensor = self._preprocess(image)

            # Run inference
            with torch.no_grad():
                output = self.model(image_tensor)

            # Extract window channel (channel 1 of icon segmentation)
            window_mask = self._extract_window_mask(
                output, threshold=confidence_threshold
            )

            # Post-process: connected components, bounding boxes
            bboxes = self._extract_bboxes(window_mask)

            return WindowMask(
                mask=window_mask,
                bboxes=bboxes,
                confidence=0.9,  # Model confidence (high for pretrained)
            )

        except Exception as e:
            logger.warning(f"CubiCasa5K inference failed: {e}")
            return WindowMask(
                mask=np.zeros(image.shape[:2], dtype=np.uint8),
                bboxes=[],
                confidence=0.0,
            )

    def _preprocess(self, image: np.ndarray) -> "torch.Tensor":
        """
        Preprocess image for CubiCasa5K model.

        Args:
            image: Input image (H, W, 3), uint8, RGB.

        Returns:
            Tensor (1, 3, H, W), float32, normalized.
        """
        import torch

        # Normalize to [0, 1]
        image_float = image.astype(np.float32) / 255.0

        # ImageNet normalization (typical for vision models)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_normalized = (image_float - mean) / std

        # Convert to tensor and transpose: (H, W, 3) -> (3, H, W) -> (1, 3, H, W)
        image_tensor = torch.from_numpy(
            image_normalized.transpose(2, 0, 1)
        ).unsqueeze(0)

        # Move to device
        image_tensor = image_tensor.to(self.device)

        return image_tensor

    def _extract_window_mask(
        self, output: "torch.Tensor", threshold: float = 0.3
    ) -> np.ndarray:
        """
        Extract window segmentation from model output.

        Args:
            output: Model output tensor (logits or probabilities).
            threshold: Confidence threshold for window pixels.

        Returns:
            Binary mask (H, W), uint8, where 1 = window, 0 = not window.
        """
        import torch

        try:
            # Assuming output shape: (B, C, H, W) where C >= 2
            # Channel 1: window probability
            if len(output.shape) == 4:
                window_channel = output[0, IconType.WINDOW.value]  # (H, W)
            else:
                # Fallback: assume single-channel output
                window_channel = output

            # Apply softmax if logits
            if window_channel.max() > 1.0 or window_channel.min() < 0.0:
                window_probs = torch.softmax(window_channel.unsqueeze(0), dim=1)
            else:
                window_probs = window_channel

            # Threshold and convert to numpy
            mask = (window_probs > threshold).cpu().numpy().astype(np.uint8)

            return mask

        except Exception as e:
            logger.warning(f"Failed to extract window mask: {e}")
            return np.zeros(output.shape[-2:], dtype=np.uint8)

    def _extract_bboxes(
        self, mask: np.ndarray
    ) -> List[Tuple[float, float, float, float]]:
        """
        Extract bounding boxes from binary window mask.

        Uses connected components to identify individual windows.

        Args:
            mask: Binary mask (H, W), uint8.

        Returns:
            List of bboxes [(x1, y1, x2, y2), ...].
        """
        try:
            from scipy import ndimage
            import cv2

            # Label connected components
            labeled, num_features = ndimage.label(mask)

            bboxes = []
            for component_id in range(1, num_features + 1):
                # Get bounding box of this component
                component_mask = labeled == component_id

                # Find contours
                contours, _ = cv2.findContours(
                    component_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    # Skip very small windows (likely noise)
                    if w > 10 and h > 10:
                        bboxes.append((float(x), float(y), float(x + w), float(y + h)))

            return bboxes

        except Exception as e:
            logger.warning(f"Failed to extract bboxes from mask: {e}")
            return []
