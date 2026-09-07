"""
OCR Adapter — Abstract layer for multiple OCR backends.

Normalizes output from different OCR engines (EasyOCR, PaddleOCR) to a common format:
    TextDetection: (bbox, text, confidence)
    where bbox = [x1, y1, x2, y2] (top-left and bottom-right corners)

Supports:
  - EasyOCR (legacy): Returns (pts, text, confidence) tuples
  - PaddleOCR: Returns [[[x,y], [x,y], [x,y], [x,y]], (text, confidence)]

Backend selection via OCRConfig.backend flag.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TextDetection:
    """Normalized OCR text detection.
    
    Attributes:
        bbox: [x1, y1, x2, y2] - top-left and bottom-right corners
        text: Recognized text string
        confidence: 0.0-1.0 confidence score
        quadrilateral: Original 4-point quadrilateral from OCR engine (for spatial analysis)
    """
    bbox: Tuple[float, float, float, float]
    text: str
    confidence: float
    quadrilateral: Optional[List[Tuple[float, float]]] = None


class OCRBackend(ABC):
    """Abstract base class for OCR backends."""

    @abstractmethod
    def extract_text(self, image_source: Union[str, Path, np.ndarray]) -> List[TextDetection]:
        """Extract text from image.
        
        Args:
            image_source: Path to image file (str or Path) or numpy array (BGR or grayscale)
            
        Returns:
            List of TextDetection objects with normalized bbox format [x1, y1, x2, y2]
        """
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Initialize OCR model (lazy loading allowed)."""
        pass
    
    @staticmethod
    def _save_temp_array(image_array: np.ndarray) -> Path:
        """Save numpy array to temporary file for backends that require file paths.
        
        Args:
            image_array: Image as numpy array
            
        Returns:
            Path to temporary file (caller responsible for cleanup)
        """
        import tempfile
        import cv2
        
        fd, temp_path = tempfile.mkstemp(suffix='.png', prefix='ocr_')
        try:
            cv2.imwrite(temp_path, image_array)
            return Path(temp_path)
        except Exception as e:
            logger.error(f"Failed to save temporary image: {e}")
            raise


class EasyOCRBackend(OCRBackend):
    """EasyOCR backend (legacy, kept for backwards compatibility)."""

    def __init__(self, config):
        """Initialize EasyOCR backend.
        
        Args:
            config: OCRConfig instance
        """
        self.config = config
        self.reader = None
        self.initialized = False

    def initialize(self) -> None:
        """Lazy-load EasyOCR model on first use."""
        if self.initialized:
            return
        try:
            import easyocr
            self.reader = easyocr.Reader(
                self.config.languages,
                gpu=(self.config.device == "cuda")
            )
            self.initialized = True
            logger.info(f"EasyOCR initialized on device: {self.config.device}")
        except ImportError:
            logger.error("EasyOCR not installed. Install via: pip install easyocr")
            raise

    def extract_text(self, image_source: Union[str, Path, np.ndarray]) -> List[TextDetection]:
        """Extract text using EasyOCR.
        
        Args:
            image_source: Path to image file or numpy array (BGR or grayscale)
            
        Returns:
            List of TextDetection objects normalized to [x1, y1, x2, y2] bbox
        """
        self.initialize()
        
        # Handle numpy array input by converting to path
        temp_path = None
        if isinstance(image_source, np.ndarray):
            temp_path = self._save_temp_array(image_source)
            image_input = str(temp_path)
        else:
            image_input = str(image_source)
        
        try:
            # EasyOCR readtext returns: List[(pts, text, confidence)]
            # where pts = [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] (quadrilateral)
            results = self.reader.readtext(image_input)
            
            detections = []
            for pts, text, confidence in results:
                # Convert quadrilateral to bbox
                quad = np.array(pts, dtype=np.float32)
                x1 = np.min(quad[:, 0])
                y1 = np.min(quad[:, 1])
                x2 = np.max(quad[:, 0])
                y2 = np.max(quad[:, 1])
                
                detection = TextDetection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    text=text.strip(),
                    confidence=float(confidence),
                    quadrilateral=[(float(p[0]), float(p[1])) for p in pts]
                )
                detections.append(detection)
            
            return detections
        finally:
            # Clean up temporary file if created
            if temp_path:
                import os
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass


class PaddleOCRBackend(OCRBackend):
    """PaddleOCR backend (recommended for document OCR)."""

    def __init__(self, config):
        """Initialize PaddleOCR backend.

        Args:
            config: OCRConfig instance
        """
        self.config = config
        self.ocr = None
        self.initialized = False
        # Whether the live model is on GPU. Tracked so a cuDNN/CUDA failure that
        # only surfaces at inference time (PaddleOCR loads its predictors lazily,
        # so init can succeed and the first .ocr() call is where cuDNN is really
        # touched) can trigger a one-time rebuild on CPU. See extract_text.
        self._use_gpu = False

    @staticmethod
    def _is_cuda_error(exc: Exception) -> bool:
        """True if the exception is a GPU/cuDNN driver-loading failure.

        These are recoverable by rebuilding the model on CPU. Any other error is
        a real fault and must propagate.
        """
        msg = str(exc).lower()
        return 'cudnn' in msg or 'cuda' in msg

    def _build_paddle(self, use_gpu: bool):
        """Construct a PaddleOCR predictor. Single source of constructor args so
        the GPU and CPU paths cannot drift apart.
        """
        from paddleocr import PaddleOCR
        return PaddleOCR(
            use_angle_cls=True,  # Enable text line orientation classification
            lang='en' if 'en' in self.config.languages else 'ch',
            use_gpu=use_gpu,
            det_limit_side_len=self._det_limit,
            det_limit_type='max',
        )

    def _rebuild_on_cpu(self, reason: str) -> None:
        """Swap the live predictor for a CPU one after a GPU/cuDNN failure."""
        logger.warning(f"{reason}. Falling back to CPU mode.")
        self.ocr = self._build_paddle(use_gpu=False)
        self._use_gpu = False
        logger.info(
            f"PaddleOCR now on device: CPU (fallback; det_limit_side_len={self._det_limit})"
        )

    def initialize(self) -> None:
        """Lazy-load PaddleOCR model on first use.

        Uses PaddleOCR 2.x API with use_gpu parameter for device control.
        Falls back to CPU if GPU initialization fails (e.g., cuDNN not installed).
        """
        if self.initialized:
            return
        try:
            # Detection input-size limit. 4608 recovers small in-plan labels
            # (measured: 960→0 unit-labels, 4608→72 unit-labels on 9600px pages).
            # CPU timing measured at 28s/page for ~546 boxes — acceptable.
            # No per-CPU cap: use the configured value on both GPU and CPU.
            self._det_limit = getattr(self.config, "det_limit_side_len", 4608)

            # PaddleOCR 2.x: use_gpu parameter controls device. Try GPU first if
            # configured for CUDA; a cuDNN failure here falls back to CPU.
            want_gpu = (self.config.device == "cuda")
            try:
                self.ocr = self._build_paddle(use_gpu=want_gpu)
                self._use_gpu = want_gpu
                logger.info(
                    f"PaddleOCR initialized on device: {'GPU' if want_gpu else 'CPU'} "
                    f"(det_limit_side_len={self._det_limit})"
                )
            except RuntimeError as e:
                if want_gpu and self._is_cuda_error(e):
                    self._rebuild_on_cpu(
                        f"GPU initialization failed (cuDNN not found or incompatible): {e}"
                    )
                else:
                    raise
            self.initialized = True
        except ImportError:
            logger.error(
                "PaddleOCR not installed. Install via: "
                "pip install paddlepaddle-gpu paddleocr"
            )
            raise

    def _invoke_paddle(self, image_input: str):
        """Call PaddleOCR, tolerating API drift in the `cls` parameter."""
        try:
            return self.ocr.ocr(image_input, cls=True)
        except TypeError:
            # Newer paddleocr versions don't accept cls parameter
            return self.ocr.ocr(image_input)

    def _run_ocr(self, image_input: str):
        """Run inference, recovering from a GPU/cuDNN failure that only surfaces
        here (predictors are loaded lazily on the first call). On such a failure
        while on GPU, rebuild once on CPU and retry; the result is identical
        because the CPU and GPU models share weights.
        """
        try:
            return self._invoke_paddle(image_input)
        except RuntimeError as e:
            if self._use_gpu and self._is_cuda_error(e):
                self._rebuild_on_cpu(f"GPU inference failed (cuDNN/CUDA): {e}")
                return self._invoke_paddle(image_input)
            raise

    def extract_text(self, image_source: Union[str, Path, np.ndarray]) -> List[TextDetection]:
        """Extract text using PaddleOCR.
        
        PaddleOCR returns: List[[[x,y], [x,y], [x,y], [x,y]], (text, confidence)]]
        where the inner lists represent a quadrilateral (4 corner points).
        
        Args:
            image_source: Path to image file or numpy array (BGR or grayscale)
            
        Returns:
            List of TextDetection objects normalized to [x1, y1, x2, y2] bbox
        """
        self.initialize()
        
        # Handle numpy array input by converting to path
        temp_path = None
        if isinstance(image_source, np.ndarray):
            temp_path = self._save_temp_array(image_source)
            image_input = str(temp_path)
        else:
            image_input = str(image_source)
        
        try:
            results = self._run_ocr(image_input)

            detections = []
            # PaddleOCR returns a list of result lines (one per detected region)
            # Each line contains detections
            for line in results:
                if line is None:
                    continue
                for detection in line:
                    quad, (text, confidence) = detection
                    
                    # Convert quadrilateral to bbox
                    quad = np.array(quad, dtype=np.float32)
                    x1 = np.min(quad[:, 0])
                    y1 = np.min(quad[:, 1])
                    x2 = np.max(quad[:, 0])
                    y2 = np.max(quad[:, 1])
                    
                    detection_obj = TextDetection(
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        text=text.strip(),
                        confidence=float(confidence),
                        quadrilateral=[(float(p[0]), float(p[1])) for p in quad]
                    )
                    detections.append(detection_obj)

            logger.info(
                f"PaddleOCR raw detections: {len(detections)} text regions found"
            )
            for det in detections:
                logger.info(
                    f"  RAW OCR | text={det.text!r:30s} conf={det.confidence:.3f} "
                    f"bbox=({det.bbox[0]:.0f},{det.bbox[1]:.0f},{det.bbox[2]:.0f},{det.bbox[3]:.0f})"
                )

            return detections
        finally:
            # Clean up temporary file if created
            if temp_path:
                import os
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass


class OCRFactory:
    """Factory for creating OCR backend instances."""

    _backends = {
        "easyocr": EasyOCRBackend,
        "paddleocr": PaddleOCRBackend,
    }

    @classmethod
    def create(cls, config) -> OCRBackend:
        """Create OCR backend instance.
        
        Args:
            config: OCRConfig instance with backend field
            
        Returns:
            OCRBackend instance
            
        Raises:
            ValueError: If backend is not supported
        """
        backend_name = config.backend.lower()
        if backend_name not in cls._backends:
            raise ValueError(
                f"Unknown OCR backend: {backend_name}. "
                f"Available: {', '.join(cls._backends.keys())}"
            )
        
        backend_class = cls._backends[backend_name]
        logger.info(f"Creating OCR backend: {backend_name}")
        return backend_class(config)

    @classmethod
    def register(cls, name: str, backend_class: type) -> None:
        """Register a new OCR backend.
        
        Args:
            name: Backend identifier (e.g., 'custom_ocr')
            backend_class: Subclass of OCRBackend
        """
        cls._backends[name] = backend_class
        logger.info(f"Registered OCR backend: {name}")
