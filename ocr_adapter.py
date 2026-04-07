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

    def initialize(self) -> None:
        """Lazy-load PaddleOCR model on first use."""
        if self.initialized:
            return
        try:
            from paddleocr import PaddleOCR
            
            # Initialize PaddleOCR with minimal parameters (latest versions are strict about params)
            # Only use parameters that are widely supported
            # device: "cpu" or "gpu" - force to CPU if device is "cpu" to save VRAM for VLM
            use_gpu = self.config.device == "cuda"
            
            self.ocr = PaddleOCR(
                use_angle_cls=True,  # Enable rotated text detection
                lang='en' if 'en' in self.config.languages else 'ch',
                use_gpu=use_gpu,  # Explicit device control
            )
            
            self.initialized = True
            logger.info(f"PaddleOCR initialized on device: {'GPU' if use_gpu else 'CPU'} (config.device={self.config.device})")
        except ImportError:
            logger.error(
                "PaddleOCR not installed. Install via: "
                "pip install paddlepaddle paddleocr"
            )
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
            # Try with cls parameter first (older versions), then without (newer versions)
            try:
                results = self.ocr.ocr(image_input, cls=True)
            except TypeError:
                # Newer paddleocr versions don't accept cls parameter
                results = self.ocr.ocr(image_input)
            
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
