"""
Configuration classes for the annotation pipeline.

All configurable parameters are centralized here to avoid magic numbers
and enable easy tuning for different floor plan types.

Environment variables:
  ANTHROPIC_API_KEY - Required: Anthropic API key
  VLM_MODEL - Optional: Model to use (default: claude-haiku-4-5-20251001)
  VLM_MAX_TOKENS - Optional: Max response tokens (default: 4096)
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
import torch


def _get_vlm_model() -> str:
    """Get VLM model from environment or use default Haiku."""
    return os.getenv("VLM_MODEL", "claude-haiku-4-5-20251001")


def _detect_device() -> str:
    """Detect available compute device with graceful fallback."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class PDFConfig:
    """Configuration for PDF extraction."""

    # DPI for rasterization. Higher values capture more detail but increase file size.
    # - 150 DPI: Fast processing, suitable for large room labels
    # - 200 DPI: Balanced (default), good for most MEP plans
    # - 300 DPI: High detail, recommended for small electrical symbols
    dpi: int = 200

    # Output image format
    output_format: str = "PNG"

    # Whether to extract all pages or specific page ranges
    page_range: Optional[tuple] = None  # None = all pages


@dataclass
class OCRConfig:
    """Configuration for OCR text extraction."""

    # Languages for EasyOCR
    languages: List[str] = field(default_factory=lambda: ["en"])

    # Device for OCR inference (auto-detected if None)
    device: Optional[str] = None

    # Minimum confidence threshold for OCR results
    confidence_threshold: float = 0.5

    # Whether to apply image preprocessing before OCR
    preprocess: bool = True

    # CLAHE parameters for contrast enhancement
    clahe_clip_limit: float = 2.0
    clahe_grid_size: tuple = (8, 8)

    # Room number pattern (regex)
    room_number_pattern: str = r"^(\d{3}[A-Z]?|\d{2,3})$"

    # Room name patterns to detect
    room_name_patterns: List[str] = field(
        default_factory=lambda: [
            r"(SUITE|OFFICE|ROOM)\s*\d*",
            r"MECHANICAL\s*(ROOM)?",
            r"ELECTRICAL\s*(ROOM)?",
            r"ELEVATOR",
            r"STAIR",
            r"CUSTODIAL",
            r"(MEN|WOMEN|RESTROOM)",
            r"ENTRANCE|ENTRY|LOBBY",
            r"CORRIDOR|HALLWAY",
            r"RISER",
            r"STORAGE",
            r"CONFERENCE",
            r"BREAK\s*ROOM",
            r"KITCHEN",
            r"SERVER\s*ROOM",
            r"IDF|MDF",
        ]
    )

    def __post_init__(self):
        if self.device is None:
            self.device = _detect_device()


@dataclass
class TemplateConfig:
    """Configuration for template-based symbol detection."""

    # Matching threshold (0.0 to 1.0). Higher = stricter matching.
    threshold: float = 0.7

    # Scale factors to try for multi-scale matching
    scales: List[float] = field(default_factory=lambda: [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])

    # Rotation angles to try (degrees) for rotation-invariant matching
    rotations: List[int] = field(default_factory=lambda: [0, 90, 180, 270])

    # Non-Maximum Suppression IoU threshold
    nms_iou_threshold: float = 0.3

    # Template matching method
    # Options: cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF_NORMED
    match_method: int = 5  # cv2.TM_CCOEFF_NORMED


@dataclass
class VLMConfig:
    """Configuration for VLM (Claude) annotation.

    Model can be set via VLM_MODEL environment variable.
    Available models:
      - claude-haiku-4-5-20251001 (default) - Fast, cheap, good for room detection
      - claude-sonnet-4-5-20250929 - Balanced performance and cost
      - claude-opus-4-6 - Most capable (expensive)
    """

    # Model to use for annotation (reads from VLM_MODEL env var)
    model: str = field(default_factory=_get_vlm_model)

    # Maximum tokens for response (reads from VLM_MAX_TOKENS env var)
    max_tokens: int = field(default_factory=lambda: int(os.getenv("VLM_MAX_TOKENS", "4096")))

    # Number of retries for API calls
    max_retries: int = 3

    # Delay between retries (seconds)
    retry_delay: float = 1.0

    # Room categories for classification (CV-focused, standardized taxonomy)
    room_categories: List[str] = field(
        default_factory=lambda: [
            "office",
            "open_plan_workspace",
            "executive_office",
            "cubicle_workstation",
            "conference_room",
            "meeting_room",
            "training_room",
            "breakout_space",
            "lobby_reception",
            "hallway_corridor",
            "restroom",
            "kitchen_break_room",
            "storage",
            "mechanical_room",
            "elevator",
            "stairwell",
            "auditorium",
            "data_center",
            "server_room",
            "other",
        ]
    )


@dataclass
class SAMConfig:
    """Configuration for SAM segmentation."""

    # SAM model variant: vit_h, vit_l, vit_b
    model_type: str = "vit_h"

    # Checkpoint path
    checkpoint: str = "sam_vit_h_4b8939.pth"

    # Device for inference (auto-detected if None)
    device: Optional[str] = None

    # Whether to output multiple masks
    multimask_output: bool = True

    def __post_init__(self):
        if self.device is None:
            self.device = _detect_device()


@dataclass
class ExportConfig:
    """Configuration for annotation export."""

    # Default image dimensions if not found in annotations
    default_width: int = 1000
    default_height: int = 1000

    # Percentage of lowest-confidence images to flag for review
    review_threshold: float = 0.25


@dataclass
class PipelineConfig:
    """Master configuration for the full annotation pipeline."""

    pdf: PDFConfig = field(default_factory=PDFConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    template: TemplateConfig = field(default_factory=TemplateConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    sam: SAMConfig = field(default_factory=SAMConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    # Pipeline options
    # NOTE: use_vlm defaults to False. Use OCR results unless explicitly enabled.
    # This ensures annotations are saved even if VLM API is unavailable.
    use_vlm: bool = False
    use_sam: bool = False
    use_template_matching: bool = False

    # Parallel processing
    num_workers: int = 4

    @classmethod
    def for_high_detail(cls) -> "PipelineConfig":
        """Factory for high-detail extraction (small symbols, dense plans)."""
        config = cls()
        config.pdf.dpi = 300
        config.ocr.preprocess = True
        config.template.scales = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
        return config

    @classmethod
    def for_fast_processing(cls) -> "PipelineConfig":
        """Factory for fast processing (large batches, lower quality OK)."""
        config = cls()
        config.pdf.dpi = 150
        config.ocr.preprocess = False
        config.template.scales = [0.8, 1.0, 1.2]
        config.template.rotations = [0]
        return config
