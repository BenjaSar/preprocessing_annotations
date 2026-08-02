"""
Example configurations for the preprocessing annotations pipeline.

Copy and modify this file for your specific use case.
"""

from preprocessing_annotations.config import PipelineConfig, PDFConfig, OCRConfig, VLMConfig, SAMConfig


def get_config_for_small_symbols():
    """
    Configuration for floor plans with small electrical symbols.

    Use when you need to capture small details like receptacles and switches.
    """
    config = PipelineConfig()

    # High DPI for detail
    config.pdf.dpi = 300

    # More aggressive OCR preprocessing
    config.ocr.confidence_threshold = 0.4
    config.ocr.preprocess = True
    config.ocr.clahe_clip_limit = 3.0

    # More template scales and rotations
    config.template.scales = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    config.template.rotations = [0, 45, 90, 135, 180, 225, 270, 315]

    # Enable all features
    config.use_vlm = True
    config.use_sam = True
    config.use_template_matching = True

    return config


def get_config_for_batch_processing():
    """
    Configuration for processing large batches quickly.

    Use when processing many floor plans and speed is more important than precision.
    """
    config = PipelineConfig()

    # Lower DPI for speed
    config.pdf.dpi = 150

    # Minimal preprocessing
    config.ocr.confidence_threshold = 0.6
    config.ocr.preprocess = False

    # Fewer template scales, no rotation
    config.template.scales = [0.8, 1.0, 1.2]
    config.template.rotations = [0]

    # VLM only, skip SAM and templates
    config.use_vlm = True
    config.use_sam = False
    config.use_template_matching = False

    # Parallel processing
    config.num_workers = 8

    return config


def get_config_for_ocr_only():
    """
    Configuration for OCR-only annotation (no VLM).

    Use for quick validation or when Claude API is unavailable.
    """
    config = PipelineConfig()

    # Standard DPI
    config.pdf.dpi = 200

    # Good OCR settings
    config.ocr.confidence_threshold = 0.5
    config.ocr.preprocess = True

    # No VLM, SAM, or templates
    config.use_vlm = False
    config.use_sam = False
    config.use_template_matching = False

    return config


def get_config_for_dense_plans():
    """
    Configuration for very dense floor plans with many rooms.

    Use when plans have many small rooms and complex layouts.
    """
    config = PipelineConfig()

    # High DPI and aggressive OCR
    config.pdf.dpi = 300
    config.ocr.confidence_threshold = 0.3  # Lower threshold to catch more
    config.ocr.preprocess = True
    config.ocr.clahe_clip_limit = 4.0  # More aggressive contrast

    # Extended room name patterns for special cases
    config.ocr.room_name_patterns.extend([
        r"CLOSET",
        r"PANTRY",
        r"LAUNDRY",
        r"FOYER",
        r"VESTIBULE",
        r"UTILITY",
        r"STORAGE",
    ])

    # VLM with SAM refinement
    config.use_vlm = True
    config.use_sam = True

    return config


def get_config_for_architectural_validation():
    """
    Configuration for high-quality validation of architectural plans.

    Use when absolute precision is required.
    """
    config = PipelineConfig()

    # Maximum quality
    config.pdf.dpi = 400
    config.ocr.confidence_threshold = 0.7  # High threshold for precision
    config.ocr.preprocess = True
    config.ocr.clahe_clip_limit = 2.0

    # Maximum template matching precision
    config.template.scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    config.template.rotations = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    config.template.threshold = 0.85  # Higher threshold

    # All features enabled
    config.use_vlm = True
    config.use_sam = True
    config.use_template_matching = True

    # Claude Opus for best results
    config.vlm.model = "claude-opus-4-20250805"
    config.vlm.max_tokens = 8192

    return config


# Quick access
CONFIGS = {
    "small_symbols": get_config_for_small_symbols,
    "batch": get_config_for_batch_processing,
    "ocr_only": get_config_for_ocr_only,
    "dense": get_config_for_dense_plans,
    "validation": get_config_for_architectural_validation,
}


if __name__ == "__main__":
    # Example: use a specific configuration
    config = get_config_for_small_symbols()

    from preprocessing_annotations.orchestration.pipeline import AnnotationPipeline

    pipeline = AnnotationPipeline(config)
    stats = pipeline.run(
        input_dir="../floorPlanVisionAIAdaptor/data",
        output_dir="./results"
    )

    print(f"Processed {stats['images_extracted']} images")
    print(f"Detected {stats['rooms_detected']} rooms")
