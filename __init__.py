"""
Preprocessing Annotations Package for MEP Floor Plan Analysis.

This package provides tools for extracting, annotating, and processing
MEP (Mechanical, Electrical, Plumbing) floor plans using a combination of:
- PDF extraction
- OCR text detection
- Template-based symbol detection
- VLM (Vision Language Model) annotation
- SAM (Segment Anything Model) segmentation

Usage:
    from preprocessing_annotations import AnnotationPipeline

    pipeline = AnnotationPipeline(config)
    pipeline.run(input_dir="./pdfs", output_dir="./dataset")
"""

from .config import PipelineConfig, OCRConfig, TemplateConfig, VLMConfig, SAMConfig
from .pdf_extractor import PDFExtractor
from .ocr_extractor import MEPTextExtractor, RoomCandidate
from .symbol_detector import SymbolTemplateDetector
from .vlm_annotator import VLMAnnotator
from .sam_segmenter import RoomSegmenter
from .exporters import LabelStudioExporter, ReviewPrioritizer
from .pipeline import AnnotationPipeline

__version__ = "0.2.0"

__all__ = [
    # Configuration
    "PipelineConfig",
    "OCRConfig",
    "TemplateConfig",
    "VLMConfig",
    "SAMConfig",
    # Core components
    "PDFExtractor",
    "MEPTextExtractor",
    "RoomCandidate",
    "SymbolTemplateDetector",
    "VLMAnnotator",
    "RoomSegmenter",
    # Exporters
    "LabelStudioExporter",
    "ReviewPrioritizer",
    # Pipeline
    "AnnotationPipeline",
]
