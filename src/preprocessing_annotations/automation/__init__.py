"""
Automation module for annotation preprocessing.

Provides tools for:
- Label normalization (standardize room type labels)
- Quality checking (validate annotation consistency)
- Region extraction (crop room regions for training)
- SFT annotation schema (mandatory output format for VLM fine-tuning)
"""

from .label_normalizer import LabelNormalizer, normalize_label
from .quality_checker import QualityChecker, ValidationIssue
from .region_extractor import RegionExtractor
from .sft_validator import (
    ImageResizer,
    SemanticRoomValidator,
    TaxonomyNormalizer,
    filter_by_confidence,
    validate_for_sft,
    prepare_sft_annotation,
)
from .annotation_schema import (
    SFTRoom,
    SFTAnnotationBuilder,
    NameUniquifier,
    ConfidenceComputer,
    build_annotation_json,
)

__all__ = [
    "LabelNormalizer",
    "normalize_label",
    "QualityChecker",
    "ValidationIssue",
    "RegionExtractor",
    "ImageResizer",
    "SemanticRoomValidator",
    "TaxonomyNormalizer",
    "filter_by_confidence",
    "validate_for_sft",
    "prepare_sft_annotation",
    "SFTRoom",
    "SFTAnnotationBuilder",
    "NameUniquifier",
    "ConfidenceComputer",
    "build_annotation_json",
]
