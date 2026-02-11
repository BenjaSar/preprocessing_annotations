"""
Automation module for annotation preprocessing.

Provides tools for:
- Label normalization (standardize room type labels)
- Quality checking (validate annotation consistency)
- Region extraction (crop room regions for training)
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
]
