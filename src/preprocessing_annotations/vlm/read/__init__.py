"""Crop text reading -- port + backend adapter."""

from .port import CropTextReader
from .unsloth_adapter import UnslothCropTextReader

__all__ = ["CropTextReader", "UnslothCropTextReader"]
