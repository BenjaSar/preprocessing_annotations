"""Crop text reading -- port + backend adapter."""

from .port import CropTextReader, RoomLabelLocator
from .unsloth_adapter import UnslothCropTextReader, UnslothRoomLabelLocator

__all__ = [
    "CropTextReader", "RoomLabelLocator",
    "UnslothCropTextReader", "UnslothRoomLabelLocator",
]
