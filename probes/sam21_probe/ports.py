"""Port (interface) the bake-off application layer depends on.

Dependency-inversion boundary: bakeoff.py imports only this Protocol, never a
concrete adapter or the sam1/sam2 packages directly. Swapping in a different
point-prompt segmenter means writing a new adapter, zero changes to bakeoff.py.
"""

from typing import Protocol

from .domain import Expansion


class PointPromptExpander(Protocol):
    """A model that expands a label-seed point (+box) into a room boundary.

    Takes image_path (not a decoded array) so each adapter can use its own
    native image-loading/caching path (SAM1's RoomSegmenter already caches by
    path; SAM2 does its own).
    """

    def expand(
        self, image_path: str, point: tuple[int, int], label_bbox: tuple[int, int, int, int]
    ) -> Expansion:
        """Return the room-boundary expansion for one seed."""
        ...
