"""Wall faces from a PDF page's own vector geometry (RoomExtentConfig).

Reads the drawing paths on wall/door/glazing layers, maps them into the
pixel space of the page image the pipeline annotated, and polygonizes them:
each face is a region the drawn walls enclose. Pages without such layers
("FLAT" or scanned PDFs) yield no faces.
"""

import re
from pathlib import Path
from typing import Iterator, List, Tuple

import fitz
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

from ..config import RoomExtentConfig

Segment = Tuple[fitz.Point, fitz.Point]


class VectorWallError(Exception):
    """The page's drawings could not be read (e.g. PyMuPDF's
    "too many nested graphics states")."""


def _item_segments(item: tuple) -> Iterator[Segment]:
    """Straight segments of one path item; a curve contributes its chord."""
    kind = item[0]
    if kind == "l":
        yield item[1], item[2]
    elif kind == "c":
        yield item[1], item[4]
    elif kind == "re":
        r = item[1]
        yield from _closed([r.tl, r.tr, r.br, r.bl])
    elif kind == "qu":
        q = item[1]
        yield from _closed([q.ul, q.ur, q.lr, q.ll])


def _closed(corners: List[fitz.Point]) -> Iterator[Segment]:
    return zip(corners, corners[1:] + corners[:1])


class VectorWallReader:
    """One reason to change: how wall geometry is read from a PDF page."""

    def __init__(self, config: RoomExtentConfig):
        self._layer = re.compile(config.layer_pattern, re.IGNORECASE)
        self._excluded = re.compile(config.excluded_layer_pattern, re.IGNORECASE)

    def _is_wall_layer(self, layer: str) -> bool:
        return bool(self._layer.search(layer)) and not self._excluded.search(layer)

    def _segments(self, page: fitz.Page) -> Iterator[Segment]:
        for drawing in page.get_drawings():
            if self._is_wall_layer(drawing.get("layer") or ""):
                for item in drawing["items"]:
                    yield from _item_segments(item)

    def faces(self, pdf_path: Path, page_index: int, image_size: Tuple[int, int]) -> List[Polygon]:
        """Wall faces in the pixel space of an image of `image_size` (width,
        height) rendered from this page, rotation included."""
        try:
            with fitz.open(pdf_path) as doc:
                page = doc[page_index]
                width, height = image_size
                scale_x, scale_y = width / page.rect.width, height / page.rect.height
                rotation = page.rotation_matrix
                lines = []
                for start, end in self._segments(page):
                    a, b = start * rotation, end * rotation
                    if a != b:
                        lines.append(LineString(
                            [(a.x * scale_x, a.y * scale_y), (b.x * scale_x, b.y * scale_y)]
                        ))
        except RuntimeError as error:
            raise VectorWallError(f"{pdf_path.name} page {page_index}: {error}") from error
        if not lines:
            return []
        return [face for face in polygonize(unary_union(lines)) if face.area > 0]
