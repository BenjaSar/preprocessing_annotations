"""
Tile-based image splitting for VLM inference on large floor plans.

Splits a PIL image into overlapping tiles, rescales tile-local bboxes back to
full-image coordinates, then deduplicates with IoU-based NMS.

Usage (from pipeline):
    splitter = TileSplitter(cols=2, rows=2, overlap_pct=0.10)
    tiles = splitter.split(pil_image)
    all_rooms = []
    for tile_img, meta in tiles:
        rooms = backend.detect_rooms_from_image(tile_img)
        rooms = splitter.rescale_rooms(rooms, meta)
        all_rooms.extend(rooms)
    merged = splitter.merge(all_rooms, iou_threshold=0.30, max_rooms=25)
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class TileMeta:
    col: int
    row: int
    x_offset: int   # pixels from full-image left edge
    y_offset: int   # pixels from full-image top edge
    tile_w: int
    tile_h: int
    full_w: int
    full_h: int


class TileSplitter:
    """Split -> detect -> rescale -> NMS-merge for tiled VLM inference."""

    def __init__(self, cols: int = 2, rows: int = 2, overlap_pct: float = 0.10) -> None:
        if cols < 1 or rows < 1:
            raise ValueError("cols and rows must be >= 1")
        if not (0.0 <= overlap_pct < 0.5):
            raise ValueError("overlap_pct must be in [0, 0.5)")
        self.cols = cols
        self.rows = rows
        self.overlap_pct = overlap_pct

    def split(self, image: Image.Image) -> List[Tuple[Image.Image, TileMeta]]:
        """Return (tile_img, TileMeta) list, left-to-right top-to-bottom."""
        full_w, full_h = image.size
        base_w = full_w / self.cols
        base_h = full_h / self.rows
        ovlp_x = int(base_w * self.overlap_pct)
        ovlp_y = int(base_h * self.overlap_pct)
        tiles: List[Tuple[Image.Image, TileMeta]] = []
        for row in range(self.rows):
            for col in range(self.cols):
                # stride = base minus overlap so adjacent tiles share ovlp pixels
                x0 = col * int(base_w - ovlp_x)
                y0 = row * int(base_h - ovlp_y)
                x1 = min(x0 + int(base_w) + ovlp_x, full_w)
                y1 = min(y0 + int(base_h) + ovlp_y, full_h)
                meta = TileMeta(
                    col=col, row=row,
                    x_offset=x0, y_offset=y0,
                    tile_w=x1 - x0, tile_h=y1 - y0,
                    full_w=full_w, full_h=full_h,
                )
                tiles.append((image.crop((x0, y0, x1, y1)), meta))
                logger.debug(
                    f"Tile ({col},{row}): [{x0},{y0},{x1},{y1}] {x1-x0}x{y1-y0}px"
                )
        return tiles

    @staticmethod
    def rescale_rooms(
        rooms: List[Dict[str, Any]], meta: TileMeta
    ) -> List[Dict[str, Any]]:
        """Translate tile-local xyxy bboxes to full-image xyxy coordinates."""
        out = []
        for room in rooms:
            r = dict(room)
            bbox = r.get("bbox")
            if bbox and len(bbox) == 4:
                tx1, ty1, tx2, ty2 = [float(v) for v in bbox]
                r["bbox"] = [
                    int(tx1 + meta.x_offset),
                    int(ty1 + meta.y_offset),
                    int(tx2 + meta.x_offset),
                    int(ty2 + meta.y_offset),
                ]
                r["_tile"] = (meta.col, meta.row)
            out.append(r)
        return out

    def merge(
        self,
        rooms: List[Dict[str, Any]],
        iou_threshold: float = 0.30,
        max_rooms: int = 25,
    ) -> List[Dict[str, Any]]:
        """NMS dedup across all tiles. Global 25-room cap applied post-merge
        (not per-tile) so the limit reflects the full-image result.
        """
        if not rooms:
            return []
        # Higher confidence wins when overlap detected
        by_conf = sorted(
            rooms, key=lambda r: float(r.get("confidence", 0.5)), reverse=True
        )
        kept: List[Dict[str, Any]] = []
        for candidate in by_conf:
            bc = candidate.get("bbox", [])
            if len(bc) != 4:
                kept.append(candidate)
                continue
            if not any(
                len(k.get("bbox", [])) == 4 and _iou(bc, k["bbox"]) >= iou_threshold
                for k in kept
            ):
                kept.append(candidate)
        if len(kept) > max_rooms:
            logger.warning(
                f"merge: {len(kept)} rooms exceeds cap ({max_rooms}), trimming"
            )
            kept = kept[:max_rooms]
        for r in kept:
            r.pop("_tile", None)
        logger.info(
            f"merge: {len(rooms)} raw -> {len(kept)} after NMS (iou={iou_threshold})"
        )
        return kept


def _iou(a: List, b: List) -> float:
    """IoU for two xyxy bboxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (
        max(0, ax2 - ax1) * max(0, ay2 - ay1)
        + max(0, bx2 - bx1) * max(0, by2 - by1)
        - inter
    )
    return inter / union if union > 0 else 0.0
