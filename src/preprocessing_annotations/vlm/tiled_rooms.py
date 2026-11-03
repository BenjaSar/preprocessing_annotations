"""Room detection over a page's tiles, in full-page coordinates.

One place for split -> detect -> rescale -> hallucination check -> NMS merge,
so every caller sees the same geometry. The pipeline's STEP 3 detects this way;
measured 2026-10-29 on sprint1_verify55, a whole-page call does not: of the 712
Pass-2 candidates whose name also appears as an OCR token, 1 had that token
inside its box, against 1773 of 1773 for Pass-1 candidates, and no uniform
rescale recovered them (best factor: 73 of 712). The grid itself comes from
TileSplitter.for_page, which is also what the SFT exporter tiles with.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from PIL import Image

from ..ingestion.tile_splitter import TileSplitter

logger = logging.getLogger(__name__)

# NMS and the room cap run once on the merged full-page set, never per tile.
# The cap is 40, not TileSplitter.merge's default: dense residential floors
# carry 30+ units, and a smaller cap discards real rooms that finer adaptive
# tiling just recovered.
MERGE_IOU_THRESHOLD = 0.30
MERGE_MAX_ROOMS = 40


class RoomDetectingBackend(Protocol):
    """The part of a VLM backend tiled detection needs -- nothing else."""

    def detect_rooms_from_image(self, image: Image.Image) -> List[Dict[str, Any]]: ...

    def detect_hallucinations(
        self, rooms: List[Dict[str, Any]], check_stripes: bool = True
    ) -> List[Dict[str, Any]]: ...


def detect_rooms_in_tiles(
    backend: RoomDetectingBackend,
    image: Image.Image,
    splitter: TileSplitter,
    source_name: str = "",
) -> List[Dict[str, Any]]:
    """Detect over `splitter`'s tiles and return rooms in full-image coords.

    A tile that fails is skipped, not fatal: the rest of the page still
    yields rooms.
    """
    tiles = splitter.split(image)
    logger.info(
        f"{source_name}: tiling {image.width}x{image.height} -> {len(tiles)} tiles "
        f"({splitter.cols}x{splitter.rows}, overlap={splitter.overlap_pct:.0%})"
    )

    detected: List[Dict[str, Any]] = []
    for tile_image, meta in tiles:
        try:
            tile_rooms = splitter.rescale_rooms(
                backend.detect_rooms_from_image(tile_image), meta
            )
        except Exception as error:  # one bad tile must not lose the page
            logger.warning(f"  Tile ({meta.col},{meta.row}) failed: {error} — skipping")
            continue
        logger.debug(f"  Tile ({meta.col},{meta.row}): {len(tile_rooms)} rooms detected")
        detected.extend(tile_rooms)

    checked = backend.detect_hallucinations(detected)
    merged = splitter.merge(
        checked, iou_threshold=MERGE_IOU_THRESHOLD, max_rooms=MERGE_MAX_ROOMS
    )
    logger.info(f"{source_name}: tiling complete — {len(detected)} raw -> {len(merged)} after NMS")
    return merged


def splitter_for_image(
    image_path: Path, config: Any
) -> Optional[Tuple[Image.Image, TileSplitter]]:
    """(opened image, splitter) for this page, or None to detect it whole.

    None covers every reason not to tile -- tiling off, page below the
    trigger, image unreadable -- so a caller has one branch, not three.
    """
    if not getattr(config, "use_tiling", False):
        return None
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as error:
        logger.warning(f"Tiling: could not open {Path(image_path).name}: {error}. Falling back.")
        return None
    splitter = TileSplitter.for_page(image.width, image.height, config)
    if splitter is None:
        logger.debug(
            f"Tiling skipped: {image.width}x{image.height} <= trigger {config.tile_trigger_px}px"
        )
        image.close()
        return None
    return image, splitter
