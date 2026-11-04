"""Dense-tile probe: room detection on a document's densest GT tiles.

Asks whether the Rockaway collapse (66 GT rooms in one tile, 1 detected by
the adapter) is the fine-tune or the base model. Selects the K tiles of each
requested document holding the most GT rooms -- same grid the pipeline
detects on (TileSplitter.for_page) -- and runs room detection on each under
every (prompt, seed) pair, with or without an adapter. Each record keeps the
raw model answer beside the parsed rooms, so a guard discarding a long answer
is told apart from a model that wrote a short one.

A GT room belongs to a tile when its polygon's centroid lies inside it. Each
detection is assigned to the smallest GT room containing its center -- the
eval's own rule (room_gt_eval.room_gt_index) -- so nested GT polygons are not
counted twice.

Holds the pipeline's single-instance lock: the VLM fits the GPU only once.
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Sequence

from PIL import Image
from shapely.affinity import translate
from shapely.geometry import Polygon, box

from preprocessing_annotations.config import PipelineConfig, RoomGtEvalConfig
from preprocessing_annotations.ingestion.tile_splitter import TileMeta, TileSplitter
from preprocessing_annotations.orchestration.run_lock import single_instance
from preprocessing_annotations.vlm.vlm_backend import UnslothQwenBackend

from report_io import emit_report
from room_gt_eval import load_gt_polygons, page_key, room_gt_index

logger = logging.getLogger(__name__)


class DenseTile(NamedTuple):
    page: str
    image_path: Path
    meta: TileMeta
    gt: List[Polygon]  # tile-local coordinates


class RecordingBackend(UnslothQwenBackend):
    """UnslothQwenBackend that keeps the last raw answer it decoded."""

    last_response: str = ""

    def _decode_generated(self, output_ids: Any, input_len: int) -> str:
        self.last_response = super()._decode_generated(output_ids, input_len)
        return self.last_response


def densest_tiles(
    images_dir: Path, gt: Dict[str, List[Polygon]], document: str,
    top: int, config: PipelineConfig,
) -> List[DenseTile]:
    """The `top` tiles of `document`'s GT pages holding the most GT rooms."""
    tiles: List[DenseTile] = []
    for image_path in sorted(images_dir.glob("*.png")):
        key = page_key(image_path.name)
        if not key.startswith(document):
            continue
        with Image.open(image_path) as image:
            splitter = TileSplitter.for_page(image.width, image.height, config)
            if splitter is None:
                continue
            metas = [meta for _, meta in splitter.split(image)]
        for meta in metas:
            x0, y0 = meta.x_offset, meta.y_offset
            area = box(x0, y0, x0 + meta.tile_w, y0 + meta.tile_h)
            inside = [
                translate(polygon, -x0, -y0)
                for polygon in gt.get(key, []) if area.contains(polygon.centroid)
            ]
            tiles.append(DenseTile(key, image_path, meta, inside))
    return sorted(tiles, key=lambda tile: len(tile.gt), reverse=True)[:top]


def localization(rooms: Sequence[Dict[str, Any]], gt: List[Polygon]) -> Dict[str, int]:
    """room_gt_eval's counts for one tile: GT rooms located, and detections
    whose center lies in a GT room (the precision numerator)."""
    boxes = [tuple(r["bbox"]) for r in rooms if len(r.get("bbox") or []) == 4]
    matched = [i for i in room_gt_index(boxes, gt) if i is not None]
    return {"located": len(set(matched)), "center_in_gt": len(matched)}


def probe_tile(
    backend: RecordingBackend, tile: DenseTile, prompts: Sequence[str], seeds: Sequence[int],
) -> Dict[str, Any]:
    meta = tile.meta
    with Image.open(tile.image_path) as image:
        crop = image.convert("RGB").crop(
            (meta.x_offset, meta.y_offset, meta.x_offset + meta.tile_w, meta.y_offset + meta.tile_h)
        )
    runs = []
    for prompt in prompts:
        for seed in seeds:
            backend.config.unsloth_room_prompt = prompt
            backend.config.seed = seed
            backend.last_response = ""
            rooms = backend.detect_rooms_from_image(crop)
            runs.append({
                "prompt": prompt,
                "seed": seed,
                "detected": len(rooms),
                **localization(rooms, tile.gt),
                "names": [r.get("room_name") for r in rooms],
                "bboxes": [r.get("bbox") for r in rooms],
                "raw_response": backend.last_response,
            })
            logger.info(
                "%s (%d,%d) prompt=%s seed=%d: %d detected (%d in GT), %d/%d located",
                tile.page, meta.col, meta.row, prompt, seed, runs[-1]["detected"],
                runs[-1]["center_in_gt"], runs[-1]["located"], len(tile.gt),
            )
    return {
        "page": tile.page,
        "tile": [meta.col, meta.row],
        "size": [meta.tile_w, meta.tile_h],
        "gt_rooms": len(tile.gt),
        "runs": runs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Room detection on each document's densest GT tiles")
    parser.add_argument("--documents", nargs="+", required=True, help="page_key prefixes, e.g. 326_rockaway")
    parser.add_argument("--top", type=int, required=True, help="Densest tiles probed per document")
    parser.add_argument("--prompts", nargs="+", required=True, help="Room prompt names (ROOM_PROMPT_BUILDERS)")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--adapter", type=Path, default=None, help="LoRA adapter; omit for the base model")
    parser.add_argument("--model", default=None, help="Unsloth model key (default: VLMConfig's)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("preprocessing_annotations").setLevel(logging.WARNING)
    pipeline_config = PipelineConfig()
    vlm_config = pipeline_config.vlm
    vlm_config.unsloth_adapter_path = args.adapter
    if args.model is not None:
        vlm_config.unsloth_model = args.model

    gt_config = RoomGtEvalConfig()
    gt = load_gt_polygons(gt_config.gt_coco_path)
    images_dir = gt_config.gt_coco_path.parent / "images"
    tiles = [
        tile for document in args.documents
        for tile in densest_tiles(images_dir, gt, document, args.top, pipeline_config)
    ]
    backend = RecordingBackend(vlm_config)
    with single_instance():
        results = [probe_tile(backend, tile, args.prompts, args.seeds) for tile in tiles]
    emit_report(
        {
            "model": vlm_config.unsloth_model,
            "adapter": str(args.adapter) if args.adapter else None,
            "do_sample": vlm_config.do_sample,
            "temperature": vlm_config.temperature,
            "room_token_budget": vlm_config.room_token_budget(),
            "tiles": results,
        },
        args.out,
    )
