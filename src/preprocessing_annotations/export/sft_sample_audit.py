"""A reviewable sample of an SFT dataset's targets (SFTDatasetConfig audit_*).

The training corpora carry no room ground truth, so nothing scores a target's
label or its box there. What is left is looking at them. This writes a seeded
sample of targets as magnified crops, each with its box drawn and its answer
recorded, so a dataset ships with a sample someone can actually review.

Crops come from the tile images the exporter has already written, so what the
audit shows is exactly what the model is trained on -- not a re-render of the
page that could differ from it.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw

from ..config.config import SFTDatasetConfig

# Audit crops live beside the JSONL splits, under the dataset directory.
AUDIT_SUBDIR = "audit"

Box = Tuple[float, float, float, float]


def _targets_of(example: dict) -> List[dict]:
    """The rooms this example asks the model to answer with."""
    return json.loads(example["messages"][-1]["content"][0]["text"])


def _window(box: Box, tile: Tuple[int, int], context_frac: float) -> Box:
    """The box plus context, clipped to the tile it was cropped from."""
    margin = context_frac * max(box[2] - box[0], box[3] - box[1])
    return (
        max(0.0, box[0] - margin), max(0.0, box[1] - margin),
        min(float(tile[0]), box[2] + margin), min(float(tile[1]), box[3] + margin),
    )


def _write_crop(
    tile_image: Path, box: Box, window: Box, scale: float, destination: Path
) -> None:
    with Image.open(tile_image) as tile:
        crop = tile.convert("RGB").crop(tuple(int(v) for v in window))
    crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))))
    ImageDraw.Draw(crop).rectangle(
        [(box[0] - window[0]) * scale, (box[1] - window[1]) * scale,
         (box[2] - window[0]) * scale, (box[3] - window[1]) * scale],
        outline=(255, 0, 0), width=3,
    )
    crop.save(destination)


def write_sample_audit(
    examples: Sequence[Tuple[str, dict]], output_dir: Path, config: SFTDatasetConfig
) -> Dict[str, Any]:
    """Write the audit crops and return the block for export_report.json.

    `examples` are (split, example) pairs. The sample is seeded, so the same
    dataset always audits the same targets.
    """
    pairs = [
        (split, example, target)
        for split, example in examples
        for target in _targets_of(example)
    ]
    if not pairs or config.audit_sample_size <= 0:
        return {"size": 0, "targets_available": len(pairs),
                "seed": config.audit_sample_seed, "items": []}

    if len(pairs) <= config.audit_sample_size:
        sample = list(pairs)
    else:
        sample = random.Random(config.audit_sample_seed).sample(pairs, config.audit_sample_size)

    audit_dir = output_dir / AUDIT_SUBDIR
    audit_dir.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, Any]] = []
    for index, (split, example, target) in enumerate(sample):
        metadata = example["metadata"]
        tile = tuple(metadata["tile_size"])
        box = tuple(float(v) for v in target["bbox"])
        window = _window(box, tile, config.audit_context_frac)
        crop_name = f"{AUDIT_SUBDIR}/{index:03d}.png"
        _write_crop(
            output_dir / metadata["image"], box, window,
            config.audit_min_scale, output_dir / crop_name,
        )
        items.append({
            "id": index,
            "split": split,
            "crop": crop_name,
            "tile_image": metadata["image"],
            "source_page": metadata["source_page"],
            "room_name": target["room_name"],
            "room_type": target["room_type"],
            "bbox": target["bbox"],
        })
    return {
        "size": len(items),
        "targets_available": len(pairs),
        "seed": config.audit_sample_seed,
        "min_scale": config.audit_min_scale,
        "context_frac": config.audit_context_frac,
        "items": items,
    }
