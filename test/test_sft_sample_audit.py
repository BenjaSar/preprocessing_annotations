"""The per-export sample audit: a seeded, magnified, reviewable sample.

What matters is that the crop shows the target the model is trained on --
the same tile image, the same box -- and that the same dataset always audits
the same targets, so two reviews can be compared.
"""

import json
from pathlib import Path

from PIL import Image

from preprocessing_annotations.config import SFTDatasetConfig
from preprocessing_annotations.export.sft_sample_audit import AUDIT_SUBDIR, write_sample_audit

TILE = (400, 300)
BOX = [100, 80, 180, 120]


def _example(dataset: Path, name: str, targets, tile=TILE) -> dict:
    image = f"images/{name}.png"
    (dataset / "images").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", tile, "white").save(dataset / image)
    return {
        "messages": [
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps(targets)}]},
        ],
        "metadata": {
            "image": image, "source_page": f"/runs/{name}.png",
            "crop": [0, 0, tile[0], tile[1]], "tile_size": list(tile), "rooms": len(targets),
        },
    }


def _target(name="OFFICE 1", room_type="PRIVATE OFFICE", bbox=None):
    return {"room_name": name, "room_type": room_type,
            "bbox": list(bbox or BOX), "confidence": 0.9}


def _dataset(tmp_path, count, targets_per_example=1):
    dataset = tmp_path / "dataset"
    return dataset, [
        ("train", _example(dataset, f"tile{i}",
                           [_target(f"ROOM {i}-{j}") for j in range(targets_per_example)]))
        for i in range(count)
    ]


def test_every_sampled_target_is_written_and_listed(tmp_path):
    dataset, examples = _dataset(tmp_path, 3)

    block = write_sample_audit(examples, dataset, SFTDatasetConfig())

    assert block["size"] == 3 and block["targets_available"] == 3
    for item in block["items"]:
        assert (dataset / item["crop"]).exists()
        assert item["crop"].startswith(f"{AUDIT_SUBDIR}/")
    assert {item["room_name"] for item in block["items"]} == {"ROOM 0-0", "ROOM 1-0", "ROOM 2-0"}


def test_the_sample_is_capped_and_seeded(tmp_path):
    dataset, examples = _dataset(tmp_path, 10)
    config = SFTDatasetConfig(audit_sample_size=4)

    first = write_sample_audit(examples, dataset, config)
    second = write_sample_audit(examples, dataset, config)
    other_seed = write_sample_audit(
        examples, dataset, SFTDatasetConfig(audit_sample_size=4, audit_sample_seed=1)
    )

    names = [item["room_name"] for item in first["items"]]
    assert len(names) == 4 and first["targets_available"] == 10
    assert names == [item["room_name"] for item in second["items"]]
    assert names != [item["room_name"] for item in other_seed["items"]]


def test_crops_are_magnified_by_at_least_the_configured_scale(tmp_path):
    dataset, examples = _dataset(tmp_path, 1)
    config = SFTDatasetConfig(audit_min_scale=3.0, audit_context_frac=0.0)

    block = write_sample_audit(examples, dataset, config)

    with Image.open(dataset / block["items"][0]["crop"]) as crop:
        assert crop.width == int((BOX[2] - BOX[0]) * config.audit_min_scale)
        assert crop.height == int((BOX[3] - BOX[1]) * config.audit_min_scale)


def test_context_is_clipped_to_the_tile(tmp_path):
    """A box at the tile edge must not ask for pixels the tile does not have."""
    dataset = tmp_path / "dataset"
    edge = [0, 0, 60, 40]
    examples = [("train", _example(dataset, "edge", [_target(bbox=edge)]))]

    block = write_sample_audit(examples, dataset, SFTDatasetConfig(audit_context_frac=5.0))

    with Image.open(dataset / block["items"][0]["crop"]) as crop:
        assert crop.width <= TILE[0] * SFTDatasetConfig().audit_min_scale
        assert crop.height <= TILE[1] * SFTDatasetConfig().audit_min_scale


def test_an_empty_dataset_audits_nothing(tmp_path):
    block = write_sample_audit([], tmp_path / "dataset", SFTDatasetConfig())

    assert block == {"size": 0, "targets_available": 0,
                     "seed": SFTDatasetConfig().audit_sample_seed, "items": []}
    assert not (tmp_path / "dataset" / AUDIT_SUBDIR).exists()
