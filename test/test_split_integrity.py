"""eval/split_integrity.py catches the two ways one drawing reaches two
splits -- a family the group map merges, and the same drawing shipped twice
under different names -- and --fix re-splits without a rerun."""

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "eval"))

from preprocessing_annotations.config import PipelineConfig, SFTDatasetConfig
from preprocessing_annotations.export.group_split import GroupMap

from split_integrity import check, resplit


def _page_image(seed: int) -> Image.Image:
    """A page whose pixels depend on seed, so two seeds are not duplicates."""
    image = Image.new("L", (64, 64), "white")
    for x in range(0, 64, 4):
        for y in range(0, 64, 4):
            image.putpixel((x, y), (seed * 37 + x * 3 + y * 7) % 256)
    return image.convert("RGB")


def _run_dir(tmp_path, documents, splits):
    """A finished-run layout: processed_annotations/, images/, coco/.

    documents: {document: pixel seed} -- equal seeds mean identical pages.
    splits:    {split: [document, ...]}
    """
    run = tmp_path / "run"
    (run / "processed_annotations").mkdir(parents=True)
    (run / "images").mkdir()
    (run / "coco").mkdir()
    for document, seed in documents.items():
        stem = f"{document}_page000"
        _page_image(seed).save(run / "images" / f"{stem}.png")
        (run / "processed_annotations" / f"{stem}.json").write_text(json.dumps({
            "image_file": f"{stem}.png",
            "image_size": {"width": 64, "height": 64},
            "sft_ready": True,
            "rooms": [
                {"room_name": "OFFICE", "type": "PRIVATE OFFICE", "bbox": [4, 4, 10, 10]},
                {"room_name": "TOILET", "type": "TOILET", "bbox": [20, 20, 10, 10]},
                {"room_name": "CORRIDOR", "type": "CORRIDOR", "bbox": [40, 4, 10, 10]},
            ],
        }))
    for split, names in splits.items():
        (run / "coco" / f"{split}.json").write_text(json.dumps({
            "images": [{"id": i, "file_name": f"{n}_page000.png"} for i, n in enumerate(names)],
            "annotations": [],
            "categories": [],
        }))
    return run


@pytest.fixture
def config():
    return SFTDatasetConfig()


def test_clean_run_passes(tmp_path, config):
    run = _run_dir(tmp_path, {"alpha": 1, "beta": 2},
                   {"train": ["alpha"], "test": ["beta"]})

    report = check(run, config, GroupMap())

    assert report["ok"]
    assert report["straddling_groups"] == []
    assert report["duplicate_documents"] == []


def test_family_straddling_splits_is_caught(tmp_path, config):
    run = _run_dir(tmp_path, {"alpha_layered": 1, "alpha_flat": 2},
                   {"train": ["alpha_layered"], "val": ["alpha_flat"]})
    group_map = GroupMap(groups={"alpha_layered": "alpha", "alpha_flat": "alpha"})

    report = check(run, config, group_map)

    assert not report["ok"]
    assert [g["group"] for g in report["straddling_groups"]] == ["alpha"]


def test_duplicate_drawing_the_map_does_not_name_is_caught(tmp_path, config):
    """Same pixels, different document names, no family entry -- the group
    map cannot see this, the raster can."""
    run = _run_dir(tmp_path, {"alpha": 1, "alpha_copy": 1},
                   {"train": ["alpha"], "test": ["alpha_copy"]})

    report = check(run, config, GroupMap())

    assert not report["ok"]
    assert report["straddling_groups"] == []
    duplicate = report["duplicate_documents"][0]
    assert duplicate["documents"] == ["alpha", "alpha_copy"]
    assert duplicate["min_abs_diff"] < config.duplicate_page_max_abs_diff


def test_duplicates_inside_one_split_are_not_a_leak(tmp_path, config):
    run = _run_dir(tmp_path, {"alpha": 1, "alpha_copy": 1},
                   {"train": ["alpha", "alpha_copy"]})

    assert check(run, config, GroupMap())["ok"]


def test_resplit_puts_a_named_family_back_together(tmp_path, config):
    run = _run_dir(tmp_path, {"alpha_layered": 1, "alpha_flat": 2},
                   {"train": ["alpha_layered"], "val": ["alpha_flat"]})
    group_map = GroupMap(groups={"alpha_layered": "alpha", "alpha_flat": "alpha"})
    assert not check(run, config, group_map)["ok"]

    resplit(run, PipelineConfig(), group_map)

    assert check(run, config, group_map)["ok"]
