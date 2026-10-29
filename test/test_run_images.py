"""eval/run_images.py: page-image lookup in a run directory."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))

from run_images import find_page_image


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_processed_page_image(tmp_path):
    image = _touch(tmp_path / "images" / "a_page000.png")
    assert find_page_image(tmp_path, {"image_file": "a_page000.png"}) == image


def test_skipped_page_image(tmp_path):
    image = _touch(tmp_path / "skipped_pages" / "a_page000.png")
    assert find_page_image(tmp_path, {"image_file": "a_page000.png"}) == image


def test_glob_metacharacters_are_literal(tmp_path):
    name = "PlanSet BP AP[74]_page000.png"
    image = _touch(tmp_path / "images" / name)
    assert find_page_image(tmp_path, {"image_file": name}) == image


def test_missing_image_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_page_image(tmp_path, {"image_file": "a_page000.png"})
