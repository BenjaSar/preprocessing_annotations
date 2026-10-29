#!/usr/bin/env python3
"""
AV-5 WallBandTool Tests.

Covers:
  - bands() on a blank image returns no bands (no wall geometry present)
  - bands() on a synthetic double-line wall returns >=1 band (same
    fixture shape test_window_strip_detector.py already validates
    find_wall_bands against directly -- this only checks the tool's
    thin wrapper, not the detection algorithm itself)
  - bands() accepts an RGB array (converts to grayscale) with the same
    result as passing the pre-converted grayscale array
  - persist() writes a PNG file and returns its path
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing_annotations.tools.wallband_tool import WallBandTool


def _blank(h=100, w=200):
    return np.full((h, w), 255, dtype=np.uint8)


def _draw_horizontal_wall(gray, y=40, x0=20, x1=180, gap=8, stroke=3):
    """Two parallel horizontal dark lines -- the double-line wall shape
    find_wall_bands looks for (same construction as
    test_window_strip_detector.py's fixtures)."""
    img = gray.copy()
    img[y:y + stroke, x0:x1] = 0
    img[y + stroke + gap:y + 2 * stroke + gap, x0:x1] = 0
    return img


def test_bands_empty_on_blank_image():
    tool = WallBandTool()
    assert tool.bands(_blank()) == []


def test_bands_finds_double_line_wall():
    gray = _draw_horizontal_wall(_blank())
    tool = WallBandTool()
    bands = tool.bands(gray)
    assert len(bands) >= 1


def test_bands_accepts_rgb_array():
    gray = _draw_horizontal_wall(_blank())
    rgb = np.stack([gray, gray, gray], axis=-1)
    tool = WallBandTool()
    assert tool.bands(rgb) == tool.bands(gray)


def test_persist_writes_png(tmp_path):
    gray = _draw_horizontal_wall(_blank())
    tool = WallBandTool()
    out_path = tmp_path / "overlay.png"
    result = tool.persist(gray, out_path)
    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0
