#!/usr/bin/env python3
"""
Red-ink mask (Prototype, 2026-08-22): removes red demolition-mark /
circuit-trace ink from sprint1_verify48 sheets before OCR/VLM/SAM see
the page. Motivating case: a red "X" crosses a room label
(CLASSROOM 3 / B106) on a flattened AVI-ON lighting-plan sheet with no
way to hide it (original CAD layer separation is gone).

Contract this file locks down:
  1. Pure red content is masked to the fill color.
  2. Near-black wall/text ink is untouched -- this is the whole
     precision argument (red and black are cleanly separated in this
     corpus, measured ~(255,0,0) vs ~(29,29,29)).
  3. True-white background is untouched (no-op on blank regions).
  4. A borderline reddish-brown (not a clean red) below the red_excess
     margin is NOT masked -- the predicate requires red to dominate
     green/blue by a real margin, not just be the largest channel.
  5. Output is a new image; input is not mutated.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing_annotations.ingestion.red_ink_mask import mask_red_ink


def _solid(color, size=(20, 20)):
    return Image.new("RGB", size, color)


def test_pure_red_is_masked_to_white():
    img = _solid((255, 0, 0))

    out = mask_red_ink(img)

    assert np.array(out).tolist() == np.array(_solid((255, 255, 255))).tolist()


def test_near_black_ink_is_untouched():
    img = _solid((29, 29, 29))

    out = mask_red_ink(img)

    assert np.array(out).tolist() == np.array(img).tolist()


def test_white_background_is_untouched():
    img = _solid((255, 255, 255))

    out = mask_red_ink(img)

    assert np.array(out).tolist() == np.array(img).tolist()


def test_borderline_reddish_brown_is_not_masked():
    """(180, 150, 140): red is the largest channel but only exceeds
    green/blue by 30/40 -- below the default red_excess=50 margin.
    Must not be treated as demolition-mark/circuit-trace red."""
    img = _solid((180, 150, 140))

    out = mask_red_ink(img)

    assert np.array(out).tolist() == np.array(img).tolist()


def test_mixed_image_only_red_region_changes():
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    arr[:, :5] = [29, 29, 29]     # black content, left half
    arr[:, 5:] = [255, 0, 0]      # red ink, right half
    img = Image.fromarray(arr)

    out = np.array(mask_red_ink(img))

    assert out[:, :5].tolist() == arr[:, :5].tolist(), "black half must be unchanged"
    assert (out[:, 5:] == 255).all(), "red half must become white"


def test_input_image_not_mutated():
    img = _solid((255, 0, 0))
    original = np.array(img).copy()

    mask_red_ink(img)

    assert np.array(img).tolist() == original.tolist()
