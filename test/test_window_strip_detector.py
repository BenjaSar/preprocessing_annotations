#!/usr/bin/env python3
"""
Window detector (Prototype, 2026-08-22, corrected same day after
self-audit): two-stage geometric detection for the window symbols YOLO
misses on sprint1_verify48 (0 shipped windows across every sampled page
despite windows being visible on 2/2 checked buildings).

Stage 1 (find_wall_bands) finds WALLS, not windows -- a wall in plan
view is two parallel lines, and that shape alone can't distinguish a
window break from a plain wall segment (both are two parallel lines).
This is a corrected understanding from a first version of this module
that called stage 1's output "window candidates" -- audited against a
real page and found it swallows all 3 known GT windows into one
1246px-wide wall-band result, 28-33x wider than any window.

Stage 2 (find_window_openings, via _interior_runs) is the actual window
discriminator: a window is ink INSIDE the wall cavity; a plain wall has
none. Measured on the real corpus: window interior coverage 0.553-0.583,
plain-wall interior coverage 0.000-0.031 -- clean separation.

Contract this file locks down:
  1. Stage 1: a single straight line is NOT a wall band (needs 2 faces).
  2. Stage 1: a double-line run (either orientation) IS a wall band --
     this is necessary scaffolding for stage 2, not a window verdict.
  3. Stage 2: a wall band with NO interior ink yields no window openings
     (plain wall, the common case -- most wall length has no window).
  4. Stage 2: a wall band WITH interior ink above threshold yields a
     window opening at the right location.
  5. Stage 2: two adjacent short interior-ink runs (mullion-split single
     window, matching the real Kennedy measurement) merge into one
     opening rather than shipping as two.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing_annotations.detection.window_strip_detector import (
    WindowStripDetectorConfig,
    find_wall_bands,
    find_window_openings,
)


def _blank(h=100, w=200):
    return np.full((h, w), 255, dtype=np.uint8)


def _draw_hline(img, y, x0, x1, thickness=2, value=0):
    img[y:y + thickness, x0:x1] = value


def test_single_wall_line_is_not_a_wall_band():
    img = _blank()
    _draw_hline(img, y=50, x0=20, x1=120, thickness=2)

    bands = find_wall_bands(img)

    assert bands == [], f"a single stroke has no second face, must not be a wall band, got {bands}"


def test_double_line_run_is_a_wall_band():
    """Two parallel strokes = a wall's two faces. This is scaffolding
    for stage 2, NOT a window verdict on its own."""
    img = _blank()
    _draw_hline(img, y=48, x0=20, x1=120, thickness=2)
    _draw_hline(img, y=56, x0=20, x1=120, thickness=2)

    bands = find_wall_bands(img)

    assert len(bands) >= 1
    x, y, w, h = bands[0]
    assert w >= 90


def test_plain_wall_with_no_interior_ink_has_no_window_openings():
    """The common case: most of a wall's length has no window. Interior
    (between the two face strokes) is blank."""
    img = _blank()
    _draw_hline(img, y=48, x0=20, x1=120, thickness=2)
    _draw_hline(img, y=56, x0=20, x1=120, thickness=2)
    # interior (rows 51-53) intentionally left blank -- no window

    windows = find_window_openings(img)

    assert windows == [], f"a plain wall interior must yield no window openings, got {windows}"


def test_wall_with_interior_ink_yields_a_window_opening():
    """A window: ink inside the wall cavity, at the measured real
    coverage level (interior rows mostly filled, not just edge pixels).

    Wall span (250px) is wide relative to the window (30px, 12% of
    span) -- matching real corpus proportions (Kennedy: ~40px window in
    a 1246px clustered wall band, ~3%). Stage 1's own double-line gap
    check reads coverage over the WHOLE clustered band; a window that
    is too large a fraction of a narrow band pushes gap-row coverage
    above band_frac_threshold and stage 1 misreads the band as having
    no gap at all (found while writing this test on a 100px wall with a
    30px window, 30% -- stage 1 returned zero bands). This proportional
    coupling between stage 1 and stage 2 is a known limitation, not
    fixed here -- see module docstring's Status section."""
    img = _blank(w=300)
    _draw_hline(img, y=48, x0=20, x1=270, thickness=2)  # top face
    _draw_hline(img, y=56, x0=20, x1=270, thickness=2)  # bottom face
    # interior ink only in x=60-90 -- the window opening
    img[50:55, 60:90] = 0

    windows = find_window_openings(img)

    assert len(windows) >= 1, "interior ink above threshold must produce a window opening"
    found = [w for w in windows if w[0] <= 75 <= w[0] + w[2]]
    assert found, f"opening must be located near x=60-90, got {windows}"


def test_adjacent_mullion_split_openings_merge_into_one():
    """Matches the real measurement: the Kennedy 1530-1574 GT window
    recovered as two adjacent interior-ink runs 1px apart -- these must
    ship as one opening, not two."""
    img = _blank(w=800)
    _draw_hline(img, y=48, x0=20, x1=770, thickness=2)
    _draw_hline(img, y=56, x0=20, x1=770, thickness=2)
    # two interior-ink runs separated by a 1px gap (a mullion bar) --
    # kept a small fraction of the wall span, see the proportional-
    # coupling note in test_wall_with_interior_ink_yields_a_window_opening
    img[50:55, 60:100] = 0
    img[50:55, 101:140] = 0

    cfg = WindowStripDetectorConfig(min_opening_px=20, opening_merge_gap_px=10)
    windows = find_window_openings(img, cfg)

    spanning = [w for w in windows if w[0] <= 60 and w[0] + w[2] >= 140]
    assert spanning, f"two mullion-split runs 1px apart must merge into one opening, got {windows}"


def test_blank_image_returns_no_wall_bands_or_windows():
    img = _blank()

    assert find_wall_bands(img) == []
    assert find_window_openings(img) == []
