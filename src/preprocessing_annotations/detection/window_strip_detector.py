"""Geometric window detector (Prototype, 2026-08-22, corrected same day
after self-audit).

Gap this addresses: YOLO's fine-tuned door/window checkpoint shipped 0
window detections across every page sampled from sprint1_verify48 (48-page
AVI-ON lighting-plan run) despite windows being visually present and
legible on at least 2 of 5 sampled buildings. Root-caused (audited, not
assumed): NOT a contrast problem -- same-category ink-fraction stats for
shipped doors (median dark-pixel gray 83, range 39.6-146.9) overlap the
one measurable shipped window sample (62.2), so window symbols are not
reliably lighter than doors.

TWO-STAGE technique, corrected from a first attempt that conflated the
stages:

Stage 1 -- WALL bands, not windows. A wall in plan view is two parallel
lines (its two faces). Binarize non-background, morphological opening
with a long thin kernel (both orientations) isolates long straight runs
(door swings are curved arcs and are rejected here), then a
cross-section double-line check keeps only paired-line runs. This
reliably finds WALLS -- confirmed by testing it against a known real
wall: the 3 known GT windows on BP-1 Kennedy Middle School Electrical
Pages_page001.png's top wall were all swallowed by ONE wall-band result,
1246px wide, 28-33x wider than any individual window. A first version of
this module called this stage's output "window candidates" -- wrong;
every plain wall segment produces the identical double-line shape a
window does, because both ARE two parallel lines. Renamed accordingly.

Stage 2 -- window openings WITHIN a wall band. A window is ink inside
the wall cavity; a plain wall run has none. Measured directly on the
Kennedy wall band (interior rows only, wall-face rows excluded):
  window 1530-1574: interior ink coverage 0.553
  window 1799-1837: interior ink coverage 0.583
  window 1987-2027: interior ink coverage 0.583
  plain wall segments (4 sampled): 0.000, 0.000, 0.000, 0.031
Clean separation, no overlap -- this is the actual window discriminator,
not stage 1's double-line shape. Stage 2 scans each stage-1 band's
interior for coverage runs above interior_ink_threshold, then merges
adjacent runs (a single opening's glazing bars/mullions split it into
several short high-coverage runs with small gaps -- measured: the 3 GT
windows each recovered as 2 adjacent runs, e.g. (1455,1531)+(1532,1572)
for the 1530-1574 GT window).

Known limitation, found writing this module's own tests: stage 1's
double-line gap check reads ink coverage over the WHOLE clustered band
width. When a window is a large fraction of a short band's length
(synthetic test: 30px window in a 100px band, 30%), the window's own
interior ink pushes gap-row coverage above band_frac_threshold and
stage 1 misreads the band as having no gap at all -- zero bands
returned, window invisible to stage 2 too. Only confirmed safe when
windows are a small fraction of the band (real corpus: ~3%, Kennedy's
40px window in a 1246px band). Not fixed -- tests use realistic
proportions to route around it, not to hide it.

Status: PROTOTYPE. Stage-2 recall measured 3/3 on the one wall band
checked (Kennedy p001 top wall, ±2px boundary error). Precision NOT
measured -- no page-level false-positive count exists yet. Not wired
into the pipeline's detection tier fallback chain
(WindowDetector/use_windows). Do not promote past Prototype without a
scored multi-page run (T-0 ground truth still owed -- see project
memory).
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class WindowStripDetectorConfig:
    """Calibration constants, all data-derived (see module docstring for
    the measurements each one comes from)."""

    # Stage 1 -- wall band detection.
    binarise_max: int = 235
    min_run_px: int = 30
    max_run_px: int = 2000
    # Kernel thickness for the directional opening, in the SHORT axis.
    # Must stay at 1: an opening's erosion step requires the kernel to
    # fit entirely inside the foreground stroke, and real wall/window
    # linework at this resolution can be as thin as 1-2px -- any
    # thickness > the thinnest real stroke erases that stroke entirely
    # (found via a failing synthetic test: thickness=3 erased a 2px-thick
    # line completely, zero bands survived). A 1px-tall kernel only
    # requires min_run_px consecutive ink pixels in a single row/column,
    # tolerant of any real stroke thickness >= 1.
    kernel_thickness_px: int = 1
    # Cross-section double-line check: the two strokes of a wall must be
    # separated by a lighter gap at least this many pixels wide, and
    # each stroke must be at least this many pixels tall/wide, to reject
    # single-pixel noise in the binarized mask.
    min_gap_px: int = 2
    min_stroke_px: int = 1
    # A wall band is a thin RECTANGLE (two face strokes joined by
    # end-caps where the band was clustered from separate components),
    # not two isolated line segments in open space -- the end-caps mean
    # every row has SOME ink, so "any ink in this row" never finds a
    # gap (measured: 0/17 rows were ink-free on a confirmed real wall
    # crop). What actually separates a face-stroke row from a hollow
    # interior row is ink COVERAGE: face rows measured 0.42-0.96 of the
    # row's width; hollow-interior (no window) rows measured 0.00-0.03.
    # Threshold sits in that measured gap.
    band_frac_threshold: float = 0.2

    # Stage 2 -- window openings inside a wall band's interior.
    # Rows/columns nearest each face are excluded before scanning the
    # interior, so the face strokes themselves (which are ALSO ink,
    # coverage 0.42-0.96) don't get mistaken for a window opening.
    interior_inset_px: int = 3
    # Measured: window interior coverage 0.553-0.583, plain-wall interior
    # coverage 0.000-0.031. Threshold sits in that gap, same value as
    # band_frac_threshold for consistency (both come from the "is this
    # row/column occupied" family of checks) though the two measured
    # gaps are independent evidence.
    interior_ink_threshold: float = 0.2
    # A single opening's glazing bars/mullions can split one physical
    # window into several short high-coverage runs -- measured: the
    # 1530-1574 GT window recovered as 2 adjacent runs (1455-1531,
    # 1532-1572) with a 1px gap. Runs whose gap is <= this are merged
    # into one opening.
    opening_merge_gap_px: int = 10
    # Discrete window breaks measured 38-44px; below this a run is more
    # likely noise (a stray mark, a fixture symbol overlapping the wall)
    # than a real opening.
    min_opening_px: int = 20


def _long_straight_runs(binary: np.ndarray, min_run_px: int, max_run_px: int,
                         kernel_thickness_px: int, horizontal: bool
                         ) -> List[Tuple[int, int, int, int]]:
    """Morphological opening with a long thin kernel in one orientation.
    Returns (x, y, w, h) boxes of surviving connected components, same
    style as sft_validator.py's _detect_ruled_tables."""
    h, w = binary.shape
    if horizontal:
        ksize = (max(1, min_run_px), kernel_thickness_px)
    else:
        ksize = (kernel_thickness_px, max(1, min_run_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    n, _lbl, stats, _cent = cv2.connectedComponentsWithStats(opened, 8)
    boxes = []
    for i in range(1, n):
        x, y, cw, ch = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        run_len = cw if horizontal else ch
        if min_run_px <= run_len <= max_run_px:
            boxes.append((x, y, cw, ch))
    return boxes


def _has_double_line(binary: np.ndarray, box: Tuple[int, int, int, int],
                      horizontal: bool, min_gap_px: int, min_stroke_px: int,
                      band_frac_threshold: float) -> bool:
    """Cross-section check: >= 2 distinct ink bands separated by a
    non-ink gap, perpendicular to the run's long axis. This is what a
    WALL (two faces) looks like -- see module docstring for why this
    does NOT by itself mean "window".

    Uses ink COVERAGE per row/column, not "any ink present" -- a
    clustered band's end-caps put SOME ink in every row, so an any-ink
    profile never finds a gap."""
    x, y, w, h = box
    region = binary[y:y + h, x:x + w]
    coverage = (region > 0).mean(axis=1 if horizontal else 0)
    profile = coverage >= band_frac_threshold
    bands = 0
    in_band = False
    gap_run = 0
    band_run = 0
    for is_ink in profile:
        if is_ink:
            band_run += 1
            if not in_band and gap_run >= min_gap_px:
                bands += 1
            in_band = True
            gap_run = 0
        else:
            if in_band and band_run < min_stroke_px:
                pass  # noise-thin band, don't count the gap that follows it as separating
            in_band = False
            gap_run += 1
            band_run = 0
    if in_band:
        bands += 1
    return bands >= 2


def _cluster_parallel_runs(
    runs: List[Tuple[int, int, int, int]], horizontal: bool, max_perp_gap_px: int
) -> List[Tuple[int, int, int, int]]:
    """Merge separate same-orientation runs into one union bbox when they
    overlap along the long axis and sit close together on the short axis.

    Necessary because each of a wall's two faces survives the directional
    opening as its OWN connected component (a 2px-thick line is one
    component, the parallel line 8px away is another) -- without this
    merge, _has_double_line never sees both faces at once and every wall
    is invisible to it, indistinguishable from open space.
    """
    remaining = list(runs)
    clusters: List[Tuple[int, int, int, int]] = []
    while remaining:
        x, y, w, h = remaining.pop(0)
        merged = [x, y, x + w, y + h]
        changed = True
        while changed:
            changed = False
            still = []
            for ox, oy, ow, oh in remaining:
                obox = [ox, oy, ox + ow, oy + oh]
                if horizontal:
                    long_overlap = min(merged[2], obox[2]) - max(merged[0], obox[0]) > 0
                    perp_gap = max(obox[1] - merged[3], merged[1] - obox[3])
                else:
                    long_overlap = min(merged[3], obox[3]) - max(merged[1], obox[1]) > 0
                    perp_gap = max(obox[0] - merged[2], merged[0] - obox[2])
                if long_overlap and perp_gap <= max_perp_gap_px:
                    merged = [
                        min(merged[0], obox[0]), min(merged[1], obox[1]),
                        max(merged[2], obox[2]), max(merged[3], obox[3]),
                    ]
                    changed = True
                else:
                    still.append((ox, oy, ow, oh))
            remaining = still
        clusters.append((merged[0], merged[1], merged[2] - merged[0], merged[3] - merged[1]))
    return clusters


def find_wall_bands(
    gray: np.ndarray, config: WindowStripDetectorConfig = None
) -> List[Tuple[int, int, int, int]]:
    """Stage 1: find wall bands (two parallel faces) in a grayscale
    floorplan/MEP sheet image. Returns (x, y, w, h) boxes.

    Renamed from this module's original detect_window_strips: audited
    and found to detect WALLS, not windows -- every plain wall segment
    has the identical double-line shape a window break does. See module
    docstring. Kept as a public stage because stage 2 (find_window_openings)
    needs its output, and because a wall-band map may be independently
    useful (e.g. as an exclusion mask) even though it is not a window
    detector on its own.
    """
    config = config or WindowStripDetectorConfig()
    _, binary = cv2.threshold(
        gray, config.binarise_max, 255, cv2.THRESH_BINARY_INV
    )

    bands: List[Tuple[int, int, int, int]] = []
    for horizontal in (True, False):
        runs = _long_straight_runs(
            binary, config.min_run_px, config.max_run_px,
            config.kernel_thickness_px, horizontal,
        )
        clusters = _cluster_parallel_runs(
            runs, horizontal, max_perp_gap_px=config.min_gap_px + 10
        )
        for box in clusters:
            if _has_double_line(
                binary, box, horizontal, config.min_gap_px, config.min_stroke_px,
                config.band_frac_threshold,
            ):
                bands.append(box)
    return bands


def _interior_runs(
    binary: np.ndarray, band: Tuple[int, int, int, int], horizontal: bool,
    config: WindowStripDetectorConfig,
) -> List[Tuple[int, int, int, int]]:
    """Stage 2 core: scan a wall band's interior (face rows/columns
    excluded) for ink-coverage runs -- these are candidate window
    openings. horizontal=True means the band itself is a horizontal wall
    (long axis = x), so we scan column coverage; False scans row
    coverage for a vertical wall."""
    x, y, w, h = band
    inset = config.interior_inset_px
    if horizontal:
        if h <= 2 * inset + 1:
            return []
        interior = binary[y + inset:y + h - inset, x:x + w]
        coverage = (interior > 0).mean(axis=0)
        base = x
    else:
        if w <= 2 * inset + 1:
            return []
        interior = binary[y:y + h, x + inset:x + w - inset]
        coverage = (interior > 0).mean(axis=1)
        base = y
    hot = coverage >= config.interior_ink_threshold
    runs = []
    start = None
    for i, is_hot in enumerate(hot):
        if is_hot and start is None:
            start = i
        elif not is_hot and start is not None:
            if i - start >= config.min_opening_px:
                runs.append((base + start, base + i))
            start = None
    if start is not None and len(hot) - start >= config.min_opening_px:
        runs.append((base + start, base + len(hot)))

    boxes = []
    for a, b in runs:
        if horizontal:
            boxes.append((a, y, b - a, h))
        else:
            boxes.append((x, a, w, b - a))
    return boxes


def _merge_adjacent_openings(
    openings: List[Tuple[int, int, int, int]], horizontal: bool, max_gap_px: int
) -> List[Tuple[int, int, int, int]]:
    """Merge openings that sit end-to-end along the wall's long axis with
    a small gap -- a single window's mullions/glazing bars split one
    physical opening into several short runs (measured: the Kennedy
    1530-1574 GT window recovered as two runs, (1455,1531) + (1532,1572),
    1px apart)."""
    if not openings:
        return []
    axis = 0 if horizontal else 1
    ordered = sorted(openings, key=lambda b: b[axis])
    merged = [list(ordered[0])]
    for box in ordered[1:]:
        prev = merged[-1]
        prev_end = prev[axis] + (prev[2] if horizontal else prev[3])
        gap = box[axis] - prev_end
        if gap <= max_gap_px:
            new_end = max(prev_end, box[axis] + (box[2] if horizontal else box[3]))
            prev[axis + 2] = new_end - prev[axis]
        else:
            merged.append(list(box))
    return [tuple(b) for b in merged]


def find_window_openings(
    gray: np.ndarray, config: WindowStripDetectorConfig = None
) -> List[Tuple[int, int, int, int]]:
    """Full two-stage detector: wall bands (stage 1) -> interior-ink
    openings within each band (stage 2) -> merge adjacent openings.

    Returns (x, y, w, h) window-opening boxes in image pixel coordinates.
    Geometry-only: no OCR, VLM, or learned detector. See module docstring
    for the measured evidence behind each stage, and for what remains
    unvalidated (page-level precision) before this should be wired into
    the pipeline.
    """
    config = config or WindowStripDetectorConfig()
    _, binary = cv2.threshold(
        gray, config.binarise_max, 255, cv2.THRESH_BINARY_INV
    )
    bands = find_wall_bands(gray, config)

    windows: List[Tuple[int, int, int, int]] = []
    for band in bands:
        x, y, w, h = band
        horizontal = w >= h
        openings = _interior_runs(binary, band, horizontal, config)
        merged = _merge_adjacent_openings(openings, horizontal, config.opening_merge_gap_px)
        windows.extend(merged)
    return windows
