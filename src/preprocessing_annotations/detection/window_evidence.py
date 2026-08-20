"""BA1/BA1b: local wall-context evidence for door-mislabeled-as-window.

Option B (2026-08-03) -- replaces the envelope-distance route
(envelope_extractor.py), killed by its own verification gate: 6/10
test_pcs sheets have no single building envelope (site plans, multi-
unit part-plan sheets, enlarged-unit details), so "distance to
envelope" isn't defined there. These features are LOCAL to each
detection's crop -- no page-level geometry, so sheet type cannot
break them the way it broke the envelope route.

One feature shipped: elongation (BA1b, undirected max(w,h)/min(w,h)),
BA0-validated on GT crops (test_pcs .910 acc, real threshold-sweep
classification accuracy, not median-gap alone) and BA3-verified on 10
pixel-confirmed real ROCKAWAY boxes (10/10 correct, this project's
actual input, not GT-only). Fixes a portrait/vertical-window
regression raw w/h had -- portrait boxes are 37-47% of all boxes and
w/h alone scored only .656/.623 there.

Two other candidates were measured and dropped, not silently: arc
evidence (HoughCircles) scored EXACTLY at the majority-class baseline
on both GT sets (zero signal). Line-pair evidence (glazing double-line
Hough signature, test_pcs .894 acc on GT) passed BA0 but failed BA2 on
real yolo_finetuned_tiled predictions -- collapses kaggle window
recall .562->.288 (77 TPs lost) chasing 1 confused box -- and BA3
missed 2/4 real ROCKAWAY windows. Removed as dead code once zero
callers remained (no combination step ever shipped); its measured
numbers stay in WindowEvidenceConfig's docstring, not reimplemented
here on a hunch.

Pure function only: bbox -> feature -> predicate. No pipeline I/O
here -- filtering objectDetections is sft_validator.py's job.
"""

from typing import List, Optional

from ..config import WindowEvidenceConfig


def elongation(bbox: List[float]) -> Optional[float]:
    """xyxy bbox max(w,h)/min(w,h) -- undirected, so a vertical
    (portrait) window scores the same as an equally-elongated
    horizontal one. BA1b: raw w/h scored only .656/.623 accuracy on
    portrait boxes (37-47% of all boxes, both GT sets) because a tall
    window's w/h sits near a door's; this fixes that without touching
    doors' separability (measured, not assumed -- doors are NOT near
    1.0 either way, median elong 1.22/1.18).

    None if either side is <= 0 (degenerate box).
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    return max(w, h) / min(w, h)


def is_window_shaped(bbox: List[float], config: Optional[WindowEvidenceConfig] = None) -> Optional[bool]:
    """Elongation predicate: True if the box is elongated (either
    orientation) beyond the BA1b-measured threshold. None if the box
    is degenerate. Bbox-only, no image I/O -- cheap enough for the
    production annotation-filtering path.
    """
    cfg = config or WindowEvidenceConfig()
    e = elongation(bbox)
    if e is None:
        return None
    return e > cfg.elongation_threshold
