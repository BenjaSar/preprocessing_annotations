"""Red-ink mask (Prototype, 2026-08-22).

Gap addressed: on sprint1_verify48's AVI-ON lighting-plan sheets, a red
demolition/exclusion "X" mark was found drawn directly across a room
label -- pixel-verified on BP-1 Masonic Heights Elementary - AVI-ON
LAYOUT FLAT_page000.png, "CLASSROOM 3 / B106", the X's diagonal strokes
cross the label glyphs. These sheets are flattened PDF exports ("FLAT"
naming) -- original CAD layer separation (which would let the
demolition-marks layer be hidden) no longer exists; the mark is baked
into the same raster every downstream consumer (OCR/VLM/SAM) reads.

Measured, not assumed: red pixels (R>150, R-G>50, R-B>50) are 0.057% of
the Masonic page and 0.031% of a Kennedy Electrical page -- rare, and
cleanly separated from near-black architectural/text ink (sampled
~(29,29,29) on the same pages). Masking red-to-white on the Masonic
crop removed the X completely; the label became fully legible; no
black content was altered (verified by re-inspecting the cleaned crop).

Known scope decision, NOT resolved here: the same red channel also
carries real content on Electrical-series sheets -- circuit-trace
wiring (seen on Kennedy p001). Masking erases both indiscriminately.
Fine for this pipeline's current consumers (room/space/door/window
detection -- none of them use circuit-trace data), but this is a
one-way scope decision: an excluded room's "not in scope" marking is
also erased, with no metadata trail. If exclusion status or circuit
data ever matters downstream, this function must not be applied
unconditionally -- see the module-level TODO in its docstring.

Status: PROTOTYPE. Validated on 2 real pages (both the removal-quality
check and the red/black color separation). NOT run across the full
48-page corpus -- unknown what fraction of pages/rooms this actually
affects, and the exclusion-semantics scope decision above is
unresolved. Not wired into the pipeline.
"""

from typing import Tuple

import numpy as np
from PIL import Image


def mask_red_ink(
    image: Image.Image,
    red_min: int = 150,
    red_excess: int = 50,
    fill: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Replace red-channel-dominant pixels with `fill` (white by default).

    A pixel counts as "red ink" when its red channel exceeds `red_min`
    AND exceeds both green and blue by at least `red_excess` -- the
    same predicate measured against this corpus's demolition-mark and
    circuit-trace content (pure-red samples read ~(255,0,0) to
    ~(255,89,89); real black wall/label ink reads ~(29,29,29), nowhere
    close to this predicate).

    Returns a new RGB image; does not modify the input in place.
    """
    rgb = image.convert("RGB")
    arr = np.array(rgb)
    r = arr[..., 0].astype(int)
    g = arr[..., 1].astype(int)
    b = arr[..., 2].astype(int)
    red_mask = (r > red_min) & (r - g > red_excess) & (r - b > red_excess)

    cleaned = arr.copy()
    cleaned[red_mask] = fill
    return Image.fromarray(cleaned)
