"""The stored page image is the coordinate space every stage records against.

STEP 2 reads the page at its rendered size and records OCR-derived rooms in
that space. A later stage that rewrites the image file smaller leaves those
rooms pointing at the old space: measured 2026-10-29 on corpus_batch2_v2, two
pages were stored smaller than rendered (1.524x, 1.111x) and their 49 SFT
targets boxed blank wall and notes text. The VLM paths resize in memory and
map their boxes back themselves, so nothing needs the file to shrink.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import preprocessing_annotations.automation as automation


def test_no_in_place_page_image_resizer_is_exposed():
    assert not hasattr(automation, "ImageResizer"), (
        "an in-place page-image resizer is back; it corrupts every coordinate "
        "already recorded against the stored image"
    )
    assert "ImageResizer" not in automation.__all__
