"""Page-image lookup in a finished run directory.

A run keeps each page's image in images/ or, for a page it skipped, in
skipped_pages/; the raw annotation records the file name in both cases.
Lookup is by that exact name, never a glob: page stems can hold glob
metacharacters ("S2-EC-22698778-PlanSet BP AP[74]_page000").
"""

from pathlib import Path
from typing import Any, Mapping

_PAGE_IMAGE_DIRS = ("images", "skipped_pages")


def find_page_image(run_dir: Path, annotation: Mapping[str, Any]) -> Path:
    """Path of the image a raw annotation was made from."""
    image_file = annotation["image_file"]
    for folder in _PAGE_IMAGE_DIRS:
        path = run_dir / folder / image_file
        if path.exists():
            return path
    raise FileNotFoundError(f"{image_file}: not in {' or '.join(_PAGE_IMAGE_DIRS)} of {run_dir}")
