#!/usr/bin/env python3
"""AV-5 (2026-09-23): spot-check wall-band geometry on a sample of images.

Persists a debug overlay per image so wall-band output (PROTOTYPE
status -- see tools/wallband_tool.py) is finally something a human can
look at, instead of only existing transiently inside
find_window_openings. Does not touch any pipeline run output.

Usage:
    python scripts/persist_wall_bands.py --input /path/to/images_dir --output /tmp/wallbands
"""

import argparse
import sys
from pathlib import Path

# Add src/ to path so the package is importable without an editable install
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from PIL import Image
import numpy as np

from preprocessing_annotations.tools.wallband_tool import WallBandTool

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", required=True, help="Directory of images")
    parser.add_argument("--output", "-o", required=True, help="Directory for overlay PNGs")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N images (sorted by name). Default: all.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    tool = WallBandTool()

    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    if args.limit:
        images = images[: args.limit]

    for img_path in images:
        with Image.open(img_path).convert("RGB") as img:
            arr = np.array(img)
        bands = tool.bands(arr)
        tool.persist(arr, output_dir / f"{img_path.stem}_wallbands.png", bands=bands)
        print(f"{img_path.name}: {len(bands)} wall band(s)")


if __name__ == "__main__":
    main()
