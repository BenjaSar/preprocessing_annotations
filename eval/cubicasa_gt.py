"""T-CC0: CubiCasa5K ground truth adapter (room + wall localization).

Wraps gt_evaluator.load_ground_truth with a path resolver for CubiCasa5K's
COCO export, whose file_name values all share the SAME basename
("F1_original.png" for every one of the 400 test images) — the default
basename resolver in gt_evaluator would collide every image onto one file,
so this module supplies a resolver that preserves the relative folder path
instead (which is where each image's real identity lives).

Scope (see CubicasaEvalConfig docstring): bbox localization only, generic
"room"/"wall" classes, residential domain. Not a source of room-type or
commercial-domain ground truth.
"""

import logging
from pathlib import Path
from typing import List, Optional

from preprocessing_annotations.config import CubicasaEvalConfig
from gt_evaluator import GTImage, load_ground_truth

logger = logging.getLogger(__name__)


def resolve_cubicasa_path(
    file_name: str, images_root: Path, kaggle_path_prefix: str
) -> Optional[Path]:
    """Resolve a CubiCasa5K file_name to a real path under `images_root`.

    Strips the export tool's kaggle-notebook path prefix and joins the
    remaining relative path (which includes the per-image folder that makes
    each file unique) onto `images_root`.

    Args:
        file_name: Raw COCO file_name, e.g.
            "/kaggle/input/cubicasa5k/cubicasa5k/cubicasa5k/high_quality_
            architectural/1191/F1_original.png".
        images_root: Local directory the stripped relative path resolves under.
        kaggle_path_prefix: Prefix to strip from `file_name`.

    Returns:
        The resolved path, or None if it does not exist on disk.
    """
    if kaggle_path_prefix not in file_name:
        logger.warning("Unexpected file_name format, skipping: %s", file_name)
        return None
    relative_path = file_name.split(kaggle_path_prefix, 1)[1]
    resolved = images_root / relative_path
    return resolved if resolved.exists() else None


def load_cubicasa_ground_truth(
    config: CubicasaEvalConfig, coco_path: Path
) -> List[GTImage]:
    """Load CubiCasa5K room+wall GT from one COCO split.

    Args:
        config: CubicasaEvalConfig supplying the resolver's fixed inputs.
        coco_path: Path to a CubiCasa5K COCO split (train/val/test).

    Returns:
        One GTImage per resolvable image, with wall/room annotations.
    """
    def resolve(file_name: str, images_dir: Path) -> Optional[Path]:
        return resolve_cubicasa_path(
            file_name, images_dir, config.kaggle_path_prefix
        )

    return load_ground_truth(
        coco_path,
        config.images_root,
        config.gt_category_merge,
        resolve_path=resolve,
    )
