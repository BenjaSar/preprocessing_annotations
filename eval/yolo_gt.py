"""T-KG0: floor-plans-500 (Kaggle, YOLOv11) ground-truth adapter.

Converts the Roboflow YOLOv11 export (door/window/zone bboxes) into this
project's GTImage format, so gt_evaluator.score_image scores detections
against it unchanged -- the same pattern cubicasa_gt.py uses for the
COCO-format CubiCasa5K GT. See KaggleFloorplanEvalConfig for scope,
license, and verified coverage.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from PIL import Image, UnidentifiedImageError

from preprocessing_annotations.bbox.bbox_metrics import BBox
from preprocessing_annotations.config import KaggleFloorplanEvalConfig
from gt_evaluator import GTAnnotation, GTImage

logger = logging.getLogger(__name__)

_DATA_YAML_NAME = "data.yaml"
_IMAGES_DIRNAME = "images"
_LABELS_DIRNAME = "labels"
_LABEL_SUFFIX = ".txt"
_YOLO_BOX_FIELD_COUNT = 5  # class_id cx cy w h, normalized [0, 1]


def _load_class_names(dataset_root: Path) -> Dict[int, str]:
    """Load the class-index -> name map from the dataset's data.yaml."""
    data_yaml = dataset_root / _DATA_YAML_NAME
    data = yaml.safe_load(data_yaml.read_text())
    return dict(enumerate(data["names"]))


def _find_image_for_label(
    label_path: Path, images_dir: Path
) -> Optional[Path]:
    """Find the image matching a label's stem, regardless of extension."""
    matches = list(images_dir.glob(f"{label_path.stem}.*"))
    if not matches:
        logger.warning("No image found for label: %s", label_path.name)
        return None
    return matches[0]


def _to_pixel_bbox(
    cx: float, cy: float, w: float, h: float, width: int, height: int
) -> BBox:
    """Convert normalized YOLO cxcywh to a pixel-space xyxy BBox."""
    return BBox(
        x1=(cx - w / 2) * width,
        y1=(cy - h / 2) * height,
        x2=(cx + w / 2) * width,
        y2=(cy + h / 2) * height,
    )


def _parse_yolo_line(
    line: str,
    class_names: Dict[int, str],
    category_merge: Dict[str, str],
    width: int,
    height: int,
) -> Optional[GTAnnotation]:
    """Parse one YOLO label line into a GTAnnotation, or None if unscored."""
    fields = line.split()
    if len(fields) != _YOLO_BOX_FIELD_COUNT:
        logger.warning("Malformed YOLO line, skipping: %r", line)
        return None
    class_id, cx, cy, w, h = fields
    try:
        name = class_names.get(int(class_id))
        box = _to_pixel_bbox(
            float(cx), float(cy), float(w), float(h), width, height
        )
    except ValueError as error:
        logger.warning("Unparsable YOLO line %r: %s", line, error)
        return None
    merged = category_merge.get(name)
    if merged is None:
        return None
    return GTAnnotation(bbox=box, category=merged)


def _parse_label_file(
    label_path: Path,
    class_names: Dict[int, str],
    category_merge: Dict[str, str],
    width: int,
    height: int,
) -> List[GTAnnotation]:
    """Parse every non-blank line of a YOLO label file into annotations."""
    lines = (l for l in label_path.read_text().splitlines() if l.strip())
    return [
        ann for ann in (
            _parse_yolo_line(line, class_names, category_merge, width,
                              height)
            for line in lines
        )
        if ann is not None
    ]


def _read_image_size(image_path: Path) -> Optional[Tuple[int, int]]:
    """Return (width, height), or None if the image can't be read."""
    try:
        with Image.open(image_path) as img:
            return img.size
    except UnidentifiedImageError as error:
        logger.warning("Unreadable image %s: %s", image_path.name, error)
        return None


def _build_gt_image(
    label_path: Path,
    images_dir: Path,
    class_names: Dict[int, str],
    category_merge: Dict[str, str],
) -> Optional[GTImage]:
    """Build one GTImage from a YOLO label file, or None if unresolvable."""
    image_path = _find_image_for_label(label_path, images_dir)
    if image_path is None:
        return None
    size = _read_image_size(image_path)
    if size is None:
        return None
    width, height = size
    annotations = _parse_label_file(
        label_path, class_names, category_merge, width, height
    )
    return GTImage(
        image_path=image_path, width=width, height=height,
        annotations=annotations,
    )


def load_kaggle_floorplan_ground_truth(
    config: KaggleFloorplanEvalConfig,
) -> List[GTImage]:
    """Load the Kaggle floor-plans-500 split as GT images.

    Args:
        config: KaggleFloorplanEvalConfig selecting the split and the
            scored category map.

    Returns:
        One GTImage per resolvable label file in the configured split.
    """
    split_dir = config.dataset_root / config.split
    images_dir = split_dir / _IMAGES_DIRNAME
    labels_dir = split_dir / _LABELS_DIRNAME
    class_names = _load_class_names(config.dataset_root)
    built = (
        _build_gt_image(
            label_path, images_dir, class_names, config.gt_category_merge
        )
        for label_path in sorted(labels_dir.glob(f"*{_LABEL_SUFFIX}"))
    )
    return [gt_image for gt_image in built if gt_image is not None]
