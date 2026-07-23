"""T-FC2: FloorPlanCAD (HF Voxel51) door/window/wall GT adapter.

Parses the FiftyOne ``samples.json`` of the Voxel51/FloorPlanCAD HF
dataset into this project's GTImage format, so gt_evaluator.score_image
scores detections against it unchanged -- the same pattern
cubicasa_gt.py (COCO) and yolo_gt.py (YOLO) use for their sources.

No FiftyOne dependency: ``samples.json`` holds every annotation
(``ground_truth.detections`` with ``label`` + ``bounding_box``
normalized ``[x, y, w, h]``, plus per-sample ``metadata`` width/height).
Image pixels live under ``data/`` in the same HF snapshot; this loader
parses annotations regardless of whether the PNGs are downloaded (the
scoring step opens images and skips any that are missing). See
FloorplancadEvalConfig for scope, license (CC BY-NC, eval-only), and the
grounded category merge.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from bbox_metrics import BBox
from config import FloorplancadEvalConfig
from gt_evaluator import GTAnnotation, GTImage

logger = logging.getLogger(__name__)

# Image files live under this prefix inside the HF dataset repo.
_IMAGE_DIR_PREFIX = "data"

_GROUND_TRUTH_FIELD = "ground_truth"
_DETECTIONS_FIELD = "detections"
_BBOX_FIELD = "bounding_box"
_LABEL_FIELD = "label"
_FILEPATH_FIELD = "filepath"
_METADATA_FIELD = "metadata"
_XYWH_LEN = 4


def _resolve_samples_path(config: FloorplancadEvalConfig) -> Path:
    """Resolve the cached samples.json from the local HF dataset snapshot.

    Uses ``local_files_only`` -- the snapshot must already be downloaded.

    Raises:
        FileNotFoundError: if samples.json is not in the local HF cache.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:
        cached = hf_hub_download(
            repo_id=config.repo_id,
            filename=config.samples_filename,
            repo_type="dataset",
            local_files_only=True,
        )
    except LocalEntryNotFoundError as error:
        raise FileNotFoundError(
            f"{config.repo_id}/{config.samples_filename} not cached; "
            f"download the dataset snapshot first"
        ) from error
    return Path(cached)


def _to_pixel_bbox(
    xywh: List[float], width: int, height: int
) -> BBox:
    """Convert a normalized [x, y, w, h] box to a pixel-space xyxy BBox."""
    x, y, w, h = xywh
    return BBox(
        x1=x * width,
        y1=y * height,
        x2=(x + w) * width,
        y2=(y + h) * height,
    )


def _to_annotation(
    detection: Dict[str, Any],
    category_merge: Dict[str, str],
    width: int,
    height: int,
) -> Optional[GTAnnotation]:
    """Convert one FiftyOne detection to a GTAnnotation, or None if dropped."""
    merged = category_merge.get(detection.get(_LABEL_FIELD))
    if merged is None:
        return None
    bbox = detection.get(_BBOX_FIELD, [])
    if len(bbox) != _XYWH_LEN:
        return None
    return GTAnnotation(
        bbox=_to_pixel_bbox(bbox, width, height), category=merged
    )


def _sample_detections(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the raw detection dicts of one sample (empty if none)."""
    ground_truth = sample.get(_GROUND_TRUTH_FIELD) or {}
    return ground_truth.get(_DETECTIONS_FIELD, [])


def _build_gt_image(
    sample: Dict[str, Any],
    snapshot_dir: Path,
    category_merge: Dict[str, str],
) -> Optional[GTImage]:
    """Build one GTImage from a FiftyOne sample, or None if unusable."""
    metadata = sample.get(_METADATA_FIELD) or {}
    width = metadata.get("width")
    height = metadata.get("height")
    if not width or not height:
        logger.warning("Sample missing metadata dims: %s", sample.get("_id"))
        return None
    annotations = [
        ann for ann in (
            _to_annotation(det, category_merge, width, height)
            for det in _sample_detections(sample)
        )
        if ann is not None
    ]
    image_path = snapshot_dir / sample.get(_FILEPATH_FIELD, "")
    return GTImage(
        image_path=image_path, width=width, height=height,
        annotations=annotations,
    )


def ensure_images(
    gt_images: List[GTImage], config: FloorplancadEvalConfig
) -> None:
    """Fetch each sample's PNG from the HF dataset repo (best-effort).

    Parsing (load_floorplancad_ground_truth) needs only samples.json;
    SCORING needs the pixels. Call this before scoring a sampled subset
    so only those PNGs are downloaded (not all 5308). Failures are logged
    and left for the scorer to skip; shared by every FloorPlanCAD caller
    (eval + diagnostic) so the fetch logic lives in one place.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
    for gt_image in gt_images:
        filename = f"{_IMAGE_DIR_PREFIX}/{gt_image.image_path.name}"
        try:
            hf_hub_download(config.repo_id, filename, repo_type="dataset")
        except (HfHubHTTPError, OSError) as error:
            logger.warning("image fetch failed %s: %s", filename, error)


def load_floorplancad_ground_truth(
    config: FloorplancadEvalConfig,
) -> List[GTImage]:
    """Load FloorPlanCAD samples.json as door/window/wall GT images.

    Args:
        config: FloorplancadEvalConfig with the repo id and category map.

    Returns:
        One GTImage per sample, images resolved under the HF snapshot dir.
    """
    samples_path = _resolve_samples_path(config)
    snapshot_dir = samples_path.parent
    data = json.loads(samples_path.read_text())
    built = (
        _build_gt_image(sample, snapshot_dir, config.gt_category_merge)
        for sample in data["samples"]
    )
    return [gt_image for gt_image in built if gt_image is not None]
