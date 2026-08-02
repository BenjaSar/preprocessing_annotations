"""T-I1: shared YOLO checkpoint loading + box decode.

Single implementation reused by kaggle_door_window_eval.py (evaluation,
T-S1/T-S2) and yolo_detector.py (pipeline integration, T-I3) -- one place
to fix if Ultralytics' result.boxes schema ever changes, instead of two
copies drifting apart.

Returns a neutral tuple, not gt_evaluator.Detection or WindowDetection --
this module has no opinion on which schema a caller wants.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

from PIL import Image
from ultralytics import YOLO

# (class_name, (x1, y1, x2, y2) pixels, confidence)
YoloBox = Tuple[str, Tuple[float, float, float, float], float]


def load_model(checkpoint_path: str) -> YOLO:
    """Load a YOLO checkpoint from an absolute path (see
    config.yolo_checkpoint_path -- never a bare filename here, that
    downloads/resolves relative to whatever CWD the caller happens to
    run from, per the T-S1 lesson)."""
    return YOLO(checkpoint_path)


def detect_objects(
    model: YOLO,
    image: Union[Path, Image.Image],
    keep_categories: Tuple[str, ...],
    confidence_threshold: Optional[float] = None,
) -> List[YoloBox]:
    """Run inference on one image, return only boxes whose predicted
    class is in keep_categories. Maps whatever the model actually
    outputs -- no special-casing of any particular checkpoint's expected
    result.

    Accepts a Path (stringified for Ultralytics) or an in-memory PIL
    Image (R2: tile crops from TileSplitter.split are PIL objects, never
    written to disk -- confirmed Ultralytics' predict(source=...) takes
    either directly, 2026-07-29).

    confidence_threshold (R1): passed to Ultralytics as `conf` only when
    given. Default None omits the kwarg entirely, so existing callers
    that don't pass it keep Ultralytics' own implicit default (0.25) --
    zero behavior change for the recorded eval-tier baselines.
    """
    source = str(image) if isinstance(image, Path) else image
    predict_kwargs = {"source": source, "verbose": False}
    if confidence_threshold is not None:
        predict_kwargs["conf"] = confidence_threshold
    result = model.predict(**predict_kwargs)[0]
    boxes: List[YoloBox] = []
    for box in result.boxes:
        class_name = model.names[int(box.cls[0])]
        if class_name not in keep_categories:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = float(box.conf[0])
        boxes.append((class_name, (x1, y1, x2, y2), confidence))
    return boxes
