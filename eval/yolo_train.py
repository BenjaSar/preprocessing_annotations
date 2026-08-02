"""T-S2 (YOLO_FINETUNING_ASSESSMENT.md Phase 9, combined plan Track A):
fine-tune YOLO26n on kaggle_floorplans500's train split for door/window
detection. Spike-only -- produces a checkpoint to evaluate via
kaggle_door_window_eval.py's `yolo_finetuned` tier. Does not touch
pipeline.py or any production call site.

Pass bar (already on record, not re-derived here): beat the CubiCasa5K
detector (door 0 TP, window P.62/R.67) and the SAM3 K=7 exemplar ceiling
(door P.78/R.83) on the same kaggle_floorplans500 test split.
"""

import argparse
import logging

from ultralytics import YOLO

from preprocessing_annotations.config import KaggleFloorplanTrainingConfig, yolo_checkpoint_path

logger = logging.getLogger(__name__)


def train(config: KaggleFloorplanTrainingConfig) -> str:
    """Fine-tune config.base_checkpoint on the Kaggle train split.

    Returns the path to the resulting best-weights checkpoint, ready to
    hand to kaggle_door_window_eval.py --tier yolo_finetuned
    --yolo-checkpoint <this path>.
    """
    model = YOLO(config.base_checkpoint)
    results = model.train(
        data=str(config.data_yaml_path),
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch=config.batch,
        mosaic=config.mosaic,
        project=str(config.runs_dir),
        name="kaggle_door_window",
    )
    return str(results.save_dir / "weights" / "best.pt")


def _parse_args() -> argparse.Namespace:
    """Parse CLI args: overrides only (all knobs default from
    KaggleFloorplanTrainingConfig -- config over hardcoding)."""
    parser = argparse.ArgumentParser(
        description="T-S2: fine-tune YOLO26n door/window detector"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override KaggleFloorplanTrainingConfig.epochs (e.g. --epochs "
             "1 for a smoke test before a full run)",
    )
    parser.add_argument(
        "--mosaic", type=float, default=None,
        help="Override KaggleFloorplanTrainingConfig.mosaic (0.0 disables "
             "mosaic augmentation, per YOLO_FINETUNING_ASSESSMENT.md's "
             "Phase 7 risk flag for structured technical drawings)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Bare Ultralytics checkpoint filename (e.g. yolo26s.pt) to "
             "fine-tune from instead of KaggleFloorplanTrainingConfig's "
             "default (yolo26n.pt) -- resolved under YOLO_CHECKPOINT_ROOT",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    config = KaggleFloorplanTrainingConfig()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.mosaic is not None:
        config.mosaic = args.mosaic
    if args.model is not None:
        config.base_checkpoint = yolo_checkpoint_path(args.model)
    best_checkpoint = train(config)
    print(best_checkpoint)


if __name__ == "__main__":
    main()
