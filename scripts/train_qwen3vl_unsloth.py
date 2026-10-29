#!/usr/bin/env python3
"""LoRA fine-tune Qwen3-VL on the SFT dataset, via Unsloth.

Consumes what export/qwen_sft_exporter.py writes: train/val/test JSONL whose
`messages` carry a relative image path resolved against the dataset dir.

Supersedes scripts/finetune_qwen.py, which builds a Qwen2.5-VL trainer over a
format nothing in this project produces.

Two things this deliberately does NOT do:
  - choose hyperparameters: every one is a CLI flag, because none has been
    measured on this data. The defaults are the previous script's documented
    starting points, not tuned values.
  - decide whether the result is good: scoring is eval/room_gt_eval.py on the
    48 pinned GT pages, through VLMConfig.unsloth_adapter_path. Training loss
    alone never earned a model a ship here.

The run holds the pipeline's own lock (orchestration/run_lock.py): training
and a pipeline run would otherwise share one 15 GiB T4 and OOM.

Smoke test first -- the memory fit of an 8B 4-bit model plus LoRA on this GPU
has never been measured:

    python scripts/train_qwen3vl_unsloth.py --dataset <dir> --out <dir> \\
        --model qwen3-vl-2b --max-steps 10

Then the real run:

    python scripts/train_qwen3vl_unsloth.py --dataset <dir> --out <dir>
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

logger = logging.getLogger(__name__)

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
MLFLOW_STAGE = "sft_train"
# log_history keys that are per-step series; the rest of a report is scalar.
STEP_SERIES = ("loss", "eval_loss")


def load_split(dataset_dir: Path, split: str) -> List[Dict[str, Any]]:
    """Read one JSONL split, resolving each image path against the dataset dir.

    The exporter writes relative paths so a dataset can be moved; the trainer
    is where they become absolute.
    """
    path = dataset_dir / f"{split}.jsonl"
    if not path.exists():
        return []
    examples = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        example = json.loads(line)
        for message in example["messages"]:
            for part in message["content"]:
                if part.get("type") == "image":
                    part["image"] = str((dataset_dir / part["image"]).resolve())
        examples.append({"messages": example["messages"]})
    logger.info("%s: %d examples from %s", split, len(examples), path)
    return examples


def load_datasets(dataset_dirs: Sequence[Path], split: str) -> List[Dict[str, Any]]:
    """One split from several exported datasets, concatenated in order.

    Each export keeps its own images/, and load_split resolves every image
    path against its own dataset dir, so corpora exported separately (one
    per pipeline run) train together without being merged on disk. Their
    group-aware splits were assigned per run, so a document can only appear
    in one split of the combined set if the runs share no document -- which
    split_integrity.py checks per run and the corpus selection checks across.
    """
    return [example for directory in dataset_dirs for example in load_split(directory, split)]


def token_length(example: Dict[str, Any], processor: Any) -> int:
    """Tokens this example becomes once its image is tokenized.

    Exact under --image-resize max, where the collator passes images through
    unchanged; an upper bound under any downscaling policy. Either way it never
    under-counts, so a preflight built on it cannot miss a truncation.
    """
    from PIL import Image

    messages = example["messages"]
    images = [
        Image.open(part["image"]).convert("RGB")
        for message in messages for part in message["content"]
        if part.get("type") == "image"
    ]
    text = processor.apply_chat_template(messages, tokenize=False)
    return int(processor(text=[text], images=images, return_tensors="pt")["input_ids"].shape[1])


def select_examples(
    examples: Sequence[Dict[str, Any]],
    lengths: Sequence[int],
    max_seq_length: int,
    drop_overlong: bool,
    only_longest: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Apply the token budget: refuse to truncate, or drop explicitly.

    The collator truncates from the END of a sequence, and the end is the
    assistant answer -- the only part that carries loss -- so a truncated
    example trains on a cut-off JSON array or on nothing. Failing loudly is
    the default; --drop-overlong turns that into a counted, reported drop.

    `only_longest` keeps just the K longest survivors: a VRAM smoke test has
    to exercise the worst case, and a shuffled first-ten-steps run need not.

    Returns:
        The examples to train on, and how many were dropped as overlong.
    """
    if len(examples) != len(lengths):
        raise ValueError("one length per example")
    fits = [(n, e) for n, e in zip(lengths, examples) if n <= max_seq_length]
    overlong = len(examples) - len(fits)
    if overlong and not drop_overlong:
        worst = sorted(lengths, reverse=True)[:overlong]
        raise SystemExit(
            f"{overlong} example(s) exceed --max-seq-length {max_seq_length} "
            f"(longest: {worst[:5]}) and would be truncated, cutting the "
            f"assistant answer. Raise --max-seq-length or pass --drop-overlong."
        )
    if only_longest:
        fits = sorted(fits, key=lambda pair: pair[0], reverse=True)[:only_longest]
    return [e for _, e in fits], overlong


def peak_vram_gb() -> float:
    """Peak allocated VRAM so far, or 0.0 without CUDA."""
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 ** 3


def log_training_run(
    report: Dict[str, Any], log_history: Sequence[Dict[str, Any]], report_path: Path
) -> None:
    """Record a finished training run in the active MLflow run.

    Logged after training, from the trainer's own history, rather than via
    SFTConfig(report_to="mlflow"): that integration would call mlflow outside
    the fail-silent guard, so an unreachable server could raise mid-training.
    No-ops when tracking is off, and never fails the run: this is called after
    the adapter and report are saved, and a chain aborts on a nonzero exit.
    """
    from preprocessing_annotations.orchestration.mlflow_tracking import (
        log_artifact, log_metrics, log_params, log_stage_metric, non_fatal,
    )

    with non_fatal("training log"):
        log_params({
            "model": report["model"],
            "datasets": report["datasets"],
            **report["hyperparameters"],
        })
        log_metrics(MLFLOW_STAGE, {
            "train_loss": report["train_loss"],
            "peak_vram_gb": report["peak_vram_gb"],
            "steps": report["steps"],
            "seconds": report["seconds"],
            "train_examples": report["train_examples"],
            "eval_examples": report["eval_examples"],
            "dropped_overlong_train": report["dropped_overlong"]["train"],
            "dropped_overlong_val": report["dropped_overlong"]["val"],
        })
        for entry in log_history:
            for series in STEP_SERIES:
                if series in entry:
                    log_stage_metric(MLFLOW_STAGE, series, entry[series], step=entry["step"])
        log_artifact(str(report_path))


def train(args: argparse.Namespace) -> Dict[str, Any]:
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTConfig, SFTTrainer

    from preprocessing_annotations.vlm.vlm_backend import UnslothQwenBackend

    # One registry of model keys for inference and training, so a fine-tune
    # cannot be built on a checkpoint the pipeline cannot then load.
    registry = UnslothQwenBackend.MODEL_REGISTRY
    if args.model not in registry:
        raise SystemExit(f"unknown model key {args.model!r}; have {sorted(registry)}")

    train_examples = load_datasets(args.datasets, TRAIN_SPLIT)
    if not train_examples:
        raise SystemExit(f"no training examples in {[str(d) for d in args.datasets]}")
    eval_examples = load_datasets(args.datasets, VAL_SPLIT) if args.eval else []

    # max_seq_length goes to from_pretrained too: the collator reads its
    # truncation limit from model.max_seq_length, not from SFTConfig, so a
    # value set only on SFTConfig silently left the old 2048 cap in force.
    model, processor = FastVisionModel.from_pretrained(
        registry[args.model],
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        max_seq_length=args.max_seq_length,
    )

    train_examples, train_overlong = select_examples(
        train_examples, [token_length(e, processor) for e in train_examples],
        args.max_seq_length, args.drop_overlong, args.only_longest,
    )
    eval_examples, eval_overlong = select_examples(
        eval_examples, [token_length(e, processor) for e in eval_examples],
        args.max_seq_length, args.drop_overlong, 0,
    )
    logger.info(
        "token budget %d: training on %d, dropped %d train / %d val as overlong",
        args.max_seq_length, len(train_examples), train_overlong, eval_overlong,
    )
    model = FastVisionModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        random_state=args.seed,
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_training(model)

    args.out.mkdir(parents=True, exist_ok=True)
    config = SFTConfig(
        output_dir=str(args.out),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        # Without an eval strategy TRL never evaluates, so a val split passed
        # via --eval was loaded and then ignored: sft_8b_run1's log holds no
        # eval_loss at all. Validation loss rising while training loss falls
        # is the overfitting signal, and with 280 examples it is the one to watch.
        eval_strategy="steps" if eval_examples else "no",
        eval_steps=args.eval_steps if eval_examples else None,
        seed=args.seed,
        optim=args.optim,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=args.max_seq_length,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_examples,
        eval_dataset=eval_examples or None,
        processing_class=processor.tokenizer,
        # resize: the collator's default "min" silently downscales every image
        # to the model's default size (512 for Qwen3-VL, which reports none),
        # so a 1237x1178 tile reached the model as ~512x488 while its prompt
        # said 1237x1178 and its boxes stayed in tile pixels. Inference does no
        # such thing -- UnslothQwenBackend only caps the longest edge at
        # VLMConfig.unsloth_detection_max_dim_px (4096), which these tiles are
        # far below. "max" disables the resize, so the pixels the model trains
        # on are the pixels it will be asked about.
        data_collator=UnslothVisionDataCollator(
            model, processor, resize=args.image_resize, max_seq_length=args.max_seq_length,
        ),
        args=config,
    )

    started = time.time()
    result = trainer.train()
    model.save_pretrained(str(args.out))
    processor.save_pretrained(str(args.out))

    report = {
        "model": args.model,
        "model_id": registry[args.model],
        "adapter_dir": str(args.out),
        "datasets": [str(d) for d in args.datasets],
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "dropped_overlong": {"train": train_overlong, "val": eval_overlong},
        "steps": result.global_step,
        "train_loss": result.training_loss,
        "eval_loss_history": [
            {"step": entry["step"], "eval_loss": entry["eval_loss"]}
            for entry in trainer.state.log_history if "eval_loss" in entry
        ],
        "peak_vram_gb": round(peak_vram_gb(), 2),
        "seconds": round(time.time() - started, 1),
        "hyperparameters": {
            "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout, "learning_rate": args.learning_rate,
            "epochs": args.epochs, "max_steps": args.max_steps,
            "batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "warmup_ratio": args.warmup_ratio, "max_seq_length": args.max_seq_length,
            "optim": args.optim, "seed": args.seed,
            "image_resize": args.image_resize,
            "drop_overlong": args.drop_overlong, "only_longest": args.only_longest,
            "eval_steps": args.eval_steps,
        },
    }
    report_path = args.out / "train_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    log_training_run(report, trainer.state.log_history, report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen3-VL on the SFT dataset")
    parser.add_argument("--dataset", type=Path, action="append", required=True, dest="datasets",
                        help="Dir holding train/val JSONL and images/; repeat to train on several")
    parser.add_argument("--out", type=Path, required=True, help="Where to write the adapter and report")
    parser.add_argument("--model", default="qwen3-vl-2b",
                        help="Model key from UnslothQwenBackend.MODEL_REGISTRY (smoke-test default: the 2B)")
    parser.add_argument("--eval", action="store_true",
                        help="Load the val split and compute validation loss every --eval-steps")
    parser.add_argument("--eval-steps", type=int, default=25,
                        help="Steps between validation-loss evaluations (with --eval)")
    parser.add_argument("--max-steps", type=int, default=-1, help="Stop after N steps (-1 = run the epochs)")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--max-seq-length", type=int, default=4224,
        help="Token budget per example, image tokens included; applied to the "
             "model, the collator and SFTConfig alike. Default: the smallest "
             "multiple of 64 that fits every corpus_batch2_v2 example but one "
             "(352/353 -- the outlier is a 34-room 2952x2107 whole page at 8206 "
             "tokens). At full resolution a 1237x1178 tile is ~1800-2360 tokens, "
             "so the old 2048 cut 26.6%% of the training set.",
    )
    parser.add_argument(
        "--drop-overlong", action="store_true",
        help="Drop examples over --max-seq-length and report how many, instead "
             "of refusing to train. Never truncates: truncation cuts the answer.",
    )
    parser.add_argument(
        "--only-longest", type=int, default=0,
        help="Train only on the K longest examples (0 = all). For the VRAM smoke "
             "test, which must exercise the worst case to prove a fit.",
    )
    parser.add_argument(
        "--image-resize", default="max",
        help="UnslothVisionDataCollator resize policy. Default 'max' = no "
             "downscaling, so training sees the same pixels as inference and "
             "the tile-pixel targets stay true. 'min' restores the collator's "
             "own default (downscale to the model's image size), an integer or "
             "WxH forces a size. Changing this invalidates the box coordinates "
             "unless the exporter changes with it.",
    )
    parser.add_argument("--optim", default="adamw_8bit")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--no-wait", action="store_true",
                        help="Exit (code 3) instead of waiting when a pipeline run holds the lock")
    # Imported here, not at module top, like every other package import in
    # this script: unsloth must be imported before transformers/trl, so the
    # package import stays out of module load. The package import this adds
    # to the parser is the one run_lock's import in __main__ already made.
    from preprocessing_annotations.orchestration.mlflow_tracking import add_mlflow_arguments

    add_mlflow_arguments(parser)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    from preprocessing_annotations.orchestration.mlflow_tracking import (
        mlflow_config_from_args, tracked_run,
    )
    from preprocessing_annotations.orchestration.run_lock import single_instance

    # Shares the pipeline's lock: both load multi-GB models onto one GPU.
    # The run opens inside the lock, so a job queued behind a pipeline run
    # does not show as an in-progress MLflow run while it is only waiting.
    with single_instance(wait=not args.no_wait):
        with tracked_run(mlflow_config_from_args(args), MLFLOW_STAGE):
            report = train(args)
        print(json.dumps(report, indent=2))
