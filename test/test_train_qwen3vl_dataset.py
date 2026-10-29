"""The trainer's dataset loading: what the exporter writes is what training
reads, with image paths resolved. No GPU and no model here -- only the seam
where a wrong path or a dropped message would train on nothing.
"""

import json
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

from train_qwen3vl_unsloth import build_parser, load_datasets, load_split, select_examples


def _dataset(tmp_path, examples, split="train"):
    (tmp_path / "images").mkdir(parents=True, exist_ok=True)
    (tmp_path / f"{split}.jsonl").write_text(
        "".join(json.dumps(example) + "\n" for example in examples)
    )
    return tmp_path


def _example(image="images/tile_0_0_10_10.png"):
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "prompt"},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "[]"}]},
        ],
        "metadata": {"rooms": 0},
    }


def test_relative_image_paths_resolve_against_the_dataset_dir(tmp_path):
    dataset = _dataset(tmp_path, [_example()])
    (dataset / "images" / "tile_0_0_10_10.png").touch()

    loaded = load_split(dataset, "train")

    image = loaded[0]["messages"][1]["content"][0]["image"]
    assert Path(image).is_absolute()
    assert Path(image) == (dataset / "images" / "tile_0_0_10_10.png").resolve()


def test_every_message_turn_is_kept(tmp_path):
    loaded = load_split(_dataset(tmp_path, [_example()]), "train")
    assert [m["role"] for m in loaded[0]["messages"]] == ["system", "user", "assistant"]


def test_only_messages_reach_the_trainer(tmp_path):
    """metadata is provenance for us, not input for the collator."""
    loaded = load_split(_dataset(tmp_path, [_example()]), "train")
    assert list(loaded[0]) == ["messages"]


def test_a_missing_split_is_empty_not_an_error(tmp_path):
    assert load_split(_dataset(tmp_path, [_example()]), "val") == []


def test_blank_lines_are_skipped(tmp_path):
    dataset = _dataset(tmp_path, [_example()])
    (dataset / "train.jsonl").write_text(
        (dataset / "train.jsonl").read_text() + "\n"
    )
    assert len(load_split(dataset, "train")) == 1


@pytest.mark.parametrize("flag,attribute,value", [
    ("--lora-r", "lora_r", 16),
    ("--learning-rate", "learning_rate", 5e-5),
    ("--max-steps", "max_steps", 10),
    ("--batch-size", "batch_size", 2),
])
def test_hyperparameters_come_from_the_cli(flag, attribute, value):
    """None is measured on this data, so none is hardcoded in the script."""
    args = build_parser().parse_args(
        ["--dataset", "d", "--out", "o", flag, str(value)]
    )
    assert getattr(args, attribute) == value


def test_images_are_not_downscaled_by_default():
    """The collator's own default ("min") downscales to the model's image size
    -- 512 for Qwen3-VL -- which would contradict both the size stated in each
    prompt and the tile-pixel boxes, and would differ from what inference
    feeds the model. "max" means no resize."""
    args = build_parser().parse_args(["--dataset", "d", "--out", "o"])
    assert args.image_resize == "max"


def _tagged(n):
    return [{"messages": [], "tag": i} for i in range(n)]


def test_everything_within_budget_is_kept():
    examples = _tagged(3)
    kept, dropped = select_examples(examples, [100, 200, 300], 300, False, 0)
    assert kept == examples and dropped == 0


def test_overlong_examples_refuse_to_train_by_default():
    """Truncation would cut the assistant answer -- the only part with loss --
    so the trainer must not silently do it."""
    with pytest.raises(SystemExit, match="would be truncated"):
        select_examples(_tagged(3), [100, 5000, 300], 4224, False, 0)


def test_drop_overlong_drops_and_counts():
    examples = _tagged(3)
    kept, dropped = select_examples(examples, [100, 5000, 300], 4224, True, 0)
    assert [e["tag"] for e in kept] == [0, 2]
    assert dropped == 1


def test_only_longest_keeps_the_worst_case_for_a_vram_smoke():
    """A shuffled first-ten-steps run need not touch the longest example; the
    smoke has to, or it proves nothing about fit."""
    kept, _ = select_examples(_tagged(4), [100, 900, 300, 700], 1000, False, 2)
    assert [e["tag"] for e in kept] == [1, 3]


def test_only_longest_never_reintroduces_an_overlong_example():
    kept, dropped = select_examples(_tagged(3), [100, 5000, 300], 4224, True, 1)
    assert [e["tag"] for e in kept] == [2] and dropped == 1


def test_lengths_must_match_examples():
    with pytest.raises(ValueError):
        select_examples(_tagged(2), [1], 100, False, 0)


def test_default_budget_is_past_the_old_2048_cap():
    """At full resolution 26.6% of batch2's training examples exceed 2048."""
    args = build_parser().parse_args(["--dataset", "d", "--out", "o"])
    assert args.max_seq_length > 2048


def test_several_datasets_concatenate_with_their_own_image_paths(tmp_path):
    """Each export resolves against its own images/, so two runs' datasets
    train together without being merged on disk."""
    first = _dataset(tmp_path / "batch2", [_example("images/a.png")])
    second = _dataset(tmp_path / "batch3", [_example("images/b.png"), _example("images/c.png")])

    loaded = load_datasets([first, second], "train")

    images = [Path(e["messages"][1]["content"][0]["image"]) for e in loaded]
    assert images == [
        (first / "images" / "a.png").resolve(),
        (second / "images" / "b.png").resolve(),
        (second / "images" / "c.png").resolve(),
    ]


def test_dataset_flag_repeats():
    args = build_parser().parse_args(["--dataset", "a", "--dataset", "b", "--out", "o"])
    assert [str(d) for d in args.datasets] == ["a", "b"]


def test_eval_steps_is_configurable():
    """--eval must actually evaluate; the interval is a flag, not a constant."""
    args = build_parser().parse_args(["--dataset", "d", "--out", "o", "--eval", "--eval-steps", "10"])
    assert args.eval and args.eval_steps == 10


def test_smoke_test_default_is_the_small_model():
    """The 8B fit on this 15 GiB T4 is unmeasured; the default must not assume it."""
    args = build_parser().parse_args(["--dataset", "d", "--out", "o"])
    assert args.model == "qwen3-vl-2b"
