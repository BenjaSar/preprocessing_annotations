"""The holdout guarantee every fine-tune number rests on: no document the room
GT scores may appear in SFT training data."""

import json
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "eval"))

from sft_holdout import check, dataset_documents, evaluation_documents


def _eval_input(tmp_path, *names):
    directory = tmp_path / "test_input"
    directory.mkdir(exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"%PDF-1.4")
    return directory


def _dataset(tmp_path, name, pages_by_split):
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    for split, pages in pages_by_split.items():
        (directory / f"{split}.jsonl").write_text("\n".join(
            json.dumps({
                "messages": [],
                "metadata": {
                    "image": f"images/{page}_0_0_100_100.png",
                    "source_page": f"/corpus/images/{page}.png",
                },
            })
            for page in pages
        ))
    return directory


def test_a_held_out_eval_set_passes(tmp_path):
    evaluation = _eval_input(tmp_path, "326 ROCKAWAY - AVI-ON LAYOUT.pdf")
    dataset = _dataset(tmp_path, "batch", {"train": ["1614_w_temple_electrical_page000"]})

    verdict = check([dataset], evaluation)

    assert verdict["held_out"]
    assert verdict["leaked_documents"] == []
    assert verdict["training_documents"] == 1
    assert verdict["evaluation_documents"] == 1


def test_a_scored_document_in_the_training_split_is_a_leak(tmp_path):
    """The failure that would turn every gate verdict into a memorisation
    score while every number still looked like an improvement."""
    evaluation = _eval_input(tmp_path, "326 ROCKAWAY - AVI-ON LAYOUT.pdf")
    dataset = _dataset(tmp_path, "batch", {"train": ["326 ROCKAWAY - AVI-ON LAYOUT_page002"]})

    verdict = check([dataset], evaluation)

    assert not verdict["held_out"]
    assert len(verdict["leaked_documents"]) == 1
    assert verdict["leaked_documents"][0]["splits"] == ["train.jsonl"]


def test_a_scored_document_in_val_or_test_is_also_a_leak(tmp_path):
    """val and test are the adapter's own splits, not the GT eval -- a GT
    document there is still a document the fine-tune was tuned against."""
    evaluation = _eval_input(tmp_path, "Bradley Fair_Avi-on Submittal.pdf")
    dataset = _dataset(tmp_path, "batch", {
        "val": ["Bradley Fair_Avi-on Submittal_page001"],
        "test": ["Bradley Fair_Avi-on Submittal_page002"],
    })

    verdict = check([dataset], evaluation)

    assert not verdict["held_out"]
    assert verdict["leaked_documents"][0]["splits"] == ["test.jsonl", "val.jsonl"]


def test_case_and_page_suffix_do_not_hide_a_leak(tmp_path):
    """The GT names PDFs in title case; the pipeline writes pages in lower
    snake case with a page suffix. A comparison that missed either would
    report a clean holdout on a corpus that leaks."""
    evaluation = _eval_input(tmp_path, "BP-1 Violet Elementary School Electrical Pages.pdf")
    dataset = _dataset(tmp_path, "batch", {
        "train": ["bp-1 violet elementary school electrical pages_page004"],
    })

    verdict = check([dataset], evaluation)

    assert not verdict["held_out"]


def test_every_dataset_is_searched(tmp_path):
    """A leak in the fifth chunk must not hide behind four clean ones."""
    evaluation = _eval_input(tmp_path, "Building A-B.pdf")
    clean = _dataset(tmp_path, "c1", {"train": ["other_page000"]})
    leaky = _dataset(tmp_path, "c2", {"train": ["Building A-B_page000"]})

    verdict = check([clean, leaky], evaluation)

    assert not verdict["held_out"]
    assert verdict["leaked_documents"][0]["dataset"] == str(leaky)


def test_an_empty_dataset_is_reported_not_silently_passed(tmp_path):
    """An export that wrote nothing has not demonstrated a holdout."""
    evaluation = _eval_input(tmp_path, "Building A-B.pdf")
    empty = tmp_path / "empty"
    empty.mkdir()

    verdict = check([empty], evaluation)

    assert verdict["empty_datasets"] == [str(empty)]
    assert verdict["training_documents"] == 0


def test_an_eval_input_with_no_pdfs_is_refused(tmp_path):
    """Zero scored documents would make any corpus look held out."""
    with pytest.raises(ValueError):
        check([_dataset(tmp_path, "batch", {"train": ["a_page000"]})], _eval_input(tmp_path))


def test_documents_are_read_from_the_source_page_not_the_tile_crop(tmp_path):
    """metadata.image carries the tile's pixel offsets, so it does not reduce
    to a document; metadata.source_page is the page the tile came from."""
    dataset = _dataset(tmp_path, "batch", {"train": ["plan_page003"]})

    assert set(dataset_documents(dataset)) == {"plan"}


def test_evaluation_documents_come_from_the_pdfs(tmp_path):
    evaluation = _eval_input(tmp_path, "One.pdf", "Two.pdf")

    assert evaluation_documents(evaluation) == {"one", "two"}
