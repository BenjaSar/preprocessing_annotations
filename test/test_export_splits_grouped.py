"""COCOExporter.export_splits assigns by group: no document straddles splits,
and pinned groups land in test."""

import json
from pathlib import Path

from PIL import Image

from preprocessing_annotations.export.exporters import COCOExporter
from preprocessing_annotations.export.group_split import GroupMap

DOCUMENTS = {"alpha": 4, "beta": 4, "gamma": 4, "delta": 4}


def _corpus(tmp_path):
    processed, images = tmp_path / "processed", tmp_path / "images"
    processed.mkdir()
    images.mkdir()
    for document, pages in DOCUMENTS.items():
        for page in range(pages):
            stem = f"{document}_page{page:03d}"
            Image.new("RGB", (100, 100), "white").save(images / f"{stem}.png")
            (processed / f"{stem}.json").write_text(json.dumps({
                "image_file": f"{stem}.png",
                "image_size": {"width": 100, "height": 100},
                "sft_ready": True,
                "rooms": [
                    {"room_name": "OFFICE", "type": "PRIVATE OFFICE", "bbox": [10, 10, 20, 20]},
                    {"room_name": "TOILET", "type": "TOILET", "bbox": [40, 40, 20, 20]},
                    {"room_name": "CORRIDOR", "type": "CORRIDOR", "bbox": [60, 10, 20, 20]},
                ],
            }))
    return processed, images


def _split_of_each_document(output_dir):
    split_of = {}
    for split in ("train", "val", "test"):
        path = output_dir / f"{split}.json"
        if not path.exists():
            continue
        for image in json.loads(path.read_text())["images"]:
            split_of.setdefault(Path(image["file_name"]).stem.rsplit("_page", 1)[0], set()).add(split)
    return split_of


def test_no_document_straddles_splits(tmp_path):
    processed, images = _corpus(tmp_path)
    output = tmp_path / "coco"

    COCOExporter().export_splits(processed, images, output, min_annotations_per_image=1)

    straddling = {d: s for d, s in _split_of_each_document(output).items() if len(s) > 1}
    assert straddling == {}


def test_pinned_group_lands_in_test(tmp_path):
    processed, images = _corpus(tmp_path)
    output = tmp_path / "coco"
    group_map = GroupMap(
        groups={"alpha": "family", "beta": "family"}, pinned_test_groups=frozenset({"family"}),
    )

    COCOExporter().export_splits(
        processed, images, output, min_annotations_per_image=1, group_map=group_map,
    )

    split_of = _split_of_each_document(output)
    assert split_of["alpha"] == {"test"}
    assert split_of["beta"] == {"test"}


def test_every_page_is_exported_exactly_once(tmp_path):
    processed, images = _corpus(tmp_path)
    output = tmp_path / "coco"

    counts = COCOExporter().export_splits(
        processed, images, output, min_annotations_per_image=1,
    )

    assert sum(counts.values()) == sum(DOCUMENTS.values())
