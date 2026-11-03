"""The Qwen3-VL SFT exporter: targets round-trip through the backend parser,
boxes are tile-local label boxes, splits come from the run's own group-aware
COCO export, and empty tiles are capped.

The round-trip is the contract that matters: an example teaches the model to
answer something UnslothQwenBackend._parse_room_response can read back as the
same boxes, or the dataset is training a format the pipeline cannot consume.
"""

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing_annotations.automation.taxonomy import DEFAULT_MANDATORY_TYPE
from preprocessing_annotations.config import PipelineConfig, SFTDatasetConfig
from preprocessing_annotations.export.qwen_sft_exporter import _tile_boxes, export, rooms_of
from preprocessing_annotations.vlm.vlm_backend import UnslothQwenBackend

TILED_PAGE = (4500, 3000)   # above PipelineConfig.tile_trigger_px
SMALL_PAGE = (2200, 1700)   # below it


def _room(name, box, convention="label", room_type="PRIVATE OFFICE", confidence=0.87):
    return {
        "name": name,
        "type": room_type,
        "confidence": confidence,
        "coordinates": {"bbox": list(box), "box_convention": convention},
    }


def _page_pixels(size, seed=7):
    """A page whose pixels vary, so a crop can be told from the whole page."""
    image = Image.new("L", size, "white")
    for x in range(0, size[0], 8):
        for y in range(0, size[1], 8):
            image.putpixel((x, y), (seed * 31 + x * 5 + y * 11) % 256)
    return image.convert("RGB")


def _run_dir(tmp_path, pages, splits):
    """A finished-run layout: processed_annotations/, images/, coco/.

    pages: {file_stem: (page_size, [room, ...], [exclusion zone, ...])}
    """
    run = tmp_path / "run"
    (run / "processed_annotations").mkdir(parents=True)
    (run / "images").mkdir()
    (run / "coco").mkdir()
    for stem, (size, rooms, zones) in pages.items():
        _page_pixels(size).save(run / "images" / f"{stem}.png")
        (run / "processed_annotations" / f"{stem}.json").write_text(json.dumps({
            "image_file": f"{stem}.png",
            "image_size": {"width": size[0], "height": size[1]},
            "sft_ready": True,
            "roomsRecognized": rooms,
            "exclusion_zones": zones,
        }))
    for split, stems in splits.items():
        (run / "coco" / f"{split}.json").write_text(json.dumps({
            "images": [{"id": i, "file_name": f"{s}.png"} for i, s in enumerate(stems)],
            "annotations": [], "categories": [],
        }))
    return run


def _examples(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _assistant_text(example):
    return example["messages"][-1]["content"][0]["text"]


def test_targets_round_trip_through_the_backend_parser(tmp_path):
    """Every target reads back as the identical box, in the tile's own pixels."""
    rooms = [_room("OFFICE 1", [100, 120, 260, 150]), _room("CORRIDOR", [300, 400, 420, 430])]
    run = _run_dir(tmp_path, {"doc_page000": (TILED_PAGE, rooms, [])},
                   {"train": ["doc_page000"]})

    export(run, tmp_path / "out")

    checked = 0
    for example in _examples(tmp_path / "out" / "train.jsonl"):
        width, height = example["metadata"]["tile_size"]
        expected = json.loads(_assistant_text(example))
        parsed = UnslothQwenBackend._parse_room_response(
            _backend(), _assistant_text(example), width, height,
        )
        assert [p["bbox"] for p in parsed] == [e["bbox"] for e in expected]
        assert [p["room_name"] for p in parsed] == [e["room_name"] for e in expected]
        checked += len(expected)
    assert checked == len(rooms)


def _backend():
    """A backend for its parser; the model loads lazily, so none is loaded."""
    return UnslothQwenBackend(PipelineConfig().vlm)


def test_the_referenced_image_is_the_tile_the_boxes_describe(tmp_path):
    """The pairing a trainer actually consumes: it reads messages[].image and
    nothing else, so that file must be the crop the prompt and boxes describe
    -- not the whole page with the crop hidden in metadata."""
    run = _run_dir(tmp_path, {"doc_page000": (TILED_PAGE, [_room("ONE", [100, 100, 200, 130])], [])},
                   {"train": ["doc_page000"]})

    export(run, tmp_path / "out")

    page = Image.open(run / "images" / "doc_page000.png").convert("RGB")
    examples = _examples(tmp_path / "out" / "train.jsonl")
    assert examples
    for example in examples:
        reference = example["messages"][1]["content"][0]["image"]
        assert reference == example["metadata"]["image"]
        crop_path = tmp_path / "out" / reference
        assert not Path(reference).is_absolute(), "dataset must stay relocatable"
        assert crop_path.exists()
        with Image.open(crop_path) as crop:
            assert list(crop.size) == example["metadata"]["tile_size"]
            assert crop.size != page.size or len(examples) == 1
            assert crop.convert("RGB").tobytes() == page.crop(
                tuple(example["metadata"]["crop"])
            ).tobytes()
        prompt = example["messages"][1]["content"][1]["text"]
        assert f"{crop.size[0]}x{crop.size[1]} pixels" in prompt


def test_crops_are_written_only_for_kept_examples(tmp_path):
    """The empty-tile cap drops examples; their crops must not be written."""
    run = _run_dir(tmp_path, {"doc_page000": (TILED_PAGE, [_room("ONE", [100, 100, 200, 130])], [])},
                   {"train": ["doc_page000"]})

    report = export(run, tmp_path / "out")

    written = list((tmp_path / "out" / "images").glob("*.png"))
    assert len(written) == report["examples"]
    referenced = {e["metadata"]["image"] for e in _examples(tmp_path / "out" / "train.jsonl")}
    assert {f"images/{p.name}" for p in written} == referenced


def test_target_confidence_comes_from_the_pipeline(tmp_path):
    """Not a constant: TileSplitter.merge breaks tile overlaps by confidence."""
    rooms = [_room("SURE", [100, 100, 200, 130], confidence=0.91),
             _room("UNSURE", [400, 100, 500, 130], confidence=0.62)]
    run = _run_dir(tmp_path, {"doc_page000": (SMALL_PAGE, rooms, [])},
                   {"train": ["doc_page000"]})

    export(run, tmp_path / "out")

    targets = {t["room_name"]: t["confidence"]
               for e in _examples(tmp_path / "out" / "train.jsonl")
               for t in json.loads(_assistant_text(e))}
    assert targets == {"SURE": 0.91, "UNSURE": 0.62}


def test_boxes_are_tile_local_and_inside_the_tile(tmp_path):
    room_box = [3000, 2000, 3100, 2030]   # in the lower-right of a tiled page
    run = _run_dir(tmp_path, {"doc_page000": (TILED_PAGE, [_room("STORE", room_box)], [])},
                   {"train": ["doc_page000"]})

    export(run, tmp_path / "out")

    targets = [
        (e, t) for e in _examples(tmp_path / "out" / "train.jsonl")
        for t in json.loads(_assistant_text(e))
    ]
    assert targets, "the room should appear in at least one tile"
    for example, target in targets:
        width, height = example["metadata"]["tile_size"]
        x1, y1, x2, y2 = target["bbox"]
        assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
        assert [x2 - x1, y2 - y1] == [room_box[2] - room_box[0], room_box[3] - room_box[1]]


def _cut_box(page, config, margin=10, height=30):
    """A box no tile contains whole: it spans the whole overlap and beyond.

    Taken from the live grid, not hardcoded. A box narrower than the overlap
    would sit inside it and be whole in two tiles, testing nothing — which is
    exactly how the first version of this test passed vacuously.
    """
    tiles = _tile_boxes(page, config)
    edge = sorted({t[2] for t in tiles})[0]            # first tile's right edge
    overlap_start = sorted({t[0] for t in tiles})[1]   # where the next tile begins
    assert overlap_start < edge, "tiles are expected to overlap"
    # Start before the second tile and end after the first: neither holds it.
    return [overlap_start - margin, 500, edge + margin, 500 + height], edge


def test_room_crossing_every_tile_edge_is_dropped_not_clipped(tmp_path):
    """D6: a box no tile contains whole yields no target, never a truncated one."""
    config = PipelineConfig()
    crossing, edge = _cut_box(TILED_PAGE, config)
    assert crossing[0] < edge < crossing[2], "the box must straddle a tile edge"
    run = _run_dir(tmp_path, {"doc_page000": (TILED_PAGE, [_room("SEAM", crossing)], [])},
                   {"train": ["doc_page000"]})

    report = export(run, tmp_path / "out")

    targets = [t for e in _examples(tmp_path / "out" / "train.jsonl")
               for t in json.loads(_assistant_text(e))]
    assert targets == []
    assert report["rooms_without_whole_tile"] == 1


def test_room_inside_the_tile_overlap_is_a_target_in_both_tiles(tmp_path):
    """The other half of D6: whole in two tiles means a target in each."""
    config = PipelineConfig()
    tiles = _tile_boxes(TILED_PAGE, config)
    overlap_start, edge = sorted({t[0] for t in tiles})[1], sorted({t[2] for t in tiles})[0]
    box = [overlap_start + 10, 500, edge - 10, 530]
    run = _run_dir(tmp_path, {"doc_page000": (TILED_PAGE, [_room("OVERLAP", box)], [])},
                   {"train": ["doc_page000"]})

    export(run, tmp_path / "out")

    targets = [t for e in _examples(tmp_path / "out" / "train.jsonl")
               for t in json.loads(_assistant_text(e))]
    assert len(targets) == 2
    for target in targets:
        x1, y1, x2, y2 = target["bbox"]
        assert [x2 - x1, y2 - y1] == [box[2] - box[0], box[3] - box[1]]


def test_only_the_trained_box_convention_becomes_a_target(tmp_path):
    rooms = [_room("LABEL ROOM", [100, 100, 200, 130]),
             _room("EXTENT ROOM", [300, 300, 900, 900], convention="room_extent")]
    run = _run_dir(tmp_path, {"doc_page000": (SMALL_PAGE, rooms, [])},
                   {"train": ["doc_page000"]})

    report = export(run, tmp_path / "out")

    names = [t["room_name"] for e in _examples(tmp_path / "out" / "train.jsonl")
             for t in json.loads(_assistant_text(e))]
    assert names == ["LABEL ROOM"]
    assert report["rooms_skipped"]["wrong_convention"] == 1


def test_a_small_page_is_one_whole_example(tmp_path):
    run = _run_dir(tmp_path, {"doc_page000": (SMALL_PAGE, [_room("LOBBY", [10, 10, 90, 40])], [])},
                   {"train": ["doc_page000"]})

    export(run, tmp_path / "out")
    examples = _examples(tmp_path / "out" / "train.jsonl")

    assert len(examples) == 1
    assert examples[0]["metadata"]["tile_size"] == list(SMALL_PAGE)
    assert examples[0]["metadata"]["crop"] == [0, 0, SMALL_PAGE[0], SMALL_PAGE[1]]


def test_splits_follow_the_runs_own_coco_export(tmp_path):
    pages = {f"{name}_page000": (SMALL_PAGE, [_room("OFFICE", [10, 10, 90, 40])], [])
             for name in ("alpha", "beta", "gamma")}
    run = _run_dir(tmp_path, pages, {"train": ["alpha_page000"], "val": ["beta_page000"],
                                     "test": ["gamma_page000"]})

    report = export(run, tmp_path / "out")

    assert report["per_split"] == {"train": 1, "val": 1, "test": 1}
    for split, document in [("train", "alpha"), ("val", "beta"), ("test", "gamma")]:
        images = {e["metadata"]["image"] for e in _examples(tmp_path / "out" / f"{split}.jsonl")}
        assert all(document in image for image in images)


def test_empty_tiles_are_capped(tmp_path):
    """One room on a tiled page leaves many empty tiles; the cap bounds them."""
    run = _run_dir(tmp_path, {"doc_page000": (TILED_PAGE, [_room("ONE", [100, 100, 200, 130])], [])},
                   {"train": ["doc_page000"]})

    report = export(run, tmp_path / "out")

    assert report["empty_tiles_available"] > report["empty_tiles_kept"]
    empty = sum(1 for e in _examples(tmp_path / "out" / "train.jsonl")
                if e["metadata"]["rooms"] == 0)
    assert empty / len(_examples(tmp_path / "out" / "train.jsonl")) <= (
        SFTDatasetConfig().max_empty_tile_fraction + 1e-9
    )


def test_export_is_reproducible(tmp_path):
    pages = {"doc_page000": (TILED_PAGE, [_room("ONE", [100, 100, 200, 130])], [])}
    run = _run_dir(tmp_path, pages, {"train": ["doc_page000"]})

    export(run, tmp_path / "a")
    export(run, tmp_path / "b")

    assert (tmp_path / "a" / "train.jsonl").read_text() == (tmp_path / "b" / "train.jsonl").read_text()


@pytest.mark.parametrize("bad,reason", [
    ({"name": "NO BOX", "type": "OTHER", "coordinates": {"box_convention": "label"}},
     "unusable_box"),
    ({"name": "DEGENERATE", "type": "OTHER",
      "coordinates": {"bbox": [100, 100, 100, 130], "box_convention": "label"}},
     "unusable_box"),
    ({"name": "", "type": "OTHER",
      "coordinates": {"bbox": [10, 10, 50, 40], "box_convention": "label"}},
     "no_name"),
])
def test_unusable_rooms_never_become_targets(bad, reason):
    rooms, skipped = rooms_of({"roomsRecognized": [bad]}, SFTDatasetConfig().box_convention)
    assert rooms == []
    assert skipped[reason] == 1


def test_name_confidence_floor_is_off_by_default():
    """Default keeps every room: the box is right even when the string is not."""
    assert SFTDatasetConfig().min_name_ocr_confidence == 0.0


def test_name_confidence_floor_drops_only_low_confidence_names():
    """D9 mechanism: the floor reads confidence_detail.ocr, the OCR text score."""
    rooms = [
        {**_room("CLEAN", [10, 10, 90, 40]), "confidence_detail": {"ocr": 0.99}},
        {**_room("MISREAD", [10, 50, 90, 80]), "confidence_detail": {"ocr": 0.87}},
    ]

    kept, skipped = rooms_of({"roomsRecognized": rooms}, "label", min_name_ocr_confidence=0.9)

    assert [room.name for room in kept] == ["CLEAN"]
    assert skipped["low_name_ocr_confidence"] == 1


def test_room_without_recorded_ocr_confidence_survives_a_floor():
    """A VLM-only room records no OCR score; a floor must not silently cut it."""
    kept, _ = rooms_of(
        {"roomsRecognized": [_room("VLM ROOM", [10, 10, 90, 40])]},
        "label", min_name_ocr_confidence=0.9,
    )
    assert [room.name for room in kept] == ["VLM ROOM"]


# --- unplaceable room types (SFTDatasetConfig.resolve_default_types_from_name) ---

def test_default_type_is_re_resolved_from_the_label():
    """A stored type that is only the taxonomy's default is re-read from the
    label, so a class added after the run reaches the dataset."""
    rooms, skipped = rooms_of(
        {"roomsRecognized": [_room("EXAM RM", [10, 10, 90, 40], room_type=DEFAULT_MANDATORY_TYPE)]},
        "label", resolve_default_from_name=True,
    )
    assert [room.room_type for room in rooms] == ["CLINIC"]
    assert skipped == {}


def test_label_the_taxonomy_cannot_place_is_dropped_not_guessed():
    rooms, skipped = rooms_of(
        {"roomsRecognized": [_room("SUITE 200", [10, 10, 90, 40], room_type=DEFAULT_MANDATORY_TYPE)]},
        "label", resolve_default_from_name=True,
    )
    assert rooms == []
    assert skipped["unresolved_type"] == 1


def test_a_real_storage_label_keeps_the_default_type():
    rooms, _ = rooms_of(
        {"roomsRecognized": [_room("STOR", [10, 10, 90, 40], room_type=DEFAULT_MANDATORY_TYPE)]},
        "label", resolve_default_from_name=True,
    )
    assert [room.room_type for room in rooms] == [DEFAULT_MANDATORY_TYPE]


def test_types_from_elsewhere_than_the_label_are_left_alone():
    """resolve_context_dependent_types turns LAB into CLINIC on a clinical
    page; re-reading the label here would undo it."""
    rooms, _ = rooms_of(
        {"roomsRecognized": [_room("LAB", [10, 10, 90, 40], room_type="CLINIC")]},
        "label", resolve_default_from_name=True,
    )
    assert [room.room_type for room in rooms] == ["CLINIC"]


def test_resolution_is_off_unless_asked_for():
    rooms, skipped = rooms_of(
        {"roomsRecognized": [_room("SUITE 200", [10, 10, 90, 40], room_type=DEFAULT_MANDATORY_TYPE)]},
        "label",
    )
    assert [room.room_type for room in rooms] == [DEFAULT_MANDATORY_TYPE]
    assert skipped == {}


def test_export_reports_the_dropped_targets_and_the_flag(tmp_path):
    run = _run_dir(
        tmp_path,
        {"plan_page000": (SMALL_PAGE, [
            _room("EXAM RM", [100, 100, 260, 140], room_type=DEFAULT_MANDATORY_TYPE),
            _room("SUITE 200", [300, 100, 460, 140], room_type=DEFAULT_MANDATORY_TYPE),
        ], [])},
        {"train": ["plan_page000"], "val": [], "test": []},
    )

    report = export(run, tmp_path / "dataset")

    assert report["resolve_default_types_from_name"] is True
    assert report["rooms_skipped"]["unresolved_type"] == 1
    targets = json.loads(_assistant_text(_examples(tmp_path / "dataset" / "train.jsonl")[0]))
    assert [t["room_type"] for t in targets] == ["CLINIC"]
