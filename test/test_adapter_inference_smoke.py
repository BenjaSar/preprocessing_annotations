"""G-PROD-1: the fine-tuned adapter, through the real inference path.

Everything else that touches the adapter is mocked. test_unsloth_adapter_path
proves the config records a path and that a missing one raises, but it never
loads weights; room_replay scores *recorded* annotations, so it reads a JSON
file a run left behind rather than exercising generation. Between them sits the
step that actually produced .6112 recall and that nothing asserts on: load the
LoRA onto the quantized base, ask it with the prompt it was trained with, and
parse what comes back.

The failure this guards is specific and quiet. `initialize` loads the adapter
and then calls `FastVisionModel.for_inference(self.model)`; the prompt comes
from `unsloth_room_prompt`, separately. Break either link -- an adapter that
loads but does not attach, a prompt default that drifts back to `production`
-- and the pipeline still emits well-formed rooms, just the base model's. Every
downstream check passes. Only the recall moves, and only after a 7 h run.

So the assertions here are not "it returned something". They are the properties
that a silently-unattached adapter or a mismatched prompt would break:
boxes land inside the page, they are label-sized rather than page-sized, and
the labels are non-empty text.

Marked `slow` (the marker's documented meaning: real GPU/eval work, minutes),
so `pytest test/` stays fast and CI opts in with `-m slow`. Skips rather than
fails wherever the environment cannot run it -- no unsloth, no CUDA, no
adapter, no page -- because a skip is honest about an untested path and a
failure there would only teach people to ignore it.
"""

import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from preprocessing_annotations.config.config import VLMConfig
from preprocessing_annotations.vlm.vlm_backend import UnslothQwenBackend

pytestmark = pytest.mark.slow

UNSLOTH_MODULE = "unsloth"
ADAPTER_ENV = "VLM_SMOKE_ADAPTER"
PAGE_ENV = "VLM_SMOKE_PAGE"
DEFAULT_ADAPTER = PROJECT_ROOT / "trained_adapters" / "sft_real_v1"
DEFAULT_PAGE_DIR = PROJECT_ROOT / "gt_segmentation" / "images"
PAGE_SUFFIX = ".png"

# A label box encloses a word, not a room. Measured on a 4500x3000 GT page the
# adapter's boxes run 0.003%-0.007% of page area, so this ceiling leaves ~140x
# headroom over the largest observed box while still sitting well below a
# wall-to-wall `room_extent` box (a classroom on that page is ~2%). That gap is
# the point: the threshold has to be tight enough to catch a convention
# regression, which an earlier 10% ceiling was not -- it cleared every real box
# by 1377x and would have waved `room_extent` through. It is not an accuracy
# bound; room_replay measures accuracy properly.
MAX_LABEL_BOX_AREA_FRACTION = 0.01


def _resolve(env_var: str, default: Path) -> Path:
    override = os.environ.get(env_var)
    return Path(override) if override else default


def _require_environment() -> tuple[Path, Path]:
    """Skip unless this machine can actually run adapter inference."""
    import importlib.util

    if importlib.util.find_spec(UNSLOTH_MODULE) is None:
        pytest.skip(f"{UNSLOTH_MODULE} is not installed in this interpreter")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device: the 4-bit base model cannot be loaded")

    adapter = _resolve(ADAPTER_ENV, DEFAULT_ADAPTER)
    if not (adapter / "adapter_config.json").exists():
        pytest.skip(f"no adapter at {adapter} (override with {ADAPTER_ENV})")

    page_dir = _resolve(PAGE_ENV, DEFAULT_PAGE_DIR)
    pages = sorted(page_dir.glob(f"*{PAGE_SUFFIX}")) if page_dir.is_dir() else []
    if page_dir.is_file():
        pages = [page_dir]
    if not pages:
        pytest.skip(f"no page image under {page_dir} (override with {PAGE_ENV})")

    return adapter, pages[0]


@pytest.fixture(scope="module")
def adapter_rooms():
    """Rooms the fine-tuned adapter detects on one real GT page.

    Module-scoped: loading an 8B 4-bit model plus LoRA is the expensive part,
    and every test here asks a different question of the same output.
    """
    adapter, page = _require_environment()

    config = VLMConfig()
    config.backend = "unsloth"
    config.unsloth_adapter_path = adapter
    config.unsloth_room_prompt = "sft_label"

    backend = UnslothQwenBackend(config)
    try:
        rooms = backend.detect_rooms(page)
        # Captured before release(), which drops the model: this is the only
        # window in which the live graph can be inspected.
        attached = {
            name: {"r": settings.r, "lora_alpha": settings.lora_alpha}
            for name, settings in getattr(backend.model, "peft_config", {}).items()
        }
        lora_layers = sum(
            1 for name, _ in backend.model.named_modules() if name.endswith("lora_A")
        )
    finally:
        backend.release()

    from PIL import Image

    with Image.open(page) as image:
        width, height = image.size

    return {
        "rooms": rooms,
        "page": page,
        "width": width,
        "height": height,
        "adapter": adapter,
        "attached": attached,
        "lora_layers": lora_layers,
    }


def test_the_adapter_is_attached_to_the_live_graph(adapter_rooms):
    """The failure this whole file exists for: a run that scores the base
    model while reporting an adapter.

    `initialize` raises on a *missing* path, so that case is already covered
    by test_unsloth_adapter_path. The uncovered one is a path that loads but
    does not attach -- an incompatible rank, a silently-swallowed PEFT error.
    Asserting on parsed output cannot see it, because the base model answers
    this prompt perfectly well. So assert on the graph: LoRA layers present,
    and their rank/alpha matching what the adapter's own config declares.
    Read from the adapter rather than written here, so the assertion cannot
    drift from the weights it is checking.
    """
    import json

    declared = json.loads((adapter_rooms["adapter"] / "adapter_config.json").read_text())
    attached = adapter_rooms["attached"]

    assert attached, "no PEFT adapter on the model: it scored the BASE model"
    assert adapter_rooms["lora_layers"] > 0, "peft_config present but no lora_A layers"
    for name, settings in attached.items():
        assert settings["r"] == declared["r"], f"{name}: rank {settings['r']} != {declared['r']}"
        assert settings["lora_alpha"] == declared["lora_alpha"], (
            f"{name}: alpha {settings['lora_alpha']} != {declared['lora_alpha']}"
        )


def test_the_live_path_returns_rooms(adapter_rooms):
    """Zero rooms on a labelled GT page means generation or parsing broke.

    Not an accuracy claim -- room_replay measures that. This separates
    "the path runs" from "the path produces nothing", which a prompt
    mismatch or a truncated generation budget would cause.
    """
    assert adapter_rooms["rooms"], f"no rooms detected on {adapter_rooms['page'].name}"


def test_every_room_carries_a_non_empty_label(adapter_rooms):
    """The task is label text plus its box. A room without text is unusable
    downstream, and the exporter's taxonomy step cannot place it."""
    unlabelled = [
        room for room in adapter_rooms["rooms"]
        if not (room.get("room_name") or "").strip()
    ]

    assert not unlabelled, f"{len(unlabelled)} rooms have no label text"


def test_every_box_lies_inside_the_page(adapter_rooms):
    """Out-of-bounds boxes are the classic coordinate-space regression: the
    model answers in one space and the parser scales into another."""
    width, height = adapter_rooms["width"], adapter_rooms["height"]
    outside = [
        room["bbox"] for room in adapter_rooms["rooms"]
        if not (
            0 <= room["bbox"][0] < room["bbox"][2] <= width
            and 0 <= room["bbox"][1] < room["bbox"][3] <= height
        )
    ]

    assert not outside, f"{len(outside)} boxes fall outside {width}x{height}: {outside[:3]}"


def test_boxes_are_label_sized_not_page_sized(adapter_rooms):
    """The adapter was trained on the `label` convention. A box covering a
    large share of the page means it reverted to `room_extent` or collapsed
    to one giant detection -- both of which still parse cleanly."""
    page_area = adapter_rooms["width"] * adapter_rooms["height"]
    oversized = [
        (room.get("room_name"), room["bbox"]) for room in adapter_rooms["rooms"]
        if (room["bbox"][2] - room["bbox"][0]) * (room["bbox"][3] - room["bbox"][1])
        > page_area * MAX_LABEL_BOX_AREA_FRACTION
    ]

    assert not oversized, (
        f"{len(oversized)} boxes exceed {MAX_LABEL_BOX_AREA_FRACTION:.0%} "
        f"of the page: {oversized[:3]}"
    )
