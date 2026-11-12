"""The release manifest, and the checks that stop a release shipping wrong.

The three failures these guard against all produce a successful-looking run:
the adapter not passed at all, a different adapter at the expected path, and
the right adapter asked with the wrong prompt. None of them raise in the
pipeline, by design -- a research harness must be free to mix a checkpoint
with any prompt. So the assertions live here.

No GPU, no model: every test reads a manifest and a config dict.
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "deploy"))

from release import (  # noqa: E402
    MANIFEST_PATH,
    build_command,
    check_adapter,
    check_band,
    check_code,
    load_manifest,
    resolve_adapter,
    verify_run,
)


# --- the manifest itself ---------------------------------------------------


def test_the_manifest_pins_the_prompt_the_adapter_was_trained_on():
    """The single most important pairing: an adapter asked the wrong way
    degrades silently, with no error anywhere."""
    flags = load_manifest()["invocation"]["flags"]

    assert "--unsloth-room-prompt" in flags
    assert flags[flags.index("--unsloth-room-prompt") + 1] == "sft_label"


def test_the_manifest_pins_the_adapter_by_content_not_only_by_path():
    """A present-but-wrong adapter loads happily; only a digest catches it."""
    digests = load_manifest()["adapter"]["digests"]

    assert "adapter_model.safetensors" in digests
    assert all(len(d) == 64 for d in digests.values())


def test_the_manifest_does_not_restate_the_recall_band():
    """The band has one home. Copied here it would eventually disagree."""
    manifest = load_manifest()

    assert manifest["expectations"]["band_pin"].endswith(".json")
    assert "band" not in manifest["expectations"]


def test_the_manifest_states_what_this_release_cannot_do():
    """Extent is 0. A consumer expecting polygons must learn it here, not
    from an empty field in production."""
    assert "room_extent" in load_manifest()["known_limits"]


def test_the_band_file_agrees_with_the_manifest_on_the_code():
    """A band measured on other code says nothing about this code."""
    verdict = check_band(load_manifest())

    assert verdict["ok"], verdict


# --- the command this release runs -----------------------------------------


def test_the_adapter_flag_is_built_from_the_resolved_path():
    """The path is stated once. A second copy in the flag list would drift."""
    manifest = load_manifest()
    command = build_command(manifest, Path("/in"), Path("/out"))
    argv = command["argv"]

    assert "--unsloth-adapter" in argv
    assert argv[argv.index("--unsloth-adapter") + 1] == str(resolve_adapter(manifest))
    assert manifest["invocation"]["flags"].count("--unsloth-adapter") == 0


def test_the_seed_and_sampling_reach_the_run_as_environment():
    command = build_command(load_manifest(), Path("/in"), Path("/out"))

    assert command["env"]["VLM_SEED"] == "1234"
    assert command["env"]["VLM_DO_SAMPLE"] == "true"


def test_the_command_keeps_the_rest_of_the_environment():
    """Dropping the inherited environment would strip PATH and CUDA vars."""
    command = build_command(load_manifest(), Path("/in"), Path("/out"))

    assert "PATH" in command["env"]


def test_an_adapter_root_override_relocates_the_weights():
    """A deployment that stages weights elsewhere must not need a new
    manifest."""
    manifest = load_manifest()

    relocated = resolve_adapter(manifest, Path("/mnt/models"))

    assert str(relocated).startswith("/mnt/models")
    assert relocated.name == Path(manifest["adapter"]["path"]).name


# --- preflight -------------------------------------------------------------


def test_a_missing_adapter_directory_fails_preflight(tmp_path):
    verdict = check_adapter(load_manifest(), tmp_path / "not_there")

    assert not verdict["ok"]
    assert all(f["status"] == "missing" for f in verdict["files"])


def test_an_adapter_with_different_weights_fails_preflight(tmp_path):
    """The failure this exists for: right path, wrong weights."""
    manifest = load_manifest()
    for name in manifest["adapter"]["digests"]:
        (tmp_path / name).write_bytes(b"not the trained weights")

    verdict = check_adapter(manifest, tmp_path)

    assert not verdict["ok"]
    assert all(f["status"] == "wrong_content" for f in verdict["files"])


def test_the_code_check_compares_against_the_live_fingerprint():
    verdict = check_code(load_manifest())

    assert verdict["actual"] == verdict["expected"], (
        "src/ has changed since this release was pinned. The recall band was "
        "measured on the pinned code, so it no longer describes this code -- "
        "re-measure every seed in the band, then re-pin the manifest."
    )


# --- postflight ------------------------------------------------------------


def _config(**overrides):
    manifest = load_manifest()
    config = {
        "code_fingerprint": manifest["code_fingerprint"],
        "vlm": {
            "unsloth_adapter_path": str(resolve_adapter(manifest)),
            "unsloth_room_prompt": "sft_label",
            "backend": "unsloth",
            "unsloth_model": "qwen3-vl-8b",
            "seed": 1234,
            "do_sample": True,
            "temperature": 1.0,
        },
    }
    config["vlm"].update(overrides.pop("vlm", {}))
    config.update(overrides)
    return config


def _run_dir(tmp_path, config):
    (tmp_path / "pipeline_config.json").write_text(json.dumps(config))
    return tmp_path


def test_a_run_matching_the_manifest_passes_postflight(tmp_path):
    verdict = verify_run(load_manifest(), _run_dir(tmp_path, _config()))

    assert verdict["ok"], verdict


def test_a_run_with_no_adapter_fails_postflight(tmp_path):
    """The quietest failure of all: the base model ran and nothing said so."""
    run = _run_dir(tmp_path, _config(vlm={"unsloth_adapter_path": None}))

    verdict = verify_run(load_manifest(), run)

    assert not verdict["ok"]
    failed = [f["field"] for f in verdict["fields"] if not f["ok"]]
    assert "vlm.unsloth_adapter_path" in failed


def test_a_run_with_the_wrong_prompt_fails_postflight(tmp_path):
    """Adapter loaded, asked the way it was not trained. No error anywhere
    in the pipeline -- only worse output."""
    run = _run_dir(tmp_path, _config(vlm={"unsloth_room_prompt": "production"}))

    verdict = verify_run(load_manifest(), run)

    assert not verdict["ok"]
    failed = [f["field"] for f in verdict["fields"] if not f["ok"]]
    assert "vlm.unsloth_room_prompt" in failed


def test_a_run_from_different_code_fails_postflight(tmp_path):
    run = _run_dir(tmp_path, _config(code_fingerprint="deadbeef1234"))

    verdict = verify_run(load_manifest(), run)

    assert not verdict["ok"]
    assert "code_fingerprint" in [f["field"] for f in verdict["fields"] if not f["ok"]]


def test_a_run_with_a_different_seed_fails_postflight(tmp_path):
    """Determinism is the release's reproducibility claim; a changed seed
    breaks it without breaking anything visible."""
    run = _run_dir(tmp_path, _config(vlm={"seed": 777}))

    verdict = verify_run(load_manifest(), run)

    assert not verdict["ok"]


def test_sampling_flags_compare_across_type(tmp_path):
    """The manifest holds env strings, the config holds parsed values.
    'true' must match True, not fail on str != bool."""
    verdict = verify_run(load_manifest(), _run_dir(tmp_path, _config()))

    sampling = [f for f in verdict["fields"] if f["field"] == "vlm.do_sample"]
    assert sampling and sampling[0]["ok"]


def test_a_directory_with_no_config_fails_rather_than_passing_vacuously(tmp_path):
    verdict = verify_run(load_manifest(), tmp_path)

    assert not verdict["ok"]
    assert "pipeline_config.json" in verdict["reason"]


def test_every_field_the_manifest_lists_is_actually_checked(tmp_path):
    """A field named in config_must_match but never compared would look like
    coverage without being any."""
    manifest = load_manifest()

    verdict = verify_run(manifest, _run_dir(tmp_path, _config()))

    checked = {f["field"] for f in verdict["fields"]}
    assert set(manifest["expectations"]["config_must_match"]) <= checked


def test_the_manifest_path_default_points_inside_the_project():
    """Not /tmp: a manifest parked there is a release that stops existing
    after a reboot."""
    assert MANIFEST_PATH.exists()
    assert PROJECT_ROOT in MANIFEST_PATH.parents
