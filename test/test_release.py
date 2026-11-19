"""The release manifest, and the checks that stop a release shipping wrong.

The four failures these guard against all produce a successful-looking run:
the adapter not passed at all, a different adapter at the expected path, the
right adapter asked with the wrong prompt, and a detector that cannot load.
None of them raise in the pipeline, by design -- a research harness must be
free to mix a checkpoint with any prompt, or to run without a detector. So
the assertions live here.

No GPU, no model: every test reads a manifest, a config dict, a log or a path.
The detector checks import ultralytics (which is what the detectors do) but
never construct a model.
"""

import json
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "deploy"))

import release as release_module  # noqa: E402
from release import (  # noqa: E402
    DETECTOR_PROBES,
    MANIFEST_PATH,
    build_command,
    check_adapter,
    check_band,
    check_code,
    check_detectors,
    check_sam3,
    check_yolo,
    draw_if_asked,
    draw_run,
    load_manifest,
    preflight,
    probe_yolo,
    resolve_adapter,
    verify_run,
)

DETECTOR_SOURCES = [
    PROJECT_ROOT / "src" / "preprocessing_annotations" / "detection" / "yolo_detector.py",
    PROJECT_ROOT / "src" / "preprocessing_annotations" / "detection" / "sam3_exemplar_detector.py",
]


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
    """Extents cover part of the input only. A consumer expecting a polygon
    on every room must learn it here, not from an empty field in
    production."""
    assert "room_extent" in load_manifest()["known_limits"]


def test_an_optional_step_this_release_enables_must_report_itself():
    """A step switched on in the flags but absent from summary_must_report
    could produce nothing for a whole corpus and still pass postflight."""
    manifest = load_manifest()

    if "--use-vector-room-extent" in manifest["invocation"]["flags"]:
        assert "room_extent" in manifest["expectations"]["summary_must_report"]


@pytest.mark.parametrize(
    "key", load_manifest()["expectations"]["summary_must_report"]
)
def test_the_summary_keys_the_manifest_requires_are_what_the_pipeline_writes(key):
    """The check is a key match against the pipeline's own summary. If the
    pipeline renames the key, this check starts failing every run instead of
    the rename going unnoticed."""
    pipeline_source = (
        PROJECT_ROOT / "src" / "preprocessing_annotations" / "orchestration" / "pipeline.py"
    ).read_text()

    assert f'"{key}"' in pipeline_source, f"pipeline.py no longer writes {key!r}"


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


# --- preflight: can the detectors this release switches on load? -----------
#
# Both detectors catch every load failure, log it and return nothing, so a
# release with no weights where the pipeline looks finishes "successfully"
# with no door/window detections. Label recall cannot see it -- detections
# are tag-only -- so preflight is the only place it can be stopped before
# the GPU is spent.


def test_yolo_weights_missing_from_where_the_pipeline_looks_fail_preflight(tmp_path):
    """The wrapper-repo layout: the checkout is one level off, so the weights
    resolve to a path that does not exist. The verdict must name that path."""
    missing = tmp_path / "yolo_checkpoints" / "best.pt"

    verdict = check_yolo(missing)

    assert not verdict["ok"]
    assert not verdict["checkpoint_exists"]
    assert verdict["checkpoint"] == str(missing)


def test_yolo_weights_in_place_with_ultralytics_importable_pass_preflight(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights")

    verdict = check_yolo(weights)

    assert verdict["ok"], verdict
    assert verdict["import_failure"] is None


def test_a_missing_ultralytics_fails_preflight_even_with_the_weights_present(tmp_path, monkeypatch):
    """A stack built without ultralytics (it was missing from requirements.txt
    until this check was added) cannot load either detector, whatever weights
    are on disk."""
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights")
    monkeypatch.setitem(sys.modules, "ultralytics", None)

    verdict = check_yolo(weights)

    assert not verdict["ok"]
    assert verdict["checkpoint_exists"]
    assert "ultralytics" in verdict["import_failure"]


def test_an_uncached_sam3_checkpoint_fails_preflight_with_the_reason():
    """resolve_sam3_checkpoint raises when facebook/sam3 is not in the HF
    cache; the detector swallows that, so the reason has to surface here."""
    def not_cached():
        raise RuntimeError("SAM3 checkpoint not cached (facebook/sam3/sam3.pt)")

    verdict = check_sam3(not_cached)

    assert not verdict["ok"]
    assert verdict["checkpoint"] is None
    assert "not cached" in verdict["resolve_failure"]


def test_a_sam3_checkpoint_that_resolves_to_a_missing_file_fails_preflight(tmp_path):
    verdict = check_sam3(lambda: str(tmp_path / "sam3.pt"))

    assert not verdict["ok"]


def test_a_cached_sam3_checkpoint_passes_preflight(tmp_path):
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"weights")

    verdict = check_sam3(lambda: str(checkpoint))

    assert verdict["ok"], verdict


def test_the_yolo_probe_checks_the_path_the_pipeline_config_names():
    """Read from the pipeline's config, not rebuilt here: a copy of the path
    would eventually disagree with the one a run uses."""
    release_module._expose_pipeline()
    from preprocessing_annotations.config import YoloObjectDetectorConfig

    assert probe_yolo()["checkpoint"] == YoloObjectDetectorConfig().checkpoint_path


def _canned(ok):
    return lambda: {"ok": ok}


def _probes(**outcomes):
    """DETECTOR_PROBES' own flags and names, with canned probe results.

    Keys come from the real table, so a mistyped flag there changes what is
    checked here instead of being copied along with the typo.
    """
    return {
        flag: (name, _canned(outcomes.get(name, True)))
        for flag, (name, _) in DETECTOR_PROBES.items()
    }


def _manifest_without(flag):
    manifest = load_manifest()
    manifest["invocation"]["flags"] = [f for f in manifest["invocation"]["flags"] if f != flag]
    return manifest


def test_this_release_switches_on_both_detectors_so_both_are_checked():
    """A manifest flag with no probe, or a renamed flag, would leave a
    detector unchecked while the verdict still read ok."""
    verdict = check_detectors(load_manifest(), _probes())

    assert verdict["enabled"] == ["sam3", "yolo"]


def test_only_the_detectors_the_manifest_enables_are_checked():
    """A release that runs without SAM3 is not failed for SAM3's cache."""
    verdict = check_detectors(_manifest_without("--use-sam3-exemplar"), _probes(sam3=False))

    assert verdict["enabled"] == ["yolo"]
    assert verdict["ok"]


def test_one_failing_detector_fails_the_detector_check():
    verdict = check_detectors(load_manifest(), _probes(sam3=False))

    assert not verdict["ok"]


def test_a_failing_detector_check_fails_the_whole_preflight(monkeypatch):
    """The check has to be wired into the gate, not merely exist."""
    passing = lambda *args, **kwargs: {"ok": True}  # noqa: E731
    for name in ("check_code", "check_adapter", "check_band"):
        monkeypatch.setattr(release_module, name, passing)
    monkeypatch.setattr(release_module, "check_detectors", lambda manifest: {"ok": False})

    verdict = preflight(load_manifest())

    assert not verdict["ok"]
    assert verdict["checks"]["detectors"]["ok"] is False


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


CLEAN_LOG = "OCR extraction complete\nVLM room detection complete\n"
DETECTOR_FAILURE_LOG = (
    "YOLO object detector: model load failed: [Errno 2] No such file or directory: 'best.pt'\n"
)


CLEAN_SUMMARY = {
    "total_annotations": 48,
    "room_extent": {"rooms_with_extent": 311, "pages_with_extent": 28},
}


def _run_dir(tmp_path, config, logs=(CLEAN_LOG,), summary=CLEAN_SUMMARY):
    """A run directory as the pipeline leaves it: config, log(s), summary."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "pipeline_config.json").write_text(json.dumps(config))
    for index, text in enumerate(logs):
        (tmp_path / f"pipeline_2026-01-01_00000{index}.log").write_text(text)
    if summary is not None:
        (tmp_path / release_module.RUN_SUMMARY_NAME).write_text(json.dumps(summary))
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


def test_a_run_whose_detector_failed_to_load_fails_postflight(tmp_path):
    """Preflight showed the detector could load; the log says whether it did.
    The run finished and exited 0 -- only the log tells."""
    run = _run_dir(tmp_path, _config(), logs=(CLEAN_LOG + DETECTOR_FAILURE_LOG,))

    verdict = verify_run(load_manifest(), run)

    assert not verdict["ok"]
    failed = [f for f in verdict["fields"] if not f["ok"]]
    assert [f["field"] for f in failed] == ["log: 'model load failed'"]
    assert failed[0]["actual"] == 1


def test_load_failures_are_counted_across_every_log_in_the_run(tmp_path):
    """A resumed run leaves more than one pipeline_*.log; a failure in any of
    them is a failure of the run."""
    run = _run_dir(tmp_path, _config(), logs=(DETECTOR_FAILURE_LOG, CLEAN_LOG, DETECTOR_FAILURE_LOG))

    verdict = verify_run(load_manifest(), run)

    failed = [f for f in verdict["fields"] if not f["ok"]]
    assert failed[0]["actual"] == 2


def test_a_run_with_no_log_fails_rather_than_passing_vacuously(tmp_path):
    """No log means nothing was checked. That must not read as clean."""
    run = _run_dir(tmp_path, _config(), logs=())

    verdict = verify_run(load_manifest(), run)

    assert not verdict["ok"]
    assert "run log" in [f["field"] for f in verdict["fields"] if not f["ok"]]


def test_every_log_phrase_the_manifest_lists_is_actually_checked(tmp_path):
    manifest = load_manifest()

    verdict = verify_run(manifest, _run_dir(tmp_path, _config()))

    checked = {f["field"] for f in verdict["fields"]}
    for phrase in manifest["expectations"]["log_must_not_contain"]:
        assert f"log: {phrase!r}" in checked


@pytest.mark.parametrize("detector_source", DETECTOR_SOURCES, ids=lambda p: p.name)
def test_the_phrase_the_manifest_forbids_is_still_what_the_detectors_log(detector_source):
    """The postflight check is a string match on the detectors' own warning.
    If either detector rewords it, the check keeps passing and catches
    nothing -- the vacuous pass this file exists to prevent."""
    text = detector_source.read_text()

    for phrase in load_manifest()["expectations"]["log_must_not_contain"]:
        assert phrase in text, f"{detector_source.name} no longer logs {phrase!r}"


# --- postflight: did the optional steps this release enables actually run? --
#
# The extent step tallies a page it cannot find a source PDF for and moves
# on, so a deployment that stages PDFs elsewhere finishes with no extents and
# exits 0. Nothing downstream requires the field, so the absence reads as
# "this input had none".


def test_a_run_whose_extent_step_never_reported_itself_fails_postflight(tmp_path):
    run = _run_dir(tmp_path, _config(), summary={"total_annotations": 48})

    verdict = verify_run(load_manifest(), run)

    assert not verdict["ok"]
    assert "summary: room_extent" in [f["field"] for f in verdict["fields"] if not f["ok"]]


def test_the_stored_run_from_before_this_release_fails_the_extent_check():
    """A real negative, not a synthetic one: the pincheck run was made before
    the flag was added, so its summary has no room_extent at all."""
    run = PROJECT_ROOT.parent / "adapter_real_v1_b1275_pincheck"
    if not (run / release_module.RUN_SUMMARY_NAME).exists():
        pytest.skip(f"{run} not on this machine")

    failed = [f["field"] for f in release_module.check_summary(load_manifest(), run) if not f["ok"]]

    assert failed == ["summary: room_extent"]


def test_a_run_whose_pages_had_no_source_pdf_fails_postflight(tmp_path):
    """The step ran but could not see its input -- the failure a zero count
    alone cannot be told apart from."""
    summary = {"room_extent": {"rooms_with_extent": 0, "pages_without_source_pdf": 48}}
    run = _run_dir(tmp_path, _config(), summary=summary)

    verdict = verify_run(load_manifest(), run)

    assert not verdict["ok"]
    failed = [f for f in verdict["fields"] if not f["ok"]]
    assert [f["field"] for f in failed] == ["summary: room_extent.pages_without_source_pdf"]
    assert failed[0]["actual"] == 48


def test_a_run_that_legitimately_found_no_extents_still_passes(tmp_path):
    """Scanned and FLAT exports carry no wall layers. An empty result is a
    property of the input, not a broken release."""
    run = _run_dir(tmp_path, _config(), summary={"room_extent": {"pages": 12}})

    assert verify_run(load_manifest(), run)["ok"]


def test_a_run_with_no_summary_fails_rather_than_passing_vacuously(tmp_path):
    run = _run_dir(tmp_path, _config(), summary=None)

    verdict = verify_run(load_manifest(), run)

    assert not verdict["ok"]
    assert "run summary" in [f["field"] for f in verdict["fields"] if not f["ok"]]


def test_a_release_that_enables_no_optional_step_is_not_held_to_a_summary(tmp_path):
    """The expectations are declared, not assumed: a manifest that switches
    the step off must not require its tally."""
    manifest = load_manifest()
    manifest["expectations"]["summary_must_report"] = []
    manifest["expectations"]["summary_must_be_zero"] = []

    run = _run_dir(tmp_path, _config(), summary=None)

    assert verify_run(manifest, run)["ok"]


def test_every_summary_key_the_manifest_lists_is_actually_checked(tmp_path):
    manifest = load_manifest()

    verdict = verify_run(manifest, _run_dir(tmp_path, _config()))

    checked = {f["field"] for f in verdict["fields"]}
    expectations = manifest["expectations"]
    for dotted in expectations["summary_must_report"] + expectations["summary_must_be_zero"]:
        assert f"summary: {dotted}" in checked


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


# --- drawing the run -------------------------------------------------------
# The visualizer has always been able to draw a finished run. From a terminal
# nothing called it, so runs ended with annotations no one could look at.

RECORDER = """
import json, os, sys
from pathlib import Path

Path(os.environ["RECORD_PATH"]).write_text(json.dumps({
    "argv": sys.argv[1:],
    "pythonpath": os.environ.get("PYTHONPATH"),
}))
sys.exit(int(os.environ.get("EXIT_CODE", "0")))
"""


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    """A stand-in visualizer that records what it was given."""
    script = tmp_path / "recorder.py"
    script.write_text(RECORDER)
    record = tmp_path / "called.json"
    monkeypatch.setenv("RECORD_PATH", str(record))
    return script, record


def test_the_visualizer_is_pointed_at_the_run_and_told_where_to_import_from(
    recorder, tmp_path
):
    """The script imports the pipeline package before it parses an argument."""
    script, record = recorder

    assert draw_run(tmp_path / "run", script) == 0

    called = json.loads(record.read_text())
    assert called["argv"] == ["--input", str(tmp_path / "run")]
    assert called["pythonpath"].split(os.pathsep)[0] == str(PROJECT_ROOT / "src")


def test_an_existing_pythonpath_is_kept(recorder, tmp_path, monkeypatch):
    script, record = recorder
    monkeypatch.setenv("PYTHONPATH", "/already/here")

    draw_run(tmp_path / "run", script)

    assert "/already/here" in json.loads(record.read_text())["pythonpath"]


def test_nothing_is_drawn_unless_it_is_asked_for(recorder, tmp_path):
    """--visualize is opt-in: the service draws runs itself and must not
    draw them twice."""
    _, record = recorder

    draw_if_asked({"ok": True}, tmp_path / "run", wanted=False)

    assert not record.exists()


def test_a_run_that_is_not_a_valid_release_is_not_drawn(recorder, tmp_path, monkeypatch):
    """Pictures of a run that failed its postflight are pictures of nothing."""
    script, record = recorder
    monkeypatch.setattr(release_module, "VISUALIZE_SCRIPT", script)

    draw_if_asked({"ok": False}, tmp_path / "run", wanted=True)

    assert not record.exists()


def test_an_accepted_run_is_drawn(recorder, tmp_path, monkeypatch):
    script, record = recorder
    monkeypatch.setattr(release_module, "VISUALIZE_SCRIPT", script)

    draw_if_asked({"ok": True}, tmp_path / "run", wanted=True)

    assert json.loads(record.read_text())["argv"] == ["--input", str(tmp_path / "run")]


def test_a_drawing_that_fails_does_not_change_the_verdict(recorder, tmp_path, monkeypatch):
    """The detections are the product; the pictures are a view of them."""
    script, _ = recorder
    monkeypatch.setattr(release_module, "VISUALIZE_SCRIPT", script)
    monkeypatch.setenv("EXIT_CODE", "1")
    verdict = {"ok": True}

    draw_if_asked(verdict, tmp_path / "run", wanted=True)

    assert verdict == {"ok": True}, "drawing rewrote the verdict"


def test_the_visualizer_default_points_at_the_script_that_exists():
    """Not a phantom path: the flag is useless if it names nothing."""
    assert release_module.VISUALIZE_SCRIPT.exists()
