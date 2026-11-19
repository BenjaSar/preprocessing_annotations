"""One release of the pipeline: pinned code, pinned adapter, pinned prompt.

The pipeline raises when an adapter directory is missing. It does not, and
should not, raise for the five ways a release goes wrong quietly:

  1. --unsloth-adapter is simply not passed. The base model runs, the output
     looks normal, and nothing in the artifacts says "this was not the
     fine-tune" unless someone reads pipeline_config.json.
  2. The adapter directory exists but holds different weights -- a retrain,
     a half-copied checkpoint, the byte-identical nested copy one level
     deeper. It loads happily.
  3. The adapter is loaded and asked with the wrong prompt. The adapter was
     trained on 'sft_label'; the backend's default is 'production'. There is
     no error, only worse output.
  4. A detector the release switches on cannot load -- its weights are not
     where the pipeline looks, ultralytics is not installed, or the SAM3
     checkpoint is not in the Hugging Face cache. Both detectors catch the
     failure, log "model load failed" and return nothing, so the run
     finishes "successfully" with no door/window detections. The recall
     band cannot see it: detections are tag-only and never accept or reject
     a room.
  5. An optional step the release switches on never actually ran. The extent
     step needs the run's source PDF, found from the page-image stem; where
     it cannot find one it tallies the page and moves on, so the run
     finishes with no extents and nothing downstream requires the field.
     The absence then reads as "this input had none".

None of those are bugs in the pipeline -- a research harness has to be free
to mix a checkpoint with any prompt, or to run without a detector. They are
bugs in *shipping*, so the check belongs here, where "this exact thing is
what we release" is the whole point. Preflight proves intent before the GPU
is spent; postflight reads the run's own pipeline_config.json and log and
proves what actually ran.

Deliberately not in src/: any edit there changes code_fingerprint, which
invalidates the pinned recall band and would cost a full re-measurement of
every seed in it. This layer sits above the frozen code and reads it.

Usage:
    python deploy/release.py --check
    python deploy/release.py --input <dir> --output <run dir> [--visualize]
    python deploy/release.py --verify-run <run dir> [--visualize]
"""

import argparse
import hashlib
import importlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "eval"))

from report_io import emit_report

logger = logging.getLogger(__name__)

MANIFEST_PATH = Path(__file__).parent / "release.json"
CHUNK = 1 << 20

# How the pipeline names the log it writes into every run directory
# (orchestration/pipeline.py, the FileHandler set up at run start).
RUN_LOG_GLOB = "pipeline_*.log"

# Where the pipeline tallies its post-processing steps, the optional ones
# included (orchestration/pipeline.py, written at the end of Step 4).
RUN_SUMMARY_NAME = "post_processing_summary.json"

# The run's own pictures. The script has always been able to draw a finished
# run; from a terminal nothing called it, so runs ended with annotations no
# one could look at. --visualize is that caller.
VISUALIZE_SCRIPT = Path(__file__).parent.parent / "scripts" / "visualize_bbox.py"


def load_manifest(path: Path = MANIFEST_PATH) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def digest(path: Path) -> str:
    """sha256 of a file, read in chunks -- the weights are ~100MB."""
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def resolve_adapter(manifest: Dict[str, Any], root: Optional[Path] = None) -> Path:
    """Where this checkout's adapter lives.

    The manifest stores a path relative to the repo root so a checkout
    elsewhere still resolves; --adapter-root overrides it for a deployment
    that stages weights somewhere else entirely.
    """
    declared = Path(manifest["adapter"]["path"])
    if root is not None:
        return (Path(root) / declared.name).resolve()
    return (project_root / declared).resolve()


def check_adapter(manifest: Dict[str, Any], adapter: Path) -> Dict[str, Any]:
    """Verify the adapter directory holds the weights this release names."""
    files = []
    for name, expected in manifest["adapter"]["digests"].items():
        path = adapter / name
        if not path.exists():
            files.append({"file": name, "status": "missing", "expected": expected})
            continue
        actual = digest(path)
        files.append({
            "file": name,
            "status": "ok" if actual == expected else "wrong_content",
            "expected": expected,
            "actual": actual,
        })
    return {
        "path": str(adapter),
        "exists": adapter.exists(),
        "files": files,
        "ok": adapter.exists() and all(f["status"] == "ok" for f in files),
    }


def _expose_pipeline() -> None:
    """Make this checkout's pipeline package importable, once."""
    src = str(project_root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def current_fingerprint() -> str:
    """The fingerprint of the code in this checkout.

    Imported from the pipeline rather than recomputed, so this can never
    disagree with what a run records about itself.
    """
    _expose_pipeline()
    from preprocessing_annotations.orchestration.code_fingerprint import source_fingerprint

    return source_fingerprint()


def check_code(manifest: Dict[str, Any]) -> Dict[str, Any]:
    actual = current_fingerprint()
    expected = manifest["code_fingerprint"]
    return {
        "expected": expected,
        "actual": actual,
        "ok": actual == expected,
    }


def check_band(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """The pinned recall band, read from its one home.

    Not copied into the manifest: a band in two files is a band that will
    eventually disagree with itself.
    """
    pin_path = project_root / manifest["expectations"]["band_pin"]
    if not pin_path.exists():
        return {"path": str(pin_path), "ok": False, "reason": "band file missing"}
    pin = json.loads(pin_path.read_text())
    return {
        "path": str(pin_path),
        "band": pin["band"],
        "code_fingerprint": pin["code_fingerprint"],
        "agrees_with_manifest": pin["code_fingerprint"] == manifest["code_fingerprint"],
        "ok": pin["code_fingerprint"] == manifest["code_fingerprint"],
    }


def _attempt(action: Callable[[], Any]) -> Tuple[Any, Optional[str]]:
    """(result, None) if `action` works, (None, why) if it raises anything.

    Exception, not ImportError or OSError: the detectors' own handlers
    swallow every type, so a check that is narrower than the handler it
    stands in for would pass a failure the run then hides.
    """
    try:
        return action(), None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


def _import_failure(module: str, name: str) -> Optional[str]:
    """None if `from <module> import <name>` works, else why it does not."""
    return _attempt(lambda: getattr(importlib.import_module(module), name))[1]


def check_yolo(checkpoint: Path) -> Dict[str, Any]:
    """Can the YOLO detector load: weights where it looks, ultralytics present."""
    import_failure = _import_failure("ultralytics", "YOLO")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.is_file(),
        "import_failure": import_failure,
        "ok": checkpoint.is_file() and import_failure is None,
    }


def check_sam3(resolve_checkpoint: Callable[[], str]) -> Dict[str, Any]:
    """Can the SAM3 detector load: checkpoint cached, predictor importable."""
    checkpoint, resolve_failure = _attempt(resolve_checkpoint)
    import_failure = _import_failure("ultralytics.models.sam", "SAM3SemanticPredictor")
    return {
        "checkpoint": checkpoint,
        "resolve_failure": resolve_failure,
        "import_failure": import_failure,
        "ok": checkpoint is not None and Path(checkpoint).is_file() and import_failure is None,
    }


def probe_yolo() -> Dict[str, Any]:
    """check_yolo against the path the pipeline itself will use.

    Read from the pipeline's config rather than rebuilt here, for the same
    reason the fingerprint is: this can then never disagree with the run.
    """
    _expose_pipeline()
    from preprocessing_annotations.config import YoloObjectDetectorConfig

    return check_yolo(Path(YoloObjectDetectorConfig().checkpoint_path))


def probe_sam3() -> Dict[str, Any]:
    _expose_pipeline()
    from preprocessing_annotations.config import resolve_sam3_checkpoint

    return check_sam3(resolve_sam3_checkpoint)


# The detectors a release can switch on, keyed by the flag that switches each
# on. The manifest says which are on; this says how to prove each can load.
DETECTOR_PROBES: Dict[str, Tuple[str, Callable[[], Dict[str, Any]]]] = {
    "--use-yolo-objects": ("yolo", probe_yolo),
    "--use-sam3-exemplar": ("sam3", probe_sam3),
}


def check_detectors(
    manifest: Dict[str, Any],
    probes: Dict[str, Tuple[str, Callable[[], Dict[str, Any]]]] = DETECTOR_PROBES,
) -> Dict[str, Any]:
    """Every detector this release's flags enable must be able to load.

    Derived from the manifest's flags, so a release that drops a detector is
    not held to it and a new flag cannot be enabled without a probe.
    """
    flags = manifest["invocation"]["flags"]
    results = {name: probe() for flag, (name, probe) in probes.items() if flag in flags}
    return {
        "enabled": sorted(results),
        "detectors": results,
        "ok": all(result["ok"] for result in results.values()),
    }


def preflight(manifest: Dict[str, Any], adapter_root: Optional[Path] = None) -> Dict[str, Any]:
    """Everything checkable before the GPU is spent. Pure; returns a verdict."""
    adapter = resolve_adapter(manifest, adapter_root)
    checks = {
        "code": check_code(manifest),
        "adapter": check_adapter(manifest, adapter),
        "band": check_band(manifest),
        "detectors": check_detectors(manifest),
    }
    return {
        "release": manifest["name"],
        "stage": "preflight",
        "checks": checks,
        "ok": all(c["ok"] for c in checks.values()),
    }


def build_command(
    manifest: Dict[str, Any],
    input_dir: Path,
    output_dir: Path,
    adapter_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """The exact argv and env this release runs. No flag is written twice.

    --unsloth-adapter is appended from the resolved path rather than stored
    in the manifest's flag list, so the path stays a single source.
    """
    adapter = resolve_adapter(manifest, adapter_root)
    argv = [
        sys.executable,
        str(project_root / "main.py"),
        "-i", str(input_dir),
        "-o", str(output_dir),
        *manifest["invocation"]["flags"],
        "--unsloth-adapter", str(adapter),
    ]
    env = dict(os.environ)
    env.update(manifest["invocation"]["env"])
    return {"argv": argv, "env": env}


def _lookup(config: Dict[str, Any], dotted: str) -> Any:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _occurrences(paths: List[Path], phrase: str) -> int:
    """How many times `phrase` appears across `paths`, streamed line by line."""
    total = 0
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as handle:
            total += sum(line.count(phrase) for line in handle)
    return total


def check_logs(manifest: Dict[str, Any], run_dir: Path) -> List[Dict[str, Any]]:
    """Hold the run's own log to the phrases a clean run never writes.

    Preflight can only show a detector *could* load. The log is the one place
    the run says whether it did -- and a run with no log proves nothing, so
    that fails rather than passing vacuously.
    """
    logs = sorted(Path(run_dir).glob(RUN_LOG_GLOB))
    if not logs:
        return [{"field": "run log", "expected": RUN_LOG_GLOB, "actual": None, "ok": False}]
    fields = []
    for phrase in manifest["expectations"]["log_must_not_contain"]:
        count = _occurrences(logs, phrase)
        fields.append({
            "field": f"log: {phrase!r}",
            "expected": 0,
            "actual": count,
            "ok": count == 0,
        })
    return fields


def check_summary(manifest: Dict[str, Any], run_dir: Path) -> List[Dict[str, Any]]:
    """Hold the run's own summary to the optional steps this release enables.

    A step that is switched on but never actually ran leaves a run that looks
    complete: nothing downstream requires its field, so its absence reads as
    "this input had none" rather than "this step did not happen". The extent
    step is the live case -- it needs the run's source PDF, found from the
    page-image stem, and a deployment that stages PDFs elsewhere produces
    zero extents and still exits 0.

    Two kinds of expectation, both declared by the manifest so a release that
    switches the step off is not held to it: keys that must be reported at
    all, and tallies that must be zero. A legitimately empty result (no CAD
    layers in the input) stays a pass -- only a step that could not see its
    input fails.
    """
    expectations = manifest["expectations"]
    required = expectations.get("summary_must_report", [])
    zeroed = expectations.get("summary_must_be_zero", [])
    if not required and not zeroed:
        return []

    path = Path(run_dir) / RUN_SUMMARY_NAME
    if not path.exists():
        return [{
            "field": "run summary",
            "expected": RUN_SUMMARY_NAME,
            "actual": None,
            "ok": False,
        }]

    summary = json.loads(path.read_text())
    fields = []
    for dotted in required:
        actual = _lookup(summary, dotted)
        fields.append({
            "field": f"summary: {dotted}",
            "expected": "reported",
            "actual": actual,
            "ok": actual is not None,
        })
    for dotted in zeroed:
        actual = _lookup(summary, dotted) or 0
        fields.append({
            "field": f"summary: {dotted}",
            "expected": 0,
            "actual": actual,
            "ok": actual == 0,
        })
    return fields


def verify_run(
    manifest: Dict[str, Any],
    run_dir: Path,
    adapter_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Read what the run recorded about itself and hold it to the manifest.

    Preflight says what we meant to run. This says what ran. They differ
    whenever a flag is dropped, renamed or overridden downstream -- which is
    exactly the failure no amount of pre-run checking can catch.
    """
    config_path = Path(run_dir) / "pipeline_config.json"
    if not config_path.exists():
        return {
            "release": manifest["name"],
            "stage": "postflight",
            "run_dir": str(run_dir),
            "ok": False,
            "reason": f"no pipeline_config.json in {run_dir}",
        }

    config = json.loads(config_path.read_text())
    expected_adapter = resolve_adapter(manifest, adapter_root)
    recorded_adapter = _lookup(config, "vlm.unsloth_adapter_path")

    fields = [{
        "field": "code_fingerprint",
        "expected": manifest["code_fingerprint"],
        "actual": config.get("code_fingerprint"),
        "ok": config.get("code_fingerprint") == manifest["code_fingerprint"],
    }, {
        "field": "vlm.unsloth_adapter_path",
        "expected": str(expected_adapter),
        "actual": recorded_adapter,
        "ok": recorded_adapter is not None
        and Path(recorded_adapter).resolve() == expected_adapter,
    }]

    declared = _declared_values(manifest)
    for dotted in manifest["expectations"]["config_must_match"]:
        actual = _lookup(config, dotted)
        expected = declared.get(dotted)
        fields.append({
            "field": dotted,
            "expected": expected,
            "actual": actual,
            "ok": expected is None or _equal(expected, actual),
        })

    fields.extend(check_logs(manifest, run_dir))
    fields.extend(check_summary(manifest, run_dir))

    return {
        "release": manifest["name"],
        "stage": "postflight",
        "run_dir": str(run_dir),
        "fields": fields,
        "ok": all(f["ok"] for f in fields),
    }


def _declared_values(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """What the manifest says each checked config field should be.

    Derived from the flags and env the release actually passes, so the
    manifest never states a value twice and the two can never drift.
    """
    flags = manifest["invocation"]["flags"]
    env = manifest["invocation"]["env"]
    pairs = {}
    for i, token in enumerate(flags):
        if token.startswith("--") and i + 1 < len(flags) and not flags[i + 1].startswith("--"):
            pairs[token] = flags[i + 1]
    return {
        "vlm.unsloth_room_prompt": pairs.get("--unsloth-room-prompt"),
        "vlm.backend": pairs.get("--vlm-backend"),
        "vlm.unsloth_model": pairs.get("--unsloth-model"),
        "vlm.seed": env.get("VLM_SEED"),
        "vlm.do_sample": env.get("VLM_DO_SAMPLE"),
        "vlm.temperature": env.get("VLM_TEMPERATURE"),
    }


def _equal(expected: Any, actual: Any) -> bool:
    """Compare a manifest string against a typed config value.

    The manifest holds env vars, which are strings; the config holds the
    parsed bool/int/float. Compare by the config's type rather than
    stringifying, so 'true' matches True and '1.0' matches 1.0.
    """
    if isinstance(actual, bool):
        return (str(expected).strip().lower() in {"true", "1", "yes"}) == actual
    if isinstance(actual, (int, float)) and not isinstance(actual, bool):
        try:
            return float(expected) == float(actual)
        except (TypeError, ValueError):
            return False
    return str(expected) == str(actual)


def draw_run(run_dir: Path, script: Optional[Path] = None) -> int:
    """Draw a finished run with the pipeline's own visualizer.

    A subprocess, not an import: the script imports the pipeline package at
    module scope and this process has only put `src` on its own sys.path, so
    the child is told where to import from.

    `script` is resolved at call time rather than bound as a default, so the
    module constant is the one source of the path.
    """
    script = VISUALIZE_SCRIPT if script is None else script
    argv = [sys.executable, str(script), "--input", str(run_dir)]
    env = dict(os.environ)
    src = str(project_root / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else src
    logger.info("drawing %s", run_dir)
    return subprocess.run(argv, env=env).returncode


def draw_if_asked(verdict: Dict[str, Any], run_dir: Path, wanted: bool) -> None:
    """Draw the run, if asked and if this release accepted it.

    Two rules, both the same as the service's:
      - a run that failed its postflight is not drawn. Pictures of an invalid
        release are pictures of nothing.
      - a drawing that fails never changes the verdict or the exit code. The
        detections are the product; the pictures are a view of them.
    """
    if not wanted:
        return
    if not verdict["ok"]:
        logger.warning("not drawing: this run is not a valid release")
        return
    exit_code = draw_run(run_dir)
    if exit_code != 0:
        logger.error(
            "the run is a valid release but could not be drawn (visualizer exited %s)",
            exit_code,
        )


def explain(verdict: Dict[str, Any]) -> str:
    if verdict["ok"]:
        return f"{verdict['release']} {verdict['stage']}: ok"
    lines = [f"{verdict['release']} {verdict['stage']}: FAILED"]
    if "reason" in verdict:
        lines.append(f"  {verdict['reason']}")
    for name, check in verdict.get("checks", {}).items():
        if not check["ok"]:
            lines.append(f"  {name}: {json.dumps(check)}")
    for field in verdict.get("fields", []):
        if not field["ok"]:
            lines.append(
                f"  {field['field']}: expected {field['expected']!r}, "
                f"got {field['actual']!r}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--adapter-root", type=Path, default=None,
                        help="Directory holding the adapter, if not the repo's sibling")
    parser.add_argument("--check", action="store_true",
                        help="Preflight only; run nothing")
    parser.add_argument("--verify-run", type=Path, default=None,
                        help="Postflight an existing run directory")
    parser.add_argument("-i", "--input", type=Path, default=None)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Also write the verdict here")
    parser.add_argument("--visualize", action="store_true",
                        help="Draw the run with scripts/visualize_bbox.py once it is accepted")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    manifest = load_manifest(args.manifest)

    if args.verify_run is not None:
        verdict = verify_run(manifest, args.verify_run, args.adapter_root)
        logger.info(explain(verdict))
        emit_report(verdict, args.out)
        draw_if_asked(verdict, args.verify_run, args.visualize)
        sys.exit(0 if verdict["ok"] else 1)

    verdict = preflight(manifest, args.adapter_root)
    logger.info(explain(verdict))
    if args.check or not verdict["ok"]:
        emit_report(verdict, args.out)
        sys.exit(0 if verdict["ok"] else 1)

    if args.input is None or args.output is None:
        parser.error("--input and --output are required to run (or pass --check)")

    command = build_command(manifest, args.input, args.output, args.adapter_root)
    logger.info("running %s", " ".join(command["argv"]))
    result = subprocess.run(command["argv"], env=command["env"])
    if result.returncode != 0:
        logger.error("pipeline exited %s", result.returncode)
        sys.exit(result.returncode)

    verdict = verify_run(manifest, args.output, args.adapter_root)
    logger.info(explain(verdict))
    emit_report(verdict, args.out)
    draw_if_asked(verdict, args.output, args.visualize)
    sys.exit(0 if verdict["ok"] else 1)
