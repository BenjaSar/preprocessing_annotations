"""One release of the pipeline: pinned code, pinned adapter, pinned prompt.

The pipeline raises when an adapter directory is missing. It does not, and
should not, raise for the three ways a release goes wrong quietly:

  1. --unsloth-adapter is simply not passed. The base model runs, the output
     looks normal, and nothing in the artifacts says "this was not the
     fine-tune" unless someone reads pipeline_config.json.
  2. The adapter directory exists but holds different weights -- a retrain,
     a half-copied checkpoint, the byte-identical nested copy one level
     deeper. It loads happily.
  3. The adapter is loaded and asked with the wrong prompt. The adapter was
     trained on 'sft_label'; the backend's default is 'production'. There is
     no error, only worse output.

None of those are bugs in the pipeline -- a research harness has to be free
to mix a checkpoint with any prompt. They are bugs in *shipping*, so the
check belongs here, where "this exact thing is what we release" is the whole
point. Preflight proves intent before the GPU is spent; postflight reads the
run's own pipeline_config.json and proves what actually ran.

Deliberately not in src/: any edit there changes code_fingerprint, which
invalidates the pinned recall band and would cost a full re-measurement of
every seed in it. This layer sits above the frozen code and reads it.

Usage:
    python deploy/release.py --check
    python deploy/release.py --input <dir> --output <run dir>
    python deploy/release.py --verify-run <run dir>
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "eval"))

from report_io import emit_report

logger = logging.getLogger(__name__)

MANIFEST_PATH = Path(__file__).parent / "release.json"
CHUNK = 1 << 20


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


def current_fingerprint() -> str:
    """The fingerprint of the code in this checkout.

    Imported from the pipeline rather than recomputed, so this can never
    disagree with what a run records about itself.
    """
    sys.path.insert(0, str(project_root / "src"))
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


def preflight(manifest: Dict[str, Any], adapter_root: Optional[Path] = None) -> Dict[str, Any]:
    """Everything checkable before the GPU is spent. Pure; returns a verdict."""
    adapter = resolve_adapter(manifest, adapter_root)
    checks = {
        "code": check_code(manifest),
        "adapter": check_adapter(manifest, adapter),
        "band": check_band(manifest),
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    manifest = load_manifest(args.manifest)

    if args.verify_run is not None:
        verdict = verify_run(manifest, args.verify_run, args.adapter_root)
        logger.info(explain(verdict))
        emit_report(verdict, args.out)
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
    sys.exit(0 if verdict["ok"] else 1)
