"""
Thin, fail-silent MLflow instrumentation layer.

Hard invariant: if mlflow is disabled, not installed, or the tracking
server is unreachable, every function here no-ops and the annotation
pipeline runs identically. This module must never raise -- observability
must never break a production annotation run.

`mlflow` is imported lazily, inside functions, never at module top level,
so any file importing this module works even without the package installed.
"""

from __future__ import annotations

import argparse
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from ..config.config import MlflowConfig

logger = logging.getLogger(__name__)

_active_run_id: Optional[str] = None


def _flatten(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(_flatten(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def start_run(config: Any) -> Optional[str]:
    """Start an MLflow run if config.enabled. Returns the run_id, or None
    if disabled/unavailable/failed -- never raises."""
    global _active_run_id
    if not getattr(config, "enabled", False):
        return None
    try:
        import mlflow

        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)
        run = mlflow.start_run()
        _active_run_id = run.info.run_id
        return _active_run_id
    except Exception as e:
        logger.warning(f"MLflow start_run failed, continuing without tracking: {e}")
        _active_run_id = None
        return None


def end_run(status: str = "FINISHED") -> None:
    """No-ops if no active run. `status` is an MLflow RunStatus name."""
    global _active_run_id
    if _active_run_id is None:
        return
    try:
        import mlflow

        mlflow.end_run(status=status)
    except Exception as e:
        logger.warning(f"MLflow end_run failed (non-fatal): {e}")
    finally:
        _active_run_id = None


def log_stage_metric(stage: str, key: str, value: float, step: Optional[int] = None) -> None:
    if _active_run_id is None:
        return
    try:
        import mlflow

        mlflow.log_metric(key, float(value), step=step)
    except Exception as e:
        logger.warning(f"MLflow log_metric failed for {stage}/{key} (non-fatal): {e}")


def log_metrics(stage: str, metrics: Dict[str, float]) -> None:
    for key, value in metrics.items():
        log_stage_metric(stage, key, value)


def log_params(params: Dict[str, Any]) -> None:
    if _active_run_id is None:
        return
    try:
        import mlflow

        flat = _flatten(params)
        # mlflow param values must be strings, and log_params has a per-key
        # value-length cap -- truncate rather than raise on oversized values
        # (e.g. argv lists, nested config blobs).
        safe = {k: str(v)[:500] for k, v in flat.items()}
        mlflow.log_params(safe)
    except Exception as e:
        logger.warning(f"MLflow log_params failed (non-fatal): {e}")


def log_tags(tags: Dict[str, Any]) -> None:
    if _active_run_id is None:
        return
    try:
        import mlflow

        mlflow.set_tags({k: str(v) for k, v in tags.items()})
    except Exception as e:
        logger.warning(f"MLflow log_tags failed (non-fatal): {e}")


def log_artifact(path: str) -> None:
    if _active_run_id is None:
        return
    try:
        import mlflow

        mlflow.log_artifact(path)
    except Exception as e:
        logger.warning(f"MLflow log_artifact failed for {path} (non-fatal): {e}")


@contextmanager
def tracked_run(config: Any, run_kind: str) -> Iterator[Optional[str]]:
    """One MLflow run around a standalone script (trainer, gate), tagged
    `run_kind` so these runs can be told apart from pipeline runs in the
    shared experiment. Yields the run_id, or None when tracking is
    disabled/unavailable.

    Ends the run FAILED if the block raises -- including SystemExit, which
    the trainer uses to refuse a truncating token budget -- and re-raises,
    so a crashed run is never recorded as FINISHED.
    """
    run_id = start_run(config)
    if run_id:
        log_tags({"run_kind": run_kind})
    try:
        yield run_id
    except BaseException:
        end_run(status="FAILED")
        raise
    else:
        end_run()


@contextmanager
def non_fatal(what: str) -> Iterator[None]:
    """Keep a failure while building a payload from breaking the run it
    observes: the log_* calls guard mlflow errors themselves, but the code
    that assembles their arguments (a report whose schema drifted, a verdict
    missing a key) is not covered by them, and it also runs when tracking is
    off. A warning, not a crash after the adapter is already saved.

    Catches Exception only: SystemExit and KeyboardInterrupt still propagate.
    """
    try:
        yield
    except Exception as e:
        logger.warning(f"MLflow {what} failed (non-fatal): {e!r}")


def add_mlflow_arguments(parser: argparse.ArgumentParser) -> None:
    """The pipeline's two MLflow flags, for standalone scripts."""
    parser.add_argument(
        "--enable-mlflow", action="store_true",
        help="Track this run in MLflow (default: off; MLFLOW_ENABLED=1 enables "
             "it without the flag). If mlflow is not installed or the server "
             "is unreachable, the run continues unaffected and a warning is logged.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri", default=None,
        help="Override MlflowConfig.tracking_uri (default: $MLFLOW_TRACKING_URI "
             "or http://127.0.0.1:5000).",
    )


def mlflow_config_from_args(args: argparse.Namespace) -> MlflowConfig:
    """Resolve MlflowConfig the way pipeline.main does: the env supplies the
    default, the flag can only turn tracking on."""
    config = MlflowConfig()
    config.enabled = config.enabled or args.enable_mlflow
    if args.mlflow_tracking_uri is not None:
        config.tracking_uri = args.mlflow_tracking_uri
    return config


@contextmanager
def traced(name: str, span_type: str = "UNKNOWN"):
    """Context manager, not a decorator: VLM/OCR/YOLO backend classes are
    defined at module import time, before argparse/config exist, so a
    decorator has no config to check at definition time. This checks the
    module-level active-run state at call time instead.

    Never swallows or alters exceptions raised inside the block.
    """
    if _active_run_id is None:
        yield None
        return
    t0 = time.time()
    try:
        yield None
    finally:
        duration_ms = (time.time() - t0) * 1000
        log_stage_metric(name, f"{name}_duration_ms", duration_ms)
