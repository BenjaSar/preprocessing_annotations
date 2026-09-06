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

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

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


def end_run() -> None:
    """No-ops if no active run."""
    global _active_run_id
    if _active_run_id is None:
        return
    try:
        import mlflow

        mlflow.end_run()
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
