"""MLflow tracking for the standalone scripts (trainer, D8 gate): the guard
contract holds when nothing is installed or enabled, a crash is never recorded
as FINISHED, and the gate verdict / training report reach MLflow as numbers.

A fake `mlflow` module stands in for the package, so this needs no server --
the live server path is exercised separately against a real tracking store.
"""

import argparse
import sys
import types
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "eval"))
sys.path.insert(0, str(project_root / "scripts"))

from preprocessing_annotations.config.config import MlflowConfig
from preprocessing_annotations.orchestration import mlflow_tracking
from preprocessing_annotations.orchestration.mlflow_tracking import (
    add_mlflow_arguments,
    mlflow_config_from_args,
    non_fatal,
    tracked_run,
)

from sft_gate import apply_gate, gate_metrics, log_verdict
from train_qwen3vl_unsloth import STEP_SERIES, log_training_run


class FakeMlflow(types.ModuleType):
    """Records every call the guard layer makes."""

    def __init__(self):
        super().__init__("mlflow")
        self.calls = []

    def set_tracking_uri(self, uri):
        self.calls.append(("set_tracking_uri", uri))

    def set_experiment(self, name):
        self.calls.append(("set_experiment", name))

    def start_run(self):
        self.calls.append(("start_run",))
        return types.SimpleNamespace(info=types.SimpleNamespace(run_id="run-1"))

    def end_run(self, status="FINISHED"):
        self.calls.append(("end_run", status))

    def log_metric(self, key, value, step=None):
        self.calls.append(("log_metric", key, value, step))

    def log_params(self, params):
        self.calls.append(("log_params", params))

    def set_tags(self, tags):
        self.calls.append(("set_tags", tags))

    def log_artifact(self, path):
        self.calls.append(("log_artifact", path))

    def named(self, name):
        return [call for call in self.calls if call[0] == name]


@pytest.fixture
def fake_mlflow(monkeypatch):
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setattr(mlflow_tracking, "_active_run_id", None)
    yield fake
    monkeypatch.setattr(mlflow_tracking, "_active_run_id", None)


@pytest.fixture
def enabled():
    return MlflowConfig(enabled=True, tracking_uri="http://tracking.invalid", experiment_name="exp")


def _report(recall, precision, category, per_document):
    return {
        "localization": {"label_recall": recall, "label_precision": precision},
        "category": {"accuracy": category},
        "per_document": {name: {"label_recall": r} for name, r in per_document.items()},
        "run_dir": "/runs/candidate",
    }


# --- guard contract ---------------------------------------------------------


def test_disabled_run_never_touches_mlflow(fake_mlflow):
    with tracked_run(MlflowConfig(enabled=False), "sft_gate") as run_id:
        assert run_id is None
    assert fake_mlflow.calls == []


def test_missing_package_runs_the_block_unaffected(monkeypatch, enabled):
    monkeypatch.setitem(sys.modules, "mlflow", None)  # `import mlflow` -> ImportError
    monkeypatch.setattr(mlflow_tracking, "_active_run_id", None)
    ran = []
    with tracked_run(enabled, "sft_gate") as run_id:
        ran.append(run_id)
    assert ran == [None]


def test_enabled_run_is_tagged_and_finished(fake_mlflow, enabled):
    with tracked_run(enabled, "sft_gate") as run_id:
        assert run_id == "run-1"
    assert fake_mlflow.named("set_tags") == [("set_tags", {"run_kind": "sft_gate"})]
    assert fake_mlflow.named("end_run") == [("end_run", "FINISHED")]


@pytest.mark.parametrize("failure", [RuntimeError("boom"), SystemExit("refuse to truncate")])
def test_a_crash_ends_the_run_failed_and_propagates(fake_mlflow, enabled, failure):
    with pytest.raises(type(failure)):
        with tracked_run(enabled, "sft_train"):
            raise failure
    assert fake_mlflow.named("end_run") == [("end_run", "FAILED")]


# --- env default and flag resolution ---------------------------------------


def _args(*argv):
    parser = argparse.ArgumentParser()
    add_mlflow_arguments(parser)
    return parser.parse_args(argv)


def test_tracking_is_off_by_default(monkeypatch):
    monkeypatch.delenv("MLFLOW_ENABLED", raising=False)
    assert mlflow_config_from_args(_args()).enabled is False


def test_env_enables_tracking_without_the_flag(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLED", "1")
    assert mlflow_config_from_args(_args()).enabled is True


def test_flag_enables_tracking_without_the_env(monkeypatch):
    monkeypatch.delenv("MLFLOW_ENABLED", raising=False)
    assert mlflow_config_from_args(_args("--enable-mlflow")).enabled is True


def test_the_absent_flag_cannot_switch_off_an_env_enable(monkeypatch):
    """The regression pipeline.main had: assigning the store_true default
    overwrote the environment's opt-in."""
    monkeypatch.setenv("MLFLOW_ENABLED", "true")
    assert mlflow_config_from_args(_args()).enabled is True


def test_tracking_uri_flag_overrides_the_env_default(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://from-env:1")
    assert mlflow_config_from_args(_args()).tracking_uri == "http://from-env:1"
    assert mlflow_config_from_args(_args("--mlflow-tracking-uri", "http://flag:2")).tracking_uri == "http://flag:2"


# --- gate -------------------------------------------------------------------


def _verdict(candidate_recall=0.60):
    baselines = [
        _report(0.55, 0.93, 0.95, {"a": 0.5, "b": 0.6}),
        _report(0.57, 0.92, 0.96, {"a": 0.52, "b": 0.58}),
    ]
    candidate = _report(candidate_recall, 0.93, 0.957, {"a": 0.55, "b": 0.5})
    return apply_gate(baselines, candidate, tolerance=0.01)


def test_gate_metrics_are_all_numeric_and_carry_the_decision():
    verdict = _verdict()
    metrics = gate_metrics(verdict)
    assert all(isinstance(v, float) for v in metrics.values())
    assert metrics["candidate_label_recall"] == pytest.approx(0.60)
    assert metrics["baseline_label_recall_max"] == pytest.approx(0.57)
    assert metrics["delta_label_recall"] == pytest.approx(verdict["deltas_vs_baseline_median"]["label_recall"])
    assert metrics["check_recall_above_baseline_spread"] == 1.0
    # document "b" fell below the worst baseline (0.5 < 0.58): the gate fails on it
    assert metrics["check_no_document_lost_recall"] == 0.0
    assert metrics["documents_losing_recall"] == 1.0
    assert metrics["passed"] == 0.0


def test_gate_metrics_have_one_series_per_check():
    verdict = _verdict()
    metrics = gate_metrics(verdict)
    for name in verdict["checks"]:
        assert f"check_{name}" in metrics


def test_gate_verdict_reaches_mlflow_with_its_report_artifact(fake_mlflow, enabled, tmp_path):
    verdict = _verdict()
    report = tmp_path / "verdict.json"
    report.write_text("{}")
    log_verdict(verdict, enabled, report)
    logged = {call[1]: call[2] for call in fake_mlflow.named("log_metric")}
    assert logged == gate_metrics(verdict)
    assert fake_mlflow.named("log_artifact") == [("log_artifact", str(report))]
    (params,) = fake_mlflow.named("log_params")
    assert params[1]["candidate_run_dir"] == "/runs/candidate"
    assert fake_mlflow.named("end_run") == [("end_run", "FINISHED")]


def test_gate_logging_is_a_noop_when_disabled(fake_mlflow):
    log_verdict(_verdict(), MlflowConfig(enabled=False), None)
    assert fake_mlflow.calls == []


# --- trainer ----------------------------------------------------------------

TRAIN_REPORT = {
    "model": "qwen3-vl-8b",
    "datasets": ["/data/batch2_sft"],
    "train_examples": 343,
    "eval_examples": 40,
    "dropped_overlong": {"train": 9, "val": 1},
    "steps": 258,
    "train_loss": 0.369,
    "peak_vram_gb": 9.75,
    "seconds": 5400.0,
    "hyperparameters": {"lora_r": 8, "learning_rate": 1e-4, "seed": 3407},
}
LOG_HISTORY = [
    {"step": 1, "loss": 2.48},
    {"step": 2, "loss": 2.1, "grad_norm": 0.7},
    {"step": 25, "eval_loss": 0.9},
    {"step": 258, "train_runtime": 5400.0},  # trainer's closing summary: no series key
]


def test_training_run_logs_scalars_and_per_step_series(fake_mlflow, enabled, tmp_path):
    report_path = tmp_path / "train_report.json"
    report_path.write_text("{}")
    with tracked_run(enabled, "sft_train"):
        log_training_run(TRAIN_REPORT, LOG_HISTORY, report_path)

    metrics = fake_mlflow.named("log_metric")
    scalars = {key: value for _, key, value, step in metrics if step is None}
    assert scalars["train_loss"] == pytest.approx(0.369)
    assert scalars["peak_vram_gb"] == pytest.approx(9.75)
    assert scalars["dropped_overlong_train"] == 9
    assert scalars["dropped_overlong_val"] == 1

    series = [(key, value, step) for _, key, value, step in metrics if step is not None]
    assert series == [("loss", 2.48, 1), ("loss", 2.1, 2), ("eval_loss", 0.9, 25)]
    assert {key for key, _, _ in series} <= set(STEP_SERIES)

    (_, params), = fake_mlflow.named("log_params")
    assert params["lora_r"] == "8" and params["model"] == "qwen3-vl-8b"
    assert fake_mlflow.named("log_artifact") == [("log_artifact", str(report_path))]


def test_training_log_is_a_noop_outside_a_run(fake_mlflow, tmp_path):
    log_training_run(TRAIN_REPORT, LOG_HISTORY, tmp_path / "train_report.json")
    assert fake_mlflow.calls == []


# --- observability never breaks the run it observes -------------------------

# sft_8b_run2's report, written before the trainer took several datasets:
# `dataset` not `datasets`, and no eval_loss_history.
DRIFTED_REPORT = {key: value for key, value in TRAIN_REPORT.items() if key != "datasets"}


@pytest.mark.parametrize("tracking", [True, False])
def test_a_drifted_report_never_fails_the_training_run(fake_mlflow, enabled, tmp_path, caplog, tracking):
    """Called after the adapter is saved, with a chain that aborts on nonzero
    exit -- and the payload is assembled even when tracking is off."""
    config = enabled if tracking else MlflowConfig(enabled=False)
    with tracked_run(config, "sft_train"):
        log_training_run(DRIFTED_REPORT, LOG_HISTORY, tmp_path / "train_report.json")
    assert "training log failed (non-fatal)" in caplog.text
    assert fake_mlflow.named("end_run") == ([("end_run", "FINISHED")] if tracking else [])


def test_a_broken_verdict_never_changes_the_gate_exit_path(fake_mlflow, enabled, caplog):
    """A traceback would exit 1 -- indistinguishable from 'the gate failed'."""
    verdict = _verdict()
    del verdict["checks"]
    log_verdict(verdict, enabled, None)
    assert "gate verdict log failed (non-fatal)" in caplog.text


def test_non_fatal_lets_exit_requests_through():
    with pytest.raises(SystemExit):
        with non_fatal("anything"):
            raise SystemExit(3)
