# MLflow self-hosted setup

Tracking is opt-in and never required: with it off, or with `mlflow` missing, or
with the store unreachable, every run behaves exactly as before and logs a warning.

Run everything below from `/home/ubuntu/floorplan_classifier/VLM` (sibling of
`preprocessing_annotations/`). Requires `mlflow==3.17.2` (pinned in
`requirements.txt`; the 2.19.0 pin conflicted with `datasets`' `pyarrow>=21`).

## Two ways to record: pick by what else is running

**No server (use this while a pipeline run or training job holds the GPU box).**
Clients write straight to the SQLite store. Nothing is left running, so it costs
no RAM beside the job. A pipeline run peaks near 14 GB of the 15.7 GB on this
machine, and a default `mlflow server` adds 1.5 GB or more across 10 processes
(RSS summed, so shared pages are double-counted).

**Server (for the UI).** Start it on demand, when nothing memory-heavy is running.

## One-time store setup

Pre-create the experiment with a fixed artifact root. Without this, a client
writing straight to SQLite puts artifacts in `./mlruns` relative to whatever
directory it was started from.

```bash
MLFLOW_TRACKING_URI=sqlite:///$(pwd)/mlflow_data/mlflow.db vlm/bin/python -c \
  "import mlflow; mlflow.create_experiment('floorplan-annotation-pipeline', artifact_location='$(pwd)/mlflow_data/artifacts')"
```

## Recording a chain of runs (no server)

One environment default covers the pipeline, the trainer and the D8 gate, so a
chain script needs no per-command flags:

```bash
export MLFLOW_ENABLED=1
export MLFLOW_TRACKING_URI=sqlite:///$(pwd)/mlflow_data/mlflow.db
export MLFLOW_DISABLE_AGENT_HINT=1   # silences a hint line mlflow prints at import
```

`MLFLOW_ENABLED` is read by `MlflowConfig`; unset means off. A flag can only turn
tracking on, never off, so `--enable-mlflow` and `MLFLOW_ENABLED=1` agree.

## Browsing the UI (server)

```bash
MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false mlflow server \
  --backend-store-uri sqlite:///$(pwd)/mlflow_data/mlflow.db \
  --default-artifact-root $(pwd)/mlflow_data/artifacts \
  --host 127.0.0.1 --port 5000 --workers 1
curl -sf http://127.0.0.1:5000/health
```

`MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false` drops the job runner and its `huey`
consumers (10 processes down to 2, ~0.5 GB RSS measured); nothing here uses them. The server reads the same store the no-server clients wrote. With the
server up, point clients at it instead: `MLFLOW_TRACKING_URI=http://127.0.0.1:5000`.

## What gets recorded

All runs land in the `floorplan-annotation-pipeline` experiment, told apart by the
`run_kind` tag (filter: `tags.run_kind = 'sft_gate'`).

| `run_kind` | Written by | Metrics | Artifact |
|---|---|---|---|
| `pipeline` | `main.py` | stage durations/outcomes, corroboration totals | — |
| `sft_train` | `scripts/train_qwen3vl_unsloth.py` | `train_loss`, `peak_vram_gb`, `steps`, `seconds`, example counts, overlong drops; per-step `loss` and `eval_loss` | `train_report.json` |
| `sft_gate` | `eval/sft_gate.py` | candidate recall/precision/category accuracy, deltas vs baseline median, baseline spread, one 0/1 series per check, `passed` | the verdict JSON (with `--out`) |

A run that raises (including the trainer refusing an overlong token budget) ends
`FAILED`, not `FINISHED`. Training is logged after it finishes, from the trainer's
own history, with `report_to="none"` kept on `SFTConfig`: the built-in MLflow
integration would call mlflow outside the fail-silent guard.

## Flags

Pipeline, trainer and gate all take:
- `--enable-mlflow` — turn on tracking for this run (default off).
- `--mlflow-tracking-uri` — override the tracking URI (default `$MLFLOW_TRACKING_URI`, else `http://127.0.0.1:5000`).

Pipeline only:
- `--eval-gt-dir <path>` — optional ground-truth annotations dir; if set, runs `bbox_metrics.evaluate_dataset` after the run and logs precision/recall/IoU-family metrics.
