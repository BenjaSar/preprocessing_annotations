# MLflow self-hosted setup

Run as a separate process, independent of any pipeline run:

```bash
mlflow server \
  --backend-store-uri sqlite:///$(pwd)/mlflow_data/mlflow.db \
  --default-artifact-root $(pwd)/mlflow_data/artifacts \
  --host 127.0.0.1 --port 5000
```

Run from `/home/ubuntu/floorplan_classifier/VLM` (sibling of `preprocessing_annotations/`). Verify with:

```bash
curl -sf http://127.0.0.1:5000/health
```

Pipeline flags:
- `--enable-mlflow` — turn on tracking for a run (default off; if the server is down or `mlflow` isn't installed, the run continues unaffected and logs a warning).
- `--mlflow-tracking-uri` — override the tracking URI (default `http://127.0.0.1:5000`, or `$MLFLOW_TRACKING_URI`).
- `--eval-gt-dir <path>` — optional ground-truth annotations dir; if set, runs `bbox_metrics.evaluate_dataset` after the run and logs precision/recall/IoU-family metrics.
