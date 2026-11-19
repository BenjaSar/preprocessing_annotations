# preprocessing_annotations — Floor Plan Room Annotation Pipeline

Turns architectural/MEP floor plan PDFs into room annotations: a labelled, located
box per room, exported as COCO, Label Studio JSON, and Qwen3-VL SFT JSONL.

The pipeline combines PDF rasterization, OCR, a vision-language model (Claude API or
a local Qwen3-VL, optionally with a fine-tuned LoRA adapter), SAM label→room
expansion, and YOLO/SAM3 door-window detection, then reconciles those sources into a
single annotation per page.

**What it produces:** room *labels* with *label-sized* boxes.
**What it does not produce:** room *spatial extents* (polygons). See
[Known limits](#known-limits) — this is stated up front because a consumer expecting
polygons will get none.

---

## Status

Measured on this checkout, 2026-11-12. Numbers below come from the artifacts and
commands cited, not from estimates.

| Item | Value | Source |
|---|---|---|
| Fast test suite | 729 passed, 1 skipped, 16 deselected (64 s) | `pytest -q` in this directory |
| Slow (GPU/network) tests | 16, deselected by default | `pytest -m slow --collect-only` |
| Test files / source modules | 78 / 78 | `test/test_*.py`, `src/preprocessing_annotations/**/*.py` |
| Shipped release | `rooms-sft-real-v1` | [`deploy/release.json`](deploy/release.json) |
| Shipped adapter label recall | 0.6072 – 0.6112 over 3 seeds, 13-document GT set | [`eval/pins/adapter_real_v1_band.json`](eval/pins/adapter_real_v1_band.json) |
| Room extent (IoU) | 0.0 — not produced | `release.json: known_limits.room_extent` |
| Room taxonomy | 44 mandatory classes, 17 extended types, 24 VLM prompt categories | `automation/taxonomy.py` |
| Pinned code fingerprint | `92ac4c7a44d3` | `deploy/release.json` |

Whole-pipeline determinism is **measured**, not assumed: a fresh 3 h 28 m GPU re-run
at `VLM_SEED=1234` on 2026-11-12 reproduced the entire scored replay report
bit-identically (`eval/adapter_pin.py`, `../adapter_pin_check.sh`).

---

## Repository layout

```
preprocessing_annotations/
├── main.py                 # CLI entry point → orchestration.pipeline:main
├── run.sh, QUICK_START.sh  # wrappers that source .env and call main.py
├── pytest.ini              # testpaths=test, `slow` marker deselected by default
├── src/preprocessing_annotations/
│   ├── config/             # all tunables (PDFConfig, OCRConfig, VLMConfig, …)
│   ├── ingestion/          # PDF extraction, OCR, tiling, crop re-read, vector walls
│   ├── detection/          # SAM, YOLO, SAM3 exemplar, door/window, attachment
│   ├── vlm/                # backends, prompts, tiled rooms, reconciliation
│   ├── export/             # COCO / Label Studio / Qwen SFT / group split
│   ├── automation/         # taxonomy, schema, validators, abbreviation recovery
│   ├── bbox/               # validation, metrics, visualization
│   ├── orchestration/      # pipeline, run lock, code fingerprint, MLflow
│   └── tools/              # vocabulary and wall-band CLIs
├── eval/                   # 37 offline evaluation / gate / probe scripts
│   └── pins/               # pinned metric bands the test suite enforces
├── deploy/                 # pinned release manifest + preflight/postflight checker
├── scripts/                # training, splits, QA utilities
├── test/                   # 78 test files + fixtures
└── docs/                   # running guide, MLflow setup, cost analysis
```

---

## Installation

```bash
# The project venv already carries paddleocr 2.x, PyMuPDF, torch and unsloth.
source /home/ubuntu/floorplan_classifier/VLM/vlm/bin/activate
cd /home/ubuntu/floorplan_classifier/VLM/preprocessing_annotations
```

For a fresh environment:

```bash
pip install -e .          # installs the `annotate-pipeline` console script
pip install -r requirements.txt
```

`main.py` inserts `src/` on `sys.path`, so the pipeline runs without an editable
install as well.

**Credentials.** Only the `claude` backend needs one:

```bash
export ANTHROPIC_API_KEY=...        # or: source .env
```

**Weights** live outside the repository and are addressed by path: the SAM
checkpoint (`weights/sam_vit_h_4b8939.pth`), the YOLO door/window checkpoint, and
LoRA adapters under `../trained_adapters/`.

---

## Quick start

```bash
# 1. Local Qwen3-VL 8B via Unsloth — the flags every production run passes
python main.py --input ../test_input --output ./results_local \
  --use-vlm --vlm-backend unsloth --unsloth-model qwen3-vl-8b \
  --use-yolo-objects --use-sam3-exemplar --export-object-detections --verbose

# 2. OCR only — no VLM, no API cost, fastest
python main.py --input ../test_input --output ./results

# 3. Claude API backend — the CLI default, but not what production runs use
python main.py --input ../test_input --output ./results_claude --use-vlm
```

Verify which backend started:

```bash
grep "Creating VLM backend" results*/pipeline_*.log
```

### Running the shipped release

Do not assemble the flags by hand for a release run — a fine-tuned adapter asked
with the wrong prompt degrades **silently**. Use the release wrapper:

```bash
python deploy/release.py --check                      # preflight, no GPU
python deploy/release.py -i ../test_input -o ../run1  # preflight → run → postflight
python deploy/release.py --verify-run ../run1         # postflight an existing run
```

Exit 0 = the release is intact; exit 1 names the field that disagrees. Preflight
checks the code fingerprint, the adapter's content digests, the recall-band pin, and
that every detector the release enables can load (YOLO weights where the pipeline
looks, `ultralytics`, the SAM3 checkpoint in the Hugging Face cache). Postflight checks
the run's recorded config and that its `pipeline_*.log` holds no `model load failed`.
See [`deploy/README.md`](deploy/README.md).

---

## Pipeline stages

`orchestration/pipeline.py` runs the stages below in order. Stage names are the ones
printed in the run log.

| Step | Stage | Implementation |
|---|---|---|
| 1 | Resolve input to images (PDF → PNG, 150–300 DPI, multi-page) | `ingestion/pdf_extractor.py` |
| 2 | OCR text extraction + abbreviation recovery | `ingestion/ocr_extractor.py`, `ingestion/two_pass_ocr_extractor.py`, `automation/abbreviation_ocr_recovery.py` |
| 3 | VLM zero-shot room annotation (tiled above 3000 px), or OCR-only annotation when `--use-vlm` is off | `vlm/vlm_backend.py`, `vlm/tiled_rooms.py`, `ingestion/tile_splitter.py` |
| 3 (same loop) | VLM crop re-read of OCR fragments that look like room keywords, invoked inside Step 3's per-image loop (`orchestration/pipeline.py:824`) while the VLM is already loaded; on by default with an unsloth backend | `orchestration/crop_reread_step.py` |
| 3.5 | SAM label→room boundary expansion; object detection → room mapping | `detection/sam_segmenter.py`, `detection/yolo_detector.py`, `detection/sam3_exemplar_detector.py` |
| 4 | Post-processing: reconciliation, taxonomy resolution, filtering, validation | `vlm/semantic_reconciler.py`, `automation/sft_validator.py`, `bbox/bbox_validator.py` |
| 6 | Review prioritization | `export/exporters.py` |
| 7 | Export: COCO, Label Studio, object detections, SFT inputs | `export/exporters.py`, `export/object_exporter.py` |

Supporting behaviour wired into these stages: hallucination detection
(`vlm/hallucination_detector.py`), the door layer filter that discards wiring
misread as doors (`detection/door_layer_filter.py`, on by default), the window
strip detector, and vector-wall room extents (`ingestion/vector_walls.py`, pilot,
off by default).

### VLM backends

Three backends share one interface (`vlm/vlm_backend.py`). **`unsloth` (Qwen3-VL) is
the backend this project actually runs**: every baseline, every gated candidate and
the shipped release use it. `claude` is the CLI *default* — a convenience for a
quick API-backed smoke run, not the production path.

| Backend | Model | Cost | Hardware | Notes |
|---|---|---|---|---|
| `unsloth` | `qwen3-vl-8b` (also `qwen3-vl-2b`, `qwen3-vl-4b`, and `-thinking` variants) | free | 8B occupies the whole T4 (`../rebaseline_1275.sh:wait_for_free_gpu`); ~70% less VRAM than plain transformers (`vlm/vlm_backend.py:1048`) | Production and release backend; the only one that accepts `--unsloth-adapter` |
| `qwen` | `Qwen/Qwen2.5-VL-7B-Instruct` | free | ~6 GB VRAM at 4-bit (`config/config.py:472`) | HuggingFace transformers path |
| `claude` (CLI default) | `claude-haiku-4-5-20251001`; override with `--vlm-model` or `$VLM_MODEL` | API | none | Needs `ANTHROPIC_API_KEY` |

A LoRA adapter must be asked with the prompt it was trained on:
`--unsloth-room-prompt sft_label`. The backend default is `production`. Nothing in
the pipeline reports the mismatch — `deploy/release.py` is the thing that does.

---

## CLI reference

Produced from `orchestration/pipeline.py`'s argument parser.

**Input/output**

| Flag | Meaning |
|---|---|
| `--input`, `-i` | Input directory of PDFs (required) |
| `--output`, `-o` | Output directory (required) |
| `--skip-existing` | Skip images that already have annotations |
| `--verbose`, `-v` | Debug logging (the file handler is always DEBUG) |
| `--no-wait` | Exit code 3 if another run holds the lock, instead of waiting |

**Rasterization**

| Flag | Meaning |
|---|---|
| `--dpi` | Override DPI (default 200) |
| `--high-detail` / `--fast` | 300 DPI, more template scales / 150 DPI, fewer |
| `--enforce-min-edge-floor` | Raise DPI so small sheets reach `PDFConfig.min_longest_edge_px` (4500). Off by default |

**OCR**

| Flag | Meaning |
|---|---|
| `--ocr-backend {paddleocr,easyocr}` | Default `paddleocr`; `easyocr` is the legacy path |
| `--ocr-vlm-fallback` | Two-pass OCR Pass 2. Off by default — its boxes are not in page coordinates |
| `--no-crop-reread` | Disable the VLM crop re-read (on by default with an unsloth backend) |

**VLM**

| Flag | Meaning |
|---|---|
| `--use-vlm` | Enable VLM annotation |
| `--vlm-backend {claude,qwen,unsloth}` | Default `claude` |
| `--vlm-model` / `--qwen-model` / `--unsloth-model` | Per-backend model id |
| `--unsloth-adapter PATH` | LoRA adapter directory, trained against the same `--unsloth-model` |
| `--unsloth-room-prompt NAME` | Room prompt; use `sft_label` with an adapter |
| `--no-tiling` | Send the full image in one pass (tiling is on above 3000 px for local backends) |
| `--use-page-type` | Advisory page-type triage; requires `--use-vlm --vlm-backend unsloth` |

**Detection**

| Flag | Meaning |
|---|---|
| `--use-sam` | SAM boundary refinement |
| `--use-windows` | Window detection (Tier 1, PDF layers only) |
| `--use-yolo-objects` | Fine-tuned YOLO door/window detector. Tag-only: writes `has_door`/`has_windows`, never accepts or rejects a room |
| `--use-sam3-exemplar` | Tiled SAM3 box-exemplar door detector seeded by YOLO's own detections; no effect without `--use-yolo-objects` |
| `--corroboration-weight` | Additive object-corroboration bonus on room confidence. Default 0.0 (inert) and unvalidated at any nonzero value |
| `--use-vector-room-extent` | Pilot: add `coordinates.extent` from the PDF's vector wall layers |

**Reconciliation, export, scoring**

| Flag | Meaning |
|---|---|
| `--use-semantic-reconciliation` | No-op; reconciliation is on by default |
| `--no-semantic-reconciliation` | Turn it off |
| `--no-door-layer-filter` | Keep door detections the PDF layers show to be wiring |
| `--export-object-detections` | Also write `coco_objects.json` in its own category id space |
| `--object-detection-min-confidence` | Threshold for that export only |
| `--min-rooms` | Minimum rooms per image for `sft_ready=True` (default 1) |
| `--eval-gt-dir` | Score the finished run against a GT annotations directory; writes `eval_metrics.json` |
| `--enable-mlflow`, `--mlflow-tracking-uri` | Opt-in run tracking; a missing or unreachable server never fails the run |

Environment variables read at run time: `VLM_SEED`, `VLM_DO_SAMPLE`,
`VLM_TEMPERATURE`, `VLM_MODEL`, `MLFLOW_ENABLED`, `MLFLOW_TRACKING_URI`,
`FLOORPLAN_LOCK`, `ANTHROPIC_API_KEY`.

---

## Outputs

A finished run directory:

```
run/
├── images/                      # extracted page PNGs
├── annotations/                 # raw per-page annotations (every drop point recorded)
├── processed_annotations/       # post-processed, SFT-facing annotations
├── coco/                        # COCO room geometry
├── coco_objects.json            # door/window geometry (own category id space)
├── label_studio_import.json
├── needs_review.json
├── room_regions/                # cropped room images
├── quality_reports/
├── coverage_report.json
├── post_processing_summary.json
├── dropped_page_attribution/    # why each dropped page was dropped
├── skipped_pages/
├── run_metrics.jsonl
├── run_manifest_<timestamp>.json
├── pipeline_config.json         # full resolved config + code_fingerprint
└── pipeline_<timestamp>.log
```

`pipeline_config.json` is the audit record: it carries the resolved configuration
and the source fingerprint, and the release postflight reads it to prove what
actually ran.

### Annotation data model

Legacy structure, which **the exporters read**:

```json
{
  "image_file": "...",
  "image_size": {"width": 4500, "height": 3375},
  "rooms": [{"room_name": "Office", "category": "office", "bbox": [100, 150, 200, 250]}]
}
```

SFT-facing structure (`automation/annotation_schema.py`):

```json
{
  "roomsRecognized": [{
    "id": "room_0",
    "type": "PRIVATE OFFICE",
    "name": "Office",
    "coordinates": {"bbox": [100, 150, 200, 250]},
    "confidence": 0.95,
    "confidence_detail": {"detection": 0.9, "classification": 0.95, "ocr": 0.9},
    "provenance": {"detection": {"method": "vlm", "model": "qwen3-vl-8b"}}
  }],
  "schema_version": "1.0"
}
```

Two box conventions coexist in the data — label boxes and mixed/oversize boxes —
and each room carries a `box_convention` tag. The SFT exporter targets label boxes
only and skips the rest rather than mixing conventions into one training target.

### Taxonomy

`automation/taxonomy.py` is the single source of truth: **44 mandatory classes**
(e.g. `PRIVATE OFFICE`, `OPEN OFFICE`, `CONFERENCE`, `ELECTRICAL ROOM`,
`MECHANICAL ROOM`, `CORRIDOR`, `STAIRWELL`, `RESTROOM`, `CLASSROOM`, `CLINIC`,
`PARKING GARAGE`, `RESIDENTIAL UNIT`, `OTHER`), 17 extended types, and 24
categories exposed in the VLM prompt. Resolution to a mandatory class is mandatory:
`test/test_taxonomy_resolve_mandatory.py` enforces it.

---

## SFT dataset and fine-tuning

```bash
# 1. Export: a finished run dir (coco/ + processed_annotations/) → Qwen3-VL chat
#    JSONL with group-aware train/val/test splits. The per-document cap is not
#    optional: without it one document supplied 573/577 of the CL targets.
python -m preprocessing_annotations.export.qwen_sft_exporter \
  --run <run dir> --out <dataset dir> --max-targets-per-document 200

# 2. Prove the GT documents are held out — run before every training job
python eval/sft_holdout.py --dataset <dataset dir> [--dataset ...] \
  --evaluation-input ../test_input

# 3. Train
python scripts/train_qwen3vl_unsloth.py --dataset <dataset dir> [--dataset ...] \
  --out ../trained_adapters/<name> --model qwen3-vl-8b [hyperparameter flags]

# 4a. Score a run against room GT (scores at the report's top level)
python eval/room_gt_eval.py --run <run dir> --out gt_<run>.json

# 4b. Or replay the filter stage under current code (scores in `as_run` /
#     `baseline` / `variant` sections) — this is what the production chains feed
#     the gate, because it re-scores every run under one version of the filter.
python eval/room_replay.py --run <run dir> --out replay_<run>.json

# 5. Gate the candidate against the baseline spread. Both report kinds are
#    accepted; a mixed set is refused (sft_gate.py:load_reports).
python eval/sft_gate.py --baseline replay_base_a.json --baseline replay_base_b.json \
  --candidate replay_cand.json --out gate_cand.json [--noise-floor <delta>] \
  [--replay-section baseline]
```

Design decisions encoded in the exporter (`export/qwen_sft_exporter.py`), each
measured before the code existed:

- Targets are **label boxes only**; other conventions are skipped, not mixed.
- One example = one **tile**, the exact view inference gives the model; the grid
  comes from the same `TileSplitter.for_page` the pipeline uses, so training and
  inference views cannot drift.
- Empty tiles are kept as negatives but capped.
- A room is a target only in tiles that contain its whole box.

Splits are **group-aware** (`export/group_split.py`): near-duplicate pages of one
document correlate 0.80–0.93, and related documents share room names at Jaccard
0.39–0.59, so the split unit is a group (document, or a wider family named in a
group map), never a page. Groups pinned in the map always land in test — that is
how the GT pages are kept out of every training set.

Hyperparameters are all CLI flags with the previous script's documented starting
points as defaults, because none has been measured on this data. Training loss
alone has never qualified a model here; scoring is `eval/room_gt_eval.py`.

---

## Evaluation and gates

| Script | What it answers |
|---|---|
| `eval/room_gt_eval.py` | Score a run against room-polygon GT (48 pages, 1497 polygons): label localization, extent IoU, extent size, door/window attachment |
| `eval/room_replay.py` | Re-run the **filter stage only** of a finished run under alternative filter settings, offline, in minutes instead of re-running the pipeline |
| `eval/sft_gate.py` | Did the fine-tune beat the baseline, or beat the noise? Compares a candidate against the *spread* of several seeded baselines; `--candidate` is repeatable and the band is judged on its worst member. Consumes either report kind (`room_gt_eval` top-level scores, or a `room_replay` section chosen with `--replay-section`) and refuses a mixed set |
| `eval/adapter_pin.py` | The shipped adapter's pinned recall band, read by both the CPU regression test and the GPU pin chain |
| `eval/sft_holdout.py` | No GT-scored document may appear in training data. Measured empty intersection 2026-11-09; this turns that measurement into a re-runnable check |
| `eval/split_integrity.py`, `eval/page_consistency.py` | Split and per-page consistency checks |
| `eval/dense_tile_probe.py`, `eval/crop_reread_probe.py`, `eval/missed_room_ocr.py` | Targeted diagnostics |

The gate exists because decoding is sampled (`VLM_DO_SAMPLE=true`,
temperature 1.0): two runs of identical code on identical pages do not return
identical numbers, and the measured baseline seed spread is 0.0140 label recall. A
candidate a point above one baseline run has not necessarily done anything. Supply
`--noise-floor` when gating against a single baseline — one baseline has no spread
of its own, so the check otherwise passes on any positive delta.

---

## Testing

```bash
pytest -q                     # fast suite — 729 passed, 1 skipped, 16 deselected, 64 s
pytest -m slow                # GPU/network suite — 16 tests, minutes to hours
pytest test/test_release.py   # release manifest checks
```

`pytest.ini` sets `testpaths = test` and `addopts = -m "not slow"`. Slow tests take
the pipeline's own run lock via `test/conftest.py::gpu_run_lock`, so a slow test
cannot collide with a pipeline run on the single GPU. The one skip is a vocabulary
test whose pre-fix run artifact is not present in this environment
(`test/test_vocab_tool.py:94`).

Notable guard tests, each pinning a bug that was found the hard way:

- `test_adapter_band_regression.py` — pin-consistency assertions run in the fast
  suite; the three stored-run re-scorings are marked `slow` (~68 s each, no GPU) and
  check `eval/pins/adapter_real_v1_band.json` at tolerance 0.0005. Catches scoring
  or filtering drift that nothing else would notice.
- `test_release.py` — the release's code check compares against the **live**
  fingerprint, so it fails until a legitimate code change is re-pinned. Also holds the
  detector checks: missing YOLO weights, missing `ultralytics` and an uncached SAM3
  checkpoint each fail preflight; a `model load failed` in any run log (or no log at
  all) fails postflight; and the manifest's forbidden phrase must still be what both
  detectors log.
- `test_page_image_coordinate_space.py` — the stored PNG must not be resized after
  OCR has read it; enforced structurally, by asserting no in-place image resizer is
  exposed at all (`test_no_in_place_page_image_resizer_is_exposed`).
- `test_merge_vlm_ocr_scan.py` — OCR rooms merge against VLM rooms only, never
  against each other. The test asserts four neighbouring 16 px labels all survive;
  the recall effect of the underlying fix is recorded in `deploy/release.json`
  (~0.025 on every baseline), not asserted here.
- `test_sft_holdout.py`, `test_group_split.py`, `test_split_integrity.py` — training
  data may not contain evaluation documents.
- `test_yolo_objects_ocr_only_guard.py`, `test_unattached_object_detections.py` —
  object detections stay tag-only and are persisted completely.

---

## Reproducibility

Three mechanisms, all enforced in code:

1. **Code fingerprint** (`orchestration/code_fingerprint.py`) — a content hash over
   the pipeline source, recorded in every run's `pipeline_config.json`. Git is not
   enough: most runs are made from a working tree with uncommitted edits, so a
   commit hash would call two different trees the same. The gate refuses to compare
   runs from different fingerprints.
2. **Run lock** (`orchestration/run_lock.py`) — one pipeline run at a time. VLM and
   SAM each load multi-GB models; two concurrent runs exhaust host RAM and freeze
   the machine. Default lock path `/tmp/floorplan_pipeline.lock`, override with
   `$FLOORPLAN_LOCK`.
3. **Seeded sampling** — `VLM_SEED` makes a run reproducible, not merely repeatable:
   re-running seed 1234 reproduced a full scored report bit-identically.

### Operational rules for long chains

- **Do not edit `src/` while a chain runs.** Every run records its fingerprint and
  the chain aborts the moment one differs — but the half-edited runs are still
  wasted GPU hours.
- **Do not edit a running shell script.** Bash reads a script by byte offset; an
  in-place edit truncates the tail of the chain. Stop it by PID, edit, relaunch.
- **Check for a running GPU job before starting one.** The 8B needs the whole T4;
  another session's probe beside a run kills it when the pipeline reloads the model.

---

## Known limits

Stated here and in `deploy/release.json` under `known_limits`, so a consumer learns
them from the documentation rather than from an empty field in production.

- **Room extents are not produced.** `gt_rooms_iou_matched` is 0 and `extent_f1` is
  0.0. The pipeline localizes room *labels*. `--use-vector-room-extent` is a pilot,
  off by default.
- **Exporters read the legacy `rooms` structure**, not `roomsRecognized`. Anything
  that consumes the richer structure must read the annotations directly.
- **Label recall is ~0.611** on the 13-document GT set with the shipped adapter; the
  band file holds the exact interval.
- **`--corroboration-weight` is inert** (default 0.0) and unvalidated at any nonzero
  value. Do not ship a nonzero default without the confidence-distribution spike.
- **`--ocr-vlm-fallback` boxes are not in page coordinates.** Off by default.
- **`--use-sam3-exemplar` is contested on real pages**: a measured F1 gain on all
  three GT datasets, but it has also lost real doors under its current tile grid.
  Read the detector docstring before enabling in production.
- **YOLO and SAM3 can be silently absent from a run — caught by `deploy/release.py`,
  not by `main.py`.** Both detectors catch their own load failure, log
  `model load failed`, and let the run finish. Detections are tag-only, so the recall
  band cannot notice either. `release.py` now fails preflight when a detector the
  manifest enables cannot load, and postflight when the run's log says one did not;
  a direct `main.py` run is still unchecked. Three causes:
  - **Weights are located by directory layout.** `config/config.py:28` takes `parents[4]`
    as the project root and resolves `yolo_checkpoints/` (`:1423`) beneath it, with no
    environment override. A checkout anywhere but the layout the chains use — for
    instance as the `floorPlanVisionAI/preprocessing` submodule — loses them. Not fixed
    in code (it would move `code_fingerprint`); preflight reports the exact path it
    looked for.
  - **`ultralytics` was not in `requirements.txt`** though the YOLO and SAM3 code import
    it. Now pinned (`8.4.104`). A stack built before that pin still lacks it, and so
    does a checkout of this repository from before it.
  - **SAM3's `facebook/sam3` checkpoint must already be in the Hugging Face cache**
    (`config/config.py:1889`, `local_files_only=True`).

  For a run made without `release.py`: `grep -c "model load failed" <run>/pipeline_*.log`
  must be 0.
- **The `floorPlanVisionAI` wrapper is now the serving layer** (`floorplan-service`): a
  FastAPI service that runs this release through `deploy/release.py` and never imports
  the pipeline. Its submodule pointer was bumped to current `main` (`8434d9d`) on
  2026-11-12; that commit predates the detector checks, so the wrapper only gains them
  when the pointer is bumped again.

---

## Related projects in this workspace

| Path | Role |
|---|---|
| `../paligemma2/src/` | 10-step PaliGemma 2 fine-tuning pipeline; `convert_preprocessing_to_paligemma.py` bridges this pipeline's `processed_annotations/` into PaliGemma `<locYYYY>` JSONL |
| `../floorPlanVisionAIAdaptor/` | Qwen2-VL 4-bit production inference application |
| `../floorPlanVisionAI/` | `floorplan-service`: an HTTP service that runs this release as background jobs via `deploy/release.py`. `preprocessing/` is a submodule of this repository, pointed at current `main` since 2026-11-12 |
| `../trained_adapters/` | LoRA adapters, addressed by path and pinned by content digest in the release manifest |
| `../rebaseline_1275.sh`, `../separate_adapters.sh`, `../adapter_pin_check.sh` | Multi-day GPU chain scripts: baselines, training, gating, pin checks. Launch by absolute path |

## Further documentation

- [`deploy/README.md`](deploy/README.md) — the deployable unit and what it stops
- [`docs/RUNNING_THE_PROJECT.md`](docs/RUNNING_THE_PROJECT.md) — step-by-step run guide
- [`docs/mlflow_setup.md`](docs/mlflow_setup.md) — self-hosted tracking, with and without a server
- [`docs/VLM_COST_ANALYSIS.md`](docs/VLM_COST_ANALYSIS.md), [`docs/QUICK_VLM_DECISION.md`](docs/QUICK_VLM_DECISION.md) — backend selection and cost
- `../PROJECT_STATE.md` — project-level status log

## Deprecated

- `scripts/finetune_qwen.py` — targets Qwen2.5-VL, guesses box conventions (wrong for
  every room this pipeline emits) and shuffles pages into splits with no group
  awareness and no seed. Superseded by `export/qwen_sft_exporter.py` +
  `scripts/train_qwen3vl_unsloth.py`.
- `scripts/annotate_pipeline_legacy.py` — pre-package entry point.
