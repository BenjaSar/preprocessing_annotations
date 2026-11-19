# Deployable unit — `rooms-sft-real-v1`

One release = pinned code + pinned adapter + pinned prompt. `release.json` is
the declaration; `release.py` enforces it. Nothing in this directory
duplicates a value that lives elsewhere.

## Run it

```bash
python deploy/release.py --check                       # preflight, no GPU
python deploy/release.py -i <input dir> -o <run dir>   # preflight, run, postflight
python deploy/release.py --verify-run <run dir>        # postflight an existing run

python deploy/release.py -i <in> -o <run> --visualize  # ...and draw it
python deploy/release.py --verify-run <run> --visualize  # draw a run already made
```

Exit 0 means the release is intact. Exit 1 names the field that disagrees.

Weights staged somewhere else: `--adapter-root /mnt/models`. No manifest edit.

### `--visualize`

`scripts/visualize_bbox.py` can draw any finished run and, from a terminal, nothing
called it — so runs ended with annotations no one could look at. `--visualize` is that
caller. It writes `<run dir>/visualizations/overview/*.png` (full page, boxes drawn) and
`<run dir>/visualizations/regions/*.png` (one crop per room).

Two rules, the same as the service's:

| Case | What happens |
|---|---|
| The run failed its postflight | Not drawn. Pictures of an invalid release are pictures of nothing |
| The drawing itself fails | Logged. **The exit code does not change** — the detections are the product, the pictures are a view of them |

Opt-in, default off, on purpose: `floorPlanVisionAI` draws every run itself after calling
this script, and the manifest's flag list does not carry `--visualize`, so a run started
by the service is not drawn twice.

**The boxes are label-sized.** They bound each room's text, not its walls.

## What it stops

The pipeline raises when an adapter directory is missing. It does not raise
for the five ways a release goes wrong *quietly*, because a research harness
has to be free to mix a checkpoint with any prompt, to run without a
detector, or to run a step over input it does not suit:

| failure | what you'd see without this | caught by |
|---|---|---|
| `--unsloth-adapter` never passed | a normal-looking run of the **base model** | postflight |
| right path, different weights | loads happily | preflight digest |
| adapter asked with the default prompt | no error, just worse output | postflight |
| a detector the release enables cannot load | a run that "succeeds" with **no door/window detections** | preflight (can it load) + postflight (did it) |
| an optional step the release enables never ran | a run with the field simply absent, which reads as "this input had none" | postflight (`summary_must_report`) |

The third is the one to care about for output quality. The adapter was trained
on `sft_label`; the backend's default is `production`. Nothing anywhere reports
the mismatch.

The fourth is invisible to the recall band: detections are tag-only and never
accept or reject a room. Both detectors catch every load failure, log
`model load failed` and return nothing. Preflight checks what it can before the
GPU is spent — for each detector the manifest's flags switch on: YOLO weights at
the path the pipeline's own config names, `ultralytics` importable, and (SAM3)
the `facebook/sam3` checkpoint in the Hugging Face cache. A failed verdict
carries the exact path it looked for. Postflight then reads the run's
`pipeline_*.log` and fails on any `model load failed` (`log_must_not_contain` in
`release.json`), and on a run with no log at all, which proves nothing.

The fifth is the extent step. It needs the run's source PDF, found from the
page-image stem; where it cannot find one it tallies the page and moves on,
so a deployment that stages PDFs elsewhere finishes with zero extents and
exits 0. Postflight requires the step's own tally in
`post_processing_summary.json` (`summary_must_report`) and requires
`room_extent.pages_without_source_pdf` to be zero (`summary_must_be_zero`).
A legitimately empty result stays a pass: a scanned or FLAT PDF carries no
wall layers, and that is a property of the input, not of the release.

Demonstrated on real runs: `adapter_real_v1_b1275_pincheck` passes; the
stored `baseline_seed1234_b1275` fails with `unsloth_adapter_path: None` and
`unsloth_room_prompt: 'production'`; the pincheck run fails the *new* extent
check, because it was made before the flag was added (`test_release.py`
asserts that, against the stored run rather than a fixture).

## Why preflight *and* postflight

Preflight proves intent before ~3.5h of GPU. Postflight reads the run's own
`pipeline_config.json` and log and proves what actually ran — the only check
that catches a flag silently dropped or renamed downstream, or a detector that
loaded in preflight and still failed in the run.

## Why this is not in `src/`

Any edit under `src/` changes `code_fingerprint`. The pinned recall band was
measured on `92ac4c7a44d3`; changing it would invalidate the band, the slow
regression test and the GPU pin chain, and cost a re-measurement of every
seed in the band (~10h GPU) to restore. This layer sits above the frozen code
and reads it.

## Room extents: a second output, and a partial one

This release switches on `--use-vector-room-extent`. Each room keeps its
label box as its location and gains `coordinates.extent` — the room outline
taken from the source PDF's own CAD wall layers — where one can be found.

Measured on this code with `eval/room_extent_eval.py` over
`adapter_real_v1_b1275_pincheck`: 311 of 1072 rooms on 28 of 48 pages get an
extent; 286 of 1497 GT rooms matched at IoU ≥ .5, so **extent recall .191 at
precision .920**, against 0 for label boxes. Coverage is bounded by the input
rather than by tuning — 5 of 10 scored documents are scanned or FLAT exports
with no layers at all, and get nothing.

The label box stays primary: substituting the extent box moves rooms' centres
out of their GT room (located 915 → 904). So the extent never becomes an
annotation's `bbox` or `segmentation`; in COCO it travels as
`attributes.extent_bbox` / `extent_segmentation` / `extent_source`, converted
into COCO's xywh from the record's xyxy. A room with no extent exports
exactly as it did before.

The **Label Studio** export carries none: its regions bind to controls in a
labeling config this repo does not hold, so a polygon there needs a project
change first. The **Qwen SFT** exporter reads label boxes only — training on
extents would be a different task. Both are stated in `release.json` under
`known_limits`.

## Re-pinning after a legitimate code change

1. Re-measure **every** seed in the band — a band half-updated is worse than
   none.
2. Update `eval/pins/adapter_real_v1_band.json`.
3. Update `code_fingerprint` in `release.json`.
4. `pytest test/test_release.py` — `test_the_code_check_compares_against_the_live_fingerprint`
   fails until step 3 is done, which is the point.

## The wrapper repo (`floorPlanVisionAI`)

`floorPlanVisionAI/preprocessing` is a git submodule of **this** repository,
pointed at `main` (re-pointed 2026-11-12; the old pin `883beac` no longer
existed after the attribution history rewrite). The wrapper is a service that
runs this release through `deploy/release.py` and never imports the pipeline.

It runs whatever commit it pins, so the checks above reach it only when its
pointer is bumped:

```bash
# in floorPlanVisionAI, after this repo's change is committed and pushed
git submodule update --remote preprocessing
git add preprocessing && git commit -m "chore: bump preprocessing pin"
```

Weights do not travel this way. The adapter (852M on disk, 98MB of real
weights), `sam_vit_h_4b8939.pth` (2.4G) and the SAM2/YOLO checkpoints stay
outside the repo. The adapter is addressed with `--adapter-root`. The YOLO
checkpoint is **not** addressable that way: the pipeline looks for
`yolo_checkpoints/` in the directory that contains the pipeline checkout
(`parents[4]` in `config/config.py:28`). Here that is `VLM/`; in the wrapper the
checkout is `preprocessing/`, so it is the wrapper's own root. Link or mount it
there; preflight names the exact path it looked for when it is missing. SAM3
comes from the Hugging Face cache (`HF_HOME`).
