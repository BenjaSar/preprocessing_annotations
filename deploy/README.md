# Deployable unit — `rooms-sft-real-v1`

One release = pinned code + pinned adapter + pinned prompt. `release.json` is
the declaration; `release.py` enforces it. Nothing in this directory
duplicates a value that lives elsewhere.

## Run it

```bash
python deploy/release.py --check                       # preflight, no GPU
python deploy/release.py -i <input dir> -o <run dir>   # preflight, run, postflight
python deploy/release.py --verify-run <run dir>        # postflight an existing run
```

Exit 0 means the release is intact. Exit 1 names the field that disagrees.

Weights staged somewhere else: `--adapter-root /mnt/models`. No manifest edit.

## What it stops

The pipeline raises when an adapter directory is missing. It does not raise
for the three ways a release goes wrong *quietly*, because a research harness
has to be free to mix a checkpoint with any prompt:

| failure | what you'd see without this | caught by |
|---|---|---|
| `--unsloth-adapter` never passed | a normal-looking run of the **base model** | postflight |
| right path, different weights | loads happily | preflight digest |
| adapter asked with the default prompt | no error, just worse output | postflight |

The third is the one to care about. The adapter was trained on `sft_label`;
the backend's default is `production`. Nothing anywhere reports the mismatch.

Demonstrated on real runs: `adapter_real_v1_b1275_pincheck` passes; the
stored `baseline_seed1234_b1275` fails with `unsloth_adapter_path: None` and
`unsloth_room_prompt: 'production'`.

## Why preflight *and* postflight

Preflight proves intent before ~3.5h of GPU. Postflight reads the run's own
`pipeline_config.json` and proves what actually ran — the only check that
catches a flag silently dropped or renamed downstream.

## Why this is not in `src/`

Any edit under `src/` changes `code_fingerprint`. The pinned recall band was
measured on `92ac4c7a44d3`; changing it would invalidate the band, the slow
regression test and the GPU pin chain, and cost a re-measurement of every
seed in the band (~10h GPU) to restore. This layer sits above the frozen code
and reads it.

## What this release does not produce

Room **extents**. `gt_rooms_iou_matched` is 0 and `extent_f1` is 0.0 — it
localizes room *labels*. Exporters read the legacy `rooms` structure, not
`roomsRecognized`. Both are stated in `release.json` under `known_limits` so
a consumer learns it here rather than from an empty field in production.

## Re-pinning after a legitimate code change

1. Re-measure **every** seed in the band — a band half-updated is worse than
   none.
2. Update `eval/pins/adapter_real_v1_band.json`.
3. Update `code_fingerprint` in `release.json`.
4. `pytest test/test_release.py` — `test_the_code_check_compares_against_the_live_fingerprint`
   fails until step 3 is done, which is the point.

## The wrapper repo (`floorPlanVisionAI`) — needs a commit and a push

`floorPlanVisionAI/preprocessing` is a git submodule of **this** repository,
pinned at `883beac` (2026-09-06). That SHA no longer exists: the attribution
history rewrite re-hashed everything, and the same commit is now `03b60eb`.
`git submodule update` fails today.

The pinned snapshot also predates all room work — 110 `.py` vs 409 here, zero
adapter-loading call sites, no `sft_label` prompt — so even a working
checkout could not run this release.

Fixing it is a pointer bump, not a code move, and it needs commits pushed
first:

```bash
# in this repo, after committing feat/sft-test and merging to main
git push origin main

# in floorPlanVisionAI
git submodule set-branch --branch main preprocessing
git submodule update --remote preprocessing
git add preprocessing .gitmodules && git commit -m "chore: re-point preprocessing submodule at current main"
```

Weights do not travel this way. The adapter (852M on disk, 98MB of real
weights), `sam_vit_h_4b8939.pth` (2.4G) and the SAM2/YOLO checkpoints stay
outside the repo and are addressed by path — which is what `--adapter-root`
is for.
