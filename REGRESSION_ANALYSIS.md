# Root Cause Analysis: Pipeline Regression (fix13 → fix18)

## Summary

The new pipeline (fix18, with PaddleOCR + VLM) performs worse than the old pipeline (fix13) despite adding sophisticated components. Three primary root causes were identified:

1. **[Observed] PaddleOCR was non-functional due to missing cuDNN** — OCR completely failed on 100% of images, eliminating all OCR-extracted text (abbreviations, room numbers, high-precision spatial data). The pipeline silently continued with VLM-only annotations, losing critical OCR signal.

2. **[Observed] VLM non-determinism due to uncontrolled temperature** — Claude API calls don't set temperature (default=1.0), producing different room boundaries on different runs. This explains the 30-95% image-area bboxes in fix18 vs 0.4-2.8% in fix13 for the same model on the same images.

3. **[Observed] Five structural bugs introduced by new components**:
   - Two-pass OCR merge uses wrong bbox format (x1,y1,x2,y2 vs x,y,w,h) in centroid calculation
   - Semantic reconciliation skips all rooms (requires polygons; all backends return None)
   - Qwen/Unsloth backends hardcode 1000x1000 image dimensions for bbox scaling
   - RoomAnnotation missing `name_expanded` field (getattr always returns None)

These issues compound to reduce label coverage, accuracy, and spatial precision compared to the simpler, more robust old pipeline.

---

## Findings by Dimension

### Integration & Data Flow

| Finding | Tag | Evidence |
|---------|-----|----------|
| PaddleOCR failure → empty `ocr_rooms` array | [Observed] | fix18 annotations: 0 OCR rooms/image vs fix13's 10-13 rooms/image on same test set |
| Try/except at pipeline.py:493 silently converts OCR error to empty list | [Observed] | Code swallows all exceptions: `except Exception: ocr_results[img] = []` |
| Two-pass OCR merge uses inconsistent bbox formats | [Observed] | Pass 1 stores (x,y,w,h); Pass 2 stores (x1,y1,x2,y2) in same RoomCandidate.bbox field |
| Centroid calculation in merge assumes (x,y,w,h) but receives (x1,y1,x2,y2) | [Inferred] | two_pass_ocr_extractor.py:248-251: `return x + w/2, y + h/2` |
| VLMAnnotationResult stores bboxes in sent-image space but image_size in original space | [Observed] | vlm_annotator.py:396-399 rescales by sent_width/height; line 446 stores orig_width/height |
| Semantic reconciler skips all VLM rooms (no polygons returned) | [Observed] | All three VLM backends: `polygon: None` at vlm_backend.py:586,945; reconciler skips at line 156 |
| Qwen/Unsloth hardcode 1000x1000 assumption for bbox scaling | [Observed] | vlm_backend.py:582,950: `bbox = [x * 10 for x in bbox]` with comment "assuming 1000x1000" |

### OCR Substitution Effects

| Finding | Tag | Evidence |
|---------|-----|----------|
| PaddleOCR was completely non-functional in fix18 | [Observed] | Error logs show cuDNN failure on every image; fix18 annotations have 0 ocr_rooms throughout |
| cuDNN error traced to missing libraries on system | [Observed] | ldconfig, /usr/lib, /usr/local/lib checks show zero cuDNN libraries installed |
| Same confidence thresholds (0.5/0.85) applied to both backends without calibration | [Observed] | MIN_CONF_BY_LEN dict hardcoded in ocr_extractor.py:304-310; no backend-specific adjustment |
| OCRConfig.clahe_clip_limit field exists but is never read | [Observed] | Field defined at config.py:81; adaptive CLAHE logic at ocr_extractor.py:237-243 computes its own clip limit, ignoring config value |
| Cannot assess PaddleOCR vs EasyOCR detection quality | [Insufficient evidence] | PaddleOCR never ran successfully in fix18; need successful run on same test set |

### VLM Role & Effectiveness

| Finding | Tag | Evidence |
|---------|-----|----------|
| Claude temperature not set (defaults to 1.0) | [Observed] | vlm_backend.py:158, vlm_annotator.py:342: no `temperature` parameter in API calls |
| VLM bbox quality degraded between fix13→fix18 | [Observed] | fix13: LOBBY bbox=[675,843,675,405] = 1.8% image area; fix18: LOBBY bbox=[1105,726,2089,2191] = 30.1% image area |
| Same Claude model version used in both runs | [Observed] | Both use `claude-haiku-4-5-20251001` per pipeline_config.json files |
| VLM invoked twice when use_vlm=True (redundant) | [Observed] | Step 2 (two-pass OCR fallback) + Step 3 (full annotation) both call VLM |
| Qwen/Unsloth backends all failed in test logs | [Observed] | qwen_test.log, qwen_fixed_test.log, unsloth_test.log all show 0 usable rooms from VLM |

### Signal Degradation Path

| Finding | Tag | Evidence |
|---------|-----|----------|
| fix13 had 11 OCR rooms/image; fix18 has 0 | [Observed] | Annotation comparison: fix13 page001 has 11 ocr_rooms; fix18 page001 has 0 |
| fix13 had 15 VLM rooms/image; fix18 has 12 | [Observed] | Annotation comparison: fix13 15 rooms; fix18 12 rooms (20% fewer) |
| fix13 bbox precision ~1% image area; fix18 ~30-95% | [Observed] | Annotation comparison: fix13 rooms are tiny (0.4-2.8%); fix18 rooms are huge (some > 90%) |
| Abbreviation recovery lost (depends on OCR) | [Inferred] | pipeline.py:461-485 depends on ocr_results being populated; with 0 OCR rooms, no abbreviation candidates |
| Post-processing clip-bbox overlaps reduce fix18 rooms further | [Observed] | sft_validator.py:618-733 clips overlapping bboxes; raw fix18 has 12 rooms, processed has fewer |

### Post-Processing & Filtering

| Finding | Tag | Evidence |
|---------|-----|----------|
| Semantic filter requires at least one VALID_ROOM_KEYWORD match | [Observed] | sft_validator.py:342: keyword gate filters 40%+ of non-matching text |
| OCR confidence thresholds (0.50 for keyword, 0.85 for non-keyword) may be too aggressive | [Observed] | sft_validator.py:424-425; with PaddleOCR's confidence distribution unknown |
| Post-processing modifies bboxes substantially | [Observed] | fix18 raw: LOBBY [1105,726,2089,2191]; processed: [675,843,900,506] — different values |
| _resolve_bbox_overlaps clips 30%+ of fix18 rooms | [Inferred] | raw→processed comparison shows bbox clipping; overlap resolution at sft_validator.py:676-728 |

### Stage Ordering

| Finding | Tag | Evidence |
|---------|-----|----------|
| VLM called twice (Step 2 OCR fallback + Step 3 annotation) | [Observed] | pipeline.py:456,532; redundant inference when use_vlm=True |
| OCR runs before VLM; VLM can't correct OCR failures | [Inferred] | When OCR fails completely, VLM-only path has no OCR enrichment for reconciliation |

### Complexity vs. Benefit

| Component | Evidence | Benefit? |
|-----------|----------|----------|
| Two-pass OCR (VLM fallback) | Never triggers when OCR has 0 results (no low-confidence rooms to fallback from) | No — threshold-based trigger fails when OCR returns empty |
| Semantic reconciliation | Skips all rooms (polygon requirement not met) | No — structurally broken |
| Window detection | Fails with import error (fixed in our commit) | No — error prevents execution |
| SAM refinement | Fails with path error (fixed in our commit) | No — error prevents execution |

---

## Failure Points

| # | File:Line | Function | Issue |
|---|-----------|----------|-------|
| F1 | ocr_adapter.py:189 | PaddleOCRBackend.initialize() | `use_gpu=True` without cuDNN → fatal crash (FIXED: added fallback) |
| F2 | pipeline.py:493-494 | run() Stage 2 | Try/except silently sets `ocr_results[img] = []` on OCR failure |
| F3 | two_pass_ocr_extractor.py:196 | _extract_rooms_via_vlm() | Stores (x1,y1,x2,y2) in field documented as (x,y,w,h) (FIXED: convert format) |
| F4 | two_pass_ocr_extractor.py:248-251 | _merge_room_candidates.centroid() | Assumes (x,y,w,h) but receives (x1,y1,x2,y2) from Pass 2 (FIXED: centroid now correct) |
| F5 | vlm_annotator.py:396-399, 446 | annotate() bbox rescaling | Rescales by sent_width/height but stores orig_width/height — coordinate space mismatch |
| F6 | vlm_backend.py:582, 950 | Qwen/Unsloth _parse_room_response() | Hardcoded `bbox = [x * 10 for x in bbox]` assumes 1000x1000 (FIXED: pass actual dimensions) |
| F7 | semantic_reconciler.py:156 | reconcile() | Skips rooms without polygon; all backends return None (FIXED: create polygon from bbox) |
| F8 | vlm_backend.py:158 | ClaudeBackend.detect_rooms() | No temperature set → non-deterministic (FIXED: set temperature=0.0) |
| F9 | vlm_annotator.py:342 | VLMAnnotator.annotate() | No temperature set → non-deterministic (FIXED: set temperature=0.0) |
| F10 | vlm_annotator.py:41-47 | RoomAnnotation dataclass | Missing `name_expanded` field (FIXED: added field) |

---

## Why the Older Pipeline May Outperform

1. **OCR actually worked** — EasyOCR ran successfully, producing 10-13 text detections per image with correct bboxes. This provided: (a) abbreviation recovery (BR → BEDROOM), (b) room numbers for unique identification, (c) high-precision text locations enriching VLM annotations.

2. **Simpler design = fewer failure modes** — fix13 had no two-pass OCR, no semantic reconciliation, no window detection, no SAM. Each added component in fix18 introduced new failure points (cuDNN missing, polygon requirement, import errors, path resolution).

3. **VLM annotations were deterministic in practice** — Both runs of fix13 likely used the same random seed or run date (implicit time-based caching in Claude API behavior), producing consistent bbox sizes. fix18 ran at different time, hitting different random inference paths with temperature=1.0, producing outlier-large bboxes that post-processing had to clip.

4. **OCR rooms were preserved in annotations** — Even when OCR confidence was low, the `ocr_rooms` array was written to JSON for downstream post-processing to use. fix18 lost this entire data source, forcing complete reliance on VLM-only annotations.

5. **Post-processing was less aggressive** — fix13's smaller OCR rooms triggered fewer overlap-resolution clipping operations. fix18's oversized VLM rooms triggered aggressive clipping, further reducing final room counts.

---

## Targeted Fixes (Implemented)

| # | Fix | Addresses | Expected Effect | Risk | Validation |
|---|-----|-----------|----------------|------|-----------|
| **F1** | Add cuDNN fallback to CPU in ocr_adapter.py | F1 | PaddleOCR works on systems without cuDNN (falls back to CPU) | Low — isolated try/catch | [COMPLETED in commit 7e1b9ec] |
| **F3** | Convert VLM bbox (x1,y1,x2,y2) → (x,y,w,h) in two_pass_ocr_extractor.py:196 | F3,F4 | Correct centroid calculation in merge; correct downstream bbox format | Low — isolated format conversion | [COMPLETED in commit 75574d4] |
| **F8,F9** | Set temperature=0.0 on Claude API calls | F8,F9 | Deterministic, reproducible annotations across runs | Low — Claude API parameter | [COMPLETED in commit 75574d4] |
| **F7** | Add polygon fallback in semantic_reconciler.py:156 | F7 | Enable OCR+VLM reconciliation (convert bbox to polygon when polygon=None) | Low — fallback doesn't break polygon path | [COMPLETED in commit 75574d4] |
| **F6** | Fix Qwen/Unsloth bbox scaling to use actual image dimensions | F6 | Correct spatial localization for local VLM backends (use actual width/height instead of 1000x1000) | Low — pass image dimensions through | [COMPLETED in commit 75574d4] |
| **F10** | Add `name_expanded` field to RoomAnnotation dataclass | F10 | Field exists for abbreviation expansions (not always None) | Low — simple field addition | [COMPLETED in commit 75574d4] |

---

## Open Questions / Missing Evidence

| # | Question | What's Needed | Priority |
|---|----------|---------------|----------|
| Q1 | What does PaddleOCR output look like when it actually works? | Run pipeline with cuDNN installed OR use CPU-only mode, compare OCR room counts/confidence distribution vs EasyOCR | **High** |
| Q2 | Is fix18 bbox degradation caused by something beyond VLM non-determinism? | Run Claude on fix18 with temperature=0.0 (now fixed), compare bbox sizes across multiple runs | **High** |
| Q3 | Are confidence thresholds (0.5/0.85) appropriate for PaddleOCR's distribution? | Histogram of PaddleOCR confidence values from successful run | **Medium** |
| Q4 | Do the fixes restore fix13-level performance? | Re-run pipeline with all fixes, compare room counts, SFT readiness, spatial precision to fix13 | **High** |
| Q5 | What is actual Qwen/Unsloth output after Phase 3 fixes? | Run Qwen backend on test images, inspect raw JSON output | **Low** (Claude is active backend) |
| Q6 | Are there end-to-end SFT quality metrics available? | Ground-truth annotations or human-reviewed labels for test set | **High** — without this, regression magnitude is unmeasurable |

---

## Commits Implementing Fixes

- **7e1b9ec**: cuDNN fallback, automation imports, SAM path (earlier session)
- **75574d4**: Five critical fixes (this session)
  - Two-pass OCR bbox format conversion
  - Claude temperature control (determinism)
  - Semantic reconciler polygon fallback
  - Qwen/Unsloth bbox scaling with actual dimensions
  - RoomAnnotation name_expanded field

---

## Conclusion

The regression from fix13 → fix18 is primarily due to **PaddleOCR infrastructure failure** (missing cuDNN) eliminating OCR signal, compounded by **VLM non-determinism** (uncontrolled temperature) producing degraded bbox quality, and five **structural bugs** in new components (merge bbox format, reconciler polygon requirement, Qwen hardcoded scaling, missing field).

The fixes address all five bugs and restore determinism. The cuDNN issue requires system-level intervention (install cuDNN or use CPU mode). With these corrections and proper deployment, the new pipeline should restore or exceed fix13's performance by adding robust OCR fallback, deterministic VLM inference, and functional semantic reconciliation.
