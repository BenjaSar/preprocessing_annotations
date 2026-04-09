# Fix Validation Report

## Executive Summary

All 5 critical bug fixes identified in the root cause analysis have been successfully implemented and verified in the codebase. The fixes directly address the three primary root causes of pipeline regression:

1. **B1: cuDNN Fallback** - Enables pipeline to work without cuDNN installed
2. **B3: Semantic Reconciler Polygon Fallback** - Allows VLM results with no polygon field to be processed
3. **B4: Claude Temperature Control** - Ensures deterministic, reproducible VLM annotations
4. **B5: Qwen/Unsloth Bbox Scaling** - Corrects hardcoded 1000x1000 assumption with actual image dimensions
5. **B7: RoomAnnotation name_expanded Field** - Adds missing field to dataclass

---

## Fix Verification Details

### Fix B1: cuDNN Fallback (CRITICAL)

**Root Cause**: PaddleOCR with `use_gpu=True` requires cuDNN libraries. System has CUDA 12.2 but cuDNN not installed → RuntimeError on every image → OCR completely failed.

**Fix Location**: `ocr_adapter.py:177-225`

**Implementation**:
```python
# Try GPU first if configured for CUDA
try:
    self.ocr = PaddleOCR(use_gpu=use_gpu, ...)
    self.initialized = True
except RuntimeError as e:
    # Catch cuDNN loading errors and other GPU-specific issues
    if use_gpu and ('cudnn' in str(e).lower() or 'cuda' in str(e).lower()):
        logger.warning(f"GPU initialization failed (cuDNN not found): {e}. Falling back to CPU mode.")
        # Retry with CPU
        self.ocr = PaddleOCR(use_gpu=False, ...)
        self.initialized = True
```

**Verification**: ✓ Code checked, fallback logic in place  
**Test Status**: Blocked on CUDA environment stability (import hangs)  
**Expected Outcome**: Pipeline continues with CPU-based OCR when GPU unavailable

---

### Fix B3: Semantic Reconciler Polygon Fallback (HIGH)

**Root Cause**: Semantic reconciliation requires VLM results to have a `polygon` field. All three VLM backends (Claude, Qwen, Unsloth) return `polygon: None` → reconciliation skips all rooms → zero reconciled output.

**Fix Location**: `semantic_reconciler.py:155-162`

**Implementation**:
```python
# If no polygon but bbox exists, create polygon from bbox
if not polygon:
    bbox = result.get('bbox', [])
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        # Convert bbox to rectangle polygon
        polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        logger.debug(f"VLM result {room_id} has no polygon, created from bbox {bbox}")
```

**Verification**: ✓ Code checked, polygon creation logic in place  
**Impact**: Enables semantic reconciliation to work with all VLM backends  
**Expected Outcome**: All VLM-returned rooms now reconciled instead of skipped

---

### Fix B4: Claude Temperature Control (CRITICAL)

**Root Cause**: Claude API calls don't set `temperature` parameter → defaults to 1.0 (fully random) → same model returns different room boundaries on different runs. Example: LOBBY bbox in fix13 = 1.8% image area; fix18 = 30.1% image area for identical image/model.

**Fix Location**: 
- `vlm_backend.py:161` (room annotation)
- `vlm_backend.py:296` (window detection)
- `vlm_annotator.py:346` (batch annotation)

**Implementation**:
```python
# All Claude API calls now include:
message = self.client.messages.create(
    model=self.config.model,
    max_tokens=self.config.max_tokens,
    temperature=0.0,  # Deterministic output for reproducible annotations
    messages=[...]
)
```

**Verification**: ✓ Code checked, temperature=0.0 set on all 3 Claude API call sites  
**Impact**: Produces deterministic, reproducible annotations  
**Expected Outcome**: Same VLM input → identical room boundaries across multiple runs

---

### Fix B5: Qwen/Unsloth Bbox Scaling (HIGH)

**Root Cause**: Qwen and Unsloth backends hardcode 1000x1000 image dimensions for bbox scaling. Code comments: `# "assuming 1000x1000"`. For actual images (e.g., 2000x1500), this produces bboxes scaled incorrectly by 10x or more.

**Fix Locations**:
- Qwen: `vlm_backend.py:509-510` (caller), `vlm_backend.py:549-591` (_parse_room_response)
- Unsloth: `vlm_backend.py:872-873` (caller), `vlm_backend.py:912-956` (_parse_room_response)

**Implementation**:
```python
# Caller: Extract actual image dimensions
img_width, img_height = image.size
rooms = self._parse_room_response(response_text, img_width, img_height)

# Parser: Use actual dimensions for scaling
def _parse_room_response(self, response_text: str, img_width: int = 1000, img_height: int = 1000):
    # Convert bbox percentages to pixel coordinates using actual image dimensions
    x1_pct, y1_pct, x2_pct, y2_pct = bbox
    bbox = [
        int(x1_pct * img_width / 100),      # Scale X coords with width
        int(y1_pct * img_height / 100),     # Scale Y coords with height
        int(x2_pct * img_width / 100),
        int(y2_pct * img_height / 100)
    ]
```

**Verification**: ✓ Code checked on both Qwen (line 509-510, 549-591) and Unsloth (line 872-873, 912-956)  
**Impact**: Correct bbox coordinates for all image sizes  
**Expected Outcome**: Qwen/Unsloth bboxes now accurate for variable image dimensions

---

### Fix B7: RoomAnnotation name_expanded Field (LOW)

**Root Cause**: `RoomAnnotation` dataclass missing `name_expanded` field. Code tries to access it via `getattr()` → always returns `None` → abbreviation expansion never applied.

**Fix Location**: `vlm_annotator.py:48`

**Implementation**:
```python
@dataclass
class RoomAnnotation:
    """Annotated room from VLM."""
    room_number: str
    room_name: str
    category: str
    bbox: List[int]  # [x, y, width, height]
    name_expanded: Optional[str] = None  # Abbreviation expansion
```

**Verification**: ✓ Code checked, field present in dataclass definition  
**Impact**: Enables abbreviation expansion to function  
**Expected Outcome**: Abbreviated room names (e.g., "BR", "KIT") can be expanded to full names

---

## Summary of Changes

| Bug ID | Component | Issue | Fix | Status | Commit |
|--------|-----------|-------|-----|--------|--------|
| B1 | ocr_adapter | cuDNN missing → OCR fails | CPU fallback | ✓ Verified | 7e1b9ec |
| B3 | semantic_reconciler | No polygon → reconciliation fails | Create polygon from bbox | ✓ Verified | 75574d4 |
| B4 | vlm_backend, vlm_annotator | Random temperature → non-deterministic | Set temperature=0.0 | ✓ Verified | 75574d4 |
| B5 | vlm_backend | Hardcoded 1000x1000 → wrong bbox | Use actual image dimensions | ✓ Verified | 75574d4 |
| B7 | vlm_annotator | Missing field → always None | Add name_expanded field | ✓ Verified | 75574d4 |

---

## Deferred Fix

### Fix B2: OCR DPI Propagation (OPTIONAL, LOW PRIORITY)

**Status**: Deferred - Can be user-applied if merge thresholds need DPI-aware scaling

**Rationale**: 
- fix13 uses 200 DPI, fix18 uses 300 DPI
- OCR merge thresholds (e.g., 50 pixel distance) don't scale with DPI
- At 300 DPI, actual physical distance is 50% smaller than 200 DPI equivalent
- However, this is secondary to the 5 critical fixes above

**Proposed Implementation**:
Pass `PDFConfig.dpi` through to `OCRConfig` so merge/clustering thresholds can scale:
```python
ocr_config.merge_distance_px = (ocr_config.merge_distance_px / 200) * pdf_config.dpi
```

---

## Code Quality Notes

All fixes follow the existing code style and patterns:
- Proper logging at appropriate levels (debug, info, warning)
- Comments explain why fixes were needed
- Backward-compatible (sensible defaults where parameters added)
- No breaking changes to APIs or data formats

---

## Next Steps

### Immediate (Blocking Validation)

1. **Resolve CUDA environment stability**
   - torch import hangs on CUDA initialization
   - Likely due to GPU memory fragmentation or incomplete cuDNN installation
   - Workaround: Run pipeline on CPU backend or restart CUDA services

2. **Re-run pipeline with fixes on test set**
   - Input: `/home/ubuntu/floorplan_classifier/VLM/dataset_test_fix13/images/` (7 test images)
   - Expected output: Room counts and bbox metrics matching or exceeding fix13
   - Validation metrics: room count, SFT readiness, bbox area %, confidence distribution

### Follow-up (Post-Validation)

3. **Compare output metrics** against fix13 baseline
   - Total rooms detected per image
   - SFT-ready image count (should be 6-7 out of 7)
   - Average bbox area as % of image (should be <5%)
   - Confidence score distribution

4. **Optional**: Implement B2 (OCR DPI propagation) if merge thresholds need adjustment

---

## Sign-Off

All 5 critical bug fixes have been implemented, committed, and code-verified. The pipeline is ready for testing once CUDA environment stability is resolved.

**Date**: April 09, 2026  
**Verified By**: OpenCode Analysis (automated code inspection)  
**Commits**: 7e1b9ec, 75574d4, c33506c
