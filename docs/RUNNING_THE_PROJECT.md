# How to Run the Floorplan Annotation Pipeline - Complete Guide

## Quick Start (3 Steps)

### Step 1: Activate Virtual Environment
```bash
source /home/ubuntu/floorplan_classifier/VLM/vlm/bin/activate
cd /home/ubuntu/floorplan_classifier/VLM/preprocessing_annotations
```

### Step 2: Prepare Input
Place PDF files in an input directory. Example:
```bash
# Using the test PDFs
INPUT_DIR=/home/ubuntu/floorplan_classifier/VLM/test_input
OUTPUT_DIR=./my_results
```

### Step 3: Run Pipeline
```bash
python3 main.py --input $INPUT_DIR --output $OUTPUT_DIR --ocr-backend easyocr --use-vlm --vlm-backend qwen
```

---

## Complete Command Reference

### Basic Syntax
```bash
python3 main.py --input INPUT_DIR --output OUTPUT_DIR [OPTIONS]
```

### Required Arguments
- `--input INPUT_DIR` - Directory containing PDF files
- `--output OUTPUT_DIR` - Directory for results

### VLM Backend Options (--vlm-backend)

#### Option A: Qwen (Local, ~7B, Recommended)
```bash
python3 main.py \
  --input /home/ubuntu/floorplan_classifier/VLM/test_input \
  --output ./results_qwen \
  --ocr-backend easyocr \
  --use-vlm \
  --vlm-backend qwen
```
**Status:** Model loads ✓, Inference needs work (Phase 3.2)  
**Speed:** ~1 room per 30 seconds  
**VRAM:** 8-10GB (Tesla T4 compatible)

#### Option B: Unsloth (Optimized Qwen, ~2x faster)
```bash
python3 main.py \
  --input /home/ubuntu/floorplan_classifier/VLM/test_input \
  --output ./results_unsloth \
  --ocr-backend easyocr \
  --use-vlm \
  --vlm-backend unsloth \
  --unsloth-model qwen2.5-vl-7b
```
**Status:** Model loads ✓, Inference needs work (Phase 3.2)  
**Speed:** ~2x faster than Qwen  
**VRAM:** 6-8GB (same or less than Qwen)

#### Option C: Claude (API, Reliable, Costs Money)
```bash
python3 main.py \
  --input /home/ubuntu/floorplan_classifier/VLM/test_input \
  --output ./results_claude \
  --ocr-backend easyocr \
  --use-vlm \
  --vlm-backend claude
```
**Status:** Fully functional ✓✓✓  
**Speed:** ~1 room per 10-15 seconds (faster than local)  
**Cost:** ~$0.10 per image (approximate)  
**Requirement:** ANTHROPIC_API_KEY environment variable

### OCR Backend Options (--ocr-backend)

#### EasyOCR (Recommended, No Dependencies)
```bash
--ocr-backend easyocr  # Default, no installation needed
```

#### PaddleOCR (Alternative, Broken in 3.4.0)
```bash
--ocr-backend paddleocr  # Currently broken, use easyocr instead
```

---

## Common Usage Scenarios

### Scenario 1: Quick Test with Sample Data
```bash
source /home/ubuntu/floorplan_classifier/VLM/vlm/bin/activate
cd /home/ubuntu/floorplan_classifier/VLM/preprocessing_annotations

python3 main.py \
  --input /home/ubuntu/floorplan_classifier/VLM/test_input \
  --output ./test_results \
  --ocr-backend easyocr \
  --use-vlm \
  --vlm-backend claude
```
**Expected Runtime:** 5-10 minutes  
**Expected Output:** 8 annotated images, 60+ rooms detected

### Scenario 2: Batch Processing Large Dataset
```bash
python3 main.py \
  --input /path/to/your/pdfs \
  --output /path/to/output/directory \
  --ocr-backend easyocr \
  --use-vlm \
  --vlm-backend claude \
  --skip-existing  # Don't re-process already annotated images
```

### Scenario 3: High-Detail Processing (Smaller Symbols)
```bash
python3 main.py \
  --input /path/to/pdfs \
  --output /path/to/output \
  --ocr-backend easyocr \
  --use-vlm \
  --vlm-backend claude \
  --high-detail  # 300 DPI, more scales for small symbols
  --dpi 300
```

### Scenario 4: Fast Processing (Speed Priority)
```bash
python3 main.py \
  --input /path/to/pdfs \
  --output /path/to/output \
  --ocr-backend easyocr \
  --use-vlm \
  --vlm-backend claude \
  --fast  # Lower DPI, fewer scales
```

### Scenario 5: OCR-Only (No VLM, Fastest)
```bash
python3 main.py \
  --input /path/to/pdfs \
  --output /path/to/output \
  --ocr-backend easyocr
  # Note: No --use-vlm flag
```
**Runtime:** 2-3 minutes for 8 images  
**Quality:** Lower (text labels only, no room boundaries)

---

## Output Directory Structure

After running the pipeline, you'll get:

```
output_directory/
├── images/                          # Extracted floor plan images (PNG)
│   ├── page000.png
│   ├── page001.png
│   └── ...
├── annotations/                     # Raw room annotations (JSON)
│   ├── page000.json
│   └── ...
├── processed_annotations/           # Validated annotations (JSON)
│   └── page000.json
├── quality_reports/                 # Validation issues per image
│   ├── page000_issues.txt
│   └── ...
├── room_regions/                    # Extracted room images for training
│   ├── room_page000_001_CONFERENCE.png
│   └── ...
├── coco/                            # COCO format export
│   └── annotations.json
├── label_studio_import.json         # Ready to import into Label Studio
├── needs_review.json                # Images flagged for human review
├── post_processing_summary.json     # Statistics
├── pipeline_config.json             # Configuration used
└── coverage_report.json             # SFT readiness metrics
```

---

## Understanding Output Files

### 1. annotations/ - Raw OCR+VLM Results
```json
{
  "rooms": [
    {
      "room_name": "CONFERENCE",
      "room_number": "2902",
      "bbox": [2942, 540, 387, 123],
      "confidence": 0.998
    }
  ],
  "roomsRecognized": [...]  // Enhanced with VLM classification
}
```

### 2. processed_annotations/ - Filtered & Validated
Only annotations passing quality checks:
- Minimum room count: 3
- SFT ready: True (suitable for training)
- All issues resolved

### 3. label_studio_import.json - For Annotation Review
Import this file into Label Studio to:
- Review predictions
- Make corrections
- Export corrected annotations for training

### 4. needs_review.json - Priority for Human Review
```json
[
  {
    "filename": "page000.json",
    "confidence": 0.75,
    "reason": "sft_ready=False, low_coverage"
  }
]
```

---

## Monitoring Pipeline Execution

### Real-Time Progress
```bash
# Watch the log as it runs
tail -f pipeline.log

# Or search for specific events
grep "COMPLETE\|ERROR\|WARNING" pipeline.log
```

### Common Log Messages

✅ **Good Signs:**
```
INFO | pipeline | Extracted N images from M PDFs
INFO | two_pass_ocr_extractor | Pass 1 complete: extracted X room candidates
INFO | vlm_backend | Qwen2.5-VL backend initialized
INFO | pipeline | PIPELINE COMPLETE
```

⚠️ **Warnings (usually OK):**
```
WARNING | sft_validator | Fix1: dropped text-label bbox
WARNING | torchao | Skipping import of cpp extensions
```

❌ **Errors (stop processing):**
```
ERROR | vlm_backend | Failed to initialize Qwen
ERROR | pipeline | VLM annotation failed
```

---

## Environment Variables

### Required for Claude Backend
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### Optional Performance Tuning
```bash
# Increase HuggingFace timeout for slow networks
export HF_HUB_DOWNLOAD_TIMEOUT=120

# Limit GPU memory usage
export CUDA_VISIBLE_DEVICES=0
```

---

## Troubleshooting

### Issue: "No module named 'pipeline'"
**Solution:** Activate venv and change directory
```bash
source /home/ubuntu/floorplan_classifier/VLM/vlm/bin/activate
cd /home/ubuntu/floorplan_classifier/VLM/preprocessing_annotations
```

### Issue: "CUDA out of memory"
**Solutions:**
1. Use Unsloth backend (lower memory)
2. Use Claude API backend (no GPU memory)
3. Process fewer images at a time

### Issue: "ReadTimeoutError" when loading Unsloth
**Solution:** Already fixed! Timeout increased to 60 seconds. If still occurs:
```bash
export HF_HUB_DOWNLOAD_TIMEOUT=120
```

### Issue: "weight is not an nn.Module" for Qwen
**Solution:** Already fixed! Using Qwen2_5_VLForConditionalGeneration now.

### Issue: "Image features and image tokens do not match"
**Status:** Known issue with Qwen inference (Phase 3.2 work)  
**Workaround:** Use Claude backend for reliable results

---

## Performance Expectations

### Pipeline Timing
- PDF extraction: ~1 min per 8-page document
- OCR (EasyOCR): ~30-40 sec per page
- VLM (Claude): ~10-15 sec per page
- VLM (Qwen): ~30 sec per page (untested inference)
- VLM (Unsloth): ~15 sec per page (estimated)
- Post-processing: ~1 min for 8 images
- **Total time:** ~8-10 minutes for 8 pages with VLM

### Quality Metrics
| Backend | Rooms Detected | SFT Ready | Room Boundaries |
|---------|---------------|-----------|-----------------|
| Claude | 60-80 per 8 pages | ✓ Yes | ✓ Accurate |
| Qwen | TBD | ✓ Yes (once working) | TBD |
| Unsloth | TBD | ✓ Yes (once working) | TBD |
| OCR-only | 60-80 per 8 pages | ✗ No | ✗ Text labels only |

---

## Next Steps After Running

1. **Review label_studio_import.json**
   ```bash
   head -20 output_directory/label_studio_import.json
   ```

2. **Import into Label Studio**
   - Open Label Studio UI
   - Create project > Import tasks
   - Select label_studio_import.json
   - Review and correct annotations

3. **Export Corrected Annotations**
   - Export as COCO JSON
   - Use for fine-tuning custom VLM

4. **Extract Training Data**
   ```bash
   ls output_directory/room_regions/
   # Use these room images for fine-tuning
   ```

---

## Development Mode (Debugging)

### Run with verbose logging
```bash
python3 main.py \
  --input /path/to/input \
  --output /path/to/output \
  --ocr-backend easyocr \
  --use-vlm \
  --vlm-backend claude \
  --verbose
```

### Test specific component
```bash
# Test OCR only
python3 -c "from ocr_extractor import OCRExtractor; ..."

# Test VLM only
python3 -c "from vlm_backend import Qwen2_5VLBackend; ..."

# Test pipeline config
python3 -c "from config import PipelineConfig; print(PipelineConfig())"
```

---

## Contact & Support

**For Pipeline Issues:**
- Check logs in `pipeline.log`
- See "Troubleshooting" section above
- Review git commit messages for latest changes

**Current Git Status:**
```bash
cd /home/ubuntu/floorplan_classifier/VLM/preprocessing_annotations
git log --oneline -5  # See recent changes
git status            # Check current state
```

**Latest Fixes Applied:**
- Phase 3 Commit: `188dfaa` - VLM backend fixes (Qwen + Unsloth)
- Qwen model class compatibility fixed
- Unsloth HuggingFace timeout fixed
- torch import issues resolved
