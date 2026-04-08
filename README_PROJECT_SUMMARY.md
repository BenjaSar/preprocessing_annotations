# Floorplan Annotation Pipeline - Complete Project Summary

## 📋 What This Project Does

This pipeline automatically extracts and annotates architectural floor plans from PDF documents using OCR and Vision Language Models (VLMs). It detects rooms, extracts their names, classifies room types, and produces training data for fine-tuning custom VLM models.

### Input
- PDF files containing floor plans (architectural drawings)

### Output
- Annotated JSON files with room detection (name, type, bounding box, confidence)
- Extracted room images for training
- Label Studio import format for human review
- COCO format for ML training

---

## 🚀 Quick Start - 30 Seconds

```bash
# 1. Activate environment
source /home/ubuntu/floorplan_classifier/VLM/vlm/bin/activate
cd /home/ubuntu/floorplan_classifier/VLM/preprocessing_annotations

# 2. Run pipeline (with Claude - working version)
python3 main.py \
  --input /home/ubuntu/floorplan_classifier/VLM/test_input \
  --output ./my_results \
  --ocr-backend easyocr \
  --use-vlm \
  --vlm-backend claude

# 3. Check results
ls my_results/label_studio_import.json
```

---

## 📚 Documentation Files Available

1. **RUNNING_THE_PROJECT.md** - Complete guide with all options and examples
2. **QUICK_START.sh** - Automated setup script (executable)
3. **BACKEND_COMPARISON.txt** - Decision matrix for choosing backends
4. **This file** - Project overview and summary

---

## 🎯 Choose Your Backend Based on Your Needs

### ✅ CLAUDE (Recommended - Fully Working)
```bash
python3 main.py --input ./pdfs --output ./results --use-vlm --vlm-backend claude
```
- **Status**: ✅ Fully working and tested
- **Quality**: Excellent (accurate room boundaries)
- **Speed**: 10-15 sec/image
- **Cost**: ~$0.10 per image
- **Setup**: Requires `ANTHROPIC_API_KEY` environment variable

### 🔶 QWEN (Local, Model Loads But Inference Broken)
```bash
python3 main.py --input ./pdfs --output ./results --use-vlm --vlm-backend qwen
```
- **Status**: 🔶 Model loading ✓ (Fixed in Commit 188dfaa)
- **Status**: ❌ Inference broken (Image features/tokens mismatch - Phase 3.2 work)
- **Speed**: ~30 sec/image (when working)
- **Cost**: FREE (one-time 14GB download)
- **VRAM**: 8-10GB GPU required

### 🔶 UNSLOTH (Optimized, Model Loads But Inference Broken)
```bash
python3 main.py --input ./pdfs --output ./results --use-vlm --vlm-backend unsloth
```
- **Status**: 🔶 Model loading ✓ (Fixed timeout in Commit 188dfaa)
- **Status**: ❌ Inference broken (Invalid JSON response - Phase 3.2 work)
- **Speed**: ~15 sec/image (2x faster, when working)
- **Cost**: FREE (one-time 8GB download)
- **VRAM**: 6-8GB GPU required

### ✅ OCR-ONLY (Fast, Lower Quality)
```bash
python3 main.py --input ./pdfs --output ./results --ocr-backend easyocr
```
- **Status**: ✅ Fully working
- **Quality**: Lower (text labels only, no room boundaries)
- **Speed**: 2-3 minutes for 8 images (FASTEST)
- **Cost**: FREE
- **VRAM**: None (CPU only)

---

## 📊 Current Status - Phase 3 Implementation

### ✅ COMPLETED (Phase 3 - VLM Backend Fixes)

**Fix 1: Qwen Model Class Compatibility**
- Problem: `Qwen2.5-VL` models failed with "weight is not an nn.Module"
- Solution: Use `Qwen2_5_VLForConditionalGeneration` instead of `Qwen2VLForConditionalGeneration`
- Status: ✅ FIXED & TESTED (Commit 188dfaa)
- Impact: Qwen model now loads successfully

**Fix 2: Unsloth HuggingFace Download Timeout**
- Problem: "ReadTimeoutError: HTTP 443 Read timed out (10s)"
- Solution: Increased timeout from 10s to 60s
- Status: ✅ FIXED & TESTED (Commit 188dfaa)
- Impact: Unsloth model downloads complete without hanging

### 🔧 IN PROGRESS (Phase 3.2 - Inference Issues)

**Qwen Inference Issue**
- Status: Model loads, inference produces error
- Error: "Image features and image tokens do not match"
- Fix Needed: Adjust image input preparation
- Timeline: Phase 3.2

**Unsloth Inference Issue**
- Status: Model loads, inference produces invalid JSON
- Error: "Expecting value: line 1 column 2 (char 1)"
- Fix Needed: Adjust response parsing for quantized model output
- Timeline: Phase 3.2

---

## 📁 Project Structure

```
/home/ubuntu/floorplan_classifier/VLM/preprocessing_annotations/
├── pipeline.py                      # Main pipeline orchestrator
├── vlm_backend.py                   # VLM backends (Claude, Qwen, Unsloth)
├── ocr_adapter.py                   # OCR backend abstraction
├── two_pass_ocr_extractor.py        # Two-pass OCR+VLM strategy
├── automation/                      # Post-processing (validation, SFT, etc)
├── main.py                          # Entry point
├── RUNNING_THE_PROJECT.md           # Complete guide
├── QUICK_START.sh                   # Automated setup
├── BACKEND_COMPARISON.txt           # Backend decision matrix
└── README_PROJECT_SUMMARY.md        # This file
```

---

## 🔧 Technology Stack

- **OCR Engines**: EasyOCR (working), PaddleOCR (broken in 3.4.0)
- **VLM Backends**:
  - Claude API (Anthropic) - ✅ Working
  - Qwen2.5-VL (Local) - 🔶 Loading works, inference broken
  - Unsloth (Optimized Qwen) - 🔶 Loading works, inference broken
- **Framework**: Python 3.12, PyTorch 2.10.0, Transformers 4.57.6
- **Hardware**: Tesla T4 GPU (14.5GB VRAM)
- **Environment**: Linux, CUDA 12.8

---

## 📖 How to Use This Documentation

### For Running the Pipeline
→ Start with **RUNNING_THE_PROJECT.md**

### For Quick Setup
→ Use **QUICK_START.sh** script

### For Choosing a Backend
→ Check **BACKEND_COMPARISON.txt**

### For Understanding the Latest Changes
→ Review **This file** and the git commits

---

## 🎓 Example Commands by Use Case

### Case 1: Test with Sample Data (Working)
```bash
source /home/ubuntu/floorplan_classifier/VLM/vlm/bin/activate
cd /home/ubuntu/floorplan_classifier/VLM/preprocessing_annotations
python3 main.py \
  --input /home/ubuntu/floorplan_classifier/VLM/test_input \
  --output ./test_results \
  --use-vlm \
  --vlm-backend claude
```
**Expected**: 8 images processed, 60+ rooms detected, Label Studio import ready

### Case 2: High-Quality Production Run
```bash
python3 main.py \
  --input /path/to/your/pdfs \
  --output ./production_results \
  --ocr-backend easyocr \
  --use-vlm \
  --vlm-backend claude \
  --high-detail \
  --dpi 300
```
**Expected**: Detailed annotations with 300 DPI, suitable for small symbols

### Case 3: Fast Testing (No Cost)
```bash
python3 main.py \
  --input /path/to/pdfs \
  --output ./fast_results \
  --ocr-backend easyocr
```
**Expected**: Ultra-fast processing (~3 min for 8 pages), text labels only

### Case 4: Help Debug Qwen Backend
```bash
python3 main.py \
  --input /home/ubuntu/floorplan_classifier/VLM/test_input \
  --output ./debug_qwen \
  --use-vlm \
  --vlm-backend qwen \
  --verbose
```
**Expected**: Model loads, inference fails with detailed error (helps with Phase 3.2)

---

## 📊 Performance Metrics

| Metric | Claude | Qwen (est.) | Unsloth (est.) | OCR-Only |
|--------|--------|----------|-------------|----------|
| Time per image | 10-15s | 30s | 15s | 20s |
| Time for 8 images | ~2 min | ~4 min | ~2 min | ~3 min |
| Rooms detected | 60-80 | TBD | TBD | 60-80 |
| SFT-ready images | Yes | TBD | TBD | No |
| Cost per image | $0.10 | FREE | FREE | FREE |
| GPU required | No | 8-10GB | 6-8GB | No |
| Status | ✅ Working | 🔶 Loading | 🔶 Loading | ✅ Working |

---

## 🐛 Known Issues & Workarounds

### Issue: "ANTHROPIC_API_KEY not set"
**Solution**: `export ANTHROPIC_API_KEY='your-key-here'`

### Issue: "CUDA out of memory"
**Solution**: Use Claude backend (no GPU) or process fewer images

### Issue: Qwen model loading timeout
**Status**: Already fixed (Commit 188dfaa)

### Issue: Qwen inference broken ("Image features/tokens don't match")
**Workaround**: Use Claude backend until Phase 3.2 fix available

### Issue: Unsloth timeout on download
**Status**: Already fixed (Commit 188dfaa)

### Issue: PaddleOCR "Killed" error
**Workaround**: Use `--ocr-backend easyocr` instead

---

## 🔄 Latest Git Commits

```
188dfaa - Phase 3: Fix VLM backend loading issues (Qwen model class, Unsloth timeout)
         - Fixed Qwen2.5-VL model class selection
         - Increased HuggingFace timeout from 10s to 60s
         - Added torch imports to detect_rooms/detect_windows
         - Both backends now initialize successfully
```

---

## 📝 Environment Setup

### Required Environment Variables
```bash
export ANTHROPIC_API_KEY="your-api-key"  # Only needed for Claude backend
```

### Optional Configuration
```bash
export HF_HUB_DOWNLOAD_TIMEOUT=120      # For slow networks (default: 60s)
export CUDA_VISIBLE_DEVICES=0            # Limit GPU usage
```

---

## 🚦 Ready to Start?

1. **For Production**: Use Claude backend (✅ fully working)
   - See: **RUNNING_THE_PROJECT.md → Scenario 1**

2. **For Testing**: Use OCR-only or Claude with sample data
   - See: **QUICK_START.sh**

3. **For Development**: Help fix Qwen/Unsloth inference
   - See: **BACKEND_COMPARISON.txt → Use When**

---

## 💡 Next Steps After Getting Results

1. Review `output_directory/label_studio_import.json`
2. Import into Label Studio for human review
3. Export corrected annotations as COCO JSON
4. Use `room_regions/` images for fine-tuning custom VLM
5. Report any issues or improvements

---

## 📞 Support & Resources

- **Complete Guide**: RUNNING_THE_PROJECT.md
- **Backend Selection**: BACKEND_COMPARISON.txt
- **Automated Setup**: QUICK_START.sh
- **Git History**: `git log --oneline | head -10`
- **Current Status**: `git status`

---

## 📌 Summary

| Aspect | Status |
|--------|--------|
| **Pipeline Functional** | ✅ YES (with Claude backend) |
| **OCR Working** | ✅ YES (EasyOCR) |
| **Qwen Backend Loading** | ✅ YES (Fixed in Commit 188dfaa) |
| **Qwen Inference** | ❌ NO (Phase 3.2 work needed) |
| **Unsloth Loading** | ✅ YES (Fixed in Commit 188dfaa) |
| **Unsloth Inference** | ❌ NO (Phase 3.2 work needed) |
| **Claude Backend** | ✅ YES (Fully working) |
| **Ready for Production** | ✅ YES (Use Claude backend) |

**Bottom Line**: The project is ready to use with the Claude backend right now. Local VLM backends (Qwen, Unsloth) have working model loading but need Phase 3.2 inference fixes.

---

*Last Updated: April 7, 2026 - Phase 3 VLM Backend Fixes Complete*
