# VLM Backend Quick Reference

## Your Command
```bash
python3 -m pipeline --input /home/ubuntu/floorplan_classifier/VLM/test_input \
                    --output ./test_output_two_pass_ocr \
                    --ocr-backend paddleocr \
                    --use-vlm
```

**Current Backend:** ❌ NOT Qwen, NOT Unsloth → **Uses CLAUDE API** (default)

---

## Change Backend: 3 Options

### 1️⃣ Claude API (Recommended for Testing)
```bash
python3 -m pipeline \
  --input /home/ubuntu/floorplan_classifier/VLM/test_input \
  --output ./test_output_two_pass_ocr \
  --ocr-backend paddleocr \
  --use-vlm
  # or --vlm-backend claude (explicit)
```
- ⚡ **Speed:** 2-5 sec/image
- 💰 **Cost:** ~$0.01-0.05/image
- 💾 **VRAM:** None (cloud)
- 🔑 **Needs API Key:** Yes
- 📡 **Needs Internet:** Yes

### 2️⃣ Qwen (Local, No Cost)
```bash
python3 -m pipeline \
  --input /home/ubuntu/floorplan_classifier/VLM/test_input \
  --output ./test_output_qwen \
  --ocr-backend paddleocr \
  --use-vlm \
  --vlm-backend qwen
```
- ⏱️ **Speed:** 30-60 sec/image
- 💰 **Cost:** $0
- 💾 **VRAM:** 15 GB
- 🔑 **Needs API Key:** No
- 📡 **Needs Internet:** No

### 3️⃣ Unsloth (Fast + Local + No Cost)
```bash
# Step 1: Install (one-time)
pip install "unsloth[cu121]" --break-system-packages

# Step 2: Run
python3 -m pipeline \
  --input /home/ubuntu/floorplan_classifier/VLM/test_input \
  --output ./test_output_unsloth \
  --ocr-backend paddleocr \
  --use-vlm \
  --vlm-backend unsloth
```
- ⚡ **Speed:** 15-30 sec/image (2x faster than Qwen!)
- 💰 **Cost:** $0
- 💾 **VRAM:** 8 GB
- 🔑 **Needs API Key:** No
- 📡 **Needs Internet:** No

---

## Decision Matrix

| Need | Best Choice | Command |
|------|------------|---------|
| Fast testing | **Claude** | `--use-vlm` |
| No API cost | **Qwen** | `--use-vlm --vlm-backend qwen` |
| Fast + No cost | **Unsloth** | `--use-vlm --vlm-backend unsloth` |
| Just OCR | **None** | Remove `--use-vlm` |

---

## Verify Which Backend is Running

After pipeline starts, check the log:
```bash
grep "Creating VLM backend" test_output_*/pipeline.log
```

Should show one of:
- `Creating VLM backend: claude`
- `Creating VLM backend: qwen`
- `Creating VLM backend: unsloth`

---

## Pro Tips

### Fastest Testing (Recommended)
```bash
# Just OCR, no VLM (5-10 minutes)
python3 -m pipeline \
  --input /home/ubuntu/floorplan_classifier/VLM/test_input \
  --output ./test_output_pass1_only \
  --ocr-backend paddleocr
```

### Cost-Free Option (No API calls)
```bash
# Local inference with Qwen (no internet needed)
python3 -m pipeline \
  --input /home/ubuntu/floorplan_classifier/VLM/test_input \
  --output ./test_output_qwen \
  --ocr-backend paddleocr \
  --use-vlm \
  --vlm-backend qwen
```

### Production Ready (Fast + Quality)
```bash
# Claude API (best quality, fastest)
python3 -m pipeline \
  --input /home/ubuntu/floorplan_classifier/VLM/test_input \
  --output ./test_output_production \
  --ocr-backend paddleocr \
  --use-vlm \
  --vlm-backend claude
```

---

## Common Questions

**Q: Why is my command using Claude and not Qwen?**  
A: Because `--vlm-backend` defaults to `claude`. You must explicitly add `--vlm-backend qwen` to use Qwen.

**Q: Is Unsloth already installed?**  
A: No. Installation times out due to heavy dependency compilation. You can skip it for now.

**Q: Can I use Unsloth without installing it?**  
A: No, you must install it first: `pip install "unsloth[cu121]" --break-system-packages`

**Q: Which backend should I use for testing Phase 3?**  
A: 
- Start with **Claude** (default, fastest, best quality)
- If you want to avoid API costs, use **Qwen**
- If you want both speed and no cost, use **Unsloth** (requires installation)

**Q: How much will Claude API cost?**  
A: ~$0.05-0.30 per floorplan (depends on page count and number of low-confidence rooms)

**Q: How long will each option take?**  
A: 
- Claude: 15-30 minutes (for 2 test PDFs)
- Qwen: 60-120 minutes
- Unsloth: 30-60 minutes
- Just OCR: 5-10 minutes

---

## Quick Decision

For **Phase 3 Testing**, I recommend:

1. **First:** Run with Claude (your current command) → see results quickly
   ```bash
   python3 -m pipeline --input /home/ubuntu/floorplan_classifier/VLM/test_input \
                       --output ./test_output_two_pass_ocr \
                       --ocr-backend paddleocr --use-vlm
   ```

2. **If API cost is concern:** Run with Qwen
   ```bash
   python3 -m pipeline --input /home/ubuntu/floorplan_classifier/VLM/test_input \
                       --output ./test_output_qwen \
                       --ocr-backend paddleocr --use-vlm --vlm-backend qwen
   ```

3. **If you want best of both:** Install Unsloth and use it
   ```bash
   pip install "unsloth[cu121]" --break-system-packages
   python3 -m pipeline --input /home/ubuntu/floorplan_classifier/VLM/test_input \
                       --output ./test_output_unsloth \
                       --ocr-backend paddleocr --use-vlm --vlm-backend unsloth
   ```

---

**Bottom Line:** Your current command uses **Claude API** by default. To use Qwen or Unsloth, add `--vlm-backend qwen` or `--vlm-backend unsloth`.
