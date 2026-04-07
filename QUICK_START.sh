#!/bin/bash
# Quick Start: Test Phase 3 Two-Pass OCR Strategy
# Copy-paste these commands to test the implementation

set -e  # Exit on error

PROJECT_ROOT="/home/ubuntu/floorplan_classifier/VLM/preprocessing_annotations"
TEST_INPUT="/home/ubuntu/floorplan_classifier/VLM/test_input"

echo "==============================================================================="
echo "                  PHASE 3 TWO-PASS OCR TESTING - QUICK START"
echo "==============================================================================="
echo ""
echo "Project directory: $PROJECT_ROOT"
echo "Test input: $TEST_INPUT"
echo ""

# Navigate to project
cd "$PROJECT_ROOT"

# ============================================================================
# TEST 1: Pass 1 Only (Baseline - No VLM)
# ============================================================================
echo "==============================================================================="
echo "TEST 1: PASS 1 ONLY (Baseline - OCR without VLM fallback)"
echo "==============================================================================="
echo ""
echo "This test runs the pipeline WITHOUT VLM to establish a baseline."
echo "Expected duration: 5-10 minutes"
echo ""
read -p "Press Enter to start TEST 1, or Ctrl+C to skip..."
echo ""

echo "Starting TEST 1: Pass 1 Only..."
python3 -m pipeline \
  --input "$TEST_INPUT" \
  --output ./test_output_pass1_only \
  --ocr-backend paddleocr

echo ""
echo "✓ TEST 1 Complete!"
echo ""
echo "Output location: ./test_output_pass1_only/"
echo "Check results:"
echo "  - Annotations: ./test_output_pass1_only/annotations/*.json"
echo "  - Extracted images: ./test_output_pass1_only/images/"
echo "  - Logs: ./test_output_pass1_only/pipeline.log"
echo ""
echo "Quick check:"
ls -lah ./test_output_pass1_only/annotations/ | head -5
echo ""

# ============================================================================
# TEST 2: Two-Pass OCR (With VLM Fallback)
# ============================================================================
echo "==============================================================================="
echo "TEST 2: TWO-PASS OCR (PaddleOCR + VLM Fallback)"
echo "==============================================================================="
echo ""
echo "This test runs the pipeline WITH VLM fallback for low-confidence results."
echo "Expected duration: 15-30 minutes"
echo "Expected API cost: $0.05-0.30 (Claude Haiku)"
echo ""
read -p "Press Enter to start TEST 2, or Ctrl+C to skip..."
echo ""

echo "Starting TEST 2: Two-Pass OCR..."
python3 -m pipeline \
  --input "$TEST_INPUT" \
  --output ./test_output_two_pass_ocr \
  --ocr-backend paddleocr \
  --use-vlm

echo ""
echo "✓ TEST 2 Complete!"
echo ""
echo "Output location: ./test_output_two_pass_ocr/"
echo "Check results:"
echo "  - Annotations: ./test_output_two_pass_ocr/annotations/*.json"
echo "  - Extracted images: ./test_output_two_pass_ocr/images/"
echo "  - Logs: ./test_output_two_pass_ocr/pipeline.log"
echo ""
echo "Quick check:"
ls -lah ./test_output_two_pass_ocr/annotations/ | head -5
echo ""

# ============================================================================
# COMPARISON ANALYSIS
# ============================================================================
echo "==============================================================================="
echo "COMPARISON ANALYSIS: Pass 1 vs Two-Pass OCR"
echo "==============================================================================="
echo ""

echo "1. Running metrics collection..."
python3 << 'PYTHON'
import json
import os
from pathlib import Path

def count_rooms(dir_path):
    total = 0
    for f in Path(dir_path).glob('*.json'):
        try:
            with open(f) as fp:
                data = json.load(fp)
                total += len(data.get('rooms', []))
        except:
            pass
    return total

def avg_confidence(dir_path):
    confs = []
    for f in Path(dir_path).glob('*.json'):
        try:
            with open(f) as fp:
                data = json.load(fp)
                for room in data.get('rooms', []):
                    confs.append(room.get('confidence', 0))
        except:
            pass
    return sum(confs) / len(confs) if confs else 0

pass1_dir = "./test_output_pass1_only/annotations"
pass2_dir = "./test_output_two_pass_ocr/annotations"

pass1_rooms = count_rooms(pass1_dir)
pass2_rooms = count_rooms(pass2_dir)
pass1_conf = avg_confidence(pass1_dir)
pass2_conf = avg_confidence(pass2_dir)

print(f"PASS 1 ONLY:")
print(f"  Total rooms: {pass1_rooms}")
print(f"  Average confidence: {pass1_conf:.3f}")
print()
print(f"TWO-PASS OCR:")
print(f"  Total rooms: {pass2_rooms}")
print(f"  Average confidence: {pass2_conf:.3f}")
print()
print(f"IMPROVEMENT:")
print(f"  Room count difference: {pass2_rooms - pass1_rooms:+d}")
print(f"  Confidence improvement: {pass2_conf - pass1_conf:+.3f} ({(pass2_conf - pass1_conf) / pass1_conf * 100:+.1f}%)")
PYTHON

echo ""
echo "2. Checking for VLM fallback activity..."
grep -c "Pass 2 (VLM Fallback)" ./test_output_two_pass_ocr/pipeline.log 2>/dev/null || echo "No VLM fallback log found"
grep "Matched low-confidence\|Adding unmatched VLM\|Merged" ./test_output_two_pass_ocr/pipeline.log | head -5 || echo "No merge details found"

echo ""
echo "3. Checking Phase 1 (SFT output) integration..."
python3 << 'PYTHON'
import json
from pathlib import Path

# Check SFT format in first annotation
for f in list(Path("./test_output_two_pass_ocr/annotations").glob('*.json'))[:1]:
    with open(f) as fp:
        data = json.load(fp)
        sft_rooms = data.get('roomsRecognized', [])
        if sft_rooms:
            print(f"✓ SFT output present: {len(sft_rooms)} rooms in 'roomsRecognized' field")
            room = sft_rooms[0]
            required = ['room_name', 'nameUnique', 'confidence', 'coverage', 'polygon', 'provenance']
            present = [k for k in required if k in room]
            print(f"✓ SFT format: {len(present)}/{len(required)} required fields present")
        else:
            print("✗ No SFT output found!")
PYTHON

echo ""
echo "4. Checking Phase 2 (Window detection) integration..."
python3 << 'PYTHON'
import json
from pathlib import Path

# Check window detection in first annotation
for f in list(Path("./test_output_two_pass_ocr/annotations").glob('*.json'))[:1]:
    with open(f) as fp:
        data = json.load(fp)
        room = data.get('roomsRecognized', [{}])[0]
        has_windows = room.get('hasWindows')
        has_skylights = room.get('hasSkylights')
        has_openings = room.get('hasOpenings')
        print(f"✓ Window detection fields present:")
        print(f"  - hasWindows: {has_windows is not None}")
        print(f"  - hasSkylights: {has_skylights is not None}")
        print(f"  - hasOpenings: {has_openings is not None}")
PYTHON

echo ""
echo "==============================================================================="
echo "TESTING COMPLETE!"
echo "==============================================================================="
echo ""
echo "Summary:"
echo "  ✓ Pass 1 (baseline): ./test_output_pass1_only/"
echo "  ✓ Two-Pass OCR: ./test_output_two_pass_ocr/"
echo ""
echo "Next steps:"
echo "  1. Review annotation files in both output directories"
echo "  2. Compare confidence scores between Pass 1 and Two-Pass OCR"
echo "  3. Verify room names are immutable (not modified by VLM)"
echo "  4. Check Phase 1 (SFT) and Phase 2 (windows) integrations"
echo "  5. Report results (approve/retest/rollback)"
echo ""
echo "For detailed analysis, see TESTING_GUIDE.md"
echo "For detailed results of pipeline, check pipeline.log in each output directory"
echo ""

# Optional: Show sample annotation
echo "Sample annotation from Two-Pass OCR (first 30 lines):"
find ./test_output_two_pass_ocr/annotations -name '*.json' -type f | head -1 | xargs head -c 2000 | python3 -m json.tool 2>/dev/null | head -30 || echo "(Could not display sample)"
