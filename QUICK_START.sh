#!/bin/bash
# Quick Start Script for Floorplan Annotation Pipeline
# Usage: ./QUICK_START.sh [backend] [input_dir] [output_dir]

set -e

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
BACKEND=${1:-claude}
INPUT_DIR=${2:-/home/ubuntu/floorplan_classifier/VLM/test_input}
OUTPUT_DIR=${3:-./results_${BACKEND}}

# Show usage
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: $0 [backend] [input_dir] [output_dir]"
    echo ""
    echo "Backends:"
    echo "  claude    - Anthropic Claude API (fully working, costs money)"
    echo "  qwen      - Local Qwen2.5-VL 7B (model loads, inference WIP)"
    echo "  unsloth   - Optimized Qwen2.5-VL (model loads, inference WIP)"
    echo "  ocr-only  - OCR text extraction only (fastest, low quality)"
    echo ""
    echo "Example:"
    echo "  $0 claude /path/to/pdfs ./output"
    echo ""
    exit 0
fi

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}Floorplan Annotation Pipeline - Quick Start${NC}"
echo -e "${BLUE}================================================${NC}"

# Step 1: Activate venv
echo -e "${YELLOW}Step 1: Activating virtual environment...${NC}"
source /home/ubuntu/floorplan_classifier/VLM/vlm/bin/activate
cd /home/ubuntu/floorplan_classifier/VLM/preprocessing_annotations
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Step 2: Validate input
echo ""
echo -e "${YELLOW}Step 2: Validating input directory...${NC}"
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}✗ Input directory not found: $INPUT_DIR${NC}"
    exit 1
fi

PDF_COUNT=$(find "$INPUT_DIR" -name "*.pdf" | wc -l)
if [ $PDF_COUNT -eq 0 ]; then
    echo -e "${RED}✗ No PDF files found in: $INPUT_DIR${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Found $PDF_COUNT PDF file(s) in $INPUT_DIR${NC}"

# Step 3: Show configuration
echo ""
echo -e "${YELLOW}Step 3: Pipeline Configuration${NC}"
echo "  Backend:      $BACKEND"
echo "  Input:        $INPUT_DIR"
echo "  Output:       $OUTPUT_DIR"
echo "  OCR:          easyocr"

# Step 4: Build command
echo ""
echo -e "${YELLOW}Step 4: Building command...${NC}"

CMD="python3 main.py --input $INPUT_DIR --output $OUTPUT_DIR --ocr-backend easyocr"

case "$BACKEND" in
    claude)
        if [ -z "$ANTHROPIC_API_KEY" ]; then
            echo -e "${RED}✗ ANTHROPIC_API_KEY not set${NC}"
            echo "   Set it with: export ANTHROPIC_API_KEY='your-key'"
            exit 1
        fi
        CMD="$CMD --use-vlm --vlm-backend claude"
        echo -e "${GREEN}✓ Using Claude API backend${NC}"
        echo "  NOTE: Each image costs ~\$0.10"
        ;;
    qwen)
        CMD="$CMD --use-vlm --vlm-backend qwen"
        echo -e "${GREEN}✓ Using Qwen2.5-VL local backend${NC}"
        echo "  NOTE: Model loads successfully"
        echo "  WARNING: Inference needs Phase 3.2 work (input format issue)"
        ;;
    unsloth)
        CMD="$CMD --use-vlm --vlm-backend unsloth --unsloth-model qwen2.5-vl-7b"
        echo -e "${GREEN}✓ Using Unsloth optimized backend${NC}"
        echo "  NOTE: Model loads successfully (timeout fixed)"
        echo "  WARNING: Inference needs Phase 3.2 work (response parsing issue)"
        ;;
    ocr-only)
        echo -e "${GREEN}✓ Using OCR-only extraction${NC}"
        echo "  NOTE: Fastest option, no VLM used"
        echo "  WARNING: Quality is lower (text labels only)"
        ;;
    *)
        echo -e "${RED}✗ Unknown backend: $BACKEND${NC}"
        echo "  Available: claude, qwen, unsloth, ocr-only"
        exit 1
        ;;
esac

# Step 5: Show command
echo ""
echo -e "${YELLOW}Step 5: Command to execute${NC}"
echo -e "${BLUE}$CMD${NC}"

# Step 6: Ask for confirmation
echo ""
read -p "Ready to run? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Step 6: Run pipeline
echo ""
echo -e "${YELLOW}Step 6: Running pipeline...${NC}"
echo -e "${GREEN}Starting at $(date)${NC}"
echo "================================================"
echo ""

eval "$CMD"

# Step 7: Show results
echo ""
echo "================================================"
echo -e "${GREEN}✓ Pipeline completed at $(date)${NC}"
echo ""
echo -e "${BLUE}Results available in: $OUTPUT_DIR${NC}"
echo ""
echo "Key output files:"
echo "  - $OUTPUT_DIR/label_studio_import.json  (Ready to review in Label Studio)"
echo "  - $OUTPUT_DIR/processed_annotations/    (Validated annotations)"
echo "  - $OUTPUT_DIR/room_regions/             (Training data)"
echo ""
echo "Next steps:"
echo "  1. Review label_studio_import.json"
echo "  2. Import into Label Studio for review"
echo "  3. Export corrected annotations for training"
echo ""
