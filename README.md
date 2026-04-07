# Preprocessing Annotations - MEP Floor Plan Pipeline

A comprehensive Python package for extracting, annotating, and processing MEP (Mechanical, Electrical, Plumbing) floor plans using a combination of PDF extraction, OCR, Vision Language Models, and SAM segmentation.

## Features

- **PDF Extraction**: High-resolution rasterization (150-300 DPI) of floor plan PDFs
- **OCR Text Detection**: Extract room labels and text using PaddleOCR (default) or EasyOCR (legacy) with preprocessing
- **Template Matching**: Multi-scale, rotation-invariant detection of electrical symbols
- **VLM Annotation**: Zero-shot annotation using Claude (Vision Language Model)
- **SAM Segmentation**: Precise room boundary refinement using Segment Anything Model
- **Label Studio Export**: Generate annotated datasets ready for human review
- **Review Prioritization**: Intelligent flagging of low-confidence annotations

## Quick Start

### From Project Root

```bash
cd /home/ubuntu/floorplan_classifier/VLM/preprocessing_annotations

# Run with OCR only (fastest)
python main.py --input ../../floorPlanVisionAIAdaptor/data --output ./results

# Run with VLM annotation (Claude)
python main.py --input ../../floorPlanVisionAIAdaptor/data --output ./results --use-vlm

# High-detail mode for small symbols
python main.py --input ../../floorPlanVisionAIAdaptor/data --output ./results --high-detail --use-vlm
```

### Installation

```bash
cd /home/ubuntu/floorplan_classifier/VLM/preprocessing_annotations

# Install in development mode
pip install -e .

# Then run from anywhere
annotate-pipeline --input ./pdfs --output ./results --use-vlm
```

## VLM Backend Guide

When using `--use-vlm`, you have three backend options:

### Backend Comparison

| Backend | Speed | Cost | VRAM | Internet | Best For |
|---------|-------|------|------|----------|----------|
| **Claude** (default) | 2-5 sec/image | ~$0.01-0.05/image | None | Yes | Fast testing, best quality |
| **Qwen** | 30-60 sec/image | Free | 15 GB | No | No API cost, offline |
| **Unsloth** | 15-30 sec/image | Free | 8 GB | No | Fast + free (2x faster than Qwen) |

### Quick Commands

```bash
# Claude API (recommended for testing)
python3 -m pipeline --input /home/ubuntu/floorplan_classifier/VLM/test_input \
                    --output ./test_output_claude \
                    --ocr-backend paddleocr --use-vlm

# Qwen (local, no cost)
python3 -m pipeline --input /home/ubuntu/floorplan_classifier/VLM/test_input \
                    --output ./test_output_qwen \
                    --ocr-backend paddleocr --use-vlm --vlm-backend qwen

# Unsloth (fast + free, requires installation)
pip install "unsloth[cu121]" --break-system-packages
python3 -m pipeline --input /home/ubuntu/floorplan_classifier/VLM/test_input \
                    --output ./test_output_unsloth \
                    --ocr-backend paddleocr --use-vlm --vlm-backend unsloth
```

### Verify Backend is Running

After starting the pipeline, check the log:
```bash
grep "Creating VLM backend" test_output_*/pipeline.log
```

---

## Command Line Interface

```bash
python main.py --help
```

### Available Options

```
--input PATH, -i PATH           Input directory containing PDF files (required)
--output PATH, -o PATH          Output directory for results (required)
--use-vlm                       Enable Claude VLM annotation
--use-sam                       Enable SAM boundary refinement
--high-detail                   Use high-detail settings (300 DPI, more scales)
--fast                          Use fast processing settings (150 DPI, fewer scales)
--dpi DPI                       Custom DPI for PDF extraction (default: 200)
--skip-existing                 Skip images that already have annotations
--verbose, -v                   Enable verbose logging
```

## Usage Examples

### 1. Basic Usage (OCR Only)

```bash
python main.py --input ./pdfs --output ./dataset
```

Output:
- Extracted images: `dataset/images/`
- OCR annotations: `dataset/annotations/`
- Label Studio import: `dataset/label_studio_import.json`

### 2. With VLM Annotation

```bash
python main.py --input ./pdfs --output ./dataset --use-vlm --verbose
```

Requires: `ANTHROPIC_API_KEY` environment variable set

### 3. High-Detail Mode

For floor plans with small electrical symbols:

```bash
python main.py --input ./pdfs --output ./dataset --high-detail --use-vlm
```

Settings:
- DPI: 300 (vs 200 default)
- Template scales: 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4
- Rotation angles: 0°, 90°, 180°, 270°

### 4. Fast Processing

For large batches where speed matters:

```bash
python main.py --input ./pdfs --output ./dataset --fast
```

Settings:
- DPI: 150
- Template scales: 0.8, 1.0, 1.2
- No rotation invariance

### 5. Programmatic Usage

```python
from preprocessing_annotations import AnnotationPipeline, PipelineConfig

# Create pipeline with custom config
config = PipelineConfig.for_high_detail()
config.use_vlm = True

pipeline = AnnotationPipeline(config)
stats = pipeline.run(
    input_dir="./pdfs",
    output_dir="./results",
    skip_existing=True
)

print(f"Processed {stats['images_extracted']} images")
print(f"Flagged {stats['flagged_for_review']} for review")
```

### 6. VLM Backend Selection

Choose the backend that best fits your needs:

```bash
# Fastest testing (use default Claude backend)
python main.py --input ./pdfs --output ./dataset --use-vlm

# No API costs (use Qwen backend)
python main.py --input ./pdfs --output ./dataset --use-vlm --vlm-backend qwen

# Best speed + free (use Unsloth backend - requires installation)
pip install "unsloth[cu121]" --break-system-packages
python main.py --input ./pdfs --output ./dataset --use-vlm --vlm-backend unsloth
```

**Expected Processing Times** (for 2 test PDFs):
- Claude: 15-30 minutes
- Qwen: 60-120 minutes
- Unsloth: 30-60 minutes
- OCR only (no VLM): 5-10 minutes

## Project Structure

```
preprocessing_annotations/
├── __init__.py                  # Package exports
├── setup.py                     # Installation configuration
├── main.py                      # CLI entry point
├── README.md                    # This file
├── config.py                    # Configuration classes
├── pipeline.py                  # Main orchestration
├── pdf_extractor.py             # PDF to image conversion
├── ocr_extractor.py             # OCR text detection
├── symbol_detector.py           # Template-based detection
├── vlm_annotator.py             # Claude VLM integration
├── sam_segmenter.py             # SAM segmentation
└── exporters.py                 # Label Studio export
```

## Configuration

All settings are centralized in `config.py`. You can customize:

### PDF Extraction
- `dpi`: Resolution (default: 200)
- `page_range`: Specific page range to extract

### OCR
- `confidence_threshold`: Minimum OCR confidence (default: 0.5)
- `preprocess`: Enable image preprocessing (default: True)
- `clahe_clip_limit`: Contrast enhancement strength

### Template Matching
- `threshold`: Detection threshold (default: 0.7)
- `scales`: Multi-scale factors for matching
- `rotations`: Rotation angles to try

### VLM
- `model`: Claude model to use (default: claude-sonnet-4-20250514)
- `max_tokens`: Response token limit (default: 4096)
- `max_retries`: API retry attempts (default: 3)

### SAM
- `model_type`: SAM variant (vit_h, vit_l, vit_b)
- `checkpoint`: Path to model checkpoint

## Output Structure

After running, results are organized as:

```
results/
├── images/                      # Extracted PNG images
│   ├── document1_page000.png
│   ├── document1_page001.png
│   └── ...
├── annotations/                 # JSON annotations
│   ├── document1_page000.json   # {rooms, panels, electrical_counts}
│   ├── document1_page001.json
│   └── ...
├── label_studio_import.json     # Ready to import into Label Studio
├── needs_review.json            # Images flagged for human review
├── pipeline_config.json         # Configuration used
└── README.md                    # Results summary
```

## Environment Setup

### 1. Set Claude API Key

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```
export ANTHROPIC_API_KEY=<your-api-key-here>
```

Then load it:
```bash
source .env
```

### 2. Install Dependencies

```bash
pip install -e .
```

### 3. Download SAM Checkpoint (optional, for refinement)

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Place in `preprocessing_annotations/` directory or specify path in config.

## Performance Considerations

### Memory Usage
- OCR Reader: ~500MB GPU memory
- SAM Model: ~2GB GPU memory
- Concurrent processing: Configure via `num_workers` in config

### Processing Speed

| Task | Speed | Notes |
|------|-------|-------|
| PDF Extraction | ~5-10 sec/page | Depends on DPI |
| OCR | ~10-30 sec/image | Faster on GPU |
| VLM Annotation | ~30-60 sec/image | API latency included |
| SAM Refinement | ~5-10 sec/room | Depends on image complexity |

### GPU vs CPU

The pipeline auto-detects GPU availability:
- CUDA (NVIDIA)
- MPS (Apple Silicon)
- Falls back to CPU if unavailable

## Troubleshooting

### Module not found error

```bash
# Make sure you're in the right directory
cd /home/ubuntu/floorplan_classifier/VLM/preprocessing_annotations

# Then run
python main.py --input ./data --output ./results
```

### API key not working

```bash
# Check environment variable is set
echo $ANTHROPIC_API_KEY

# If not set, source the .env file
source .env
```

### Out of memory

- Reduce DPI: `--dpi 150`
- Use fast mode: `--fast`
- Process fewer images at a time

### Slow processing

- Use fast mode: `--fast`
- Disable preprocessing: modify config
- Reduce template scales

## Frequently Asked Questions (VLM Backends)

### Which VLM backend should I use?

For **Phase 3 Testing**, choose based on your priorities:

1. **Claude API** (Recommended for most users)
   - Fastest results (2-5 sec/image)
   - Best quality annotations
   - Minimal setup
   - Requires API key and internet
   - Cost: ~$0.05-0.30 per floorplan

2. **Qwen** (Best for avoiding API costs)
   - Free to run
   - Works offline
   - Slower (30-60 sec/image)
   - Requires 15 GB VRAM
   - No installation challenges

3. **Unsloth** (Best of both worlds)
   - Free and fast (15-30 sec/image)
   - Works offline
   - 2x faster than Qwen
   - Requires 8 GB VRAM
   - Installation can take time due to compilation

### Why is my pipeline using Claude and not Qwen?

Because `--vlm-backend` defaults to `claude`. You must explicitly add `--vlm-backend qwen` to use Qwen.

### Is Unsloth already installed?

No. Installation requires Unsloth to compile against your CUDA version. You can install it with:
```bash
pip install "unsloth[cu121]" --break-system-packages
```

### Can I use Unsloth without installing it?

No, you must install it first. Once installed, use: `--vlm-backend unsloth`

### How much does Claude API cost?

Approximately:
- ~$0.01-0.05 per image (depends on complexity)
- ~$0.05-0.30 per floorplan (varies by page count and room complexity)

### How do I verify which backend is running?

Check the pipeline log after starting:
```bash
grep "Creating VLM backend" test_output_*/pipeline.log
```

Should show:
- `Creating VLM backend: claude`
- `Creating VLM backend: qwen`
- `Creating VLM backend: unsloth`

## Development

### Running Tests

```bash
pytest tests/
```

### Adding Custom Room Patterns

Edit `config.py`:
```python
@dataclass
class OCRConfig:
    room_name_patterns: List[str] = field(
        default_factory=lambda: [
            r"YOUR_PATTERN_HERE",
            # ... other patterns
        ]
    )
```

### Extending with Custom Exporters

```python
from preprocessing_annotations.exporters import LabelStudioExporter

class CustomExporter(LabelStudioExporter):
    def export(self, annotations_dir, images_dir, output_file):
        # Your custom export logic
        pass
```

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{preprocessing_annotations_2024,
  title={Preprocessing Annotations: MEP Floor Plan Pipeline},
  author={Computer Vision Team},
  year={2024},
  url={https://github.com/...}
}
```

## License

[Your License Here]

## Support

For issues, questions, or contributions, please contact the Computer Vision Team.
