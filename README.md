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

## Architecture

The preprocessing pipeline transforms raw PDF floor plans into SFT-ready annotations through a multi-stage processing flow with pluggable VLM backends and comprehensive validation.

### System Overview

```
                           PDF Input
                              │
                              ▼
              ┌──────────────────────────────┐
              │ preprocessing_annotations    │
              │  (Multi-stage Pipeline)      │
              ├──────────────────────────────┤
              │ 1. PDF Extract → Images      │
              │ 2. OCR Detection             │
              │ 3. VLM Annotation (3 paths)  │
              │ 4. Reconciliation            │
              │ 5. Validation & Filtering    │
              │ 6. Metrics & Quality Check   │
              │ 7. Export                    │
              └──────────────────────────────┘
                    ├──────┬──────┬──────────┐
                    ▼      ▼      ▼          ▼
              Processed  COCO  Label    SFT JSONL
              Annotations    Studio     (with bboxes)
                    │                       │
                    └───────┬───────────────┘
                            ▼
              ┌──────────────────────────────┐
              │  Training Pipelines          │
              ├──────────────────────────────┤
              │ • paligemma2/ (10-step)      │
              │ • finetune_qwen.py           │
              │ • floorPlanVisionAIAdaptor   │
              │   (production inference)     │
              └──────────────────────────────┘
```

### Pipeline Stages

The preprocessing pipeline executes 7 sequential stages:

1. **PDF Extraction** (`pdf_extractor.py`)
   - Converts floor plan PDFs to high-resolution PNG images (150-300 DPI)
   - Handles multi-page documents with page-by-page extraction

2. **Text Detection** (`ocr_extractor.py`, `two_pass_ocr_extractor.py`)
   - Primary: PaddleOCR or EasyOCR for room label extraction
   - Secondary: Two-pass strategy with VLM fallback for low-confidence regions
   - Symbol detection (`symbol_detector.py`): Template-based electrical symbol matching

3. **VLM Room Annotation** (3 parallel paths)
   - **Claude API** (`vlm_annotator.py`): Standalone Anthropic integration, 2-5 sec/image
   - **Qwen2.5-VL** (`vlm_backend.py`): HuggingFace local inference, 30-60 sec/image
   - **Unsloth Qwen** (`vlm_backend.py`): Optimized local inference, 15-30 sec/image
   - Shared hallucination detection (`hallucination_detector.py`): Detects repetitive/grid patterns

4. **Semantic Reconciliation** (`semantic_reconciler.py`)
   - Hybrid merge of OCR text detections with VLM room polygons
   - Spatial containment-based label assignment
   - Resolves conflicts between multiple detection sources

5. **Validation & Filtering** (`automation/sft_validator.py`, `bbox_validator.py`)
   - OOB (out-of-bounds) bbox detection and filtering
   - NaN/infinity checks, coordinate normalization
   - Atomic writes with schema versioning (`bbox_validator.py`)

6. **Quality Evaluation** (`automation/quality_checker.py`, `bbox_metrics.py`)
   - Field validation, taxonomy synchronization
   - Confidence thresholding and overlap detection
   - Comprehensive metrics: mAP, GIoU, DIoU, CIoU, Acc@0.5/0.75
   - Hallucination rate and coverage analysis

7. **Export** (`exporters.py`, bridge converter)
   - COCO format for general ML frameworks
   - Label Studio JSON for human review workflows
   - SFT-ready JSONL with full provenance and confidence metadata
   - PaliGemma2 format with native `<locYYYY><locXXXX>` spatial tokens

### VLM Backend Architecture

The pipeline supports three pluggable VLM backends through a common interface:

```
VLMBackend (Abstract Base Class)
├── ClaudeBackend
│   └── Uses Anthropic API
├── Qwen2_5VLBackend
│   └── Uses HuggingFace Transformers (local)
└── UnslothQwenBackend
    └── Uses Unsloth optimized inference

VLMAnnotator (Standalone Claude path)
└── Alternative to ClaudeBackend (separate architecture)
```

**Shared components across all backends:**
- `detect_hallucinations()`: Detects and truncates autoregressive loops (F4)
- `_parse_room_response()`: Normalized response parsing with image dimension awareness
- Bbox coordinate rescaling to original image space (F3)

### Annotation Data Model

The pipeline produces two output formats for different use cases:

**Backward-Compatible Output** (`rooms[]`):
```json
{
  "image_file": "...",
  "image_size": {"width": 4500, "height": 3375},
  "rooms": [
    {
      "room_name": "Office",
      "category": "office",
      "bbox": [100, 150, 200, 250]
    }
  ]
}
```

**SFT-Ready Output** (`roomsRecognized[]`):
```json
{
  "image_file": "...",
  "roomsRecognized": [
    {
      "id": "room_0",
      "type": "PRIVATE OFFICE",
      "name": "Office",
      "coordinates": {"bbox": [...], "polygon": [...]},
      "confidence": 0.95,
      "confidence_detail": {"detection": 0.9, "classification": 0.95, "ocr": 0.9},
      "coverage": {"spatial_fraction": 0.12, "text_tokens_matched": 4},
      "provenance": {"detection": {"method": "vlm", "model": "claude", "score": 0.9}},
      "name_expanded": "Office - A Wing - 01"
    }
  ],
  "schema_version": "1.0"
}
```

**Room Taxonomy**: 31-class standardized schema (`automation/taxonomy.py`)
- Commercial: PRIVATE OFFICE, OPEN OFFICE, CONFERENCE, etc.
- Utility: ELECTRICAL ROOM, MECHANICAL, STORAGE, etc.
- Circulation: LOBBY, CORRIDOR, STAIRWELL, ELEVATOR, etc.
- Sanitary: RESTROOM, SHOWER, etc.
- Special: RESIDENTIAL units with automatic bedroom/bathroom counting

### Module Reference

#### Core Pipeline (5 files)
| Module | Purpose |
|--------|---------|
| `main.py` | CLI entry point; bootstraps and delegates to pipeline |
| `config.py` | Centralized configuration with PDFConfig, OCRConfig, VLMConfig, SAMConfig dataclasses |
| `pipeline.py` | Main AnnotationPipeline orchestrator coordinating 7-stage flow |
| `exporters.py` | COCO format, Label Studio JSON, review prioritization |
| `__init__.py`, `setup.py` | Package initialization and distribution configuration |

#### Text Detection (4 files)
| Module | Purpose |
|--------|---------|
| `ocr_extractor.py` | Primary OCR text extraction using PaddleOCR or EasyOCR |
| `ocr_adapter.py` | Abstraction layer normalizing output from multiple OCR backends |
| `two_pass_ocr_extractor.py` | Two-pass strategy: baseline + VLM fallback for low-confidence regions |
| `symbol_detector.py` | Template-based multi-scale, rotation-invariant electrical symbol detection |

#### VLM Annotation (4 files)
| Module | Purpose |
|--------|---------|
| `vlm_annotator.py` | Claude standalone integration (alternative to vlm_backend.py) |
| `vlm_backend.py` | Pluggable backend abstraction: Claude API, Qwen2.5-VL, Unsloth (1576 lines) |
| `semantic_reconciler.py` | Hybrid merge of OCR and VLM outputs via spatial containment |
| `hallucination_detector.py` | Shared detector for repetitive/grid/marching patterns (F4) |

#### Spatial Analysis (3 files)
| Module | Purpose |
|--------|---------|
| `sam_segmenter.py` | SAM (Segment Anything Model) integration for boundary refinement |
| `window_detector.py` | Three-tier window detection: PDF layers (Tier 1), CubiCasa5K (Tier 2), VLM (Tier 3) |
| `cubicasa5k_detector.py` | CubiCasa5K multi-task CNN for wall/icon/junction detection |

#### Validation & Metrics (3 files)
| Module | Purpose |
|--------|---------|
| `bbox_validator.py` | OOB/NaN checks, coordinate normalization (F10), atomic writes (F11), filtering (F12) |
| `bbox_metrics.py` | Comprehensive evaluation: mAP, GIoU/DIoU/CIoU, Acc@0.5/0.75, hallucination rate (F7) |
| `visualize_bbox.py` | Standalone tool to render color-coded bboxes with labels for visual QA |

#### SFT & Training (2 files)
| Module | Purpose |
|--------|---------|
| `finetune_qwen.py` | Qwen2.5-VL fine-tuning infrastructure with SFT data prep (includes bboxes via F8) |
| `production_monitor.py` | Production hardening: confidence thresholding, error recovery, quality gates |

#### Automation Subsystem (12 files in `automation/`)
| Module | Purpose |
|--------|---------|
| `taxonomy.py` | Single source of truth: 31-class room taxonomy with mandatory mappings |
| `annotation_schema.py` | Defines SFT-ready JSON schema (RoomCoordinates, SFTRoom, roomsRecognized) |
| `label_normalizer.py` | Backward-compatible wrapper around taxonomy |
| `quality_checker.py` | Room annotation validation: field names, format, taxonomy sync, overlap detection |
| `sft_validator.py` | SFT-grade validation: confidence thresholds, spatial text filtering (1092 lines) |
| `region_extractor.py` | Extracts cropped room regions as individual training images |
| `abbreviations.py` | Canonical abbreviation map (BR→BEDROOM, CONF→CONFERENCE, etc.) |
| `abbreviation_ocr_recovery.py` | Layer 2 recovery: image preprocessing + upscaling for small text OCR |
| `residential_unit_detector.py` | Layer 1: VLM-based residential apartment detection (bedroom/bathroom counting) |
| `synthetic_label_generator.py` | Layer 3: Synthetic label generation when OCR/VLM detection insufficient |
| `residential_pipeline.py` | Orchestrates 3-layer residential abbreviation recovery |

### Connected Projects

#### PaliGemma2 Training Pipeline (`paligemma2/src/`)
A complete 10-step fine-tuning pipeline for Google PaliGemma 2 (Vision Language Model):
- **Steps 1-5**: PDF organization, image conversion, metadata generation, classification/VQA annotations
- **Step 6**: Detection annotation setup with Label Studio converter to PaliGemma native `<loc0000>...<loc1023>` format
- **Steps 7-10**: Train/val/test split creation, LoRA fine-tuning, evaluation metrics, error analysis
- **Bridge**: `convert_preprocessing_to_paligemma.py` reads `processed_annotations/*.json` and outputs PaliGemma JSONL (F1)
- **Advantage**: Native spatial tokens eliminate custom bbox format invention

#### floorPlanVisionAIAdaptor (Production Inference)
Production-ready application using Qwen2-VL with 4-bit quantization:
- Takes PDF floor plans as input, outputs structured JSON analysis
- Model lifecycle management with context manager support
- GPU memory optimization via bit-width reduction
- Configuration management for model selection, inference parameters, system prompts

#### floorPlanVisionAI (Planned Modular Architecture)
Scaffold-only design documenting a modular production system:
- Separated concerns: config, preprocessing, detection, refinement, validation, association, orchestration
- Currently empty; intended as future refactoring target
- Provides architectural guidance for componentization

### Configuration

All settings are centralized in `config.py` and can be customized per use case (see Configuration section below). Key dataclasses:

- **PDFConfig**: DPI, output format, page range
- **OCRConfig**: Backend selection, confidence threshold, preprocessing, 53 room name patterns
- **VLMConfig**: Backend selection (claude/qwen/unsloth), model ID, token limits, room taxonomy
- **SAMConfig**: Model variant, checkpoint path, device auto-detection
- **PipelineConfig**: Master config with factory methods (`for_high_detail()`, `for_fast_processing()`)

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
  year={2026},
  url={https://github.com/...}
}
```

## License

[Your License Here]

## Support

For issues, questions, or contributions, please contact the Computer Vision Team.
