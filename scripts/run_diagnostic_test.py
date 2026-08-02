#!/usr/bin/env python
"""
Test script to run the pipeline with diagnostic instrumentation.
Uses Qwen2.5-VL local inference on the test dataset.
"""

import sys
import logging
from pathlib import Path

# Add preprocessing_annotations to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing_annotations.orchestration.pipeline import AnnotationPipeline
from preprocessing_annotations.config import PipelineConfig, VLMConfig

# Configure logging for diagnostic output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)-8s [%(name)s:%(lineno)d] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/home/ubuntu/floorplan_classifier/VLM/dataset_test_fix24_verbose/diagnostic_run.log')
    ]
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Use the official test_input directory with PDFs
    input_dir = Path("/home/ubuntu/floorplan_classifier/VLM/test_input")
    output_dir = Path("/home/ubuntu/floorplan_classifier/VLM/test_output_diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create config with defaults
    config = PipelineConfig()
    config.use_vlm = True  # Enable VLM annotation
    
    # Configure VLM backend to use Qwen2.5-VL
    config.vlm.backend = "qwen"
    config.vlm.model = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    logger.info("=" * 80)
    logger.info("DIAGNOSTIC RUN: Pipeline with instrumented logging")
    logger.info(f"Input: {input_dir}")
    logger.info(f"VLM Enabled: {config.use_vlm}")
    logger.info(f"Backend: {config.vlm.backend}")
    logger.info(f"Model: {config.vlm.model}")
    logger.info("=" * 80)
    
    pipeline = AnnotationPipeline(config)
    pipeline.run(input_dir=str(input_dir), output_dir=str(output_dir))
    
    logger.info("=" * 80)
    logger.info("DIAGNOSTIC RUN COMPLETE")
    logger.info("Check debug_overlays/ and diagnostic_run.log for details")
    logger.info("=" * 80)
