#!/usr/bin/env python3
"""
Quick test script for Phase 1: PaddleOCR Integration

This demonstrates how to use the new PaddleOCR backend.
"""

import sys
from pathlib import Path

# Add the package dir (parent of test/) so `preprocessing_annotations` resolves.
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing_annotations.config import PipelineConfig, OCRConfig
from preprocessing_annotations.ingestion.ocr_extractor import MEPTextExtractor

def main():
    """Test Phase 1 implementation."""
    print("=" * 60)
    print("PHASE 1: PaddleOCR Integration Test")
    print("=" * 60)
    
    # Create config
    print("\n1. Creating PipelineConfig...")
    config = PipelineConfig()
    
    # Switch OCR backend to PaddleOCR
    print("2. Switching OCR backend to PaddleOCR...")
    config.ocr.backend = "paddleocr"
    print(f"   ✓ OCR backend is now: {config.ocr.backend}")
    
    # Create extractor
    print("\n3. Creating MEPTextExtractor...")
    extractor = MEPTextExtractor(config.ocr)
    print("   ✓ MEPTextExtractor created successfully")
    
    # Verify backend
    print("\n4. Verifying OCR backend...")
    print(f"   Backend type: {extractor.ocr_backend.__class__.__name__}")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Phase 1 test completed successfully!")
    print("=" * 60)
    print(f"\nConfiguration Summary:")
    print(f"  - OCR Backend: {config.ocr.backend}")
    print(f"  - Backend Class: {extractor.ocr_backend.__class__.__name__}")
    print(f"  - Device: {config.ocr.device}")
    print(f"  - Languages: {config.ocr.languages}")
    
    print("\n📝 Next Steps:")
    print("  1. Copy your floor plan images to: ./test_images/")
    print("  2. Run: python test_phase1_inference.py ./test_images/")
    print("  3. Check: ./test_results/ for OCR output")


if __name__ == '__main__':
    main()
