#!/usr/bin/env python3
"""
Quick validation script to test that all 5 bug fixes are working.

This script tests each fix in isolation without requiring full pipeline execution:
- B1: cuDNN fallback (OCR adapter)
- B3: Polygon fallback (semantic reconciler)
- B4: Temperature control (VLM backend)
- B5: Bbox scaling (Qwen/Unsloth)
- B7: name_expanded field (VLM annotator)
"""

import sys
import json
from pathlib import Path
from dataclasses import fields

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_b7_room_annotation():
    """Test B7: RoomAnnotation has name_expanded field."""
    print("\n=== Testing B7: RoomAnnotation name_expanded field ===")
    try:
        from preprocessing_annotations.vlm.vlm_annotator import RoomAnnotation
        
        # Check if name_expanded field exists
        field_names = [f.name for f in fields(RoomAnnotation)]
        
        if 'name_expanded' in field_names:
            print("✓ PASS: name_expanded field exists in RoomAnnotation")
            
            # Test creating an annotation with it
            room = RoomAnnotation(
                room_number="101",
                room_name="BR",
                category="Bedroom",
                bbox=[100, 100, 200, 200],
                name_expanded="Bedroom"  # Should accept this now
            )
            print(f"✓ PASS: Can create RoomAnnotation with name_expanded: {room.name_expanded}")
            return True
        else:
            print(f"✗ FAIL: name_expanded field missing. Fields: {field_names}")
            return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def test_b4_temperature_control():
    """Test B4: Temperature is set to 0.0 in VLM backend."""
    print("\n=== Testing B4: Claude temperature=0.0 ===")
    try:
        # Read vlm_backend.py and check for temperature settings
        backend_file = Path(__file__).resolve().parent.parent / "src" / "preprocessing_annotations" / "vlm" / "vlm_backend.py"
        content = backend_file.read_text()
        
        # Check for temperature=0.0 in multiple locations
        occurrences = content.count('temperature=0.0')
        
        if occurrences >= 3:  # Should be at least 3 places (annotation, window detection, etc.)
            print(f"✓ PASS: Found {occurrences} occurrences of temperature=0.0")
            
            # Verify specific locations
            if 'temperature=0.0,  # Deterministic output' in content:
                print("✓ PASS: Temperature control comment found")
                return True
        else:
            print(f"✗ FAIL: Expected 3+ occurrences of temperature=0.0, found {occurrences}")
            return False
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def test_b5_bbox_scaling():
    """Test B5: Qwen/Unsloth use actual image dimensions for bbox scaling."""
    print("\n=== Testing B5: Bbox dimension scaling ===")
    try:
        backend_file = Path(__file__).resolve().parent.parent / "src" / "preprocessing_annotations" / "vlm" / "vlm_backend.py"
        content = backend_file.read_text()
        
        # Check for img_width/img_height usage in parse methods
        if 'img_width, img_height = image.size' in content:
            print("✓ PASS: Found image.size extraction")
        else:
            print("✗ FAIL: image.size extraction not found")
            return False
        
        # Check for correct scaling formula
        if 'int(x1_pct * img_width / 100)' in content:
            print("✓ PASS: Found correct percentage-to-pixel scaling")
        else:
            print("✗ FAIL: Correct scaling formula not found")
            return False
        
        # Check that hardcoded 1000x1000 is no longer used for scaling
        if '1000x1000' in content and 'bbox = [x * 10' in content:
            print("✗ FAIL: Old hardcoded scaling still present")
            return False
        else:
            print("✓ PASS: Old hardcoded scaling removed")
        
        return True
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def test_b3_polygon_fallback():
    """Test B3: Semantic reconciler creates polygon from bbox when missing."""
    print("\n=== Testing B3: Polygon fallback ===")
    try:
        reconciler_file = Path(__file__).resolve().parent.parent / "src" / "preprocessing_annotations" / "vlm" / "semantic_reconciler.py"
        content = reconciler_file.read_text()
        
        # Check for polygon creation logic
        if 'if not polygon:' in content and '[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]' in content:
            print("✓ PASS: Polygon fallback logic found")
            
            # Check for logging
            if 'created from bbox' in content:
                print("✓ PASS: Polygon creation logging found")
            
            return True
        else:
            print("✗ FAIL: Polygon fallback logic not found")
            return False
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def test_b1_cudnn_fallback():
    """Test B1: OCR adapter has cuDNN fallback to CPU."""
    print("\n=== Testing B1: cuDNN fallback ===")
    try:
        ocr_file = Path(__file__).resolve().parent.parent / "src" / "preprocessing_annotations" / "ingestion" / "ocr_adapter.py"
        content = ocr_file.read_text()
        
        # Check for fallback logic
        if 'cudnn' in content.lower() and 'use_gpu=False' in content:
            print("✓ PASS: cuDNN fallback logic found")
            
            # Check for try/except with GPU initialization
            if 'except RuntimeError' in content and 'Falling back to CPU' in content:
                print("✓ PASS: GPU-to-CPU fallback mechanism found")
                return True
            else:
                print("✗ FAIL: Fallback exception handling not found")
                return False
        else:
            print("✗ FAIL: cuDNN fallback logic not found")
            return False
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("PIPELINE BUG FIX VALIDATION TESTS")
    print("=" * 70)
    
    results = {}
    
    # Test each fix
    results['B7 (name_expanded)'] = test_b7_room_annotation()
    results['B4 (temperature)'] = test_b4_temperature_control()
    results['B5 (bbox scaling)'] = test_b5_bbox_scaling()
    results['B3 (polygon fallback)'] = test_b3_polygon_fallback()
    results['B1 (cuDNN fallback)'] = test_b1_cudnn_fallback()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All fixes validated successfully!")
        return 0
    else:
        print(f"\n✗ {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
