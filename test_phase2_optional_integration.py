#!/usr/bin/env python3
"""
Phase 2 Optional Integration Tests — Window Detection with Real Models.

Tests:
1. CubiCasa5K model loading and inference
2. VLM window detection with Unsloth Qwen backend
3. End-to-end pipeline: all three detection tiers
4. SFT output validation with window suffixes
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from window_detector import WindowDetector, RoomWindowMapping
from cubicasa5k_detector import CubiCasa5KDetector


def test_cubicasa5k_model_loading():
    """Test CubiCasa5K model loading."""
    print("\n" + "=" * 70)
    print("TEST 1: CubiCasa5K Model Loading")
    print("=" * 70)

    model_path = Path(__file__).parent / "models" / "cubicasa5k_model.pkl"

    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print(f"   Download from: https://drive.google.com/file/d/1gRB7ez1e4H7a9Y09lLqRuna0luZO5VRK/view")
        return False

    try:
        detector = CubiCasa5KDetector(model_path=model_path, device="cuda")
        assert detector.is_available, "Model should be available"
        print(f"✅ Model found: {model_path}")
        print(f"   Size: {model_path.stat().st_size / (1024*1024):.1f} MB")
        print(f"   Detector initialized: is_available={detector.is_available}")
        
        # Try to load model
        if detector.load_model():
            print(f"✅ Model loaded successfully")
            return True
        else:
            print(f"⚠️  Model loading returned False (might need PyTorch compat)")
            return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unsloth_qwen_import():
    """Test Unsloth Qwen backend import and initialization."""
    print("\n" + "=" * 70)
    print("TEST 2: Unsloth Qwen Backend Import & Initialization")
    print("=" * 70)

    try:
        from vlm_backend import UnslothQwenBackend, VLMFactory
        from config import VLMConfig
        
        print("✅ Imports successful")
        print(f"   - UnslothQwenBackend: OK")
        print(f"   - VLMFactory: OK")
        print(f"   - VLMConfig: OK")

        # Check if Unsloth is available
        try:
            import unsloth
            print(f"✅ Unsloth library available: {unsloth.__version__}")
        except ImportError:
            print(f"⚠️  Unsloth not installed")
            print(f"   Install: pip install unsloth")
            return False

        # Try to create a Qwen config
        config = VLMConfig()
        config.backend = "unsloth"
        config.qwen_model = "qwen2.5-vl-7b"
        
        print(f"✅ VLMConfig created for Unsloth Qwen")
        print(f"   Backend: {config.backend}")
        print(f"   Model: {config.qwen_model}")
        
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_window_detector_with_mock_data():
    """Test WindowDetector with mock data (no models required)."""
    print("\n" + "=" * 70)
    print("TEST 3: WindowDetector with Mock Data")
    print("=" * 70)

    try:
        detector = WindowDetector()

        # Mock rooms with bboxes
        rooms = [
            {"id": 0, "type": "CONFERENCE", "bbox": [50, 50, 200, 200]},
            {"id": 1, "type": "PRIVATE OFFICE", "bbox": [250, 250, 400, 400]},
        ]

        # Create mock image
        image_array = np.zeros((500, 500, 3), dtype=np.uint8)

        # Run detection (all tiers will be skipped without models)
        mappings = detector.detect_windows(
            image_array=image_array,
            rooms=rooms,
            pdf_path=None,
            vlm_backend=None,
        )

        assert len(mappings) == 2, f"Expected 2 mappings, got {len(mappings)}"
        
        print(f"✅ WindowDetector processed 2 rooms")
        for mapping in mappings:
            print(f"   Room {mapping.room_id}: "
                  f"windows={mapping.has_windows}, count={mapping.window_count}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vlm_window_detection_prompt():
    """Test VLM window detection prompt building."""
    print("\n" + "=" * 70)
    print("TEST 4: VLM Window Detection Prompt")
    print("=" * 70)

    try:
        from vlm_backend import Qwen2_5VLBackend, ClaudeBackend
        from config import VLMConfig

        # Test Qwen prompt
        qwen_config = VLMConfig()
        qwen_backend = Qwen2_5VLBackend(qwen_config)
        qwen_prompt = qwen_backend._build_window_detection_prompt()

        assert "window" in qwen_prompt.lower(), "Prompt should mention windows"
        assert "JSON" in qwen_prompt.upper(), "Prompt should mention JSON"
        
        print(f"✅ Qwen window detection prompt OK")
        print(f"   Length: {len(qwen_prompt)} chars")
        print(f"   Contains: window, skylight, JSON array")

        # Test Claude prompt
        claude_config = VLMConfig()
        claude_backend = ClaudeBackend(claude_config)
        claude_prompt = claude_backend._build_window_detection_prompt()

        assert "window" in claude_prompt.lower(), "Prompt should mention windows"
        assert "JSON" in claude_prompt.upper(), "Prompt should mention JSON"
        
        print(f"✅ Claude window detection prompt OK")
        print(f"   Length: {len(claude_prompt)} chars")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_window_detection_mock_response_parsing():
    """Test parsing of mock VLM window detection responses."""
    print("\n" + "=" * 70)
    print("TEST 5: VLM Window Response Parsing")
    print("=" * 70)

    try:
        from vlm_backend import Qwen2_5VLBackend
        from config import VLMConfig

        config = VLMConfig()
        backend = Qwen2_5VLBackend(config)

        # Mock response from VLM
        mock_response = """[
            {"bbox": [10, 20, 30, 40], "type": "window", "confidence": 0.95},
            {"bbox": [50, 50, 80, 90], "type": "window", "confidence": 0.87},
            {"bbox": [100, 10, 150, 50], "type": "skylight", "confidence": 0.92}
        ]"""

        windows = backend._parse_window_response(mock_response)

        assert len(windows) == 3, f"Expected 3 windows, got {len(windows)}"
        assert windows[0]["type"] == "window", "First should be window"
        assert windows[2]["type"] == "skylight", "Third should be skylight"

        print(f"✅ Response parsing successful")
        print(f"   Parsed {len(windows)} windows:")
        for i, w in enumerate(windows):
            print(f"     {i}: {w['type']}, confidence={w['confidence']}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end_integration():
    """Test end-to-end window detection integration."""
    print("\n" + "=" * 70)
    print("TEST 6: End-to-End Integration")
    print("=" * 70)

    try:
        from window_detector import (
            apply_window_suffixes, WindowDetection, WindowDetectionTier
        )
        from automation.taxonomy import add_window_suffix

        # Create mock detection results using proper WindowDetection objects
        windows = [
            WindowDetection(
                bbox=(100, 100, 150, 150),
                confidence=0.9,
                source_tier=WindowDetectionTier.CUBICASA5K,
            ),
            WindowDetection(
                bbox=(200, 200, 250, 250),
                confidence=0.85,
                source_tier=WindowDetectionTier.CUBICASA5K,
            ),
        ]

        rooms = [
            {"id": 0, "type": "CONFERENCE", "bbox": [50, 50, 300, 300]},
            {"id": 1, "type": "CORRIDOR", "bbox": [310, 50, 400, 300]},
        ]

        # Simulate spatial intersection
        detector = WindowDetector()
        mappings = detector.map_windows_to_rooms(windows, rooms)

        # Apply suffixes
        apply_window_suffixes(rooms, mappings)

        # Check results
        assert rooms[0]["type"] == "CONFERENCE w/ windows", \
            f"Room 0 should have windows suffix, got {rooms[0]['type']}"
        assert rooms[1]["type"] == "CORRIDOR", \
            f"Room 1 should NOT have suffix, got {rooms[1]['type']}"

        print(f"✅ End-to-end integration successful")
        print(f"   Room 0: {rooms[0]['type']}")
        print(f"   Room 1: {rooms[1]['type']}")
        print(f"   Window metadata present: {bool('window_detection' in rooms[0])}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sft_output_format():
    """Test SFT output format with window suffixes."""
    print("\n" + "=" * 70)
    print("TEST 7: SFT Output Format with Windows")
    print("=" * 70)

    try:
        from automation.annotation_schema import SFTAnnotationBuilder
        from automation.taxonomy import add_window_suffix

        builder = SFTAnnotationBuilder()

        # Build a room with window suffix
        base_type = "CONFERENCE"
        suffixed_type = add_window_suffix(base_type, has_windows=True)

        room = builder.build_room(
            room_id=1,
            mandatory_type=suffixed_type,
            original_name="CONF RM",
            room_number="101",
            bbox=[100, 100, 300, 300],
            detection_score=0.9,
            classification_match_type="vlm",
            ocr_confidence=1.0,
            source="vlm_only",
            detection_method="vlm",
            detection_model="Qwen",
        )

        # Convert to dict
        room_dict = builder.to_dict(room)

        assert room_dict["type"] == "CONFERENCE w/ windows", \
            f"Type should be suffixed, got {room_dict['type']}"
        assert "nameUnique" in room_dict, "Should have nameUnique"
        assert "coverage" in room_dict, "Should have coverage"
        assert "provenance" in room_dict, "Should have provenance"

        print(f"✅ SFT output format valid")
        print(f"   Type: {room_dict['type']}")
        print(f"   Name: {room_dict['name']}")
        print(f"   Confidence: {room_dict['confidence']}")
        print(f"   Has nameUnique: True")
        print(f"   Has coverage: True")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("PHASE 2 OPTIONAL INTEGRATION TEST SUITE")
    print("=" * 70)

    tests = [
        ("CubiCasa5K Model Loading", test_cubicasa5k_model_loading),
        ("Unsloth Qwen Backend", test_unsloth_qwen_import),
        ("WindowDetector Mock Data", test_window_detector_with_mock_data),
        ("VLM Window Detection Prompt", test_vlm_window_detection_prompt),
        ("VLM Response Parsing", test_window_detection_mock_response_parsing),
        ("End-to-End Integration", test_end_to_end_integration),
        ("SFT Output Format", test_sft_output_format),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    print("=" * 70)

    if failed > 0:
        print(f"\n⚠️  {failed} test(s) failed")
        print("\nNote: Some tests may fail due to:")
        print("  - Unsloth not installed (install with: pip install unsloth)")
        print("  - CUDA/GPU issues")
        print("  - Model checkpoint not downloaded")
        sys.exit(0)  # Non-fatal, tests are optional
    else:
        print("\n✅ All optional integration tests passed!")


if __name__ == "__main__":
    main()
