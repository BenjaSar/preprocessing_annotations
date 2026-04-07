"""
Unit tests for Phase 3: Two-Pass OCR Strategy.

Tests the TwoPassOCRExtractor component with PaddleOCR (Pass 1) + VLM fallback (Pass 2).

Test Coverage:
1. TwoPassOCRExtractor initialization
2. Confidence threshold validation
3. Room candidate merging logic (spatial matching)
4. Integration with MEPTextExtractor (Pass 1)
5. VLM fallback behavior
6. End-to-end two-pass extraction
"""

import unittest
from typing import List, Tuple
from unittest.mock import Mock, patch, MagicMock

try:
    from ..config import OCRConfig
    from ..ocr_extractor import RoomCandidate, TextDetection
    from ..two_pass_ocr_extractor import TwoPassOCRExtractor
except ImportError:
    from config import OCRConfig
    from ocr_extractor import RoomCandidate, TextDetection
    from test.two_pass_ocr_extractor import TwoPassOCRExtractor


class TestTwoPassOCRExtractorInit(unittest.TestCase):
    """Test TwoPassOCRExtractor initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ocr_config = OCRConfig(
            backend="paddleocr",
            languages=["en"],
            device="cpu",
        )
    
    def test_init_without_vlm_backend(self):
        """Test initialization without VLM backend (Pass 1 only)."""
        extractor = TwoPassOCRExtractor(
            ocr_config=self.ocr_config,
            vlm_backend=None,
            confidence_threshold=0.7,
        )
        self.assertIsNotNone(extractor.ocr_extractor)
        self.assertIsNone(extractor.vlm_backend)
        self.assertEqual(extractor.confidence_threshold, 0.7)
    
    def test_init_with_vlm_backend(self):
        """Test initialization with VLM backend."""
        mock_vlm = Mock()
        extractor = TwoPassOCRExtractor(
            ocr_config=self.ocr_config,
            vlm_backend=mock_vlm,
            confidence_threshold=0.65,
        )
        self.assertIsNotNone(extractor.ocr_extractor)
        self.assertEqual(extractor.vlm_backend, mock_vlm)
        self.assertEqual(extractor.confidence_threshold, 0.65)
    
    def test_init_invalid_confidence_threshold(self):
        """Test initialization with invalid confidence threshold."""
        with self.assertRaises(ValueError):
            TwoPassOCRExtractor(
                ocr_config=self.ocr_config,
                confidence_threshold=1.5,  # Invalid: > 1.0
            )
        
        with self.assertRaises(ValueError):
            TwoPassOCRExtractor(
                ocr_config=self.ocr_config,
                confidence_threshold=-0.1,  # Invalid: < 0.0
            )
    
    def test_init_valid_confidence_thresholds(self):
        """Test initialization with valid confidence thresholds."""
        for threshold in [0.0, 0.5, 0.7, 1.0]:
            extractor = TwoPassOCRExtractor(
                ocr_config=self.ocr_config,
                confidence_threshold=threshold,
            )
            self.assertEqual(extractor.confidence_threshold, threshold)


class TestMergeRoomCandidates(unittest.TestCase):
    """Test room candidate merging logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ocr_config = OCRConfig(backend="paddleocr", device="cpu")
        self.extractor = TwoPassOCRExtractor(
            ocr_config=self.ocr_config,
            vlm_backend=Mock(),
            confidence_threshold=0.7,
        )
    
    def _create_room_candidate(
        self,
        bbox: Tuple[int, int, int, int],
        name: str,
        confidence: float,
    ) -> RoomCandidate:
        """Helper to create a RoomCandidate."""
        return RoomCandidate(
            bbox=bbox,
            room_number="",
            room_name=name,
            confidence=confidence,
            raw_text=name,
        )
    
    def test_merge_all_high_confidence_ocr(self):
        """Test merging when all OCR results are high-confidence."""
        pass1_rooms = [
            self._create_room_candidate((100, 100, 50, 50), "OFFICE", 0.9),
            self._create_room_candidate((300, 300, 50, 50), "MEETING", 0.85),
        ]
        pass2_rooms = [
            self._create_room_candidate((105, 105, 50, 50), "OFFICE_VLM", 0.95),
        ]
        
        merged = self.extractor._merge_room_candidates(pass1_rooms, pass2_rooms)
        
        # High-confidence OCR results should be kept
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].room_name, "OFFICE")
        self.assertEqual(merged[1].room_name, "MEETING")
    
    def test_merge_low_confidence_ocr_with_good_match(self):
        """Test merging when low-confidence OCR matches high-confidence VLM."""
        pass1_rooms = [
            self._create_room_candidate((100, 100, 50, 50), "BR", 0.5),  # Low confidence
        ]
        pass2_rooms = [
            self._create_room_candidate((105, 105, 50, 50), "BEDROOM", 0.95),  # High confidence, near match
        ]
        
        merged = self.extractor._merge_room_candidates(pass1_rooms, pass2_rooms)
        
        # Low-confidence OCR should be replaced by high-confidence VLM
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].room_name, "BEDROOM")
        self.assertGreaterEqual(merged[0].confidence, 0.95)
    
    def test_merge_low_confidence_ocr_no_match(self):
        """Test merging when low-confidence OCR has no matching VLM result."""
        pass1_rooms = [
            self._create_room_candidate((100, 100, 50, 50), "BR", 0.5),  # Low confidence
        ]
        pass2_rooms = [
            self._create_room_candidate((500, 500, 50, 50), "KITCHEN", 0.95),  # Far away, no match
        ]
        
        merged = self.extractor._merge_room_candidates(pass1_rooms, pass2_rooms)
        
        # Low-confidence OCR should be kept (no good match), VLM should be added
        self.assertEqual(len(merged), 2)
        room_names = {r.room_name for r in merged}
        self.assertEqual(room_names, {"BR", "KITCHEN"})
    
    def test_merge_adds_unmatched_vlm_results(self):
        """Test that unmatched VLM results are added to merged list."""
        pass1_rooms = [
            self._create_room_candidate((100, 100, 50, 50), "OFFICE", 0.9),
        ]
        pass2_rooms = [
            self._create_room_candidate((100, 100, 50, 50), "OFFICE_VLM", 0.95),
            self._create_room_candidate((500, 500, 50, 50), "KITCHEN", 0.92),  # Unmatched
        ]
        
        merged = self.extractor._merge_room_candidates(pass1_rooms, pass2_rooms)
        
        # Should have high-conf OCR office + unmatched VLM kitchen
        self.assertEqual(len(merged), 2)
        room_names = {r.room_name for r in merged}
        self.assertIn("OFFICE", room_names)
        self.assertIn("KITCHEN", room_names)
    
    def test_merge_bbox_distance_threshold(self):
        """Test that bbox distance threshold is respected."""
        pass1_rooms = [
            self._create_room_candidate((100, 100, 50, 50), "BR", 0.5),  # Low confidence
        ]
        pass2_rooms = [
            self._create_room_candidate((300, 300, 50, 50), "BEDROOM", 0.95),  # Far away
        ]
        
        # With tight threshold, should not match
        merged = self.extractor._merge_room_candidates(
            pass1_rooms,
            pass2_rooms,
            bbox_distance_threshold=50.0,  # Tight threshold
        )
        self.assertEqual(len(merged), 2)  # Both kept separate
        
        # With loose threshold, should match
        merged = self.extractor._merge_room_candidates(
            pass1_rooms,
            pass2_rooms,
            bbox_distance_threshold=500.0,  # Loose threshold
        )
        self.assertEqual(len(merged), 1)  # VLM replaces OCR


class TestTwoPassOCRExtraction(unittest.TestCase):
    """Test end-to-end two-pass OCR extraction."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ocr_config = OCRConfig(backend="paddleocr", device="cpu")
    
    def _create_room_candidate(
        self,
        bbox: Tuple[int, int, int, int],
        name: str,
        confidence: float,
    ) -> RoomCandidate:
        """Helper to create a RoomCandidate."""
        return RoomCandidate(
            bbox=bbox,
            room_number="",
            room_name=name,
            confidence=confidence,
            raw_text=name,
        )
    
    @patch('two_pass_ocr_extractor.MEPTextExtractor')
    def test_pass1_only_when_vlm_backend_none(self, mock_ocr_class):
        """Test that only Pass 1 runs when VLM backend is None."""
        # Mock Pass 1 results
        pass1_rooms = [
            self._create_room_candidate((100, 100, 50, 50), "OFFICE", 0.95),
        ]
        pass1_raw = []
        
        mock_ocr_instance = Mock()
        mock_ocr_instance.extract_and_find_rooms.return_value = (pass1_rooms, pass1_raw)
        mock_ocr_class.return_value = mock_ocr_instance
        
        extractor = TwoPassOCRExtractor(
            ocr_config=self.ocr_config,
            vlm_backend=None,
        )
        
        # Should return Pass 1 results without calling VLM
        rooms, raw = extractor.extract_and_find_rooms_with_vlm_fallback("dummy.png")
        self.assertEqual(rooms, pass1_rooms)
        self.assertEqual(raw, pass1_raw)
    
    @patch('two_pass_ocr_extractor.MEPTextExtractor')
    def test_pass1_only_when_all_high_confidence(self, mock_ocr_class):
        """Test that Pass 2 is skipped when all results are high-confidence."""
        # Mock Pass 1 results (all high confidence)
        pass1_rooms = [
            self._create_room_candidate((100, 100, 50, 50), "OFFICE", 0.95),
            self._create_room_candidate((300, 300, 50, 50), "MEETING", 0.90),
        ]
        pass1_raw = []
        
        mock_ocr_instance = Mock()
        mock_ocr_instance.extract_and_find_rooms.return_value = (pass1_rooms, pass1_raw)
        mock_ocr_class.return_value = mock_ocr_instance
        
        mock_vlm = Mock()
        
        extractor = TwoPassOCRExtractor(
            ocr_config=self.ocr_config,
            vlm_backend=mock_vlm,
            confidence_threshold=0.7,
        )
        
        # Should skip Pass 2 since all results are high-confidence
        rooms, raw = extractor.extract_and_find_rooms_with_vlm_fallback("dummy.png")
        self.assertEqual(rooms, pass1_rooms)
        mock_vlm.detect_rooms.assert_not_called()  # VLM should not be called
    
    @patch('two_pass_ocr_extractor.MEPTextExtractor')
    def test_two_pass_with_low_confidence_results(self, mock_ocr_class):
        """Test two-pass extraction with low-confidence results."""
        # Mock Pass 1 results (mixed confidence)
        pass1_rooms = [
            self._create_room_candidate((100, 100, 50, 50), "BR", 0.5),  # Low
            self._create_room_candidate((300, 300, 50, 50), "OFFICE", 0.95),  # High
        ]
        pass1_raw = []
        
        mock_ocr_instance = Mock()
        mock_ocr_instance.extract_and_find_rooms.return_value = (pass1_rooms, pass1_raw)
        mock_ocr_class.return_value = mock_ocr_instance
        
        # Mock Pass 2 results
        pass2_rooms = [
            self._create_room_candidate((105, 105, 50, 50), "BEDROOM", 0.95),
        ]
        
        mock_vlm = Mock()
        mock_vlm.detect_rooms.return_value = [
            {"bbox": [105, 105, 155, 155], "room_name": "BEDROOM", "confidence": 0.95, "room_id": ""},
        ]
        
        extractor = TwoPassOCRExtractor(
            ocr_config=self.ocr_config,
            vlm_backend=mock_vlm,
            confidence_threshold=0.7,
        )
        
        rooms, raw = extractor.extract_and_find_rooms_with_vlm_fallback("dummy.png")
        
        # Should have called VLM for low-confidence fallback
        mock_vlm.detect_rooms.assert_called_once()
        
        # Results should include high-confidence OCR OFFICE + merged BEDROOM
        room_names = {r.room_name for r in rooms}
        self.assertIn("OFFICE", room_names)
        self.assertIn("BEDROOM", room_names)


class TestVLMRoomExtraction(unittest.TestCase):
    """Test VLM room extraction conversion."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ocr_config = OCRConfig(backend="paddleocr", device="cpu")
        self.mock_vlm = Mock()
        self.extractor = TwoPassOCRExtractor(
            ocr_config=self.ocr_config,
            vlm_backend=self.mock_vlm,
        )
    
    def test_extract_rooms_via_vlm_valid_format(self):
        """Test VLM room extraction with valid detection format."""
        vlm_detections = [
            {
                "bbox": [100, 100, 150, 150],
                "room_name": "OFFICE",
                "confidence": 0.95,
                "room_id": "room_1",
            },
            {
                "bbox": [300, 300, 350, 350],
                "room_name": "MEETING",
                "confidence": 0.90,
                "room_id": "room_2",
            },
        ]
        
        self.mock_vlm.detect_rooms.return_value = vlm_detections
        
        rooms = self.extractor._extract_rooms_via_vlm("dummy.png")
        
        self.assertEqual(len(rooms), 2)
        self.assertEqual(rooms[0].room_name, "OFFICE")
        self.assertEqual(rooms[1].room_name, "MEETING")
    
    def test_extract_rooms_via_vlm_missing_bbox(self):
        """Test VLM room extraction with missing bbox."""
        vlm_detections = [
            {
                "room_name": "OFFICE",
                "confidence": 0.95,
                # Missing bbox
            },
        ]
        
        self.mock_vlm.detect_rooms.return_value = vlm_detections
        
        rooms = self.extractor._extract_rooms_via_vlm("dummy.png")
        
        # Should skip detection with missing bbox
        self.assertEqual(len(rooms), 0)
    
    def test_extract_rooms_via_vlm_empty_response(self):
        """Test VLM room extraction with empty response."""
        self.mock_vlm.detect_rooms.return_value = []
        
        rooms = self.extractor._extract_rooms_via_vlm("dummy.png")
        
        self.assertEqual(len(rooms), 0)
    
    def test_extract_rooms_via_vlm_exception_handling(self):
        """Test VLM room extraction error handling."""
        self.mock_vlm.detect_rooms.side_effect = Exception("API error")
        
        with self.assertRaises(Exception):
            self.extractor._extract_rooms_via_vlm("dummy.png")


class TestIntegrationWithPipeline(unittest.TestCase):
    """Test integration with the annotation pipeline."""
    
    def test_extractor_type_compatibility(self):
        """Test that TwoPassOCRExtractor is compatible with MEPTextExtractor interface."""
        ocr_config = OCRConfig(backend="paddleocr", device="cpu")
        extractor = TwoPassOCRExtractor(
            ocr_config=ocr_config,
            vlm_backend=None,
        )
        
        # Should have extract_and_find_rooms_with_vlm_fallback method
        self.assertTrue(hasattr(extractor, "extract_and_find_rooms_with_vlm_fallback"))
        self.assertTrue(callable(getattr(extractor, "extract_and_find_rooms_with_vlm_fallback")))


if __name__ == "__main__":
    unittest.main()
