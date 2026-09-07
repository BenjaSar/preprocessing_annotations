#!/usr/bin/env python3
"""
Validation Script for Multimodal OCR Implementation.

Tests all phases of the Solution C (Hybrid) implementation:
  - Phase 1: PaddleOCR integration
  - Phase 2: Semantic reconciliation
  - Phase 3: Qwen2.5-VL infrastructure
  - Phase 4: Production hardening

Usage:
    python validate_implementation.py [--full]
"""

import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class ValidationResult:
    """Store validation test result."""
    
    def __init__(self, test_name: str, passed: bool, details: str = "", error: str = ""):
        self.test_name = test_name
        self.passed = passed
        self.details = details
        self.error = error
    
    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        msg = f"{status}: {self.test_name}"
        if self.details:
            msg += f"\n  Details: {self.details}"
        if self.error:
            msg += f"\n  Error: {self.error}"
        return msg


class Phase1Validator:
    """Validate Phase 1: OCR Adapter."""
    
    def run_all(self) -> list[ValidationResult]:
        """Run all Phase 1 tests."""
        results = []
        
        logger.info("=" * 60)
        logger.info("PHASE 1: OCR Adapter & PaddleOCR Integration")
        logger.info("=" * 60)
        
        # Test 1: Import ocr_adapter module
        results.append(self._test_imports())
        
        # Test 2: OCRFactory creation
        results.append(self._test_factory())
        
        # Test 3: Configuration
        results.append(self._test_config())
        
        # Test 4: Backend registration
        results.append(self._test_backend_registration())
        
        return results
    
    def _test_imports(self) -> ValidationResult:
        """Test if ocr_adapter module can be imported."""
        try:
            from preprocessing_annotations.ingestion.ocr_adapter import OCRFactory, TextDetection, OCRBackend
            return ValidationResult(
                "Import ocr_adapter module",
                True,
                "Successfully imported OCRFactory, TextDetection, OCRBackend"
            )
        except Exception as e:
            return ValidationResult(
                "Import ocr_adapter module",
                False,
                error=str(e)
            )
    
    def _test_factory(self) -> ValidationResult:
        """Test OCRFactory can create backends."""
        try:
            from preprocessing_annotations.ingestion.ocr_adapter import OCRFactory
            from preprocessing_annotations.config import OCRConfig
            
            # Test EasyOCR backend creation (don't initialize)
            config = OCRConfig(backend="easyocr")
            backend = OCRFactory.create(config)
            
            if backend.__class__.__name__ != "EasyOCRBackend":
                return ValidationResult(
                    "OCRFactory creates correct backend",
                    False,
                    f"Expected EasyOCRBackend, got {backend.__class__.__name__}"
                )
            
            return ValidationResult(
                "OCRFactory creates correct backend",
                True,
                "EasyOCRBackend created successfully (not initialized)"
            )
        except Exception as e:
            return ValidationResult(
                "OCRFactory creates correct backend",
                False,
                error=str(e)
            )
    
    def _test_config(self) -> ValidationResult:
        """Test OCRConfig with backend field."""
        try:
            from preprocessing_annotations.config import OCRConfig
            
            config = OCRConfig(backend="paddleocr")
            if config.backend != "paddleocr":
                return ValidationResult(
                    "OCRConfig.backend field",
                    False,
                    f"Expected 'paddleocr', got {config.backend}"
                )
            
            return ValidationResult(
                "OCRConfig.backend field",
                True,
                "backend='paddleocr' configuration works"
            )
        except Exception as e:
            return ValidationResult(
                "OCRConfig.backend field",
                False,
                error=str(e)
            )
    
    def _test_backend_registration(self) -> ValidationResult:
        """Test custom backend registration."""
        try:
            from preprocessing_annotations.ingestion.ocr_adapter import OCRFactory, OCRBackend
            
            class DummyBackend(OCRBackend):
                def initialize(self): pass
                def extract_text(self, image_source): return []
            
            OCRFactory.register("dummy", DummyBackend)
            
            return ValidationResult(
                "Custom backend registration",
                True,
                "Successfully registered dummy backend"
            )
        except Exception as e:
            return ValidationResult(
                "Custom backend registration",
                False,
                error=str(e)
            )


class Phase2Validator:
    """Validate Phase 2: Semantic Reconciliation."""
    
    def run_all(self) -> list[ValidationResult]:
        """Run all Phase 2 tests."""
        results = []
        
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: Semantic Reconciliation")
        logger.info("=" * 60)
        
        # Test 1: Import semantic_reconciler
        results.append(self._test_imports())
        
        # Test 2: SemanticReconciler initialization
        results.append(self._test_initialization())
        
        # Test 3: Reconciliation logic
        results.append(self._test_reconciliation())
        
        # Test 4: Shapely integration check
        results.append(self._test_shapely_check())
        
        return results
    
    def _test_imports(self) -> ValidationResult:
        """Test if semantic_reconciler can be imported."""
        try:
            from preprocessing_annotations.vlm.semantic_reconciler import SemanticReconciler, ReconciledRoom, VLMRoom, OCRText
            return ValidationResult(
                "Import semantic_reconciler module",
                True,
                "Successfully imported SemanticReconciler and related classes"
            )
        except Exception as e:
            return ValidationResult(
                "Import semantic_reconciler module",
                False,
                error=str(e)
            )
    
    def _test_initialization(self) -> ValidationResult:
        """Test SemanticReconciler initialization."""
        try:
            from preprocessing_annotations.vlm.semantic_reconciler import SemanticReconciler
            from preprocessing_annotations.config import OCRConfig
            
            config = OCRConfig()
            reconciler = SemanticReconciler(config)
            
            return ValidationResult(
                "SemanticReconciler initialization",
                True,
                "SemanticReconciler created with config"
            )
        except Exception as e:
            return ValidationResult(
                "SemanticReconciler initialization",
                False,
                error=str(e)
            )
    
    def _test_reconciliation(self) -> ValidationResult:
        """Test reconciliation with mock data."""
        try:
            from preprocessing_annotations.vlm.semantic_reconciler import SemanticReconciler
            
            reconciler = SemanticReconciler()
            
            # Mock data
            ocr_results = [
                {
                    'text': 'BR',
                    'bbox': (10, 10, 50, 50),
                    'confidence': 0.8
                }
            ]
            
            vlm_results = [
                {
                    'room_id': 'room_0',
                    'polygon': [(0, 0), (100, 0), (100, 100), (0, 100)],
                    'room_type': 'bedroom',
                    'confidence': 0.9
                }
            ]
            
            # This should work (Shapely optional)
            results = reconciler.reconcile(ocr_results, vlm_results)
            
            if not isinstance(results, list):
                return ValidationResult(
                    "Reconciliation with mock data",
                    False,
                    f"Expected list, got {type(results)}"
                )
            
            return ValidationResult(
                "Reconciliation with mock data",
                True,
                f"Reconciled {len(results)} rooms (Shapely optional)"
            )
        except Exception as e:
            return ValidationResult(
                "Reconciliation with mock data",
                False,
                error=str(e)
            )
    
    def _test_shapely_check(self) -> ValidationResult:
        """Check Shapely availability."""
        try:
            import shapely
            return ValidationResult(
                "Shapely library availability",
                True,
                f"Shapely {shapely.__version__} available"
            )
        except ImportError:
            return ValidationResult(
                "Shapely library availability",
                False,
                details="Shapely not installed. Optional: pip install shapely",
                error="ImportError"
            )


class Phase3Validator:
    """Validate Phase 3: Qwen2.5-VL Infrastructure."""
    
    def run_all(self) -> list[ValidationResult]:
        """Run all Phase 3 tests."""
        results = []
        
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 3: Qwen2.5-VL Infrastructure")
        logger.info("=" * 60)
        
        # Test 1: Import vlm_backend
        results.append(self._test_imports())
        
        # Test 2: VLMFactory
        results.append(self._test_factory())
        
        # Test 3: Fine-tuning script
        results.append(self._test_finetune_script())
        
        # Test 4: LoRA config
        results.append(self._test_lora_config())
        
        return results
    
    def _test_imports(self) -> ValidationResult:
        """Test if vlm_backend can be imported."""
        try:
            from preprocessing_annotations.vlm.vlm_backend import VLMFactory, VLMBackend, ClaudeBackend, Qwen2_5VLBackend
            return ValidationResult(
                "Import vlm_backend module",
                True,
                "Successfully imported VLM backends and factory"
            )
        except Exception as e:
            return ValidationResult(
                "Import vlm_backend module",
                False,
                error=str(e)
            )
    
    def _test_factory(self) -> ValidationResult:
        """Test VLMFactory can create backends."""
        try:
            from preprocessing_annotations.vlm.vlm_backend import VLMFactory
            from preprocessing_annotations.config import VLMConfig
            
            # Test Claude backend
            config = VLMConfig(backend="claude")
            backend = VLMFactory.create(config)
            
            if backend.__class__.__name__ != "ClaudeBackend":
                return ValidationResult(
                    "VLMFactory creates Claude backend",
                    False,
                    f"Expected ClaudeBackend, got {backend.__class__.__name__}"
                )
            
            # Test Qwen backend (without initializing)
            config2 = VLMConfig(backend="qwen")
            backend2 = VLMFactory.create(config2)
            
            if backend2.__class__.__name__ != "Qwen2_5VLBackend":
                return ValidationResult(
                    "VLMFactory creates Qwen backend",
                    False,
                    f"Expected Qwen2_5VLBackend, got {backend2.__class__.__name__}"
                )
            
            return ValidationResult(
                "VLMFactory creates correct backends",
                True,
                "Both Claude and Qwen backends created successfully"
            )
        except Exception as e:
            return ValidationResult(
                "VLMFactory creates backends",
                False,
                error=str(e)
            )
    
    def _test_finetune_script(self) -> ValidationResult:
        """Test fine-tuning data preparator."""
        try:
            from finetune_qwen import QwenFinetuneDataPreparator, LoRAConfig
            
            # Check LoRA config
            config = LoRAConfig.to_dict()
            if 'r' not in config or 'lora_alpha' not in config:
                return ValidationResult(
                    "LoRA configuration",
                    False,
                    "Missing LoRA hyperparameters"
                )
            
            return ValidationResult(
                "Fine-tuning infrastructure",
                True,
                f"LoRA config with r={config['r']}, alpha={config['lora_alpha']}"
            )
        except Exception as e:
            return ValidationResult(
                "Fine-tuning infrastructure",
                False,
                error=str(e)
            )
    
    def _test_lora_config(self) -> ValidationResult:
        """Test LoRA configuration export."""
        try:
            from finetune_qwen import LoRAConfig
            
            config_dict = LoRAConfig.to_dict()
            required_keys = ['r', 'lora_alpha', 'learning_rate', 'num_train_epochs']
            
            for key in required_keys:
                if key not in config_dict:
                    return ValidationResult(
                        "LoRA config export",
                        False,
                        f"Missing key: {key}"
                    )
            
            return ValidationResult(
                "LoRA config export",
                True,
                f"All {len(config_dict)} LoRA hyperparameters present"
            )
        except Exception as e:
            return ValidationResult(
                "LoRA config export",
                False,
                error=str(e)
            )


class Phase4Validator:
    """Validate Phase 4: Production Hardening."""
    
    def run_all(self) -> list[ValidationResult]:
        """Run all Phase 4 tests."""
        results = []
        
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 4: Production Hardening & Monitoring")
        logger.info("=" * 60)
        
        # Test 1: Import production_monitor
        results.append(self._test_imports())
        
        # Test 2: Confidence thresholder
        results.append(self._test_confidence_thresholder())
        
        # Test 3: Quality gate
        results.append(self._test_quality_gate())
        
        # Test 4: Error recovery
        results.append(self._test_error_recovery())
        
        return results
    
    def _test_imports(self) -> ValidationResult:
        """Test if production_monitor can be imported."""
        try:
            from preprocessing_annotations.orchestration.production_monitor import (
                ConfidenceThresholder,
                QualityGate,
                ErrorRecovery,
                ProductionLogger,
                ProductionPipeline
            )
            return ValidationResult(
                "Import production_monitor module",
                True,
                "Successfully imported all production monitoring classes"
            )
        except Exception as e:
            return ValidationResult(
                "Import production_monitor module",
                False,
                error=str(e)
            )
    
    def _test_confidence_thresholder(self) -> ValidationResult:
        """Test confidence thresholding."""
        try:
            from preprocessing_annotations.orchestration.production_monitor import ConfidenceThresholder
            
            thresholder = ConfidenceThresholder(
                ocr_threshold=0.70,
                vlm_threshold=0.75,
                combined_threshold=0.65
            )
            
            # Mock annotations
            annotations = [
                {'confidence': 0.8, 'room_name': 'Room 1'},
                {'confidence': 0.5, 'room_name': 'Room 2'},
            ]
            
            keep, reject = thresholder.filter_annotations(annotations)
            
            if len(keep) != 1 or len(reject) != 1:
                return ValidationResult(
                    "Confidence thresholding",
                    False,
                    f"Expected 1 keep, 1 reject. Got {len(keep)}, {len(reject)}"
                )
            
            return ValidationResult(
                "Confidence thresholding",
                True,
                "Correctly filtered annotations by confidence"
            )
        except Exception as e:
            return ValidationResult(
                "Confidence thresholding",
                False,
                error=str(e)
            )
    
    def _test_quality_gate(self) -> ValidationResult:
        """Test quality validation."""
        try:
            from preprocessing_annotations.orchestration.production_monitor import QualityGate
            
            # Test with valid annotations
            annotations = [
                {'confidence': 0.8, 'text_labels': ['BR', 'Kitchen']},
                {'confidence': 0.75, 'text_labels': ['Bath']},
                {'confidence': 0.7, 'text_labels': ['Bedroom']},
            ]
            
            assessment = QualityGate.validate(annotations)
            
            if assessment.room_count != 3:
                return ValidationResult(
                    "Quality gate validation",
                    False,
                    f"Expected 3 rooms, got {assessment.room_count}"
                )
            
            return ValidationResult(
                "Quality gate validation",
                True,
                f"Quality assessment: {assessment.overall_confidence:.2f} confidence"
            )
        except Exception as e:
            return ValidationResult(
                "Quality gate validation",
                False,
                error=str(e)
            )
    
    def _test_error_recovery(self) -> ValidationResult:
        """Test error recovery mechanisms."""
        try:
            from preprocessing_annotations.orchestration.production_monitor import ErrorRecovery
            
            # Mock error
            error = Exception("Test error")
            
            # Mock OCR results
            ocr_results = [
                {'room_name': 'BR', 'room_number': '101', 'confidence': 0.8}
            ]
            
            # Fallback should work
            fallback = ErrorRecovery.fallback_to_ocr_only(ocr_results, [], error)
            
            if len(fallback) != 1:
                return ValidationResult(
                    "Error recovery fallback",
                    False,
                    f"Expected 1 fallback annotation, got {len(fallback)}"
                )
            
            return ValidationResult(
                "Error recovery fallback",
                True,
                "OCR-only fallback generated successfully"
            )
        except Exception as e:
            return ValidationResult(
                "Error recovery fallback",
                False,
                error=str(e)
            )


def main():
    """Run full validation suite."""
    print("\n" + "=" * 60)
    print("MULTIMODAL OCR IMPLEMENTATION VALIDATION")
    print("=" * 60)
    print("Validating all 4 phases of Solution C (Hybrid) implementation\n")
    
    all_results = []
    
    # Phase 1
    phase1 = Phase1Validator()
    all_results.extend(phase1.run_all())
    
    # Phase 2
    phase2 = Phase2Validator()
    all_results.extend(phase2.run_all())
    
    # Phase 3
    phase3 = Phase3Validator()
    all_results.extend(phase3.run_all())
    
    # Phase 4
    phase4 = Phase4Validator()
    all_results.extend(phase4.run_all())
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    for result in all_results:
        print(result)
        print()
    
    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n✅ All validations passed! Implementation is ready.")
        print("\nNext steps:")
        print("1. Integrate Phase 1 (PaddleOCR) into pipeline.py")
        print("2. Test on sample floor plans")
        print("3. Benchmark against EasyOCR")
        print("4. Enable Phase 2 (semantic reconciliation)")
        print("5. If needed, proceed to Phase 3 (fine-tuning)")
        return 0
    else:
        print("\n❌ Some validations failed. See details above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
