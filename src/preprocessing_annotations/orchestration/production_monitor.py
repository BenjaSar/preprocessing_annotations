"""
Production Monitoring & Hardening Module (Phase 4).

Implements production-grade features:
  - Confidence thresholding and fallback strategies
  - Error recovery and graceful degradation
  - Detailed logging and telemetry
  - Validation gates and quality assurance
  - Performance monitoring

This module ensures the Solution C (Hybrid) pipeline is robust in production.
"""

import logging
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)


class ConfidenceThreshold(Enum):
    """Confidence-based decision thresholds."""
    HIGH = 0.85      # High confidence - use without validation
    MEDIUM = 0.70    # Medium confidence - validate or flag for review
    LOW = 0.50       # Low confidence - fallback strategy
    REJECT = 0.30    # Too low - reject annotation


@dataclass
class QualityMetric:
    """Single quality metric for an annotation."""
    metric_name: str
    value: float
    threshold: float
    passed: bool
    details: str = ""


@dataclass
class AnnotationQuality:
    """Quality assessment for a single annotation."""
    image_id: str
    overall_confidence: float
    room_count: int
    metrics: List[QualityMetric] = field(default_factory=list)
    passed_quality_gate: bool = False
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ConfidenceThresholder:
    """Apply confidence-based thresholding to annotations."""
    
    def __init__(
        self,
        ocr_threshold: float = 0.70,
        vlm_threshold: float = 0.75,
        combined_threshold: float = 0.65
    ):
        """Initialize thresholder.
        
        Args:
            ocr_threshold: Minimum OCR confidence to use text labels
            vlm_threshold: Minimum VLM confidence to trust room detection
            combined_threshold: Minimum confidence after reconciliation
        """
        self.ocr_threshold = ocr_threshold
        self.vlm_threshold = vlm_threshold
        self.combined_threshold = combined_threshold
    
    def filter_annotations(
        self,
        annotations: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Filter annotations by confidence thresholds.
        
        Args:
            annotations: List of reconciled room annotations
        
        Returns:
            (keep, reject) - tuple of kept and rejected annotations
        """
        keep = []
        reject = []
        
        for ann in annotations:
            confidence = ann.get('confidence', 0.0)
            
            if confidence >= self.combined_threshold:
                keep.append(ann)
            else:
                # Add rejection reason
                ann['rejection_reason'] = f"Low confidence: {confidence:.3f} < {self.combined_threshold}"
                reject.append(ann)
        
        logger.info(
            f"Confidence filter: kept {len(keep)}, rejected {len(reject)} annotations"
        )
        
        return keep, reject


class QualityGate:
    """Validate annotations against quality gates."""
    
    # Quality gate thresholds
    ROOM_COUNT_MIN = 3  # Minimum rooms per image
    ROOM_COUNT_MAX = 100  # Maximum rooms per image
    AVG_CONFIDENCE_MIN = 0.65  # Average confidence across rooms
    TEXT_LABELS_MIN = 3  # Minimum total text labels detected
    
    @staticmethod
    def validate(annotations: List[Dict[str, Any]]) -> AnnotationQuality:
        """
        Validate annotations against quality gates.
        
        Args:
            annotations: List of room annotations for one image
        
        Returns:
            AnnotationQuality assessment
        """
        image_id = annotations[0].get('image_id', 'unknown') if annotations else 'unknown'
        room_count = len(annotations)
        
        assessment = AnnotationQuality(
            image_id=image_id,
            overall_confidence=0.0,
            room_count=room_count,
        )
        
        if room_count == 0:
            assessment.issues.append("No rooms detected")
            return assessment
        
        # Calculate overall confidence
        confidences = [a.get('confidence', 0.5) for a in annotations]
        overall_conf = sum(confidences) / len(confidences)
        assessment.overall_confidence = overall_conf
        
        # Room count validation
        if room_count < QualityGate.ROOM_COUNT_MIN:
            assessment.issues.append(
                f"Too few rooms: {room_count} < {QualityGate.ROOM_COUNT_MIN}"
            )
        elif room_count > QualityGate.ROOM_COUNT_MAX:
            assessment.issues.append(
                f"Too many rooms: {room_count} > {QualityGate.ROOM_COUNT_MAX}"
            )
        
        assessment.metrics.append(QualityMetric(
            metric_name="room_count",
            value=float(room_count),
            threshold=float(QualityGate.ROOM_COUNT_MIN),
            passed=QualityGate.ROOM_COUNT_MIN <= room_count <= QualityGate.ROOM_COUNT_MAX
        ))
        
        # Average confidence validation
        if overall_conf < QualityGate.AVG_CONFIDENCE_MIN:
            assessment.warnings.append(
                f"Low average confidence: {overall_conf:.3f} < {QualityGate.AVG_CONFIDENCE_MIN}"
            )
        
        assessment.metrics.append(QualityMetric(
            metric_name="avg_confidence",
            value=overall_conf,
            threshold=float(QualityGate.AVG_CONFIDENCE_MIN),
            passed=overall_conf >= QualityGate.AVG_CONFIDENCE_MIN
        ))
        
        # Text labels validation
        total_labels = sum(len(a.get('text_labels', [])) for a in annotations)
        if total_labels < QualityGate.TEXT_LABELS_MIN:
            assessment.warnings.append(
                f"Few text labels: {total_labels} < {QualityGate.TEXT_LABELS_MIN}"
            )
        
        assessment.metrics.append(QualityMetric(
            metric_name="text_labels",
            value=float(total_labels),
            threshold=float(QualityGate.TEXT_LABELS_MIN),
            passed=total_labels >= QualityGate.TEXT_LABELS_MIN
        ))
        
        # Determine if it passes overall
        assessment.passed_quality_gate = len(assessment.issues) == 0
        
        return assessment


class ErrorRecovery:
    """Error recovery and fallback strategies."""
    
    @staticmethod
    def fallback_to_ocr_only(
        ocr_results: List[Dict[str, Any]],
        vlm_results: List[Dict[str, Any]],
        error: Exception
    ) -> List[Dict[str, Any]]:
        """
        Fallback strategy: use OCR-only results if VLM fails.
        
        Args:
            ocr_results: OCR-only room candidates (no VLM semantic enhancement)
            vlm_results: VLM detections (may be partial/corrupted)
            error: Exception that triggered fallback
        
        Returns:
            Fallback annotations
        """
        logger.warning(
            f"VLM failed ({type(error).__name__}), falling back to OCR-only. "
            f"Error: {str(error)[:100]}"
        )
        
        # Convert OCR candidates to annotation format
        fallback = []
        for idx, ocr in enumerate(ocr_results):
            fallback.append({
                "room_id": f"ocr_fallback_{idx}",
                "room_name": ocr.get('room_name', 'Room'),
                "room_number": ocr.get('room_number'),
                "room_type": "other",  # Cannot infer type without VLM
                "bbox": ocr.get('bbox'),
                "confidence": ocr.get('confidence', 0.6) * 0.8,  # Discount confidence
                "text_labels": [ocr.get('room_name', 'Room')],
                "provenance": {
                    "source": "ocr_fallback",
                    "error_type": type(error).__name__
                }
            })
        
        return fallback
    
    @staticmethod
    def handle_exception(
        error: Exception,
        context: str,
        fallback_fn: Optional[Callable] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Handle exceptions with logging and recovery.
        
        Args:
            error: Exception that occurred
            context: Description of what was being attempted
            fallback_fn: Optional fallback function to call
        
        Returns:
            Fallback result or None
        """
        logger.error(
            f"Error in {context}: {type(error).__name__}: {str(error)}",
            exc_info=True
        )
        
        if fallback_fn:
            try:
                return fallback_fn()
            except Exception as fallback_error:
                logger.error(
                    f"Fallback also failed: {type(fallback_error).__name__}",
                    exc_info=True
                )
        
        return None


class ProductionLogger:
    """Structured logging for production monitoring."""
    
    def __init__(self, log_dir: Path):
        """Initialize production logger.
        
        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create handlers for different log levels
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup file-based logging handlers."""
        # Main logger
        file_handler = logging.FileHandler(
            self.log_dir / "pipeline.log"
        )
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def log_processing_stats(self, stats: Dict[str, Any]) -> None:
        """Log processing statistics."""
        stats_file = self.log_dir / f"stats_{datetime.now().strftime('%Y%m%d')}.json"
        
        with open(stats_file, 'a') as f:
            f.write(json.dumps({
                'timestamp': datetime.now().isoformat(),
                **stats
            }) + '\n')
    
    def log_quality_assessments(self, assessments: List[AnnotationQuality]) -> None:
        """Log quality assessments."""
        quality_file = self.log_dir / "quality_assessments.jsonl"
        
        with open(quality_file, 'a') as f:
            for assessment in assessments:
                f.write(json.dumps(asdict(assessment)) + '\n')


class ProductionPipeline:
    """
    Production-grade pipeline wrapper with hardening features.
    
    Wraps the annotation pipeline with:
    - Confidence thresholding
    - Error recovery
    - Quality gates
    - Comprehensive logging
    """
    
    def __init__(self, pipeline, config=None):
        """Initialize production wrapper.
        
        Args:
            pipeline: Underlying AnnotationPipeline instance
            config: Optional configuration with thresholds
        """
        self.pipeline = pipeline
        self.config = config
        
        # Initialize hardening components
        self.thresholder = ConfidenceThresholder()
        self.logger = ProductionLogger(Path("./logs"))
    
    def process_with_hardening(
        self,
        image_path: str,
        enable_fallback: bool = True
    ) -> tuple[List[Dict[str, Any]], AnnotationQuality, Dict[str, Any]]:
        """
        Process image with all hardening features.
        
        Args:
            image_path: Path to floor plan image
            enable_fallback: Whether to enable error recovery fallback
        
        Returns:
            (annotations, quality_assessment, telemetry)
        """
        telemetry = {
            'image_path': str(image_path),
            'start_time': datetime.now().isoformat(),
            'stages': {}
        }
        
        try:
            # Stage 1: Run pipeline
            logger.info(f"Processing {image_path}")
            annotations = self.pipeline.process(image_path)
            
            # Stage 2: Apply confidence filtering
            keep, reject = self.thresholder.filter_annotations(annotations)
            telemetry['stages']['filtering'] = {
                'kept': len(keep),
                'rejected': len(reject)
            }
            
            # Stage 3: Quality assessment
            quality = QualityGate.validate(keep)
            telemetry['stages']['quality'] = asdict(quality)
            
            # Stage 4: Logging
            self.logger.log_quality_assessments([quality])
            self.logger.log_processing_stats(telemetry)
            
            logger.info(
                f"Processing complete: {len(keep)} annotations kept, "
                f"quality gate {'PASSED' if quality.passed_quality_gate else 'FAILED'}"
            )
            
            return keep, quality, telemetry
            
        except Exception as e:
            telemetry['error'] = {
                'type': type(e).__name__,
                'message': str(e)
            }
            
            if enable_fallback:
                logger.warning(f"Pipeline failed, attempting OCR-only fallback")
                # In real implementation, this would extract OCR results from error state
                fallback = ErrorRecovery.fallback_to_ocr_only([], [], e)
                quality = QualityGate.validate(fallback)
                telemetry['fallback'] = True
                return fallback, quality, telemetry
            else:
                raise
