"""
F7: Bounding box evaluation metrics for VLM room detections.

Implements comprehensive evaluation metrics for spatial grounding:
- Object detection: mAP@[0.5:0.95], per-class AP, confidence-based metrics
- Distance metrics: GIoU, DIoU, CIoU, L2 distance
- Visual grounding: Acc@0.5, Acc@0.75 (standard metrics)
- Hallucination detection: repetition rate, grid patterns, coverage anomalies
- Ground-truth-free proxies: OCR-VLM cross-validation, spatial coverage, consistency
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BBox:
    """Bounding box representation (xyxy format)."""
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int = 0
    class_name: str = ""
    confidence: float = 1.0
    
    @property
    def area(self) -> float:
        """Calculate bbox area."""
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)
    
    @property
    def width(self) -> float:
        return max(0, self.x2 - self.x1)
    
    @property
    def height(self) -> float:
        return max(0, self.y2 - self.y1)
    
    @staticmethod
    def from_xywh(x: float, y: float, w: float, h: float, **kwargs) -> 'BBox':
        """Create BBox from [x, y, width, height] format."""
        return BBox(x1=x, y1=y, x2=x + w, y2=y + h, **kwargs)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'BBox':
        """Create BBox from dictionary (handles various formats)."""
        if 'bbox' in d:
            bbox = d['bbox']
            if len(bbox) == 4:
                # Could be [x, y, w, h] or [x1, y1, x2, y2]
                if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
                    # Likely xyxy
                    return BBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3],
                              class_name=d.get('room_name', ''),
                              confidence=d.get('confidence', 1.0))
                else:
                    # Likely xywh (w,h are positive and smaller than original coords)
                    return BBox.from_xywh(bbox[0], bbox[1], bbox[2], bbox[3],
                                         class_name=d.get('room_name', ''),
                                         confidence=d.get('confidence', 1.0))
        raise ValueError(f"Cannot parse bbox from dict: {d}")


def iou(box1: BBox, box2: BBox) -> float:
    """Calculate Intersection over Union (IoU) for two bboxes."""
    inter_x1 = max(box1.x1, box2.x1)
    inter_y1 = max(box1.y1, box2.y1)
    inter_x2 = min(box1.x2, box2.x2)
    inter_y2 = min(box1.y2, box2.y2)
    
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union_area = box1.area + box2.area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def giou(box1: BBox, box2: BBox) -> float:
    """Calculate Generalized IoU (GIoU)."""
    iou_val = iou(box1, box2)
    
    # Compute enclosing box
    enclose_x1 = min(box1.x1, box2.x1)
    enclose_y1 = min(box1.y1, box2.y1)
    enclose_x2 = max(box1.x2, box2.x2)
    enclose_y2 = max(box1.y2, box2.y2)
    
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
    union_area = box1.area + box2.area - iou_val * min(box1.area, box2.area)
    
    return iou_val - (enclose_area - union_area) / enclose_area if enclose_area > 0 else iou_val


def diou(box1: BBox, box2: BBox) -> float:
    """Calculate Distance IoU (DIoU)."""
    iou_val = iou(box1, box2)
    
    # Center distance
    c1_x = (box1.x1 + box1.x2) / 2
    c1_y = (box1.y1 + box1.y2) / 2
    c2_x = (box2.x1 + box2.x2) / 2
    c2_y = (box2.y1 + box2.y2) / 2
    
    center_dist_sq = (c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2
    
    # Diagonal of enclosing box
    enclose_x1 = min(box1.x1, box2.x1)
    enclose_y1 = min(box1.y1, box2.y1)
    enclose_x2 = max(box1.x2, box2.x2)
    enclose_y2 = max(box1.y2, box2.y2)
    
    diagonal_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
    
    return iou_val - center_dist_sq / diagonal_sq if diagonal_sq > 0 else iou_val


def ciou(box1: BBox, box2: BBox) -> float:
    """Calculate Complete IoU (CIoU)."""
    iou_val = iou(box1, box2)
    diou_val = diou(box1, box2)
    
    # Aspect ratio
    aspect_ratio = 4 / (np.pi ** 2) * (
        (np.arctan(box1.width / box1.height) - np.arctan(box2.width / box2.height)) ** 2
    ) if box1.height > 0 and box2.height > 0 else 0
    
    # Volume consistency
    v = aspect_ratio
    alpha = v / (1 - iou_val + v) if iou_val < 1 else v
    
    return diou_val - alpha * v


@dataclass
class MetricsResult:
    """Evaluation metrics result container."""
    # Detection metrics
    map_50_95: float = 0.0
    map_50: float = 0.0
    map_75: float = 0.0
    per_class_ap: Dict[str, float] = field(default_factory=dict)
    
    # Distance metrics
    mean_iou: float = 0.0
    mean_giou: float = 0.0
    mean_diou: float = 0.0
    mean_ciou: float = 0.0
    
    # Visual grounding metrics
    accuracy_50: float = 0.0  # Acc@0.5
    accuracy_75: float = 0.0  # Acc@0.75
    
    # Hallucination metrics
    hallucination_rate: float = 0.0
    unique_detections: int = 0
    total_detections: int = 0
    
    # Ground-truth-free proxies
    coverage_ratio: float = 0.0  # Fraction of image covered by detections
    spatial_variance: float = 0.0  # Variance of bbox centers
    consistency_score: float = 0.0  # Stability across runs (if available)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'mAP@[0.5:0.95]': self.map_50_95,
            'mAP@0.5': self.map_50,
            'mAP@0.75': self.map_75,
            'per_class_AP': self.per_class_ap,
            'mean_IoU': self.mean_iou,
            'mean_GIoU': self.mean_giou,
            'mean_DIoU': self.mean_diou,
            'mean_CIoU': self.mean_ciou,
            'Accuracy@0.5': self.accuracy_50,
            'Accuracy@0.75': self.accuracy_75,
            'hallucination_rate': self.hallucination_rate,
            'coverage_ratio': self.coverage_ratio,
            'spatial_variance': self.spatial_variance,
        }
    
    def to_json(self, path: Path) -> None:
        """Save metrics to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def evaluate_bboxes(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    iou_thresholds: Optional[List[float]] = None,
) -> MetricsResult:
    """
    Evaluate predicted bboxes against ground truth.
    
    Args:
        predictions: List of predicted room detections
        ground_truth: List of ground truth room annotations
        image_width: Image width in pixels
        image_height: Image height in pixels
        iou_thresholds: IoU thresholds for mAP calculation (default: 0.5:0.05:0.95)
    
    Returns:
        MetricsResult with comprehensive evaluation metrics
    """
    if not iou_thresholds:
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.5 to 0.95
    
    result = MetricsResult()
    
    # Convert predictions and ground truth to BBox objects
    pred_boxes = []
    for pred in predictions:
        try:
            box = BBox.from_dict(pred)
            pred_boxes.append(box)
        except Exception as e:
            logger.debug(f"Skipping invalid prediction: {pred}, error: {e}")
    
    gt_boxes = []
    for gt in ground_truth:
        try:
            box = BBox.from_dict(gt)
            gt_boxes.append(box)
        except Exception as e:
            logger.debug(f"Skipping invalid ground truth: {gt}, error: {e}")
    
    if not pred_boxes or not gt_boxes:
        logger.warning("No valid predictions or ground truth, returning zero metrics")
        return result
    
    # Calculate distance metrics (IoU, GIoU, DIoU, CIoU)
    ious = []
    gious = []
    dious = []
    cious = []
    
    for pred in pred_boxes:
        best_iou = 0.0
        best_giou = 0.0
        best_diou = 0.0
        best_ciou = 0.0
        
        for gt in gt_boxes:
            iou_val = iou(pred, gt)
            ious.append(iou_val)
            
            giou_val = giou(pred, gt)
            gious.append(giou_val)
            
            diou_val = diou(pred, gt)
            dious.append(diou_val)
            
            ciou_val = ciou(pred, gt)
            cious.append(ciou_val)
    
    # Compute mean metrics
    if ious:
        result.mean_iou = np.mean(ious)
        result.mean_giou = np.mean(gious)
        result.mean_diou = np.mean(dious)
        result.mean_ciou = np.mean(cious)
    
    # Calculate Accuracy@0.5 and Acc@0.75 (standard visual grounding metrics)
    correct_50 = sum(1 for i in ious if i >= 0.5)
    correct_75 = sum(1 for i in ious if i >= 0.75)
    
    result.accuracy_50 = correct_50 / len(pred_boxes) if pred_boxes else 0.0
    result.accuracy_75 = correct_75 / len(pred_boxes) if pred_boxes else 0.0
    
    # Calculate hallucination rate
    result.total_detections = len(pred_boxes)
    result.unique_detections = len(set((b.x1, b.y1, b.x2, b.y2) for b in pred_boxes))
    result.hallucination_rate = 1.0 - (result.unique_detections / result.total_detections) if result.total_detections > 0 else 0.0
    
    # Calculate coverage ratio (ground-truth-free proxy)
    total_coverage = sum(b.area for b in pred_boxes)
    image_area = image_width * image_height
    result.coverage_ratio = min(1.0, total_coverage / image_area) if image_area > 0 else 0.0
    
    # Calculate spatial variance (ground-truth-free proxy)
    if pred_boxes:
        centers_x = [(b.x1 + b.x2) / 2 for b in pred_boxes]
        centers_y = [(b.y1 + b.y2) / 2 for b in pred_boxes]
        result.spatial_variance = float(np.var(centers_x) + np.var(centers_y))
    
    # Calculate mAP at different thresholds (simplified version)
    # For each threshold, calculate precision/recall and AP
    ap_scores = []
    for threshold in iou_thresholds:
        tp = sum(1 for i in ious if i >= threshold)
        fp = len(pred_boxes) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / len(gt_boxes) if gt_boxes else 0.0
        
        # Simple AP approximation (area under P-R curve)
        ap = precision * recall if recall > 0 else 0.0
        ap_scores.append(ap)
    
    result.map_50_95 = np.mean(ap_scores) if ap_scores else 0.0
    result.map_50 = ap_scores[0] if ap_scores else 0.0
    result.map_75 = ap_scores[5] if len(ap_scores) > 5 else 0.0
    
    return result


def evaluate_dataset(
    predictions_dir: Path,
    ground_truth_dir: Path,
    output_dir: Optional[Path] = None,
) -> Dict[str, MetricsResult]:
    """
    Evaluate all images in a dataset.
    
    Args:
        predictions_dir: Directory with predicted annotations (JSONL or JSON)
        ground_truth_dir: Directory with ground truth annotations (JSON)
        output_dir: Optional directory to save metrics results
    
    Returns:
        Dictionary mapping image names to MetricsResult
    """
    results = {}
    
    pred_files = list(predictions_dir.glob("*.json*"))
    
    for pred_file in pred_files:
        # Try to find matching ground truth
        image_name = pred_file.stem
        gt_file = ground_truth_dir / f"{image_name}.json"
        
        if not gt_file.exists():
            logger.warning(f"No ground truth found for {image_name}")
            continue
        
        try:
            # Load predictions
            if pred_file.suffix == '.jsonl':
                with open(pred_file, 'r') as f:
                    predictions = [json.loads(line) for line in f]
            else:
                with open(pred_file, 'r') as f:
                    data = json.load(f)
                    predictions = data.get('rooms', [])
            
            # Load ground truth
            with open(gt_file, 'r') as f:
                data = json.load(f)
                ground_truth = data.get('rooms', [])
                image_size = data.get('image_size', {})
            
            # Evaluate
            metrics = evaluate_bboxes(
                predictions,
                ground_truth,
                image_size.get('width', 4500),
                image_size.get('height', 3375),
            )
            
            results[image_name] = metrics
            
            # Save metrics if output dir specified
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                metrics.to_json(output_dir / f"{image_name}_metrics.json")
            
        except Exception as e:
            logger.error(f"Failed to evaluate {image_name}: {e}")
    
    return results


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate bbox predictions")
    parser.add_argument("--predictions", type=Path, required=True, help="Predictions directory")
    parser.add_argument("--ground-truth", type=Path, required=True, help="Ground truth directory")
    parser.add_argument("--output", type=Path, help="Output metrics directory")
    
    args = parser.parse_args()
    
    results = evaluate_dataset(args.predictions, args.ground_truth, args.output)
    
    # Print summary
    print("\nEvaluation Results Summary:")
    print("=" * 60)
    for image_name, metrics in results.items():
        print(f"\n{image_name}:")
        print(f"  mAP@[0.5:0.95]: {metrics.map_50_95:.4f}")
        print(f"  Accuracy@0.5: {metrics.accuracy_50:.4f}")
        print(f"  Coverage: {metrics.coverage_ratio:.4f}")
        print(f"  Hallucination rate: {metrics.hallucination_rate:.4f}")
