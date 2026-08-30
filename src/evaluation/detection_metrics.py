"""
src/evaluation/detection_metrics.py

A self-contained, numpy-only mean Average Precision (mAP) calculator for
object detection -- the standard Pascal VOC / COCO-style algorithm
(per-class precision-recall curve via greedy IoU matching in descending
confidence order, all-point interpolated area under the curve).

Deliberately NOT a `torchmetrics.detection.MeanAveragePrecision` wrapper:
that requires torch + torchmetrics installed just to run an evaluation
script, whereas this implementation needs only numpy (already a
dependency of everything in this project) and is small enough to be
fully auditable in one file -- matching the same reasoning already
applied to `src/tracking/bytetrack_tracker.py` (self-own the well-known
algorithm rather than add a heavy/version-sensitive dependency for it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.core.types import BoundingBox, Detection


@dataclass(frozen=True)
class GroundTruthBox:
    """A single labeled object in one image (no confidence -- ground
    truth is certain by definition)."""

    bbox: BoundingBox
    class_id: int
    class_name: str


@dataclass(frozen=True)
class ClassAveragePrecision:
    class_id: int
    class_name: str
    average_precision: float
    num_ground_truth: int
    num_predictions: int


@dataclass(frozen=True)
class EvaluationReport:
    mean_average_precision: float
    iou_threshold: float
    per_class: List[ClassAveragePrecision]
    num_images: int

    def to_dict(self) -> dict:
        return {
            "mean_average_precision": self.mean_average_precision,
            "iou_threshold": self.iou_threshold,
            "num_images": self.num_images,
            "per_class": [
                {
                    "class_id": c.class_id,
                    "class_name": c.class_name,
                    "average_precision": c.average_precision,
                    "num_ground_truth": c.num_ground_truth,
                    "num_predictions": c.num_predictions,
                }
                for c in self.per_class
            ],
        }


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _average_precision_from_curve(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """All-point interpolated AP (COCO / Pascal VOC 2010+ convention):
    precision is made monotonically non-increasing from the right before
    integrating, so noisy fluctuations in the raw precision curve don't
    understate AP."""
    # Prepend/append sentinel points so the curve spans recall in [0, 1].
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Integrate via the trapezoidal-on-steps rule: sum of
    # (recall_i - recall_{i-1}) * precision_i wherever recall changes.
    change_points = np.where(recalls[1:] != recalls[:-1])[0] + 1
    return float(np.sum((recalls[change_points] - recalls[change_points - 1]) * precisions[change_points]))


def evaluate_detections(
    predictions_by_image: Dict[str, List[Detection]],
    ground_truths_by_image: Dict[str, List[GroundTruthBox]],
    iou_threshold: float = 0.5,
) -> EvaluationReport:
    """Compute per-class AP and overall mAP at a single IoU threshold.

    Args:
        predictions_by_image: image identifier -> that image's detector
            output (same identifiers as `ground_truths_by_image`).
        ground_truths_by_image: image identifier -> that image's labeled
            ground-truth boxes.
        iou_threshold: minimum IoU for a prediction to count as a match
            (0.5 is the conventional "mAP@0.5" threshold).
    """
    class_ids = sorted(
        {gt.class_id for gts in ground_truths_by_image.values() for gt in gts}
    )

    per_class_results: List[ClassAveragePrecision] = []
    for class_id in class_ids:
        class_name = _class_name_for(class_id, ground_truths_by_image)
        ap, num_gt, num_pred = _average_precision_for_class(
            class_id, predictions_by_image, ground_truths_by_image, iou_threshold
        )
        per_class_results.append(
            ClassAveragePrecision(
                class_id=class_id,
                class_name=class_name,
                average_precision=ap,
                num_ground_truth=num_gt,
                num_predictions=num_pred,
            )
        )

    mean_ap = float(np.mean([c.average_precision for c in per_class_results])) if per_class_results else 0.0
    return EvaluationReport(
        mean_average_precision=mean_ap,
        iou_threshold=iou_threshold,
        per_class=per_class_results,
        num_images=len(ground_truths_by_image),
    )


def _class_name_for(class_id: int, ground_truths_by_image: Dict[str, List[GroundTruthBox]]) -> str:
    for gts in ground_truths_by_image.values():
        for gt in gts:
            if gt.class_id == class_id:
                return gt.class_name
    return str(class_id)


def _average_precision_for_class(
    class_id: int,
    predictions_by_image: Dict[str, List[Detection]],
    ground_truths_by_image: Dict[str, List[GroundTruthBox]],
    iou_threshold: float,
) -> tuple:
    # Flatten this class's predictions across all images, sorted by
    # confidence descending -- the order greedy TP/FP assignment depends on.
    flat_predictions = []
    for image_id, detections in predictions_by_image.items():
        for det in detections:
            if det.class_id == class_id:
                flat_predictions.append((image_id, det))
    flat_predictions.sort(key=lambda item: item[1].confidence, reverse=True)

    gt_by_image: Dict[str, List[GroundTruthBox]] = {
        image_id: [gt for gt in gts if gt.class_id == class_id]
        for image_id, gts in ground_truths_by_image.items()
    }
    num_gt = sum(len(gts) for gts in gt_by_image.values())
    matched: Dict[str, List[bool]] = {image_id: [False] * len(gts) for image_id, gts in gt_by_image.items()}

    if num_gt == 0 or not flat_predictions:
        return 0.0, num_gt, len(flat_predictions)

    tp = np.zeros(len(flat_predictions))
    fp = np.zeros(len(flat_predictions))

    for i, (image_id, pred) in enumerate(flat_predictions):
        candidates = gt_by_image.get(image_id, [])
        best_iou = 0.0
        best_idx = -1
        for gt_idx, gt in enumerate(candidates):
            if matched[image_id][gt_idx]:
                continue
            iou = _iou(pred.bbox, gt.bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = gt_idx

        if best_iou >= iou_threshold and best_idx >= 0:
            tp[i] = 1.0
            matched[image_id][best_idx] = True
        else:
            fp[i] = 1.0

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / num_gt
    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)

    ap = _average_precision_from_curve(recalls, precisions)
    return ap, num_gt, len(flat_predictions)
