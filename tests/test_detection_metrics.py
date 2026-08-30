"""
tests/test_detection_metrics.py

Hand-computable scenarios for `evaluate_detections`, verifying the
self-contained mAP implementation against known-correct answers rather
than just checking it runs.
"""

from __future__ import annotations

from src.core.types import BoundingBox, Detection
from src.evaluation.detection_metrics import GroundTruthBox, evaluate_detections


class TestEvaluateDetections:
    def test_one_tp_one_fp_gives_ap_half(self):
        # 2 images, 1 class. Image1: prediction perfectly matches GT (TP).
        # Image2: prediction doesn't overlap GT at all (FP), and that GT
        # is never matched. Hand-computed expected AP: 0.5 (see the
        # module-level derivation in the original verification run).
        gt = {
            "img1": [GroundTruthBox(bbox=BoundingBox(0, 0, 10, 10), class_id=0, class_name="car")],
            "img2": [GroundTruthBox(bbox=BoundingBox(0, 0, 10, 10), class_id=0, class_name="car")],
        }
        preds = {
            "img1": [Detection(bbox=BoundingBox(0, 0, 10, 10), confidence=0.9, class_id=0, class_name="car")],
            "img2": [Detection(bbox=BoundingBox(50, 50, 60, 60), confidence=0.8, class_id=0, class_name="car")],
        }
        report = evaluate_detections(preds, gt, iou_threshold=0.5)
        assert abs(report.mean_average_precision - 0.5) < 1e-6

    def test_perfect_detector_gives_ap_one(self):
        gt = {
            "img1": [
                GroundTruthBox(bbox=BoundingBox(0, 0, 10, 10), class_id=0, class_name="car"),
                GroundTruthBox(bbox=BoundingBox(20, 20, 30, 30), class_id=0, class_name="car"),
            ]
        }
        preds = {
            "img1": [
                Detection(bbox=BoundingBox(0, 0, 10, 10), confidence=0.95, class_id=0, class_name="car"),
                Detection(bbox=BoundingBox(20, 20, 30, 30), confidence=0.9, class_id=0, class_name="car"),
            ]
        }
        report = evaluate_detections(preds, gt, iou_threshold=0.5)
        assert abs(report.mean_average_precision - 1.0) < 1e-6

    def test_no_predictions_gives_ap_zero(self):
        gt = {"img1": [GroundTruthBox(bbox=BoundingBox(0, 0, 10, 10), class_id=0, class_name="car")]}
        report = evaluate_detections({"img1": []}, gt, iou_threshold=0.5)
        assert report.mean_average_precision == 0.0

    def test_no_ground_truth_gives_empty_report(self):
        report = evaluate_detections({"img1": []}, {"img1": []}, iou_threshold=0.5)
        assert report.mean_average_precision == 0.0
        assert report.per_class == []
