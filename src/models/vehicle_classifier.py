"""
src/models/vehicle_classifier.py

Vehicle attribute classification (type/color/brand), implementing
`BaseClassifier` (`src/core/base_model.py`) three ways that compose via
dependency injection rather than one another's inheritance:

    HistogramColorClassifier   -- real, dependency-light (opencv + numpy
        only), classifies dominant color via an HSV histogram binned into
        named color buckets. Runtime-verified in `tests/test_classifiers.py`.

    TimmBackboneClassifier     -- deep-learning type/brand classification
        via a `timm` backbone + linear heads. NOT runtime-verified here
        (torch/timm unavailable in this sandbox) -- reviewed against
        `timm`'s documented API, same status as `YoloAdapter`.

    CompositeVehicleClassifier -- DI composition: runs multiple injected
        `BaseClassifier`s over the same crop and merges their
        `VehicleAttributes`, so color (cheap, classical CV) and
        type/brand (deep learning) can be produced by two independently
        swappable/testable classifiers rather than one monolithic model
        that would force retraining to fix just one attribute.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.base_model import BaseClassifier
from src.core.exceptions import ModelLoadError
from src.core.registry import register
from src.core.types import VehicleAttributes

logger = logging.getLogger(__name__)

# HSV hue ranges (OpenCV's 0-179 hue scale) for each named color bucket.
# Red wraps around 0/179, so it gets two ranges. Black/white/gray are
# distinguished by low saturation / extreme value rather than hue.
_HUE_COLOR_BUCKETS: List[Tuple[str, Tuple[int, int]]] = [
    ("red", (0, 8)),
    ("orange", (9, 20)),
    ("yellow", (21, 33)),
    ("green", (34, 78)),
    ("blue", (79, 130)),
    ("purple", (131, 155)),
    ("red", (156, 179)),
]


@register("classifier", "histogram_color")
class HistogramColorClassifier(BaseClassifier):
    """Classifies a vehicle crop's dominant color via an HSV histogram.

    Deliberately not a learned model: color is one of the few vehicle
    attributes where classical CV (dominant hue + saturation/value
    gating for achromatic colors) is genuinely competitive with a small
    trained classifier, and it needs zero training data or GPU -- a
    reasonable default to ship immediately, with `TimmBackboneClassifier`
    as the upgrade path once labeled color data exists.
    """

    def __init__(
        self,
        low_saturation_threshold: int = 40,
        white_value_threshold: int = 200,
        black_value_threshold: int = 50,
        center_crop_fraction: float = 0.6,
    ) -> None:
        self.low_saturation_threshold = low_saturation_threshold
        self.white_value_threshold = white_value_threshold
        self.black_value_threshold = black_value_threshold
        self.center_crop_fraction = center_crop_fraction

    def classify(self, vehicle_crop: np.ndarray) -> VehicleAttributes:
        if vehicle_crop is None or vehicle_crop.size == 0:
            return VehicleAttributes()

        import cv2

        crop = self._center_crop(vehicle_crop)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        mean_s = float(np.mean(s))
        mean_v = float(np.mean(v))

        if mean_v <= self.black_value_threshold:
            return VehicleAttributes(color="black", color_confidence=self._achromatic_confidence(v, mean_v, "dark"))
        if mean_s <= self.low_saturation_threshold:
            if mean_v >= self.white_value_threshold:
                return VehicleAttributes(color="white", color_confidence=self._achromatic_confidence(v, mean_v, "light"))
            return VehicleAttributes(color="gray", color_confidence=1.0 - (mean_s / self.low_saturation_threshold))

        color_name, confidence = self._dominant_hue_bucket(h, s)
        return VehicleAttributes(color=color_name, color_confidence=confidence)

    def _center_crop(self, image: np.ndarray) -> np.ndarray:
        """Crop toward the image center to reduce background contamination
        (road surface, sky) skewing the dominant-color estimate."""
        height, width = image.shape[:2]
        crop_h = int(height * self.center_crop_fraction)
        crop_w = int(width * self.center_crop_fraction)
        top = (height - crop_h) // 2
        left = (width - crop_w) // 2
        return image[top : top + crop_h, left : left + crop_w]

    @staticmethod
    def _achromatic_confidence(value_channel: np.ndarray, mean_v: float, direction: str) -> float:
        """Rough confidence proxy: how tightly the value channel clusters
        near the extreme that triggered black/white classification."""
        spread = float(np.std(value_channel)) + 1e-6
        return float(np.clip(1.0 - spread / 128.0, 0.0, 1.0))

    def _dominant_hue_bucket(self, hue_channel: np.ndarray, sat_channel: np.ndarray) -> Tuple[str, float]:
        # Weight the hue histogram by saturation so washed-out pixels
        # (near-gray, still technically some hue) don't dominate the vote.
        bucket_weights: Dict[str, float] = {}
        flat_hue = hue_channel.flatten()
        flat_sat = sat_channel.flatten().astype(np.float64)

        for name, (lo, hi) in _HUE_COLOR_BUCKETS:
            mask = (flat_hue >= lo) & (flat_hue <= hi)
            bucket_weights[name] = bucket_weights.get(name, 0.0) + float(flat_sat[mask].sum())

        total_weight = sum(bucket_weights.values())
        if total_weight <= 0:
            return "unknown", 0.0

        best_color = max(bucket_weights, key=bucket_weights.get)
        confidence = bucket_weights[best_color] / total_weight
        return best_color, float(confidence)


@register("classifier", "timm_backbone")
class TimmBackboneClassifier(BaseClassifier):
    """Deep-learning vehicle type + brand classification via a shared
    `timm` backbone with two linear heads.

    NOT runtime-verified in this environment -- see module docstring.
    Weights are expected to be a checkpoint produced by training this
    same architecture (backbone + `type_head` + `brand_head`); there is
    intentionally no `color_head` here (see `HistogramColorClassifier`
    and `CompositeVehicleClassifier` for why color is handled
    separately).
    """

    def __init__(
        self,
        weights_path: str,
        backbone_name: str,
        type_labels: List[str],
        brand_labels: List[str],
        image_size: int = 224,
        device: str = "cpu",
        min_confidence: float = 0.3,
    ) -> None:
        self.type_labels = type_labels
        self.brand_labels = brand_labels
        self.image_size = image_size
        self.device = device
        self.min_confidence = min_confidence

        try:
            import timm
            import torch
            from torch import nn
        except ImportError as exc:
            raise ModelLoadError(
                "torch and timm are required for TimmBackboneClassifier. "
                "Install via `pip install torch timm`."
            ) from exc

        self._torch = torch
        backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        feature_dim = backbone.num_features

        self._model = nn.ModuleDict(
            {
                "backbone": backbone,
                "type_head": nn.Linear(feature_dim, len(type_labels)),
                "brand_head": nn.Linear(feature_dim, len(brand_labels)),
            }
        ).to(device)

        checkpoint = torch.load(weights_path, map_location=device)
        self._model.load_state_dict(checkpoint)
        self._model.eval()

    def classify(self, vehicle_crop: np.ndarray) -> VehicleAttributes:
        if vehicle_crop is None or vehicle_crop.size == 0:
            return VehicleAttributes()

        torch = self._torch
        tensor = self._preprocess(vehicle_crop)

        with torch.no_grad():
            features = self._model["backbone"](tensor)
            type_logits = self._model["type_head"](features)
            brand_logits = self._model["brand_head"](features)

        type_name, type_conf = self._top1(type_logits, self.type_labels)
        brand_name, brand_conf = self._top1(brand_logits, self.brand_labels)

        return VehicleAttributes(
            vehicle_type=type_name if type_conf >= self.min_confidence else None,
            vehicle_type_confidence=type_conf,
            brand=brand_name if brand_conf >= self.min_confidence else None,
            brand_confidence=brand_conf,
        )

    def _preprocess(self, image: np.ndarray):
        import cv2

        torch = self._torch
        resized = cv2.resize(image, (self.image_size, self.image_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (rgb - mean) / std
        chw = normalized.transpose(2, 0, 1)
        return torch.from_numpy(chw).unsqueeze(0).to(self.device)

    def _top1(self, logits, labels: List[str]) -> Tuple[str, float]:
        probs = logits.softmax(dim=1)[0]
        idx = int(probs.argmax())
        return labels[idx], float(probs[idx])


class CompositeVehicleClassifier(BaseClassifier):
    """Runs multiple injected `BaseClassifier`s over the same crop and
    merges their non-`None` fields into a single `VehicleAttributes`.

    Later classifiers in `classifiers` only fill in fields the earlier
    ones left `None` -- so ordering expresses priority (e.g. put a
    cheap/fast classifier first, an expensive fallback second) without
    needing any conditional logic in this class itself.
    """

    def __init__(self, classifiers: List[BaseClassifier]) -> None:
        if not classifiers:
            raise ValueError("CompositeVehicleClassifier requires at least one classifier.")
        self._classifiers = classifiers

    def classify(self, vehicle_crop: np.ndarray) -> VehicleAttributes:
        merged: Dict[str, Optional[object]] = {}
        for classifier in self._classifiers:
            attrs = classifier.classify(vehicle_crop)
            for field_name in (
                "vehicle_type", "vehicle_type_confidence",
                "color", "color_confidence",
                "brand", "brand_confidence",
            ):
                if merged.get(field_name) is None:
                    value = getattr(attrs, field_name)
                    if value is not None:
                        merged[field_name] = value
        return VehicleAttributes(**merged)
