"""
src/models/yolo_adapter.py

`YoloAdapter` bridges Ultralytics YOLOv8/v11 into two roles at once:

  1. `BaseDetector` (src.core.base_model) -- the inference-time contract
     consumed by `src/pipelines/stream_tracker.py` and `src/api/main.py`.
     This path calls Ultralytics' own, highly-optimized `model.predict(...)`
     directly and is what production inference should use.

  2. `pytorch_lightning.LightningModule` -- so this task can be trained
     under the same `pl.Trainer` (DDP, AMP, MLflow/W&B logging, checkpoint
     callbacks) as the other task modules in this platform (ALPR,
     classification), giving one uniform training entry point
     (`src/pipelines/train.py`) across the whole system.

HONEST CAVEAT on (2): Ultralytics' own `YOLO.train(...)` trainer
reimplements a large amount of YOLO-specific training machinery
internally -- mosaic/close-mosaic augmentation scheduling, EMA weight
averaging, per-scale loss balancing, warmup bias/LR schedules, and more.
`training_step`/`validation_step` below delegate to the *same* underlying
loss function Ultralytics uses (`DetectionModel.loss`), so gradients and
the loss value are faithful to native YOLO training, but none of the
scheduling extras above are reproduced by this adapter. For a pure
YOLO-only project, prefer calling `YoloAdapter.native_train(...)` (a thin
wrapper around `YOLO.train`) instead. Use the Lightning path when you
specifically need YOLO training to participate in a larger, uniformly
orchestrated multi-task Lightning run.

Batch contract for the Lightning path (matches Ultralytics'
`build_dataloader`/`YOLODataset` collation -- see
`src/data/datasets.py::YoloFormatDetectionDataset`):
    batch = {
        "img": FloatTensor[B, 3, H, W],      # normalized to [0, 1]
        "batch_idx": FloatTensor[N],         # which image each box belongs to
        "cls": FloatTensor[N, 1],            # class id per box
        "bboxes": FloatTensor[N, 4],         # normalized xywh per box
    }
where N is the total number of boxes across the batch (Ultralytics uses a
flat, ragged-friendly representation rather than per-image padding).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from src.core.base_model import BaseDetector
from src.core.exceptions import InferenceError, ModelLoadError
from src.core.types import BoundingBox, Detection

logger = logging.getLogger(__name__)


class YoloAdapter(pl.LightningModule, BaseDetector):
    """Lightning + `BaseDetector` adapter around an Ultralytics YOLO model.

    Config (see configs/model/yolov8_detect.yaml):
        weights (str): a `.pt` checkpoint name/path ("yolov8n.pt" pulls
            COCO-pretrained weights on first use) or a `.yaml` model
            definition for training from scratch.
        num_classes (int): number of target classes for this dataset.
        class_names (List[str]): ordered class names, length == num_classes.
        image_size (int): square input resolution used for train/predict.
        optimizer / scheduler: Hydra `_target_` configs, only consumed by
            the Lightning training path.
    """

    def __init__(self, model_cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(
            {k: v for k, v in model_cfg.items() if k not in ("optimizer", "scheduler")}
        )
        self.cfg = model_cfg
        self._class_names: List[str] = list(model_cfg.class_names)

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise ModelLoadError(
                "ultralytics is required for YoloAdapter. Install via "
                "`pip install ultralytics`."
            ) from exc

        try:
            self._yolo = YOLO(model_cfg.weights)
        except Exception as exc:  # noqa: BLE001 - surface any load failure uniformly
            raise ModelLoadError(f"Failed to load YOLO weights '{model_cfg.weights}': {exc}") from exc

        # The underlying trainable nn.Module (Ultralytics' `DetectionModel`).
        # Registering it as `self.net` makes it a proper Lightning submodule
        # (device placement, DDP wrapping, checkpointing all "just work").
        self.net = self._yolo.model
        self._is_warmed_up = False

    # ------------------------------------------------------------------ #
    # BaseDetector implementation (inference path -- production use).
    # ------------------------------------------------------------------ #
    def predict(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        classes: Optional[Sequence[int]] = None,
    ) -> List[Detection]:
        try:
            results = self._yolo.predict(
                source=frame,
                conf=conf_threshold,
                iou=iou_threshold,
                classes=list(classes) if classes is not None else None,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(f"YOLO predict() failed: {exc}") from exc

        if not results:
            return []
        result = results[0]  # single-frame input -> single Results object
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        cls_ids = boxes.cls.detach().cpu().numpy().astype(int)

        detections: List[Detection] = []
        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, cls_ids):
            class_name = self._class_names[cls_id] if cls_id < len(self._class_names) else str(cls_id)
            detections.append(
                Detection(
                    bbox=BoundingBox(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=class_name,
                )
            )
        return detections

    def warmup(self, image_size: int = 640) -> None:
        if self._is_warmed_up:
            return
        dummy = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        self.predict(dummy)
        self._is_warmed_up = True
        logger.info("YoloAdapter warmed up at image_size=%d.", image_size)

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    # ------------------------------------------------------------------ #
    # Native Ultralytics training (recommended default -- see module
    # docstring for why this differs from the Lightning path below).
    # ------------------------------------------------------------------ #
    def native_train(self, **train_kwargs: Any) -> Any:
        """Thin wrapper around `ultralytics.YOLO.train(...)`, exposing the
        full native trainer (mosaic scheduling, EMA, warmup, etc.) for
        YOLO-only training runs. `train_kwargs` are passed straight
        through -- see Ultralytics' documentation for the full argument
        list (data, epochs, imgsz, batch, device, ...)."""
        return self._yolo.train(**train_kwargs)

    def export(self, format: str, output_path: str, **kwargs: Any) -> str:
        """Thin wrapper around `ultralytics.YOLO.export(...)`, satisfying
        the `Exportable` protocol for `ExportUseCase`
        (`src/pipelines/use_cases.py`).

        Deliberately NOT a hand-rolled `torch.onnx.export` + TensorRT
        `trtexec` pipeline: Ultralytics already implements ONNX,
        OpenVINO, and TensorRT engine export (plus INT8/FP16
        quantization) directly against its own model graph, correctly
        handling YOLO-specific export quirks (e.g. the detection head's
        NMS-free output format) that a generic exporter would get wrong.
        Reinventing that here would be strictly worse engineering than
        delegating to it.

        Args:
            format: one of Ultralytics' supported export formats --
                "onnx", "openvino", "engine" (TensorRT), "torchscript", etc.
            output_path: where the exported artifact should end up.
                Ultralytics itself chooses the exact output filename
                (based on `format` and the source weights name); this
                wrapper moves/renames its result to `output_path` so
                callers (e.g. `ExportUseCase`) get a predictable path.
            **kwargs: forwarded to `YOLO.export` (e.g. `half=True`,
                `dynamic=True`, `imgsz=640`, `int8=True`).
        """
        exported_path = self._yolo.export(format=format, **kwargs)

        import shutil
        from pathlib import Path

        exported_path = Path(exported_path)
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if exported_path.resolve() != destination.resolve():
            shutil.move(str(exported_path), str(destination))
        return str(destination)

    # ------------------------------------------------------------------ #
    # Lightning training path (opt-in, multi-task orchestration use case).
    # ------------------------------------------------------------------ #
    def forward(self, batch: Dict[str, Any]) -> Any:
        return self.net(batch["img"])

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, loss_items = self.net.loss(batch)
        self._log_yolo_loss(loss_items, stage="train", batch_size=batch["img"].shape[0])
        return loss.sum()

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, loss_items = self.net.loss(batch)
        self._log_yolo_loss(loss_items, stage="val", batch_size=batch["img"].shape[0])
        return loss.sum()

    def _log_yolo_loss(self, loss_items: torch.Tensor, stage: str, batch_size: int) -> None:
        # Ultralytics' DetectionModel.loss returns per-component loss items
        # in a fixed order: [box_loss, cls_loss, dfl_loss].
        names = ["box_loss", "cls_loss", "dfl_loss"]
        for name, value in zip(names, loss_items.detach().cpu().tolist()):
            self.log(
                f"{stage}/{name}",
                value,
                on_step=(stage == "train"),
                on_epoch=True,
                prog_bar=(name == "box_loss"),
                batch_size=batch_size,
            )

    def configure_optimizers(self) -> Dict[str, Any]:
        from hydra.utils import instantiate

        if "optimizer" not in self.cfg or self.cfg.optimizer is None:
            raise ValueError(
                "model config is missing an `optimizer` block for Lightning "
                "training (not required for `native_train`)."
            )
        optimizer = instantiate(self.cfg.optimizer, params=self.net.parameters())
        result: Dict[str, Any] = {"optimizer": optimizer}

        scheduler_cfg = self.cfg.get("scheduler", None)
        if scheduler_cfg is not None:
            scheduler = instantiate(scheduler_cfg, optimizer=optimizer)
            result["lr_scheduler"] = {"scheduler": scheduler, "interval": "epoch"}
        return result
