# Vehicle CV Platform

An enterprise-grade refactor of a legacy single-script YOLOv5 vehicle
detector into a modular, multi-task Computer Vision platform: detection
(YOLOv8/v11) + multi-object tracking (ByteTrack/BoT-SORT) + ALPR +
vehicle attribute classification, with MLOps tracking, edge export, and
an async FastAPI + Streamlit serving layer.

## Status of the 5 pillars

| Pillar | Status |
|---|---|
| 1. Performance & Architecture (YOLOv8/v11, tracking, ALPR, classification) | Detector adapter, ByteTrack/BoT-SORT, ALPR (`src/models/plate_ocr.py`), and vehicle attribute classification (`src/models/vehicle_classifier.py`) all **implemented**; the dependency-light pieces are **unit-verified** (see below). |
| 2. SOLID + Hydra package structure | **Implemented**: `core/models/data/pipelines/tracking/api` exactly as specified, extended with the Clean Architecture use-case/repository layer for `detect.py`/`train.py`/`val.py`/`export.py`. |
| 3. MLOps (MLflow/W&B) + DVC | Logger configs (`configs/logger/{mlflow,wandb}.yaml`) and the full `dvc.yaml` pipeline (`prepare_data -> train -> validate -> export`, with `params.yaml`) **implemented**. |
| 4. Optimization & edge deployment (ONNX/TensorRT/OpenVINO, async streaming) | Multi-threaded RTSP pipeline and the export/benchmark path (`export.py` + `ExportUseCase`) **implemented and unit-verified** (orchestration logic; real ONNX Runtime benchmarking is reviewed, not executed here -- see below). |
| 5. Serving & UI (FastAPI, Streamlit) | FastAPI service **implemented**. Streamlit dashboard (`dashboard/app.py`) **implemented** -- image detection, RTSP stream lifecycle + live MJPEG preview, real-time traffic statistics. Its HTTP client (`dashboard/api_client.py`) is **unit-verified** (see below). |

## Clean Architecture refactor: `detect.py` / `train.py` / `val.py` / `export.py`

All four legacy scripts are now thin **composition roots** over the same
layered architecture:

```
detect.py / train.py / val.py / export.py   <- composition roots: parse config, wire objects, nothing else
        |
        v
src/pipelines/use_cases.py   <- application layer: Detect/Train/Evaluate/ExportUseCase
        |                        (depends ONLY on src/core interfaces)
        v
src/core/base_model.py, repositories.py   <- interfaces (BaseDetector, Trainable, Exportable,
        |                                     WeightsRepository, DatasetRepository, RunRepository,
        |                                     DetectionResultsRepository, ReportRepository)
        v
src/models/yolo_adapter.py, src/data/repositories.py   <- concrete adapters
                                                            (YoloAdapter, LocalWeightsRepository, ...)
```

- **Dependency Injection** throughout: every use case takes its
  detector/trainable/exportable and repositories as constructor
  arguments. `detect.py`/`train.py`/`val.py`/`export.py`'s
  `build_*_use_case` functions are the only places that know concrete
  class names.
- **Repository Pattern**: `WeightsRepository`, `DatasetRepository`,
  `RunRepository`, `DetectionResultsRepository`, `ReportRepository`
  (`src/core/repositories.py`) abstract away *where* weights, datasets,
  and results/reports live. `src/data/repositories.py` provides local-
  filesystem implementations, including the legacy `runs/exp`, `exp2`,
  `exp3`, ... incrementing convention.
- **Config, deliberately different tools per script, each justified in
  its own docstring**:
  - `detect.py`, `export.py` -> **Pydantic Settings** (`configs/model.yaml`,
    `configs/export.yaml` + `VEHICLE_CV_{DETECT,EXPORT}_*` env overrides)
    -- single deterministic actions, no sweep need.
  - `train.py`, `val.py` -> **Hydra** (`configs/train.yaml`,
    `configs/val.yaml`, supports `python val.py --multirun
    conf_threshold=0.1,0.25,0.5`) -- both genuinely benefit from
    override/sweep ergonomics (hyperparameter sweeps; threshold sweeps
    for a PR curve).
- **`Trainable`/`Exportable` are `typing.Protocol`s, not ABCs**:
  training and export are capabilities only some detectors have
  (`YoloAdapter` does, via `native_train`/`export`, both delegating to
  Ultralytics' own native implementations rather than hand-rolling YOLO
  training or ONNX/TensorRT conversion -- see those methods' docstrings
  for why that's the correct call, not a shortcut).

## ALPR & vehicle attribute classification

`src/models/plate_ocr.py` splits ALPR into two narrow, independently
swappable strategies composed via DI into `AlprPipeline` (which
implements `BaseOCR`):
- `PlateLocator` -- `HaarCascadePlateLocator` (OpenCV's bundled cascade;
  zero extra dependencies, real out-of-the-box default, but the weakest
  link -- see its docstring for the honest accuracy caveat and the
  drop-in YOLO-based upgrade path).
- `TextReader` -- `EasyOcrTextReader` (wraps `easyocr`; not
  runtime-verified here since `easyocr` transitively needs `torch`).

`src/models/vehicle_classifier.py` similarly composes rather than
inherits:
- `HistogramColorClassifier` -- real, dependency-light (HSV histogram
  dominant-color estimation), verified against actual known colors.
- `TimmBackboneClassifier` -- deep-learning type/brand classification
  (not runtime-verified; torch/timm unavailable here).
- `CompositeVehicleClassifier` -- DI-merges multiple classifiers'
  outputs (e.g. cheap classical color + deep-learning type/brand) into
  one `VehicleAttributes`, priority-ordered by injection order.

## DVC pipeline

`dvc.yaml` wires `prepare_data -> train -> validate -> export` directly
to `train.py`/`val.py`/`export.py` -- there is exactly one code path,
not a separate "DVC version" and "manual version" of each step.
`params.yaml` holds the subset of config DVC tracks for cache
invalidation and `dvc exp run --set-param` sweeps.
`scripts/prepare_dataset.py` (the `prepare_data` stage) splits a flat
raw labeled dataset into train/val and writes an Ultralytics-format
`data.yaml` -- dependency-free (stdlib only) and unit-verified for real,
including split determinism.

## What's been verified, not just written

This sandbox has no network access, so `torch`/`ultralytics`/`fastapi`/
`hydra`/`pydantic`/`onnx` couldn't be installed to run everything
end-to-end. Every file byte-compiles cleanly, but compiling only proves
syntax. Where a module's logic was dependency-light enough to actually
execute here (opencv/numpy/scipy/stdlib are all available), I did --
these aren't just written tests, they were actually run and passed:

- **`src/core/*`**: every dataclass invariant, registry
  conflict/lookup/copy-safety behavior.
- **`src/tracking/bytetrack_tracker.py`**: real two-stage matching +
  Kalman filter on a synthetic 10-frame sequence; velocity estimate
  converged to exactly 5.00 px/frame against known ground truth.
- **`src/tracking/speed_estimator.py`**: px/frame -> km/h arithmetic
  verified against a hand-computed value.
- **`src/pipelines/use_cases.py` (Detect/Train)**: 7 scenarios via real
  `Local*Repository`s + fakes -- image-directory processing, incrementing
  run dirs, opt-out persistence, video frame-capping, error handling,
  full training-path hyperparameter forwarding.
- **`src/evaluation/detection_metrics.py`**: the self-contained mAP
  calculator against 3 hand-computed scenarios (AP=0.5, AP=1.0, AP=0.0),
  matching the derived math exactly.
- **`src/pipelines/use_cases.py` (Evaluate/Export)**: `EvaluateUseCase`
  end-to-end against real image files + real YOLO-format label files on
  disk, reproducing the same AP=0.5 result independently; `ExportUseCase`
  fully exercised via a fake exportable + injected fake benchmark
  function, including the "only benchmark onnx" branch.
- **`src/models/plate_ocr.py`**: the real bundled Haar cascade loads and
  runs (0 detections on random noise, as expected); `AlprPipeline`'s DI
  composition and confidence filtering verified against fakes.
- **`src/models/vehicle_classifier.py`**: `HistogramColorClassifier`
  against 7 known ground-truth colors (pure red/green/blue/yellow/white/
  black/gray) -- all correct; `CompositeVehicleClassifier`'s
  priority-ordered merge verified against fakes.
- **`scripts/prepare_dataset.py`**: full split/materialize/write-yaml
  flow against a synthetic raw dataset, including determinism under a
  fixed seed and correct skipping of unlabeled images.
- **`dashboard/api_client.py`**: 9 scenarios against a mocked
  `requests.Session` (request shape, payload correctness, error
  propagation for both connection failures and non-2xx responses), PLUS
  a from-scratch end-to-end integration test against a real
  `http.server` instance on a real socket -- no mocks at all in that
  second pass -- covering the health check, stream listing, a real
  multipart image upload, stream start/stop (including 204 No Content
  handling), and 404 error propagation. Both passes gave identical
  results, which is exactly the point of injecting `requests.Session`
  rather than hardcoding it: the mocked tests are trustworthy because
  they've been cross-checked against a real transport.

**Not runtime-verified** (reviewed against documented APIs only):
`YoloAdapter` (Ultralytics/Lightning), the FastAPI service itself, the
RTSP grabber, `EasyOcrTextReader`, `TimmBackboneClassifier`,
`dashboard/app.py` (imports `streamlit`, unavailable here -- only its
extracted `api_client.py` logic is verified), and the real ONNX Runtime
benchmarking path inside `benchmark_onnx_model` (the `ExportUseCase`
*orchestration* around it is verified; the function itself needs a real
onnxruntime session against a real exported model).

## Project layout

See the full tree in the original response. Implemented so far:

```
src/core/           base_model.py, registry.py, types.py, exceptions.py, repositories.py   ✅ implemented + tested
src/models/         yolo_adapter.py                                        ✅ implemented (untested)
                    plate_ocr.py                                           ✅ implemented (Haar locator tested; EasyOCR untested)
                    vehicle_classifier.py                                  ✅ implemented (color classifier tested; timm untested)
src/tracking/       base_tracker.py, bytetrack_tracker.py                  ✅ implemented + tested
                    botsort_tracker.py                                     ✅ implemented (Re-ID branch is an extension point)
                    speed_estimator.py                                     ✅ implemented + tested
src/pipelines/      stream_tracker.py                                      ✅ implemented (untested)
                    use_cases.py (Detect/Train/Evaluate/Export)            ✅ implemented + tested
                    onnx_benchmark.py                                      ✅ implemented (orchestration tested; real onnxruntime path untested)
src/evaluation/     detection_metrics.py                                   ✅ implemented + tested
src/data/           repositories.py, yolo_label_reader.py                  ✅ implemented + tested
                    datasets.py, transforms.py (Lightning DataModule path) ⏳ next step
src/api/            main.py, schemas.py, dependencies.py                   ✅ implemented (untested)
detect.py / train.py / val.py / export.py                                  ✅ implemented (use-case layer tested; composition roots untested)
scripts/prepare_dataset.py                                                 ✅ implemented + tested
dvc.yaml, params.yaml                                                       ✅ implemented
dashboard/app.py                                                           ✅ implemented (untested -- imports streamlit)
dashboard/api_client.py                                                    ✅ implemented + tested
```

## Quickstart

```bash
pip install -r requirements.txt

# Dependency-free tests (run right now, no torch needed)
pytest tests/ -v -k "not stream_tracker and not api"

# Detect (Pydantic Settings config)
python detect.py --config configs/model.yaml --source path/to/video.mp4

# Train (Hydra config, supports CLI overrides and --multirun sweeps)
python train.py epochs=50 device=cuda:0

# Validate (Hydra config, supports threshold sweeps)
python val.py weights=runs/train/train/weights/best.pt
python val.py --multirun conf_threshold=0.1,0.25,0.5,0.75

# Export + benchmark (Pydantic Settings config)
python export.py --weights runs/train/train/weights/best.pt --format onnx

# Full DVC pipeline (prepare_data -> train -> validate -> export)
dvc init && dvc repro

# Serve
export VEHICLE_CV_DETECTOR_WEIGHTS=runs/train/train/weights/best.pt
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Start tracking an RTSP feed
curl -X POST localhost:8000/streams -H 'Content-Type: application/json' \
  -d '{"stream_id": "cam-01", "source": "rtsp://192.168.1.100:554/stream1", "tracker_backend": "bytetrack"}'

# Watch it live
open http://localhost:8000/streams/cam-01/mjpeg

# Or use the dashboard instead of curl/raw MJPEG URLs
VEHICLE_CV_API_URL=http://localhost:8000 streamlit run dashboard/app.py
```

## Architecture notes worth knowing before extending this

- **Why `YoloAdapter` isn't trained purely via Lightning by default**:
  Ultralytics' own trainer reimplements mosaic scheduling, EMA, and
  per-scale loss balancing that would be substantial, error-prone work to
  reproduce faithfully inside a hand-rolled `training_step`. The Lightning
  path is provided (and uses the *same* underlying loss function) for
  teams that need YOLO training to participate in a larger, uniformly
  orchestrated multi-task `pl.Trainer` run; `native_train()` is the
  recommended default otherwise. See the docstring in
  `src/models/yolo_adapter.py` for the full reasoning.
- **Why ByteTrack is hand-implemented rather than wrapping
  `ultralytics.trackers.BYTETracker`**: that class is an internal
  implementation detail with an API that has changed across Ultralytics
  releases and isn't published as stable. `src/tracking/bytetrack_tracker.py`
  implements the same published algorithm against this platform's own
  stable `Detection`/`Track` types instead.
- **Why the API is env-var configured while training is Hydra
  configured**: the API is a long-running deployed service (env vars /
  `.env` is the conventional 12-factor surface for that); training is a
  research CLI invocation, where Hydra's override/sweep ergonomics matter
  more. `src/api/dependencies.py::Settings` and `configs/config.yaml` are
  intentionally two separate configuration surfaces, not a Hydra config
  reused for both.
- **MJPEG endpoint correctness**: `StreamTracker` retains the last raw
  frame (`get_last_frame()`, thread-safe) specifically so
  `/streams/{id}/mjpeg` can draw real overlays on the real frame rather
  than a placeholder canvas.
