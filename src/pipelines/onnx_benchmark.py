"""
src/pipelines/onnx_benchmark.py

Standalone ONNX Runtime latency/throughput benchmark, injected into
`ExportUseCase` (`src/pipelines/use_cases.py`) as its default
`benchmark_fn`. Kept as a plain function (not a class) since it has no
state to encapsulate -- and kept in its own module specifically so it's
trivially swappable/fakeable in tests without needing a real onnxruntime
session or a real exported model file.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def benchmark_onnx_model(
    onnx_path: str, image_size: int, num_warmup_runs: int, num_benchmark_runs: int
) -> Dict[str, Any]:
    """Run `num_benchmark_runs` inference calls (after `num_warmup_runs`
    untimed warmup calls) against `onnx_path` with a random
    `[1, 3, image_size, image_size]` input, returning latency percentiles
    and throughput.
    """
    import onnxruntime as ort

    providers = ort.get_available_providers()
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, image_size, image_size).astype(np.float32)

    for _ in range(num_warmup_runs):
        session.run(None, {input_name: dummy_input})

    latencies_ms: List[float] = []
    for _ in range(num_benchmark_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        latencies_ms.append((time.perf_counter() - start) * 1000)

    latencies_arr = np.array(latencies_ms)
    result = {
        "mean_latency_ms": float(latencies_arr.mean()),
        "p50_latency_ms": float(np.percentile(latencies_arr, 50)),
        "p95_latency_ms": float(np.percentile(latencies_arr, 95)),
        "p99_latency_ms": float(np.percentile(latencies_arr, 99)),
        "throughput_fps": float(1000.0 / latencies_arr.mean()),
        "providers": providers,
    }
    logger.info(
        "ONNX benchmark: mean=%.2fms p95=%.2fms throughput=%.1f fps (providers=%s)",
        result["mean_latency_ms"],
        result["p95_latency_ms"],
        result["throughput_fps"],
        providers,
    )
    return result
