"""
dashboard/api_client.py

A thin, typed HTTP client for the FastAPI service (`src/api/main.py`),
used by the Streamlit dashboard (`dashboard/app.py`).

This is deliberately the ONLY file in `dashboard/` that talks to the
backend, and it deliberately does NOT import anything from `src/` --
the dashboard is a separate deployable (its own container, its own
`requirements-dashboard.txt` in a real deployment) that only ever speaks
HTTP/JSON to the API, exactly like any other API consumer would. That's
what makes `src.core.base_model.BaseDetector`/`Trainable`/`Exportable`
irrelevant here: from the dashboard's point of view, "the model" is just
whatever `POST /detect` returns.

The `requests.Session` is injected (constructor argument, defaulting to a
fresh one) specifically so `tests/test_api_client.py` can substitute a
mocked session and verify every method's request shape and error
handling without a real server running -- same DI principle applied
throughout the rest of this project, extended to the one component that
lives outside `src/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


class ApiClientError(Exception):
    """Raised for any failed call to the API -- connection failure, a
    non-2xx response, or a malformed JSON body. Kept as a single
    exception type (rather than a hierarchy) since the dashboard only
    ever needs to catch-and-display an error message, never branch on
    failure kind."""


@dataclass(frozen=True)
class DetectionResult:
    bbox: Dict[str, float]
    confidence: float
    class_id: int
    class_name: str


@dataclass(frozen=True)
class DetectResponse:
    detections: List[DetectionResult]
    inference_time_ms: float


@dataclass(frozen=True)
class StreamStatus:
    stream_id: str
    source: str
    current_fps: float
    traffic_counts: Dict[str, int]


class VehicleCvApiClient:
    """Wraps every endpoint in `src/api/main.py`. See that file's
    docstring for the exact request/response shapes this mirrors."""

    def __init__(self, base_url: str, timeout_s: float = 10.0, session: Optional[requests.Session] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._session = session or requests.Session()

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    def detect(
        self,
        image_bytes: bytes,
        filename: str,
        content_type: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> DetectResponse:
        data = self._request(
            "POST",
            "/detect",
            params={"conf_threshold": conf_threshold, "iou_threshold": iou_threshold},
            files={"file": (filename, image_bytes, content_type)},
        )
        return DetectResponse(
            detections=[DetectionResult(**d) for d in data["detections"]],
            inference_time_ms=data["inference_time_ms"],
        )

    def start_stream(
        self,
        stream_id: str,
        source: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        tracker_backend: str = "bytetrack",
        counting_line_y: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload = {
            "stream_id": stream_id,
            "source": source,
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "tracker_backend": tracker_backend,
            "counting_line_y": counting_line_y,
        }
        return self._request("POST", "/streams", json=payload)

    def stop_stream(self, stream_id: str) -> None:
        self._request("DELETE", f"/streams/{stream_id}", expect_json=False)

    def list_streams(self) -> List[StreamStatus]:
        data = self._request("GET", "/streams")
        return [StreamStatus(**item) for item in data]

    def get_latest(self, stream_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/streams/{stream_id}/latest")

    def mjpeg_url(self, stream_id: str) -> str:
        """No HTTP call -- just builds the URL the dashboard embeds
        directly in an `<img>` tag (browsers natively render an MJPEG
        multipart stream inside `<img src="...">`; no JS/polling needed)."""
        return f"{self.base_url}/streams/{stream_id}/mjpeg"

    def _request(
        self,
        method: str,
        path: str,
        expect_json: bool = True,
        **kwargs: Any,
    ) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = self._session.request(method, url, timeout=self.timeout_s, **kwargs)
        except requests.exceptions.RequestException as exc:
            raise ApiClientError(f"Could not reach {url}: {exc}") from exc

        if not response.ok:
            detail = _extract_error_detail(response)
            raise ApiClientError(f"{method} {path} failed ({response.status_code}): {detail}")

        if not expect_json:
            return None

        try:
            return response.json()
        except ValueError as exc:
            raise ApiClientError(f"{method} {path} returned a non-JSON response: {exc}") from exc


def _extract_error_detail(response: requests.Response) -> str:
    try:
        body = response.json()
        return body.get("detail", response.text)
    except ValueError:
        return response.text
