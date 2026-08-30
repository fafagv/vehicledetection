"""
tests/test_api_client.py

`dashboard/api_client.py` exercised against a mocked `requests.Session`
(injected via the constructor) -- no real server needed, and no
dependency on anything under `src/`, matching the dashboard's intended
isolation from the ML backend's Python package.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

requests = pytest.importorskip("requests")

from dashboard.api_client import ApiClientError, DetectResponse, StreamStatus, VehicleCvApiClient


def _mock_session(status_code=200, json_body=None, ok=True, text=""):
    session = MagicMock(spec=requests.Session)
    response = MagicMock()
    response.ok = ok
    response.status_code = status_code
    response.json.return_value = json_body
    response.text = text
    session.request.return_value = response
    return session, response


class TestVehicleCvApiClient:
    def test_health(self):
        session, _ = _mock_session(json_body={"status": "ok", "detector_loaded": True, "active_streams": 2})
        client = VehicleCvApiClient("http://localhost:8000", session=session)

        result = client.health()

        assert result == {"status": "ok", "detector_loaded": True, "active_streams": 2}
        assert session.request.call_args[0] == ("GET", "http://localhost:8000/health")

    def test_detect_builds_multipart_request_and_parses_response(self):
        session, _ = _mock_session(
            json_body={
                "detections": [
                    {"bbox": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}, "confidence": 0.9, "class_id": 0, "class_name": "car"}
                ],
                "inference_time_ms": 12.5,
            }
        )
        client = VehicleCvApiClient("http://localhost:8000", session=session)

        result = client.detect(b"fakebytes", "test.jpg", "image/jpeg", conf_threshold=0.4, iou_threshold=0.5)

        assert isinstance(result, DetectResponse)
        assert len(result.detections) == 1
        assert result.detections[0].class_name == "car"
        assert result.inference_time_ms == 12.5

        method, url = session.request.call_args[0]
        kwargs = session.request.call_args[1]
        assert (method, url) == ("POST", "http://localhost:8000/detect")
        assert kwargs["params"] == {"conf_threshold": 0.4, "iou_threshold": 0.5}
        assert kwargs["files"]["file"] == ("test.jpg", b"fakebytes", "image/jpeg")

    def test_start_stream_sends_correct_payload(self):
        session, _ = _mock_session(json_body={"stream_id": "cam-01", "status": "started"})
        client = VehicleCvApiClient("http://localhost:8000", session=session)

        result = client.start_stream("cam-01", "rtsp://x", conf_threshold=0.3, tracker_backend="botsort")

        assert result == {"stream_id": "cam-01", "status": "started"}
        payload = session.request.call_args[1]["json"]
        assert payload["stream_id"] == "cam-01"
        assert payload["tracker_backend"] == "botsort"

    def test_stop_stream_does_not_parse_json(self):
        session, _ = _mock_session(json_body=None, ok=True)
        client = VehicleCvApiClient("http://localhost:8000", session=session)

        result = client.stop_stream("cam-01")

        assert result is None
        method, url = session.request.call_args[0]
        assert (method, url) == ("DELETE", "http://localhost:8000/streams/cam-01")

    def test_list_streams_parses_into_dataclasses(self):
        session, _ = _mock_session(
            json_body=[{"stream_id": "cam-01", "source": "rtsp://a", "current_fps": 24.5, "traffic_counts": {"car": 10}}]
        )
        client = VehicleCvApiClient("http://localhost:8000", session=session)

        streams = client.list_streams()

        assert len(streams) == 1
        assert isinstance(streams[0], StreamStatus)
        assert streams[0].current_fps == 24.5

    def test_mjpeg_url_strips_trailing_slash(self):
        client = VehicleCvApiClient("http://localhost:8000/")
        assert client.mjpeg_url("cam-01") == "http://localhost:8000/streams/cam-01/mjpeg"

    def test_non_2xx_response_raises_with_detail(self):
        session, _ = _mock_session(status_code=404, ok=False, json_body={"detail": "No active stream"})
        client = VehicleCvApiClient("http://localhost:8000", session=session)

        with pytest.raises(ApiClientError, match="No active stream"):
            client.get_latest("nonexistent")

    def test_connection_failure_raises_api_client_error(self):
        session = MagicMock(spec=requests.Session)
        session.request.side_effect = requests.exceptions.ConnectionError("refused")
        client = VehicleCvApiClient("http://localhost:8000", session=session)

        with pytest.raises(ApiClientError, match="Could not reach"):
            client.health()

    def test_non_json_2xx_response_raises_api_client_error(self):
        session, response = _mock_session(ok=True)
        response.json.side_effect = ValueError("not json")
        client = VehicleCvApiClient("http://localhost:8000", session=session)

        with pytest.raises(ApiClientError, match="non-JSON"):
            client.health()
