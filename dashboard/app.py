"""
dashboard/app.py

Interactive Streamlit dashboard for the Vehicle CV Platform: image
upload detection with adjustable confidence/IoU thresholds, RTSP stream
lifecycle management, a live annotated video preview, and real-time
traffic statistics. Talks to `src/api/main.py` exclusively over HTTP via
`dashboard/api_client.py` -- this file contains no ML code and imports
nothing from `src/`.

Run with:
    streamlit run dashboard/app.py

Configure the backend URL either via the sidebar at runtime or the
`VEHICLE_CV_API_URL` environment variable (defaults to
`http://localhost:8000`).
"""

from __future__ import annotations

import os
import time

import streamlit as st

from dashboard.api_client import ApiClientError, VehicleCvApiClient

st.set_page_config(page_title="Vehicle CV Platform", page_icon="🚗", layout="wide")


@st.cache_resource
def get_client(base_url: str) -> VehicleCvApiClient:
    """Cached per `base_url` so switching the API endpoint in the sidebar
    creates a fresh client (and connection pool) rather than reusing a
    stale one."""
    return VehicleCvApiClient(base_url)


def render_sidebar() -> VehicleCvApiClient:
    st.sidebar.title("🚗 Vehicle CV Platform")

    default_url = os.environ.get("VEHICLE_CV_API_URL", "http://localhost:8000")
    base_url = st.sidebar.text_input("API base URL", value=default_url)
    client = get_client(base_url)

    st.sidebar.subheader("Detection thresholds")
    st.session_state.setdefault("conf_threshold", 0.25)
    st.session_state.setdefault("iou_threshold", 0.45)
    st.session_state["conf_threshold"] = st.sidebar.slider(
        "Confidence threshold", 0.0, 1.0, st.session_state["conf_threshold"], 0.05
    )
    st.session_state["iou_threshold"] = st.sidebar.slider(
        "IoU (NMS) threshold", 0.0, 1.0, st.session_state["iou_threshold"], 0.05
    )

    st.sidebar.divider()
    try:
        health = client.health()
        status_icon = "🟢" if health.get("detector_loaded") else "🟡"
        st.sidebar.markdown(
            f"{status_icon} API reachable — detector loaded: **{health.get('detector_loaded')}**, "
            f"active streams: **{health.get('active_streams')}**"
        )
    except ApiClientError as exc:
        st.sidebar.error(f"🔴 API unreachable: {exc}")

    return client


def render_detect_tab(client: VehicleCvApiClient) -> None:
    st.header("Image detection")
    st.caption("Upload an image to run detection with the thresholds set in the sidebar.")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded_file is None:
        return

    col_image, col_results = st.columns([2, 1])
    col_image.image(uploaded_file, caption="Uploaded image", use_container_width=True)

    if col_results.button("Run detection", type="primary"):
        with st.spinner("Running detection..."):
            try:
                result = client.detect(
                    image_bytes=uploaded_file.getvalue(),
                    filename=uploaded_file.name,
                    content_type=uploaded_file.type or "image/jpeg",
                    conf_threshold=st.session_state["conf_threshold"],
                    iou_threshold=st.session_state["iou_threshold"],
                )
            except ApiClientError as exc:
                col_results.error(f"Detection failed: {exc}")
                return

        col_results.metric("Inference time", f"{result.inference_time_ms:.1f} ms")
        col_results.metric("Objects detected", len(result.detections))

        if result.detections:
            table_rows = [
                {
                    "class": d.class_name,
                    "confidence": round(d.confidence, 3),
                    "x1": round(d.bbox["x1"], 1),
                    "y1": round(d.bbox["y1"], 1),
                    "x2": round(d.bbox["x2"], 1),
                    "y2": round(d.bbox["y2"], 1),
                }
                for d in result.detections
            ]
            col_results.dataframe(table_rows, use_container_width=True, hide_index=True)
        else:
            col_results.info("No objects detected above the current confidence threshold.")


def render_streams_tab(client: VehicleCvApiClient) -> None:
    st.header("RTSP / video stream tracking")

    with st.form("start_stream_form"):
        st.subheader("Start a new stream")
        col1, col2, col3 = st.columns(3)
        stream_id = col1.text_input("Stream ID", placeholder="camera-01")
        source = col2.text_input("Source", placeholder="rtsp://192.168.1.100:554/stream1")
        tracker_backend = col3.selectbox("Tracker", ["bytetrack", "botsort"])
        counting_line_y = st.number_input(
            "Counting line (pixel row, optional)", min_value=0, value=0, step=10,
            help="Leave at 0 to disable traffic-density line counting for this stream.",
        )
        submitted = st.form_submit_button("Start tracking", type="primary")

    if submitted:
        if not stream_id or not source:
            st.warning("Both Stream ID and Source are required.")
        else:
            try:
                client.start_stream(
                    stream_id=stream_id,
                    source=source,
                    conf_threshold=st.session_state["conf_threshold"],
                    iou_threshold=st.session_state["iou_threshold"],
                    tracker_backend=tracker_backend,
                    counting_line_y=int(counting_line_y) if counting_line_y > 0 else None,
                )
                st.success(f"Started stream '{stream_id}'.")
            except ApiClientError as exc:
                st.error(f"Could not start stream: {exc}")

    st.divider()
    st.subheader("Active streams")

    try:
        streams = client.list_streams()
    except ApiClientError as exc:
        st.error(f"Could not list streams: {exc}")
        return

    if not streams:
        st.info("No active streams. Start one above.")
        return

    for stream in streams:
        with st.expander(f"📹 {stream.stream_id} — {stream.source}", expanded=True):
            col_preview, col_stats = st.columns([2, 1])

            col_preview.markdown(
                f'<img src="{client.mjpeg_url(stream.stream_id)}" style="width:100%;" />',
                unsafe_allow_html=True,
            )

            col_stats.metric("FPS", f"{stream.current_fps:.1f}")
            if stream.traffic_counts:
                col_stats.write("**Traffic counts**")
                col_stats.dataframe(
                    [{"class": k, "count": v} for k, v in stream.traffic_counts.items()],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                col_stats.caption("No counting line configured for this stream.")

            if col_stats.button("Stop", key=f"stop_{stream.stream_id}"):
                try:
                    client.stop_stream(stream.stream_id)
                    st.success(f"Stopped '{stream.stream_id}'.")
                    st.rerun()
                except ApiClientError as exc:
                    st.error(f"Could not stop stream: {exc}")


def render_statistics_tab(client: VehicleCvApiClient) -> None:
    st.header("Real-time traffic statistics")

    auto_refresh = st.checkbox("Auto-refresh every 3 seconds", value=False)

    try:
        streams = client.list_streams()
    except ApiClientError as exc:
        st.error(f"Could not fetch statistics: {exc}")
        return

    if not streams:
        st.info("No active streams to report statistics for.")
    else:
        total_vehicles = sum(sum(s.traffic_counts.values()) for s in streams)
        col1, col2, col3 = st.columns(3)
        col1.metric("Active streams", len(streams))
        col2.metric("Total vehicles counted", total_vehicles)
        col3.metric("Avg FPS across streams", f"{sum(s.current_fps for s in streams) / len(streams):.1f}")

        st.bar_chart(
            {s.stream_id: sum(s.traffic_counts.values()) for s in streams},
        )

        st.dataframe(
            [
                {
                    "stream_id": s.stream_id,
                    "fps": round(s.current_fps, 1),
                    **s.traffic_counts,
                }
                for s in streams
            ],
            use_container_width=True,
            hide_index=True,
        )

    if auto_refresh:
        time.sleep(3)
        st.rerun()


def main() -> None:
    client = render_sidebar()

    tab_detect, tab_streams, tab_stats = st.tabs(["🖼️ Image Detection", "📹 Live Streams", "📊 Statistics"])
    with tab_detect:
        render_detect_tab(client)
    with tab_streams:
        render_streams_tab(client)
    with tab_stats:
        render_statistics_tab(client)


if __name__ == "__main__":
    main()
