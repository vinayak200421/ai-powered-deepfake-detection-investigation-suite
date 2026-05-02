from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from app.api_client import DEFAULT_ANALYZE_URL, analyze_video_bytes, load_bundled_sample_result

st.title("Upload")

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_video_bytes" not in st.session_state:
    st.session_state.last_video_bytes = None

inference_mode = st.radio(
    "Inference mode",
    (
        "Bundled sample JSON (offline, no GPU)",
        "HTTP API (tunnel to GPU server)",
        "Local CPU pipeline (very slow; needs weights)",
    ),
    index=1,
)

api_url = st.text_input("Analyze URL", value=DEFAULT_ANALYZE_URL)
enable_gradcam = st.checkbox("Generate Grad-CAM Heatmaps (Slower)", value=True)
timeout_s = st.slider("Request timeout (s)", min_value=10, max_value=300, value=120)
max_retries = st.slider("HTTP retries", min_value=1, max_value=6, value=3)

if inference_mode.startswith("Local CPU"):
    st.warning(
        "Local CPU inference can take many minutes per clip (PROJECT_PLAN §13). "
        "Use the HTTP API through an SSH tunnel for demos when possible."
    )

video_file = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"])

if video_file is not None:
    data = video_file.getvalue()
    st.session_state.last_video_bytes = data
    from app.components.video_player import show_uploaded_video

    show_uploaded_video(data, mime=video_file.type or "video/mp4")

col_a, col_b = st.columns(2)
with col_a:
    run_upload = st.button("Analyze uploaded video", type="primary")
with col_b:
    load_sample = st.button("Load bundled sample only (no video)")


def _find_xception_weights() -> Path | None:
    root = _REPO_ROOT / "models"
    for p in root.rglob("full_c23.p"):
        return p
    return None


if load_sample:
    st.session_state.last_result = load_bundled_sample_result()
    st.success("Loaded `app/sample_results/sample_result.json`. Open **Results**.")

if run_upload:
    if inference_mode == "Bundled sample JSON (offline, no GPU)":
        st.session_state.last_result = load_bundled_sample_result()
        st.success("Applied bundled sample (offline). Open **Results**.")
    elif inference_mode == "HTTP API (tunnel to GPU server)":
        raw = st.session_state.last_video_bytes
        if not raw:
            st.warning("Upload a video first for API inference.")
        else:
            with st.spinner("Calling remote API..."):
                try:
                    url_to_call = api_url
                    if enable_gradcam:
                        url_to_call += "&gradcam=true" if "?" in url_to_call else "?gradcam=true"
                        
                    result = analyze_video_bytes(
                        raw,
                        url=url_to_call,
                        timeout_s=timeout_s,
                        max_retries=max_retries,
                    )
                except Exception as e:
                    st.error(f"Request failed: {e}")
                else:
                    st.session_state.last_result = result
                    st.success("Done. Open **Results**.")
    else:
        raw = st.session_state.last_video_bytes
        if not raw:
            st.warning("Upload a video for local CPU inference.")
        else:
            wpath = _find_xception_weights()
            if wpath is None:
                st.error(
                    "Missing `full_c23.p` under `models/`. "
                    "Local CPU mode needs FaceForensics++ weights."
                )
            else:
                try:
                    import cv2  # noqa: F401
                    import torch  # noqa: F401
                except ImportError as e:
                    st.error(f"Local CPU inference requires PyTorch and OpenCV: {e}")
                else:
                    from src.pipeline import Pipeline, PipelineConfig

                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                        tmp.write(raw)
                        tmp.flush()
                        tpath = Path(tmp.name)
                    try:
                        with st.spinner(
                            "Running local CPU pipeline (minutes possible; plan §13)..."
                        ):
                            pipe = Pipeline(
                                device="cpu",
                                cfg=PipelineConfig(xception_weights=wpath),
                            )
                            st.session_state.last_result = pipe.run_on_video(tpath)
                    except Exception as e:
                        st.error(f"Pipeline failed: {e}")
                    else:
                        st.success("Done. Open **Results**.")
                    finally:
                        tpath.unlink(missing_ok=True)
