import streamlit as st
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import UPLOADED_VIDEOS_DIR, OUTPUT_SUMMARY_DIR
from src.utils import load_json, get_active_video_stem
from src.pipeline import run_pipeline

st.set_page_config(
    page_title="DASEM - Intelligent Traffic Monitoring & Safety Suite",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Dark Theme CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    .metric-card {
        background-color: #1E293B;
        padding: 18px 22px;
        border-radius: 10px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 12px;
    }
    .status-badge-critical {
        background-color: #7F1D1D;
        color: #FCA5A5;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.88rem;
    }
    .status-badge-warning {
        background-color: #7C2D12;
        color: #FDBA74;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.88rem;
    }
    .status-badge-normal {
        background-color: #064E3B;
        color: #6EE7B7;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.88rem;
    }
    .status-badge-info {
        background-color: #1E3A8A;
        color: #93C5FD;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.88rem;
    }
    div.stButton > button {
        background-color: #2563EB;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("DASEM: Dynamic Accident Severity Estimation System")
st.caption("AI-Powered Road Traffic Detection, Persistent Tracking, Severity Classification & Emergency Support Suite")

st.sidebar.header("Video Input Selection")
uploaded_file = st.sidebar.file_uploader("Upload Traffic Video (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    UPLOADED_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = UPLOADED_VIDEOS_DIR / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    stem = save_path.stem
    if st.session_state.get("active_video_stem") != stem:
        st.session_state["uploaded_video_path"] = str(save_path)
        st.session_state["selected_video"] = str(save_path)
        st.session_state["active_video_stem"] = stem
        st.session_state["video_stem"] = stem
        st.session_state.pop("summary_res", None)
    st.sidebar.success(f"Uploaded: {uploaded_file.name}")

active_stem = get_active_video_stem(st.session_state)
uploaded_video_path = st.session_state.get("uploaded_video_path")

if not active_stem or not uploaded_video_path or not Path(uploaded_video_path).exists():
    st.info("No video uploaded. Please upload a video to begin analysis.")
else:
    video_path = Path(uploaded_video_path)
    stem = active_stem

    st.subheader(f"Selected Source: {video_path.name}")

    col1, col2 = st.columns([1.5, 1])

    with col1:
        if video_path.exists():
            st.video(str(video_path))
        else:
            st.warning("Selected video file not found.")

    with col2:
        st.markdown("### Pipeline Control")
        if st.button("Process Video Pipeline", type="primary", use_container_width=True):
            with st.spinner("Executing DASEM AI Pipeline (YOLOv8 Detection, ByteTrack, Accident & Severity Analysis)..."):
                start_t = time.time()
                summary_res = run_pipeline(str(video_path))
                elapsed = round(time.time() - start_t, 2)
                st.success(f"Pipeline Execution Complete in {elapsed} seconds!")
                st.session_state["summary_res"] = summary_res

        summary_file = OUTPUT_SUMMARY_DIR / f"{stem}_summary.json"
        if summary_file.exists():
            summary_data = load_json(summary_file)
            st.markdown("### Executive Overview")
            
            det_sum = summary_data.get("detection_summary", {})
            acc_sum = summary_data.get("accident_summary", {})
            sev_sum = summary_data.get("severity_summary", {})
            emg_sum = summary_data.get("emergency_summary", {})

            m1, m2 = st.columns(2)
            m1.metric("Unique Vehicles", det_sum.get("total_unique_objects", 0))
            m2.metric("Total Accidents", acc_sum.get("total_accidents", 0))

            m3, m4 = st.columns(2)
            m3.metric("Overall Severity", sev_sum.get("overall_severity", "None"))
            m4.metric("Emergency Priority", emg_sum.get("response_priority", "N/A"))

            with st.expander("Technical Details", expanded=False):
                st.json(summary_data)
        else:
            st.info(f"Video uploaded: '{video_path.name}'. Click 'Process Video Pipeline' to begin AI analysis.")

st.markdown("---")
st.caption("DASEM System Specification | Powered by Ultralytics YOLOv8 & PyTorch")
