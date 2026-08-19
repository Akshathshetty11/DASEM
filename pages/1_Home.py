import streamlit as st
import shutil
from pathlib import Path
from src.config import PROJECT_ROOT, UPLOADED_VIDEOS_DIR, ensure_directories
from src.utils import get_active_video_stem
from src.pipeline import run_full_pipeline

ensure_directories()

st.set_page_config(page_title="Home - DASEM", page_icon="🏠", layout="wide")

st.title("DASEM System Console")
st.markdown("""
Welcome to the **Dynamic Accident Severity Estimation System**.
Please upload a traffic video file below to begin AI computer vision analysis.
""")

uploaded_file = st.file_uploader("Upload Traffic Video (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"], key="home_uploader")

if uploaded_file is not None:
    UPLOADED_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = UPLOADED_VIDEOS_DIR / uploaded_file.name
    with open(save_path, "wb") as f:
        shutil.copyfileobj(uploaded_file, f)
    
    stem = save_path.stem
    st.session_state["uploaded_video_path"] = str(save_path)
    st.session_state["selected_video"] = str(save_path)
    st.session_state["active_video_stem"] = stem
    st.session_state["video_stem"] = stem
    st.sidebar.success(f"Uploaded: {uploaded_file.name}")

active_stem = get_active_video_stem(st.session_state)
uploaded_video_path = st.session_state.get("uploaded_video_path")

if not active_stem or not uploaded_video_path or not Path(uploaded_video_path).exists():
    st.info("No video uploaded. Please upload a video to begin analysis.")
else:
    selected_path = Path(uploaded_video_path)
    st.video(str(selected_path))

    if st.button("Execute AI Pipeline Analysis", key="btn_home_pipeline", type="primary"):
        with st.spinner("Processing video through AI detection and tracking modules..."):
            try:
                res = run_full_pipeline(selected_path)
                st.session_state["pipeline_result"] = res
                st.session_state["active_video_stem"] = selected_path.stem
                st.session_state["video_stem"] = selected_path.stem
                st.session_state["active_video_name"] = selected_path.name
                st.session_state["selected_video"] = str(selected_path)
                st.success("Analysis complete. Select any sidebar tab to inspect detailed module outputs.")
            except Exception as e:
                st.error(f"Execution error: {e}")
