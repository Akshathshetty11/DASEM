import streamlit as st
from pathlib import Path
from src.config import OUTPUT_DETECTED_DIR, OUTPUT_DETECTIONS_DIR
from src.utils import load_json, get_active_video_stem

st.set_page_config(page_title="Detection & Tracking - DASEM", page_icon="🚘", layout="wide")

st.title("OBJECT DETECTION & TRACKING")
st.caption("YOLOv8 Multi-Class Object Detection & Persistent ByteTrack Multi-Object Tracking")

stem = get_active_video_stem(st.session_state)
uploaded_video_path = st.session_state.get("uploaded_video_path")

if not stem or not uploaded_video_path:
    st.info("No video uploaded. Please upload a video to begin analysis.")
else:
    video_name = Path(uploaded_video_path).name
    json_path = OUTPUT_DETECTIONS_DIR / f"{stem}.json"
    video_path = OUTPUT_DETECTED_DIR / f"{stem}_detected.mp4"

    if json_path.exists():
        data = load_json(json_path)
        total_unique = data.get("total_unique_objects", 0)
        counts = data.get("unique_object_counts", data.get("unique_counts", {}))

        st.subheader("Unique Tracked Objects")
        st.info("Counts represent unique physical objects tracked over the entire video duration (counted once per track ID).")

        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
        col1.metric("Total Unique", total_unique)
        col2.metric("Cars", counts.get("car", 0))
        col3.metric("Motorcycles", counts.get("motorcycle", 0))
        col4.metric("Buses", counts.get("bus", 0))
        col5.metric("Trucks", counts.get("truck", 0))
        col6.metric("Auto Rickshaws", counts.get("auto_rickshaw", 0))
        col7.metric("Persons", counts.get("person", 0))
        col8.metric("Bicycles", counts.get("bicycle", 0))

        st.markdown("---")
        st.subheader("Annotated Tracking Video")
        if video_path.exists() and video_path.stat().st_size > 0:
            st.video(str(video_path))
        else:
            st.warning("Processed detection video not generated or empty.")

        with st.expander("Technical Details", expanded=False):
            st.json(data)

    else:
        st.info(f"Video uploaded: '{video_name}'. Click 'Process Video Pipeline' on the Main page to begin AI analysis.")
