import streamlit as st
from pathlib import Path
from src.config import OUTPUT_FIRE_DIR, OUTPUT_SMOKE_DIR
from src.utils import load_json, get_active_video_stem

st.set_page_config(page_title="Hazard Detection - DASEM", page_icon="🔥", layout="wide")

st.title("HAZARD DETECTION")

stem = get_active_video_stem(st.session_state)
uploaded_video_path = st.session_state.get("uploaded_video_path")

if not stem or not uploaded_video_path:
    st.info("No video uploaded. Please upload a video to begin analysis.")
else:
    video_name = Path(uploaded_video_path).name
    fire_json = OUTPUT_FIRE_DIR / f"{stem}_fire.json"
    fire_video = OUTPUT_FIRE_DIR / f"{stem}_fire.mp4"
    smoke_json = OUTPUT_SMOKE_DIR / f"{stem}_smoke.json"
    smoke_video = OUTPUT_SMOKE_DIR / f"{stem}_smoke.mp4"

    if fire_json.exists() or smoke_json.exists():
        fire_data = load_json(fire_json) if fire_json.exists() else {}
        smoke_data = load_json(smoke_json) if smoke_json.exists() else {}

        fire_detected = fire_data.get("fire_detected", False)
        smoke_detected = smoke_data.get("smoke_detected", False)
        fire_events = fire_data.get("fire_events_count", 0)
        smoke_events = smoke_data.get("smoke_events_count", 0)

        max_conf = max(fire_data.get("fire_confidence", 0.0), smoke_data.get("smoke_confidence", 0.0))

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Fire Status", "DETECTED" if fire_detected else "NOT DETECTED")
        col2.metric("Smoke Status", "DETECTED" if smoke_detected else "NOT DETECTED")
        col3.metric("Fire Events", fire_events)
        col4.metric("Smoke Events", smoke_events)
        col5.metric("Maximum Confidence", f"{max_conf:.0f}%")

        st.markdown("---")

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Fire Detection Analysis")
            if fire_detected:
                st.error(f"Fire Detected (Confidence: {fire_data.get('fire_confidence', 0.0):.1f}%)")
            else:
                st.success("No Fire Detected")

            if fire_video.exists() and fire_video.stat().st_size > 0:
                st.video(str(fire_video))

        with col_b:
            st.subheader("Smoke Detection Analysis")
            if smoke_detected:
                st.warning(f"Smoke Plume Detected (Confidence: {smoke_data.get('smoke_confidence', 0.0):.1f}%)")
            else:
                st.success("No Smoke Detected")

            if smoke_video.exists() and smoke_video.stat().st_size > 0:
                st.video(str(smoke_video))

        with st.expander("Technical Details", expanded=False):
            st.json({"fire_data": fire_data, "smoke_data": smoke_data})
    else:
        st.info(f"Video uploaded: '{video_name}'. Click 'Process Video Pipeline' on the Main page to begin AI analysis.")
