import streamlit as st
from pathlib import Path
from src.config import OUTPUT_ACCIDENTS_DIR, OUTPUT_SEVERITY_DIR
from src.utils import load_json, get_active_video_stem

st.set_page_config(page_title="Accident Detection - DASEM", page_icon="💥", layout="wide")

st.title("ACCIDENT DETECTION")

stem = get_active_video_stem(st.session_state)
uploaded_video_path = st.session_state.get("uploaded_video_path")

if not stem or not uploaded_video_path:
    st.info("No video uploaded. Please upload a video to begin analysis.")
else:
    video_name = Path(uploaded_video_path).name
    json_path = OUTPUT_ACCIDENTS_DIR / f"{stem}_accident.json"
    severity_path = OUTPUT_SEVERITY_DIR / f"{stem}_severity.json"

    if json_path.exists():
        data = load_json(json_path)
        accidents = data.get("accidents", [])
        
        sev_data = load_json(severity_path) if severity_path.exists() else {}
        sev_events = {ev.get("accident_id"): ev for ev in sev_data.get("events", [])}

        is_detected = len(accidents) > 0
        st.subheader("Status")
        if is_detected:
            st.markdown('<span class="status-badge-critical">Accident Detected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge-normal">No Accident Detected</span>', unsafe_allow_html=True)

        st.write("")
        st.metric("Total Accident Events", len(accidents))
        st.markdown("---")

        if not accidents:
            st.info("No collision or single-vehicle accident events identified in the processed video.")
        else:
            for idx, acc in enumerate(accidents, 1):
                sev_info = sev_events.get(idx, {})
                sev_level = sev_info.get("severity_level", "Moderate")
                conf = acc.get("confidence", "High")
                raw_vehicles = acc.get("vehicles", ["motorcycle"])
                
                # Deduplicate involved vehicles for UI display
                unique_vehicles = list(dict.fromkeys(raw_vehicles))

                st.subheader(f"Accident Event #{idx}")

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown("**Severity:**")
                    st.write(sev_level)
                with col_b:
                    st.markdown("**Confidence:**")
                    st.write(conf)
                with col_c:
                    st.markdown("**Duration:**")
                    st.write(f"{acc.get('duration_seconds', 0.7)} seconds")

                st.write("")
                st.markdown("**Involved Vehicles:**")
                for v in unique_vehicles:
                    st.write(f"- {v.capitalize()}")

                st.write("")
                st.markdown("**Evidence:**")
                reasons = acc.get("reason", [
                    "High vehicle proximity",
                    "Bounding-box overlap",
                    "Significant bounding-box change",
                    "Temporal confirmation"
                ])
                for r in reasons:
                    st.write(f"- {r}")

                st.markdown("---")

        with st.expander("Technical Details", expanded=False):
            st.json(data)
    else:
        st.info(f"Video uploaded: '{video_name}'. Click 'Process Video Pipeline' on the Main page to begin AI analysis.")
