import streamlit as st
from pathlib import Path
from src.config import OUTPUT_SEVERITY_DIR
from src.utils import load_json, get_active_video_stem

st.set_page_config(page_title="Severity Classification - DASEM", page_icon="📈", layout="wide")

st.title("SEVERITY CLASSIFICATION")

stem = get_active_video_stem(st.session_state)
uploaded_video_path = st.session_state.get("uploaded_video_path")

if not stem or not uploaded_video_path:
    st.info("No video uploaded. Please upload a video to begin analysis.")
else:
    video_name = Path(uploaded_video_path).name
    json_path = OUTPUT_SEVERITY_DIR / f"{stem}_severity.json"

    if json_path.exists():
        data = load_json(json_path)
        events = data.get("events", [])

        mild_count = sum(1 for e in events if e.get("severity_level") in ["Minor", "Mild"])
        mod_count = sum(1 for e in events if e.get("severity_level") == "Moderate")
        sev_count = sum(1 for e in events if e.get("severity_level") in ["Severe", "Critical", "Major"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Mild Events", mild_count)
        col2.metric("Moderate Events", mod_count)
        col3.metric("Severe Events", sev_count)

        st.markdown("---")

        if not events:
            st.info("No severe impact events evaluated in the selected video.")
        else:
            for ev in events:
                idx = ev.get("accident_id", 1)
                sev_lvl = ev.get("severity_level", "Moderate").upper()
                score = ev.get("severity_score", 45)
                conf = "HIGH" if score >= 70 else "MODERATE"
                raw_vehicles = ev.get("participating_vehicles", ["motorcycle"])

                # Deduplicate vehicle labels
                unique_vehicles = list(dict.fromkeys(raw_vehicles))

                st.subheader(f"Accident Event #{idx}")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Severity:**")
                    st.write(sev_lvl)
                with c2:
                    st.markdown("**Severity Score:**")
                    st.write(f"{score} / 100")
                with c3:
                    st.markdown("**Confidence:**")
                    st.write(conf)

                st.write("")
                st.markdown("**Involved Vehicles:**")
                for v in unique_vehicles:
                    st.write(f"- {v.capitalize()}")

                st.markdown("---")

        with st.expander("Technical Details", expanded=False):
            st.json(data)
    else:
        st.info(f"Video uploaded: '{video_name}'. Click 'Process Video Pipeline' on the Main page to begin AI analysis.")
