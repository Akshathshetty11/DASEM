import streamlit as st
import pandas as pd
from pathlib import Path
from src.config import OUTPUT_SUMMARY_DIR
from src.utils import load_json, get_active_video_stem

st.set_page_config(page_title="Analytics - DASEM", page_icon="📊", layout="wide")

st.title("SYSTEM OVERVIEW & ANALYTICS")

stem = get_active_video_stem(st.session_state)
uploaded_video_path = st.session_state.get("uploaded_video_path")

if not stem or not uploaded_video_path:
    st.info("No video uploaded. Please upload a video to begin analysis.")
else:
    video_name = Path(uploaded_video_path).name
    summary_path = OUTPUT_SUMMARY_DIR / f"{stem}_summary.json"

    if summary_path.exists():
        sdata = load_json(summary_path)

        det_sum = sdata.get("detection_summary", {})
        acc_sum = sdata.get("accident_summary", {})
        sev_sum = sdata.get("severity_summary", {})
        haz_sum = sdata.get("hazard_summary", {})

        st.subheader("SYSTEM OVERVIEW")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Unique Objects", det_sum.get("total_unique_objects", 0))
        col2.metric("Accident Events", acc_sum.get("total_accidents", 0))
        col3.metric("Overall Severity", sev_sum.get("overall_severity", "None"))
        col4.metric("Fire Events", 1 if haz_sum.get("fire_detected", False) else 0)
        col5.metric("Smoke Events", 1 if haz_sum.get("smoke_detected", False) else 0)

        st.markdown("---")
        st.subheader("UNIQUE OBJECT BREAKDOWN")

        counts = det_sum.get("unique_counts", {})
        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
        m1.metric("Cars", counts.get("car", 0))
        m2.metric("Motorcycles", counts.get("motorcycle", 0))
        m3.metric("Buses", counts.get("bus", 0))
        m4.metric("Trucks", counts.get("truck", 0))
        m5.metric("Auto Rickshaws", counts.get("auto_rickshaw", 0))
        m6.metric("Persons", counts.get("person", 0))
        m7.metric("Bicycles", counts.get("bicycle", 0))

        st.write("")
        if counts:
            chart_df = pd.DataFrame([
                {"Category": k.replace("_", " ").title(), "Unique Count": v}
                for k, v in counts.items()
            ])
            st.bar_chart(chart_df.set_index("Category"))

        with st.expander("Technical Details", expanded=False):
            st.json(sdata)
    else:
        st.info(f"Video uploaded: '{video_name}'. Click 'Process Video Pipeline' on the Main page to begin AI analysis.")
