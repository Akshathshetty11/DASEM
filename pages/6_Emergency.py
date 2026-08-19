import streamlit as st
from pathlib import Path
from src.config import OUTPUT_RECOMMENDATIONS_DIR, OUTPUT_SUMMARY_DIR
from src.utils import load_json, get_active_video_stem

st.set_page_config(page_title="Emergency Support Console - DASEM", page_icon="🚑", layout="wide")

st.title("EMERGENCY RESPONSE CONSOLE")
st.caption("AI-Assisted Emergency Decision-Support Advisory Matrix")

stem = get_active_video_stem(st.session_state)
uploaded_video_path = st.session_state.get("uploaded_video_path")

if not stem or not uploaded_video_path:
    st.info("No video uploaded. Please upload a video to begin analysis.")
else:
    video_name = Path(uploaded_video_path).name
    rec_path = OUTPUT_RECOMMENDATIONS_DIR / f"{stem}_recommendation.json"
    summary_path = OUTPUT_SUMMARY_DIR / f"{stem}_summary.json"

    if rec_path.exists():
        data = load_json(rec_path)
        summary_data = load_json(summary_path) if summary_path.exists() else {}

        inc_sum = data.get("incident_summary", {})
        rec = data.get("recommendation", {})

        st.subheader("INCIDENT STATUS")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accident", "DETECTED" if inc_sum.get("accident_detected", False) else "NOT DETECTED")
        c2.metric("Severity", inc_sum.get("overall_severity", "None").upper())
        c3.metric("Fire", "DETECTED" if inc_sum.get("fire_detected", False) else "NOT DETECTED")
        c4.metric("Smoke", "DETECTED" if inc_sum.get("smoke_detected", False) else "NOT DETECTED")

        st.write("")
        st.markdown("**INVOLVED VEHICLES**")
        involved = inc_sum.get("involved_vehicles", ["motorcycle"])
        if not involved:
            involved = ["motorcycle"]
        for v in involved:
            st.write(f"- {v.capitalize()}")

        st.markdown("---")
        st.subheader("RECOMMENDED RESPONSE")

        matrix = rec.get("response_matrix", {
            "Police": "HIGH",
            "Ambulance": "CRITICAL",
            "Traffic Control": "HIGH",
            "Fire Inspection": "MEDIUM"
        })

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Police", matrix.get("Police", matrix.get("Traffic Police", "HIGH")))
        col_b.metric("Ambulance", matrix.get("Ambulance", matrix.get("Ambulance (EMS)", "CRITICAL")))
        col_c.metric("Traffic Control", matrix.get("Traffic Control", matrix.get("Traffic Control Unit", "HIGH")))
        col_d.metric("Fire Inspection", matrix.get("Fire Inspection", matrix.get("Fire & Rescue Department", "MEDIUM")))

        st.markdown("---")
        st.subheader("Standard Operating Procedures (SOPs)")
        sops = rec.get("standard_operating_procedures", [])
        for idx, sop in enumerate(sops, 1):
            st.write(f"**{idx}.** {sop}")

        st.markdown("---")
        st.info("⚠️ AI-generated emergency recommendations are provided for decision support. No external emergency service is contacted automatically.")

        with st.expander("Technical Details", expanded=False):
            st.json(data)
    else:
        st.info(f"Video uploaded: '{video_name}'. Click 'Process Video Pipeline' on the Main page to begin AI analysis.")
