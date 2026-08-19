import streamlit as st

st.set_page_config(page_title="About - DASEM", page_icon="ℹ️", layout="wide")

st.title("ℹ️ About DASEM System & Technology Stack")
st.markdown("""
### DASEM – Accident Detection, Severity Classification, Fire and Smoke Detection System

**DASEM** (*Dynamic Accident Severity Estimation & Multi-Modal Hazard Detection System*) is an AI platform designed for automated road-traffic video analysis.

---

### 🛠️ Core Technology & Computer Vision Stack

1. **YOLOv8 Object Detection & Persistent Multi-Object Tracking**:
   - Primary Object Detection powered by **Ultralytics YOLOv8**.
   - Multi-class road user detection (`person`, `car`, `bus`, `truck`, `motorcycle`, `bicycle`).
   - Persistent Object Tracking powered by **ByteTrack** / `VehicleTracker` assigning unique integer track IDs to every physical object.

2. **Unique Object Counting Engine**:
   - Computes counts strictly from unique persistent track ID sets per class:
     $$\text{Unique Cars} = |\text{set}(\text{car\_track\_ids})|$$
   - Each physical car, motorcycle, bus, truck, person, or bicycle is counted **only ONCE** across the entire video.

3. **Multi-Signal Accident Collision Engine**:
   - Evaluates vehicle proximity, bounding box overlap (IoU), bounding box area changes, and 8-frame temporal confirmation using persistent track IDs.

4. **Dynamic Severity Classification**:
   - Classifies collisions into **Mild**, **Moderate**, or **Severe** based on kinetic impact scores, duration, and involved objects.

5. **Dedicated Fire & Smoke Event Aggregation**:
   - Dedicated YOLOv8 / HSV flame flickering & spatial texture variance engines.
   - Merges continuous frame detections into discrete aggregated events.

6. **Emergency Recommendation Engine**:
   - Generates automated emergency dispatch protocols (`Police`, `Ambulance`, `Fire Rescue`, `Traffic Control`).

---

### 📂 File Path & Data Architecture:
- Built with `pathlib` relative to `PROJECT_ROOT`.
- Modular standalone Python CLI execution for every module (`detect.py`, `tracker.py`, `accident.py`, `severity.py`, `fire.py`, `smoke.py`, `emergency.py`, `pipeline.py`).
""")
