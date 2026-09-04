DASEM – Accident Detection, Severity Classification, Fire and Smoke Detection System

DASEM (Dynamic Accident Severity Estimation & Multi-Modal Hazard Detection) is a modular Computer Vision and AI analytics platform built with Python 3.12, Ultralytics YOLOv8, OpenCV, and Streamlit.

The system processes only user-uploaded MP4 videos and generates structured JSON outputs and annotated MP4 videos at every stage.

📁 Directory Structure

DASEM/

├── app.py — Streamlit Main Dashboard Entry Point

├── pages/ — Streamlit Navigation Pages
│ ├── 1_Home.py
│ ├── 2_Detection.py
│ ├── 3_Accident.py
│ ├── 4_Severity.py
│ ├── 5_Fire_Smoke.py
│ ├── 6_Emergency.py
│ ├── 7_Analytics.py
│ └── 8_About.py

├── src/ — Modular Python CLI Engines
│ ├── config.py — Central Configuration & Path Manager
│ ├── utils.py — Pathlib, JSON, and Video I/O Helpers
│ ├── detect.py — Module 1: Vehicle Detection
│ ├── tracker.py — Module 2: Vehicle Tracker
│ ├── accident.py — Module 3: Accident Detection Engine
│ ├── severity.py — Module 4: Severity Classification
│ ├── fire.py — Module 5: Fire Detection Engine
│ ├── smoke.py — Module 6: Smoke Detection Engine
│ ├── emergency.py — Module 7: Emergency Recommendation
│ └── pipeline.py — Module 8: Full Pipeline Runner

├── models/ — Model Weight Directories
│ ├── vehicle/
│ ├── accident/
│ ├── fire/
│ └── smoke/

├── videos/ — Video Storage
│ ├── uploaded/
│ └── samples/

└── output/ — Stage Artifact Outputs
├── detected/
├── detections/
├── accidents/
├── severity/
├── fire/
├── smoke/
├── emergency/
└── combined/

🧪 Independent CLI Module Execution

Every module in src/ can be executed independently from the terminal.

1. Vehicle Detection

Command: python src/detect.py "videos/uploaded/example.mp4"

Output:

output/detected/<video>_detected.mp4

output/detections/<video>.json

2. Accident Detection

Command: python src/accident.py "output/detections/example.json"

Output:

output/accidents/<video>_accident.json

3. Severity Classification

Command: python src/severity.py "output/accidents/example_accident.json"

Output:

output/severity/<video>_severity.json

4. Fire Detection

Command: python src/fire.py "videos/uploaded/example.mp4"

Output:

output/fire/<video>_fire.json

5. Smoke Detection

Command: python src/smoke.py "videos/uploaded/example.mp4"

Output:

output/smoke/<video>_smoke.json

6. Emergency Recommendation

Command:

python src/emergency.py "output/severity/example_severity.json" "output/fire/example_fire.json" "output/smoke/example_smoke.json"

Output:

output/emergency/<video>_emergency.json

7. Full Pipeline Execution

Command:

python src/pipeline.py "videos/uploaded/example.mp4"

8. Streamlit Dashboard

Command:

streamlit run app.py