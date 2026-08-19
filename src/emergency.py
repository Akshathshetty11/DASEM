import sys
import os
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, OUTPUT_RECOMMENDATIONS_DIR, OUTPUT_FIRE_DIR, OUTPUT_SMOKE_DIR
from src.utils import validate_file_path, load_json, save_json

logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
logger = logging.getLogger("DASEM.emergency")

def generate_emergency_recommendation(severity_json_path, fire_json_path=None, smoke_json_path=None):
    """
    Process severity, fire, and smoke outputs.
    Generates structured emergency response priority matrix and SOPs.
    Saves JSON to output/recommendations/<video_stem>_recommendation.json.
    """
    input_path = validate_file_path(severity_json_path, extension=".json")
    severity_data = load_json(input_path)

    video_name = severity_data.get("video", input_path.name.replace("_severity.json", ".mp4"))
    stem = Path(video_name).stem
    overall_severity = severity_data.get("overall_severity", "None")
    overall_score = severity_data.get("overall_max_score", 0)
    events = severity_data.get("events", [])

    # Load fire & smoke data if available
    fire_detected = False
    if fire_json_path and Path(fire_json_path).exists():
        fdata = load_json(Path(fire_json_path))
        fire_detected = fdata.get("fire_detected", False)
    else:
        fallback_fire = OUTPUT_FIRE_DIR / f"{stem}_fire.json"
        if fallback_fire.exists():
            fdata = load_json(fallback_fire)
            fire_detected = fdata.get("fire_detected", False)

    smoke_detected = False
    if smoke_json_path and Path(smoke_json_path).exists():
        sdata = load_json(Path(smoke_json_path))
        smoke_detected = sdata.get("smoke_detected", False)
    else:
        fallback_smoke = OUTPUT_SMOKE_DIR / f"{stem}_smoke.json"
        if fallback_smoke.exists():
            sdata = load_json(fallback_smoke)
            smoke_detected = sdata.get("smoke_detected", False)

    # Collect participating vehicles from events
    involved_vehicles = []
    for ev in events:
        for v in ev.get("participating_vehicles", []):
            if v not in involved_vehicles:
                involved_vehicles.append(v)

    # Response priority calculation
    if fire_detected and overall_severity in ["Severe", "Critical"]:
        priority = "P1 - Critical Urgent Dispatch"
        eta = "3 - 5 mins"
    elif overall_severity in ["Severe", "Critical"] or len(events) >= 2:
        priority = "P1 - High Priority Emergency Dispatch"
        eta = "4 - 7 mins"
    elif overall_severity == "Major" or smoke_detected:
        priority = "P2 - Priority Response Dispatch"
        eta = "8 - 12 mins"
    elif overall_severity in ["Mild", "Minor"]:
        priority = "P3 - Standard Patrol Dispatch"
        eta = "12 - 18 mins"
    else:
        priority = "P4 - Normal Monitoring"
        eta = "N/A"

    # Emergency Service Response Matrix (Service -> Priority Level)
    response_matrix = {}
    if overall_severity in ["Severe", "Critical"] or len(events) > 0:
        response_matrix["Traffic Police"] = "HIGH" if overall_severity != "Critical" else "CRITICAL"
        response_matrix["Ambulance (EMS)"] = "CRITICAL" if overall_severity in ["Severe", "Critical"] else "HIGH"
        response_matrix["Traffic Control Unit"] = "HIGH"
        response_matrix["Fire & Rescue Department"] = "CRITICAL" if fire_detected else ("HIGH" if smoke_detected else "MEDIUM")
    else:
        response_matrix["Traffic Police"] = "LOW"
        response_matrix["Ambulance (EMS)"] = "STANDBY"
        response_matrix["Traffic Control Unit"] = "NORMAL"
        response_matrix["Fire & Rescue Department"] = "STANDBY"

    sops = []
    if overall_severity in ["Severe", "Critical"]:
        sops.append("Immediately dispatch primary EMS and traffic management units to incident coordinates.")
        sops.append("Reroute oncoming traffic to prevent secondary multi-vehicle collisions.")
    if fire_detected:
        sops.append("Deploy HAZMAT fire suppression units. Maintain 50m safety perimeter.")
    if smoke_detected:
        sops.append("Activate electronic variable message signs (VMS) warning drivers of low visibility.")
    if not sops:
        sops.append("Maintain routine automated camera monitoring.")

    disclaimer = "AI-generated emergency recommendations are provided for decision support. No external emergency service is contacted automatically."

    output_data = {
        "video": video_name,
        "incident_summary": {
            "accident_detected": len(events) > 0,
            "overall_severity": overall_severity,
            "overall_max_score": overall_score,
            "fire_detected": fire_detected,
            "smoke_detected": smoke_detected,
            "involved_vehicles": involved_vehicles
        },
        "recommendation": {
            "response_priority": priority,
            "estimated_arrival": eta,
            "response_matrix": response_matrix,
            "dispatch_services": [f"{k} ({v})" for k, v in response_matrix.items() if v in ["HIGH", "CRITICAL"]],
            "standard_operating_procedures": sops,
            "disclaimer": disclaimer
        }
    }

    OUTPUT_RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_json_path = OUTPUT_RECOMMENDATIONS_DIR / f"{stem}_recommendation.json"

    save_json(output_data, output_json_path)

    logger.info(f"Response Priority: {priority}")
    logger.info(f"Response Matrix: {response_matrix}")
    logger.info(f"Recommendation JSON saved to: {output_json_path}")

    return {
        "json_path": str(output_json_path),
        "response_priority": priority,
        "dispatch_services": [f"{k} ({v})" for k, v in response_matrix.items() if v in ["HIGH", "CRITICAL"]]
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DASEM Module 7 - Smart Emergency Advisory Console")
    parser.add_argument("severity_json_path", nargs="?", default=None, help="Path to severity JSON file")
    args = parser.parse_args()

    try:
        generate_emergency_recommendation(args.severity_json_path)
    except Exception as e:
        logger.error(f"Emergency recommendation failed: {e}")
        sys.exit(1)
