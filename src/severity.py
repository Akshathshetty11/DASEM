import sys
import os
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, OUTPUT_SEVERITY_DIR
from src.utils import validate_file_path, load_json, save_json

logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
logger = logging.getLogger("DASEM.severity")

def classify_severity_from_json(accident_json_path, fire_json_path=None, smoke_json_path=None):
    """
    Process accident JSON, fire JSON, and smoke JSON to estimate dynamic accident severity scores (0-100).
    Categorizes events dynamically into: Mild (0-39), Moderate (40-64), Severe (65-84), Critical (85-100).
    Links Fire/Smoke hazards to accident events.
    Saves output to output/severity/<video_stem>_severity.json.
    """
    acc_path = validate_file_path(accident_json_path, extension=".json")
    acc_data = load_json(acc_path)
    
    video_name = acc_data.get("video", acc_path.name.replace("_accident.json", ".mp4"))
    stem = Path(video_name).stem
    accidents = acc_data.get("accidents", [])

    fire_data = load_json(fire_json_path) if fire_json_path and Path(fire_json_path).exists() else {}
    smoke_data = load_json(smoke_json_path) if smoke_json_path and Path(smoke_json_path).exists() else {}

    fire_detected = fire_data.get("fire_detected", False)
    smoke_detected = smoke_data.get("smoke_detected", False)

    fire_vehicles = {v.get("track_id") for v in fire_data.get("affected_vehicles", []) if v.get("track_id") is not None}
    smoke_vehicles = {v.get("track_id") for v in smoke_data.get("affected_vehicles", []) if v.get("track_id") is not None}

    logger.info(f"DASEM Module 4 - Accident Severity Estimation Engine")
    logger.info(f"Input Accident JSON: {acc_path.name} ({len(accidents)} events to evaluate)")

    severity_events = []
    max_severity_score = 0
    overall_severity_level = "None"

    if not accidents:
        result_json = {
            "video": video_name,
            "overall_severity": "None",
            "overall_max_score": 0,
            "total_events": 0,
            "events": []
        }
        OUTPUT_SEVERITY_DIR.mkdir(parents=True, exist_ok=True)
        output_json_path = OUTPUT_SEVERITY_DIR / f"{stem}_severity.json"
        save_json(result_json, output_json_path)
        return {
            "json_path": str(output_json_path),
            "overall_severity": "None",
            "overall_max_score": 0,
            "total_events": 0,
            "events": []
        }

    for idx, acc in enumerate(accidents, 1):
        base_score = float(acc.get("score", 40.0))
        vehicles = acc.get("vehicles", ["car"])
        vehicle_ids = set(acc.get("vehicle_ids", []))
        
        # Deduplicate vehicle list
        unique_vehicles = list(dict.fromkeys([v.capitalize() for v in vehicles]))

        # Hazard Association Check (Is fire/smoke attached to an accident event?)
        is_accident_fire = fire_detected and (bool(vehicle_ids.intersection(fire_vehicles)) or not fire_vehicles or acc.get("event_type") == "fire_incident")
        is_accident_smoke = smoke_detected and (bool(vehicle_ids.intersection(smoke_vehicles)) or not smoke_vehicles)

        hazard_bonus = 0.0
        if is_accident_fire:
            hazard_bonus += 35.0
        elif is_accident_smoke:
            hazard_bonus += 15.0

        # Multi-vehicle interaction factor
        vehicle_count_factor = min(15.0, max(0.0, (len(unique_vehicles) - 1) * 10.0))
        
        # Heavy transport collision momentum bonus (Bus / Truck involvement in a collision)
        has_heavy_transport = any(v.lower() in ['bus', 'truck'] for v in unique_vehicles)
        heavy_transport_factor = 15.0 if (has_heavy_transport and len(unique_vehicles) >= 2) else 0.0

        total_score = min(100, int(base_score + hazard_bonus + vehicle_count_factor + heavy_transport_factor))

        # Dynamic Severity Level Mapping
        if total_score >= 85 or is_accident_fire:
            sev_level = "Critical"
        elif total_score >= 65:
            sev_level = "Severe"
        elif total_score >= 40:
            sev_level = "Moderate"
        else:
            sev_level = "Mild"

        if total_score > max_severity_score:
            max_severity_score = total_score
            overall_severity_level = sev_level

        severity_events.append({
            "accident_id": idx,
            "start_time": acc.get("start_time", 0.0),
            "end_time": acc.get("end_time", 0.0),
            "duration_seconds": acc.get("duration_seconds", 0.7),
            "severity_score": total_score,
            "severity_level": sev_level,
            "participating_vehicles": unique_vehicles,
            "participating_vehicle_ids": list(vehicle_ids),
            "fire_involved": is_accident_fire,
            "smoke_involved": is_accident_smoke,
            "impact_metrics": {
                "distance": acc.get("distance", 0.0),
                "iou": acc.get("iou", 0.0),
                "bbox_change": acc.get("bbox_change", 0.0)
            }
        })

    OUTPUT_SEVERITY_DIR.mkdir(parents=True, exist_ok=True)
    output_json_path = OUTPUT_SEVERITY_DIR / f"{stem}_severity.json"

    result_json = {
        "video": video_name,
        "overall_severity": overall_severity_level,
        "overall_max_score": max_severity_score,
        "total_events": len(severity_events),
        "events": severity_events
    }

    save_json(result_json, output_json_path)

    logger.info(f"Evaluated {len(severity_events)} accident events.")
    logger.info(f"Overall Severity: {overall_severity_level} (Max Score: {max_severity_score}/100)")
    logger.info(f"Severity JSON saved to: {output_json_path}")

    return {
        "json_path": str(output_json_path),
        "overall_severity": overall_severity_level,
        "overall_max_score": max_severity_score,
        "total_events": len(severity_events),
        "events": severity_events
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DASEM Module 4 - Accident Severity Estimation Engine")
    parser.add_argument("accident_json", nargs="?", default=None, help="Path to accident JSON file")
    parser.add_argument("--fire_json", default=None, help="Path to fire JSON file")
    parser.add_argument("--smoke_json", default=None, help="Path to smoke JSON file")
    args = parser.parse_args()

    try:
        classify_severity_from_json(args.accident_json, args.fire_json, args.smoke_json)
    except Exception as e:
        logger.error(f"Severity classification failed: {e}")
        sys.exit(1)
