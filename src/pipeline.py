import sys
import os
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    PROJECT_ROOT, OUTPUT_DIR, OUTPUT_SUMMARY_DIR, UPLOADED_VIDEOS_DIR
)
from src.utils import validate_file_path, save_json
from src.detect import detect_vehicles, resolve_input_video
from src.accident import detect_accidents_from_json
from src.severity import classify_severity_from_json
from src.fire import detect_fire
from src.smoke import detect_smoke
from src.emergency import generate_emergency_recommendation

logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
logger = logging.getLogger("DASEM.pipeline")

def run_pipeline(video_path=None):
    """
    Complete Sequential DASEM AI Computer Vision Pipeline & Centralized Single Source of Truth Builder.
    Dynamically processes ONLY the uploaded video file.
    Runs Module 1 (Detection & Tracking) -> Module 3 (Accident) -> Module 5 (Fire) -> Module 6 (Smoke) -> Module 4 (Severity) -> Module 7 (Emergency).
    Saves aggregated master single-source-of-truth output to output/summary/<video_stem>_summary.json.
    """
    if not video_path:
        uploaded_files = sorted(
            list(UPLOADED_VIDEOS_DIR.glob("*.mp4")) + list(UPLOADED_VIDEOS_DIR.glob("*.avi")) + list(UPLOADED_VIDEOS_DIR.glob("*.mov")),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if uploaded_files:
            video_path = uploaded_files[0]
        else:
            raise ValueError("No video uploaded. Please upload a video to begin analysis.")

    input_path = validate_file_path(video_path)
    video_name = input_path.name
    stem = input_path.stem

    logger.info(f"============================================================")
    logger.info(f"STARTING DASEM SEQUENTIAL AI PIPELINE FOR: {video_name}")
    logger.info(f"============================================================")

    # Step 1: Multi-Class YOLOv8 Detection & ByteTrack Persistent Tracking (Moving Vehicles Only)
    logger.info("\n--- STEP 1: Object Detection & Persistent Tracking ---")
    det_res = detect_vehicles(str(input_path))
    det_json_path = det_res["json_path"]

    # Step 2: Accident Detection Engine
    logger.info("\n--- STEP 2: Accident Candidate Evaluation ---")
    acc_res = detect_accidents_from_json(det_json_path)
    acc_json_path = acc_res["json_path"]

    # Step 3: Fire Detection Engine (Anti-False-Positive Texture & Flicker Verified)
    logger.info("\n--- STEP 3: Fire Detection & Spatial Vehicle Linking ---")
    fire_res = detect_fire(str(input_path), det_json_path)
    fire_json_path = fire_res["json_path"]

    # Step 4: Smoke Detection Engine (Anti-False-Positive Motion Verified)
    logger.info("\n--- STEP 4: Smoke Detection & Spatial Vehicle Linking ---")
    smoke_res = detect_smoke(str(input_path), det_json_path)
    smoke_json_path = smoke_res["json_path"]

    # Step 5: Severity Score Classification (Minor, Major, Critical)
    logger.info("\n--- STEP 5: Calibrated Severity Score Classification ---")
    sev_res = classify_severity_from_json(acc_json_path, fire_json_path, smoke_json_path)
    sev_json_path = sev_res["json_path"]

    # Step 6: Emergency Recommendation Advisory
    logger.info("\n--- STEP 6: Emergency Response Recommendation ---")
    emg_res = generate_emergency_recommendation(sev_json_path, fire_json_path, smoke_json_path)
    emg_json_path = emg_res["json_path"]

    # Build Master SINGLE SOURCE OF TRUTH Centralized Analysis Object
    OUTPUT_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    output_summary_path = OUTPUT_SUMMARY_DIR / f"{stem}_summary.json"

    master_summary = {
        "video": video_name,
        "video_stem": stem,
        "video_path": str(input_path),
        "detection_summary": {
            "total_unique_vehicles": det_res["total_unique_vehicles"],
            "total_unique_objects": det_res["total_unique_objects"],
            "unique_counts": det_res["unique_counts"],
            "track_class_map": det_res["track_class_map"],
            "moving_track_class_map": det_res.get("moving_track_class_map", det_res["track_class_map"])
        },
        "accident_summary": {
            "total_accidents": acc_res["total_accidents"]
        },
        "severity_summary": {
            "overall_severity": sev_res["overall_severity"],
            "overall_max_score": sev_res.get("overall_max_score", 0),
            "total_events": sev_res["total_events"]
        },
        "hazard_summary": {
            "fire_detected": fire_res["fire_detected"],
            "fire_confidence": fire_res["fire_confidence"],
            "affected_burning_vehicles": fire_res.get("affected_vehicles", []),
            "smoke_detected": smoke_res["smoke_detected"],
            "smoke_confidence": smoke_res["smoke_confidence"],
            "affected_smoke_vehicles": smoke_res.get("affected_vehicles", [])
        },
        "emergency_summary": {
            "response_priority": emg_res["response_priority"],
            "dispatch_services": emg_res["dispatch_services"]
        },
        "generated_files": {
            "detection_json": det_res["json_path"],
            "detected_video": det_res["video_path"],
            "accident_json": acc_res["json_path"],
            "severity_json": sev_res["json_path"],
            "fire_json": fire_res["json_path"],
            "fire_video": fire_res["video_path"],
            "smoke_json": smoke_res["json_path"],
            "smoke_video": smoke_res["video_path"],
            "recommendation_json": emg_res["json_path"]
        }
    }

    save_json(master_summary, output_summary_path)

    logger.info(f"\n============================================================")
    logger.info(f"DASEM PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info(f"Master Single Source of Truth Summary saved to: {output_summary_path}")
    logger.info(f"============================================================")

    return master_summary

# Export alias for backward compatibility with pages/1_Home.py
run_full_pipeline = run_pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DASEM Master AI Sequential Pipeline")
    parser.add_argument("video_path", nargs="?", default=None, help="Path to input uploaded video file")
    args = parser.parse_args()

    try:
        run_pipeline(args.video_path)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)
