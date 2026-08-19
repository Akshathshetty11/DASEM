import sys
import os
import argparse
import logging
import math
import cv2
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, OUTPUT_FIRE_DIR, OUTPUT_DETECTIONS_DIR
from src.utils import validate_file_path, get_video_metadata, save_json, load_json
from src.tracker import calculate_iou, euclidean_distance

logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
logger = logging.getLogger("DASEM.fire")

def detect_fire(video_path, detection_json_path=None):
    """
    Process ONLY the uploaded video file passed through CLI arguments.
    Combines HSV color analysis, texture variance filtering (std_dev >= 18.0), frame-to-frame flickering (delta I >= 4.0),
    and temporal persistence to eliminate false positives from orange/yellow vehicles, text overlays, and banners.
    Links verified fire events ONLY to tracked vehicles (e.g. Burning Vehicle: Car Track 8), ignoring humans.
    Saves output to output/fire/<video>_fire.json and output video to output/fire/<video>_fire.mp4.
    """
    input_path = validate_file_path(video_path)
    video_name = input_path.name
    stem = input_path.stem

    metadata = get_video_metadata(input_path)
    fps = metadata['fps']
    width = metadata['width']
    height = metadata['height']
    total_frames = metadata['frame_count']
    total_pixels = max(1, width * height)

    # Load detection data if available for spatial vehicle linking
    detection_data = {}
    if detection_json_path and Path(detection_json_path).exists():
        detection_data = load_json(Path(detection_json_path))
    else:
        fallback_det = OUTPUT_DETECTIONS_DIR / f"{stem}.json"
        if fallback_det.exists():
            detection_data = load_json(fallback_det)

    track_class_map = detection_data.get("moving_track_class_map", detection_data.get("track_class_map", {}))
    det_frames_data = {f.get("frame_number"): f.get("detections", []) for f in detection_data.get("frames", [])}

    logger.info(f"DASEM Module 5 - Fire Detection Engine (Anti-False-Positive Verified)")
    logger.info(f"Input video: {video_name} ({width}x{height} @ {fps} FPS, {total_frames} frames)")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"[ERROR] Cannot open VideoCapture for {input_path}")

    OUTPUT_FIRE_DIR.mkdir(parents=True, exist_ok=True)
    output_video_path = OUTPUT_FIRE_DIR / f"{stem}_fire.mp4"
    output_json_path = OUTPUT_FIRE_DIR / f"{stem}_fire.json"

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    fire_frames = 0
    max_flame_area_pct = 0.0
    fire_intervals = []
    current_interval = None
    frame_idx = 0
    prev_gray = None

    # HSV Fire Mask Ranges (Bright Orange / Yellow / Red Flames)
    lower_fire1 = np.array([0, 110, 180], dtype=np.uint8)
    upper_fire1 = np.array([35, 255, 255], dtype=np.uint8)
    
    lower_fire2 = np.array([165, 110, 180], dtype=np.uint8)
    upper_fire2 = np.array([180, 255, 255], dtype=np.uint8)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_idx += 1
        timestamp = round(frame_idx / fps, 2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
        mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
        fire_mask = cv2.bitwise_or(mask1, mask2)

        # Morphological filter
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)

        valid_flame_contours = []
        frame_flame_pixels = 0

        # Evaluate candidate contours for texture variance & flickering
        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area > 120:
                x, y, w, h = cv2.boundingRect(c)
                
                roi_val = hsv[y:y+h, x:x+w, 2]
                texture_std = float(np.std(roi_val))

                flicker_score = 0.0
                if prev_gray is not None and prev_gray.shape == gray.shape:
                    roi_gray_curr = gray[y:y+h, x:x+w]
                    roi_gray_prev = prev_gray[y:y+h, x:x+w]
                    flicker_score = float(np.mean(cv2.absdiff(roi_gray_curr, roi_gray_prev)))

                if texture_std >= 18.0 and (flicker_score >= 4.0 or area > 800):
                    valid_flame_contours.append((x, y, w, h, area))
                    frame_flame_pixels += area

        prev_gray = gray.copy()

        flame_area_pct = round((frame_flame_pixels / total_pixels) * 100.0, 3)
        if flame_area_pct > max_flame_area_pct:
            max_flame_area_pct = flame_area_pct

        is_fire_frame = len(valid_flame_contours) > 0 and flame_area_pct > 0.05

        if is_fire_frame:
            fire_frames += 1

            # Match fire contours to active tracked vehicles in current frame (excluding persons)
            frame_dets = det_frames_data.get(frame_idx, [])
            linked_vehicles_in_frame = []

            for (x, y, w, h, c_area) in valid_flame_contours:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                fire_box = [x, y, x + w, y + h]
                fire_cx = x + w / 2.0
                fire_cy = y + h / 2.0

                for d in frame_dets:
                    tid = d.get("track_id")
                    if tid is not None:
                        st_cls = track_class_map.get(str(tid), track_class_map.get(tid, d.get("class", "car")))
                        if st_cls in ['person', 'human']:
                            continue

                        vb = d["bbox"]
                        v_cx = (vb["x1"] + vb["x2"]) / 2.0
                        v_cy = (vb["y1"] + vb["y2"]) / 2.0
                        v_diag = math.sqrt((vb["x2"]-vb["x1"])**2 + (vb["y2"]-vb["y1"])**2)
                        
                        dist = euclidean_distance((fire_cx, fire_cy), (v_cx, v_cy))
                        iou = calculate_iou(fire_box, [vb["x1"], vb["y1"], vb["x2"], vb["y2"]])

                        if iou > 0.05 or dist < max(80.0, v_diag * 0.70):
                            linked_vehicles_in_frame.append({
                                "track_id": tid,
                                "class": st_cls
                            })

            if linked_vehicles_in_frame:
                best_link = linked_vehicles_in_frame[0]
                label_txt = f"BURNING VEHICLE: {best_link['class'].upper()} #{best_link['track_id']}"
            else:
                label_txt = "FIRE DETECTED"

            cv2.putText(frame, label_txt, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if current_interval is None:
                current_interval = {
                    "start_time": timestamp,
                    "start_frame": frame_idx,
                    "max_pct": flame_area_pct,
                    "linked_vehicles": linked_vehicles_in_frame
                }
            else:
                current_interval["max_pct"] = max(current_interval["max_pct"], flame_area_pct)
                existing_ids = {v["track_id"] for v in current_interval["linked_vehicles"]}
                for lv in linked_vehicles_in_frame:
                    if lv["track_id"] not in existing_ids:
                        current_interval["linked_vehicles"].append(lv)
                        existing_ids.add(lv["track_id"])
        else:
            if current_interval is not None:
                current_interval["end_time"] = timestamp
                current_interval["end_frame"] = frame_idx
                current_interval["duration_seconds"] = round(timestamp - current_interval["start_time"], 1)
                fire_intervals.append(current_interval)
                current_interval = None

        out.write(frame)

    if current_interval is not None:
        current_interval["end_time"] = round(total_frames / fps, 2)
        current_interval["end_frame"] = total_frames
        current_interval["duration_seconds"] = round(current_interval["end_time"] - current_interval["start_time"], 1)
        fire_intervals.append(current_interval)

    cap.release()
    out.release()

    overall_fire_detected = len(fire_intervals) > 0 and fire_frames >= 4
    fire_confidence = min(98.0, round(float(max_flame_area_pct * 40.0) + 45.0, 1)) if overall_fire_detected else 0.0

    affected_vehicles = []
    seen_tids = set()
    if overall_fire_detected:
        for intv in fire_intervals:
            for lv in intv.get("linked_vehicles", []):
                tid = lv["track_id"]
                if tid not in seen_tids:
                    affected_vehicles.append({
                        "track_id": tid,
                        "class": lv["class"],
                        "description": f"Burning Vehicle: {lv['class'].capitalize()} (Track #{tid})"
                    })
                    seen_tids.add(tid)

    output_data = {
        "video": video_name,
        "fire_detected": overall_fire_detected,
        "fire_confidence": fire_confidence if overall_fire_detected else 0.0,
        "max_flame_area_percentage": max_flame_area_pct,
        "total_fire_frames": fire_frames,
        "fire_events_count": len(fire_intervals) if overall_fire_detected else 0,
        "affected_vehicles": affected_vehicles,
        "fire_intervals": fire_intervals if overall_fire_detected else []
    }

    save_json(output_data, output_json_path)

    logger.info(f"Fire Detected: {overall_fire_detected} (Confidence: {fire_confidence}%)")
    logger.info(f"Affected Burning Vehicles: {affected_vehicles}")
    logger.info(f"Fire JSON saved to: {output_json_path}")

    return {
        "json_path": str(output_json_path),
        "video_path": str(output_video_path),
        "fire_detected": overall_fire_detected,
        "fire_confidence": fire_confidence,
        "affected_vehicles": affected_vehicles
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DASEM Module 5 - Anti-False-Positive Fire Detection Engine")
    parser.add_argument("video_path", nargs="?", default=None, help="Path to input video file")
    parser.add_argument("--detection_json", type=str, default=None, help="Path to detection JSON file")
    args = parser.parse_args()

    try:
        detect_fire(args.video_path, args.detection_json)
    except Exception as e:
        logger.error(f"Fire detection failed: {e}")
        sys.exit(1)
