import sys
import os
import argparse
import logging
import math
import cv2
import gc
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, OUTPUT_FIRE_DIR, OUTPUT_DETECTIONS_DIR, OUTPUT_SMOKE_DIR
from src.utils import validate_file_path, get_video_metadata, save_json, load_json
from src.tracker import calculate_iou, euclidean_distance

logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
logger = logging.getLogger("DASEM.fire")

def detect_fire(video_path, detection_json_path=None, smoke_json_path=None):
    """
    Process ONLY the uploaded video file passed through CLI arguments.
    Anti-False-Positive Fire Detection Engine with Rigid Vehicle Body Filtering & Incandescent Core Verification:
      - Color, color contrast, brightness, or warm pixels alone NEVER independently trigger Fire Detected.
      - Red buses, red cars, orange trucks, yellow vehicles, brake lights, signs, reflections MUST NEVER trigger Fire Detected = Yes.
      - Rejects flat painted vehicle surfaces & vehicle translation:
        1. Rigid Vehicle Body Filter: If contour covers > 40% of a vehicle bounding box AND solidity >= 0.72, REJECT (it's a car/bus/truck body).
        2. Incandescent Glow Core: Requires V_max >= 215 to ensure an incandescent flame core is present.
        3. Internal Turbulence: Requires non-uniform intensity turbulence std(delta_ROI) >= 5.5 to distinguish flame flickering from rigid vehicle motion.
      - Qualifies candidate flame regions when S_flame >= 5.0 and texture_std >= 14.0.
      - Confirms Fire Detected = Yes when flame evidence persists across >= 5 consecutive frames.
      - Does NOT depend on smoke presence.
      - Saves output to output/fire/<video>_fire.json and video to output/fire/<video>_fire.mp4.
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

    detection_data = {}
    if detection_json_path and Path(detection_json_path).exists():
        detection_data = load_json(Path(detection_json_path))
    else:
        fallback_det = OUTPUT_DETECTIONS_DIR / f"{stem}.json"
        if fallback_det.exists():
            detection_data = load_json(fallback_det)

    track_class_map = detection_data.get("moving_track_class_map", detection_data.get("track_class_map", {}))
    det_frames_data = {f.get("frame_number"): f.get("detections", []) for f in detection_data.get("frames", [])}

    logger.info(f"DASEM Module 5 - Anti-False-Positive Fire Detection Engine (Rigid Body Rejection & Incandescent Core)")
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

    max_flame_area_pct = 0.0
    frame_idx = 0
    prev_gray = None

    # HSV Flame Core Mask Ranges (Orange/Yellow/Red Flames, Saturation S >= 110, Value V >= 180)
    lower_fire1 = np.array([0, 110, 180], dtype=np.uint8)
    upper_fire1 = np.array([32, 255, 255], dtype=np.uint8)
    
    lower_fire2 = np.array([165, 110, 180], dtype=np.uint8)
    upper_fire2 = np.array([180, 255, 255], dtype=np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    candidate_fire_streak = 0
    confirmed_fire_frames_data = []

    try:
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
            fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)

            valid_flame_contours = []
            frame_flame_pixels = 0

            frame_dets = det_frames_data.get(frame_idx, [])

            contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                perimeter = cv2.arcLength(c, True)
                if area > 120 and perimeter > 0:
                    x, y, w, h = cv2.boundingRect(c)
                    bbox_area = max(1.0, float(w * h))
                    fill_ratio = float(area) / bbox_area
                    p2_over_a = (perimeter * perimeter) / area

                    hull = cv2.convexHull(c)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / max(1.0, hull_area)

                    roi_val = hsv[y:y+h, x:x+w, 2]
                    roi_gray = gray[y:y+h, x:x+w]
                    texture_std = float(np.std(roi_val))
                    v_max = float(np.max(roi_val))
                    
                    sobel_edges = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 1, ksize=3)
                    sobel_std = float(np.std(sobel_edges))

                    # 1. RIGID VEHICLE BODY REJECTION FILTER
                    # If this contour covers a significant portion of a detected vehicle AND has high structural solidity (>= 0.72),
                    # it is a normal vehicle body (car/bus/truck paint surface), NOT a non-rigid flame plume!
                    is_vehicle_body = False
                    fire_box = [x, y, x + w, y + h]
                    for d in frame_dets:
                        tid = d.get("track_id")
                        st_cls = track_class_map.get(str(tid), track_class_map.get(tid, d.get("class", "car")))
                        if st_cls in ['person', 'human']:
                            continue
                        vb = d["bbox"]
                        v_box = [vb["x1"], vb["y1"], vb["x2"], vb["y2"]]
                        v_area = max(1.0, (vb["x2"] - vb["x1"]) * (vb["y2"] - vb["y1"]))
                        iou = calculate_iou(fire_box, v_box)
                        overlap_ratio = area / v_area
                        if (iou > 0.35 or overlap_ratio > 0.40) and solidity >= 0.72:
                            is_vehicle_body = True
                            break

                    if is_vehicle_body:
                        continue  # REJECT NORMAL VEHICLE BODY!

                    # 2. INCANDESCENT FLAME CORE REQUIREMENT
                    # Genuine flames contain incandescent bright glow peaks (V_max >= 215). Normal paint has uniform lower brightness (V_max < 205).
                    is_incandescent_core = v_max >= 215.0

                    # 3. MOTION-COMPENSATED FLAME TURBULENCE ANALYSIS
                    flicker_score = 0.0
                    changed_ratio = 0.0
                    internal_turbulence = 0.0

                    if prev_gray is not None and prev_gray.shape == gray.shape:
                        roi_gray_curr = gray[y:y+h, x:x+w]
                        roi_gray_prev = prev_gray[y:y+h, x:x+w]
                        roi_delta = cv2.absdiff(roi_gray_curr, roi_gray_prev)
                        internal_turbulence = float(np.std(roi_delta))
                        flicker_score = float(np.mean(roi_delta))
                        changed_ratio = float(np.mean(roi_delta >= 16))

                    is_real_flicker = (internal_turbulence >= 5.5 and (flicker_score >= 4.0 or changed_ratio >= 0.08))
                    s_flicker = 2.5 if is_real_flicker else 0.0

                    is_irregular_flame_shape = (solidity < 0.72 and (p2_over_a >= 20.0 or fill_ratio <= 0.60))
                    s_shape = 2.5 if is_irregular_flame_shape else 0.0

                    is_flame_texture = (texture_std >= 22.0 or sobel_std >= 25.0)
                    s_texture = 2.5 if is_flame_texture else 0.0

                    s_core = 2.5 if (float(np.mean(roi_val)) >= 185.0 and is_incandescent_core) else 0.0

                    # MANDATORY GENUINE FLAME EVIDENCE RULE:
                    # Color saturation (s_core) and smooth vehicle translation CANNOT trigger Fire Detected!
                    # Fire requires ACTUAL flame characteristics: irregular flame shape OR internal flame turbulence!
                    # AND high spatial flame texture (is_flame_texture) AND incandescent core (is_incandescent_core).
                    has_genuine_flame_evidence = (s_shape > 0.0 or s_flicker > 0.0) and is_flame_texture and is_incandescent_core

                    if has_genuine_flame_evidence:
                        s_flame = s_shape + s_texture + s_flicker + s_core
                    else:
                        s_flame = 0.0

                    if s_flame >= 5.0 and texture_std >= 14.0:
                        valid_flame_contours.append((x, y, w, h, area, s_flame))
                        frame_flame_pixels += area

            if prev_gray is not None:
                del prev_gray
            prev_gray = gray

            flame_area_pct = round((frame_flame_pixels / total_pixels) * 100.0, 3)

            linked_vehicles_in_frame = []
            for (x, y, w, h, flame_area, _) in valid_flame_contours:
                fire_box = [x, y, x + w, y + h]
                fire_cx = x + w / 2.0
                fire_cy = y + h / 2.0
                for d in frame_dets:
                    tid = d.get("track_id")
                    if tid is None:
                        continue
                    st_cls = track_class_map.get(str(tid), track_class_map.get(tid, d.get("class", "car")))
                    if st_cls in ['person', 'human']:
                        continue
                    vb = d["bbox"]
                    v_cx = (vb["x1"] + vb["x2"]) / 2.0
                    v_cy = (vb["y1"] + vb["y2"]) / 2.0
                    v_diag = math.sqrt((vb["x2"]-vb["x1"])**2 + (vb["y2"]-vb["y1"])**2)
                    vehicle_area = max(1.0, (vb["x2"] - vb["x1"]) * (vb["y2"] - vb["y1"]))
                    iou = calculate_iou(fire_box, [vb["x1"], vb["y1"], vb["x2"], vb["y2"]])
                    local_flame = flame_area <= vehicle_area * 0.85
                    if local_flame and (iou > 0.05 or euclidean_distance((fire_cx, fire_cy), (v_cx, v_cy)) < v_diag * 0.55):
                        if tid not in {v["track_id"] for v in linked_vehicles_in_frame}:
                            linked_vehicles_in_frame.append({"track_id": tid, "class": st_cls})

            is_candidate_frame = len(valid_flame_contours) > 0 and flame_area_pct > 0.03

            if is_candidate_frame:
                candidate_fire_streak += 1
            else:
                candidate_fire_streak = max(0, candidate_fire_streak - 1)

            is_confirmed_fire_frame = candidate_fire_streak >= 5

            if is_confirmed_fire_frame:
                if flame_area_pct > max_flame_area_pct:
                    max_flame_area_pct = flame_area_pct

                confirmed_fire_frames_data.append({
                    "frame_idx": frame_idx,
                    "timestamp": timestamp,
                    "flame_area_pct": flame_area_pct,
                    "linked_vehicles": linked_vehicles_in_frame
                })

                for (x, y, w, h, c_area, _) in valid_flame_contours:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                if linked_vehicles_in_frame:
                    best_link = linked_vehicles_in_frame[0]
                    label_txt = f"BURNING VEHICLE: {best_link['class'].upper()} #{best_link['track_id']}"
                else:
                    label_txt = "FIRE DETECTED"

                cv2.putText(frame, label_txt, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            out.write(frame)
            del frame, hsv, mask1, mask2, fire_mask

            if frame_idx % 60 == 0:
                gc.collect()

    finally:
        cap.release()
        out.release()
        if prev_gray is not None:
            del prev_gray
        gc.collect()

    fire_intervals = []
    current_interval = None

    for fdata in confirmed_fire_frames_data:
        f_num = fdata["frame_idx"]
        t_stamp = fdata["timestamp"]
        pct = fdata["flame_area_pct"]
        l_vehs = fdata["linked_vehicles"]

        if current_interval is None:
            current_interval = {
                "start_time": t_stamp,
                "start_frame": f_num,
                "end_time": t_stamp,
                "end_frame": f_num,
                "max_pct": pct,
                "linked_vehicles": list(l_vehs)
            }
        else:
            if (t_stamp - current_interval["end_time"]) <= 2.0:
                current_interval["end_time"] = t_stamp
                current_interval["end_frame"] = f_num
                current_interval["max_pct"] = max(current_interval["max_pct"], pct)
                existing_ids = {v["track_id"] for v in current_interval["linked_vehicles"]}
                for lv in l_vehs:
                    if lv["track_id"] not in existing_ids:
                        current_interval["linked_vehicles"].append(lv)
                        existing_ids.add(lv["track_id"])
            else:
                current_interval["duration_seconds"] = round(current_interval["end_time"] - current_interval["start_time"], 1)
                fire_intervals.append(current_interval)
                current_interval = {
                    "start_time": t_stamp,
                    "start_frame": f_num,
                    "end_time": t_stamp,
                    "end_frame": f_num,
                    "max_pct": pct,
                    "linked_vehicles": list(l_vehs)
                }

    if current_interval is not None:
        current_interval["duration_seconds"] = round(current_interval["end_time"] - current_interval["start_time"], 1)
        fire_intervals.append(current_interval)

    overall_fire_detected = len(fire_intervals) > 0 and len(confirmed_fire_frames_data) >= 5
    fire_confidence = min(98.0, round(float(max_flame_area_pct * 40.0) + 55.0, 1)) if overall_fire_detected else 0.0

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
        "total_fire_frames": len(confirmed_fire_frames_data) if overall_fire_detected else 0,
        "fire_events_count": len(fire_intervals) if overall_fire_detected else 0,
        "affected_vehicles": affected_vehicles if overall_fire_detected else [],
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
    parser.add_argument("--smoke_json", type=str, default=None, help="Path to smoke JSON file")
    args = parser.parse_args()

    try:
        detect_fire(args.video_path, args.detection_json, args.smoke_json)
    except Exception as e:
        logger.error(f"Fire detection failed: {e}")
        sys.exit(1)
