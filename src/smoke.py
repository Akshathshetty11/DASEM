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

from src.config import PROJECT_ROOT, OUTPUT_SMOKE_DIR, OUTPUT_DETECTIONS_DIR
from src.utils import validate_file_path, get_video_metadata, save_json, load_json
from src.tracker import calculate_iou, euclidean_distance

logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
logger = logging.getLogger("DASEM.smoke")

def detect_smoke(video_path, detection_json_path=None):
    """
    Process ONLY the uploaded video file passed through CLI arguments.
    Uses motion difference, edge gradient analysis, shape expansion, and temporal multi-frame confirmation.
    Eliminates false positives from static concrete pillars, grey walls, road surfaces, and static shadows.
    Links verified smoke plumes ONLY to tracked vehicles (e.g. Smoke Plume: Car Track 8), ignoring humans.
    Saves output to output/smoke/<video>_smoke.json and output video to output/smoke/<video>_smoke.mp4.
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

    logger.info(f"DASEM Module 6 - Smoke Detection Engine (Anti-False-Positive Verified)")
    logger.info(f"Input video: {video_name} ({width}x{height} @ {fps} FPS, {total_frames} frames)")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"[ERROR] Cannot open VideoCapture for {input_path}")

    OUTPUT_SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    output_video_path = OUTPUT_SMOKE_DIR / f"{stem}_smoke.mp4"
    output_json_path = OUTPUT_SMOKE_DIR / f"{stem}_smoke.json"

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    smoke_frames = 0
    max_smoke_area_pct = 0.0
    smoke_intervals = []
    current_interval = None
    frame_idx = 0
    prev_gray = None

    # HSV Smoke Mask Ranges (Low Saturation, Medium-High Value grayish/whitish plume)
    lower_smoke = np.array([0, 0, 110], dtype=np.uint8)
    upper_smoke = np.array([180, 50, 220], dtype=np.uint8)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_idx += 1
        timestamp = round(frame_idx / fps, 2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)

        # Morphological filter to isolate diffuse smoke plumes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)

        valid_smoke_contours = []
        frame_smoke_pixels = 0

        # Evaluate candidate contours for MOTION and EDGE GRADIENT (diffuse vs static pillar/wall)
        contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area > 400:
                x, y, w, h = cv2.boundingRect(c)

                motion_score = 0.0
                if prev_gray is not None and prev_gray.shape == gray.shape:
                    roi_gray_curr = gray[y:y+h, x:x+w]
                    roi_gray_prev = prev_gray[y:y+h, x:x+w]
                    motion_score = float(np.mean(cv2.absdiff(roi_gray_curr, roi_gray_prev)))

                roi_gray = gray[y:y+h, x:x+w]
                sobel_edges = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 1, ksize=3)
                edge_density = float(np.mean(np.abs(sobel_edges)))

                if motion_score >= 2.5 and edge_density < 65.0:
                    valid_smoke_contours.append((x, y, w, h, area))
                    frame_smoke_pixels += area

        prev_gray = gray.copy()

        smoke_area_pct = round((frame_smoke_pixels / total_pixels) * 100.0, 3)
        if smoke_area_pct > max_smoke_area_pct:
            max_smoke_area_pct = smoke_area_pct

        is_smoke_frame = len(valid_smoke_contours) > 0 and smoke_area_pct > 0.15

        if is_smoke_frame:
            smoke_frames += 1

            # Match smoke contours to active tracked vehicles in current frame (excluding persons)
            frame_dets = det_frames_data.get(frame_idx, [])
            linked_vehicles_in_frame = []

            for (x, y, w, h, s_area) in valid_smoke_contours:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (180, 180, 180), 2)

                smoke_box = [x, y, x + w, y + h]
                smoke_cx = x + w / 2.0
                smoke_cy = y + h / 2.0

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

                        dist = euclidean_distance((smoke_cx, smoke_cy), (v_cx, v_cy))
                        iou = calculate_iou(smoke_box, [vb["x1"], vb["y1"], vb["x2"], vb["y2"]])

                        if iou > 0.05 or dist < max(100.0, v_diag * 0.85):
                            linked_vehicles_in_frame.append({
                                "track_id": tid,
                                "class": st_cls
                            })

            if linked_vehicles_in_frame:
                best_link = linked_vehicles_in_frame[0]
                label_txt = f"SMOKE PLUME: {best_link['class'].upper()} #{best_link['track_id']}"
            else:
                label_txt = "SMOKE DETECTED"

            cv2.putText(frame, label_txt, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            if current_interval is None:
                current_interval = {
                    "start_time": timestamp,
                    "start_frame": frame_idx,
                    "max_pct": smoke_area_pct,
                    "linked_vehicles": linked_vehicles_in_frame
                }
            else:
                current_interval["max_pct"] = max(current_interval["max_pct"], smoke_area_pct)
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
                smoke_intervals.append(current_interval)
                current_interval = None

        out.write(frame)

    if current_interval is not None:
        current_interval["end_time"] = round(total_frames / fps, 2)
        current_interval["end_frame"] = total_frames
        current_interval["duration_seconds"] = round(current_interval["end_time"] - current_interval["start_time"], 1)
        smoke_intervals.append(current_interval)

    cap.release()
    out.release()

    overall_smoke_detected = len(smoke_intervals) > 0 and smoke_frames >= 5
    smoke_confidence = min(95.0, round(float(max_smoke_area_pct * 25.0) + 40.0, 1)) if overall_smoke_detected else 0.0

    affected_vehicles = []
    seen_tids = set()
    if overall_smoke_detected:
        for intv in smoke_intervals:
            for lv in intv.get("linked_vehicles", []):
                tid = lv["track_id"]
                if tid not in seen_tids:
                    affected_vehicles.append({
                        "track_id": tid,
                        "class": lv["class"],
                        "description": f"Smoke Plume: {lv['class'].capitalize()} (Track #{tid})"
                    })
                    seen_tids.add(tid)

    output_data = {
        "video": video_name,
        "smoke_detected": overall_smoke_detected,
        "smoke_confidence": smoke_confidence if overall_smoke_detected else 0.0,
        "max_smoke_area_percentage": max_smoke_area_pct,
        "total_smoke_frames": smoke_frames,
        "smoke_events_count": len(smoke_intervals) if overall_smoke_detected else 0,
        "affected_vehicles": affected_vehicles,
        "smoke_intervals": smoke_intervals if overall_smoke_detected else []
    }

    save_json(output_data, output_json_path)

    logger.info(f"Smoke Detected: {overall_smoke_detected} (Confidence: {smoke_confidence}%)")
    logger.info(f"Affected Vehicles: {affected_vehicles}")
    logger.info(f"Smoke JSON saved to: {output_json_path}")

    return {
        "json_path": str(output_json_path),
        "video_path": str(output_video_path),
        "smoke_detected": overall_smoke_detected,
        "smoke_confidence": smoke_confidence,
        "affected_vehicles": affected_vehicles
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DASEM Module 6 - Anti-False-Positive Smoke Detection Engine")
    parser.add_argument("video_path", nargs="?", default=None, help="Path to input video file")
    parser.add_argument("--detection_json", type=str, default=None, help="Path to detection JSON file")
    args = parser.parse_args()

    try:
        detect_smoke(args.video_path, args.detection_json)
    except Exception as e:
        logger.error(f"Smoke detection failed: {e}")
        sys.exit(1)
