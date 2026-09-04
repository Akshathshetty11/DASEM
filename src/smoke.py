import sys
import os
import argparse
import logging
import math
import cv2
import gc
import numpy as np
from collections import deque
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
    Optimized DASEM Smoke Detection Engine:
      - Uses motion difference, edge gradient analysis, shape expansion, and temporal multi-frame confirmation.
      - Safe Frame Sampling Stride (N=2 for 20+ FPS surveillance videos to optimize processing).
      - Zero RAM accumulation & efficient memory garbage collection.
      - Saves output to output/smoke/<video>_smoke.json and output video to output/smoke/<video>_smoke.mp4.
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

    logger.info(f"DASEM Module 6 - Smoke Detection Engine (Optimized Stream)")
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
    smoke_candidates = {}
    next_candidate_id = 1

    lower_smoke = np.array([0, 0, 130], dtype=np.uint8)
    upper_smoke = np.array([180, 35, 215], dtype=np.uint8)

    all_linked_vehicles = {}
    last_valid_contours = []
    last_smoke_area_pixels = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            timestamp = round(frame_idx / fps, 2)

            should_analyze = (frame_idx == 1 or frame_idx % 2 == 1 or not last_valid_contours)

            if should_analyze:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)

                valid_smoke_contours = []
                frame_smoke_pixels = 0

                contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    area = cv2.contourArea(c)
                    if area > 400:
                        x, y, w, h = cv2.boundingRect(c)
                        bbox_area = max(1.0, float(w * h))
                        fill_ratio = float(area) / bbox_area

                        roi_gray = gray[y:y+h, x:x+w]
                        laplacian_var = float(cv2.Laplacian(roi_gray, cv2.CV_64F).var())

                        motion_score = 0.0
                        if prev_gray is not None and prev_gray.shape == gray.shape:
                            roi_gray_curr = gray[y:y+h, x:x+w]
                            roi_gray_prev = prev_gray[y:y+h, x:x+w]
                            diff = cv2.absdiff(roi_gray_curr, roi_gray_prev)
                            motion_score = float(np.mean(diff))

                        is_diffuse = fill_ratio < 0.70 or laplacian_var < 350.0
                        is_moving_smoke = motion_score >= 1.25

                        if is_diffuse and is_moving_smoke:
                            valid_smoke_contours.append((x, y, w, h, area, motion_score, laplacian_var))
                            frame_smoke_pixels += area

                if prev_gray is not None:
                    del prev_gray
                prev_gray = gray

                last_valid_contours = valid_smoke_contours
                last_smoke_area_pixels = frame_smoke_pixels
                del hsv, smoke_mask
            else:
                valid_smoke_contours = last_valid_contours
                frame_smoke_pixels = last_smoke_area_pixels

            smoke_area_pct = round((frame_smoke_pixels / total_pixels) * 100.0, 3)

            current_centroids = []
            for (x, y, w, h, s_area, m_score, l_var) in valid_smoke_contours:
                cx = x + w / 2.0
                cy = y + h / 2.0
                current_centroids.append({
                    "bbox": (x, y, w, h),
                    "centroid": (cx, cy),
                    "area": s_area,
                    "motion": m_score
                })

            updated_candidates = {}
            matched_curr_indices = set()

            for cand_id, cand_data in smoke_candidates.items():
                last_c = cand_data["centroid"]
                best_idx = None
                best_dist = float('inf')

                for idx, curr in enumerate(current_centroids):
                    if idx in matched_curr_indices:
                        continue
                    d = euclidean_distance(last_c, curr["centroid"])
                    if d < max(120.0, math.sqrt(curr["area"]) * 1.2) and d < best_dist:
                        best_dist = d
                        best_idx = idx

                if best_idx is not None:
                    curr_match = current_centroids[best_idx]
                    matched_curr_indices.add(best_idx)
                    prev_area = cand_data["area"]
                    area_expansion = (curr_match["area"] - prev_area) / max(1.0, prev_area)

                    cand_data["streak"] += 1
                    cand_data["last_frame"] = frame_idx
                    cand_data["centroid"] = curr_match["centroid"]
                    cand_data["bbox"] = curr_match["bbox"]
                    cand_data["area"] = curr_match["area"]
                    cand_data["area_history"].append(curr_match["area"])
                    cand_data["expansion_rate"] = area_expansion
                    updated_candidates[cand_id] = cand_data
                else:
                    cand_data["disappeared"] = cand_data.get("disappeared", 0) + 1
                    if cand_data["disappeared"] <= 10:
                        updated_candidates[cand_id] = cand_data

            for idx, curr in enumerate(current_centroids):
                if idx not in matched_curr_indices:
                    updated_candidates[next_candidate_id] = {
                        "id": next_candidate_id,
                        "streak": 1,
                        "start_frame": frame_idx,
                        "last_frame": frame_idx,
                        "centroid": curr["centroid"],
                        "bbox": curr["bbox"],
                        "area": curr["area"],
                        "area_history": [curr["area"]],
                        "expansion_rate": 0.0,
                        "disappeared": 0
                    }
                    next_candidate_id += 1

            smoke_candidates = updated_candidates

            confirmed_candidates = [
                c for c in smoke_candidates.values()
                if c["streak"] >= 5 and (c["area_history"][-1] >= c["area_history"][0] * 1.05 or c["area"] > 800)
            ]

            frame_dets = det_frames_data.get(frame_idx, [])
            linked_vehicles_in_frame = []

            for cand in confirmed_candidates:
                cx, cy, cw, ch = cand["bbox"]
                smoke_box = [cx, cy, cx + cw, cy + ch]
                s_center = cand["centroid"]
                s_area = cand["area"]

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
                    iou = calculate_iou(smoke_box, [vb["x1"], vb["y1"], vb["x2"], vb["y2"]])

                    if iou > 0.05 or euclidean_distance(s_center, (v_cx, v_cy)) < v_diag * 0.70:
                        if tid not in {v["track_id"] for v in linked_vehicles_in_frame}:
                            linked_vehicles_in_frame.append({
                                "track_id": tid,
                                "class": st_cls
                            })
                            all_linked_vehicles[tid] = st_cls

            if confirmed_candidates and smoke_area_pct > 0.03:
                smoke_frames += 1
                if smoke_area_pct > max_smoke_area_pct:
                    max_smoke_area_pct = smoke_area_pct

                if current_interval is None:
                    current_interval = {
                        "start_time": timestamp,
                        "start_frame": frame_idx,
                        "end_time": timestamp,
                        "end_frame": frame_idx,
                        "max_pct": smoke_area_pct,
                        "linked_vehicles": list(linked_vehicles_in_frame)
                    }
                else:
                    current_interval["end_time"] = timestamp
                    current_interval["end_frame"] = frame_idx
                    current_interval["max_pct"] = max(current_interval["max_pct"], smoke_area_pct)
                    for lv in linked_vehicles_in_frame:
                        if lv["track_id"] not in {v["track_id"] for v in current_interval["linked_vehicles"]}:
                            current_interval["linked_vehicles"].append(lv)
            else:
                if current_interval is not None:
                    current_interval["duration_seconds"] = round(current_interval["end_time"] - current_interval["start_time"], 1)
                    if current_interval["duration_seconds"] >= 0.4:
                        smoke_intervals.append(current_interval)
                    current_interval = None

            if confirmed_candidates:
                for cand in confirmed_candidates:
                    x, y, w, h = cand["bbox"]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (180, 180, 180), 2)
                    
                if linked_vehicles_in_frame:
                    best_link = linked_vehicles_in_frame[0]
                    label_txt = f"SMOKE PLUME: {best_link['class'].upper()} #{best_link['track_id']}"
                else:
                    label_txt = "SMOKE DETECTED"

                cv2.putText(frame, label_txt, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            out.write(frame)
            del frame

            if frame_idx % 60 == 0:
                gc.collect()

    finally:
        cap.release()
        out.release()
        if prev_gray is not None:
            del prev_gray
        gc.collect()

    if current_interval is not None:
        current_interval["duration_seconds"] = round(current_interval["end_time"] - current_interval["start_time"], 1)
        if current_interval["duration_seconds"] >= 0.4:
            smoke_intervals.append(current_interval)

    overall_smoke_detected = len(smoke_intervals) > 0 and smoke_frames >= 4
    smoke_confidence = min(95.0, round(float(max_smoke_area_pct * 30.0) + 50.0, 1)) if overall_smoke_detected else 0.0

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
                        "description": f"Vehicle with Smoke Emission: {lv['class'].capitalize()} (Track #{tid})"
                    })
                    seen_tids.add(tid)

    output_data = {
        "video": video_name,
        "smoke_detected": overall_smoke_detected,
        "smoke_confidence": smoke_confidence if overall_smoke_detected else 0.0,
        "max_smoke_area_percentage": max_smoke_area_pct,
        "total_smoke_frames": smoke_frames if overall_smoke_detected else 0,
        "smoke_events_count": len(smoke_intervals) if overall_smoke_detected else 0,
        "affected_vehicles": affected_vehicles if overall_smoke_detected else [],
        "smoke_intervals": smoke_intervals if overall_smoke_detected else []
    }

    save_json(output_data, output_json_path)

    logger.info(f"Smoke Detected: {overall_smoke_detected} (Confidence: {smoke_confidence}%)")
    logger.info(f"Affected Smoke Emission Vehicles: {affected_vehicles}")
    logger.info(f"Smoke JSON saved to: {output_json_path}")

    return {
        "json_path": str(output_json_path),
        "video_path": str(output_video_path),
        "smoke_detected": overall_smoke_detected,
        "smoke_confidence": smoke_confidence,
        "affected_vehicles": affected_vehicles
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DASEM Module 6 - Smoke Detection Engine")
    parser.add_argument("video_path", nargs="?", default=None, help="Path to input video file")
    parser.add_argument("--detection_json", type=str, default=None, help="Path to detection JSON file")
    args = parser.parse_args()

    try:
        detect_smoke(args.video_path, args.detection_json)
    except Exception as e:
        logger.error(f"Smoke detection failed: {e}")
        sys.exit(1)
