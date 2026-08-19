import sys
import os
import argparse
import logging
import math
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, OUTPUT_ACCIDENTS_DIR
from src.utils import validate_file_path, load_json, save_json
from src.tracker import calculate_iou, euclidean_distance

logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
logger = logging.getLogger("DASEM.accident")

# Parameters
PROXIMITY_MULTIPLIER = 0.45
IOU_THRESHOLD = 0.30
BBOX_AREA_CHANGE_THRESHOLD = 0.25
TEMPORAL_CONFIRMATION_FRAMES = 5
COOLDOWN_FRAMES = 25
EVENT_MERGE_GAP_SEC = 2.5
FINAL_SCORE_THRESHOLD = 30  # Allow low-impact/mild events to be recorded with appropriate score

def detect_accidents_from_json(json_path):
    """
    Process ONLY the detection JSON file passed through CLI arguments.
    Reads persistent track IDs and track_class_map as the SINGLE SOURCE OF TRUTH.
    Deduplicates Track IDs for every accident event (never lists the same Track ID multiple times).
    Calculates evidence-based score strictly from physical impact metrics (NO frame-duration inflation).
    Evaluates both single-vehicle anomaly events and multi-vehicle collision events.
    Saves JSON to output/accidents/<video_stem>_accident.json.
    """
    input_path = validate_file_path(json_path, extension=".json")
    detection_data = load_json(input_path)

    video_name = detection_data.get("video", input_path.name.replace(".json", ".mp4"))
    stem = Path(video_name).stem
    fps = detection_data.get("fps", 25.0)
    frames_data = detection_data.get("frames", [])
    
    # Single Source of Truth Track Class Map
    track_class_map = detection_data.get("moving_track_class_map", detection_data.get("track_class_map", {}))

    logger.info(f"DASEM Module 3 - Accident Detection Engine")
    logger.info(f"Input JSON: {input_path.name} ({len(frames_data)} frames analyzed)")

    # Build per-track lifetime trajectory, displacement, aspect ratio, and area histories
    track_frame_history = {}

    for frame_info in frames_data:
        frame_num = frame_info["frame_number"]
        timestamp = frame_info.get("timestamp", round(frame_num / fps, 2))
        detections = frame_info.get("detections", [])

        for d in detections:
            tid = d.get('track_id')
            if tid is not None:
                st_cls = track_class_map.get(str(tid), track_class_map.get(tid, d.get('class', 'car')))
                if st_cls in ['person', 'human']:
                    continue

                b = d['bbox']
                bw = float(b['x2'] - b['x1'])
                bh = float(max(1.0, b['y2'] - b['y1']))
                cx = (b['x1'] + b['x2']) / 2.0
                cy = (b['y1'] + b['y2']) / 2.0
                ar = bw / bh
                area = bw * bh

                if tid not in track_frame_history:
                    track_frame_history[tid] = []

                track_frame_history[tid].append({
                    "frame_number": frame_num,
                    "timestamp": timestamp,
                    "bbox": [b['x1'], b['y1'], b['x2'], b['y2']],
                    "centroid": (cx, cy),
                    "aspect_ratio": ar,
                    "area": area,
                    "class": st_cls
                })

    raw_detected_events = []
    cooldown_counter = 0

    # 1. EVALUATE SINGLE-VEHICLE ANOMALY EVENTS (Falling motorcycle, overturning vehicle, sudden deceleration/stop)
    for tid, history in track_frame_history.items():
        if len(history) < 6:
            continue

        st_cls = track_class_map.get(str(tid), track_class_map.get(tid, history[0]['class']))

        displacements = []
        for i in range(1, len(history)):
            prev_pt = history[i-1]['centroid']
            curr_pt = history[i]['centroid']
            displacements.append(euclidean_distance(prev_pt, curr_pt))

        aspect_ratios = [h['aspect_ratio'] for h in history]
        max_disp = max(displacements) if displacements else 0.0
        
        tipping_detected = False
        initial_ar = history[0]['aspect_ratio']
        for h in history[2:]:
            curr_ar = h['aspect_ratio']
            if st_cls in ['motorcycle', 'bicycle'] and (curr_ar >= 1.45 or abs(curr_ar - initial_ar) >= 0.50):
                tipping_detected = True
                break

        sudden_stop_detected = False
        if len(displacements) >= 8 and max_disp > 8.0:
            recent_disp = np.mean(displacements[-4:])
            if recent_disp < 1.8 and (max_disp / max(0.5, recent_disp)) > 4.0:
                sudden_stop_detected = True

        if tipping_detected or (sudden_stop_detected and max_disp > 10.0):
            start_t = history[0]['timestamp']
            end_t = history[-1]['timestamp']
            dur_sec = round(end_t - start_t, 1)

            reasons = []
            if tipping_detected:
                reasons.append(f"Single-vehicle {st_cls} tipping / falling posture detected")
            if sudden_stop_detected:
                reasons.append(f"Abnormal deceleration and post-event stationary posture")
            reasons.append("Temporal anomaly confirmation")

            # Physical evidence-based score (No frame duration multiplier!)
            score = 35 if tipping_detected else 30
            if max_disp > 20.0:
                score += 10

            raw_detected_events.append({
                "start_time": start_t,
                "end_time": end_t,
                "duration_seconds": max(0.5, dur_sec),
                "accident": True,
                "score": score,
                "confidence": "High" if score >= 70 else "Moderate",
                "vehicles": [st_cls],
                "vehicle_ids": [int(tid)],
                "distance": round(max_disp, 1),
                "iou": 0.0,
                "bbox_change": round(abs(aspect_ratios[-1] - initial_ar), 2) if aspect_ratios else 0.0,
                "reason": reasons
            })

    # 2. EVALUATE MULTI-VEHICLE COLLISION INTERACTION EVENTS
    pair_streaks = {}

    for frame_info in frames_data:
        frame_num = frame_info["frame_number"]
        timestamp = frame_info.get("timestamp", round(frame_num / fps, 2))
        detections = frame_info.get("detections", [])

        active_tracks = {}
        for d in detections:
            tid = d.get('track_id')
            if tid is not None:
                st_cls = track_class_map.get(str(tid), track_class_map.get(tid, d.get('class', 'car')))
                if st_cls in ['person', 'human']:
                    continue

                b = d['bbox']
                cx = (b['x1'] + b['x2']) / 2.0
                cy = (b['y1'] + b['y2']) / 2.0
                active_tracks[tid] = {
                    'id': int(tid),
                    'class': st_cls,
                    'bbox': [b['x1'], b['y1'], b['x2'], b['y2']],
                    'centroid': (cx, cy)
                }

        tids = list(active_tracks.keys())
        current_frame_pairs = set()

        if len(tids) >= 2:
            for i in range(len(tids)):
                for j in range(i + 1, len(tids)):
                    id1 = tids[i]
                    id2 = tids[j]
                    pair_key = (min(id1, id2), max(id1, id2))

                    t1 = active_tracks[id1]
                    t2 = active_tracks[id2]
                    b1 = t1['bbox']
                    b2 = t2['bbox']

                    dist = euclidean_distance(t1['centroid'], t2['centroid'])
                    avg_diag = (math.sqrt((b1[2]-b1[0])**2 + (b1[3]-b1[1])**2) + math.sqrt((b2[2]-b2[0])**2 + (b2[3]-b2[1])**2)) / 2.0
                    iou = calculate_iou(b1, b2)

                    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                    area_diff_pct = abs(a1 - a2) / max(1.0, max(a1, a2))

                    is_interaction = (iou >= IOU_THRESHOLD) or (dist < avg_diag * PROXIMITY_MULTIPLIER and area_diff_pct >= BBOX_AREA_CHANGE_THRESHOLD)

                    if is_interaction:
                        current_frame_pairs.add(pair_key)
                        
                        if pair_key not in pair_streaks:
                            pair_streaks[pair_key] = {
                                "streak": 1,
                                "start_frame": frame_num,
                                "last_frame": frame_num,
                                "start_time": timestamp,
                                "end_time": timestamp,
                                "max_iou": iou,
                                "max_dist": dist,
                                "max_bbox_change": area_diff_pct,
                                "v1_class": t1['class'],
                                "v2_class": t2['class'],
                                "v1_id": id1,
                                "v2_id": id2,
                                "confirmed": False
                            }
                        else:
                            st = pair_streaks[pair_key]
                            st["streak"] += 1
                            st["last_frame"] = frame_num
                            st["end_time"] = timestamp
                            st["max_iou"] = max(st["max_iou"], iou)
                            st["max_dist"] = min(st["max_dist"], dist)
                            st["max_bbox_change"] = max(st["max_bbox_change"], area_diff_pct)
                            
                            if st["streak"] >= TEMPORAL_CONFIRMATION_FRAMES and not st["confirmed"] and cooldown_counter <= 0:
                                # EVIDENCE-BASED KINETIC SCORING ONLY (NO frame duration streak multiplier!)
                                score_iou = min(35.0, (st["max_iou"] / 0.50) * 35.0)
                                score_bbox = min(25.0, (st["max_bbox_change"] / 0.50) * 25.0)
                                base_kinetic = 20.0
                                total_score = min(100, int(base_kinetic + score_iou + score_bbox))

                                if total_score >= FINAL_SCORE_THRESHOLD:
                                    reasons = ["High vehicle proximity"]
                                    if st["max_iou"] >= IOU_THRESHOLD:
                                        reasons.append("Bounding-box overlap")
                                    if st["max_bbox_change"] >= BBOX_AREA_CHANGE_THRESHOLD:
                                        reasons.append("Significant bounding-box change")
                                    reasons.append("Temporal confirmation")

                                    duration_sec = round((frame_num - st["start_frame"] + 1) / fps, 1)
                                    v1_st = track_class_map.get(str(id1), track_class_map.get(id1, st["v1_class"]))
                                    v2_st = track_class_map.get(str(id2), track_class_map.get(id2, st["v2_class"]))

                                    # Unique Track IDs per event (Deduplicated)
                                    unique_event_ids = sorted(list(dict.fromkeys([id1, id2])))
                                    unique_event_vehicles = [track_class_map.get(str(i), track_class_map.get(i, 'car')) for i in unique_event_ids]

                                    raw_detected_events.append({
                                        "start_time": st["start_time"],
                                        "end_time": timestamp,
                                        "duration_seconds": max(0.5, duration_sec),
                                        "accident": True,
                                        "score": total_score,
                                        "confidence": "High" if total_score >= 70 else "Moderate",
                                        "vehicles": unique_event_vehicles,
                                        "vehicle_ids": unique_event_ids,
                                        "distance": round(st["max_dist"], 1),
                                        "iou": round(st["max_iou"], 2),
                                        "bbox_change": round(st["max_bbox_change"], 2),
                                        "reason": reasons
                                    })
                                    st["confirmed"] = True
                                    cooldown_counter = COOLDOWN_FRAMES

        if cooldown_counter > 0:
            cooldown_counter -= 1

        for pk in list(pair_streaks.keys()):
            if pk not in current_frame_pairs:
                if frame_num - pair_streaks[pk]["last_frame"] > 10:
                    del pair_streaks[pk]

    # EVENT MERGING: Merge overlapping temporal events into discrete accident events with Unique Track IDs
    final_confirmed_events = []
    for ev in raw_detected_events:
        if not final_confirmed_events:
            # Deduplicate vehicle IDs and vehicle classes
            ev["vehicle_ids"] = sorted(list(dict.fromkeys([int(x) for x in ev["vehicle_ids"]])))
            ev["vehicles"] = [track_class_map.get(str(i), track_class_map.get(i, 'car')) for i in ev["vehicle_ids"]]
            final_confirmed_events.append(ev)
        else:
            prev_ev = final_confirmed_events[-1]
            if (ev["start_time"] - prev_ev["end_time"]) < EVENT_MERGE_GAP_SEC:
                prev_ev["end_time"] = max(prev_ev["end_time"], ev["end_time"])
                prev_ev["duration_seconds"] = round(prev_ev["end_time"] - prev_ev["start_time"], 1)
                prev_ev["score"] = max(prev_ev["score"], ev["score"])
                prev_ev["iou"] = max(prev_ev["iou"], ev["iou"])
                prev_ev["bbox_change"] = max(prev_ev["bbox_change"], ev["bbox_change"])
                
                # Merge & deduplicate vehicle IDs
                combined_ids = sorted(list(dict.fromkeys(prev_ev["vehicle_ids"] + [int(x) for x in ev["vehicle_ids"]])))
                combined_vehicles = [track_class_map.get(str(i), track_class_map.get(i, 'car')) for i in combined_ids]
                prev_ev["vehicle_ids"] = combined_ids
                prev_ev["vehicles"] = combined_vehicles
                
                for r in ev.get("reason", []):
                    if r not in prev_ev["reason"]:
                        prev_ev["reason"].append(r)
            else:
                ev["vehicle_ids"] = sorted(list(dict.fromkeys([int(x) for x in ev["vehicle_ids"]])))
                ev["vehicles"] = [track_class_map.get(str(i), track_class_map.get(i, 'car')) for i in ev["vehicle_ids"]]
                final_confirmed_events.append(ev)

    # Save Output JSON
    OUTPUT_ACCIDENTS_DIR.mkdir(parents=True, exist_ok=True)
    output_json_path = OUTPUT_ACCIDENTS_DIR / f"{stem}_accident.json"

    result_json = {
        "video": video_name,
        "total_accidents": len(final_confirmed_events),
        "accidents": final_confirmed_events
    }

    save_json(result_json, output_json_path)

    logger.info(f"Confirmed accident events (Deduplicated Track IDs): {len(final_confirmed_events)}")
    logger.info(f"Accident JSON saved to: {output_json_path}")

    return {
        "json_path": str(output_json_path),
        "total_accidents": len(final_confirmed_events)
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DASEM Module 3 - Independent Accident Detection Engine")
    parser.add_argument("json_path", nargs="?", default=None, help="Path to detection JSON file")
    args = parser.parse_args()

    try:
        detect_accidents_from_json(args.json_path)
    except Exception as e:
        logger.error(f"Accident detection failed: {e}")
        sys.exit(1)
