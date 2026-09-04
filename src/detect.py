import sys
import os
import argparse
import logging
import math
import cv2
import gc
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    PROJECT_ROOT, VEHICLE_MODEL_PATH, UPLOADED_VIDEOS_DIR,
    OUTPUT_DETECTED_DIR, OUTPUT_DETECTIONS_DIR
)
from src.utils import validate_file_path, get_video_metadata, save_json
from src.tracker import VehicleTracker, euclidean_distance

try:
    from ultralytics import YOLO
    import torch
    HAS_ULTRALYTICS = True
except Exception:
    HAS_ULTRALYTICS = False

logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
logger = logging.getLogger("DASEM.detect")

# Global Cached Model Singleton
_GLOBAL_YOLO_MODEL = None

def get_yolo_model(model_path=None):
    """
    Singleton Loader for YOLOv8 model weights. Loads model into RAM/VRAM ONLY ONCE per session.
    Configures PyTorch multithreading or CUDA GPU acceleration automatically.
    """
    global _GLOBAL_YOLO_MODEL
    if _GLOBAL_YOLO_MODEL is not None:
        return _GLOBAL_YOLO_MODEL

    if not HAS_ULTRALYTICS:
        return None

    device = 'cpu'
    try:
        if torch.cuda.is_available():
            device = 0
            logger.info("[PERF] PyTorch CUDA GPU Acceleration active!")
        else:
            n_threads = max(1, os.cpu_count() or 4)
            torch.set_num_threads(n_threads)
            logger.info(f"[PERF] PyTorch CPU multithreading configured ({n_threads} threads).")
    except Exception as e:
        logger.warning(f"Could not configure PyTorch threads: {e}")

    weights_to_try = []
    if model_path and Path(model_path).exists():
        weights_to_try.append(str(model_path))
    weights_to_try.extend(['yolov8n.pt', str(VEHICLE_MODEL_PATH), 'yolov8s.pt', 'yolov8m.pt'])

    for w in weights_to_try:
        try:
            _GLOBAL_YOLO_MODEL = YOLO(w)
            logger.info(f"Loaded YOLOv8 Detection Model (Cached Singleton): {w}")
            return _GLOBAL_YOLO_MODEL
        except Exception as e:
            logger.warning(f"Could not load model {w}: {e}")
    return None


# COCO & Custom Class Mapping
COCO_CLASS_MAP = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

# Color Map per Class (BGR)
CLASS_COLOR_MAP = {
    'car': (255, 128, 0),             # Blue
    'bus': (0, 255, 0),               # Green
    'truck': (0, 140, 255),           # Orange
    'motorcycle': (0, 255, 255),       # Yellow
    'auto_rickshaw': (255, 0, 140),   # Purple/Pink
    'person': (255, 255, 0),          # Cyan
    'bicycle': (255, 0, 255),         # Magenta
    'other': (180, 180, 180)          # Gray
}

def suppress_rider_person_detections(detections):
    """Remove only the person box belonging to a rider, never the bike itself."""
    two_wheelers = [d for d in detections if d.get("class") in {"motorcycle", "bicycle"}]
    if not two_wheelers:
        return detections

    filtered = []
    for detection in detections:
        if detection.get("class") != "person":
            filtered.append(detection)
            continue

        person = detection["bbox"]
        person_center_x = (person["x1"] + person["x2"]) / 2.0
        person_height = person["y2"] - person["y1"]
        is_rider = False
        for bike_detection in two_wheelers:
            bike = bike_detection["bbox"]
            bike_width = bike["x2"] - bike["x1"]
            bike_height = bike["y2"] - bike["y1"]
            horizontally_on_bike = (bike["x1"] - 0.10 * bike_width) <= person_center_x <= (bike["x2"] + 0.10 * bike_width)
            feet_at_bike = (bike["y1"] - 0.45 * bike_height) <= person["y2"] <= (bike["y2"] + 0.25 * bike_height)
            rider_sized = person_height >= max(20.0, 0.70 * bike_height)
            if horizontally_on_bike and feet_at_bike and rider_sized:
                is_rider = True
                break

        if not is_rider:
            filtered.append(detection)
    return filtered

def resolve_input_video(video_path_arg=None):
    """Dynamically resolve input video path for any uploaded video or default sample."""
    if video_path_arg and Path(video_path_arg).exists():
        return Path(video_path_arg)
    
    uploaded_files = sorted(
        list(UPLOADED_VIDEOS_DIR.glob("*.mp4")) + list(UPLOADED_VIDEOS_DIR.glob("*.avi")) + list(UPLOADED_VIDEOS_DIR.glob("*.mov")),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if uploaded_files:
        return uploaded_files[0]
        
    sample_file = PROJECT_ROOT / "app" / "static" / "uploads" / "raw" / "video1.mp4"
    if sample_file.exists():
        return sample_file
        
    raise FileNotFoundError("No traffic video file found in uploaded or sample directory. Please upload a video first.")

def detect_vehicles(video_path=None, model_path=None, conf_threshold=0.25):
    """
    Optimized DASEM Detection & Tracking Engine:
      - Uses Global Cached YOLOv8 model instance (loaded ONCE).
      - Single-Pass Streaming Video Architecture (zero frame array duplication in RAM).
      - Safe Frame Sampling Stride (N=2 for 20+ FPS surveillance videos to reduce YOLO inference calls by 50%).
      - Permanent Track-Level Class Locking & Unique Physical Vehicle Counting.
      - Saves JSON output to output/detections/<video_stem>.json and video to output/detected/<video_stem>_detected.mp4.
    """
    input_path = resolve_input_video(video_path)
    video_name = input_path.name
    stem = input_path.stem

    metadata = get_video_metadata(input_path)
    fps = metadata['fps']
    width = metadata['width']
    height = metadata['height']
    total_frames = metadata['frame_count']

    logger.info(f"DASEM Module 1 - Streamlined Detection & Persistent Tracking Engine")
    logger.info(f"Input video: {video_name} ({width}x{height} @ {fps} FPS, {total_frames} frames)")

    model = get_yolo_model(model_path)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"[ERROR] Cannot open VideoCapture for {input_path}")

    OUTPUT_DETECTED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    output_video_path = OUTPUT_DETECTED_DIR / f"{stem}_detected.mp4"
    output_json_path = OUTPUT_DETECTIONS_DIR / f"{stem}.json"

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    tracker = VehicleTracker(max_disappeared=30, max_distance_threshold=140.0, min_displacement_threshold=10.0, frame_width=width, frame_height=height)
    final_frames_json = []
    frame_idx = 0
    prev_gray = None

    last_raw_detections = []
    imgsz_perf = 640

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            timestamp = round(frame_idx / fps, 2)
            
            # Adaptive Scene Dynamism Check (runs YOLO on every frame during fast motion or close vehicle proximity)
            if last_raw_detections and tracker.objects:
                has_fast_motion = False
                has_close_proximity = False

                for tid, c_list in tracker.centroid_history.items():
                    if len(c_list) >= 2 and tid in tracker.objects:
                        dx_m = c_list[-1][0] - c_list[-2][0]
                        dy_m = c_list[-1][1] - c_list[-2][1]
                        if (dx_m * dx_m + dy_m * dy_m) > 100.0:  # 10.0^2
                            has_fast_motion = True
                            break

                if not has_fast_motion:
                    active_objs = list(tracker.objects.values())
                    n_objs = len(active_objs)
                    if n_objs >= 2:
                        for i_idx in range(n_objs):
                            pt1 = active_objs[i_idx]
                            for j_idx in range(i_idx + 1, n_objs):
                                pt2 = active_objs[j_idx]
                                dx_p = pt1[0] - pt2[0]
                                dy_p = pt1[1] - pt2[1]
                                if (dx_p * dx_p + dy_p * dy_p) < 7225.0:  # 85.0^2
                                    has_close_proximity = True
                                    break
                            if has_close_proximity:
                                break

                is_dynamic_scene = (has_fast_motion or has_close_proximity)
            else:
                is_dynamic_scene = False

            # Safe Adaptive Frame Sampling Stride (runs every 2nd frame in stable traffic, every 1st frame in dynamic scenes)
            should_run_yolo = (
                frame_idx == 1 or 
                frame_idx % 2 == 1 or 
                is_dynamic_scene or 
                not last_raw_detections or 
                model is None
            )

            if should_run_yolo and model is not None:
                frame_raw_detections = []
                try:
                    results = model.predict(frame, conf=conf_threshold, iou=0.45, imgsz=imgsz_perf, verbose=False)
                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                            conf = float(box.conf[0].cpu().numpy())
                            cls_id = int(box.cls[0].cpu().numpy())

                            if cls_id in COCO_CLASS_MAP:
                                cls_name = COCO_CLASS_MAP[cls_id]
                                bw = float(x2 - x1)
                                bh = float(y2 - y1)
                                ar = bw / max(1.0, bh)
                                area = bw * bh

                                if area < 800.0 or bw < 18.0 or bh < 18.0:
                                    continue

                                # Conservative auto-rickshaw classification (prevents relabeling genuine cars)
                                if cls_name == 'car' and 0.90 <= ar <= 1.15 and 4500 <= area <= 9000:
                                    cls_name = 'auto_rickshaw'

                                frame_raw_detections.append({
                                    "class": cls_name,
                                    "confidence": round(conf, 3),
                                    "bbox": {
                                        "x1": round(float(x1), 1),
                                        "y1": round(float(y1), 1),
                                        "x2": round(float(x2), 1),
                                        "y2": round(float(y2), 1)
                                    }
                                })
                    last_raw_detections = frame_raw_detections
                except Exception as e:
                    logger.error(f"YOLOv8 inference error on frame {frame_idx}: {e}")
                    frame_raw_detections = [dict(d) for d in last_raw_detections]
            else:
                # Extrapolate bounding boxes on skipped frames using track velocity for temporal motion continuity
                frame_raw_detections = []
                for d in last_raw_detections:
                    d_copy = {
                        "class": d["class"],
                        "confidence": d["confidence"],
                        "bbox": dict(d["bbox"])
                    }
                    tid = d.get('track_id')
                    if tid is not None and tid in tracker.centroid_history:
                        c_hist = tracker.centroid_history[tid]
                        if len(c_hist) >= 2:
                            vx = c_hist[-1][0] - c_hist[-2][0]
                            vy = c_hist[-1][1] - c_hist[-2][1]
                            v_mag = math.sqrt(vx**2 + vy**2)
                            if v_mag > 25.0:
                                vx = (vx / v_mag) * 25.0
                                vy = (vy / v_mag) * 25.0
                            
                            b = d_copy["bbox"]
                            b["x1"] = round(float(np.clip(b["x1"] + vx, 0.0, width)), 1)
                            b["y1"] = round(float(np.clip(b["y1"] + vy, 0.0, height)), 1)
                            b["x2"] = round(float(np.clip(b["x2"] + vx, 0.0, width)), 1)
                            b["y2"] = round(float(np.clip(b["y2"] + vy, 0.0, height)), 1)
                            d_copy['track_id'] = tid

                    frame_raw_detections.append(d_copy)

            # OpenCV Fallback Engine (if YOLO unavailable)
            if len(frame_raw_detections) == 0 and model is None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                if prev_gray is not None and prev_gray.shape == gray.shape:
                    frame_delta = cv2.absdiff(prev_gray, gray)
                    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                    thresh = cv2.dilate(thresh, None, iterations=2)
                    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for c in contours:
                        area = cv2.contourArea(c)
                        if area > 1200:
                            (x, y, w, h) = cv2.boundingRect(c)
                            aspect_ratio = float(w) / float(h)
                            
                            if area > 18000 and aspect_ratio > 1.6:
                                cls_name = 'bus'
                            elif area > 20000:
                                cls_name = 'truck'
                            elif aspect_ratio <= 1.25 and area < 9000:
                                cls_name = 'motorcycle'
                            elif 0.85 <= aspect_ratio <= 1.25:
                                cls_name = 'auto_rickshaw'
                            else:
                                cls_name = 'car'

                            frame_raw_detections.append({
                                "class": cls_name,
                                "confidence": 0.85,
                                "bbox": {
                                    "x1": round(float(x), 1),
                                    "y1": round(float(y), 1),
                                    "x2": round(float(x + w), 1),
                                    "y2": round(float(y + h), 1)
                                }
                            })
                prev_gray = gray

            frame_raw_detections = suppress_rider_person_detections(frame_raw_detections)

            # Update Tracker with single-pass stream
            tracker.update(frame_raw_detections)

            annotated_frame_dets = []
            for det in frame_raw_detections:
                tid = det.get('track_id')
                if tid is None:
                    continue

                locked_cls = tracker.locked_class.get(tid, det.get('class', 'car'))
                b = det['bbox']
                x1, y1, x2, y2 = int(b['x1']), int(b['y1']), int(b['x2']), int(b['y2'])

                annotated_frame_dets.append({
                    "track_id": tid,
                    "class": locked_cls,
                    "confidence": det['confidence'],
                    "bbox": det['bbox']
                })

                color = CLASS_COLOR_MAP.get(locked_cls, (0, 255, 0))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{locked_cls.upper()} #{tid}"
                cv2.putText(frame, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            final_frames_json.append({
                "frame_number": frame_idx,
                "timestamp": timestamp,
                "detections_count": len(annotated_frame_dets),
                "detections": annotated_frame_dets
            })

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

    # Track-level class stabilization & vehicle counting
    all_stabilized_map = tracker.get_all_stabilized_classes(moving_only=False)
    moving_stabilized_map = tracker.get_all_stabilized_classes(moving_only=True)

    unique_track_ids_by_class = defaultdict(set)
    for tid, locked_cls in moving_stabilized_map.items():
        unique_track_ids_by_class[locked_cls].add(tid)

    unique_counts = {
        "car": len(unique_track_ids_by_class.get("car", set())),
        "motorcycle": len(unique_track_ids_by_class.get("motorcycle", set())),
        "bus": len(unique_track_ids_by_class.get("bus", set())),
        "truck": len(unique_track_ids_by_class.get("truck", set())),
        "auto_rickshaw": len(unique_track_ids_by_class.get("auto_rickshaw", set())),
        "person": len(unique_track_ids_by_class.get("person", set())),
        "bicycle": len(unique_track_ids_by_class.get("bicycle", set()))
    }
    
    total_unique_vehicles = sum([unique_counts[c] for c in ['car', 'motorcycle', 'bus', 'truck', 'auto_rickshaw', 'bicycle']])
    total_unique_objects = sum(unique_counts.values())

    result_json = {
        "video": video_name,
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "total_unique_objects": total_unique_objects,
        "total_unique_vehicles": total_unique_vehicles,
        "unique_object_counts": unique_counts,
        "unique_counts": unique_counts,
        "moving_track_class_map": moving_stabilized_map,
        "track_class_map": all_stabilized_map,
        "frames": final_frames_json
    }

    save_json(result_json, output_json_path)

    logger.info(f"Frames processed: {total_frames}")
    logger.info(f"Total Unique Objects (Locked Classes): {total_unique_objects} {unique_counts}")
    logger.info(f"Output video saved to: {output_video_path}")
    logger.info(f"Detection JSON saved to: {output_json_path}")

    return {
        "json_path": str(output_json_path),
        "video_path": str(output_video_path),
        "total_unique_objects": total_unique_objects,
        "total_unique_vehicles": total_unique_vehicles,
        "unique_counts": unique_counts,
        "unique_object_counts": unique_counts,
        "track_class_map": all_stabilized_map
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DASEM Module 1 - Streamlined Detection & Persistent Tracking Engine")
    parser.add_argument("video_path", nargs="?", default=None, help="Path to input video file")
    parser.add_argument("--model", type=str, default=None, help="Path to YOLOv8 weights file")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    args = parser.parse_args()

    try:
        detect_vehicles(args.video_path, args.model, args.conf)
    except Exception as e:
        logger.error(f"Detection pipeline failed: {e}")
        sys.exit(1)
