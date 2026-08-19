import sys
import os
import argparse
import logging
import cv2
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
from src.tracker import VehicleTracker

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except Exception:
    HAS_ULTRALYTICS = False

logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
logger = logging.getLogger("DASEM.detect")

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

def resolve_input_video(video_path_arg=None):
    """
    Dynamically resolve input video path for any uploaded video or default sample.
    """
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

def detect_vehicles(video_path=None, model_path=None, conf_threshold=0.30):
    """
    Process ANY uploaded video file dynamically.
    Performs YOLOv8 object detection & ByteTrack persistent tracking with PERMANENT TRACK-LEVEL CLASS LOCKING.
    Supports Person, Car, Bus, Truck, Motorcycle, Auto Rickshaw, Bicycle.
    Filters partial edge fragments, tiny bounding boxes (< 1200px^2), and static roadside scrap.
    Counts ONLY REAL UNIQUE PHYSICAL OBJECTS across persistent Track IDs.
    Saves JSON output to output/detections/<video_stem>.json and video to output/detected/<video_stem>_detected.mp4.
    """
    input_path = resolve_input_video(video_path)
    video_name = input_path.name
    stem = input_path.stem

    metadata = get_video_metadata(input_path)
    fps = metadata['fps']
    width = metadata['width']
    height = metadata['height']
    total_frames = metadata['frame_count']

    logger.info(f"DASEM Module 1 - YOLOv8 Multi-Class Detection & Persistent Tracking")
    logger.info(f"Input video: {video_name} ({width}x{height} @ {fps} FPS, {total_frames} frames)")

    # Model Loading
    model = None
    if HAS_ULTRALYTICS:
        weights_to_try = []
        if model_path and Path(model_path).exists():
            weights_to_try.append(str(model_path))
        weights_to_try.extend(['yolov8n.pt', str(VEHICLE_MODEL_PATH), 'yolov8s.pt', 'yolov8m.pt'])

        for w in weights_to_try:
            try:
                model = YOLO(w)
                logger.info(f"Loaded YOLOv8 Detection Model: {w}")
                break
            except Exception as e:
                logger.warning(f"Could not load model {w}: {e}")

    if model is None:
        logger.warning("[WARNING] YOLO model file missing/unavailable. Utilizing OpenCV Motion & Contour Engine.")

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

    # Reset Tracker state freshly for each execution
    tracker = VehicleTracker(max_disappeared=30, max_distance_threshold=140.0, min_displacement_threshold=10.0)
    raw_frames_data = []
    total_raw_detections_count = 0
    frame_idx = 0
    prev_gray = None

    track_bbox_aspect_ratios = defaultdict(list)
    track_bbox_areas = defaultdict(list)

    imgsz_perf = 320 if max(width, height) > 640 else 416

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_idx += 1
        timestamp = round(frame_idx / fps, 2)
        frame_raw_detections = []

        # 1. Primary YOLOv8 Inference with performance optimization
        if model is not None:
            try:
                results = model.predict(frame, conf=conf_threshold, iou=0.50, imgsz=imgsz_perf, verbose=False)
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

                            # Partial-object box filter: skip boxes smaller than 1000px^2
                            if area < 1000.0 or bw < 20.0 or bh < 20.0:
                                continue

                            # Auto Rickshaw Detection Rule
                            if cls_name == 'car' and 0.85 <= ar <= 1.25 and 4500 <= area <= 14000:
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
            except Exception as e:
                logger.error(f"YOLOv8 inference error on frame {frame_idx}: {e}")

        # 2. OpenCV Fallback Engine (if YOLO unavailable)
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

        # 3. Update Tracker to Assign Persistent Track IDs
        tracker.update(frame_raw_detections)

        for det in frame_raw_detections:
            tid = det.get('track_id')
            if tid is not None:
                b = det['bbox']
                w = b['x2'] - b['x1']
                h = max(1.0, b['y2'] - b['y1'])
                track_bbox_aspect_ratios[tid].append(w / h)
                track_bbox_areas[tid].append(w * h)

        total_raw_detections_count += len(frame_raw_detections)
        raw_frames_data.append({
            "frame_number": frame_idx,
            "timestamp": timestamp,
            "frame_img": frame.copy(),
            "detections": frame_raw_detections
        })

    cap.release()

    # 4. TEMPORAL CLASS LOCKING & UNIQUE PHYSICAL TRACK OBJECT COUNTING
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

    # Build final frames JSON & write annotated video with LOCKED CLASSES
    final_frames_json = []

    for f_data in raw_frames_data:
        frame_num = f_data["frame_number"]
        timestamp = f_data["timestamp"]
        frame = f_data["frame_img"]

        stabilized_frame_detections = []
        for det in f_data["detections"]:
            tid = det.get("track_id")
            raw_cls = det.get("class", "car")
            conf = det.get("confidence", 0.85)
            b = det["bbox"]

            locked_cls = all_stabilized_map.get(tid, raw_cls) if tid is not None else raw_cls
            is_valid = tracker.is_valid_vehicle_track(tid) if tid is not None else True

            color = CLASS_COLOR_MAP.get(locked_cls, CLASS_COLOR_MAP['other'])
            x1, y1, x2, y2 = int(b['x1']), int(b['y1']), int(b['x2']), int(b['y2'])
            
            thickness = 2 if is_valid else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            motion_tag = "" if is_valid else " [Filtered]"
            label = f"{locked_cls.replace('_', ' ').capitalize()} #{tid if tid else 'N/A'}{motion_tag} {int(conf * 100)}%"
            cv2.putText(frame, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            stabilized_frame_detections.append({
                "class": locked_cls,
                "track_id": tid,
                "confidence": conf,
                "is_moving": is_valid,
                "bbox": b
            })

        out.write(frame)
        final_frames_json.append({
            "frame_number": frame_num,
            "timestamp": timestamp,
            "detections": stabilized_frame_detections
        })

    out.release()

    json_data = {
        "video": video_name,
        "fps": fps,
        "width": width,
        "height": height,
        "total_unique_vehicles": total_unique_vehicles,
        "total_unique_objects": total_unique_objects,
        "unique_object_counts": unique_counts,
        "track_class_map": {str(k): v for k, v in all_stabilized_map.items()},
        "moving_track_class_map": {str(k): v for k, v in moving_stabilized_map.items()},
        "frames": final_frames_json
    }

    save_json(json_data, output_json_path)

    logger.info(f"Frames processed: {frame_idx}")
    logger.info(f"Total Unique Objects (Locked Classes): {total_unique_objects} {unique_counts}")
    logger.info(f"Output video saved to: {output_video_path}")
    logger.info(f"Detection JSON saved to: {output_json_path}")

    return {
        "json_path": str(output_json_path),
        "video_path": str(output_video_path),
        "total_unique_objects": total_unique_objects,
        "total_unique_vehicles": total_unique_vehicles,
        "unique_counts": unique_counts,
        "track_class_map": all_stabilized_map,
        "moving_track_class_map": moving_stabilized_map
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DASEM Module 1 - YOLOv8 Object Detection & Persistent Tracking")
    parser.add_argument("video_path", nargs="?", default=None, help="Path to input video file")
    args = parser.parse_args()

    try:
        detect_vehicles(args.video_path)
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        sys.exit(1)
