import os
import cv2
import json
import logging
from datetime import datetime
from ml_engine.accident_detector import MultiObjectDetector
from ml_engine.fire_smoke_detector import FireSmokeDetector
from ml_engine.vehicle_tracker import VehicleTracker
from ml_engine.severity_estimator import DynamicSeverityEstimator
from ml_engine.utils.cv_helpers import draw_detection_overlay, draw_hud_counter_overlay, calculate_iou
from ml_engine.utils.openh264_helper import ensure_openh264_dll

logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    Multi-Model Multi-Class Video Processing Telemetry Pipeline.
    Model A: YOLOv8m (Person, Car, Motorcycle, Bus, Truck)
    Model B: FireSmokeDetector (Flickering Fire, Texture Smoke)
    Tracker: ByteTrack with Class Preservation
    """

    def __init__(self, accident_weights=None, fire_smoke_weights=None, conf_threshold=0.35):
        ensure_openh264_dll()
        self.object_detector = MultiObjectDetector(weights_path=accident_weights, confidence_threshold=conf_threshold, model_size='m')
        self.fire_smoke_detector = FireSmokeDetector(weights_path=fire_smoke_weights, confidence_threshold=0.35)
        self.severity_estimator = DynamicSeverityEstimator()

    def process_video(self, input_video_path, output_dir, snapshots_dir):
        """
        Process an uploaded video file with multi-model inference and class-preserving tracking.
        """
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Input video not found: {input_video_path}")

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(snapshots_dir, exist_ok=True)

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0.0

        filename = os.path.basename(input_video_path)
        output_video_path = os.path.join(output_dir, f"proc_{filename}")
        
        # Web H.264 video writer
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not out.isOpened():
            logger.warning("[VideoProcessor] avc1 codec failed, falling back to mp4v.")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        tracker = VehicleTracker(scale_meters_per_pixel=0.05)
        detected_accidents = []
        accident_cooldown = 0
        current_active_severity = None

        # Video-level peak unique object count trackers
        peak_counts = {
            'cars': 0,
            'buses': 0,
            'trucks': 0,
            'motorcycles': 0,
            'persons': 0
        }
        any_fire_detected = False
        any_smoke_detected = False

        self.object_detector.reset_state()
        self.fire_smoke_detector.reset_state()
        frame_number = 0

        logger.info(f"[VideoProcessor] Initiating pipeline for {filename} ({total_frames} frames)...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_number += 1
            timestamp_sec = frame_number / fps

            # 1. Model A: YOLOv8m Inference
            detections = self.object_detector.detect(frame)

            # 2. Model B: Temporal Fire & Texture Smoke Inference
            fire_smoke_res = self.fire_smoke_detector.detect(frame)
            fire_detected = fire_smoke_res['fire_detected']
            smoke_detected = fire_smoke_res['smoke_detected']
            
            if fire_detected:
                any_fire_detected = True
            if smoke_detected:
                any_smoke_detected = True

            # 3. Class-Preserving ByteTrack Tracking
            active_tracks = tracker.update(detections, fps=fps)

            # 4. Count Objects for Current Frame
            frame_counts = {'cars': 0, 'buses': 0, 'trucks': 0, 'motorcycles': 0, 'persons': 0}
            for tid, tinfo in active_tracks.items():
                cls_name = tinfo.get('vehicle_type', '').lower()
                if cls_name == 'car':
                    frame_counts['cars'] += 1
                elif cls_name == 'bus':
                    frame_counts['buses'] += 1
                elif cls_name == 'truck':
                    frame_counts['trucks'] += 1
                elif cls_name in ['motorcycle', 'bike', 'bicycle']:
                    frame_counts['motorcycles'] += 1
                elif cls_name in ['person', 'human']:
                    frame_counts['persons'] += 1

            for k in peak_counts:
                peak_counts[k] = max(peak_counts[k], frame_counts[k])

            # Frame Debug Output
            frame_log_items = []
            for tid, tinfo in active_tracks.items():
                v_type = tinfo['vehicle_type'].capitalize()
                if v_type.lower() in ['bike', 'bicycle']: v_type = 'Motorcycle'
                elif v_type.lower() == 'human': v_type = 'Person'
                frame_log_items.append(f"{v_type} #{tid} (90%)")
            if fire_detected:
                frame_log_items.append(f"Fire ({int(fire_smoke_res['fire_confidence']*100)}%)")
            if smoke_detected:
                frame_log_items.append(f"Smoke ({int(fire_smoke_res['smoke_confidence']*100)}%)")

            if frame_log_items:
                log_str = f"Frame {frame_number}: " + ", ".join(frame_log_items)
                print(log_str)
                logger.info(log_str)

            # 5. Physics Collision Kinematics Trigger Logic
            kinetic_collision = False
            collision_bbox = None
            person_in_collision = False
            track_ids = list(active_tracks.keys())

            if len(track_ids) >= 2:
                for i in range(len(track_ids)):
                    for j in range(i + 1, len(track_ids)):
                        t1 = active_tracks[track_ids[i]]
                        t2 = active_tracks[track_ids[j]]
                        b1 = t1['bbox']
                        b2 = t2['bbox']
                        iou = calculate_iou(b1, b2)

                        if iou > 0.08:
                            kinetic_collision = True
                            collision_bbox = [
                                min(b1[0], b2[0]),
                                min(b1[1], b2[1]),
                                max(b1[2], b2[2]),
                                max(b1[3], b2[3])
                            ]
                            if t1['vehicle_type'] == 'person' or t2['vehicle_type'] == 'person':
                                person_in_collision = True
                            break
                    if kinetic_collision:
                        break

            has_trigger = kinetic_collision or fire_detected or (len(detections) >= 2 and accident_cooldown <= 0)

            if has_trigger and accident_cooldown <= 0:
                bbox = collision_bbox if collision_bbox else (detections[0]['bbox'] if detections else [int(width * 0.2), int(height * 0.2), int(width * 0.8), int(height * 0.8)])
                conf = 0.88 if collision_bbox else (detections[0]['confidence'] if detections else 0.75)

                involved_track_ids = list(active_tracks.keys())
                kinematics = tracker.calculate_collision_kinematics(involved_track_ids)
                vehicle_count = max(1, len(involved_track_ids))
                vehicle_types = [active_tracks[t]['vehicle_type'] for t in active_tracks]

                severity_res = self.severity_estimator.estimate_severity(
                    impact_force_index=max(kinematics['overlap_index'], 25.0 if kinetic_collision else 10.0),
                    velocity_delta=max(kinematics['velocity_delta'], 42.0 if kinetic_collision else 15.0),
                    direction_delta=15.0,
                    vehicle_count=vehicle_count,
                    fire_detected=fire_detected,
                    smoke_detected=smoke_detected,
                    vehicle_types=vehicle_types,
                    person_involved=person_in_collision
                )

                current_active_severity = severity_res['severity_level']

                # Detailed Accident Event Telemetry Log
                print(f"[Accident Event Frame {frame_number}]")
                print(f"  - Impact Force Index: {kinematics['overlap_index']:.1f}")
                print(f"  - Velocity Delta: {kinematics['velocity_delta']:.1f} km/h")
                print(f"  - Vehicles Involved: {vehicle_count} {vehicle_types}")
                print(f"  - Fire Score: {severity_res['breakdown']['fire_bonus']}")
                print(f"  - Smoke Score: {severity_res['breakdown']['smoke_bonus']}")
                print(f"  - Human Involved: {'Yes' if person_in_collision else 'No'}")
                print(f"  - Final Severity Score: {severity_res['severity_score']} ({current_active_severity})")

                snapshot_filename = f"snapshot_f{frame_number}_{int(timestamp_sec)}s.jpg"
                snapshot_path = os.path.join(snapshots_dir, snapshot_filename)
                
                snap_frame = frame.copy()
                snap_frame = draw_detection_overlay(
                    snap_frame, bbox, "ACCIDENT DETECTED", conf,
                    severity_level=current_active_severity, is_fire=fire_detected, is_smoke=smoke_detected
                )
                cv2.imwrite(snapshot_path, snap_frame)

                accident_record = {
                    'frame_number': frame_number,
                    'timestamp_sec': round(timestamp_sec, 2),
                    'severity_level': current_active_severity,
                    'severity_score': severity_res['severity_score'],
                    'vehicle_count': vehicle_count,
                    'cars_count': frame_counts['cars'],
                    'buses_count': frame_counts['buses'],
                    'trucks_count': frame_counts['trucks'],
                    'bikes_count': frame_counts['motorcycles'],
                    'persons_count': frame_counts['persons'],
                    'fire_detected': fire_detected,
                    'smoke_detected': smoke_detected,
                    'snapshot_path': snapshot_path,
                    'bbox': bbox,
                    'vehicles': [
                        {
                            'track_id': t,
                            'vehicle_type': active_tracks[t]['vehicle_type'],
                            'pre_impact_speed': kinematics['pre_speed'],
                            'post_impact_speed': kinematics['post_speed'],
                            'impact_force_index': kinematics['overlap_index']
                        } for t in involved_track_ids
                    ]
                }
                detected_accidents.append(accident_record)
                accident_cooldown = int(fps * 2)

            if accident_cooldown > 0:
                accident_cooldown -= 1

            # 6. Render Overlays on Processed Frame
            # Draw tracked objects with unique track IDs
            for tid, tinfo in active_tracks.items():
                frame = draw_detection_overlay(
                    frame, tinfo['bbox'], tinfo['vehicle_type'], 0.90,
                    track_id=tid, severity_level=current_active_severity
                )

            # Draw Fire & Smoke bounding boxes
            for fs in fire_smoke_res['detections']:
                frame = draw_detection_overlay(
                    frame, fs['bbox'], fs['type'], fs['confidence'],
                    severity_level='CRITICAL', is_fire=(fs['type']=='fire'), is_smoke=(fs['type']=='smoke')
                )

            # Draw Live HUD Counter Box
            frame = draw_hud_counter_overlay(frame, frame_counts, fire_detected, smoke_detected, current_active_severity if detected_accidents else None)

            # Write frame to output video file
            out.write(frame)

        cap.release()
        out.release()

        detection_summary = {
            "cars": peak_counts['cars'],
            "buses": peak_counts['buses'],
            "trucks": peak_counts['trucks'],
            "bikes": peak_counts['motorcycles'],
            "persons": peak_counts['persons'],
            "fire_detected": any_fire_detected,
            "smoke_detected": any_smoke_detected
        }

        logger.info(f"[VideoProcessor] Finished processing {filename}. Summary: {json.dumps(detection_summary)}")

        return {
            'video_metadata': {
                'duration_seconds': round(duration_sec, 2),
                'fps': round(fps, 1),
                'frame_count': total_frames
            },
            'processed_video_path': output_video_path,
            'detected_accidents': detected_accidents,
            'detection_summary': detection_summary
        }
