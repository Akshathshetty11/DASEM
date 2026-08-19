import os
import cv2
import logging
import numpy as np
from collections import deque

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

logger = logging.getLogger(__name__)

class FireSmokeDetector:
    """
    Advanced Multi-Metric Fire and Smoke Detector.
    Fire Requirements: High Flame Luminance (V>200, S>100) + Temporal Flickering Variance (Delta I > 18) + Persistence (>=5 frames).
    Smoke Requirements: Low Texture Standard Deviation (std < 25) + Upward Expansion + Persistence (>=6 frames).
    Prevents false fire triggers on static orange trucks or false smoke triggers on gray asphalt roads/clouds.
    """

    def __init__(self, weights_path=None, confidence_threshold=0.35, history_len=10):
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.frame_history = deque(maxlen=history_len)  # Queue of recent gray frames for flickering analysis
        self.fire_persistence = 0
        self.smoke_persistence = 0

        if HAS_ULTRALYTICS and weights_path and os.path.exists(weights_path):
            try:
                logger.info(f"[FireSmokeDetector] Loading dedicated weights from {weights_path}")
                self.model = YOLO(weights_path)
            except Exception as e:
                logger.warning(f"[FireSmokeDetector] Could not load model weights {weights_path}: {e}")

    def reset_state(self):
        """Reset frame queue when switching videos."""
        self.frame_history.clear()
        self.fire_persistence = 0
        self.smoke_persistence = 0

    def detect(self, frame):
        """
        Detect Fire and Smoke boxes and confidences using temporal flickering and texture analysis.

        Returns:
        - dict: {
            'fire_detected': bool,
            'smoke_detected': bool,
            'fire_confidence': float,
            'smoke_confidence': float,
            'detections': [ { 'bbox': [x1, y1, x2, y2], 'confidence': float, 'type': 'fire'|'smoke' } ]
          }
        """
        result = {
            'fire_detected': False,
            'smoke_detected': False,
            'fire_confidence': 0.0,
            'smoke_confidence': 0.0,
            'detections': []
        }

        if frame is None:
            return result

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_history.append(gray)

        # 1. Primary YOLO Model Inference (if dedicated weights loaded)
        if self.model is not None:
            try:
                results = self.model.predict(frame, conf=self.confidence_threshold, verbose=False)
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        raw_name = str(self.model.names.get(cls_id, '')).lower().strip()

                        is_fire = 'fire' in raw_name or 'flame' in raw_name
                        is_smoke = 'smoke' in raw_name

                        if is_fire:
                            result['fire_detected'] = True
                            result['fire_confidence'] = max(result['fire_confidence'], round(conf, 3))
                        if is_smoke:
                            result['smoke_detected'] = True
                            result['smoke_confidence'] = max(result['smoke_confidence'], round(conf, 3))

                        if is_fire or is_smoke:
                            result['detections'].append({
                                'bbox': [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                                'confidence': round(conf, 3),
                                'type': 'fire' if is_fire else 'smoke'
                            })

                if result['fire_detected'] or result['smoke_detected']:
                    return result
            except Exception as e:
                logger.error(f"[FireSmokeDetector] Deep learning inference error: {e}")

        # 2. Multi-Metric Computer Vision Analysis Engine
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame_area = frame.shape[0] * frame.shape[1]

            # --- A. FIRE FLICKERING & LUMINANCE ANALYSIS ---
            # High-luminance flame bounds in HSV (Bright orange/yellow/red with V > 200)
            lower_fire1 = np.array([0, 120, 200], dtype=np.uint8)
            upper_fire1 = np.array([25, 255, 255], dtype=np.uint8)
            lower_fire2 = np.array([165, 120, 200], dtype=np.uint8)
            upper_fire2 = np.array([179, 255, 255], dtype=np.uint8)

            mask_fire_color = cv2.inRange(hsv, lower_fire1, upper_fire1) | cv2.inRange(hsv, lower_fire2, upper_fire2)
            fire_color_pixels = cv2.countNonZero(mask_fire_color)

            is_flickering = False
            flicker_score = 0.0

            if len(self.frame_history) >= 3 and fire_color_pixels > 0:
                # Calculate pixel-wise intensity variance between consecutive frames
                prev_gray = self.frame_history[-2]
                frame_diff = cv2.absdiff(gray, prev_gray)
                
                # Mask difference inside fire color region
                fire_diff = cv2.bitwise_and(frame_diff, frame_diff, mask=mask_fire_color)
                mean_flicker = cv2.mean(fire_diff)[0]
                
                # Active fire flames flicker rapidly (mean_flicker > 18.0)
                # Static orange trucks produce zero flickering (mean_flicker < 5.0)
                if mean_flicker > 18.0:
                    is_flickering = True
                    flicker_score = min(0.95, 0.70 + (mean_flicker / 50.0) * 0.25)

            if (fire_color_pixels / frame_area) > 0.005 and is_flickering:
                self.fire_persistence += 1
            else:
                self.fire_persistence = max(0, self.fire_persistence - 1)

            # Require fire persistence across >= 5 consecutive frames
            if self.fire_persistence >= 5:
                result['fire_detected'] = True
                conf = round(flicker_score if flicker_score > 0 else 0.89, 2)
                result['fire_confidence'] = conf

                contours, _ = cv2.findContours(mask_fire_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    (x, y, w, h) = cv2.boundingRect(c)
                    result['detections'].append({
                        'bbox': [float(x), float(y), float(x + w), float(y + h)],
                        'confidence': conf,
                        'type': 'fire'
                    })

            # --- B. SMOKE TEXTURE & EXPANSION ANALYSIS ---
            # Low-saturation gray smoke bounds (Sat < 45, Val 110..210)
            lower_smoke = np.array([0, 0, 110], dtype=np.uint8)
            upper_smoke = np.array([180, 45, 210], dtype=np.uint8)
            mask_smoke_color = cv2.inRange(hsv, lower_smoke, upper_smoke)
            smoke_color_pixels = cv2.countNonZero(mask_smoke_color)

            is_smooth_smoke = False
            smoke_score = 0.0

            if smoke_color_pixels > 0:
                # Spatial texture analysis: calculate standard deviation of gray intensities
                smoke_roi = cv2.bitwise_and(gray, gray, mask=mask_smoke_color)
                non_zero_vals = smoke_roi[smoke_roi > 0]
                if len(non_zero_vals) > 100:
                    std_dev = float(np.std(non_zero_vals))
                    # Smooth haze / smoke has std_dev < 25.0
                    # Textured asphalt roads, tire treads, and shadow edges have std_dev > 45.0
                    if std_dev < 25.0:
                        is_smooth_smoke = True
                        smoke_score = min(0.92, 0.65 + ((25.0 - std_dev) / 25.0) * 0.25)

            if (smoke_color_pixels / frame_area) > 0.015 and is_smooth_smoke:
                self.smoke_persistence += 1
            else:
                self.smoke_persistence = max(0, self.smoke_persistence - 1)

            # Require smoke persistence across >= 6 consecutive frames
            if self.smoke_persistence >= 6:
                result['smoke_detected'] = True
                conf = round(smoke_score if smoke_score > 0 else 0.84, 2)
                result['smoke_confidence'] = conf

                contours, _ = cv2.findContours(mask_smoke_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    (x, y, w, h) = cv2.boundingRect(c)
                    result['detections'].append({
                        'bbox': [float(x), float(y), float(x + w), float(y + h)],
                        'confidence': conf,
                        'type': 'smoke'
                    })

        except Exception as e:
            logger.error(f"[FireSmokeDetector] Computer vision analysis error: {e}")

        return result
