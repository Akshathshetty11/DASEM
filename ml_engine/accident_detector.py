import os
import logging
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

logger = logging.getLogger(__name__)

# Strict Vehicle-Only COCO Class Mapping (Persons Excluded Completely)
COCO_CLASS_MAP = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

# Calibrated Vehicle Confidence Thresholds
CLASS_CONFIDENCE_THRESHOLDS = {
    'car': 0.35,
    'motorcycle': 0.35,
    'bus': 0.35,
    'truck': 0.35
}

class MultiObjectDetector:
    """
    Vehicle-Only Deep Learning Detector (YOLOv8m).
    Detects ONLY Cars, Buses, Trucks, and Motorcycles.
    Person detection is completely removed.
    """

    def __init__(self, weights_path=None, confidence_threshold=0.35, model_size='m'):
        self.weights_path = weights_path
        self.default_conf = confidence_threshold
        self.model = None

        weights_to_try = []
        if weights_path and os.path.exists(weights_path):
            weights_to_try.append(weights_path)
        weights_to_try.extend([f'yolov8{model_size}.pt', 'yolov8m.pt', 'yolov8s.pt', 'yolov8n.pt'])

        if HAS_ULTRALYTICS:
            for w in weights_to_try:
                try:
                    logger.info(f"[MultiObjectDetector] Loading YOLO model: {w}")
                    self.model = YOLO(w)
                    logger.info(f"[MultiObjectDetector] Successfully loaded vehicle YOLO model: {w}")
                    break
                except Exception as e:
                    logger.warning(f"[MultiObjectDetector] Could not load {w}: {e}")

    def reset_state(self):
        """Reset detector state."""
        pass

    def detect(self, frame):
        """
        Run vehicle detection on a single frame.

        Returns:
        - list of dicts: [ { 'bbox': [x1, y1, x2, y2], 'confidence': float, 'class_name': str, 'class_id': int } ]
        """
        if frame is None or self.model is None:
            return []

        detections = []

        try:
            # Soft NMS (iou=0.55) to prevent dropping adjacent vehicles
            results = self.model.predict(frame, conf=0.25, iou=0.55, verbose=False)
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())

                    # Strict Vehicle Mapping (Persons Excluded)
                    if cls_id in COCO_CLASS_MAP:
                        class_name = COCO_CLASS_MAP[cls_id]
                    else:
                        names = self.model.names
                        if isinstance(names, dict):
                            raw_name = str(names.get(cls_id, '')).lower().strip()
                        elif isinstance(names, list) and cls_id < len(names):
                            raw_name = str(names[cls_id]).lower().strip()
                        else:
                            raw_name = ''

                        if raw_name in ['motorcycle', 'motorbike', 'bike', 'bicycle']:
                            class_name = 'motorcycle'
                        elif raw_name == 'car':
                            class_name = 'car'
                        elif raw_name == 'bus':
                            class_name = 'bus'
                        elif raw_name == 'truck':
                            class_name = 'truck'
                        else:
                            continue # Ignore all non-vehicle classes (including persons)

                    required_conf = CLASS_CONFIDENCE_THRESHOLDS.get(class_name, self.default_conf)
                    
                    if conf >= required_conf:
                        detections.append({
                            'bbox': [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                            'confidence': round(conf, 3),
                            'class_name': class_name,
                            'class_id': cls_id
                        })

        except Exception as e:
            logger.error(f"[MultiObjectDetector] Error during vehicle inference: {e}")

        return detections


# Alias for backward compatibility
AccidentDetector = MultiObjectDetector
