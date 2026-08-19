import cv2
import numpy as np
import math

# Vehicle & Hazard BGR Color Mapping
# Car -> Blue, Motorcycle -> Yellow, Bus -> Green, Truck -> Orange, Fire -> Red, Smoke -> Gray
CLASS_COLORS = {
    'car': (255, 128, 0),          # Vivid Blue (BGR)
    'motorcycle': (0, 255, 255),    # Vivid Yellow (BGR)
    'bike': (0, 255, 255),          # Vivid Yellow (BGR)
    'bicycle': (0, 255, 255),       # Vivid Yellow (BGR)
    'bus': (0, 255, 0),             # Vivid Green (BGR)
    'truck': (0, 140, 255),         # Vivid Orange (BGR)
    'fire': (0, 0, 255),            # Vivid Red (BGR)
    'smoke': (128, 128, 128),       # Gray (BGR)
    'accident': (0, 0, 255),        # Red
    'unknown': (200, 200, 200)      # Light Gray
}

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0
    return float(intersection_area / union_area)


def euclidean_distance(pt1, pt2):
    """Calculate 2D Euclidean distance between two points (x, y)."""
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def estimate_speed_kmh(pixel_distance, fps, scale_meters_per_pixel=0.05):
    """Estimate speed in km/h based on pixel movement."""
    if fps <= 0:
        return 0.0
    meters_per_second = (pixel_distance * scale_meters_per_pixel) * fps
    return round(meters_per_second * 3.6, 1)


def get_class_color(class_name):
    """Return BGR color tuple for a given vehicle/hazard class."""
    name_clean = str(class_name).lower().strip()
    return CLASS_COLORS.get(name_clean, (200, 200, 200))


def draw_detection_overlay(frame, bbox, label, confidence, track_id=None, severity_level='MINOR', is_fire=False, is_smoke=False):
    """
    Draw class-colored bounding box with class name, tracking ID, and confidence percentage.
    Examples:
      - 'Car #4 94%'
      - 'Bus #2 91%'
      - 'Truck #5 96%'
      - 'Motorcycle #7 93%'
      - 'Fire 95%'
      - 'Smoke 90%'
    """
    x1, y1, x2, y2 = map(int, bbox)
    color = get_class_color(label)
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    conf_pct = int(confidence * 100) if confidence <= 1.0 else int(confidence)
    
    clean_label = label.strip().capitalize()
    if clean_label.lower() in ['bike', 'bicycle']:
        clean_label = 'Motorcycle'

    if track_id is not None:
        text = f"{clean_label} #{track_id} {conf_pct}%"
    else:
        text = f"{clean_label} {conf_pct}%"
        
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    
    # Text background pill
    cv2.rectangle(frame, (x1, max(0, y1 - text_height - 8)), (x1 + text_width + 8, max(text_height + 8, y1)), color, -1)
    
    # White text inside background pill
    cv2.putText(frame, text, (x1 + 4, max(text_height + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return frame


def draw_hud_counter_overlay(frame, counts_dict, fire_detected=False, smoke_detected=False, severity_level=None):
    """
    Render live vehicle & hazard telemetry HUD box in top-left corner of video frames.
    """
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (350, 195), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (15, 15), (350, 195), (51, 65, 85), 1)

    cv2.putText(frame, "VEHICLE & HAZARD TELEMETRY", (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (59, 130, 246), 2)

    lines = [
        (f"Cars: {counts_dict.get('cars', 0)}", CLASS_COLORS['car']),
        (f"Buses: {counts_dict.get('buses', 0)}", CLASS_COLORS['bus']),
        (f"Trucks: {counts_dict.get('trucks', 0)}", CLASS_COLORS['truck']),
        (f"Motorcycles: {counts_dict.get('motorcycles', 0)}", CLASS_COLORS['motorcycle']),
        (f"Fire: {'YES' if fire_detected else 'NO'}", (0, 0, 255) if fire_detected else (148, 163, 184)),
        (f"Smoke: {'YES' if smoke_detected else 'NO'}", (128, 128, 128) if smoke_detected else (148, 163, 184))
    ]

    y_offset = 60
    for line_text, color in lines[:4]:
        cv2.putText(frame, line_text, (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        y_offset += 20

    cv2.putText(frame, lines[4][0], (190, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, lines[4][1], 2)
    cv2.putText(frame, lines[5][0], (190, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.45, lines[5][1], 2)

    if severity_level:
        sev_color = (0, 0, 255) if severity_level == 'Critical' else ((0, 140, 255) if severity_level == 'Major' else (0, 220, 255))
        cv2.putText(frame, f"Severity: {severity_level}", (25, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, sev_color, 2)

    return frame
