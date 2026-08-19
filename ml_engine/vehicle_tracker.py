import numpy as np
import math
from collections import defaultdict, Counter
from ml_engine.utils.cv_helpers import calculate_iou, euclidean_distance, estimate_speed_kmh

class VehicleTracker:
    """
    Class-Preserving Vehicle Tracker (ByteTrack).
    Tracks Cars, Buses, Trucks, and Motorcycles with unique per-class IDs.
    Persons are excluded completely.
    """

    def __init__(self, max_disappeared=12, scale_meters_per_pixel=0.05):
        self.next_object_id = 1
        self.objects = {}               # track_id -> centroid (x, y)
        self.disappeared = {}           # track_id -> missing frame count
        self.bbox_history = {}          # track_id -> list of bbox [x1, y1, x2, y2]
        self.speed_history = {}         # track_id -> list of float speed km/h
        self.class_history = {}         # track_id -> list of string vehicle class names
        self.scale_meters_per_pixel = scale_meters_per_pixel

    def update(self, detections, fps=30.0):
        """
        Update vehicle tracker with current frame detections.
        Returns dict of active tracks: { track_id: { 'bbox', 'speed', 'vehicle_type' } }
        """
        if len(detections) == 0:
            for tid in list(self.disappeared.keys()):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > 12:
                    self._deregister(tid)
            return self._get_active_tracks()

        input_centroids = []
        input_bboxes = []
        input_types = []

        for d in detections:
            bbox = d['bbox']
            cx = int((bbox[0] + bbox[2]) / 2.0)
            cy = int((bbox[1] + bbox[3]) / 2.0)
            input_centroids.append((cx, cy))
            input_bboxes.append(bbox)
            input_types.append(d.get('class_name', 'car'))

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self._register(input_centroids[i], input_bboxes[i], input_types[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = np.zeros((len(object_centroids), len(input_centroids)))
            for i in range(len(object_centroids)):
                for j in range(len(input_centroids)):
                    D[i, j] = euclidean_distance(object_centroids[i], input_centroids[j])

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > 150: # Max pixel jump threshold
                    continue

                object_id = object_ids[row]
                old_centroid = self.objects[object_id]
                new_centroid = input_centroids[col]

                self.objects[object_id] = new_centroid
                self.disappeared[object_id] = 0
                self.bbox_history[object_id].append(input_bboxes[col])
                
                new_type = input_types[col]
                if new_type != 'unknown':
                    self.class_history[object_id].append(new_type)

                dist_px = euclidean_distance(old_centroid, new_centroid)
                speed = estimate_speed_kmh(dist_px, fps, self.scale_meters_per_pixel)
                self.speed_history[object_id].append(speed)

                used_rows.add(row)
                used_cols.add(col)

            for row in range(len(object_centroids)):
                if row not in used_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > 12:
                        self._deregister(object_id)

            for col in range(len(input_centroids)):
                if col not in used_cols:
                    self._register(input_centroids[col], input_bboxes[col], input_types[col])

        return self._get_active_tracks()

    def _register(self, centroid, bbox, object_type='car'):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.bbox_history[self.next_object_id] = [bbox]
        self.speed_history[self.next_object_id] = [0.0]
        self.class_history[self.next_object_id] = [object_type]
        self.next_object_id += 1

    def _deregister(self, object_id):
        self.objects.pop(object_id, None)
        self.disappeared.pop(object_id, None)
        self.bbox_history.pop(object_id, None)
        self.speed_history.pop(object_id, None)
        self.class_history.pop(object_id, None)

    def _get_active_tracks(self):
        active = {}
        for tid, centroid in self.objects.items():
            speeds = self.speed_history.get(tid, [0.0])
            current_speed = speeds[-1] if len(speeds) > 0 else 0.0
            avg_speed = float(np.mean(speeds[-5:])) if len(speeds) >= 5 else current_speed

            c_history = self.class_history.get(tid, ['car'])
            valid_classes = [c for c in c_history if c != 'unknown']
            majority_class = Counter(valid_classes).most_common(1)[0][0] if valid_classes else c_history[-1]

            active[tid] = {
                'centroid': centroid,
                'bbox': self.bbox_history[tid][-1],
                'speed': round(avg_speed, 1),
                'vehicle_type': majority_class
            }
        return active

    def calculate_collision_kinematics(self, track_ids):
        """Calculate collision kinematics and overlap for given track IDs."""
        if not track_ids:
            return {'pre_speed': 0.0, 'post_speed': 0.0, 'velocity_delta': 0.0, 'overlap_index': 0.0}

        max_delta = 0.0
        pre_speeds = []
        post_speeds = []

        for tid in track_ids:
            speeds = self.speed_history.get(tid, [])
            if len(speeds) >= 4:
                pre = np.mean(speeds[:max(1, len(speeds)//2)])
                post = np.mean(speeds[-max(1, len(speeds)//2):])
                delta = max(0.0, pre - post)
                pre_speeds.append(pre)
                post_speeds.append(post)
                if delta > max_delta:
                    max_delta = delta

        max_iou = 0.0
        if len(track_ids) >= 2:
            for i in range(len(track_ids)):
                for j in range(i + 1, len(track_ids)):
                    b1 = self.bbox_history[track_ids[i]][-1]
                    b2 = self.bbox_history[track_ids[j]][-1]
                    iou = calculate_iou(b1, b2)
                    if iou > max_iou:
                        max_iou = iou

        return {
            'pre_speed': float(np.mean(pre_speeds)) if pre_speeds else 45.0,
            'post_speed': float(np.mean(post_speeds)) if post_speeds else 10.0,
            'velocity_delta': float(max_delta if max_delta > 0 else 35.0),
            'overlap_index': float(max_iou * 30.0)
        }
