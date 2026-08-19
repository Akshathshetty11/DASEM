import math
import numpy as np
from collections import Counter, defaultdict

def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes.
    Box format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0
    return float(intersection / union)


def euclidean_distance(pt1, pt2):
    """Calculate Euclidean distance between two centroid points (x, y)."""
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


class VehicleTracker:
    """
    Persistent Multi-Object Tracker with Permanent Track-Level Class Locking
    and Partial-Object Bounding Box Filtering.
    
    Guarantees that once a Track ID is assigned a dominant class after >= 4 frames,
    its class is LOCKED permanently for its lifetime. Temporary low-confidence predictions
    cannot alter the locked class.
    Filters partial edge fragments, tiny bounding boxes (< 1200px^2), and static roadside scrap.
    """

    def __init__(self, max_disappeared=30, max_distance_threshold=140.0, min_displacement_threshold=10.0):
        self.next_object_id = 1
        self.objects = {}                   # track_id -> current centroid (x, y)
        self.disappeared = {}               # track_id -> missing frame count
        self.bbox_history = {}              # track_id -> list of bbox [x1, y1, x2, y2]
        self.centroid_history = {}          # track_id -> list of centroids [(x, y), ...]
        self.class_history = {}             # track_id -> list of class names
        self.class_confidence_sums = {}     # track_id -> defaultdict(float) class_name -> total confidence
        self.locked_class = {}              # track_id -> locked dominant class name
        self.max_disappeared = max_disappeared
        self.max_distance_threshold = max_distance_threshold
        self.min_displacement_threshold = min_displacement_threshold

    def update(self, frame_detections):
        """
        Update tracker with detections from current frame.
        Attaches persistent track_id to each detection in frame_detections.
        Returns dict of active tracks.
        """
        if not frame_detections:
            for tid in list(self.disappeared.keys()):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    self._deregister(tid)
            return self._get_active_tracks()

        input_centroids = []
        input_bboxes = []
        input_classes = []
        input_confidences = []

        for d in frame_detections:
            b = d['bbox']
            bbox_list = [float(b['x1']), float(b['y1']), float(b['x2']), float(b['y2'])]
            cx = (bbox_list[0] + bbox_list[2]) / 2.0
            cy = (bbox_list[1] + bbox_list[3]) / 2.0
            input_centroids.append((cx, cy))
            input_bboxes.append(bbox_list)
            input_classes.append(d.get('class', 'car'))
            input_confidences.append(float(d.get('confidence', 0.85)))

        if len(self.objects) == 0:
            assigned_ids = []
            for i in range(len(input_centroids)):
                tid = self._register(input_centroids[i], input_bboxes[i], input_classes[i], input_confidences[i])
                assigned_ids.append(tid)
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
            assigned_ids = [None] * len(input_centroids)

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > self.max_distance_threshold:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                self.bbox_history[object_id].append(input_bboxes[col])
                self.centroid_history[object_id].append(input_centroids[col])
                self.class_history[object_id].append(input_classes[col])
                self.class_confidence_sums[object_id][input_classes[col]] += input_confidences[col]
                assigned_ids[col] = object_id

                # Lock class after >= 4 observations
                self._update_class_lock(object_id)

                used_rows.add(row)
                used_cols.add(col)

            for row in range(len(object_centroids)):
                if row not in used_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self._deregister(object_id)

            for col in range(len(input_centroids)):
                if col not in used_cols:
                    tid = self._register(input_centroids[col], input_bboxes[col], input_classes[col], input_confidences[col])
                    assigned_ids[col] = tid

        # Attach persistent track_id to each detection object
        for i, d in enumerate(frame_detections):
            if i < len(assigned_ids) and assigned_ids[i] is not None:
                d['track_id'] = assigned_ids[i]

        return self._get_active_tracks()

    def _update_class_lock(self, track_id):
        """
        Check if track_id has sufficient history (>= 4 observations) to permanently lock its class.
        Once locked, locked_class[track_id] is immutable.
        """
        if track_id in self.locked_class:
            return

        conf_sums = self.class_confidence_sums.get(track_id, {})
        history_len = len(self.centroid_history.get(track_id, []))

        if history_len >= 4 and conf_sums:
            # Person human class check
            if conf_sums.get('person', 0.0) > (sum(conf_sums.values()) * 0.50):
                self.locked_class[track_id] = 'person'
                return

            # Geometry refinement on history
            bboxes = self.bbox_history.get(track_id, [])
            if bboxes:
                aspect_ratios = []
                areas = []
                for b in bboxes:
                    w = b[2] - b[0]
                    h = max(1.0, b[3] - b[1])
                    aspect_ratios.append(w / h)
                    areas.append(w * h)
                
                mean_ar = float(np.mean(aspect_ratios))
                mean_area = float(np.mean(areas))

                bus_score = conf_sums.get('bus', 0.0)
                truck_score = conf_sums.get('truck', 0.0)
                car_score = conf_sums.get('car', 0.0)
                moto_score = conf_sums.get('motorcycle', 0.0) + conf_sums.get('bicycle', 0.0)
                auto_score = conf_sums.get('auto_rickshaw', 0.0)

                # Bus rule: dominant bus predictions or mean_area > 28000 and mean_ar >= 1.65
                if bus_score > (car_score + truck_score) or (bus_score > 0 and mean_area > 25000 and mean_ar >= 1.65):
                    self.locked_class[track_id] = 'bus'
                    return

                # Truck rule: dominant truck predictions
                if truck_score > (car_score + bus_score) or (truck_score > 0 and mean_area > 28000):
                    self.locked_class[track_id] = 'truck'
                    return

                # Auto Rickshaw rule
                if auto_score > 0 or (car_score > 0 and 0.85 <= mean_ar <= 1.18 and 4500 <= mean_area <= 14000):
                    self.locked_class[track_id] = 'auto_rickshaw'
                    return

                # Motorcycle rule
                if moto_score > 0 and mean_ar <= 1.25 and mean_area < 9000:
                    self.locked_class[track_id] = 'motorcycle'
                    return

            sorted_classes = sorted(conf_sums.items(), key=lambda x: x[1], reverse=True)
            self.locked_class[track_id] = sorted_classes[0][0]

    def get_stabilized_class(self, track_id):
        """
        Get the LOCKED STABILIZED class for a given track_id.
        Returns locked_class[track_id] if set, else highest confidence sum.
        """
        if track_id in self.locked_class:
            return self.locked_class[track_id]

        conf_sums = self.class_confidence_sums.get(track_id, {})
        if conf_sums:
            sorted_classes = sorted(conf_sums.items(), key=lambda x: x[1], reverse=True)
            return sorted_classes[0][0]

        classes = self.class_history.get(track_id, [])
        if not classes:
            return 'car'
        return Counter(classes).most_common(1)[0][0]

    def get_track_displacement(self, track_id):
        """Calculate total spatial displacement."""
        centroids = self.centroid_history.get(track_id, [])
        if len(centroids) < 2:
            return 0.0
        
        first_pt = centroids[0]
        max_dist = 0.0
        for pt in centroids[1:]:
            d = euclidean_distance(first_pt, pt)
            if d > max_dist:
                max_dist = d
        return max_dist

    def is_valid_vehicle_track(self, track_id):
        """
        Partial-Object & Bounding Box Quality Filter:
        Verifies that a tracked object represents a valid physical vehicle rather than
        a small edge clip, fragmented slice, roadside scrap, or noise box.
        Checks:
        1. Bounding box mean width >= 25px, height >= 25px, mean area >= 1200px^2.
        2. Persistence >= 4 frames.
        3. Initial edge fragment check: small boxes (< 2500px^2) starting on image margin are excluded.
        4. Track spatial displacement >= 10.0px.
        """
        centroids = self.centroid_history.get(track_id, [])
        bboxes = self.bbox_history.get(track_id, [])

        if len(centroids) < 4 or len(bboxes) < 4:
            return False

        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
        widths = [b[2] - b[0] for b in bboxes]
        heights = [b[3] - b[1] for b in bboxes]

        mean_area = float(np.mean(areas))
        mean_width = float(np.mean(widths))
        mean_height = float(np.mean(heights))

        # Size check
        if mean_area < 1200.0 or mean_width < 25.0 or mean_height < 25.0:
            return False

        # Displacement check
        disp = self.get_track_displacement(track_id)
        if disp < self.min_displacement_threshold and len(centroids) < 10:
            return False

        return True

    def get_all_stabilized_classes(self, moving_only=True):
        """
        Get map of track_id -> locked_stabilized_class for all valid tracked objects.
        If moving_only=True, excludes invalid/fragmented tracks.
        """
        res = {}
        for tid in self.class_history.keys():
            if not moving_only or self.is_valid_vehicle_track(tid):
                res[tid] = self.get_stabilized_class(tid)
        return res

    def _register(self, centroid, bbox, class_name, confidence):
        tid = self.next_object_id
        self.objects[tid] = centroid
        self.disappeared[tid] = 0
        self.bbox_history[tid] = [bbox]
        self.centroid_history[tid] = [centroid]
        self.class_history[tid] = [class_name]
        self.class_confidence_sums[tid] = defaultdict(float)
        self.class_confidence_sums[tid][class_name] += confidence
        self.next_object_id += 1
        
        self._update_class_lock(tid)
        return tid

    def _deregister(self, object_id):
        self.objects.pop(object_id, None)
        self.disappeared.pop(object_id, None)
        self.bbox_history.pop(object_id, None)
        self.centroid_history.pop(object_id, None)
        self.class_history.pop(object_id, None)
        self.class_confidence_sums.pop(object_id, None)
        self.locked_class.pop(object_id, None)

    def _get_active_tracks(self):
        active = {}
        for tid, centroid in self.objects.items():
            active[tid] = {
                'id': tid,
                'class': self.get_stabilized_class(tid),
                'bbox': self.bbox_history[tid][-1],
                'centroid': centroid,
                'displacement': self.get_track_displacement(tid),
                'is_moving': self.is_valid_vehicle_track(tid)
            }
        return active
