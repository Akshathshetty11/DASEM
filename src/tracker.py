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
    Persistent Multi-Object Tracker with Class Hysteresis & Temporal Class Stabilization.
    
    Guarantees that once a Track ID is assigned a dominant class after >= 6 frames,
    its class is locked and stabilized. Isolated YOLO prediction noise or single-frame fluctuations
    cannot alter the locked class. Class hysteresis prevents label flickering.
    Filters partial edge fragments, tiny bounding boxes (< 800px^2), and static roadside scrap.
    Explicitly preserves PERSON detections entering from the side.
    """

    def __init__(self, max_disappeared=30, max_distance_threshold=140.0, min_displacement_threshold=10.0, frame_width=640, frame_height=352):
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
        self.frame_width = frame_width
        self.frame_height = frame_height

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
            N = len(object_ids)
            M = len(input_centroids)
            cost_matrix = np.full((N, M), 1e6, dtype=float)

            for i in range(N):
                tid = object_ids[i]
                centroids = self.centroid_history.get(tid, [])
                bboxes = self.bbox_history.get(tid, [])
                d_count = self.disappeared.get(tid, 0)

                last_cx, last_cy = centroids[-1]
                last_b = bboxes[-1]
                last_w = max(1.0, last_b[2] - last_b[0])
                last_h = max(1.0, last_b[3] - last_b[1])
                last_area = last_w * last_h
                last_ar = last_w / last_h
                track_cls = self.get_stabilized_class(tid)

                # Estimate trajectory motion vector (vx, vy) per frame
                if len(centroids) >= 3:
                    vx = (centroids[-1][0] - centroids[-3][0]) / 2.0
                    vy = (centroids[-1][1] - centroids[-3][1]) / 2.0
                elif len(centroids) == 2:
                    vx = centroids[-1][0] - centroids[-2][0]
                    vy = centroids[-1][1] - centroids[-2][1]
                else:
                    vx, vy = 0.0, 0.0

                v_mag = math.sqrt(vx**2 + vy**2)
                if v_mag > 40.0:
                    vx = (vx / v_mag) * 40.0
                    vy = (vy / v_mag) * 40.0

                pred_cx = last_cx + vx * (1 + d_count)
                pred_cy = last_cy + vy * (1 + d_count)
                pred_bbox = [pred_cx - last_w / 2.0, pred_cy - last_h / 2.0, pred_cx + last_w / 2.0, pred_cy + last_h / 2.0]

                effective_max_dist = max(self.max_distance_threshold, last_w * 1.2, last_h * 1.2)
                gate_thresh_sq = (effective_max_dist * 1.6) ** 2

                for j in range(M):
                    cand_cx, cand_cy = input_centroids[j]

                    # Stage 1: Cheap Spatial Centroid Gating (Squared distance check avoids math.sqrt, IoU, and shape overhead)
                    dx_pred = pred_cx - cand_cx
                    dy_pred = pred_cy - cand_cy
                    dist_pred_sq = dx_pred * dx_pred + dy_pred * dy_pred

                    dx_last = last_cx - cand_cx
                    dy_last = last_cy - cand_cy
                    dist_last_sq = dx_last * dx_last + dy_last * dy_last

                    if dist_pred_sq > gate_thresh_sq and dist_last_sq > gate_thresh_sq:
                        continue

                    # Stage 2: Detailed Signal & Cost Evaluation
                    cand_b = input_bboxes[j]
                    cand_cls = input_classes[j]

                    cand_w = max(1.0, cand_b[2] - cand_b[0])
                    cand_h = max(1.0, cand_b[3] - cand_b[1])
                    cand_area = cand_w * cand_h
                    cand_ar = cand_w / cand_h

                    dist_pred = math.sqrt(dist_pred_sq)
                    dist_last = math.sqrt(dist_last_sq)
                    spatial_dist = min(dist_pred, 0.65 * dist_pred + 0.35 * dist_last)

                    if spatial_dist > effective_max_dist * 1.6 and dist_last > effective_max_dist * 1.6:
                        continue

                    # 2. Bounding-Box IoU Signal
                    iou_last = calculate_iou(last_b, cand_b)
                    iou_pred = calculate_iou(pred_bbox, cand_b)
                    eff_iou = max(iou_last, iou_pred)

                    # 3. Shape & Size Consistency
                    area_diff = abs(last_area - cand_area) / max(1.0, max(last_area, cand_area))
                    ar_diff = abs(last_ar - cand_ar) / max(0.1, max(last_ar, cand_ar))
                    size_cost = (area_diff * 30.0) + (ar_diff * 15.0)

                    # 4. Class Compatibility Signal
                    if track_cls == cand_cls:
                        class_cost = 0.0
                    elif (track_cls == 'person') != (cand_cls == 'person'):
                        class_cost = 200.0  # Heavy gating for person vs vehicle
                    else:
                        class_cost = 18.0   # Minor vehicle label noise penalty

                    # 5. Motion Trajectory Direction Consistency
                    direction_penalty = 0.0
                    if v_mag > 3.0:
                        dx = cand_cx - last_cx
                        dy = cand_cy - last_cy
                        d_mag = math.sqrt(dx**2 + dy**2)
                        if d_mag > 3.0:
                            dot = (vx * dx + vy * dy) / (v_mag * d_mag)
                            if dot < -0.4:
                                direction_penalty = 25.0

                    # Combined Cost
                    total_cost = spatial_dist - (eff_iou * 70.0) + size_cost + class_cost + direction_penalty + (d_count * 5.0)
                    cost_matrix[i, j] = total_cost

            # Solve 1-to-1 optimal assignment
            try:
                from scipy.optimize import linear_sum_assignment
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                matched_indices = list(zip(row_ind, col_ind))
            except Exception:
                matched_indices = []
                cost_pairs = []
                for i in range(N):
                    for j in range(M):
                        if cost_matrix[i, j] < 1e5:
                            cost_pairs.append((cost_matrix[i, j], i, j))
                cost_pairs.sort(key=lambda x: x[0])
                used_r = set()
                used_c = set()
                for c_val, r, c in cost_pairs:
                    if r not in used_r and c not in used_c:
                        matched_indices.append((r, c))
                        used_r.add(r)
                        used_c.add(c)

            used_rows = set()
            used_cols = set()
            assigned_ids = [None] * len(input_centroids)

            for row, col in matched_indices:
                cost = cost_matrix[row, col]
                if cost >= 1e5:
                    continue

                object_id = object_ids[row]
                last_b = self.bbox_history[object_id][-1]
                last_w = max(1.0, last_b[2] - last_b[0])
                last_h = max(1.0, last_b[3] - last_b[1])
                effective_max_dist = max(self.max_distance_threshold, last_w * 1.2, last_h * 1.2)

                dist_last = euclidean_distance(self.objects[object_id], input_centroids[col])
                if cost > (effective_max_dist + 40.0) and dist_last > effective_max_dist * 1.25:
                    continue

                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                self.bbox_history[object_id].append(input_bboxes[col])
                self.centroid_history[object_id].append(input_centroids[col])
                self.class_history[object_id].append(input_classes[col])
                self.class_confidence_sums[object_id][input_classes[col]] += input_confidences[col]
                assigned_ids[col] = object_id

                self._update_class_lock(object_id)

                used_rows.add(row)
                used_cols.add(col)

            for row in range(len(object_ids)):
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
        Class Hysteresis Lock Engine:
        Evaluates cumulative confidence-weighted scores over track history.
        Locks the dominant class after >= 6 observations.
        To override an existing locked class, a new candidate class must exceed
        3.0x the confidence sum of the locked class over at least 15 observations.
        """
        conf_sums = self.class_confidence_sums.get(track_id, {})
        history_len = len(self.centroid_history.get(track_id, []))

        if history_len < 6 or not conf_sums:
            return

        # Hysteresis override check if already locked
        if track_id in self.locked_class:
            current_locked = self.locked_class[track_id]
            current_score = conf_sums.get(current_locked, 0.0)

            sorted_candidates = sorted(conf_sums.items(), key=lambda x: x[1], reverse=True)
            best_candidate, best_score = sorted_candidates[0]

            # Require 3.0x higher confidence and >= 15 observations to flip locked class
            if best_candidate != current_locked and history_len >= 15 and best_score >= (current_score * 3.0):
                self.locked_class[track_id] = best_candidate
            return

        # Initial Lock Determination (after >= 6 observations)
        if conf_sums.get('person', 0.0) > (sum(conf_sums.values()) * 0.50):
            self.locked_class[track_id] = 'person'
            return

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

            # Bus rule: dominant bus predictions OR mean_area > 12000 and mean_ar >= 1.15 with bus presence
            if bus_score >= max(car_score, truck_score) * 0.5 or (bus_score > 0 and mean_area > 12000 and mean_ar >= 1.15):
                self.locked_class[track_id] = 'bus'
                return

            # Truck rule: dominant truck predictions
            if truck_score > (car_score + bus_score) or (truck_score > 0 and mean_area > 28000):
                self.locked_class[track_id] = 'truck'
                return

            # Auto-rickshaw rule
            if auto_score >= max(1.0, car_score * 0.75) and auto_score >= (sum(conf_sums.values()) * 0.30):
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
        Active Roadway Vehicle Filter:
        Verifies that a tracked object represents an active physical roadway vehicle or person.
        Checks:
        1. PERSON EXEMPTION: Person tracks entering from the side are ALWAYS valid.
        2. Bounding box size & persistence: mean area >= 800px^2, persistence >= 2 frames.
        3. Static Roadside Scrap Filter: Excludes permanently stationary vehicle objects (displacement < 12.0px)
           located in far outer roadside margins outside the active roadway lane.
        """
        st_cls = self.get_stabilized_class(track_id)

        # 1. EXPLICIT PERSON EXEMPTION: Pedestrians/persons entering from side are always preserved
        if st_cls == 'person':
            centroids = self.centroid_history.get(track_id, [])
            return len(centroids) >= 1

        centroids = self.centroid_history.get(track_id, [])
        bboxes = self.bbox_history.get(track_id, [])

        if len(centroids) < 2 or len(bboxes) < 2:
            return False

        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
        mean_area = float(np.mean(areas))

        # Minimum size check
        if mean_area < 800.0:
            return False

        disp = self.get_track_displacement(track_id)
        last_b = bboxes[-1]
        cx = (last_b[0] + last_b[2]) / 2.0
        cy = (last_b[1] + last_b[3]) / 2.0

        # Roadside Margin Borders
        margin_x = min(40.0, self.frame_width * 0.06)
        margin_top = min(40.0, self.frame_height * 0.06)

        is_far_roadside_margin = (cx <= margin_x or cx >= (self.frame_width - margin_x) or cy <= margin_top)

        # Filter out static roadside scrap (permanently stationary in outer margins)
        if disp < self.min_displacement_threshold and is_far_roadside_margin and len(centroids) < 15:
            return False

        return True

    def get_all_stabilized_classes(self, moving_only=True):
        """
        Get map of track_id -> locked_stabilized_class for all valid tracked objects.
        If moving_only=True, excludes static roadside scrap.
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
        # Retain class_history, centroid_history, bbox_history, class_confidence_sums, locked_class for lifetime unique counting

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
