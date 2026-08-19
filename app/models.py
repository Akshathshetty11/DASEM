from datetime import datetime
import json
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Video(db.Model):
    """Uploaded Video Entity with Vehicle Telemetry (Persons Excluded)."""
    __tablename__ = 'videos'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    processed_path = db.Column(db.String(500), nullable=True)
    duration_seconds = db.Column(db.Float, default=0.0)
    fps = db.Column(db.Float, default=0.0)
    status = db.Column(db.String(50), default='uploaded')
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Vehicle Summary Counts
    cars_count = db.Column(db.Integer, default=0)
    buses_count = db.Column(db.Integer, default=0)
    trucks_count = db.Column(db.Integer, default=0)
    bikes_count = db.Column(db.Integer, default=0)
    fire_detected = db.Column(db.Boolean, default=False)
    smoke_detected = db.Column(db.Boolean, default=False)
    detection_summary_json = db.Column(db.Text, nullable=True)

    accidents = db.relationship('Accident', backref='video', lazy=True, cascade='all, delete-orphan')

    def get_summary(self):
        if self.detection_summary_json:
            try:
                return json.loads(self.detection_summary_json)
            except Exception:
                pass
        return {
            "cars": self.cars_count,
            "buses": self.buses_count,
            "trucks": self.trucks_count,
            "bikes": self.bikes_count,
            "fire_detected": self.fire_detected,
            "smoke_detected": self.smoke_detected
        }

    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'file_path': self.file_path,
            'processed_path': self.processed_path,
            'duration_seconds': self.duration_seconds,
            'fps': self.fps,
            'status': self.status,
            'uploaded_at': self.uploaded_at.isoformat() if self.uploaded_at else None,
            'accident_count': len(self.accidents),
            'object_counts': self.get_summary(),
            'detection_summary': self.get_summary()
        }


class Accident(db.Model):
    """Detected Accident Event Entity."""
    __tablename__ = 'accidents'
    
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey('videos.id'), nullable=False)
    timestamp_sec = db.Column(db.Float, nullable=False)
    frame_number = db.Column(db.Integer, nullable=False)
    severity_level = db.Column(db.String(20), nullable=False)  # Minor, Major, Critical
    severity_score = db.Column(db.Float, nullable=False)        # 0.0 - 100.0
    vehicle_count = db.Column(db.Integer, default=1)
    cars_count = db.Column(db.Integer, default=0)
    buses_count = db.Column(db.Integer, default=0)
    trucks_count = db.Column(db.Integer, default=0)
    bikes_count = db.Column(db.Integer, default=0)
    fire_detected = db.Column(db.Boolean, default=False)
    smoke_detected = db.Column(db.Boolean, default=False)
    snapshot_path = db.Column(db.String(500), nullable=True)
    clip_path = db.Column(db.String(500), nullable=True)
    bbox_json = db.Column(db.Text, nullable=True)
    verification_status = db.Column(db.String(30), default='unverified')
    detected_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    vehicles = db.relationship('AccidentVehicle', backref='accident', lazy=True, cascade='all, delete-orphan')

    def get_bbox(self):
        if self.bbox_json:
            try:
                return json.loads(self.bbox_json)
            except Exception:
                return []
        return []

    def to_dict(self):
        return {
            'id': self.id,
            'video_id': self.video_id,
            'timestamp_sec': round(self.timestamp_sec, 2),
            'frame_number': self.frame_number,
            'severity_level': self.severity_level,
            'severity_score': round(self.severity_score, 2),
            'vehicle_count': self.vehicle_count,
            'cars_count': self.cars_count,
            'buses_count': self.buses_count,
            'trucks_count': self.trucks_count,
            'bikes_count': self.bikes_count,
            'fire_detected': self.fire_detected,
            'smoke_detected': self.smoke_detected,
            'snapshot_path': self.snapshot_path,
            'clip_path': self.clip_path,
            'bbox': self.get_bbox(),
            'verification_status': self.verification_status,
            'detected_at': self.detected_at.isoformat() if self.detected_at else None,
            'vehicles': [v.to_dict() for v in self.vehicles]
        }


class AccidentVehicle(db.Model):
    """Specific Vehicle Dynamic Parameters involved in an Event."""
    __tablename__ = 'accident_vehicles'
    
    id = db.Column(db.Integer, primary_key=True)
    accident_id = db.Column(db.Integer, db.ForeignKey('accidents.id'), nullable=False)
    track_id = db.Column(db.Integer, nullable=True)
    vehicle_type = db.Column(db.String(50), default='car')  # car, bus, truck, motorcycle
    pre_impact_speed = db.Column(db.Float, default=0.0)
    post_impact_speed = db.Column(db.Float, default=0.0)
    impact_force_index = db.Column(db.Float, default=0.0)

    def to_dict(self):
        return {
            'id': self.id,
            'accident_id': self.accident_id,
            'track_id': self.track_id,
            'vehicle_type': self.vehicle_type,
            'pre_impact_speed': round(self.pre_impact_speed, 2),
            'post_impact_speed': round(self.post_impact_speed, 2),
            'impact_force_index': round(self.impact_force_index, 2)
        }


class Report(db.Model):
    """Generated PDF / CSV Audit Report Entity."""
    __tablename__ = 'reports'
    
    id = db.Column(db.Integer, primary_key=True)
    report_name = db.Column(db.String(255), nullable=False)
    report_type = db.Column(db.String(10), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    generated_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'report_name': self.report_name,
            'report_type': self.report_type,
            'file_path': self.file_path,
            'generated_at': self.generated_at.isoformat() if self.generated_at else None
        }
