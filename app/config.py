import os
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    """Flask Application Configuration."""
    BASE_DIR = str(_BASE_DIR)
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # SQLite Database URI
    DB_PATH = os.path.join(BASE_DIR, 'database', 'app.db')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', f'sqlite:///{DB_PATH}')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Upload directories
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'app', 'static', 'uploads')
    RAW_UPLOADS = os.path.join(UPLOAD_FOLDER, 'raw')
    PROCESSED_UPLOADS = os.path.join(UPLOAD_FOLDER, 'processed')
    CLIPS_UPLOADS = os.path.join(UPLOAD_FOLDER, 'clips')
    SNAPSHOTS_UPLOADS = os.path.join(UPLOAD_FOLDER, 'snapshots')
    REPORTS_FOLDER = os.path.join(BASE_DIR, 'reports')
    
    # Max video upload size (500 MB)
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024
    
    # Computer Vision & Detection Config
    MODEL_CONFIDENCE_THRESHOLD = float(os.environ.get('MODEL_CONFIDENCE_THRESHOLD', '0.45'))
    TRACKER_TYPE = 'bytetrack'
    
    # Severity Score Thresholds (0 - 100)
    SEVERITY_MINOR_MAX = 40.0
    SEVERITY_MAJOR_MAX = 74.9
    # Anything >= 75.0 is Critical
    
    # ML Weights Paths
    YOLO_ACCIDENT_WEIGHTS = os.path.join(BASE_DIR, 'ml_engine', 'weights', 'yolov8n_accident.pt')
    YOLO_FIRE_SMOKE_WEIGHTS = os.path.join(BASE_DIR, 'ml_engine', 'weights', 'yolov8n_fire_smoke.pt')
