from pathlib import Path

# Base Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Core Model Paths (Configurable)
MODELS_DIR = PROJECT_ROOT / "models"
VEHICLE_MODEL_PATH = MODELS_DIR / "vehicle" / "yolov8m.pt"
ACCIDENT_MODEL_PATH = MODELS_DIR / "accident" / "yolov8m_accident.pt"
FIRE_MODEL_PATH = MODELS_DIR / "fire" / "yolov8n_fire.pt"
SMOKE_MODEL_PATH = MODELS_DIR / "smoke" / "yolov8n_smoke.pt"

# Video Input Directories
VIDEOS_DIR = PROJECT_ROOT / "videos"
UPLOADED_VIDEOS_DIR = VIDEOS_DIR / "uploaded"
SAMPLE_VIDEOS_DIR = VIDEOS_DIR / "samples"

# Output Artifact Directories
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DETECTED_DIR = OUTPUT_DIR / "detected"
OUTPUT_DETECTIONS_DIR = OUTPUT_DIR / "detections"
OUTPUT_ACCIDENTS_DIR = OUTPUT_DIR / "accidents"
OUTPUT_SEVERITY_DIR = OUTPUT_DIR / "severity"
OUTPUT_FIRE_DIR = OUTPUT_DIR / "fire"
OUTPUT_SMOKE_DIR = OUTPUT_DIR / "smoke"
OUTPUT_EMERGENCY_DIR = OUTPUT_DIR / "emergency"
OUTPUT_RECOMMENDATIONS_DIR = OUTPUT_DIR / "recommendations"
OUTPUT_SUMMARY_DIR = OUTPUT_DIR / "summary"
OUTPUT_COMBINED_DIR = OUTPUT_DIR / "combined"

# Ensure All Directories Exist
def ensure_directories():
    """Create all required project directories if missing."""
    dirs = [
        MODELS_DIR / "vehicle", MODELS_DIR / "accident", MODELS_DIR / "fire", MODELS_DIR / "smoke",
        UPLOADED_VIDEOS_DIR, SAMPLE_VIDEOS_DIR,
        OUTPUT_DETECTED_DIR, OUTPUT_DETECTIONS_DIR, OUTPUT_ACCIDENTS_DIR,
        OUTPUT_SEVERITY_DIR, OUTPUT_FIRE_DIR, OUTPUT_SMOKE_DIR,
        OUTPUT_EMERGENCY_DIR, OUTPUT_RECOMMENDATIONS_DIR, OUTPUT_SUMMARY_DIR, OUTPUT_COMBINED_DIR
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

# Call directory check on import
ensure_directories()
