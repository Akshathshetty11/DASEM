import json
import logging
import cv2
from pathlib import Path
from src.config import PROJECT_ROOT, OUTPUT_SUMMARY_DIR, OUTPUT_DETECTIONS_DIR, ensure_directories

logger = logging.getLogger("DASEM.utils")

def validate_file_path(path_input, extension=None):
    """Validate input file existence using pathlib."""
    if not path_input:
        raise ValueError("[ERROR] Empty file path provided.")
    
    path = Path(path_input)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Required file not found: {path}")

    if extension and path.suffix.lower() != extension.lower():
        raise ValueError(f"[ERROR] Expected file extension '{extension}', got '{path.suffix}'")

    return path


def save_json(data, output_path):
    """Save dictionary/list data to a JSON file using pathlib."""
    path = Path(output_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    logger.info(f"[INFO] JSON successfully saved to: {path}")
    return path


def load_json(json_path):
    """Load JSON file using pathlib."""
    path = validate_file_path(json_path, extension=".json")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def get_video_metadata(video_path):
    """Extract fps, frame count, width, and height using OpenCV."""
    path = validate_file_path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"[ERROR] Could not open video file: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if fps > 0 else 0.0

    cap.release()
    return {
        'fps': round(fps, 2),
        'width': width,
        'height': height,
        'frame_count': frame_count,
        'duration_seconds': round(duration_sec, 2)
    }


def get_active_video_stem(st_session_state=None):
    """
    Dynamically resolve the active video stem for user uploaded video.
    Returns None if no video has been uploaded by the user.
    """
    if st_session_state is not None:
        if st_session_state.get("active_video_stem"):
            return st_session_state.get("active_video_stem")
        if st_session_state.get("video_stem"):
            return st_session_state.get("video_stem")
    
    return None
