import os
import json
import threading
from werkzeug.utils import secure_filename
from flask import Blueprint, jsonify, request, current_app, send_from_directory
from app.models import db, Video, Accident, AccidentVehicle
from ml_engine.utils.video_processor import VideoProcessor

video_bp = Blueprint('video', __name__)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def format_static_url(filepath, base_dir):
    """Format file path into clean Flask static URL path (static/uploads/...)."""
    if not filepath:
        return None
    rel_path = os.path.relpath(filepath, base_dir).replace('\\', '/')
    if rel_path.startswith('app/static/'):
        return rel_path.replace('app/static/', 'static/', 1)
    return rel_path


def background_video_processing_task(app, video_id, input_path):
    """Background worker function for ML processing video."""
    with app.app_context():
        video = Video.query.get(video_id)
        if not video:
            return

        try:
            video.status = 'processing'
            db.session.commit()

            processor = VideoProcessor(
                accident_weights=app.config['YOLO_ACCIDENT_WEIGHTS'],
                fire_smoke_weights=app.config['YOLO_FIRE_SMOKE_WEIGHTS'],
                conf_threshold=app.config['MODEL_CONFIDENCE_THRESHOLD']
            )

            result = processor.process_video(
                input_video_path=input_path,
                output_dir=app.config['PROCESSED_UPLOADS'],
                snapshots_dir=app.config['SNAPSHOTS_UPLOADS']
            )

            summary = result.get('detection_summary', {})

            video.duration_seconds = result['video_metadata']['duration_seconds']
            video.fps = result['video_metadata']['fps']
            video.processed_path = format_static_url(result['processed_video_path'], app.config['BASE_DIR'])
            video.status = 'completed'

            video.cars_count = summary.get('cars', 0)
            video.buses_count = summary.get('buses', 0)
            video.trucks_count = summary.get('trucks', 0)
            video.bikes_count = summary.get('bikes', 0)
            video.fire_detected = summary.get('fire_detected', False)
            video.smoke_detected = summary.get('smoke_detected', False)
            video.detection_summary_json = json.dumps(summary)

            db.session.commit()

            for acc in result['detected_accidents']:
                formatted_snap = format_static_url(acc['snapshot_path'], app.config['BASE_DIR'])
                
                accident_record = Accident(
                    video_id=video.id,
                    timestamp_sec=acc['timestamp_sec'],
                    frame_number=acc['frame_number'],
                    severity_level=acc['severity_level'],
                    severity_score=acc['severity_score'],
                    vehicle_count=acc['vehicle_count'],
                    cars_count=summary.get('cars', 0),
                    buses_count=summary.get('buses', 0),
                    trucks_count=summary.get('trucks', 0),
                    bikes_count=summary.get('bikes', 0),
                    fire_detected=acc['fire_detected'],
                    smoke_detected=acc['smoke_detected'],
                    snapshot_path=formatted_snap,
                    bbox_json=str(acc['bbox']),
                    verification_status='unverified'
                )
                db.session.add(accident_record)
                db.session.commit()

                for v in acc['vehicles']:
                    veh_record = AccidentVehicle(
                        accident_id=accident_record.id,
                        track_id=v['track_id'],
                        vehicle_type=v['vehicle_type'],
                        pre_impact_speed=v['pre_impact_speed'],
                        post_impact_speed=v['post_impact_speed'],
                        impact_force_index=v['impact_force_index']
                    )
                    db.session.add(veh_record)
                db.session.commit()

            app.logger.info(f"Video {video.filename} processed successfully. Summary: {summary}")

        except Exception as e:
            app.logger.error(f"Error processing video ID {video_id}: {e}")
            video.status = 'failed'
            db.session.commit()


@video_bp.route('/upload', methods=['POST'])
def upload_video():
    """Upload a new video for vehicle & hazard analysis."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(current_app.config['RAW_UPLOADS'], filename)
        file.save(save_path)

        formatted_rel_path = format_static_url(save_path, current_app.config['BASE_DIR'])

        video = Video(
            filename=filename,
            file_path=formatted_rel_path,
            status='uploaded'
        )
        db.session.add(video)
        db.session.commit()

        app = current_app._get_current_object()
        thread = threading.Thread(
            target=background_video_processing_task,
            args=(app, video.id, save_path)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'message': 'Video uploaded successfully. Vehicle processing started.',
            'video': video.to_dict()
        }), 202

    return jsonify({'error': 'Invalid video format'}), 400


@video_bp.route('/', methods=['GET'])
def list_videos():
    """List all uploaded videos with vehicle counts."""
    videos = Video.query.order_by(Video.uploaded_at.desc()).all()
    return jsonify([v.to_dict() for v in videos])


@video_bp.route('/<int:video_id>/status', methods=['GET'])
def check_status(video_id):
    """Check processing status and summary of a specific video."""
    video = Video.query.get_or_404(video_id)
    return jsonify({
        'video_id': video.id,
        'status': video.status,
        'accident_count': len(video.accidents),
        'detection_summary': video.get_summary(),
        'video': video.to_dict()
    })
