import sys
import os

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app
from app.models import db, Video, Accident, AccidentVehicle, Report

def init_database(seed_sample_data=False, reset_db=True):
    """Initialize SQLite database tables with vehicle telemetry support."""
    app = create_app()
    with app.app_context():
        if reset_db:
            print("[DB Init] Re-creating database tables for vehicle telemetry...")
            db.drop_all()
        
        db.create_all()
        print("[DB Init] Database tables created successfully.")

        if seed_sample_data:
            print("[DB Init] Seeding initial sample vehicle telemetry data...")
            sample_video = Video(
                filename="sample_highway_traffic.mp4",
                file_path="static/uploads/raw/sample_highway_traffic.mp4",
                duration_seconds=45.0,
                fps=30.0,
                status="completed",
                cars_count=8,
                buses_count=1,
                trucks_count=2,
                bikes_count=3,
                fire_detected=False,
                smoke_detected=True,
                detection_summary_json='{"cars": 8, "buses": 1, "trucks": 2, "bikes": 3, "fire_detected": false, "smoke_detected": true}'
            )
            db.session.add(sample_video)
            db.session.commit()

            sample_accident = Accident(
                video_id=sample_video.id,
                timestamp_sec=14.5,
                frame_number=435,
                severity_level="Major",
                severity_score=68.4,
                vehicle_count=2,
                cars_count=5,
                buses_count=1,
                trucks_count=1,
                bikes_count=2,
                fire_detected=False,
                smoke_detected=True,
                snapshot_path="static/uploads/snapshots/acc_14s_sample.jpg",
                bbox_json='[120, 200, 350, 420]',
                verification_status="unverified"
            )
            db.session.add(sample_accident)
            db.session.commit()

            veh1 = AccidentVehicle(
                accident_id=sample_accident.id,
                track_id=101,
                vehicle_type="car",
                pre_impact_speed=75.0,
                post_impact_speed=15.0,
                impact_force_index=60.0
            )
            veh2 = AccidentVehicle(
                accident_id=sample_accident.id,
                track_id=104,
                vehicle_type="truck",
                pre_impact_speed=60.0,
                post_impact_speed=25.0,
                impact_force_index=45.0
            )
            db.session.add_all([veh1, veh2])
            db.session.commit()
            print("[DB Init] Sample vehicle telemetry data seeded successfully.")

if __name__ == '__main__':
    seed = '--seed' in sys.argv
    init_database(seed_sample_data=seed, reset_db=True)
