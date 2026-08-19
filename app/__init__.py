import os
from flask import Flask
from flask_cors import CORS
from app.config import Config
from app.models import db

def create_app(config_class=Config):
    """Flask Application Factory."""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize extensions
    db.init_app(app)
    CORS(app)

    # Ensure upload directories exist
    upload_dirs = [
        app.config['RAW_UPLOADS'],
        app.config['PROCESSED_UPLOADS'],
        app.config['CLIPS_UPLOADS'],
        app.config['SNAPSHOTS_UPLOADS'],
        app.config['REPORTS_FOLDER']
    ]
    for d in upload_dirs:
        os.makedirs(d, exist_ok=True)

    # Register Blueprints
    from app.routes.main_routes import main_bp
    from app.routes.api_routes import api_bp
    from app.routes.video_routes import video_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix='/api/v1')
    app.register_blueprint(video_bp, url_prefix='/api/v1/videos')

    return app
