import os
from flask import Blueprint, render_template, redirect, url_for, send_from_directory, current_app
from app.models import Video, Accident, Report

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Redirect root route to dashboard."""
    return redirect(url_for('main.dashboard'))

@main_bp.route('/dashboard')
def dashboard():
    """Main Dashboard view."""
    return render_template('dashboard.html')

@main_bp.route('/video-analysis')
def video_analysis():
    """Video Upload and Real-Time Analysis view."""
    return render_template('video_analysis.html')

@main_bp.route('/history')
def history():
    """Searchable Accident History view."""
    return render_template('history.html')

@main_bp.route('/accident/<int:accident_id>')
def accident_detail(accident_id):
    """Accident Event Detail view."""
    return render_template('accident_detail.html', accident_id=accident_id)

@main_bp.route('/reports')
def reports():
    """Exportable Reports view."""
    return render_template('reports.html')

@main_bp.route('/app/static/<path:filename>')
def serve_legacy_app_static(filename):
    """Fallback route to serve static assets requested with legacy /app/static/... URL prefix."""
    static_dir = os.path.join(current_app.config['BASE_DIR'], 'app', 'static')
    return send_from_directory(static_dir, filename)
