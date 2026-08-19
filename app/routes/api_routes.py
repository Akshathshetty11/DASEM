import os
import csv
from datetime import datetime
from flask import Blueprint, jsonify, request, current_app, send_from_directory
from app.models import db, Video, Accident, AccidentVehicle, Report
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

api_bp = Blueprint('api', __name__)

@api_bp.route('/analytics/summary', methods=['GET'])
def get_analytics_summary():
    """Get aggregated vehicle analytics metrics for dashboard."""
    total_videos = Video.query.count()
    total_accidents = Accident.query.count()
    
    minor_count = Accident.query.filter_by(severity_level='Minor').count()
    major_count = Accident.query.filter_by(severity_level='Major').count()
    critical_count = Accident.query.filter_by(severity_level='Critical').count()
    
    fire_count = Accident.query.filter_by(fire_detected=True).count()
    smoke_count = Accident.query.filter_by(smoke_detected=True).count()
    
    total_cars = db.session.query(db.func.sum(Video.cars_count)).scalar() or 0
    total_buses = db.session.query(db.func.sum(Video.buses_count)).scalar() or 0
    total_trucks = db.session.query(db.func.sum(Video.trucks_count)).scalar() or 0
    total_bikes = db.session.query(db.func.sum(Video.bikes_count)).scalar() or 0

    recent_accidents = Accident.query.order_by(Accident.detected_at.desc()).limit(5).all()

    return jsonify({
        'total_videos': total_videos,
        'total_accidents': total_accidents,
        'severity_breakdown': {
            'Minor': minor_count,
            'Major': major_count,
            'Critical': critical_count
        },
        'object_counts': {
            'cars': total_cars,
            'buses': total_buses,
            'trucks': total_trucks,
            'bikes': total_bikes
        },
        'hazards': {
            'fire_incidents': fire_count,
            'smoke_incidents': smoke_count
        },
        'recent_accidents': [a.to_dict() for a in recent_accidents]
    })


@api_bp.route('/accidents', methods=['GET'])
def get_accidents():
    """Get accidents list with optional filters."""
    severity = request.args.get('severity')
    status = request.args.get('status')
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 20, type=int)

    query = Accident.query

    if severity and severity != 'all':
        query = query.filter(Accident.severity_level == severity)
    if status and status != 'all':
        query = query.filter(Accident.verification_status == status)

    pagination = query.order_by(Accident.detected_at.desc()).paginate(page=page, per_page=limit, error_out=False)

    return jsonify({
        'accidents': [a.to_dict() for a in pagination.items],
        'total': pagination.total,
        'page': page,
        'pages': pagination.pages
    })


@api_bp.route('/accidents/<int:accident_id>', methods=['GET'])
def get_accident_detail(accident_id):
    """Get detailed record for a specific accident."""
    accident = Accident.query.get_or_404(accident_id)
    return jsonify(accident.to_dict())


@api_bp.route('/accidents/<int:accident_id>', methods=['PATCH'])
def update_accident_status(accident_id):
    """Update verification status."""
    accident = Accident.query.get_or_404(accident_id)
    data = request.get_json() or {}
    
    new_status = data.get('status')
    if new_status not in ['unverified', 'confirmed', 'false_positive']:
        return jsonify({'error': 'Invalid status'}), 400

    accident.verification_status = new_status
    db.session.commit()
    return jsonify(accident.to_dict())


@api_bp.route('/reports/export', methods=['POST'])
def export_report():
    """Generate PDF or CSV report of detected accidents."""
    data = request.get_json() or {}
    report_format = data.get('format', 'pdf').lower()
    
    accidents = Accident.query.order_by(Accident.detected_at.desc()).all()
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    reports_dir = current_app.config['REPORTS_FOLDER']
    os.makedirs(reports_dir, exist_ok=True)

    if report_format == 'csv':
        filename = f"accident_report_{timestamp_str}.csv"
        file_path = os.path.join(reports_dir, filename)
        
        with open(file_path, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['ID', 'Video ID', 'Timestamp (s)', 'Frame', 'Severity Level', 'Severity Score', 'Cars', 'Buses', 'Trucks', 'Motorcycles', 'Fire', 'Smoke', 'Status'])
            for acc in accidents:
                writer.writerow([
                    acc.id, acc.video_id, acc.timestamp_sec, acc.frame_number,
                    acc.severity_level, acc.severity_score, acc.cars_count, acc.buses_count, acc.trucks_count, acc.bikes_count,
                    acc.fire_detected, acc.smoke_detected, acc.verification_status
                ])
                
        report_record = Report(report_name=filename, report_type='CSV', file_path=file_path)
        db.session.add(report_record)
        db.session.commit()
        
        return jsonify({
            'message': 'CSV report generated successfully',
            'report': report_record.to_dict(),
            'download_url': f"/api/v1/reports/download/{filename}"
        })

    else:
        filename = f"accident_report_{timestamp_str}.pdf"
        file_path = os.path.join(reports_dir, filename)
        
        doc = SimpleDocTemplate(file_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#0f172a'),
            spaceAfter=12
        )
        
        elements.append(Paragraph("Dynamic Vehicle Accident Severity & Telemetry Report", title_style))
        elements.append(Paragraph(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Spacer(1, 16))

        table_data = [['ID', 'Timestamp', 'Severity', 'Score', 'Vehicles', 'Fire', 'Status']]
        for acc in accidents:
            total_v = (acc.cars_count + acc.buses_count + acc.trucks_count + acc.bikes_count) or acc.vehicle_count
            table_data.append([
                str(acc.id),
                f"{acc.timestamp_sec}s",
                acc.severity_level,
                f"{acc.severity_score:.1f}",
                str(total_v),
                "Yes" if acc.fire_detected else "No",
                acc.verification_status.capitalize()
            ])

        t = Table(table_data, colWidths=[40, 80, 85, 75, 75, 55, 90])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e293b')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 9),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1'))
        ]))
        elements.append(t)
        doc.build(elements)

        report_record = Report(report_name=filename, report_type='PDF', file_path=file_path)
        db.session.add(report_record)
        db.session.commit()

        return jsonify({
            'message': 'PDF report generated successfully',
            'report': report_record.to_dict(),
            'download_url': f"/api/v1/reports/download/{filename}"
        })


@api_bp.route('/reports/download/<filename>', methods=['GET'])
def download_report(filename):
    """Download a generated report file."""
    reports_dir = current_app.config['REPORTS_FOLDER']
    return send_from_directory(reports_dir, filename, as_attachment=True)
