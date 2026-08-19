import pytest
from app import create_app
from app.models import db, Video, Accident

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
            yield client


def test_analytics_summary_endpoint(client):
    response = client.get('/api/v1/analytics/summary')
    assert response.status_code == 200
    data = response.get_json()
    assert 'total_videos' in data
    assert 'total_accidents' in data
    assert 'severity_breakdown' in data


def test_accidents_list_endpoint(client):
    response = client.get('/api/v1/accidents')
    assert response.status_code == 200
    data = response.get_json()
    assert 'accidents' in data
    assert 'total' in data
