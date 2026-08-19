import pytest
from ml_engine.severity_estimator import DynamicSeverityEstimator

def test_minor_severity_calculation():
    estimator = DynamicSeverityEstimator()
    res = estimator.estimate_severity(
        impact_force_index=5.0,
        velocity_delta=10.0,
        vehicle_count=1,
        fire_detected=False,
        smoke_detected=False
    )
    assert res['severity_level'] == 'Minor'
    assert res['severity_score'] < 40.0


def test_critical_severity_calculation():
    estimator = DynamicSeverityEstimator()
    res = estimator.estimate_severity(
        impact_force_index=25.0,
        velocity_delta=60.0,
        vehicle_count=3,
        fire_detected=True,
        smoke_detected=True,
        vehicle_types=['truck', 'car', 'car']
    )
    assert res['severity_level'] == 'Critical'
    assert res['severity_score'] >= 75.0
