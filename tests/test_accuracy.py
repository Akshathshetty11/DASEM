import pytest
from ml_engine.utils.accuracy_evaluator import AccuracyEvaluator

def test_accuracy_evaluator():
    evaluator = AccuracyEvaluator()
    
    # Ground truth vs Predictions test dataset
    samples = [
        ('car', 'car'), ('car', 'car'), ('car', 'car'), ('car', 'bus'),
        ('bus', 'bus'), ('bus', 'bus'),
        ('truck', 'truck'), ('truck', 'truck'),
        ('motorcycle', 'motorcycle'), ('motorcycle', 'motorcycle'), ('motorcycle', 'car'),
        ('person', 'person'), ('person', 'person'),
        ('fire', 'fire'), ('smoke', 'smoke')
    ]

    for true_c, pred_c in samples:
        evaluator.add_sample(true_c, pred_c)

    metrics = evaluator.compute_metrics()
    assert metrics['overall_accuracy'] > 0.80
    assert 'car' in metrics['per_class']
    assert 'motorcycle' in metrics['per_class']
    assert 'person' in metrics['per_class']
    assert 'fire' in metrics['per_class']
    assert 'smoke' in metrics['per_class']

    text_report = evaluator.generate_report_text()
    assert "MULTI-CLASS CLASSIFICATION METRICS REPORT" in text_report
