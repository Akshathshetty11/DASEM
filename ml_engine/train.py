import os
import sys
import argparse
import numpy as np

# Vehicle classes only
VEHICLE_CLASSES = ['car', 'bus', 'truck', 'motorcycle']

class DatasetEvaluator:
    """
    Dataset & Model Training Evaluator.
    Computes class distributions, dataset balance metrics, Precision, Recall, F1-Score, and Confusion Matrix
    across Car, Bus, Truck, and Motorcycle.
    """

    def __init__(self, target_classes=None):
        self.classes = target_classes or VEHICLE_CLASSES
        self.num_classes = len(self.classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)

    def add_sample(self, true_class, pred_class):
        t_clean = str(true_class).lower().strip()
        p_clean = str(pred_class).lower().strip()

        if t_clean in ['bike', 'bicycle']: t_clean = 'motorcycle'
        if p_clean in ['bike', 'bicycle']: p_clean = 'motorcycle'

        if t_clean in self.class_to_idx and p_clean in self.class_to_idx:
            t_idx = self.class_to_idx[t_clean]
            p_idx = self.class_to_idx[p_clean]
            self.confusion_matrix[t_idx, p_idx] += 1

    def compute_metrics(self):
        cm = self.confusion_matrix
        total_samples = np.sum(cm)
        correct_samples = np.trace(cm)
        overall_accuracy = float(correct_samples / total_samples) if total_samples > 0 else 0.0

        per_class = {}
        for i, c_name in enumerate(self.classes):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp

            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            support = int(np.sum(cm[i, :]))

            per_class[c_name] = {
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1, 4),
                'support': support
            }

        return {
            'overall_accuracy': round(overall_accuracy, 4),
            'per_class': per_class,
            'confusion_matrix': cm.tolist()
        }

    def generate_report(self):
        metrics = self.compute_metrics()
        report = []
        report.append("=====================================================================")
        report.append("           VEHICLE CLASSIFICATION METRICS REPORT                    ")
        report.append("=====================================================================")
        report.append(f"Overall Accuracy: {metrics['overall_accuracy']:.2%}\n")
        report.append(f"{'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
        report.append("-" * 65)

        for c_name, m in metrics['per_class'].items():
            report.append(f"{c_name.capitalize():<12} | {m['precision']:<10.2%} | {m['recall']:<10.2%} | {m['f1_score']:<10.2%} | {m['support']:<8}")

        report.append("=====================================================================")
        return "\n".join(report)

def run_evaluation():
    evaluator = DatasetEvaluator()
    # Sample validation predictions vs ground truth
    sample_data = [
        ('car', 'car'), ('car', 'car'), ('car', 'car'), ('car', 'car'), ('car', 'bus'),
        ('bus', 'bus'), ('bus', 'bus'), ('bus', 'bus'), ('bus', 'truck'),
        ('truck', 'truck'), ('truck', 'truck'), ('truck', 'truck'),
        ('motorcycle', 'motorcycle'), ('motorcycle', 'motorcycle'), ('motorcycle', 'motorcycle')
    ]
    for t, p in sample_data:
        evaluator.add_sample(t, p)

    print(evaluator.generate_report())

if __name__ == '__main__':
    run_evaluation()
