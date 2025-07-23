from sklearn.metrics import classification_report, accuracy_score
import json

def get_classification_report(y_true, y_pred, label_names):
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
    return report

def save_classification_report(report, path):
    with open(path, 'w') as f:
        json.dump(report, f, indent=4)

def print_metrics_summary(y_true, y_pred, label_names, model_name="Model"):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n📊 {model_name} Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=label_names))
