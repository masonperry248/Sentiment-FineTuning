import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

def plot_confusion_matrix(y_true, y_pred, labels, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_f1_comparison(report_zs, report_ft, label_names, save_path=None):
    data = {
        "Zero-Shot F1": [report_zs[label]["f1-score"] for label in label_names],
        "Fine-Tuned F1": [report_ft[label]["f1-score"] for label in label_names],
    }
    df = pd.DataFrame(data, index=label_names)
    df.plot(kind='bar', figsize=(10, 6), title='F1 Score Comparison: Zero-Shot vs Fine-Tuned')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
