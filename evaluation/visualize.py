import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(metrics, results_dir, prefix=""):
    """
    Saves a confusion matrix heatmap as {prefix}confusion_matrix.png
    using data already computed in metrics.json (no retraining needed).
    """
    cm = np.array(metrics["confusion_matrix"])
    labels = metrics["confusion_matrix_labels"]

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    out_path = os.path.join(results_dir, f"{prefix}confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_per_class_metrics(metrics, results_dir, prefix=""):
    """
    Saves a grouped bar chart of precision/recall/f1 per class.
    """
    per_class = metrics["per_class"]
    labels = list(per_class.keys())
    precision = [per_class[l]["precision"] for l in labels]
    recall = [per_class[l]["recall"] for l in labels]
    f1 = [per_class[l]["f1"] for l in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1")

    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()

    out_path = os.path.join(results_dir, f"{prefix}per_class_metrics.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def generate_visuals(metrics, results_dir, prefix=""):
    """
    Generates all standard visuals for one experiment run.
    Called from run_experiment.py after metrics are computed.
    """
    cm_path = plot_confusion_matrix(metrics, results_dir, prefix=prefix)
    bar_path = plot_per_class_metrics(metrics, results_dir, prefix=prefix)
    return {"confusion_matrix_png": cm_path, "per_class_metrics_png": bar_path}