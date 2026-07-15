import sys
import os
import json
import yaml
import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def load_run(run_folder):
    base = os.path.join("results", run_folder)

    with open(os.path.join(base, "config.yaml")) as f:
        config = yaml.safe_load(f)
    with open(os.path.join(base, "metrics.json")) as f:
        metrics = json.load(f)
    with open(os.path.join(base, "benchmark.json")) as f:
        benchmark = json.load(f)

    val_metrics = None
    val_path = os.path.join(base, "val_metrics.json")
    if os.path.exists(val_path):
        with open(val_path) as f:
            val_metrics = json.load(f)

    return {
        "run_name": run_folder,
        "config": config,
        "metrics": metrics,
        "val_metrics": val_metrics,
        "benchmark": benchmark,
    }


def select_runs():
    all_folders = sorted(
        f for f in os.listdir("results")
        if os.path.isdir(os.path.join("results", f)) and f != "comparisons"
    )

    if not all_folders:
        print("No experiment runs found in results/")
        sys.exit(1)

    print("Available runs:")
    for i, name in enumerate(all_folders):
        print(f"  [{i}] {name}")

    choice = input("Select run numbers to compare (comma-separated, e.g. 0,2,3): ").strip()
    try:
        indices = [int(x.strip()) for x in choice.split(",")]
        selected = [all_folders[i] for i in indices]
    except (ValueError, IndexError):
        print("Invalid selection.")
        sys.exit(1)

    return selected


def _fmt(val, width, prec=4):
    if val is None:
        return f"{'--':<{width}}"
    return f"{val:<{width}.{prec}f}"


def print_console_table(runs):
    # Test accuracy/latency/size, plus fuzzy and impersonation recall
    # (test and val side by side) since those two classes have been the
    # focus of recent split/feature debugging.
    header = (
        f"{'Run':35} {'Model':10} {'Feats':14} "
        f"{'TestAcc':9} {'ValAcc':9} "
        f"{'FuzzyR(t/v)':14} {'ImpR(t/v)':14} "
        f"{'Lat(ms)':9} {'Size(MB)':9}"
    )
    print(header)
    print("-" * len(header))
    for run in runs:
        name = run["run_name"][:33]
        model = run["config"].get("model", "?")[:10]
        feats = ",".join(run["config"].get("features", []))[:14]

        test_acc = run["metrics"]["accuracy"]
        val_acc = run["val_metrics"]["accuracy"] if run["val_metrics"] else None

        fuzzy_t = run["metrics"]["per_class"].get("fuzzy", {}).get("recall")
        fuzzy_v = (run["val_metrics"]["per_class"].get("fuzzy", {}).get("recall")
                   if run["val_metrics"] else None)
        imp_t = run["metrics"]["per_class"].get("impersonation", {}).get("recall")
        imp_v = (run["val_metrics"]["per_class"].get("impersonation", {}).get("recall")
                 if run["val_metrics"] else None)

        fuzzy_str = f"{_fmt(fuzzy_t, 6, 3).strip()}/{_fmt(fuzzy_v, 6, 3).strip()}"
        imp_str = f"{_fmt(imp_t, 6, 3).strip()}/{_fmt(imp_v, 6, 3).strip()}"

        latency = run["benchmark"]["latency_ms"]
        size = run["benchmark"]["model_size_mb"]

        print(
            f"{name:35} {model:10} {feats:14} "
            f"{_fmt(test_acc, 9)} {_fmt(val_acc, 9)} "
            f"{fuzzy_str:14} {imp_str:14} "
            f"{latency:<9.4f} {size:<9.2f}"
        )


def save_csv(runs, out_path):
    all_classes = sorted({
        cls
        for run in runs
        for cls in list(run["metrics"]["per_class"].keys())
                  + (list(run["val_metrics"]["per_class"].keys()) if run["val_metrics"] else [])
    })

    fieldnames = ["run_name", "dataset", "model", "features",
                  "test_accuracy", "val_accuracy",
                  "latency_ms", "model_size_mb", "memory_mb"]
    for cls in all_classes:
        fieldnames += [
            f"test_{cls}_precision", f"test_{cls}_recall", f"test_{cls}_f1",
            f"val_{cls}_precision", f"val_{cls}_recall", f"val_{cls}_f1",
        ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for run in runs:
            row = {
                "run_name": run["run_name"],
                "dataset": run["config"].get("dataset"),
                "model": run["config"].get("model"),
                "features": ",".join(run["config"].get("features", [])),
                "test_accuracy": run["metrics"]["accuracy"],
                "val_accuracy": run["val_metrics"]["accuracy"] if run["val_metrics"] else None,
                "latency_ms": run["benchmark"]["latency_ms"],
                "model_size_mb": run["benchmark"]["model_size_mb"],
                "memory_mb": run["benchmark"]["memory_mb"],
            }
            for cls in all_classes:
                test_pc = run["metrics"]["per_class"].get(cls)
                if test_pc:
                    row[f"test_{cls}_precision"] = test_pc["precision"]
                    row[f"test_{cls}_recall"] = test_pc["recall"]
                    row[f"test_{cls}_f1"] = test_pc["f1"]

                if run["val_metrics"]:
                    val_pc = run["val_metrics"]["per_class"].get(cls)
                    if val_pc:
                        row[f"val_{cls}_precision"] = val_pc["precision"]
                        row[f"val_{cls}_recall"] = val_pc["recall"]
                        row[f"val_{cls}_f1"] = val_pc["f1"]
            writer.writerow(row)


def _plot_confusion_grid(runs, metrics_key, out_path, title_suffix=""):
    """
    metrics_key: "metrics" for test, "val_metrics" for val.
    Skips runs missing that key (e.g. older runs with no val_metrics.json).
    """
    usable_runs = [r for r in runs if r.get(metrics_key) is not None]
    if not usable_runs:
        print(f"No runs have {metrics_key} — skipping {out_path}")
        return

    n = len(usable_runs)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1) if n > 1 else [axes]

    for i, run in enumerate(usable_runs):
        cm = np.array(run[metrics_key]["confusion_matrix"])
        labels = run[metrics_key]["confusion_matrix_labels"]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=axes[i], cmap="Blues", values_format="d", colorbar=False)
        axes[i].set_title(f"{run['run_name']}{title_suffix}", fontsize=9)

    for j in range(len(usable_runs), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    run_folders = select_runs()
    runs = [load_run(f) for f in run_folders]

    print_console_table(runs)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir = os.path.join("results", "comparisons", f"comparison_{timestamp}")
    os.makedirs(comparison_dir, exist_ok=True)

    csv_path = os.path.join(comparison_dir, "comparison.csv")
    save_csv(runs, csv_path)

    test_cm_path = os.path.join(comparison_dir, "confusion_matrices_test.png")
    _plot_confusion_grid(runs, "metrics", test_cm_path, title_suffix=" (test)")

    val_cm_path = os.path.join(comparison_dir, "confusion_matrices_val.png")
    _plot_confusion_grid(runs, "val_metrics", val_cm_path, title_suffix=" (val)")

    print(f"\nComparison saved to: {comparison_dir}")


if __name__ == "__main__":
    main()