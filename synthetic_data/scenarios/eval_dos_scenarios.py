"""
eval_dos_scenarios.py
Runs each synthetic DoS scenario CSV through the real feature pipeline and a
previously-trained model, reporting recall/precision on the "dos" class per
scenario. This is the actual hypothesis test.

Usage:
    python eval_dos_scenarios.py --results-dir results/xgb_btp_<timestamp>
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import yaml

# adjust these two lines to match your actual project layout/imports
sys.path.append(str(Path(__file__).resolve().parents[2]))
from features.pipeline import build_features
from evaluation.metrics import compute_metrics

SCENARIO_DIR = Path(__file__).resolve().parents[1] / "output"

SCENARIO_FILES = [
    "dos_fixed_payload_baseline.csv",
    "dos_varied_payload_same_id.csv",
    "dos_different_id.csv",
    "dos_different_id_varied_payload.csv",
    "dos_partial_flood.csv",
    "dos_multi_id_flood.csv",
]


def load_trained_run(results_dir):
    results_dir = Path(results_dir)
    with open(results_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    with open(results_dir / "feature_params.json") as f:
        feature_params = json.load(f)
    with open(results_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)
    return config, feature_params, model


def eval_scenario(csv_path, config, feature_params, model):
    df = pd.read_csv(csv_path)
    X, y, _ = build_features(df, config["features"], feature_params)
    y_pred = model.predict(X)
    metrics = compute_metrics(y, y_pred)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, help="path to a completed results/<run>/ dir")
    args = ap.parse_args()

    config, feature_params, model = load_trained_run(args.results_dir)
    print(f"Loaded model from {args.results_dir}")
    print(f"Features used: {config['features']}\n")

    summary_rows = []
    for fname in SCENARIO_FILES:
        path = SCENARIO_DIR / fname
        if not path.exists():
            print(f"SKIP {fname} (not found — run run_dos_scenarios.py first)")
            continue
        metrics = eval_scenario(path, config, feature_params, model)
        dos_metrics = metrics.get("per_class", {}).get("dos", {})
        row = {
            "scenario": fname.replace(".csv", ""),
            "accuracy": metrics.get("accuracy"),
            "dos_precision": dos_metrics.get("precision"),
            "dos_recall": dos_metrics.get("recall"),
            "dos_f1": dos_metrics.get("f1"),
        }
        summary_rows.append(row)
        print(f"{row['scenario']:40s} | acc={row['accuracy']:.3f} | "
              f"dos_recall={row['dos_recall']:.3f} | dos_precision={row['dos_precision']:.3f}")

    summary_df = pd.DataFrame(summary_rows)
    out_csv = SCENARIO_DIR / "dos_scenario_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"\nSummary saved to {out_csv}")


if __name__ == "__main__":
    main()