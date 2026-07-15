import sys
import os
import json
import pickle
import shutil
import datetime
import re
import time
import threading
from tqdm import tqdm

from utils.config import load_config
from utils.split import temporal_split
from utils.split import temporal_split_augmented
from data.loaders.registry import get_loader
from features.pipeline import build_features
from models.registry import get_model
from evaluation.metrics import compute_metrics
from evaluation.benchmark import run_benchmark
from evaluation.visualize import generate_visuals


def select_config():
    config_files = sorted(
        f for f in os.listdir("configs")
        if f.endswith(".yaml") or f.endswith(".yml")
    )
    if not config_files:
        print("No config files found in configs/")
        sys.exit(1)
    print("\nAvailable configs:")
    for i, name in enumerate(config_files):
        print(f"  [{i}] {name}")
    choice = input("\nSelect a config by number: ").strip()
    try:
        index = int(choice)
        return os.path.join("configs", config_files[index])
    except (ValueError, IndexError):
        print("Invalid selection.")
        sys.exit(1)


def train_with_spinner(model, X_train, y_train):
    done = {"value": False}
    def spinner():
        with tqdm(
            desc=f"Training {model.name}",
            bar_format="{desc}: {elapsed}",
            dynamic_ncols=True
        ) as pbar:
            while not done["value"]:
                pbar.update(0)
                time.sleep(0.5)
    t = threading.Thread(target=spinner)
    t.start()
    try:
        model.train(X_train, y_train)
    finally:
        done["value"] = True
        t.join()


# ── Config ────────────────────────────────────────────────────────────────────
config_path = select_config()
config = load_config(config_path)

# ── Load dataset ──────────────────────────────────────────────────────────────
loader = get_loader(config["dataset"])
df = loader.load()

# ── Feature extraction ────────────────────────────────────────────────────────
# must happen BEFORE split so windowed features have full chronological context.
# shuffling/splitting after this point is safe since each row is now a
# self-contained feature vector — window context is already baked in.
feature_params = config.get("feature_params", {})
X, y, global_median_iat = build_features(
    df,
    feature_list=config["features"],
    feature_params=feature_params,
)

# ── Temporal split ────────────────────────────────────────────────────────────
# per-file time-based split to avoid temporal leakage between adjacent windows.
# see utils/split.py for full rationale.
split_cfg = config["split"]
if "origin" in df.columns:
    train_df, val_df, test_df = temporal_split_augmented(
        df, train_size=split_cfg["train_size"],
        val_size=split_cfg["val_size"], test_size=split_cfg["test_size"],
    )
else:
    train_df, val_df, test_df = temporal_split(
        df, train_size=split_cfg["train_size"],
        val_size=split_cfg["val_size"], test_size=split_cfg["test_size"],
    )

X_train = X.loc[train_df.index]
y_train = y.loc[train_df.index]
X_val   = X.loc[val_df.index]
y_val   = y.loc[val_df.index]
X_test  = X.loc[test_df.index]
y_test  = y.loc[test_df.index]

# ── Model training ────────────────────────────────────────────────────────────
model = get_model(config["model"], params=config["params"])
train_with_spinner(model, X_train, y_train)

# ── Evaluation ────────────────────────────────────────────────────────────────
predictions     = model.predict(X_test)
val_predictions = model.predict(X_val)

metrics     = compute_metrics(y_test, predictions)
val_metrics = compute_metrics(y_val, val_predictions)
benchmark   = run_benchmark(model, X_test, thresholds=config.get("thresholds"))

# ── Results folder ────────────────────────────────────────────────────────────
run_name   = input("\nName this experiment run: ").strip()
safe_name  = re.sub(r"[^a-zA-Z0-9_-]", "_", run_name.replace(" ", "_"))
timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results/{safe_name}_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# ── Save results ──────────────────────────────────────────────────────────────
with open(os.path.join(results_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

with open(os.path.join(results_dir, "val_metrics.json"), "w") as f:
    json.dump(val_metrics, f, indent=2)

generate_visuals(metrics, results_dir)
generate_visuals(val_metrics, results_dir, prefix="val_")

with open(os.path.join(results_dir, "benchmark.json"), "w") as f:
    json.dump(benchmark, f, indent=2)

with open(os.path.join(results_dir, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

# save feature params and global_median_iat for inference reproducibility
with open(os.path.join(results_dir, "feature_params.json"), "w") as f:
    json.dump(feature_params, f, indent=2)

if global_median_iat is not None:
    with open(os.path.join(results_dir, "global_median_iat.json"), "w") as f:
        json.dump({"global_median_iat": global_median_iat}, f, indent=2)

shutil.copy(config_path, os.path.join(results_dir, "config.yaml"))

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nTest  accuracy:   {metrics['accuracy']:.4f}")
print(f"Val   accuracy:   {val_metrics['accuracy']:.4f}")
print(f"Benchmark:        {benchmark}")
print(f"Results saved to: {results_dir}")