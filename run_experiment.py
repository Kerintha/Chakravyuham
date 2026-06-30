import sys
import os
import json
import pickle
import shutil
import datetime
import re
import time
import threading
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.config import load_config
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


# ── Config selection ──────────────────────────────────────────────────────────
config_path = select_config()
config = load_config(config_path)

# ── Load dataset ──────────────────────────────────────────────────────────────
loader = get_loader(config["dataset"])
df = loader.load()

# ── Feature extraction ────────────────────────────────────────────────────────
X, y = build_features(df, feature_list=config["features"])

# ── Train / val / test split ──────────────────────────────────────────────────
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y,
    test_size=config["split"]["test_size"],
    stratify=y if config["split"]["stratify"] else None,
    shuffle=config["split"]["shuffle"],
    random_state=config["params"].get("random_state", 42),
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=config["split"]["val_size"],
    stratify=y_train_full if config["split"]["stratify"] else None,
    shuffle=config["split"]["shuffle"],
    random_state=config["params"].get("random_state", 42),
)

# ── Model training ────────────────────────────────────────────────────────────
model = get_model(config["model"], params=config["params"])
train_with_spinner(model, X_train, y_train)

# ── Evaluation ────────────────────────────────────────────────────────────────
predictions = model.predict(X_test)
val_predictions = model.predict(X_val)

metrics = compute_metrics(y_test, predictions)
val_metrics = compute_metrics(y_val, val_predictions)
benchmark = run_benchmark(model, X_test, thresholds=config.get("thresholds"))

# ── Results folder naming ─────────────────────────────────────────────────────
run_name = input("\nName this experiment run: ").strip()
safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", run_name.replace(" ", "_"))
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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

shutil.copy(config_path, os.path.join(results_dir, "config.yaml"))

# ── Print summary ─────────────────────────────────────────────────────────────
print(f"\nTest  accuracy:  {metrics['accuracy']:.4f}")
print(f"Val   accuracy:  {val_metrics['accuracy']:.4f}")
print(f"Benchmark:       {benchmark}")
print(f"Results saved to: {results_dir}")