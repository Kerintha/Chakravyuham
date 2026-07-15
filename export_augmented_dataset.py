import glob
import os
import pandas as pd
from data.loaders.registry import get_loader

REAL_DATASET_NAME = "otids"
SYNTHETIC_GLOB = "synthetic_data/output/train/*.csv"
OUTPUT_NAME = "otids_augmented"   # must match OTIDSAugmentedLoader.name below

loader = get_loader(REAL_DATASET_NAME)
real_df = loader.load()
real_df["origin"] = "real"

synthetic_paths = sorted(glob.glob(SYNTHETIC_GLOB))
if not synthetic_paths:
    raise FileNotFoundError(f"No synthetic files at {SYNTHETIC_GLOB} -- run run_augmented_scenarios.py first")

synthetic_dfs = []
for p in synthetic_paths:
    df = pd.read_csv(p)
    df["origin"] = "synthetic"
    synthetic_dfs.append(df)
synthetic_df = pd.concat(synthetic_dfs, ignore_index=True)

combined = pd.concat([real_df, synthetic_df], ignore_index=True)

processed_dir = "data/processed"
os.makedirs(processed_dir, exist_ok=True)
out_path = os.path.join(processed_dir, f"{OUTPUT_NAME}_clean.csv")
combined.to_csv(out_path, index=False)

print(f"Saved {len(combined)} rows -> {out_path}")
print(combined["origin"].value_counts())
print(combined["label"].value_counts())