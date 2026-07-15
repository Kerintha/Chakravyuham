"""
normal.py
Generates clean, label="normal" CAN traffic from a vehicle profile.
Output schema matches the existing pipeline's expected clean dataframe:
timestamp, can_id, dlc, data_0..7, label, source_file
"""

import yaml
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from synth_utils.scheduler import generate_timestamps
from synth_utils.payloads import PayloadGenerator


def load_profile(path):
    with open(path) as f:
        return yaml.safe_load(f)


def generate_normal_traffic(profile_path, duration_s, seed=42, source_file="synthetic_normal"):
    """Returns a dataframe of normal traffic for the full ECU population in the profile."""
    profile = load_profile(profile_path)
    rng = np.random.default_rng(seed)

    rows = []
    for ecu in profile["ecus"]:
        can_id_int = int(ecu["id"], 16)
        timestamps = generate_timestamps(
            period_ms=ecu["period_ms"],
            jitter_ms=ecu["jitter_ms"],
            duration_s=duration_s,
            rng=rng,
        )
        payload_gen = PayloadGenerator(ecu["bytes"], rng=rng)
        for ts in timestamps:
            payload = payload_gen.next_payload()
            rows.append((ts, can_id_int, ecu["dlc"], *payload, "normal", source_file))

    cols = ["timestamp", "can_id", "dlc"] + [f"data_{i}" for i in range(8)] + ["label", "source_file"]
    df = pd.DataFrame(rows, columns=cols)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_normal_traffic("synthetic_data/profiles/vehicle_a.yaml", duration_s=60)
    print(df.head(10))
    print(f"\nTotal rows: {len(df)}")
    print(f"Unique IDs: {df['can_id'].nunique()}")