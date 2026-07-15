"""
build_vehicle_profile.py
Builds a vehicle profile YAML from per-ID stats CSVs produced by extract_stats.py.

Usage:
    python build_vehicle_profile.py --input data/analysis/per_id_stats_attack_free.csv \
                                     --output synthetic_data/profiles/vehicle_a.yaml \
                                     --name vehicle_a
"""

import argparse
import pandas as pd
import yaml

STATIC_STD_THRESHOLD = 0.5  # byte-position std below this => treated as constant


def build_ecu_entry(row):
    entry = {
        "id": row["can_id"],
        "period_ms": round(float(row["mean_iat_ms"]), 3),
        "jitter_ms": round(float(row["std_iat_ms"]), 3),
        "dlc": int(round(row["mean_dlc"])),
        "bytes": {},
    }
    for i in range(8):
        mean = row.get(f"byte{i}_mean")
        std = row.get(f"byte{i}_std")
        bmin = row.get(f"byte{i}_min")
        bmax = row.get(f"byte{i}_max")
        if pd.isna(mean):
            continue
        if std < STATIC_STD_THRESHOLD:
            entry["bytes"][i] = {"mode": "static", "value": int(round(mean))}
        else:
            entry["bytes"][i] = {
                "mode": "range",
                "min": int(bmin),
                "max": int(bmax),
                "mean": round(float(mean), 2),
                "std": round(float(std), 2),
            }
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="per_id_stats_<source>.csv (attack_free rows)")
    ap.add_argument("--output", required=True, help="output profile YAML path")
    ap.add_argument("--name", required=True, help="vehicle profile name")
    ap.add_argument("--source-filter", default=None,
                     help="if input has a 'source' column, filter to this value (e.g. attack_free)")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.source_filter and "source" in df.columns:
        df = df[df["source"] == args.source_filter]

    ecus = [build_ecu_entry(row) for _, row in df.iterrows()]
    ecus.sort(key=lambda e: e["period_ms"])

    profile = {
        "vehicle_name": args.name,
        "description": f"Profile derived from {args.input}",
        "num_ecus": len(ecus),
        "ecus": ecus,
    }

    with open(args.output, "w") as f:
        yaml.dump(profile, f, sort_keys=False, default_flow_style=False)

    print(f"Wrote {args.output} with {len(ecus)} ECUs")


if __name__ == "__main__":
    main()