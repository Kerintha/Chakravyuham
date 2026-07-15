"""
extract_stats.py
Pulls per-ID timing, payload, and entropy statistics from the raw HCRL
Car-Hacking dataset files to inform synthetic vehicle profiles and
attack injector calibration.

Usage: python extract_stats.py
Expects files in data/raw/otids/ (per the existing project layout).
"""

import math
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

RAW_DIR = Path("data/raw/otids")
OUT_DIR = Path("data/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "attack_free": "Attack_free_dataset.txt",
    "dos": "DoS_attack_dataset.txt",
    "fuzzy": "Fuzzy_attack_dataset.txt",
    "impersonation": "Impersonation_attack_dataset.txt",
}


def parse_file(path):
    """Parse raw HCRL-format lines into a list of dicts.
    Format: Timestamp: <t>   ID: <hex>   000   DLC: <n>   <n hex bytes>
    """
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                ts = float(parts[1])
                can_id = int(parts[3], 16)
                dlc = int(parts[6])
                data_bytes = [int(b, 16) for b in parts[7:7 + dlc]]
                # pad short payloads with -1 sentinel so array lengths align
                while len(data_bytes) < 8:
                    data_bytes.append(-1)
                rows.append((ts, can_id, dlc, *data_bytes))
            except (ValueError, IndexError):
                continue
    cols = ["timestamp", "can_id", "dlc"] + [f"data_{i}" for i in range(8)]
    df = pd.DataFrame(rows, columns=cols)
    return df


def shannon_entropy(byte_vals):
    """Shannon entropy (bits) over a list/array of byte values (0-255), ignoring -1 padding."""
    vals = [b for b in byte_vals if b >= 0]
    if not vals:
        return 0.0
    counts = np.bincount(vals, minlength=256)
    probs = counts[counts > 0] / len(vals)
    return float(-np.sum(probs * np.log2(probs)))


def per_id_stats(df, label):
    """Compute per-CAN-ID timing + payload + entropy stats for one dataframe."""
    results = []
    for can_id, g in df.groupby("can_id"):
        g = g.sort_values("timestamp")
        iats = g["timestamp"].diff().dropna().values
        byte_cols = [f"data_{i}" for i in range(8)]

        row_entropies = g[byte_cols].apply(
            lambda r: shannon_entropy(r.values), axis=1
        )

        stat = {
            "source": label,
            "can_id": hex(can_id),
            "count": len(g),
            "mean_iat_ms": np.mean(iats) * 1000 if len(iats) else np.nan,
            "std_iat_ms": np.std(iats) * 1000 if len(iats) else np.nan,
            "min_iat_ms": np.min(iats) * 1000 if len(iats) else np.nan,
            "max_iat_ms": np.max(iats) * 1000 if len(iats) else np.nan,
            "mean_dlc": g["dlc"].mean(),
            "mean_entropy": row_entropies.mean(),
            "std_entropy": row_entropies.std(),
        }

        # per-byte-position stats (only over valid, non-padded values)
        for i in range(8):
            col = g[f"data_{i}"]
            valid = col[col >= 0]
            stat[f"byte{i}_mean"] = valid.mean() if len(valid) else np.nan
            stat[f"byte{i}_std"] = valid.std() if len(valid) else np.nan
            stat[f"byte{i}_min"] = valid.min() if len(valid) else np.nan
            stat[f"byte{i}_max"] = valid.max() if len(valid) else np.nan
            stat[f"byte{i}_unique_vals"] = valid.nunique() if len(valid) else 0

        results.append(stat)
    return pd.DataFrame(results)


def check_dos_signature(dos_df):
    """Check whether DoS-attributed frames (can_id == 0x000) share a
    constant/near-constant payload, and report actual flood rate."""
    dos_rows = dos_df[dos_df["can_id"] == 0x000].sort_values("timestamp")
    byte_cols = [f"data_{i}" for i in range(8)]

    unique_payloads = dos_rows[byte_cols].drop_duplicates()
    iats = dos_rows["timestamp"].diff().dropna().values

    print("\n--- DoS signature check (can_id == 0x000) ---")
    print(f"Total 0x000 frames: {len(dos_rows)}")
    print(f"Unique payload combinations: {len(unique_payloads)}")
    if len(unique_payloads) <= 5:
        print("Payload values found:")
        print(unique_payloads.to_string(index=False))
    print(f"Mean IAT: {np.mean(iats)*1000:.4f} ms | Std IAT: {np.std(iats)*1000:.4f} ms")
    print(f"Min IAT: {np.min(iats)*1000:.4f} ms | Max IAT: {np.max(iats)*1000:.4f} ms")
    return unique_payloads, iats


def check_fuzzy_id_randomization(fuzzy_df, cutoff_s=250):
    """Check whether fuzzy attack frames (t >= cutoff) use random CAN IDs
    or only randomize payload on pre-existing legitimate IDs."""
    fuzzy_df = fuzzy_df.copy()
    fuzzy_df["t_norm"] = fuzzy_df["timestamp"] - fuzzy_df["timestamp"].min()
    normal_region = fuzzy_df[fuzzy_df["t_norm"] < cutoff_s]
    attack_region = fuzzy_df[fuzzy_df["t_norm"] >= cutoff_s]

    normal_ids = set(normal_region["can_id"].unique())
    attack_ids = set(attack_region["can_id"].unique())
    new_ids_in_attack = attack_ids - normal_ids

    print("\n--- Fuzzy ID randomization check ---")
    print(f"Unique IDs in normal region (t<{cutoff_s}s): {len(normal_ids)}")
    print(f"Unique IDs in attack region (t>={cutoff_s}s): {len(attack_ids)}")
    print(f"IDs appearing ONLY in attack region: {len(new_ids_in_attack)}")
    if len(new_ids_in_attack) > 0:
        sample = sorted(list(new_ids_in_attack))[:15]
        print(f"Sample of attack-only IDs: {[hex(x) for x in sample]}")
        print(">> Fuzzy attack DOES inject novel/random CAN IDs, not just payload on known IDs.")
    else:
        print(">> Fuzzy attack appears to reuse existing IDs only (payload-only fuzzing).")


def check_impersonation_payload(imp_df, target_id=0x164, cutoff_s=250):
    """Compare 0x164's normal payload distribution vs its payload during
    the labeled impersonation window."""
    imp_df = imp_df.copy()
    imp_df["t_norm"] = imp_df["timestamp"] - imp_df["timestamp"].min()
    id_rows = imp_df[imp_df["can_id"] == target_id]

    normal_rows = id_rows[id_rows["t_norm"] < cutoff_s]
    attack_rows = id_rows[id_rows["t_norm"] >= cutoff_s]

    byte_cols = [f"data_{i}" for i in range(8)]

    print(f"\n--- Impersonation payload check (id={hex(target_id)}) ---")
    print(f"Normal-region frames: {len(normal_rows)} | Attack-region frames: {len(attack_rows)}")
    if len(normal_rows) and len(attack_rows):
        print("Normal payload byte means: ", normal_rows[byte_cols].mean().round(2).tolist())
        print("Attack payload byte means: ", attack_rows[byte_cols].mean().round(2).tolist())
        print("Normal payload byte stds:  ", normal_rows[byte_cols].std().round(2).tolist())
        print("Attack payload byte stds:  ", attack_rows[byte_cols].std().round(2).tolist())

        norm_iats = normal_rows.sort_values("timestamp")["timestamp"].diff().dropna().values
        att_iats = attack_rows.sort_values("timestamp")["timestamp"].diff().dropna().values
        print(f"Normal mean IAT: {np.mean(norm_iats)*1000:.3f} ms | Attack mean IAT: {np.mean(att_iats)*1000:.3f} ms")


def main():
    dfs = {}
    for label, fname in FILES.items():
        path = RAW_DIR / fname
        if not path.exists():
            print(f"WARNING: {path} not found, skipping.")
            continue
        print(f"Parsing {fname} ...")
        dfs[label] = parse_file(path)
        print(f"  -> {len(dfs[label])} rows parsed")

    # per-ID stats for every file, saved separately
    all_stats = []
    for label, df in dfs.items():
        stats = per_id_stats(df, label)
        stats.to_csv(OUT_DIR / f"per_id_stats_{label}.csv", index=False)
        all_stats.append(stats)
        print(f"Saved per_id_stats_{label}.csv ({len(stats)} unique IDs)")

    if all_stats:
        pd.concat(all_stats, ignore_index=True).to_csv(
            OUT_DIR / "per_id_stats_all.csv", index=False
        )

    # targeted checks
    if "dos" in dfs:
        check_dos_signature(dfs["dos"])
    if "fuzzy" in dfs:
        check_fuzzy_id_randomization(dfs["fuzzy"])
    if "impersonation" in dfs:
        check_impersonation_payload(dfs["impersonation"])

    print(f"\nAll per-ID CSVs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()