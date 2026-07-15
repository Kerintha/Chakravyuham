"""
fuzzy.py
Fuzzy injector. Uses only legitimate IDs already in the vehicle profile (real
HCRL fuzzy attacks introduce zero novel IDs). Payload deviation is calibrated
relative to each ID's OWN baseline range, not an absolute threshold.

payload_mode:
  "artifact"  - fully random bytes 0-255, ignoring the ID's real structure
                entirely (closest to a naive/obvious fuzzer, useful as a
                control scenario, same role as dos_fixed_payload_baseline).
  "realistic" - deviates from the ID's real static/range rules but stays
                targeted: static bytes get shifted off their fixed value,
                range bytes get widened beyond their real span. This is the
                mode to prioritize for training.
"""

import numpy as np
import pandas as pd


def inject_fuzzy(
    normal_df,
    target_id,
    ecu_profile,
    start_t,
    end_t,
    payload_mode="realistic",
    fuzz_rate_multiplier=1.0,
    seed=42,
):
    rng = np.random.default_rng(seed)

    period_ms = ecu_profile["period_ms"] / max(fuzz_rate_multiplier, 0.01)
    jitter_ms = ecu_profile["jitter_ms"]
    dlc = ecu_profile["dlc"]
    byte_rules = ecu_profile["bytes"]

    period_s = period_ms / 1000.0
    jitter_s = max(jitter_ms, 0.0) / 1000.0

    fuzz_rows = []
    t = start_t
    while t < end_t:
        if payload_mode == "artifact":
            payload = list(rng.integers(0, 256, size=8))
        elif payload_mode == "realistic":
            payload = [0] * 8
            for idx in range(8):
                rule = byte_rules.get(idx)
                if rule is None:
                    payload[idx] = int(rng.integers(0, 256))
                    continue
                if rule["mode"] == "static":
                    offset = rng.integers(20, 235)
                    payload[idx] = int((rule["value"] + offset) % 256)
                else:
                    span = max(rule["max"] - rule["min"], 1)
                    lo = max(0, rule["min"] - span)
                    hi = min(255, rule["max"] + span)
                    payload[idx] = int(rng.integers(lo, hi + 1))
        else:
            raise ValueError(f"Unknown payload_mode: {payload_mode}")

        fuzz_rows.append((t, target_id, dlc, *payload, "fuzzy", "synthetic_fuzzy"))
        step = period_s + rng.normal(0, jitter_s)
        step = max(step, period_s * 0.1)
        t += step

    cols = ["timestamp", "can_id", "dlc"] + [f"data_{i}" for i in range(8)] + ["label", "source_file"]
    fuzz_df = pd.DataFrame(fuzz_rows, columns=cols)
    merged = pd.concat([normal_df, fuzz_df], ignore_index=True)
    return merged.sort_values("timestamp").reset_index(drop=True)