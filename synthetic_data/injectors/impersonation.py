"""
impersonation.py
Spoofed frames stack on top of the target ID's existing traffic (matches real
HCRL: spoofed count adds to baseline count, doesn't replace it).

payload_mode:
  "artifact" - forces the target ID's normally-static bytes to nonzero
               (reproduces the real 0x164 shortcut; control scenario only).
  "subtle"   - payload stays WITHIN the ID's real per-byte range (same walk
               logic as normal generation), only timing jitter is widened.
               This is the actual stress-test mode, prioritize training on it.
"""

import numpy as np
import pandas as pd


def inject_impersonation(
    normal_df,
    target_id,
    ecu_profile,
    start_t,
    end_t,
    payload_mode="subtle",
    jitter_multiplier=1.5,
    seed=42,
):
    rng = np.random.default_rng(seed)

    period_ms = ecu_profile["period_ms"]
    jitter_ms = ecu_profile["jitter_ms"] * jitter_multiplier
    dlc = ecu_profile["dlc"]
    byte_rules = ecu_profile["bytes"]

    period_s = period_ms / 1000.0
    jitter_s = max(jitter_ms, 0.0) / 1000.0

    state = {idx: rule["mean"] for idx, rule in byte_rules.items() if rule["mode"] == "range"}

    spoof_rows = []
    t = start_t
    while t < end_t:
        payload = [0] * 8
        if payload_mode == "artifact":
            for idx in range(8):
                rule = byte_rules.get(idx)
                if rule is None:
                    continue
                if rule["mode"] == "static":
                    payload[idx] = int((rule["value"] + rng.integers(1, 255)) % 256)
                else:
                    payload[idx] = int(round(rule["mean"]))
        elif payload_mode == "subtle":
            for idx in range(8):
                rule = byte_rules.get(idx)
                if rule is None:
                    continue
                if rule["mode"] == "static":
                    payload[idx] = rule["value"]
                else:
                    step_scale = max(rule["std"] * 0.15, 0.5)
                    new_val = float(np.clip(state[idx] + rng.normal(0, step_scale), rule["min"], rule["max"]))
                    state[idx] = new_val
                    payload[idx] = int(round(new_val))
        else:
            raise ValueError(f"Unknown payload_mode: {payload_mode}")

        spoof_rows.append((t, target_id, dlc, *payload, "impersonation", "synthetic_impersonation"))
        step = period_s + rng.normal(0, jitter_s)
        step = max(step, period_s * 0.1)
        t += step

    cols = ["timestamp", "can_id", "dlc"] + [f"data_{i}" for i in range(8)] + ["label", "source_file"]
    spoof_df = pd.DataFrame(spoof_rows, columns=cols)
    merged = pd.concat([normal_df, spoof_df], ignore_index=True)
    return merged.sort_values("timestamp").reset_index(drop=True)