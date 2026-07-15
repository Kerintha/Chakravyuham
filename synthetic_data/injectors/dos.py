"""
dos.py
DoS injector. Floods a target CAN ID at high frequency into an existing
normal traffic stream, over [start_t, end_t]. Modular — does not import or
know about any other injector.

payload_mode:
  "fixed"           - reproduces real HCRL behavior: literal constant all-zero
                       payload on every injected frame (the dataset artifact).
  "varied_realistic" - payload varies per frame using a plausible byte-range
                       generator, isolating the timing signal from the payload
                       artifact.
"""

import numpy as np
import pandas as pd


def inject_dos(
    normal_df,
    target_id,
    start_t,
    end_t,
    flood_iat_mean_ms=0.91,
    flood_iat_std_ms=1.16,
    payload_mode="fixed",
    seed=42,
):
    """
    normal_df: dataframe with schema timestamp, can_id, dlc, data_0..7, label, source_file
    target_id: int, CAN ID to flood (use 0x000 to replicate real HCRL, or any other
               legitimate ID present in the profile to test ID-independence)
    Returns: new dataframe = normal_df with DoS frames merged in and labeled "dos".
    """
    rng = np.random.default_rng(seed)

    flood_rows = []
    t = start_t
    while t < end_t:
        if payload_mode == "fixed":
            payload = [0] * 8
        elif payload_mode == "varied_realistic":
            payload = list(rng.integers(0, 256, size=8))
        else:
            raise ValueError(f"Unknown payload_mode: {payload_mode}")

        flood_rows.append((t, target_id, 8, *payload, "dos", "synthetic_dos"))

        step_ms = rng.normal(flood_iat_mean_ms, flood_iat_std_ms)
        step_ms = max(step_ms, 0.05)  # floor, avoid non-positive/degenerate steps
        t += step_ms / 1000.0

    cols = ["timestamp", "can_id", "dlc"] + [f"data_{i}" for i in range(8)] + ["label", "source_file"]
    flood_df = pd.DataFrame(flood_rows, columns=cols)

    merged = pd.concat([normal_df, flood_df], ignore_index=True)
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    return merged