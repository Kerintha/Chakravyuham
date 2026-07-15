"""
scheduler.py
Timestamp generation: never generate timestamps independently.
Each next timestamp = previous + period + small jitter.
"""

import numpy as np


def generate_timestamps(period_ms, jitter_ms, duration_s, start_t=0.0, rng=None):
    """Generate a monotonically increasing timestamp series for one ECU.

    period_ms: nominal transmission period
    jitter_ms: std of gaussian jitter added to each period
    duration_s: how long to generate for
    start_t: starting timestamp (seconds)
    Returns: list of timestamps (seconds), strictly increasing.
    """
    rng = rng or np.random.default_rng()
    period_s = period_ms / 1000.0
    jitter_s = max(jitter_ms, 0.0) / 1000.0

    timestamps = []
    t = start_t
    end_t = start_t + duration_s
    while t < end_t:
        timestamps.append(t)
        step = period_s + rng.normal(0, jitter_s)
        step = max(step, period_s * 0.1)  # never go negative/near-zero
        t += step
    return timestamps