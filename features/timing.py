import math

TIMING_FEATURE_NAMES = [
    "iat",
    "rolling_mean_iat",
    "jitter",
    "rolling_id_frequency",
]


def extract_incremental(current_iat, id_state, n_incl):
    """
    Computes timing features using incremental state.
    O(1) — pure arithmetic on running sums, no array rebuilding.

    current_iat: IAT for current message, already computed in pipeline.py
    id_state:    per-ID state with running sums (from features/state.py)
    n_incl:      window count including current message (id_state["n"] + 1)

    Features:
      iat                  — time since last message of same CAN ID.
                             directly detects DoS (compressed timing).
      rolling_mean_iat     — mean IAT over window including current message.
      jitter               — std of IAT over window.
                             detects impersonation (disturbed transmission schedule).
      rolling_id_frequency — count of this CAN ID in current window.
                             directly detects ID flooding.
    """
    iat_sum_incl    = id_state["iat_sum"] + current_iat
    rolling_mean_iat = iat_sum_incl / n_incl

    if n_incl > 1:
        iat_sum_sq_incl = id_state["iat_sum_sq"] + current_iat * current_iat
        variance = iat_sum_sq_incl / n_incl - (iat_sum_incl / n_incl) ** 2
        jitter = math.sqrt(max(variance, 0.0))
    else:
        jitter = 0.0

    return {
        "iat":                  current_iat,
        "rolling_mean_iat":     rolling_mean_iat,
        "jitter":               jitter,
        "rolling_id_frequency": n_incl,
    }


def extract(message, id_state, global_median_iat):
    """
    Single-message extraction for real-time inference.
    Same as extract_incremental — just computes current_iat
    from the message and state, then delegates.
    Kept as a separate entry point so realtime.py doesn't
    need to know about the incremental internals.
    """
    current_ts = message["timestamp"]
    if id_state["timestamps"]:
        current_iat = current_ts - id_state["timestamps"][-1]
    else:
        current_iat = global_median_iat

    n_incl = id_state["n"] + 1
    return extract_incremental(current_iat, id_state, n_incl)