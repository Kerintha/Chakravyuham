import numpy as np
from features.state import shannon_entropy

PAYLOAD_FEATURE_NAMES = (
    ["byte_mean", "byte_variance", "hamming", "window_entropy"]
    + [f"rolling_byte_mean_{i}" for i in range(8)]
    + [f"rolling_byte_std_{i}" for i in range(8)]
)


def extract_incremental(current_payload, current_payload_arr,
                        current_entropy, id_state, n_incl):
    """
    Computes payload features using incremental state.
    O(1) — pure arithmetic on running sums, no array rebuilding,
    no repeated entropy computation.

    current_payload:     list of 8 ints (raw byte values)
    current_payload_arr: np.array(current_payload) — precomputed in pipeline
    current_entropy:     shannon_entropy(current_payload) — precomputed in pipeline
    id_state:            per-ID state with running sums (from features/state.py)
    n_incl:              window count including current message

    Features:
      byte_mean            — mean of current message's 8 bytes.
      byte_variance        — variance of current message's 8 bytes.
      hamming              — bytes differing from previous payload of same ID.
                             catches fuzzy (random payloads differ maximally)
                             and impersonation (spoofed payload differs from normal).
                             padded with 0 if no previous payload.
      window_entropy       — mean Shannon entropy across window payloads.
                             catches fuzzy (random injection = sustained high entropy).
                             more robust than per-message entropy alone.
      rolling_byte_mean_i  — mean of byte position i across window.
                             establishes per-byte fingerprint per CAN ID.
      rolling_byte_std_i   — std of byte position i across window.
                             high std = byte varying abnormally.
    """
    # per-message (no window context needed)
    byte_mean     = float(current_payload_arr.mean())
    byte_variance = float(current_payload_arr.var())

    # hamming vs previous payload of same CAN ID — one-step lookback
    if id_state["payloads"]:
        prev_payload = id_state["payloads"][-1]
        hamming = float(sum(a != b for a, b in zip(current_payload, prev_payload)))
    else:
        hamming = 0.0  # padding: no previous payload, assume nothing changed

    # window entropy — incremental, O(1)
    entropy_sum_incl = id_state["entropy_sum"] + current_entropy
    window_entropy   = entropy_sum_incl / n_incl

    # per-byte rolling mean and std — incremental, O(1)
    byte_sums_incl    = id_state["byte_sums"]    + current_payload_arr
    byte_sum_sqs_incl = id_state["byte_sum_sqs"] + current_payload_arr * current_payload_arr

    rolling_byte_mean = byte_sums_incl / n_incl

    if n_incl > 1:
        variance         = byte_sum_sqs_incl / n_incl - (byte_sums_incl / n_incl) ** 2
        rolling_byte_std = np.sqrt(np.maximum(variance, 0.0))
    else:
        rolling_byte_std = np.zeros(8)

    features = {
        "byte_mean":      byte_mean,
        "byte_variance":  byte_variance,
        "hamming":        hamming,
        "window_entropy": window_entropy,
    }
    for i in range(8):
        features[f"rolling_byte_mean_{i}"] = float(rolling_byte_mean[i])
        features[f"rolling_byte_std_{i}"]  = float(rolling_byte_std[i])

    return features


def extract(message, id_state):
    """
    Single-message extraction for real-time inference.
    Computes current payload values then delegates to extract_incremental.
    Kept as separate entry point so realtime.py stays clean.
    """
    current_payload     = [message[f"data_{i}"] for i in range(8)]
    current_payload_arr = np.array(current_payload, dtype=np.float64)
    current_entropy     = shannon_entropy(current_payload)
    n_incl              = id_state["n"] + 1

    return extract_incremental(
        current_payload, current_payload_arr,
        current_entropy, id_state, n_incl
    )