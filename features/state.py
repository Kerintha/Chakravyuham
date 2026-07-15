from collections import deque
import numpy as np


def shannon_entropy(byte_list):
    """
    Shannon entropy of a list of byte values.
    Computed once per message in pipeline, cached in state.
    Moved here so pipeline.py and realtime.py share identical computation.
    """
    arr = np.array(byte_list, dtype=np.float32)
    _, counts = np.unique(arr, return_counts=True)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def make_id_state():
    """
    Initializes incremental rolling state for one CAN ID.

    Uses deques (O(1) popleft) instead of lists (O(n) pop(0)).
    Maintains running sums so mean/std/entropy are computed
    incrementally in O(1) instead of rebuilding full arrays each message.

    Fields:
      timestamps     — deque of recent timestamps within window
      payloads       — deque of recent payloads (list of 8 ints each)
      entropies      — deque of precomputed entropy per payload
      iats           — deque of computed IATs within window
      iat_sum        — running sum of IATs in window
      iat_sum_sq     — running sum of squared IATs (for std)
      byte_sums      — running sum per byte position (8 floats)
      byte_sum_sqs   — running sum of squares per byte position
      entropy_sum    — running sum of entropies in window
      n              — current window count
    """
    return {
        "timestamps":   deque(),
        "payloads":     deque(),
        "entropies":    deque(),
        "iats":         deque(),
        "iat_sum":      0.0,
        "iat_sum_sq":   0.0,
        "byte_sums":    np.zeros(8, dtype=np.float64),
        "byte_sum_sqs": np.zeros(8, dtype=np.float64),
        "entropy_sum":  0.0,
        "n":            0,
    }


def prune_state(id_state, cutoff):
    """
    Removes entries older than cutoff from state.
    O(k) where k = number of entries being removed.
    Amortized O(1) per message across the full dataset since
    each entry is added once and removed once.
    """
    while id_state["timestamps"] and id_state["timestamps"][0] < cutoff:
        id_state["timestamps"].popleft()
        old_payload  = id_state["payloads"].popleft()
        old_entropy  = id_state["entropies"].popleft()
        old_iat      = id_state["iats"].popleft()

        id_state["iat_sum"]    -= old_iat
        id_state["iat_sum_sq"] -= old_iat * old_iat

        old_arr = np.array(old_payload, dtype=np.float64)
        id_state["byte_sums"]    -= old_arr
        id_state["byte_sum_sqs"] -= old_arr * old_arr

        id_state["entropy_sum"] -= old_entropy
        id_state["n"]           -= 1


def update_state(id_state, timestamp, payload_bytes, iat, entropy):
    """
    Adds current message to state after feature extraction.
    O(1) — single append + scalar arithmetic.
    Must be called AFTER feature extraction so current message
    is not included in its own window statistics.
    """
    id_state["timestamps"].append(timestamp)
    id_state["payloads"].append(payload_bytes)
    id_state["entropies"].append(entropy)
    id_state["iats"].append(iat)

    id_state["iat_sum"]    += iat
    id_state["iat_sum_sq"] += iat * iat

    arr = np.array(payload_bytes, dtype=np.float64)
    id_state["byte_sums"]    += arr
    id_state["byte_sum_sqs"] += arr * arr

    id_state["entropy_sum"] += entropy
    id_state["n"]           += 1