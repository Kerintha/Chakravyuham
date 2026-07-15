import numpy as np
import pandas as pd
from tqdm import tqdm

from features import basic, timing, payload
from features.state import shannon_entropy, make_id_state, prune_state, update_state

FEATURE_REGISTRY = {
    "basic":   True,
    "timing":  True,
    "payload": True,
    # "behavioral": True,   # future addition
}


def _compute_global_median_iat(df):
    """
    Computes global median inter-arrival time across all CAN IDs.
    Used as padding value for timing features when a CAN ID has
    no previous message in the current window (cold start).
    Computed once before the main feature extraction loop.
    """
    iats    = []
    last_ts = {}
    for row in df.itertuples():
        can_id = row.can_id
        ts     = row.timestamp
        if can_id in last_ts:
            iats.append(ts - last_ts[can_id])
        last_ts[can_id] = ts
    return float(np.median(iats)) if iats else 0.01


def build_features(df, feature_list, feature_params=None):
    """
    Builds feature matrix X and label vector y from the clean dataframe.

    df must be in chronological order (as returned by loader) since timing
    and payload features depend on message order. Shuffling/splitting happens
    AFTER this function in run_experiment.py, never before.

    Complexity: O(N) amortized — each message is added to state once and
    pruned once. Feature computation is O(1) per message via incremental
    statistics (running sums), not O(window_size) per message.

    feature_list:   list of strings from config, e.g. ["basic", "timing", "payload"]
    feature_params: dict from config, e.g. {"window_size_ms": 50}

    Returns: (X, y, global_median_iat)
      X:                 feature matrix (DataFrame)
      y:                 label vector (Series)
      global_median_iat: float, saved for inference cold-start padding
    """
    for name in feature_list:
        if name not in FEATURE_REGISTRY:
            raise ValueError(f"Unknown feature set: '{name}'")

    feature_params  = feature_params or {}
    window_size_ms  = feature_params.get("window_size_ms", 50)
    window_size_s   = window_size_ms / 1000.0
    needs_windowed  = any(f in feature_list for f in ["timing", "payload"])

    basic_df = basic.extract(df) if "basic" in feature_list else None

    if not needs_windowed:
        X = basic_df.reset_index(drop=True)
        y = df["label"].reset_index(drop=True)
        return X, y, None

    global_median_iat = _compute_global_median_iat(df)

    timing_rows  = [] if "timing"  in feature_list else None
    payload_rows = [] if "payload" in feature_list else None

    # per-ID rolling state — incremental, O(1) updates/pruning
    # CAN ID is the grouping key — never a model feature (see basic.py comment)
    per_id_state = {}
    previous_source_file = None

    for row in tqdm(df.itertuples(), total=len(df),
                    desc="Extracting windowed features", unit="msgs"):

        can_id          = row.can_id
        current_ts      = row.timestamp
        current_payload = [getattr(row, f"data_{i}") for i in range(8)]

        # reset all per-ID rolling state at every session boundary -- prevents a
        # message's IAT/window features from being computed against a previous,
        # unrelated session's timestamps and payload history. source_file is
        # constant within one real file or one synthetic scenario, and changes
        # exactly at each of the 16 session boundaries in the combined dataset.
        if row.source_file != previous_source_file:
            per_id_state = {}
            previous_source_file = row.source_file

        if can_id not in per_id_state:
            per_id_state[can_id] = make_id_state()

        id_state = per_id_state[can_id]

        # O(k) prune — amortized O(1) per message
        prune_state(id_state, current_ts - window_size_s)

        # compute entropy once — reused by both payload features and state update
        current_entropy     = shannon_entropy(current_payload)
        current_payload_arr = np.array(current_payload, dtype=np.float64)

        # compute current IAT — used by timing features and state update
        if id_state["timestamps"]:
            current_iat = current_ts - id_state["timestamps"][-1]
        else:
            current_iat = global_median_iat  # cold-start padding

        n_incl = id_state["n"] + 1  # window count including current message

        # extract features — O(1) each, using incremental state
        if timing_rows is not None:
            t_feats = timing.extract_incremental(current_iat, id_state, n_incl)
            timing_rows.append(t_feats)

        if payload_rows is not None:
            p_feats = payload.extract_incremental(
                current_payload, current_payload_arr,
                current_entropy, id_state, n_incl
            )
            payload_rows.append(p_feats)

        # update state AFTER feature extraction — O(1)
        # current message excluded from its own window statistics
        update_state(id_state, current_ts, current_payload, current_iat, current_entropy)

    parts = []
    if basic_df is not None:
        parts.append(basic_df.reset_index(drop=True))
    if timing_rows is not None:
        parts.append(pd.DataFrame(timing_rows))
    if payload_rows is not None:
        parts.append(pd.DataFrame(payload_rows))

    X = pd.concat(parts, axis=1)
    y = df["label"].reset_index(drop=True)

    return X, y, global_median_iat