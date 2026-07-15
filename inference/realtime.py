import os
import json
import pickle
import yaml
import numpy as np
import pandas as pd

from features import basic, timing, payload
from features.state import shannon_entropy, make_id_state, prune_state, update_state


class RealtimeInference:
    """
    Real-time CAN IDS inference engine.

    Maintains a rolling state buffer per CAN ID and classifies each
    incoming CAN message as it arrives, one at a time.

    Uses identical feature computation logic as the training pipeline
    (same timing.extract(), payload.extract(), same state structure
    from features/state.py, same window_size_ms) so training and
    inference produce identical feature vectors for the same message sequence.

    State initializes empty at startup and fills over the first
    window_size_ms of messages. During cold-start, timing features
    are padded with global_median_iat (same rule as training).

    Usage:
        engine = RealtimeInference.from_results_dir(
            "results/xgboost_all_features_20260629_143012"
        )
        label = engine.process_message(
            timestamp=0.001234,
            can_id=0x0316,
            dlc=8,
            data_bytes=[0x05, 0x1c, 0x6a, 0x0a, 0x1c, 0x13, 0x00, 0x7f]
        )
        # returns: "normal" | "dos" | "fuzzy" | "impersonation"
    """

    def __init__(self, model, feature_list, window_size_ms, global_median_iat):
        self.model             = model
        self.feature_list      = feature_list
        self.window_size_ms    = window_size_ms
        self.window_size_s     = window_size_ms / 1000.0
        self.global_median_iat = global_median_iat
        self.per_id_state      = {}

    @classmethod
    def from_results_dir(cls, results_dir):
        """
        Loads a trained model and its full feature configuration from a
        results directory produced by run_experiment.py.
        """
        with open(os.path.join(results_dir, "model.pkl"), "rb") as f:
            model = pickle.load(f)

        with open(os.path.join(results_dir, "config.yaml")) as f:
            config = yaml.safe_load(f)

        with open(os.path.join(results_dir, "feature_params.json")) as f:
            feature_params = json.load(f)

        gmi_path = os.path.join(results_dir, "global_median_iat.json")
        if os.path.exists(gmi_path):
            with open(gmi_path) as f:
                global_median_iat = json.load(f)["global_median_iat"]
        else:
            global_median_iat = 0.01

        return cls(
            model             = model,
            feature_list      = config["features"],
            window_size_ms    = feature_params.get("window_size_ms", 50),
            global_median_iat = global_median_iat,
        )

    def process_message(self, timestamp, can_id, dlc, data_bytes):
        """
        Classifies one incoming CAN message in real time.

        timestamp:  float, seconds relative to session start.
                    must be monotonically increasing within a session.
        can_id:     int, CAN message identifier.
        dlc:        int, data length code (0-8).
        data_bytes: list of ints (0-255), length == dlc.
                    padded to 8 bytes internally if dlc < 8.

        Returns: predicted label string
                 ("normal" | "dos" | "fuzzy" | "impersonation")

        Inference flow per message:
          1. initialize state for new CAN IDs (make_id_state)
          2. prune state: drop entries outside current window (prune_state)
          3. compute features using pruned state — O(1) incremental
          4. update state with current message (update_state) — O(1)
          5. predict and return label
        """
        data_bytes = list(data_bytes)
        while len(data_bytes) < 8:
            data_bytes.append(0)

        message = {
            "timestamp": timestamp,
            "can_id":    can_id,
            "dlc":       dlc,
            **{f"data_{i}": data_bytes[i] for i in range(8)},
        }

        if can_id not in self.per_id_state:
            self.per_id_state[can_id] = make_id_state()

        id_state = self.per_id_state[can_id]

        prune_state(id_state, timestamp - self.window_size_s)

        # precompute values shared across feature extraction and state update
        current_payload_arr = np.array(data_bytes, dtype=np.float64)
        current_entropy     = shannon_entropy(data_bytes)

        if id_state["timestamps"]:
            current_iat = timestamp - id_state["timestamps"][-1]
        else:
            current_iat = self.global_median_iat

        n_incl = id_state["n"] + 1

        feature_row = {}

        if "basic" in self.feature_list:
            feature_row.update(basic.extract_row(message))

        if "timing" in self.feature_list:
            feature_row.update(
                timing.extract_incremental(current_iat, id_state, n_incl)
            )

        if "payload" in self.feature_list:
            feature_row.update(
                payload.extract_incremental(
                    data_bytes, current_payload_arr,
                    current_entropy, id_state, n_incl
                )
            )

        # update state AFTER feature extraction — O(1)
        update_state(id_state, timestamp, data_bytes, current_iat, current_entropy)

        X          = pd.DataFrame([feature_row])
        prediction = self.model.predict(X)[0]

        return prediction

    def reset_state(self):
        """
        Clears all rolling state buffers.
        Call at vehicle startup or start of a new drive session.
        """
        self.per_id_state = {}