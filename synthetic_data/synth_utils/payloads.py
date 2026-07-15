"""
payloads.py
Payload generation per ECU byte rules (static / range) from a vehicle profile.
Range bytes use a bounded random walk (not i.i.d. noise) so consecutive values
drift smoothly, closer to real physical-signal behavior, rather than jumping
randomly frame to frame.
"""

import numpy as np


class PayloadGenerator:
    """Stateful per-ECU payload generator. Call next_payload() once per message."""

    def __init__(self, byte_rules, rng=None):
        self.byte_rules = byte_rules  # dict: {byte_idx: {"mode": ..., ...}}
        self.rng = rng or np.random.default_rng()
        self._state = {}  # current value per range byte, for smooth walk
        for idx, rule in byte_rules.items():
            if rule["mode"] == "range":
                self._state[idx] = rule["mean"]

    def next_payload(self):
        payload = [0] * 8
        for idx, rule in self.byte_rules.items():
            if rule["mode"] == "static":
                payload[idx] = rule["value"]
            else:
                # bounded random walk: small step, clipped to [min, max]
                step_scale = max(rule["std"] * 0.15, 0.5)
                step = self.rng.normal(0, step_scale)
                new_val = self._state[idx] + step
                new_val = float(np.clip(new_val, rule["min"], rule["max"]))
                self._state[idx] = new_val
                payload[idx] = int(round(new_val))
        return payload