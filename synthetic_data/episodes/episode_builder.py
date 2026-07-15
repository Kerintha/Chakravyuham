"""
episode_builder.py

Confirmed against features/pipeline.py, timing.py, payload.py: all features
are computed per-CAN-ID (per_id_state keyed by can_id), with zero cross-ID
dependency. Therefore each episode's buffer only needs to seed the TARGET
ID(s)' own rolling state before the attack starts -- other ECUs contribute
no feature signal and are pure dilution if included.

Buffer duration is sized to comfortably exceed the feature pipeline's
window_size_ms (50ms default) so rolling stats aren't cold-starting right
at the attack boundary, without reintroducing full-session/full-ECU dilution.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from generators.normal import generate_normal_traffic
from injectors.dos import inject_dos
from injectors.fuzzy import inject_fuzzy
from injectors.impersonation import inject_impersonation

BUFFER_PAD_S = 0.3


def _generate_target_only_buffer(profile_path, duration_s, seed, target_ids_int, anchor_t):
    raw = generate_normal_traffic(profile_path, duration_s=duration_s, seed=seed)
    raw["timestamp"] = raw["timestamp"] + anchor_t
    return raw[raw["can_id"].isin(target_ids_int)].reset_index(drop=True)


def _build_single_episode(cfg, burst_idx, start_t, end_t, ecus_by_id, profile_path):
    target_ids_int = [int(t, 16) for t in cfg["target_ids"]]

    pre_start = max(0.0, start_t - BUFFER_PAD_S)
    pre_duration = start_t - pre_start
    post_duration = BUFFER_PAD_S

    seed_pre = cfg["seed"] + burst_idx * 7919
    seed_post = seed_pre + 3853

    blocks = []
    if pre_duration > 0:
        blocks.append(_generate_target_only_buffer(
            profile_path, pre_duration, seed_pre, target_ids_int, pre_start))
    if post_duration > 0:
        blocks.append(_generate_target_only_buffer(
            profile_path, post_duration, seed_post, target_ids_int, end_t))

    if blocks:
        episode_df = pd.concat(blocks, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    else:
        episode_df = pd.DataFrame(columns=["timestamp", "can_id", "dlc"] + [f"data_{i}" for i in range(8)] + ["label", "source_file"])

    for id_idx, tid_hex in enumerate(cfg["target_ids"]):
        can_id_int = int(tid_hex, 16)
        inject_seed = cfg["seed"] + burst_idx * 100 + id_idx

        if cfg["kind"] == "dos":
            episode_df = inject_dos(
                episode_df, target_id=can_id_int, start_t=start_t, end_t=end_t,
                payload_mode=cfg["payload_mode"], seed=inject_seed,
            )
        elif cfg["kind"] == "fuzzy":
            ecu = ecus_by_id[tid_hex]
            episode_df = inject_fuzzy(
                episode_df, target_id=can_id_int, ecu_profile=ecu,
                start_t=start_t, end_t=end_t,
                payload_mode=cfg["payload_mode"], seed=inject_seed,
            )
        elif cfg["kind"] == "impersonation":
            ecu = ecus_by_id[tid_hex]
            episode_df = inject_impersonation(
                episode_df, target_id=can_id_int, ecu_profile=ecu,
                start_t=start_t, end_t=end_t,
                payload_mode=cfg["payload_mode"], seed=inject_seed,
            )
        else:
            raise ValueError(cfg["kind"])

    episode_df = episode_df.sort_values("timestamp").reset_index(drop=True)
    episode_df["source_file"] = f"{cfg['name']}_b{burst_idx}"
    return episode_df


def build_episodes(cfg, ecus_by_id, profile_path):
    blocks = []
    for burst_idx, (start_t, end_t) in enumerate(cfg["bursts"]):
        block = _build_single_episode(cfg, burst_idx, start_t, end_t, ecus_by_id, profile_path)
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)