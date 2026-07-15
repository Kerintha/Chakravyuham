import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))

import yaml
from episodes.episode_builder import build_episodes

OUT_DIR = Path(__file__).resolve().parents[1] / "output" / "train"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_PATH = str(Path(__file__).resolve().parents[1] / "profiles" / "vehicle_a.yaml")

TRAIN_SEED_BASE = 1000   # distinct from the 42 used for held-out eval scenarios -- do not change

with open(PROFILE_PATH) as f:
    PROFILE = yaml.safe_load(f)

ECUS_BY_ID = {e["id"]: e for e in PROFILE["ecus"]}

TARGET_IDS_FUZZY = ["0x164", "0x316", "0x153", "0x220", "0x2b0", "0x329", "0x18f"]
TARGET_IDS_IMPERSONATION = ["0x164", "0x316", "0x153", "0x220", "0x2b0"]
TARGET_IDS_DOS = ["0x000", "0x316"]

FUZZY_GROUPS = [
    TARGET_IDS_FUZZY[0:4],
    TARGET_IDS_FUZZY[4:7],
]
IMPERSONATION_GROUPS = [
    TARGET_IDS_IMPERSONATION[0:3],
    TARGET_IDS_IMPERSONATION[3:5],
]


def make_bursts(total_duration_s, n_bursts, min_len=10, max_len=35, seed=0):
    """
    Generate n_bursts non-overlapping (start_t, end_t) windows within
    [0, total_duration_s]. These are now episode windows (each gets its
    own small normal buffer), not slices of one continuous session.
    Deterministic given seed.
    """
    import random
    rng = random.Random(seed)
    bursts = []
    slot_size = total_duration_s / n_bursts
    for i in range(n_bursts):
        slot_start = i * slot_size
        slot_end = (i + 1) * slot_size
        burst_len = rng.uniform(min_len, max_len)
        burst_len = min(burst_len, slot_size * 0.8)
        max_start = slot_end - burst_len
        start_t = rng.uniform(slot_start, max(slot_start, max_start))
        end_t = start_t + burst_len
        bursts.append((round(start_t, 2), round(end_t, 2)))
    return bursts


SCENARIOS = []

for i, group in enumerate(FUZZY_GROUPS):
    for mode in ["realistic", "artifact"]:
        seed = TRAIN_SEED_BASE + i * 1000 + (0 if mode == "realistic" else 500)
        duration_s = 90
        SCENARIOS.append({
            "name": f"train_fuzzy_group{i}_{mode}",
            "kind": "fuzzy",
            "target_ids": group,
            "payload_mode": mode,
            "seed": seed,
            "duration_s": duration_s,
            "bursts": make_bursts(duration_s, n_bursts=2, min_len=30, max_len=40, seed=seed),
        })

for i, group in enumerate(IMPERSONATION_GROUPS):
    for mode in ["subtle", "artifact"]:
        seed = TRAIN_SEED_BASE + 10000 + i * 1000 + (0 if mode == "subtle" else 500)
        duration_s = 90
        SCENARIOS.append({
            "name": f"train_impersonation_group{i}_{mode}",
            "kind": "impersonation",
            "target_ids": group,
            "payload_mode": mode,
            "seed": seed,
            "duration_s": duration_s,
            "bursts": make_bursts(duration_s, n_bursts=2, min_len=30, max_len=40, seed=seed),
        })

for i, tid in enumerate(TARGET_IDS_DOS):
    seed = TRAIN_SEED_BASE + 20000 + i * 1000
    duration_s = 120
    SCENARIOS.append({
        "name": f"train_dos_{tid}_varied",
        "kind": "dos",
        "target_ids": [tid],
        "payload_mode": "varied_realistic",
        "seed": seed,
        "duration_s": duration_s,
        "bursts": [(40, 90)],
    })


def build_scenario(cfg):
    df = build_episodes(cfg, ECUS_BY_ID, PROFILE_PATH)

    out_path = OUT_DIR / f"{cfg['name']}.csv"
    df.to_csv(out_path, index=False)
    n_attack = int((df["label"] != "normal").sum())
    density = n_attack / len(df) * 100
    print(f"[{cfg['name']}] -> {len(df)} rows, {n_attack} attack-labeled "
          f"({density:.1f}% density), {len(cfg['bursts'])} episodes x {len(cfg['target_ids'])} IDs, "
          f"saved to {out_path}")
    return out_path


if __name__ == "__main__":
    for cfg in SCENARIOS:
        build_scenario(cfg)