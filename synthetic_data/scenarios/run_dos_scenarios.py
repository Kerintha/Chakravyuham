"""
run_dos_scenarios.py
Builds the 6 DoS stress-test scenarios and saves each as a labeled CSV
matching the pipeline's expected schema, ready to feed into build_features().
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from generators.normal import generate_normal_traffic
from injectors.dos import inject_dos

OUT_DIR = Path(__file__).resolve().parents[1] / "output"
OUT_DIR.mkdir(exist_ok=True)

PROFILE = str(Path(__file__).resolve().parents[1] / "profiles" / "vehicle_a.yaml")

# Real HCRL DoS timing, used as the flood-rate baseline (from extract_stats.py output)
REAL_DOS_IAT_MEAN_MS = 0.912622
REAL_DOS_IAT_STD_MS = 1.158867

SCENARIOS = [
    {
        "name": "dos_fixed_payload_baseline",
        "question": "Control: does synthetic DoS behave like real HCRL DoS?",
        "target_id": 0x000,
        "payload_mode": "fixed",
        "flood_iat_mean_ms": REAL_DOS_IAT_MEAN_MS,
        "flood_iat_std_ms": REAL_DOS_IAT_STD_MS,
    },
    {
        "name": "dos_varied_payload_same_id",
        "question": "Same ID/frequency, payload now varies — is detection timing-based or payload-memorization?",
        "target_id": 0x000,
        "payload_mode": "varied_realistic",
        "flood_iat_mean_ms": REAL_DOS_IAT_MEAN_MS,
        "flood_iat_std_ms": REAL_DOS_IAT_STD_MS,
    },
    {
        "name": "dos_different_id",
        "question": "Flood a different legitimate ID, fixed payload — does ID choice matter (it shouldn't; can_id isn't a feature)?",
        "target_id": 0x316,
        "payload_mode": "fixed",
        "flood_iat_mean_ms": REAL_DOS_IAT_MEAN_MS,
        "flood_iat_std_ms": REAL_DOS_IAT_STD_MS,
    },
    {
        "name": "dos_different_id_varied_payload",
        "question": "Different ID AND varied payload — removes every dataset-specific shortcut at once.",
        "target_id": 0x316,
        "payload_mode": "varied_realistic",
        "flood_iat_mean_ms": REAL_DOS_IAT_MEAN_MS,
        "flood_iat_std_ms": REAL_DOS_IAT_STD_MS,
    },
    {
        "name": "dos_partial_flood",
        "question": "Moderate frequency increase (~3x normal, not extreme) — does detection hold at realistic attacker throttling?",
        "target_id": 0x000,
        "payload_mode": "fixed",
        "flood_iat_mean_ms": 3.0,   # ~3x normal 0x153-class rate, well above extreme flood
        "flood_iat_std_ms": 1.0,
    },
    {
        "name": "dos_multi_id_flood",
        "question": "Flood two IDs concurrently — does per-ID state handle simultaneous DoS sources?",
        "target_id": [0x000, 0x316],
        "payload_mode": "fixed",
        "flood_iat_mean_ms": REAL_DOS_IAT_MEAN_MS,
        "flood_iat_std_ms": REAL_DOS_IAT_STD_MS,
    },
]

DURATION_S = 120
ATTACK_START_T = 40
ATTACK_END_T = 90


def build_scenario(cfg):
    normal_df = generate_normal_traffic(PROFILE, duration_s=DURATION_S, seed=42)

    target_ids = cfg["target_id"] if isinstance(cfg["target_id"], list) else [cfg["target_id"]]
    df = normal_df
    for tid in target_ids:
        df = inject_dos(
            df,
            target_id=tid,
            start_t=ATTACK_START_T,
            end_t=ATTACK_END_T,
            flood_iat_mean_ms=cfg["flood_iat_mean_ms"],
            flood_iat_std_ms=cfg["flood_iat_std_ms"],
            payload_mode=cfg["payload_mode"],
            seed=42,
        )

    out_path = OUT_DIR / f"{cfg['name']}.csv"
    df.to_csv(out_path, index=False)
    print(f"[{cfg['name']}] {cfg['question']}")
    print(f"  -> {len(df)} rows, {sum(df['label']=='dos')} dos-labeled, saved to {out_path}\n")
    return out_path


if __name__ == "__main__":
    for cfg in SCENARIOS:
        build_scenario(cfg)