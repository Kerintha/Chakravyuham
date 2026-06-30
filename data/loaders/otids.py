import os
import pandas as pd
from data.loaders.base_loader import BaseLoader, COLUMN_NAMES

RAW_SUBDIR = "otids"

FILES = {
    "Attack_free_dataset.txt": "attack_free",
    "DoS_attack_dataset.txt": "dos",
    "Fuzzy_attack_dataset.txt": "fuzzy",
    "Impersonation_attack_dataset.txt": "impersonation",
}


def _label_attack_free(df):
    df["label"] = "normal"
    return df


def _label_dos(df):
    df["label"] = df["can_id"].apply(lambda x: "dos" if x == 0x000 else "normal")
    return df


def _label_fuzzy(df):
    df["label"] = df["timestamp"].apply(lambda t: "normal" if t < 250 else "fuzzy")
    return df


def _label_impersonation(df):
    df["label"] = df["timestamp"].apply(lambda t: "normal" if t < 250 else "impersonation")
    return df


_LABEL_FUNCS = {
    "attack_free": _label_attack_free,
    "dos": _label_dos,
    "fuzzy": _label_fuzzy,
    "impersonation": _label_impersonation,
}


class OTIDSLoader(BaseLoader):
    name = "otids"

    def _parse_line(self, line):
        """
        Parses one raw HCRL Car-Hacking dataset line.
        Format: Timestamp: <val> ID: <hex> 000 DLC: <val> <byte0> ... <byteN>
        """
        parts = line.strip().split()
        if not parts:
            return None
        try:
            timestamp = float(parts[1])
            can_id = int(parts[3], 16)
            dlc = int(parts[6])
            data_bytes = [int(b, 16) for b in parts[7:7 + dlc]]
            while len(data_bytes) < 8:
                data_bytes.append(0)
            return [timestamp, can_id, dlc] + data_bytes
        except (IndexError, ValueError):
            return None

    def _build(self, raw_dir):
        all_dfs = []
        for filename, key in FILES.items():
            filepath = os.path.join(raw_dir, RAW_SUBDIR, filename)
            df = self._parse_file_with_progress(filepath, COLUMN_NAMES)
            df = _LABEL_FUNCS[key](df)
            df["source_file"] = key
            all_dfs.append(df)

        return pd.concat(all_dfs, ignore_index=True)