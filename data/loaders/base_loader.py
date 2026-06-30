from abc import ABC, abstractmethod
import os
import pandas as pd
from tqdm import tqdm

COLUMN_NAMES = ["timestamp", "can_id", "dlc",
                "data_0", "data_1", "data_2", "data_3",
                "data_4", "data_5", "data_6", "data_7"]

REQUIRED_COLUMNS = COLUMN_NAMES + ["label"]


class BaseLoader(ABC):
    name = "base"

    @abstractmethod
    def _parse_line(self, line):
        """
        Dataset-specific line parser.
        Each loader implements this for its own raw file format.
        Returns a list of values matching COLUMN_NAMES, or None if line is malformed.
        """
        raise NotImplementedError

    @abstractmethod
    def _build(self, raw_dir):
        """
        Dataset-specific build logic.
        Calls self._parse_file_with_progress() for each raw file,
        applies labels, returns one combined dataframe.
        """
        raise NotImplementedError

    def _parse_file_with_progress(self, filepath, column_names):
        """
        Generic file reader with tqdm progress bar.
        Lives in base class so every loader gets it automatically.
        Each loader's _parse_line() handles the actual line format.
        """
        with open(filepath, "r") as f:
            lines = f.readlines()

        rows = []
        for line in tqdm(lines, desc=f"Loading {os.path.basename(filepath)}", unit="lines"):
            row = self._parse_line(line)
            if row is not None:
                rows.append(row)

        return pd.DataFrame(rows, columns=column_names)

    def validate(self, df):
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"[{self.name}] Schema broken: missing column '{col}'")

        if not pd.api.types.is_float_dtype(df["timestamp"]):
            raise ValueError(f"[{self.name}] timestamp must be float")

        if not pd.api.types.is_integer_dtype(df["can_id"]):
            raise ValueError(f"[{self.name}] can_id must be int")

        for i in range(8):
            col = f"data_{i}"
            if not df[col].between(0, 255).all():
                raise ValueError(f"[{self.name}] {col} has values outside 0-255")

        if df[REQUIRED_COLUMNS].isnull().any().any():
            raise ValueError(f"[{self.name}] Schema broken: missing values found")

        return True

    def load(self, raw_dir="data/raw", processed_dir="data/processed", use_cache=True):
        cache_path = os.path.join(processed_dir, f"{self.name}_clean.csv")

        if use_cache and os.path.exists(cache_path):
            print(f"Loading cached dataset from {cache_path}")
            return pd.read_csv(cache_path)

        df = self._build(raw_dir)
        self.validate(df)

        os.makedirs(processed_dir, exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"Dataset cached to {cache_path}")

        return df