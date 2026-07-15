# NOTE: can_id is intentionally excluded from this feature set.
#
# Including raw CAN ID as a model input causes data leakage:
# the model memorizes which IDs are attack-associated (e.g. 0x000 for DoS)
# rather than learning behavioral anomaly patterns. This makes the model
# brittle — a DoS attack on a different CAN ID would not be caught.
#
# CAN ID is still used as a grouping key inside timing/payload feature
# extraction (per-ID window state), it is just never fed to the model
# as a raw numeric feature.

BASIC_FEATURE_COLUMNS = [
    "dlc",
    "data_0", "data_1", "data_2", "data_3",
    "data_4", "data_5", "data_6", "data_7",
]


def extract(df):
    """
    Bulk extraction for training pipeline.
    Input: full validated dataframe from loader.
    Output: dataframe with BASIC_FEATURE_COLUMNS only.
    """
    return df[BASIC_FEATURE_COLUMNS].copy()


def extract_row(message):
    """
    Single-message extraction for real-time inference.
    message: dict with keys dlc, data_0..data_7
    Returns: dict of basic feature values.
    """
    return {col: message[col] for col in BASIC_FEATURE_COLUMNS}