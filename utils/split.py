import pandas as pd


def temporal_split(df, train_size=0.70, val_size=0.15, test_size=0.15):
    """
    Window-based temporal split applied independently per (source_file, label) group.

    Why per-(file, label) rather than per-file:
      Files like fuzzy/impersonation contain two structurally different time
      regions — a normal region (t < 250s) and an attack region (t >= 250s).
      Splitting purely by file-level time position can push an entire label's
      region past the train cutoff, leaving train with zero examples of that
      class. Confirmed: fuzzy's attack region (250-347s) sat entirely past the
      70% time mark of the fuzzy file's full range (0-347s), so train received
      0 fuzzy rows under the old per-file split. Impersonation showed a milder
      version of the same problem (~62% of its attack region pushed into
      val/test instead of train), consistent with its weak (not zero) recall.

      Splitting within each (file, label) group separately guarantees every
      class appears in train, val, and test, while still preserving
      chronological order within each group (no shuffling). Verified safe for
      dos (label regions span the full file timeline, fully interleaved) and
      attack_free (single label, no boundary risk).

    Preserves original df indices so X and y can be sliced with .loc after
    build_features() returns.

    df must have columns: timestamp, source_file, label.
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-9, \
        "train_size + val_size + test_size must equal 1.0"

    train_dfs, val_dfs, test_dfs = [], [], []

    for (source, label), group_df in df.groupby(["source_file", "label"]):
        group_df = group_df.sort_values("timestamp")

        t_min = group_df["timestamp"].min()
        t_max = group_df["timestamp"].max()
        t_range = t_max - t_min

        train_cutoff = t_min + t_range * train_size
        val_cutoff   = t_min + t_range * (train_size + val_size)

        train_dfs.append(group_df[group_df["timestamp"] <= train_cutoff])
        val_dfs.append(group_df[
            (group_df["timestamp"] > train_cutoff) &
            (group_df["timestamp"] <= val_cutoff)
        ])
        test_dfs.append(group_df[group_df["timestamp"] > val_cutoff])

    train = pd.concat(train_dfs)
    val   = pd.concat(val_dfs)
    test  = pd.concat(test_dfs)

    return train, val, test

def temporal_split_augmented(df, train_size=0.70, val_size=0.15, test_size=0.15):
    """
    Same per-(source_file, label) temporal logic as temporal_split, but rows
    with origin == "synthetic" are always assigned to train regardless of
    timestamp -- real val/test must stay untouched by synthetic data.
    Requires an 'origin' column (added by export_augmented_dataset.py).
    """
    real_mask = df["origin"] == "real"
    synthetic_df = df[~real_mask]

    real_train, val, test = temporal_split(
        df[real_mask], train_size=train_size, val_size=val_size, test_size=test_size
    )

    train = pd.concat([real_train, synthetic_df])
    return train, val, test