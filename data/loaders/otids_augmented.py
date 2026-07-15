from data.loaders.base_loader import BaseLoader


class OTIDSAugmentedLoader(BaseLoader):
    name = "otids_augmented"

    def _parse_line(self, line):
        raise NotImplementedError(
            "OTIDSAugmentedLoader has no raw text format. Run export_augmented_dataset.py "
            "to build data/processed/otids_augmented_clean.csv -- load() will use that cache directly."
        )

    def _build(self, raw_dir):
        raise FileNotFoundError(
            "data/processed/otids_augmented_clean.csv not found. "
            "Run export_augmented_dataset.py first -- this loader has no raw source to build from."
        )