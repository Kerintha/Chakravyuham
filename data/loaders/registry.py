from data.loaders.otids import OTIDSLoader
from data.loaders.otids_augmented import OTIDSAugmentedLoader

LOADER_REGISTRY = {
    "otids": OTIDSLoader,
    "otids_augmented": OTIDSAugmentedLoader,
    # "car_hacking": CarHackingLoader,   # added when you build a second dataset
}


def get_loader(name):
    if name not in LOADER_REGISTRY:
        raise ValueError(f"Unknown dataset: '{name}'")
    return LOADER_REGISTRY[name]()