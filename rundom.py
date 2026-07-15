from data.loaders.otids_augmented import OTIDSAugmentedLoader
loader = OTIDSAugmentedLoader()
df = loader.load()
loader.validate(df)
print(df.groupby(["origin", "label"]).size())