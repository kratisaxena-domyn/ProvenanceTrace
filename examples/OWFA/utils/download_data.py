from datasets import load_dataset

ds = load_dataset(
    "allenai/olmo-mix-1124",
    name="wiki",
    split="train",
    streaming=False,
    cache_dir="../data/data_cache"
)

print(next(iter(ds)))  # Should yield a sample document from the wiki subset
print("Dataset loaded successfully.")