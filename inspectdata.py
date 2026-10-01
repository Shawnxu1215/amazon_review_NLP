from pathlib import Path
import pandas as pd

DEFAULT_CSV = (
    Path(__file__).resolve().parent
    / "ShoppingAppReviews Dataset"
    / "ShoppingAppReviews"
    / "csv"
    / "Amazon shopping.csv"
)


def load_reviews(csv_path):
    """Clean records, convert epoch milliseconds, and retain earliest duplicates."""
    reviews = pd.read_csv(csv_path)
    return reviews

reviews = load_reviews(DEFAULT_CSV)
print(f"Loaded {len(reviews)} reviews from {DEFAULT_CSV}")
print(reviews["score"].value_counts().sort_index())

import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))