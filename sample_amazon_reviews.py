"""Create a reproducible year/rating-stratified annotation sample.

Run: python sample_amazon_reviews.py
Requires pandas and numpy. All original columns are preserved.
Year quotas mix 30% original year shares with 70% recency shares (two-year
half-life relative to the newest observed year). Rating quotas within each
year follow its original distribution, with largest-remainder rounding.
No deduplication, sentiment filtering, or train/validation/test split is done.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "ShoppingAppReviews Dataset/ShoppingAppReviews/csv/Amazon shopping.csv"


def detect_column(columns, aliases, role):
    normalized = {"".join(c for c in name.lower() if c.isalnum()): name for name in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    raise ValueError(f"Cannot detect {role} column. Available columns: {list(columns)}")


def extract_year(values):
    """Recognize year-only, epoch seconds/ms/us/ns, or date strings."""
    numeric = pd.to_numeric(values, errors="coerce")
    present = values.notna()
    if numeric[present].notna().all() and present.any():
        if numeric[present].between(1900, 2100).all():
            return numeric.where(numeric.eq(numeric.round())).astype("Int64")
        magnitude = numeric.abs().median()
        unit = "ns" if magnitude >= 1e17 else "us" if magnitude >= 1e14 else "ms" if magnitude >= 1e11 else "s"
        dates = pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)
    else:
        dates = pd.to_datetime(values, format="mixed", errors="coerce", utc=True)
    return dates.dt.year.astype("Int64")


def allocate(total, weights, capacities):
    """Integer quotas; redistribute capped strata and round by remainder."""
    capacities = np.asarray(capacities, dtype=int)
    weights = np.asarray(weights, dtype=float)
    if total < 0 or total > capacities.sum():
        raise ValueError("Requested sample size exceeds eligible rows or is negative.")
    quotas = np.zeros(len(capacities), dtype=int)
    active = capacities > 0
    remaining = total
    while remaining:
        indices = np.flatnonzero(active)
        shares = remaining * weights[indices] / weights[indices].sum()
        capped = shares >= capacities[indices]
        if capped.any():
            selected = indices[capped]
            quotas[selected] = capacities[selected]
            remaining -= int(capacities[selected].sum())
            active[selected] = False
            continue
        rounded = np.floor(shares).astype(int)
        extra = remaining - int(rounded.sum())
        order = np.argsort(-(shares - rounded), kind="stable")
        rounded[order[:extra]] += 1
        quotas[indices] = rounded
        break
    return quotas


def print_distribution(title, before, after):
    counts_before = before.value_counts().sort_index()
    counts_after = after.value_counts().reindex(counts_before.index, fill_value=0)
    table = pd.DataFrame({
        "before_n": counts_before,
        "before_%": counts_before / len(before) * 100,
        "after_n": counts_after,
        "after_%": counts_after / len(after) * 100,
    })
    print(f"\n{title}\n{table.to_string(float_format=lambda x: f'{x:.2f}')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=ROOT / "sampled_30k.csv")
    parser.add_argument("--n", type=int, default=30_000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    if args.n <= 0:
        parser.error("--n must be positive")
    if args.csv.resolve() == args.output.resolve():
        parser.error("Output must differ from the source CSV")
    if args.output.exists():
        parser.error("Output already exists; choose another --output to protect annotations")

    # Read original fields as strings to retain IDs and original date representations.
    raw = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
    rating_col = detect_column(raw.columns, ["score", "rating", "stars", "starrating", "reviewrating", "overall"], "rating")
    date_col = detect_column(raw.columns, ["at", "reviewtime", "reviewdate", "date", "timestamp", "createdat", "time", "year"], "date")
    text_col = detect_column(raw.columns, ["content", "reviewtext", "review", "text", "body"], "review text")
    for column in ["issue_label", "label_source"]:
        if column in raw and raw[column].str.strip().ne("").any():
            raise ValueError(f"Source has existing {column} annotations; refusing to clear them")
    years = extract_year(raw[date_col])
    ratings = pd.to_numeric(raw[rating_col], errors="coerce")
    valid = years.notna() & ratings.isin([1, 2, 3, 4, 5]) & raw[text_col].str.strip().ne("")
    reviews = raw.loc[valid].copy()
    reviews["year"] = years.loc[valid].astype(int)
    ratings = ratings.loc[valid].astype(int)
    print(f"Detected: text={text_col}, rating={rating_col}, date={date_col}")
    print(f"Source: {len(raw):,}; eligible: {len(reviews):,}; excluded: {(~valid).sum():,}")
    print(f"Invalid date: {years.isna().sum():,}; invalid rating: {(~pd.to_numeric(raw[rating_col], errors='coerce').isin([1, 2, 3, 4, 5])).sum():,}; blank text: {raw[text_col].str.strip().eq('').sum():,}")
    if len(reviews) < args.n:
        raise ValueError(f"Need {args.n:,} eligible rows; found {len(reviews):,}")

    year_counts = reviews["year"].value_counts().sort_index()
    recency = 2.0 ** ((year_counts.index.to_numpy() - year_counts.index.max()) / 2.0)
    weights = 0.30 * year_counts.to_numpy() / year_counts.sum() + 0.70 * recency / recency.sum()
    year_quotas = allocate(args.n, weights, year_counts.to_numpy())
    rng = np.random.RandomState(args.random_state)
    sampled_indices = []
    for year, quota in zip(year_counts.index, year_quotas):
        year_indices = reviews.index[reviews["year"].eq(year)]
        counts = ratings.loc[year_indices].value_counts().sort_index()
        rating_quotas = allocate(int(quota), counts.to_numpy(), counts.to_numpy())
        for rating, count in zip(counts.index, rating_quotas):
            candidates = year_indices[ratings.loc[year_indices].eq(rating)]
            sampled_indices.extend(rng.choice(candidates, size=int(count), replace=False).tolist())

    sample = reviews.loc[sampled_indices].sample(frac=1, random_state=args.random_state).copy()
    sample["issue_label"] = ""
    sample["label_source"] = ""
    assert len(sample) == args.n and sample.index.is_unique
    print_distribution("Year distribution (eligible source vs sample)", reviews["year"], sample["year"])
    print_distribution("Overall rating distribution", ratings, ratings.loc[sample.index])
    for year in year_counts.index:
        print_distribution(f"Rating distribution within {year}", ratings.loc[reviews.index[reviews.year.eq(year)]], ratings.loc[sample.index[sample.year.eq(year)]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(sample):,} rows (random_state={args.random_state})")


if __name__ == "__main__":
    main()
