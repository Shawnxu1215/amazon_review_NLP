"""Chronological TF-IDF + Logistic Regression sentiment baseline.

Install dependencies: python -m pip install pandas scikit-learn
Run: python sentiment_baseline.py
Optional: python sentiment_baseline.py --csv path/to/reviews.csv

The source CSV is read only. All cleaning happens in memory.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline


LABELS = ["negative", "positive"]
RATING_TO_SENTIMENT = {
    1: "negative", 2: "negative", 4: "positive", 5: "positive"
}
MAX_REVIEWS = 20_000
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
    required = {"content", "score", "at"}
    missing = required.difference(reviews.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    original_count = len(reviews)
    reviews = reviews[["content", "score", "at"]].copy()
    reviews["content"] = (
        reviews["content"].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    )
    reviews["score"] = pd.to_numeric(reviews["score"], errors="coerce")
    reviews["at"] = pd.to_datetime(
        pd.to_numeric(reviews["at"], errors="coerce"),
        unit="ms", errors="coerce", utc=True,
    )
    reviews = reviews.dropna(subset=["content", "score", "at"])
    reviews = reviews.loc[
        reviews["content"].ne("") & reviews["score"].isin(RATING_TO_SENTIMENT)
    ].copy()
    valid_count = len(reviews)

    reviews = reviews.sort_values("at", kind="stable")
    # Deduplicate by text, regardless of rating, so repeated text cannot cross splits.
    reviews["duplicate_key"] = reviews["content"].str.casefold()
    reviews = reviews.drop_duplicates("duplicate_key", keep="first")
    reviews = reviews.drop(columns="duplicate_key").reset_index(drop=True)
    if len(reviews) > MAX_REVIEWS:
        reviews = (
            reviews.sample(n=MAX_REVIEWS, random_state=42)
            .sort_values("at", kind="stable")
            .reset_index(drop=True)
        )
    reviews["sentiment"] = reviews["score"].map(RATING_TO_SENTIMENT)
    print(f"Loaded: {original_count:,} reviews")
    print(f"Removed missing/empty text, invalid rating or timestamp: {original_count - valid_count:,}")
    print(f"Removed duplicate reviews: {valid_count - len(reviews):,}")
    print(f"Limited to {len(reviews):,} reviews (maximum {MAX_REVIEWS:,})")
    print(f"Retained: {len(reviews):,} reviews")
    return reviews


def chronological_split(reviews):
    """Split sorted rows at 80% and 90%, with no shuffling or stratification."""
    train_end = int(len(reviews) * 0.8)
    validation_end = int(len(reviews) * 0.9)
    train = reviews.iloc[:train_end]
    validation = reviews.iloc[train_end:validation_end]
    test = reviews.iloc[validation_end:]
    if any(part.empty for part in (train, validation, test)):
        raise ValueError("Not enough cleaned reviews for three nonempty splits.")
    if train["sentiment"].nunique() < 2:
        raise ValueError("The training split needs at least two sentiment classes.")
    return train, validation, test


def evaluate(model, reviews, name):
    y_true = reviews["sentiment"]
    y_pred = model.predict(reviews["content"])
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )
    print(f"\n{name} evaluation")
    print(f"Accuracy:        {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1:        {f1:.4f}")
    print(f"Macro precision: {precision:.4f}")
    print(f"Macro recall:    {recall:.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred, labels=LABELS, digits=4, zero_division=0
    ))
    matrix = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=LABELS),
        index=pd.Index(LABELS, name="Actual"),
        columns=pd.Index(LABELS, name="Predicted"),
    )
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(matrix.to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    reviews = load_reviews(args.csv)
    train, validation, test = chronological_split(reviews)
    for name, part in (("Train", train), ("Validation", validation), ("Test", test)):
        print(f"\n{name}: {len(part):,} rows ({len(part) / len(reviews):.1%})")
        print(f"Time range: {part['at'].iloc[0]} to {part['at'].iloc[-1]}")
        print(part["sentiment"].value_counts().reindex(LABELS, fill_value=0).to_string())

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50_000, ngram_range=(1, 2), sublinear_tf=True
        )),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    # Both vocabulary/IDF and classifier are fitted exclusively on training data.
    X_train = train["content"]
    y_train = train["sentiment"]
    model.fit(X_train, y_train)
    evaluate(model, validation, "Validation")
    # Fixed baseline: no tuning or refitting based on validation/test results.
    evaluate(model, test, "Test")


if __name__ == "__main__":
    main()
