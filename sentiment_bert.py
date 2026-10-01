"""Fine-tune BERT to predict review sentiment from star ratings.

Install dependencies:
    python -m pip install torch transformers datasets accelerate pandas
    scikit-learn
Run: python sentiment_bert.py
Optional: python sentiment_bert.py --csv path/to/reviews.csv --epochs 3

The source CSV is read only. The fine-tuned model is saved to the
output directory.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


LABELS = ["negative", "positive"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
RATING_TO_SENTIMENT = {
    1: "negative",
    2: "negative",
    4: "positive",
    5: "positive",
}
DEFAULT_CSV = (
    Path(__file__).resolve().parent
    / "ShoppingAppReviews Dataset"
    / "ShoppingAppReviews"
    / "csv"
    / "Amazon shopping.csv"
)
DEFAULT_MODEL = "google-bert/bert-base-uncased"
MAX_REVIEWS = 20_000


def load_reviews(csv_path):
    """Clean reviews and retain the earliest occurrence of duplicate text."""
    reviews = pd.read_csv(csv_path)
    required = {"content", "score", "at"}
    missing = required.difference(reviews.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    original_count = len(reviews)
    reviews = reviews[["content", "score", "at"]].copy()
    reviews["content"] = (
        reviews["content"]
        .astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    reviews["score"] = pd.to_numeric(reviews["score"], errors="coerce")
    reviews["at"] = pd.to_datetime(
        pd.to_numeric(reviews["at"], errors="coerce"),
        unit="ms",
        errors="coerce",
        utc=True,
    )
    reviews = reviews.dropna(subset=["content", "score", "at"])
    reviews = reviews.loc[
        reviews["content"].ne("") & reviews["score"].isin(RATING_TO_SENTIMENT)
    ].copy()
    valid_count = len(reviews)

    reviews = reviews.sort_values("at", kind="stable")
    reviews["duplicate_key"] = reviews["content"].str.casefold()
    reviews = reviews.drop_duplicates("duplicate_key", keep="first")
    reviews = reviews.drop(columns="duplicate_key").reset_index(drop=True)
    deduplicated_count = len(reviews)
    if deduplicated_count > MAX_REVIEWS:
        reviews = (
            reviews.sample(n=MAX_REVIEWS, random_state=42)
            .sort_values("at", kind="stable")
            .reset_index(drop=True)
        )
    reviews["sentiment"] = reviews["score"].map(RATING_TO_SENTIMENT)
    reviews["labels"] = reviews["sentiment"].map(LABEL_TO_ID)

    print(f"Loaded: {original_count:,} reviews")
    print(
        "Removed missing/empty text, invalid rating or timestamp: "
        f"{original_count - valid_count:,}"
    )
    print(f"Removed duplicate reviews: {valid_count - deduplicated_count:,}")
    print(f"Limited to {len(reviews):,} reviews (maximum {MAX_REVIEWS:,})")
    print(f"Retained: {len(reviews):,} reviews")
    return reviews


def chronological_split(reviews):
    """Split chronologically at 80% and 90%, without shuffling the data."""
    train_end = int(len(reviews) * 0.8)
    validation_end = int(len(reviews) * 0.9)
    train = reviews.iloc[:train_end]
    validation = reviews.iloc[train_end:validation_end]
    test = reviews.iloc[validation_end:]
    if any(part.empty for part in (train, validation, test)):
        raise ValueError(
            "Not enough cleaned reviews for three nonempty splits."
        )
    if train["sentiment"].nunique() != len(LABELS):
        raise ValueError(
            "The training split must contain both sentiment classes."
        )
    return train, validation, test


def tokenize_split(reviews, tokenizer, max_length, name):
    dataset = Dataset.from_pandas(
        reviews[["content", "labels"]].reset_index(drop=True),
        preserve_index=False,
    )

    def tokenize_batch(batch):
        return tokenizer(
            batch["content"],
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["content"],
        desc=f"Tokenizing {name} reviews",
    )


def compute_metrics(prediction):
    predictions = np.argmax(prediction.predictions, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        prediction.label_ids,
        predictions,
        labels=list(range(len(LABELS))),
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(prediction.label_ids, predictions),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        inputs = inputs.copy()
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits,
            labels,
            weight=self.class_weights.to(
                device=outputs.logits.device,
                dtype=outputs.logits.dtype,
            ),
        )
        return (loss, outputs) if return_outputs else loss


def report_test_results(trainer, dataset):
    result = trainer.predict(dataset, metric_key_prefix="test")
    predictions = np.argmax(result.predictions, axis=-1)
    print("\nTest evaluation")
    print(f"Accuracy: {result.metrics['test_accuracy']:.4f}")
    print(f"Macro F1: {result.metrics['test_macro_f1']:.4f}")
    print(f"Macro precision: {result.metrics['test_macro_precision']:.4f}")
    print(f"Macro recall: {result.metrics['test_macro_recall']:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            result.label_ids,
            predictions,
            labels=list(range(len(LABELS))),
            target_names=LABELS,
            digits=4,
            zero_division=0,
        )
    )
    matrix = pd.DataFrame(
        confusion_matrix(
            result.label_ids,
            predictions,
            labels=list(range(len(LABELS))),
        ),
        index=pd.Index(LABELS, name="Actual"),
        columns=pd.Index(LABELS, name="Predicted"),
    )
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(matrix.to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("bert_sentiment_model")
    )
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.max_length <= 0:
        parser.error("epochs, batch-size, and max-length must be positive")

    set_seed(42)
    reviews = load_reviews(args.csv)
    train, validation, test = chronological_split(reviews)
    for name, part in (
        ("Train", train),
        ("Validation", validation),
        ("Test", test),
    ):
        print(f"\n{name}: {len(part):,} rows")
        counts = part["sentiment"].value_counts().reindex(LABELS, fill_value=0)
        print(counts.to_string())

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_dataset = tokenize_split(train, tokenizer, args.max_length, "train")
    validation_dataset = tokenize_split(
        validation, tokenizer, args.max_length, "validation"
    )
    test_dataset = tokenize_split(test, tokenizer, args.max_length, "test")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id=LABEL_TO_ID,
    )
    class_counts = train["labels"].value_counts().reindex(
        range(len(LABELS)), fill_value=0
    )
    class_weights = (len(train) / (len(LABELS) * class_counts)).tolist()
    gradient_accumulation_steps = 2
    steps_per_epoch = math.ceil(
        len(train) / (args.batch_size * gradient_accumulation_steps)
    )
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=math.ceil(steps_per_epoch * args.epochs * 0.1),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        report_to="none",
        seed=42,
        data_seed=42,
    )
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print(f"\nDevice: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Model: {args.model}")
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"\nBest model saved to: {args.output_dir.resolve()}")
    report_test_results(trainer, test_dataset)


if __name__ == "__main__":
    main()
