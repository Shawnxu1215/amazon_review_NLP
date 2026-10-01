"""Fine-tune BERT uncased on multiple review issue labels.

Install: python -m pip install torch transformers datasets accelerate pandas scikit-learn
Run: python review_labels_bert.py
Inspect data without downloading/training: python review_labels_bert.py --check-data

Reads the first 20,000 data rows (excluding the header) by default. Column A
is review text, column B is ignored, and columns C:N are twelve binary labels.
After cleaning and deduplication, split 80/10/10 in CSV order, matching the
previous sentiment script's unshuffled split. This CSV has no timestamps,
so CSV order is not necessarily chronological. The source CSV is read only.

Each output uses sigmoid with a configurable threshold (default 0.5), allowing
zero, one, or multiple predicted labels. No_Issue is treated like any label;
no exclusivity rule is imposed. Blank labels are rejected, not filled with 0.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    hamming_loss,
    multilabel_confusion_matrix,
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


DEFAULT_CSV = Path(__file__).resolve().parent / "sampled_30k_annotated (2).csv"
DEFAULT_MODEL = "google-bert/bert-base-uncased"


def load_reviews(csv_path, max_rows=20_000):
    """Validate annotations and keep the first occurrence of normalized text."""
    source = pd.read_csv(csv_path, nrows=max_rows, encoding="utf-8-sig")
    if source.shape[1] < 14:
        raise ValueError("Expected review text in A and twelve binary labels in C:N.")
    label_names = source.columns[2:14].tolist()
    targets = source.iloc[:, 2:14].apply(pd.to_numeric, errors="coerce")
    valid = targets.isin([0, 1])
    if not valid.to_numpy().all():
        examples = [
            f"data row {row + 1}, label {label_names[col]}"
            for row, col in np.argwhere(~valid.to_numpy())[:10]
        ]
        raise ValueError("Every label must be 0 or 1; invalid/missing at " + "; ".join(examples))

    reviews = pd.DataFrame({
        "source_row": np.arange(1, len(source) + 1),
        "content": source.iloc[:, 0].astype("string").str.replace(
            r"\s+", " ", regex=True
        ).str.strip(),
        "labels": targets.to_numpy(dtype=np.float32).tolist(),
    })
    reviews = reviews.loc[reviews["content"].notna() & reviews["content"].ne("")].copy()
    valid_count = len(reviews)
    reviews["duplicate_key"] = reviews["content"].str.casefold()
    # Report conflicting annotations instead of silently combining their labels.
    signatures = reviews["labels"].map(tuple)
    conflicts = signatures.groupby(reviews["duplicate_key"]).nunique().gt(1).sum()
    if conflicts:
        print(f"Warning: {conflicts:,} duplicate texts have conflicting labels; keeping first occurrence.")
    reviews = reviews.drop_duplicates("duplicate_key", keep="first")
    reviews = reviews.drop(columns="duplicate_key").reset_index(drop=True)
    print(f"Read first {len(source):,} data rows (limit {max_rows:,})")
    print(f"Removed empty text: {len(source) - valid_count:,}")
    print(f"Removed duplicate text: {valid_count - len(reviews):,}")
    print(f"Retained: {len(reviews):,} reviews")
    return reviews, label_names


def ordered_split(reviews):
    """Keep CSV order and use the previous model's 80/10/10 boundaries."""
    train_end = int(len(reviews) * 0.8)
    validation_end = int(len(reviews) * 0.9)
    parts = (reviews.iloc[:train_end], reviews.iloc[train_end:validation_end],
             reviews.iloc[validation_end:])
    if any(part.empty for part in parts):
        raise ValueError("Not enough cleaned reviews for three nonempty splits.")
    return parts


def tokenize_split(reviews, tokenizer, max_length, name):
    dataset = Dataset.from_pandas(
        reviews[["content", "labels"]].reset_index(drop=True), preserve_index=False
    )
    return dataset.map(
        lambda batch: tokenizer(batch["content"], truncation=True, max_length=max_length),
        batched=True, remove_columns=["content"], desc=f"Tokenizing {name}",
    )


def predict_labels(logits, threshold):
    probabilities = torch.sigmoid(torch.as_tensor(logits, dtype=torch.float32)).numpy()
    return probabilities, (probabilities >= threshold).astype(np.int32)


def compute_metrics(prediction, threshold=0.5):
    _, predicted = predict_labels(prediction.predictions, threshold)
    actual = prediction.label_ids
    metrics = {
        # Exact match requires every label on a review to be correct.
        "exact_match_accuracy": accuracy_score(actual, predicted),
        "hamming_loss": hamming_loss(actual, predicted),
    }
    for average in ("micro", "macro"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual, predicted, average=average, zero_division=0
        )
        metrics.update({f"{average}_precision": precision,
                        f"{average}_recall": recall, f"{average}_f1": f1})
    return metrics


class MultilabelTrainer(Trainer):
    """Train with binary cross entropy and train-only positive class weights."""

    def __init__(self, pos_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32)
        # This loss is a batch mean; let Trainer scale gradient accumulation.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = inputs.copy()
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels.to(device=logits.device, dtype=torch.float32),
            pos_weight=self.pos_weight.to(logits.device),
        )
        return (loss, outputs) if return_outputs else loss


def report_test_results(trainer, dataset, reviews, label_names, threshold, output_dir):
    result = trainer.predict(dataset, metric_key_prefix="test")
    probabilities, predicted = predict_labels(result.predictions, threshold)
    print("\nTest evaluation:")
    print(json.dumps(result.metrics, indent=2))
    report = classification_report(
        result.label_ids, predicted, target_names=label_names, digits=4, zero_division=0
    )
    print(report)
    (output_dir / "test_classification_report.txt").write_text(report, encoding="utf-8")
    trainer.save_metrics("test", result.metrics)
    matrices = multilabel_confusion_matrix(result.label_ids, predicted)
    pd.DataFrame(
        matrices.reshape(len(label_names), 4), index=pd.Index(label_names, name="label"),
        columns=["true_negative", "false_positive", "false_negative", "true_positive"],
    ).to_csv(output_dir / "test_confusion_matrices.csv")
    predictions = reviews[["source_row", "content"]].reset_index(drop=True).copy()
    for i, label in enumerate(label_names):
        predictions[f"true_{label}"] = result.label_ids[:, i].astype(int)
        predictions[f"pred_{label}"] = predicted[:, i]
        predictions[f"prob_{label}"] = probabilities[:, i]
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)


def main():
    # Windows redirected consoles may not represent Unicode folder/label names.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("bert_review_labels_model"))
    parser.add_argument("--max-rows", type=int, default=20_000)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.epochs) or args.epochs <= 0 or min(
        args.batch_size, args.max_length, args.max_rows
    ) <= 0:
        parser.error("epochs, batch-size, max-length and max-rows must be positive")
    if args.max_length > 512:
        parser.error("BERT supports a maximum sequence length of 512 tokens")
    if not 0 < args.threshold < 1:
        parser.error("threshold must be between 0 and 1 (exclusive)")

    set_seed(42)
    reviews, label_names = load_reviews(args.csv, args.max_rows)
    train, validation, test = ordered_split(reviews)
    counts = pd.DataFrame({
        name: np.asarray(part["labels"].tolist()).sum(axis=0).astype(int)
        for name, part in (("train", train), ("validation", validation), ("test", test))
    }, index=label_names)
    print(f"\nTrain: {len(train):,}; validation: {len(validation):,}; test: {len(test):,}")
    print("Positive examples per label:")
    print(counts.to_string())
    for name, part in (("train", train), ("validation", validation), ("test", test)):
        constant = counts.index[(counts[name] == 0) | (counts[name] == len(part))].tolist()
        if constant:
            print(f"Warning: {name} has only one class for labels: {constant}")
    if args.check_data:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_dataset = tokenize_split(train, tokenizer, args.max_length, "train")
    validation_dataset = tokenize_split(validation, tokenizer, args.max_length, "validation")
    test_dataset = tokenize_split(test, tokenizer, args.max_length, "test")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(label_names), problem_type="multi_label_classification",
        id2label=dict(enumerate(label_names)),
        label2id={label: i for i, label in enumerate(label_names)},
    )
    # Save inference settings alongside the model's label mapping.
    model.config.multilabel_threshold = args.threshold
    model.config.review_max_length = args.max_length
    positives = counts["train"].to_numpy(dtype=np.float32)
    negatives = len(train) - positives
    pos_weight = np.ones(len(label_names), dtype=np.float32)
    nonconstant = (positives > 0) & (negatives > 0)
    pos_weight[nonconstant] = negatives[nonconstant] / positives[nonconstant]
    gradient_accumulation_steps = 2
    steps_per_epoch = math.ceil(len(train) / (args.batch_size * gradient_accumulation_steps))
    training_args = TrainingArguments(
        output_dir=str(args.output_dir), num_train_epochs=args.epochs,
        learning_rate=2e-5, weight_decay=0.01,
        warmup_steps=math.ceil(steps_per_epoch * args.epochs * 0.1),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1", greater_is_better=True,
        save_total_limit=2, fp16=torch.cuda.is_available(), logging_steps=100,
        report_to="none", seed=42, data_seed=42,
    )
    trainer = MultilabelTrainer(
        pos_weight=pos_weight, model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=validation_dataset,
        processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=lambda prediction: compute_metrics(prediction, args.threshold),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([
        part[["source_row"]].assign(split=name)
        for name, part in (("train", train), ("validation", validation), ("test", test))
    ]).to_csv(args.output_dir / "data_splits.csv", index=False)
    print(f"\nDevice: {trainer.args.device}; model: {args.model}")
    training_result = trainer.train()
    trainer.save_metrics("train", training_result.metrics)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    trainer.save_metrics("validation", trainer.evaluate(metric_key_prefix="validation"))
    report_test_results(
        trainer, test_dataset, test, label_names, args.threshold, args.output_dir
    )
    print(f"\nBest model and evaluation files saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
