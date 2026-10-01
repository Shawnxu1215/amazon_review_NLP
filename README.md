# Amazon Review NLP Project

This repository contains an NLP project for Amazon shopping app reviews. It includes
review sampling, sentiment modeling, multi-label issue annotation data, and BERT
fine-tuning scripts for predicting review issue categories.

## Project Contents

- `sample_amazon_reviews.py` - creates a reproducible 30,000-row review sample.
- `sentiment_baseline.py` - TF-IDF + Logistic Regression sentiment baseline.
- `sentiment_bert.py` - BERT sentiment classifier using star ratings.
- `review_labels_bert.py` - BERT multi-label classifier for review issue labels.
- `sampled_30k.csv` - sampled review dataset.
- `sampled_30k_annotated (2).csv` - annotated multi-label dataset.
- `bert_review_labels_model/` - final trained multi-label BERT model artifacts.

The original raw `ShoppingAppReviews Dataset/` folder is not committed because
it is large. The repository includes the sampled CSVs used for this project.

## Issue Labels

The annotated dataset uses the following binary labels:

- `App_Performance`
- `Search_Discovery`
- `Ordering_Checkout`
- `Payment_Billing`
- `Delivery_Tracking`
- `Returns_Refunds`
- `Product_Seller`
- `Pricing_Promotions`
- `Account_Prime`
- `Customer_Service`
- `No_Issue`
- `other_issues`

Multiple issue labels may apply to the same review. `No_Issue` is reserved for
5-star reviews that do not contain a clear issue.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

For model weights, install Git LFS before cloning or pushing:

```bash
git lfs install
```

## Data Sampling

Create a reproducible sampled dataset:

```bash
python sample_amazon_reviews.py
```

The sampler preserves original columns and creates `sampled_30k.csv`.

## Sentiment Baseline

Run the TF-IDF baseline. If the raw source dataset is not present locally, pass
an available CSV path with `--csv`:

```bash
python sentiment_baseline.py
```

Run the BERT sentiment model:

```bash
python sentiment_bert.py
```

## Multi-Label Issue Model

Validate the annotated data without training:

```bash
python review_labels_bert.py --check-data
```

Train the multi-label BERT model:

```bash
python review_labels_bert.py
```

By default, this reads the first 20,000 annotated rows and saves the model to
`bert_review_labels_model/`.

## Model Artifacts

The final model weight file is:

```text
bert_review_labels_model/model.safetensors
```

This file is large and should be stored with Git LFS. Checkpoint folders and
optimizer states are ignored because they are very large and not needed for
normal inference or project review.

## Notes

- `.venv/`, `__pycache__/`, backup CSVs, and training checkpoints are excluded.
- If you need to reproduce training exactly, keep the scripts, annotated CSV,
  config/tokenizer files, and final `model.safetensors`.
- Large raw model checkpoints should be shared separately or tracked with Git LFS
  only if required.
