# Sentiment Analysis with DistilBERT

Fine-tuning a pre-trained Transformer (DistilBERT) for binary sentiment classification on the IMDB dataset using PyTorch and Hugging Face Transformers.

---

## Project Overview

This project demonstrates:

- Loading and preprocessing the IMDB dataset
- Tokenization using a pre-trained DistilBERT tokenizer
- Fine-tuning DistilBERT for sequence classification
- Evaluating performance using accuracy, precision, recall, and F1 score
- Performing inference on custom movie reviews

The model is trained on a 500-sample subset and evaluated on 100 samples for fast experimentation.

---

## Tech Stack

- PyTorch 2.4
- Hugging Face Transformers
- Hugging Face Datasets
- scikit-learn
- Accelerate

---

## Model

- Base model: `distilbert-base-uncased`
- Task: Binary sentiment classification
- Max sequence length: 128
- Optimizer: AdamW (via Trainer)
- Epochs: 2

---

## Results

The fine-tuned model achieves performance significantly above random baseline (50%) on the IMDB test subset.

Metrics computed:
- Accuracy
- Precision
- Recall
- F1 Score

---

## Example Predictions

"This movie was absolutely amazing!"
→ POSITIVE

"Terrible movie. Waste of time."
→ NEGATIVE


---

## How to Run

1. Install dependencies:

pip install -r requirements.txt

2. Open the notebook:

jupyter notebook sentiment_analysis_distilbert.ipynb


3. Run all cells.

---

## Why This Project Matters

This project demonstrates practical understanding of:

- Transformer architectures
- Transfer learning
- Model fine-tuning
- NLP preprocessing
- Evaluation methodology

It serves as a foundational NLP pipeline that can be extended to production or larger datasets.
