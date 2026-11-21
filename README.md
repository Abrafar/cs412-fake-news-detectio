# CS 412 Fake News Detection

Compares TF-IDF + Logistic Regression, a BiLSTM/BiGRU, and a fine-tuned DistilBERT on the Kaggle Fake and Real News dataset.

## Setup
1. Install PyTorch with a CUDA wheel from https://pytorch.org (command from the selector).
2. `pip install -r requirements.txt`
3. Put `True.csv` and `Fake.csv` in `data/raw/`.

## Run (from the repo root)
```
python src/prepare_data.py
python src/baseline_tfidf_lr.py --data-dir data/processed_raw --name tfidf_lr_raw
python src/baseline_tfidf_lr.py --data-dir data/processed_cleaned --name tfidf_lr_cleaned
python src/train_lstm.py --data-dir data/processed_cleaned
python src/train_lstm.py --data-dir data/processed_raw
python src/finetune_distilbert.py --data-dir data/processed_cleaned
python src/finetune_distilbert.py --data-dir data/processed_raw
python src/collect_results.py
```
Results are written to `results/*.json` and `results/summary.md`. Seed is 42 everywhere; preprocessing is fit on the training split only.

`raw` keeps the source cues (Reuters datelines); `cleaned` strips them. Comparing the two shows how much a model relies on the cue.
