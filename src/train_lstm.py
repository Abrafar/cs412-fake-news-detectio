import argparse
import re
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common import compute_metrics, load_splits, save_result, set_seed

TOKEN = re.compile(r"[a-z0-9']+")
PAD, UNK = 0, 1


# Lowercase word tokens.
def tokenize(text):
    return TOKEN.findall(str(text).lower())


# Map each text to a fixed-length id array and its true length.
def encode(texts, vocab, max_len):
    ids = np.zeros((len(texts), max_len), dtype=np.int64)
    lengths = np.zeros(len(texts), dtype=np.int64)
    for i, text in enumerate(texts):
        tokens = [vocab.get(t, UNK) for t in tokenize(text)[:max_len]]
        ids[i, : len(tokens)] = tokens
        lengths[i] = max(len(tokens), 1)
    return torch.from_numpy(ids), torch.from_numpy(lengths)


class Classifier(nn.Module):
    def __init__(self, vocab_size, cell, embed_dim, hidden, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden * 2, 1)

    def forward(self, ids):
        mask = (ids != PAD).unsqueeze(-1)
        states, _ = self.rnn(self.drop(self.embed(ids)))
        pooled = states.masked_fill(~mask, -1e9).max(dim=1).values
        return self.out(self.drop(pooled)).squeeze(-1)


# Return sigmoid scores for every row of a loader.
def predict(model, loader, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for ids, _ in loader:
            scores.append(torch.sigmoid(model(ids.to(device))).cpu())
    return torch.cat(scores).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed_cleaned")
    parser.add_argument("--name", default=None)
    parser.add_argument("--cell", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=300)
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-subset", type=int, default=0)
    args = parser.parse_args()
    mode = "raw" if "raw" in args.data_dir else "cleaned"
    name = args.name or f"{args.cell}_{mode}"

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, test = load_splits(args.data_dir)
    if args.train_subset:
        train = train.sample(n=min(args.train_subset, len(train)), random_state=42)

    counts = Counter(t for text in train.text for t in tokenize(text))
    words = [w for w, _ in counts.most_common(args.vocab_size - 2)]
    vocab = {w: i + 2 for i, w in enumerate(words)}

    def loader(df, shuffle):
        ids, lengths = encode(df.text.tolist(), vocab, args.max_len)
        labels = torch.tensor(df.label.values, dtype=torch.float32)
        return DataLoader(TensorDataset(ids, labels), batch_size=args.batch_size, shuffle=shuffle)

    train_loader, val_loader = loader(train, True), loader(val, False)

    model = Classifier(len(vocab) + 2, args.cell, args.embed_dim, args.hidden, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    best_f1, best_epoch, best_state, history = -1.0, 0, None, []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for ids, labels in train_loader:
            ids, labels = ids.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(ids), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(ids)
        val_metrics = compute_metrics(val.label, predict(model, val_loader, device))
        history.append({"epoch": epoch, "train_loss": total / len(train), "val_f1": val_metrics["f1"], "val_accuracy": val_metrics["accuracy"]})
        print(history[-1])
        if val_metrics["f1"] > best_f1:
            best_f1, best_epoch = val_metrics["f1"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_seconds = time.perf_counter() - start

    model.load_state_dict(best_state)
    val_scores = predict(model, val_loader, device)
    start = time.perf_counter()
    test_scores = predict(model, loader(test, False), device)
    infer_ms = (time.perf_counter() - start) * 1000 / len(test) * 1000

    result = {
        "name": name,
        "model": "BiLSTM" if args.cell == "lstm" else "BiGRU",
        "data_dir": args.data_dir,
        "seed": 42,
        "params": {k: getattr(args, k.replace("-", "_")) for k in ("cell", "epochs", "batch-size", "max-len", "vocab-size", "embed-dim", "hidden", "dropout", "lr", "train-subset")},
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "val": compute_metrics(val.label, val_scores),
        "test": compute_metrics(test.label, test_scores),
        "history": history,
        "best_epoch": best_epoch,
        "train_seconds": train_seconds,
        "infer_ms_per_1000_docs": infer_ms,
        "infer_includes_tokenization": True,
        "device": device.type,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
        "n_parameters": sum(p.numel() for p in model.parameters()),
    }
    print({k: round(v, 4) for k, v in result["test"].items() if k != "confusion_matrix"})
    save_result(name, result)


if __name__ == "__main__":
    main()
