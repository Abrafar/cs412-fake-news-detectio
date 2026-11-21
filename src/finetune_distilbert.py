import argparse
import time

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import compute_metrics, load_splits, save_result, set_seed


# Tokenize a dataframe into a DataLoader of (input_ids, attention_mask, label).
def make_loader(df, tokenizer, max_len, batch_size, shuffle):
    enc = tokenizer(df.text.tolist(), truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
    labels = torch.tensor(df.label.values, dtype=torch.long)
    return DataLoader(TensorDataset(enc["input_ids"], enc["attention_mask"], labels), batch_size=batch_size, shuffle=shuffle)


# Return the probability of the fake class for every row of a loader.
def predict(model, loader, device, use_amp):
    model.eval()
    scores = []
    with torch.no_grad():
        for ids, mask, _ in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            scores.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu())
    return torch.cat(scores).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed_cleaned")
    parser.add_argument("--name", default=None)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train-subset", type=int, default=0)
    args = parser.parse_args()
    mode = "raw" if "raw" in args.data_dir else "cleaned"
    name = args.name or f"distilbert_{mode}"

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    train, val, test = load_splits(args.data_dir)
    if args.train_subset:
        train = train.sample(n=min(args.train_subset, len(train)), random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2).to(device)
    train_loader = make_loader(train, tokenizer, args.max_len, args.batch_size, True)
    val_loader = make_loader(val, tokenizer, args.max_len, args.batch_size * 2, False)
    test_loader = make_loader(test, tokenizer, args.max_len, args.batch_size * 2, False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    best_f1, best_state, history = -1.0, None, []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for step, (ids, mask, labels) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device)).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item() * len(ids)
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {loss.item():.4f}", flush=True)
        val_metrics = compute_metrics(val.label, predict(model, val_loader, device, use_amp))
        history.append({"epoch": epoch, "train_loss": total / len(train), "val_f1": val_metrics["f1"], "val_accuracy": val_metrics["accuracy"]})
        print(history[-1], flush=True)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_seconds = time.perf_counter() - start

    model.load_state_dict(best_state)
    val_scores = predict(model, val_loader, device, use_amp)
    start = time.perf_counter()
    test_scores = predict(model, test_loader, device, use_amp)
    infer_ms = (time.perf_counter() - start) * 1000 / len(test) * 1000

    result = {
        "name": name,
        "model": "DistilBERT (fine-tuned)",
        "data_dir": args.data_dir,
        "seed": 42,
        "params": {"model_name": args.model_name, "epochs": args.epochs, "batch_size": args.batch_size, "max_len": args.max_len, "lr": args.lr, "train_subset": args.train_subset, "fp16_autocast": use_amp},
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "val": compute_metrics(val.label, val_scores),
        "test": compute_metrics(test.label, test_scores),
        "history": history,
        "train_seconds": train_seconds,
        "infer_ms_per_1000_docs": infer_ms,
        "device": device.type,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
        "n_parameters": sum(p.numel() for p in model.parameters()),
    }
    print({k: round(v, 4) for k, v in result["test"].items() if k != "confusion_matrix"})
    save_result(name, result)


if __name__ == "__main__":
    main()
