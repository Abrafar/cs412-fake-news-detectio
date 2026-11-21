import argparse
import json
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from common import ROOT, SEED

DATELINE = re.compile(r"^\s*(?:[A-Za-z][A-Za-z.,'/&\- ]{0,60}\s)?\(\s*reuters\s*\)\s*[-–—:]?\s*", re.IGNORECASE)
SOURCE_WORD = re.compile(r"\breuters\b", re.IGNORECASE)


# Remove the news-agency dateline and any remaining source name.
def strip_source_cues(text):
    text = DATELINE.sub("", text, count=1)
    return SOURCE_WORD.sub(" ", text)


# Collapse runs of whitespace.
def squash(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--min-chars", type=int, default=20)
    args = parser.parse_args()

    raw = ROOT / args.raw_dir
    real = pd.read_csv(raw / "True.csv")
    fake = pd.read_csv(raw / "Fake.csv")
    real["label"] = 0
    fake["label"] = 1
    df = pd.concat([real, fake], ignore_index=True)
    stats = {"rows_loaded": {"real": len(real), "fake": len(fake)}}

    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["body_chars"] = df["text"].str.strip().str.len()
    stats["real_with_reuters_dateline"] = int(real["text"].fillna("").str.contains(DATELINE).sum())
    stats["fake_with_reuters_dateline"] = int(fake["text"].fillna("").str.contains(DATELINE).sum())

    short = df["body_chars"] < args.min_chars
    stats["dropped_empty_or_short_body"] = int(short.sum())
    df = df[~short].copy()

    df["raw_text"] = (df["title"] + ". " + df["text"]).map(squash)
    df["key"] = df["raw_text"].str.lower()

    both = df.groupby("key")["label"].nunique()
    conflicting = set(both[both > 1].index)
    stats["dropped_conflicting_label_texts"] = int(df["key"].isin(conflicting).sum())
    df = df[~df["key"].isin(conflicting)]

    before = len(df)
    df = df.drop_duplicates("key")
    stats["dropped_duplicate_texts"] = before - len(df)
    df = df.reset_index(drop=True)
    stats["rows_kept"] = {"real": int((df.label == 0).sum()), "fake": int((df.label == 1).sum())}

    idx_train, idx_rest = train_test_split(df.index, test_size=0.30, stratify=df["label"], random_state=SEED)
    idx_val, idx_test = train_test_split(idx_rest, test_size=0.50, stratify=df.loc[idx_rest, "label"], random_state=SEED)
    splits = {"train": idx_train, "val": idx_val, "test": idx_test}

    cleaned = df["raw_text"].map(strip_source_cues).map(squash)
    stats["cleaned_changed_rows"] = int((cleaned != df["raw_text"]).sum())
    stats["median_words_by_class"] = {
        "real": float(df.loc[df.label == 0, "raw_text"].str.split().str.len().median()),
        "fake": float(df.loc[df.label == 1, "raw_text"].str.split().str.len().median()),
    }

    for mode, column in (("raw", df["raw_text"]), ("cleaned", cleaned)):
        folder = ROOT / "data" / f"processed_{mode}"
        folder.mkdir(parents=True, exist_ok=True)
        mode_stats = dict(stats)
        mode_stats["split_sizes"] = {}
        for name, idx in splits.items():
            out = pd.DataFrame({"text": column.loc[idx].values, "label": df.loc[idx, "label"].values})
            out.to_csv(folder / f"{name}.csv", index=False)
            mode_stats["split_sizes"][name] = {
                "total": len(out),
                "real": int((out.label == 0).sum()),
                "fake": int((out.label == 1).sum()),
            }
        (folder / "stats.json").write_text(json.dumps(mode_stats, indent=2))

    print(json.dumps(mode_stats, indent=2))


if __name__ == "__main__":
    main()
