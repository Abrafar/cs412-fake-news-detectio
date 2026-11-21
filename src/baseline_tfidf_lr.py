import argparse
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import compute_metrics, load_splits, save_result, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed_cleaned")
    parser.add_argument("--name", default="tfidf_lr_cleaned")
    args = parser.parse_args()

    set_seed()
    train, val, test = load_splits(args.data_dir)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
    x_train = vec.fit_transform(train.text)
    x_val = vec.transform(val.text)

    best = None
    for c in (0.1, 1, 10, 100):
        clf = LogisticRegression(C=c, max_iter=1000, solver="liblinear")
        clf.fit(x_train, train.label)
        val_f1 = compute_metrics(val.label, clf.predict_proba(x_val)[:, 1])["f1"]
        print(f"C={c}: val F1 {val_f1:.4f}")
        if best is None or val_f1 > best[0]:
            best = (val_f1, c)
    best_c = best[1]

    start = time.perf_counter()
    vec_final = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
    x_train = vec_final.fit_transform(train.text)
    clf = LogisticRegression(C=best_c, max_iter=1000, solver="liblinear").fit(x_train, train.label)
    train_seconds = time.perf_counter() - start

    x_val = vec_final.transform(val.text)
    start = time.perf_counter()
    test_prob = clf.predict_proba(vec_final.transform(test.text))[:, 1]
    infer_ms = (time.perf_counter() - start) * 1000 / len(test) * 1000

    names = np.array(vec_final.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    result = {
        "name": args.name,
        "model": "TF-IDF + Logistic Regression",
        "data_dir": args.data_dir,
        "seed": 42,
        "params": {"ngram_range": [1, 2], "min_df": 2, "max_features": 200000, "C": best_c, "n_features": int(x_train.shape[1])},
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "val": compute_metrics(val.label, clf.predict_proba(x_val)[:, 1]),
        "test": compute_metrics(test.label, test_prob),
        "train_seconds": train_seconds,
        "infer_ms_per_1000_docs": infer_ms,
        "top_features_fake": names[order[::-1][:20]].tolist(),
        "top_features_real": names[order[:20]].tolist(),
    }
    print({k: round(v, 4) for k, v in result["test"].items() if k != "confusion_matrix"})
    print("top fake cues:", result["top_features_fake"][:12])
    print("top real cues:", result["top_features_real"][:12])
    save_result(args.name, result)


if __name__ == "__main__":
    main()
