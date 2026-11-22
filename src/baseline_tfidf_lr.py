import argparse
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import SEED, compute_metrics, load_splits, save_result, set_seed

C_GRID = (0.1, 1, 10, 100, 1000, 10000)


# Word 1-2-gram TF-IDF features, fit on training text only.
def make_vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed_cleaned")
    parser.add_argument("--name", default="tfidf_lr_cleaned")
    args = parser.parse_args()

    set_seed()
    train, val, test = load_splits(args.data_dir)

    vec = make_vectorizer()
    x_train = vec.fit_transform(train.text)
    x_val = vec.transform(val.text)

    val_f1_by_c = {}
    for c in C_GRID:
        clf = LogisticRegression(C=c, max_iter=1000, solver="liblinear", random_state=SEED)
        clf.fit(x_train, train.label)
        val_f1_by_c[str(c)] = compute_metrics(val.label, clf.predict_proba(x_val)[:, 1])["f1"]
        print(f"C={c}: val F1 {val_f1_by_c[str(c)]:.4f}")
    best_c = max(C_GRID, key=lambda c: val_f1_by_c[str(c)])

    start = time.perf_counter()
    vec_final = make_vectorizer()
    x_train = vec_final.fit_transform(train.text)
    clf = LogisticRegression(C=best_c, max_iter=1000, solver="liblinear", random_state=SEED).fit(x_train, train.label)
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
        "seed": SEED,
        "params": {"ngram_range": [1, 2], "min_df": 2, "max_features": 200000, "C": best_c, "n_features": int(x_train.shape[1])},
        "val_f1_by_C": val_f1_by_c,
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "val": compute_metrics(val.label, clf.predict_proba(x_val)[:, 1]),
        "test": compute_metrics(test.label, test_prob),
        "train_seconds": train_seconds,
        "infer_ms_per_1000_docs": infer_ms,
        "infer_includes_tokenization": True,
        "n_parameters": int(clf.coef_.size + clf.intercept_.size),
        "top_features_fake": names[order[::-1][:20]].tolist(),
        "top_features_real": names[order[:20]].tolist(),
    }
    print({k: round(v, 4) for k, v in result["test"].items() if k != "confusion_matrix"})
    print("top fake cues:", result["top_features_fake"][:12])
    print("top real cues:", result["top_features_real"][:12])
    save_result(args.name, result)


if __name__ == "__main__":
    main()
