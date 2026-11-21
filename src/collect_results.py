import json

from common import ROOT

ORDER = ["tfidf_lr", "lstm", "gru", "distilbert"]


# Sort key: model family first, then raw before cleaned.
def sort_key(result):
    family = next((i for i, f in enumerate(ORDER) if result["name"].startswith(f)), 99)
    return family, result["name"]


def main():
    results = [json.loads(p.read_text()) for p in (ROOT / "results").glob("*.json")]
    results.sort(key=sort_key)
    header = "| Run | Data | Acc | Prec | Rec | F1 | AUROC | Train (s) | ms/1k docs | VRAM (MB) |"
    lines = [header, "|" + "---|" * 10]
    latex = []
    for r in results:
        t = r["test"]
        mode = "raw" if "raw" in r["data_dir"] else "cleaned"
        vram = f"{r['peak_vram_mb']:.0f}" if r.get("peak_vram_mb") else "-"
        lines.append(f"| {r['model']} | {mode} | {t['accuracy']:.4f} | {t['precision']:.4f} | {t['recall']:.4f} | {t['f1']:.4f} | {t['auroc']:.4f} | {r['train_seconds']:.0f} | {r['infer_ms_per_1000_docs']:.0f} | {vram} |")
        latex.append(f"{r['model']} & {mode} & {t['accuracy']:.3f} & {t['precision']:.3f} & {t['recall']:.3f} & {t['f1']:.3f} & {t['auroc']:.3f} & {r['train_seconds']:.0f} \\\\")
    table = "\n".join(lines)
    print(table)
    (ROOT / "results" / "summary.md").write_text(table + "\n")
    (ROOT / "results" / "summary_latex_rows.txt").write_text("\n".join(latex) + "\n")
    print("\nconfusion matrices [[TN, FP], [FN, TP]] (positive = fake):")
    for r in results:
        print(r["name"], r["test"]["confusion_matrix"])


if __name__ == "__main__":
    main()
