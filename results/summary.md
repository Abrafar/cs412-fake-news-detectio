| Run | Data | Acc | Prec | Rec | F1 | AUROC | Train (s) | ms/1k docs | VRAM (MB) |
|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | cleaned | 0.9933 | 0.9950 | 0.9901 | 0.9925 | 0.9997 | 25 | 843 | - |
| TF-IDF + Logistic Regression | raw | 0.9950 | 0.9962 | 0.9927 | 0.9944 | 0.9998 | 20 | 425 | - |
| BiLSTM | cleaned | 0.9881 | 0.9870 | 0.9866 | 0.9868 | 0.9990 | 26 | 89 | 386 |
| BiLSTM | raw | 0.9978 | 0.9966 | 0.9985 | 0.9975 | 0.9998 | 26 | 89 | 386 |
| DistilBERT (fine-tuned) | cleaned | 0.9997 | 0.9996 | 0.9996 | 0.9996 | 1.0000 | 795 | 1498 | 1697 |
| DistilBERT (fine-tuned) | raw | 0.9995 | 0.9992 | 0.9996 | 0.9994 | 1.0000 | 379 | 1561 | 1697 |
