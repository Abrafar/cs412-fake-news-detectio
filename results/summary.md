| Run | Data | Acc | Prec | Rec | F1 | AUROC | Train (s) | ms/1k docs | VRAM (MB) |
|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | cleaned | 0.9941 | 0.9958 | 0.9912 | 0.9935 | 0.9997 | 21 | 427 | - |
| TF-IDF + Logistic Regression | raw | 0.9959 | 0.9969 | 0.9939 | 0.9954 | 0.9998 | 21 | 458 | - |
| BiLSTM | cleaned | 0.9857 | 0.9941 | 0.9740 | 0.9840 | 0.9989 | 26 | 181 | 386 |
| BiLSTM | raw | 0.9978 | 0.9966 | 0.9985 | 0.9975 | 0.9998 | 27 | 195 | 386 |
| DistilBERT (fine-tuned) | cleaned | 0.9983 | 0.9981 | 0.9981 | 0.9981 | 1.0000 | 376 | 1977 | 1697 |
| DistilBERT (fine-tuned) | raw | 0.9997 | 0.9992 | 1.0000 | 0.9996 | 1.0000 | 378 | 2065 | 1697 |
