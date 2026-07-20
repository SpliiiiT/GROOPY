# Word-sign bake-off results

| Rank | Model | Acc | Macro-F1 | Latency (ms) | Size (MB) | Score |
|------|-------|-----|----------|--------------|-----------|-------|
| 1 | gru | 0.7542 | 0.7472 | 142.22 | 2.35 | **0.724** |
| 2 | transformer | 0.7458 | 0.7494 | 21.35 | 3.13 | **0.7108** |
| 3 | bilstm | 0.7797 | 0.7914 | 237.23 | 4.54 | **0.6** |
| 4 | lstm | 0.6864 | 0.6487 | 118.18 | 3.08 | **0.2436** |

_Weights: accuracy 60%, latency 20%, size 20%. Test set = held-out landmark sequences (no augmentation)._