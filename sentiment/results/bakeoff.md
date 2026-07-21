# Sentiment bake-off results

| Rank | Model | Accuracy | Latency (ms) | Size (MB) | Score |
|------|-------|----------|---------------|-----------|-------|
| 1 | distilbert | 0.86 | 40.46 | 267.82 | **0.7443** |
| 2 | scratch | 0.768 | 0.6 | 0.84 | **0.6618** |
| 3 | twitter_roberta | 0.724 | 81.18 | 498.59 | **0.0** |

_Scorecard weights: accuracy 50%, latency 30%, size 20%. Accuracy is on a held-out IMDB slice (binary ground truth); a 'neutral' prediction always counts as wrong._