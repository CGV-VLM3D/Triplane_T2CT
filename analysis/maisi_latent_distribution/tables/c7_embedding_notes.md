# C7: Per-sample Embedding Notes

## Parameters
- t-SNE: n_components=2, perplexity=30, init='pca', random_state=0
- UMAP: n_components=2, n_neighbors=15, min_dist=0.1, random_state=0 (available)
- Feature matrix: 28-dim (7 stats × 4 channels), standardized (z-score)
- Samples: 6000 total (train=5000, valid=1000)

## Label Join
- Join rate: 1.000 (6000/6000 samples matched)
- Most common abnormality: 'Lung nodule' (2670 positives across all splits)
- Additional panels: binary 'Lung nodule' color, total positive label count color

## Observations
- train and valid points are strongly interleaved with no visible cluster separation (between-split variance ratio = 0.000 ≈ 0) — good: no domain drift in the latent space.
- Between-split variance ratio: 0.0000
- t-SNE KL divergence: 2.1014
