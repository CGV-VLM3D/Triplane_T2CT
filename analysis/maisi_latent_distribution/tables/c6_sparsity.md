# C6: Sparsity Profile

## P(|x| < ε) per channel

| Channel | ε=0.01 | ε=0.05 | ε=0.1 | ε=0.5 | ε=1.0 |
|---------|--------|--------|--------|--------|--------|
| ch0     | 0.0000 | 0.0433 | 0.0877 | 0.4739 | 0.7817 |
| ch1     | 0.0000 | 0.1399 | 0.2908 | 0.9999 | 1.0000 |
| ch2     | 0.0000 | 0.0689 | 0.1390 | 0.6157 | 0.8602 |
| ch3     | 0.0000 | 0.0876 | 0.1692 | 0.6770 | 0.8967 |

**Average P(|x|<0.05) across channels: 0.0849**

> Decision rule: if avg > 0.60 → upgrade FSQ and VQ small codebook


## Gini Coefficient, Entropy, Effective Sparsity Rank

| Channel | Gini | Entropy (bits) | Eff. rank (2^H) |
|---------|------|----------------|-----------------|
| ch0     | 0.4236 | 6.1473 | 70.88 |
| ch1     | 0.3373 | 3.7879 | 13.81 |
| ch2     | 0.4621 | 5.8690 | 58.44 |
| ch3     | 0.4741 | 5.6742 | 51.06 |

**Mean effective sparsity rank: 48.55**

> If < 10 → upgrade quantization candidates; if > 50 → continuous baseline remains top.
