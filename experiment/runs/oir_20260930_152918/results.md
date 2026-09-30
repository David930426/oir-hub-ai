# Results — oir  (2026-09-30 15:29)

Labels: `data/labels_oir_draft.csv` · cases: 200 · dev 79 / test 121 · seed 13

## Class distribution

| split | UPDATE | NEW | REDUNDANT |
|---|---|---|---|
| dev | 54 | 23 | 2 |
| test | 82 | 35 | 4 |

Test cases by sampling pool: random {'UPDATE': 18, 'NEW': 23, 'REDUNDANT': 1}, semantic {'UPDATE': 24, 'NEW': 2}, title {'UPDATE': 40, 'NEW': 10, 'REDUNDANT': 3}

Gold target (UPDATE/REDUNDANT, n=86) in e5 top-1: 0.76, top-5: 1.00, top-10: 1.00 — M2 cannot name a target that is not in its top-5.

## Main table (test split)

| condition | macro-F1 [95% CI] | acc | F1 UPD | F1 NEW | F1 RED | target cond. | target joint | s/decision | tokens/decision | LLM calls |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 majority | 0.269 [0.25, 0.29] | 0.678 | 0.81 | 0.00 | 0.00 | 0.00 | 0.00 | ≈0 | 0 | – |
| B1 metadata/title | 0.607 [0.43, 0.79] | 0.777 | 0.85 | 0.57 | 0.40 | 0.77 | 0.72 | ≈0 | 0 | – |
| B2 cosine (e5) | 0.519 [0.46, 0.57] | 0.802 | 0.87 | 0.69 | 0.00 | 0.75 | 0.70 | ≈0 | 0 | – |
| B3 BM25 | 0.727 [0.48, 0.87] | 0.810 | 0.87 | 0.64 | 0.67 | 0.78 | 0.73 | ≈0 | 0 | – |
| B4 reranker | 0.481 [0.38, 0.59] | 0.628 | 0.73 | 0.49 | 0.22 | 0.43 | 0.30 | ≈0 | 0 | – |

*s/decision and tokens/decision are averaged over all test cases, so the cascade's cost includes the cases it settled without the LLM. Latency is wall-clock on the local GPU and includes model load for the first call.*

## Tuned parameters (from dev)

- B0 majority: label=UPDATE
- B1 metadata/title: t_title=0.7, t_body=1.01
- B2 cosine (e5): t_upd=0.9158, t_red=0.9673
- B3 BM25: t_upd=0.3592, t_red=0.9976
- B4 reranker: t_upd=0.9365, t_red=0.9974

## Macro-F1 by sampling pool (test)

| condition | random | semantic | title |
|---|---|---|---|
| B0 majority | 0.20 (n=42) | 0.32 (n=26) | 0.29 (n=53) |
| B1 metadata/title | 0.84 (n=42) | 0.42 (n=26) | 0.29 (n=53) |
| B2 cosine (e5) | 0.54 (n=42) | 0.32 (n=26) | 0.38 (n=53) |
| B3 BM25 | 0.85 (n=42) | 0.32 (n=26) | 0.55 (n=53) |
| B4 reranker | 0.41 (n=42) | 0.34 (n=26) | 0.49 (n=53) |

## Confusion matrices (rows = gold, cols = predicted: UPD / NEW / RED)

**B0 majority**
```
UPD    82    0    0
NEW    35    0    0
RED     4    0    0
```
**B1 metadata/title**
```
UPD    77    5    0
NEW    19   16    0
RED     3    0    1
```
**B2 cosine (e5)**
```
UPD    76    4    2
NEW    14   21    0
RED     3    1    0
```
**B3 BM25**
```
UPD    77    5    0
NEW    16   19    0
RED     2    0    2
```
**B4 reranker**
```
UPD    58   14   10
NEW    17   16    2
RED     2    0    2
```

## H1 error counts (test)

| condition | false-UPDATE (gold NEW) | missed-UPDATE | wrong target | of which flagged template_trap / reworded |
|---|---|---|---|---|
| B0 majority | 35 | 0 | 82 | 6 / 0 |
| B1 metadata/title | 19 | 5 | 18 | 4 / 3 |
| B2 cosine (e5) | 14 | 6 | 19 | 2 / 2 |
| B3 BM25 | 16 | 5 | 17 | 3 / 2 |
| B4 reranker | 17 | 24 | 33 | 3 / 5 |