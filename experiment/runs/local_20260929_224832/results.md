# Results — local  (2026-09-29 22:48)

> **SMOKE RUN: dev = test = all cases. Thresholds are fit on the very cases they are scored on. These numbers check that the code runs; they are not results.**

Labels: `C:/Users/DAVIDG~1/AppData/Local/Temp/claude/C--Users-David-Gunawan-Wisno-Documents-project-final-project-oir-hub-ai/2b2bfd3a-f18d-4f32-8fc2-3cce8392c2bc/scratchpad/smoke_labels_local.csv` · cases: 17 · dev 17 / test 17 · seed 13

## Class distribution

| split | UPDATE | NEW | REDUNDANT |
|---|---|---|---|
| dev | 6 | 10 | 1 |
| test | 6 | 10 | 1 |

Test cases by sampling pool: pilot {'REDUNDANT': 1, 'NEW': 10, 'UPDATE': 6}

Gold target (UPDATE/REDUNDANT, n=7) in e5 top-1: 1.00, top-5: 1.00, top-10: 1.00 — M2 cannot name a target that is not in its top-5.

## Main table (test split)

| condition | macro-F1 [95% CI] | acc | F1 UPD | F1 NEW | F1 RED | target cond. | target joint | s/decision | tokens/decision | LLM calls |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 majority | 0.247 [0.17, 0.30] | 0.588 | 0.00 | 0.74 | 0.00 | – | 0.00 | ≈0 | 0 | – |
| B1 metadata/title | 0.954 [0.55, 1.00] | 0.941 | 0.91 | 0.95 | 1.00 | 0.80 | 0.67 | ≈0 | 0 | – |
| B2 cosine (e5) | 0.689 [0.44, 1.00] | 0.824 | 0.67 | 1.00 | 0.40 | 1.00 | 0.50 | ≈0 | 0 | – |
| B3 BM25 | 0.954 [0.55, 1.00] | 0.941 | 0.91 | 0.95 | 1.00 | 1.00 | 0.83 | ≈0 | 0 | – |
| B4 reranker | 0.775 [0.45, 0.96] | 0.824 | 0.77 | 0.89 | 0.67 | 0.60 | 0.50 | ≈0 | 0 | – |
| M1 LLM zero-shot | 0.296 [0.15, 0.44] | 0.529 | 0.22 | 0.67 | 0.00 | 0.00 | 0.00 | 9.10 | 1664 | 17/17 |
| M2 LLM + candidates | 0.545 [0.38, 0.67] | 0.824 | 0.73 | 0.91 | 0.00 | 0.25 | 0.17 | 7.22 | 4063 | 17/17 |
| M3 LLM + few-shot | 0.444 [0.26, 0.59] | 0.706 | 0.50 | 0.83 | 0.00 | 1.00 | 0.33 | 5.31 | 7193 | 17/17 |
| M4 cascade | 0.845 [0.38, 1.00] | 0.824 | 0.67 | 0.87 | 1.00 | 0.33 | 0.17 | 2.13 | 1091 | 5/17 |

*s/decision and tokens/decision are averaged over all test cases, so the cascade's cost includes the cases it settled without the LLM. Latency is wall-clock on the local GPU and includes model load for the first call.*

## Significance vs best baseline (B1 metadata/title, chosen on dev)

Correct = right label, and for UPDATE also the right target. Exact McNemar, Holm-adjusted across the comparisons. The macro-F1 difference uses a paired bootstrap.

| condition | only baseline right | only condition right | McNemar p | Holm p | ΔmacroF1 [95% CI] |
|---|---|---|---|---|---|
| M1 LLM zero-shot | 7 | 0 | 0.0156 | 0.0625 | -0.658 [-0.83, -0.17] |
| M2 LLM + candidates | 4 | 0 | 0.1250 | 0.3750 | -0.408 [-0.59, +0.08] |
| M3 LLM + few-shot | 4 | 1 | 0.3750 | 0.5000 | -0.509 [-0.73, -0.04] |
| M4 cascade | 3 | 0 | 0.2500 | 0.5000 | -0.108 [-0.29, +0.00] |

## Tuned parameters (from dev)

- B0 majority: label=NEW
- B1 metadata/title: t_title=0.7, t_body=0.95
- B2 cosine (e5): t_upd=0.898, t_red=0.9519
- B3 BM25: t_upd=0.2589, t_red=0.9638
- B4 reranker: t_upd=0.8815, t_red=0.9989
- M4 cascade: t_new=0.9121450195775757

## Macro-F1 by sampling pool (test)

| condition | pilot |
|---|---|
| B0 majority | 0.25 (n=17) |
| B1 metadata/title | 0.95 (n=17) |
| B2 cosine (e5) | 0.69 (n=17) |
| B3 BM25 | 0.95 (n=17) |
| B4 reranker | 0.77 (n=17) |
| M1 LLM zero-shot | 0.30 (n=17) |
| M2 LLM + candidates | 0.55 (n=17) |
| M3 LLM + few-shot | 0.44 (n=17) |
| M4 cascade | 0.85 (n=17) |

## Confusion matrices (rows = gold, cols = predicted: UPD / NEW / RED)

**B0 majority**
```
UPD     0    6    0
NEW     0   10    0
RED     0    1    0
```
**B1 metadata/title**
```
UPD     5    1    0
NEW     0   10    0
RED     0    0    1
```
**B2 cosine (e5)**
```
UPD     3    0    3
NEW     0   10    0
RED     0    0    1
```
**B3 BM25**
```
UPD     5    1    0
NEW     0   10    0
RED     0    0    1
```
**B4 reranker**
```
UPD     5    0    1
NEW     2    8    0
RED     0    0    1
```
**M1 LLM zero-shot**
```
UPD     1    5    0
NEW     2    8    0
RED     0    1    0
```
**M2 LLM + candidates**
```
UPD     4    2    0
NEW     0   10    0
RED     1    0    0
```
**M3 LLM + few-shot**
```
UPD     2    3    1
NEW     0   10    0
RED     0    1    0
```
**M4 cascade**
```
UPD     3    3    0
NEW     0   10    0
RED     0    0    1
```

## H1 error counts (test)

| condition | false-UPDATE (gold NEW) | missed-UPDATE | wrong target | of which flagged template_trap / reworded |
|---|---|---|---|---|
| B0 majority | 0 | 6 | 0 | 0 / 2 |
| B1 metadata/title | 0 | 1 | 1 | 0 / 1 |
| B2 cosine (e5) | 0 | 3 | 0 | 0 / 0 |
| B3 BM25 | 0 | 1 | 0 | 0 / 1 |
| B4 reranker | 2 | 1 | 2 | 0 / 1 |
| M1 LLM zero-shot | 2 | 5 | 1 | 0 / 1 |
| M2 LLM + candidates | 0 | 2 | 3 | 0 / 1 |
| M3 LLM + few-shot | 0 | 4 | 0 | 0 / 1 |
| M4 cascade | 0 | 3 | 2 | 0 / 2 |