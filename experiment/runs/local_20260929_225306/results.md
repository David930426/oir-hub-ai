# Results — local  (2026-09-29 22:53)

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
| B1 metadata/title | 0.954 [0.56, 1.00] | 0.941 | 0.91 | 0.95 | 1.00 | 0.80 | 0.67 | ≈0 | 0 | – |
| M2 LLM + candidates | 0.426 [0.25, 0.58] | 0.706 | 0.44 | 0.83 | 0.00 | 1.00 | 0.33 | 7.33 | 4063 | 17/17 |
| M3 LLM + few-shot | 0.207 [0.06, 0.34] | 0.235 | 0.40 | 0.00 | 0.22 | 1.00 | 0.50 | 5.35 | 14153 | 17/17 |
| M4 cascade | 0.695 [0.26, 0.86] | 0.706 | 0.29 | 0.80 | 1.00 | 1.00 | 0.17 | 2.06 | 1089 | 5/17 |

*s/decision and tokens/decision are averaged over all test cases, so the cascade's cost includes the cases it settled without the LLM. Latency is wall-clock on the local GPU and includes model load for the first call.*

## Significance vs best baseline (B1 metadata/title, chosen on dev)

Correct = right label, and for UPDATE also the right target. Exact McNemar, Holm-adjusted across the comparisons. The macro-F1 difference uses a paired bootstrap.

| condition | only baseline right | only condition right | McNemar p | Holm p | ΔmacroF1 [95% CI] |
|---|---|---|---|---|---|
| M2 LLM + candidates | 4 | 1 | 0.3750 | 0.5000 | -0.528 [-0.74, -0.04] |
| M3 LLM + few-shot | 13 | 2 | 0.0074 | 0.0222 | -0.746 [-0.89, -0.38] |
| M4 cascade | 3 | 0 | 0.2500 | 0.5000 | -0.259 [-0.42, -0.08] |

## Tuned parameters (from dev)

- B1 metadata/title: t_title=0.7, t_body=0.95
- M4 cascade: t_new=0.9121450195775757

## Macro-F1 by sampling pool (test)

| condition | pilot |
|---|---|
| B1 metadata/title | 0.95 (n=17) |
| M2 LLM + candidates | 0.43 (n=17) |
| M3 LLM + few-shot | 0.21 (n=17) |
| M4 cascade | 0.70 (n=17) |

## Confusion matrices (rows = gold, cols = predicted: UPD / NEW / RED)

**B1 metadata/title**
```
UPD     5    1    0
NEW     0   10    0
RED     0    0    1
```
**M2 LLM + candidates**
```
UPD     2    4    0
NEW     0   10    0
RED     1    0    0
```
**M3 LLM + few-shot**
```
UPD     3    0    3
NEW     6    0    4
RED     0    0    1
```
**M4 cascade**
```
UPD     1    5    0
NEW     0   10    0
RED     0    0    1
```

## H1 error counts (test)

| condition | false-UPDATE (gold NEW) | missed-UPDATE | wrong target | of which flagged template_trap / reworded |
|---|---|---|---|---|
| B1 metadata/title | 0 | 1 | 1 | 0 / 1 |
| M2 LLM + candidates | 0 | 4 | 0 | 0 / 1 |
| M3 LLM + few-shot | 6 | 3 | 0 | 0 / 0 |
| M4 cascade | 0 | 5 | 0 | 0 / 2 |