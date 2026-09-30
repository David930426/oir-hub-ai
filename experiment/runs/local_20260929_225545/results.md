# Results — local  (2026-09-29 22:55)

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
| M3 LLM + few-shot | 0.648 [0.35, 0.88] | 0.706 | 0.67 | 0.78 | 0.50 | 1.00 | 0.67 | 5.90 | 7779 | 17/17 |

*s/decision and tokens/decision are averaged over all test cases, so the cascade's cost includes the cases it settled without the LLM. Latency is wall-clock on the local GPU and includes model load for the first call.*

## Tuned parameters (from dev)


## Macro-F1 by sampling pool (test)

| condition | pilot |
|---|---|
| M3 LLM + few-shot | 0.65 (n=17) |

## Confusion matrices (rows = gold, cols = predicted: UPD / NEW / RED)

**M3 LLM + few-shot**
```
UPD     4    1    1
NEW     2    7    1
RED     0    0    1
```

## H1 error counts (test)

| condition | false-UPDATE (gold NEW) | missed-UPDATE | wrong target | of which flagged template_trap / reworded |
|---|---|---|---|---|
| M3 LLM + few-shot | 2 | 2 | 0 | 0 / 1 |