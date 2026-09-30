# 專題 plan — ingestion-time supersession detection

Revised 2026-09-29. It replaces the data and annotation parts of the earlier plan. The research
question, hypotheses, conditions and statistics are unchanged.

## Question

When a new OIR announcement arrives, can a small local LLM judge (Qwen3-4B, Q4_K_M) decide
**UPDATE / NEW / REDUNDANT**, and name the document being replaced, more accurately than
metadata, cosine, BM25 and reranker thresholds? Does a cascade keep the LLM's accuracy at a
fraction of its cost?

- **H1:** similarity baselines make two systematic errors. They call template-similar but
  different programs UPDATE (*false-UPDATE*), and they miss reworded successors (*missed-UPDATE*).
- **H2:** M2 (LLM + candidates) beats the best baseline on macro-F1, but by less than intuition
  suggests. A realistic ceiling for a 4B quantized model on this task is ~0.60–0.75.
- **H3:** M4 (cascade) keeps most of M2's accuracy while settling a large share of cases
  without an LLM call.

A null result (baselines ≈ LLM) is a valid finding: it says metadata + recency is enough here.

## Data — our own, no office-provided material

| Set | Source | Size | Role |
|---|---|---|---|
| **Main** | Public news posts at `oir.thu.edu.tw`, scraped by `scrape_oir.py` (Sn 1–1350, 2013–2026) | 1,295 posts → **200 sampled cases, ≥150 labeled** | dev / test |
| **Pilot** | The 18 brochure PDFs in `data/` | 17 cases | code and prompt debugging only; never reported |

**Why the public feed works:** OIR re-posts recurring programs every cycle (exchange selection
rounds, Japan Foundation grants, JASSO, UMAP, partner summer schools). The same feed therefore
contains real UPDATEs, real re-posts, and many same-template-but-different-program traps.

**Case = chronological replay.** The incoming document is one post. Its knowledge base is every
post dated strictly before it, which is exactly what the pipeline would have seen at the time.

**Sampling** (`build_cases.py`). About 9 in 10 posts are NEW, so a purely random sample would
contain almost no UPDATEs. Cases are drawn from three pools:
- 90 **title**: an older post has a near-identical normalised title
- 39 **semantic**: no title match, but the nearest older post is in the top 20 % of e5
  similarity (cos ≥ 0.940). Only 39 posts qualify; most near-duplicates also share a title.
- 71 **random**: uniform over posts from 2015 on; this pool absorbs the semantic shortfall

Cases are shuffled so the annotator never sees the pool. Results are reported overall and per
pool. The pools are built from the same kinds of signals the baselines use, which is a threat
to validity. It is stated in the report and mitigated by the random pool.

## Labels

- **Rubric:** `RUBRIC.md`, frozen before labeling starts. It includes the edge cases: yearly
  edition = UPDATE, translation = REDUNDANT, template trap = NEW, extension notice = UPDATE, and
  results list = NEW.
- **Annotator 1 (you):** all cases, in `annotate_oir.html`. It is offline, uses u/n/r and
  1–9 shortcuts, autosaves, and exports to CSV. Budget ≈ 1.5–2 min per case, so about 5 hours
  for 150 cases.
- **Annotator 2 (a classmate who did not build the system):** the fixed 40-case κ subset
  ("κ subset only" toggle), using only the rubric.
- **Gate:** Cohen's κ ≥ 0.6 → proceed. Below that → revise the rubric, bump its version, and
  re-label.

## Conditions (all implemented in `methods.py`)

| | Condition | Notes |
|---|---|---|
| B0 | majority class | |
| B1 | metadata: normalised-title match + body identity | the "deterministic camp" |
| B2 | e5 cosine, two thresholds | |
| B3 | BM25, two thresholds | character bigrams, so no segmenter; avoids jieba's zh-TW weakness |
| B4 | bge-reranker-v2-m3 over the cosine top-10, two thresholds | |
| M1 | LLM, incoming document only | no target possible |
| M2 | LLM + top-5 cosine candidates | the core LLM condition |
| M3 | M2 + one dev exemplar per class | exemplar reasons come from the annotator's notes |
| M4 | cascade: identical body → REDUNDANT, cosine below a dev-tuned cut → NEW, otherwise M2 | rules kept only if ≥ 0.9 precise on dev |

Every threshold, and every exemplar, is chosen on **dev only** (stratified 40 %, seed 13).

The deployed router in `pipeline.py` is not the M2 LLM judge. It is a reranker plus section
coverage rule, so it is a deterministic method. Adding it as **B5** is a cheap extension if
time allows, and it would give the "what we ship today" row.

## Metrics and tests (`metrics.py`, unit-tested in `test_metrics.py`)

- Macro-F1 (primary), per-class P/R/F1, confusion matrix, accuracy
- Target accuracy for gold UPDATE: conditional (given a correct UPDATE) and joint
- Retrieval ceiling: is the gold target in the e5 top-1/5/10? M2 cannot name a document it
  never sees.
- Cost: seconds, tokens and LLM calls per decision
- Exact McNemar test of each LLM condition vs the best baseline (chosen on dev), Holm-adjusted
- Paired-bootstrap 95 % CI on the macro-F1 difference, plus a bootstrap CI on each macro-F1
- Cohen's κ between annotators

## Timeline (from today)

| When | Work | Done when |
|---|---|---|
| **Tue 29 Sep – Fri 2 Oct** | Scrape finishes → `build_cases.py oir` → label 150 cases → send the κ subset to a classmate | `labels_oir_<you>.csv` has ≥ 150 rows |
| **Sat 3 – Tue 6 Oct** | `run.py --no-llm` (baselines, seconds), then the full run (≈ 1 h of GPU time) → **MVP table** | B0–B4 vs M2 with CI and McNemar |
| **Wed 7 – Tue 13 Oct** | κ; M3/M4; error analysis from `errors.csv` (read every false-/missed-UPDATE) | error taxonomy with examples |
| **Wed 14 – Fri 23 Oct** | write-up and slides; optional B5 and a Q4-vs-Q8 check | report |

**Cut order if short on time:** B5 → Q8 check → M3 → B3/B4 → n = 150 down to 100.

## Honest limits to state in the report

- The sample is small (≈ 90 test cases), so CIs are wide. It is a single office in a single domain.
- Pool-based sampling favours cases where the signals are informative. The random pool is the
  unbiased slice.
- Most cases are labeled by one annotator. κ covers 40 cases.
- Post text only: attachments (PDF brochures) are not read. Some cases can only be judged from
  the attachment; the annotator flags them `unsure`.
- Q4_K_M on 8 GB may understate what Qwen3-4B can do.
- Pilot observations (17 cases, not results): an Optional `target` field let the model answer
  UPDATE without naming a document. Placeholder exemplar rationales made few-shot collapse.
  Both are fixed. They are worth a sentence in the method section, as design lessons.
