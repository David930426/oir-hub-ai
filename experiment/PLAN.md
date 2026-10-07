# 專題 plan — ingestion-time supersession detection

Revised 2026-09-29 (data and annotation) and 2026-10-07 (inputs are files). The research
question, hypotheses, conditions and statistics are unchanged.

## Question

When a new OIR document arrives - a PDF, Word, .txt or .md file - can a small local LLM judge (Qwen3-4B, Q4_K_M) decide
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
| **Library** | Public news posts at `oir.thu.edu.tw`, scraped by `scrape_oir.py` (Sn 1–1350, 2013–2026) | 1,295 posts, 1,164 dated | what each input is compared against |
| **Inputs** | One file per case (`fetch_files.py`): the bulletin's PDF/Word attachment, or the bulletin as `.md` if it has none | 200 cases: 61 attachments, 139 `.md`; ≥150 labeled | dev / test |

**Why the public feed works:** OIR re-posts recurring programs every cycle (exchange selection
rounds, Japan Foundation grants, JASSO, UMAP, partner summer schools). The same feed therefore
contains real UPDATEs, real re-posts, and many same-template-but-different-program traps.

**Case = chronological replay.** The input is one file, dated like the bulletin it belongs to.
It is compared with every bulletin dated strictly before it, which is exactly what the system
would have seen at the time. The answer key is per case, so it holds for the file as for the
bulletin: an attachment relates to the older bulletins exactly as its bulletin does.

**Why files.** In use, staff hand the system a document (a brochure, a typed notice), not a
finished bulletin. Attachments are the realistic input; `.md` files stand in for typed notices.
OIR no longer hosts attachments from before about 2022 (the server returns empty files), so
older cases use the `.md`. Results are reported for both input types separately.

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
- **Annotator 1 (you):** all cases, in `labeling/annotate_oir.html`. It is offline, uses u/n/r and
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
| M1 | LLM, input file only | no target possible |
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
- Only 61 of 200 inputs are real attachments; the rest are bulletins written as `.md`, which
  are easier to match (same wording as the post). Compare the two input types, not just the total.
- Labels were made by reading the bulletins; for an attachment that covers more or less than its
  bulletin, the label may fit the file less well.
- Q4_K_M on 8 GB may understate what Qwen3-4B can do.
- Pilot observations (17 brochure PDFs, not results; the pilot set has since been retired): an Optional `target` field let the model answer
  UPDATE without naming a document. Placeholder exemplar rationales made few-shot collapse.
  Both are fixed. They are worth a sentence in the method section, as design lessons.
