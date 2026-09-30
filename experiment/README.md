# Supersession experiment

Research design: [PLAN.md](PLAN.md). Labeling rules: [RUBRIC.md](RUBRIC.md).

Run everything from the repo root with the project venv. Ollama must be serving `qwen3:4b`.
Qdrant is **not** needed: the experiment keeps its small corpus in memory, so it stays
reproducible.

```bash
# 1. data (once; cached, so safe to re-run)
.venv/Scripts/python experiment/scrape_oir.py            # -> experiment/data/oir_news.jsonl
.venv/Scripts/python experiment/build_cases.py oir       # -> data/cases_oir.jsonl + annotate_oir.html

# 2. label: open experiment/annotate_oir.html in a browser, then "Export CSV" into experiment/data/

# 3. check annotator agreement on the 40-case subset
.venv/Scripts/python experiment/run.py kappa experiment/data/labels_oir_A.csv experiment/data/labels_oir_B.csv

# 4. run
.venv/Scripts/python experiment/run.py oir experiment/data/labels_oir_A.csv --no-llm   # baselines, seconds
.venv/Scripts/python experiment/run.py oir experiment/data/labels_oir_A.csv            # all conditions

# metric code self-test
cd experiment && ../.venv/Scripts/python test_metrics.py
```

Each run writes `runs/<corpus>_<time>/results.md` (every table for the report),
`predictions.csv`, and `errors.csv` (every false-UPDATE and missed-UPDATE, with the model's
rationale, for the error analysis).

| File | What it does |
|---|---|
| `scrape_oir.py` | public OIR news posts → jsonl (1 req/s, HTML cached in `data/raw/`) |
| `corpus.py` | loading, title/body normalisation, e5 embeddings, BM25 |
| `build_cases.py` | pool sampling, candidate lists, labeling page |
| `annotate_template.html` | the labeling tool |
| `methods.py` | B0–B4, M1–M4 |
| `metrics.py` | macro-F1, target accuracy, McNemar, Holm, bootstrap, κ |
| `run.py` | split, tune on dev, score on test, write the report |

LLM calls are cached in `data/cache/llm_calls.jsonl`, keyed by the exact prompt. Re-running an
analysis is free, and the originally measured latency is kept. Delete the file to re-measure.
