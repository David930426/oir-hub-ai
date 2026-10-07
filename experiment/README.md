# Supersession experiment

Research design: [PLAN.md](PLAN.md). Labeling rules: [RUBRIC.md](RUBRIC.md).
Plain-language walkthrough of every file: [docs/OIR-Experiment-File-Guide.pdf](../docs/OIR-Experiment-File-Guide.pdf).

Run everything from the repo root with the project venv. Ollama must be serving `qwen3:4b`
for the LLM conditions. Qdrant is **not** needed: the experiment keeps its corpus in memory,
so it stays reproducible.

```bash
# 1. data (already done; cached, so re-running is safe and fast)
.venv/Scripts/python experiment/scrape_oir.py            # -> data/oir_news.jsonl
.venv/Scripts/python experiment/build_cases.py oir       # -> data/cases_oir.jsonl + labeling/annotate_oir.html

# 2. label: open experiment/labeling/annotate_oir.html, Import CSV (data/labels_oir_draft.csv),
#    fix the answers, then Export CSV as experiment/data/labels_oir_<you>.csv

# 3. annotator agreement on the 40-case subset
.venv/Scripts/python experiment/run.py kappa experiment/data/labels_oir_A.csv experiment/data/labels_oir_B.csv

# 4. the experiment
.venv/Scripts/python experiment/run.py oir experiment/data/labels_oir_DG.csv --no-llm   # baselines, seconds
.venv/Scripts/python experiment/run.py oir experiment/data/labels_oir_DG.csv            # all conditions, ~1 h

# 5. check your own document against every bulletin
.venv/Scripts/python experiment/check_doc.py "data/2026年日本國際交流基金會經費資助計畫.pdf"

# metric self-test
.venv/Scripts/python experiment/test_metrics.py
```

Each run writes `runs/<corpus>_<time>/results.md` (every table for the report),
`predictions.csv`, and `errors.csv` (every wrong answer with the model's rationale, for the
error analysis).

## Layout

```
experiment/
├── PLAN.md  RUBRIC.md  README.md
├── scrape_oir.py      public OIR news posts -> data/oir_news.jsonl (1 req/s)
├── corpus.py          loading, title/body normalisation, e5 embeddings, BM25   (helper)
├── build_cases.py     pool sampling, candidate lists, labeling page
├── methods.py         conditions B0-B4 and M1-M4                               (helper)
├── metrics.py         macro-F1, target accuracy, McNemar, Holm, bootstrap, kappa (helper)
├── test_metrics.py    hand-checked tests for metrics.py
├── run.py             split, tune on dev, score on test, write the report
├── check_doc.py       your own PDF/txt/md -> related bulletins + UPDATE/NEW/REDUNDANT
├── labeling/
│   ├── annotate_template.html   the labeling tool (source)
│   └── annotate_oir.html        built from it by build_cases.py   (not committed)
├── data/
│   ├── oir_news.jsonl           1,295 scraped bulletins
│   ├── cases_oir.jsonl          the 200 test questions
│   ├── cases_local.jsonl        17 pilot questions from the PDFs in ../data/
│   ├── labels_oir_draft.csv     model-drafted answer key - review before use
│   ├── raw/  cache/             scraped HTML, embeddings, LLM answers   (not committed)
│   └── oir_news.csv             Excel copy of oir_news.jsonl            (not committed)
└── runs/                        one folder per run                      (not committed)
```

Everything marked "not committed" can be regenerated: `raw/` by `scrape_oir.py`, the
labeling pages by `build_cases.py`, `cache/` and `runs/` by `run.py`. LLM answers are cached
in `data/cache/llm_calls.jsonl` keyed by the exact prompt, so re-running an analysis is free
and keeps the originally measured latency. Delete that file to re-measure.
