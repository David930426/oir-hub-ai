# Supersession experiment

Research design: [PLAN.md](PLAN.md). Labeling rules: [RUBRIC.md](RUBRIC.md).
Plain-language walkthrough of every file: [docs/OIR-Experiment-File-Guide.pdf](../docs/OIR-Experiment-File-Guide.pdf).

**The task:** you have a document (PDF, Word, .txt or .md). Compare it with every OIR bulletin
posted before it and decide whether it is an **UPDATE** of one of them (and which), something
**NEW**, or **REDUNDANT** (already posted). `check_doc.py` does this for your own files;
`run.py` measures how accurately it is done, on 200 labeled test files.

Run everything from the repo root with the project venv. Ollama must be serving `qwen3:4b`
for the LLM conditions. Qdrant is **not** needed.

```bash
# use it: your own file
.venv/Scripts/python experiment/check_doc.py "data/2026年日本國際交流基金會經費資助計畫.pdf"
.venv/Scripts/python experiment/check_doc.py my_notice.md --apply          # also update the library
.venv/Scripts/python experiment/check_doc.py --history                     # what --apply changed
.venv/Scripts/python experiment/check_doc.py --undo                        # take back the last one

# measure it (the 專題 experiment)
.venv/Scripts/python experiment/run.py experiment/data/labels_oir_DG.csv --no-llm   # baselines, ~1 min
.venv/Scripts/python experiment/run.py experiment/data/labels_oir_DG.csv            # all conditions, ~1 h
.venv/Scripts/python experiment/run.py kappa experiment/data/labels_oir_A.csv experiment/data/labels_oir_B.csv

# rebuild the data (already done; cached, so safe to re-run)
.venv/Scripts/python experiment/scrape_oir.py      # bulletins + their attachment links -> data/oir_news.jsonl
.venv/Scripts/python experiment/build_cases.py     # the 200 test cases -> data/cases_oir.jsonl + labeling page
.venv/Scripts/python experiment/fetch_files.py     # one input file per case -> data/files/, files_manifest.jsonl

# metric self-test
.venv/Scripts/python experiment/test_metrics.py
```

## The test files

Each of the 200 cases is one bulletin. Its **input file** is that bulletin's PDF/Word attachment
when OIR still hosts one (61 cases), otherwise the bulletin written out as a `.md` file (139 cases,
standing in for a notice you type). Attachments from before about 2022 return empty files from
OIR's server and cannot be recovered. The answer key is unchanged: an attachment relates to the
older bulletins exactly as its bulletin does. `results.md` reports scores separately for the
two file types.

## Updating the library (`check_doc.py --apply`)

The scraped bulletins are never edited. Changes go into `data/kb_changes.jsonl`:

| Decision | What `--apply` does |
|---|---|
| UPDATE | adds your file; marks the target bulletin "replaced by" it, so it is no longer offered as the current version (it stays in the history) |
| NEW | adds your file |
| REDUNDANT | nothing; the library already has it |

The AI's decision can be overridden with `--decision UPDATE --target oir-1100`. `--undo` reverses
the last `--apply`. The experiment (`run.py`) always uses the scraped bulletins only, so its
results never depend on what you added.

## Layout

```
experiment/
├── PLAN.md  RUBRIC.md  README.md
├── check_doc.py       your file -> related bulletins + decision; --apply updates the library
├── run.py             the experiment: split, tune on dev, score on test, write the report
├── scrape_oir.py      OIR news posts and attachment links -> data/oir_news.jsonl (1 req/s)
├── build_cases.py     pool sampling, candidate lists, labeling page
├── fetch_files.py     downloads each case's attachment, or writes the bulletin as .md
├── corpus.py          library, file reading, normalisation, e5 vectors, BM25        (helper)
├── methods.py         conditions B0-B4 and M1-M4, shared by run.py and check_doc.py   (helper)
├── metrics.py         macro-F1, target accuracy, McNemar, Holm, bootstrap, kappa       (helper)
├── test_metrics.py    hand-checked tests for metrics.py
├── labeling/
│   ├── annotate_template.html   the labeling tool (source)
│   └── annotate_oir.html        built from it by build_cases.py   (not committed)
├── data/
│   ├── oir_news.jsonl           1,295 scraped bulletins (1,164 dated)
│   ├── cases_oir.jsonl          the 200 test cases
│   ├── files_manifest.jsonl     which input file each case uses
│   ├── labels_oir_draft.csv     model-drafted answer key - review before use
│   ├── kb_changes.jsonl         what check_doc.py --apply changed (appears on first use)
│   ├── files/                   the 200 input files                     (not committed)
│   ├── raw/  cache/             scraped HTML, vectors, LLM answers       (not committed)
│   └── oir_news.csv             Excel copy of oir_news.jsonl             (not committed)
└── runs/                        one folder per run.py run                (not committed)
```

LLM answers are cached in `data/cache/llm_calls.jsonl` keyed by the exact prompt, so re-running
an analysis is free and keeps the originally measured latency. Delete that file to re-measure.
