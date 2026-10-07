# OIR Hub AI

AI layer for the Tunghai University **Office of International Relations (OIR)** hub.

When OIR publishes a new announcement, someone has to decide whether it **replaces** an
earlier one (`UPDATE`), is about something not covered yet (`NEW`), or repeats what is
already posted (`REDUNDANT`). Getting this wrong means students read outdated deadlines.
This repo holds the router that makes that decision and the 專題 experiment that measures
how well it does.

## Layout

```
oir-hub-ai/
├── app/          the document router: upload a file, get UPDATE / NEW and the matching document
├── experiment/   the 專題 study: scraped OIR bulletins, labeled cases, baselines vs LLM, statistics
├── data/         sample OIR documents (PDF brochures) used by the app demo and the pilot set
├── docs/         design notes and the plain-language file guide
└── prototype/    the first exploratory notebook and its mock data (kept for reference)
```

| Folder | Start here |
|---|---|
| `experiment/` | [experiment/README.md](experiment/README.md), then [PLAN.md](experiment/PLAN.md) and [RUBRIC.md](experiment/RUBRIC.md) |
| `docs/` | [OIR-Experiment-File-Guide.pdf](docs/OIR-Experiment-File-Guide.pdf): what every experiment file does, step by step |
| | [Labeling-Guide.pdf](docs/Labeling-Guide.pdf): how to fill the answer page, with screenshots |
| `app/` | the section below |

## Prerequisites

- **Python 3.12.** The pinned PyTorch build has no wheels for newer Pythons (the machine default is 3.14).
- **Ollama** serving `qwen3:4b` on `http://localhost:11434`:
  ```bash
  ollama pull qwen3:4b
  ```
- **Qdrant**, for the app only (the experiment does not use it). Use Qdrant Cloud, or run it locally:
  ```bash
  docker run -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
  ```
- A **Hugging Face token**, to download the embedding model.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements-gpu.txt    # CUDA build of torch; skip on CPU-only machines
pip install -r requirements.txt
cp .env.example .env                   # then fill in the values below
```

| Variable | Purpose |
|---|---|
| `QDRANT_URL` | Qdrant endpoint (`http://localhost:6333` for a local container) |
| `QDRANT_API_KEY` | Qdrant Cloud API key; leave blank for an unsecured local instance |
| `HUGGINGFACE_TOKEN` | Token used to pull `intfloat/multilingual-e5-base` |

`.env` is git-ignored, so keep real keys out of commits.

## The app (`app/`)

| File | Role |
|---|---|
| `pipeline.py` | Ingest documents into Qdrant; route an incoming document (section-level match, `bge-reranker-v2-m3` scores, text-coverage rule) |
| `extract.py` | Plain text from `.pdf`, `.docx`, `.txt`/`.md` |
| `api.py` | FastAPI server: `POST /api/check`, `POST /api/ingest`, `GET /api/stats`, upload page at `/` |
| `run_demo.py` | Ingest the 2024 brochures from `data/`, then route the 2025 ones |
| `static/index.html` | Upload page |

```bash
.venv/Scripts/python app/run_demo.py                  # command-line demo
.venv/Scripts/uvicorn api:app --app-dir app --reload  # web app at http://localhost:8000
```

How routing works: each document is split into ~800-character sections and embedded with
`multilingual-e5-base`. Each section is matched against the knowledge base in Qdrant and
scored by the reranker. Sections already present word-for-word count as unchanged.
Chunk IDs are deterministic (`uuid5` over `"<filename>:<index>"`), so re-ingesting a file
overwrites it instead of duplicating it.

## The experiment (`experiment/`)

You give it a document (PDF, Word, .txt or .md). It finds the related OIR bulletins and decides
whether the document is an UPDATE of one of them, something NEW, or REDUNDANT, and can update
the library accordingly (`check_doc.py --apply`). The 專題 part measures how accurately this is
done: 200 test files (bulletin attachments, or bulletins written as .md) are compared with
the 1,164 scraped bulletins, and 5 rule baselines are compared with 4 LLM setups using significance
tests. Everything is in [experiment/README.md](experiment/README.md).

## Prototype (`prototype/`)

`document-similarity.ipynb` is the first notebook that explored the idea on the mock files in
`prototype/mock-data/` (`mock-arc.txt` and `mock-dorm.txt` should be UPDATEs,
`mock-scooter.txt` should be NEW). It is superseded by `app/` and kept only for reference.
