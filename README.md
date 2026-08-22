# OIR Hub AI

AI layer for the Tunghai University **Office of International Relations (OIR)** hub.

When OIR receives a new document — an immigration notice, a dorm fee change, a campus
policy update — someone has to decide whether it *replaces* information already published
or is genuinely *new*. This project prototypes that decision:

1. **Chunk & embed** the existing knowledge base into a Qdrant vector collection.
2. **Compare** an incoming document against that collection by semantic similarity.
3. **Route** the result through a local LLM that returns a structured verdict:
   `UPDATE` (with the target file) or `NEW_BULLETIN`.

Current state: a single exploratory notebook, [document-similarity.ipynb](document-similarity.ipynb),
plus sample corpora under [mock-data/](mock-data/).

---

## How it works

| Stage | Component | Notes |
|---|---|---|
| Splitting | `RecursiveCharacterTextSplitter` | 800-char chunks, 200-char overlap, punctuation kept at chunk end |
| Embedding | `intfloat/multilingual-e5-base` | Multilingual (EN/中文); `passage:` / `query:` prefixes applied, vectors L2-normalized |
| Storage | Qdrant | Cosine distance; payload holds `source_file` + `content` |
| Retrieval | `client.query_points` | Top-k nearest chunks; the top score drives the routing decision |
| Routing | Ollama (`qwen3:4b`) via the OpenAI-compatible API | JSON-schema-constrained output validated by a Pydantic `RoutingDecision` model |

Chunk IDs are deterministic (`uuid5` over `"<filename>:<index>"`), so re-ingesting the same
file overwrites its points instead of duplicating them.

## Prerequisites

- **Python 3.12**
- **Qdrant** — Qdrant Cloud, or locally:
  ```bash
  docker run -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
  ```
- **Ollama** with the routing model pulled, serving on `http://localhost:11434`:
  ```bash
  ollama pull qwen3:4b
  ```
- A **Hugging Face token** (needed to download the embedding model).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install qdrant-client sentence-transformers langchain-text-splitters \
            python-dotenv openai pydantic jupyter
```

Copy the env template and fill in your values:

```bash
cp .env.example .env
```

| Variable | Purpose |
|---|---|
| `QDRANT_URL` | Qdrant endpoint (`http://localhost:6333` for a local container) |
| `QDRANT_API_KEY` | Qdrant Cloud API key; leave blank for an unsecured local instance |
| `HUGGINGFACE_TOKEN` | Token used to pull `intfloat/multilingual-e5-base` |

`.env` is git-ignored — keep real keys out of commits.

## Running

Open the notebook and run the cells top to bottom:

```bash
jupyter notebook document-similarity.ipynb
```

The notebook, in order:

1. Sets config constants (embedding model, chunk sizes, collection name `test_bulletins`).
2. Connects to Qdrant, loads the embedding model, builds the splitter.
3. **Drops and recreates the collection** — every run starts from a clean slate.
4. Ingests the files listed in `documents_to_upload`
   (`tunghai_academic_guide.txt`, `tunghai_life_guide.txt`).
5. Embeds a candidate document (`test_new_file`, default `mock-data/mock-arc.txt`) and
   queries for the closest chunks.
6. Sends the new text plus the best-matching chunk to the LLM and prints the parsed
   `RoutingDecision`.

To try a different candidate, change `test_new_file` and re-run from that cell down.

## Mock data

| File | Role |
|---|---|
| `tunghai_academic_guide.txt` | Baseline KB — ARC registration, credit transfer, academics |
| `tunghai_life_guide.txt` | Baseline KB — dorms, campus life, facilities |
| `oir_sample_knowledge_base.txt` | Larger combined OIR guide |
| `mock-arc.txt` | Incoming doc that **contradicts** the ARC section → expect `UPDATE` |
| `mock-dorm.txt` | Incoming doc that **revises dorm fees** → expect `UPDATE` |
| `mock-scooter.txt` | Incoming doc on an **uncovered topic** → expect `NEW_BULLETIN` |

The three `mock-*` files are the test cases: two should be recognized as updates to
existing content, one as a genuinely new bulletin.

## Notes & next steps

- Reranking (`BAAI/bge-reranker-v2-m3`) is stubbed out in the config but not wired up.
- The similarity threshold that separates `UPDATE` from `NEW_BULLETIN` is not yet fixed —
  the top score is printed so it can be calibrated against the mock cases.
- The notebook is a prototype; extracting the ingest/query/route steps into modules is the
  natural next move before this is called from the OIR hub backend.
