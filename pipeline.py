"""Core logic: ingest documents into Qdrant, and route an incoming document."""

import difflib
import os
import re
import uuid
from collections import Counter

import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, PayloadSchemaType, QueryRequest,
)
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter

from extract import extract_text

load_dotenv()

EMBED_MODEL = "intfloat/multilingual-e5-base"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
COLLECTION = "oir_documents"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
# Most specific first: spaced sentence endings before bare punctuation, so we break on real
# sentence boundaries rather than inside "NT$1,000." Full-width forms for Chinese text.
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；",
              ". ", "! ", "? ", "; ", "!", "?", ";", " ", ""]

TOP_K = 5        # KB candidates recalled per section, before reranking
BATCH_SIZE = 32

# Raw cosine is not usable as a threshold: any two documents from the same domain score
# 0.85-0.95, and four editions of one bulletin score ~0.95 against each other. The
# cross-encoder returns a calibrated probability, so these thresholds mean something.
UPDATE_THRESHOLD = 0.50  # reranker prob: a section has a counterpart in the KB
IDENTICAL_MIN = 0.95     # share of the section's text already present in the KB

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
model = SentenceTransformer(
    EMBED_MODEL,
    device=DEVICE,
    token=os.getenv("HUGGINGFACE_TOKEN"),
    model_kwargs={"torch_dtype": torch.float16} if DEVICE == "cuda" else {},
)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=SEPARATORS,
    keep_separator="end",
)

_reranker = None


def get_reranker():
    """Loaded on first use - it is 2.2GB, and ingestion does not need it."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL, device=DEVICE, max_length=512)
    return _reranker


def strip_page_furniture(text, min_repeats=4, max_len=120):
    """Drop running headers/footers. A 12-page PDF repeats them on every page, and they
    otherwise become near-identical chunks that match each other instead of the content.

    max_len has to clear the longest banner: these bulletins carry a 71-character bilingual
    header on all 12 pages, and a tighter cap silently leaves it in every chunk."""
    lines = text.splitlines()
    norm = lambda l: re.sub(r"P\s*\d+\s*$", "", l.strip())
    counts = Counter(norm(l) for l in lines if l.strip())
    return "\n".join(
        l for l in lines
        if not (norm(l) and counts[norm(l)] >= min_repeats and len(norm(l)) <= max_len)
    )


def embed_passages(texts):
    return model.encode(
        [f"passage: {t}" for t in texts], normalize_embeddings=True, batch_size=BATCH_SIZE
    ).tolist()


def embed_query(text):
    return model.encode(f"query: {text}", normalize_embeddings=True).tolist()


def embed_queries(texts):
    return model.encode(
        [f"query: {t}" for t in texts], normalize_embeddings=True, batch_size=BATCH_SIZE
    ).tolist()


def ensure_collection(reset=False):
    exists = client.collection_exists(COLLECTION)
    if exists and reset:
        client.delete_collection(COLLECTION)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=model.get_sentence_embedding_dimension(),
                distance=Distance.COSINE,
            ),
        )
        # Keeps filtered lookups ("everything from this file") fast as the KB grows.
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name="source_file",
            field_schema=PayloadSchemaType.KEYWORD,
        )


def ingest_text(name, text):
    """Chunk, embed and store one document. Re-ingesting the same name overwrites it."""
    ensure_collection()
    chunks = splitter.split_text(strip_page_furniture(text))
    vectors = embed_passages(chunks)
    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{name}:{i}")),
            vector=vec,
            payload={"source_file": name, "content": chunk, "index": i},
        )
        for i, (chunk, vec) in enumerate(zip(chunks, vectors))
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    return len(points)


def ingest_file(path):
    """Ingest any format extract.py understands - .pdf and .docx included, not just text."""
    name = os.path.basename(path)
    with open(path, "rb") as f:
        return ingest_text(name, extract_text(name, f.read()))


def _normalise(s):
    """Ignore whitespace and the year label, so a pure 2024->2025 relabel is not a change."""
    return re.sub(r"\s+", "", re.sub(r"20\d{2}", "<Y>", s))


def _coverage(section, corpus):
    """How much of this section's text already exists in the knowledge base, 0-1.

    Compared against the whole corpus rather than one chunk on purpose: chunk boundaries
    shift between editions, so a chunk-to-chunk diff reports everything as changed even
    when the wording is identical.

    autojunk must stay off. SequenceMatcher's heuristic treats characters appearing in
    more than 1% of a long sequence as junk, which for CJK text throws the ratio away.
    """
    a, b = _normalise(section), _normalise(corpus)
    if not a:
        return 1.0
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks()) / len(a)


def route_document(text, limit=3):
    """Compare an incoming document against the knowledge base and decide what to do.

    Matching is per section, not per document. Whole-document embedding averages away the
    one clause that changed, and truncates anything past the encoder's 512-token window -
    which matters because successive editions of a bulletin are ~95% identical.
    """
    ensure_collection()
    sections = splitter.split_text(strip_page_furniture(text))

    # One Qdrant round-trip for every section, rather than a query per section.
    results = client.query_batch_points(
        collection_name=COLLECTION,
        requests=[
            QueryRequest(query=v, limit=TOP_K, with_payload=True)
            for v in embed_queries(sections)
        ],
    )

    pairs, flat = [], []
    for i, res in enumerate(results):
        for hit in res.points:
            pairs.append((sections[i], hit.payload["content"]))
            flat.append((i, hit))

    if not pairs:
        return {
            "decision": "NEW_BULLETIN", "target_file": None, "top_score": 0.0,
            "threshold": UPDATE_THRESHOLD, "matches": [],
            "sections": {"total": len(sections), "unchanged": 0, "revised": 0,
                         "new": len(sections)},
        }

    scores = get_reranker().predict(
        pairs, activation_fn=torch.nn.Sigmoid(), batch_size=BATCH_SIZE
    )

    # Best knowledge-base counterpart per section.
    best = {}
    for (i, hit), score in zip(flat, scores):
        if i not in best or float(score) > best[i]["score"]:
            best[i] = {"hit": hit, "score": float(score)}

    # Every distinct knowledge-base chunk that surfaced, as one corpus to diff against.
    seen, corpus = set(), []
    for _, hit in flat:
        if hit.id not in seen:
            seen.add(hit.id)
            corpus.append(hit.payload["content"])
    corpus = "\n".join(corpus)

    # Coverage is checked before the reranker score, not after. The reranker judges
    # relevance, not near-duplication, and scores some boilerplate low even when it is
    # present in the knowledge base word for word - gating on it first files text that
    # plainly already exists under "new".
    counts = Counter()
    revisions = []
    for i, section in enumerate(sections):
        b = best.get(i)
        coverage = _coverage(section, corpus)
        if coverage >= IDENTICAL_MIN:
            counts["unchanged"] += 1
        elif b is not None and b["score"] >= UPDATE_THRESHOLD:
            counts["revised"] += 1
            revisions.append((b["score"], i, b["hit"], coverage))
        else:
            counts["new"] += 1

    ranked = sorted(best.values(), key=lambda b: b["score"], reverse=True)[:limit]
    matches = [
        {
            "source_file": b["hit"].payload["source_file"],
            "score": round(b["score"], 4),          # reranker probability, not cosine
            "vector_score": round(b["hit"].score, 4),
            "excerpt": b["hit"].payload["content"][:300],
        }
        for b in ranked
    ]

    # A document is an update when some section changed; unchanged sections prove nothing
    # new, and a document whose sections have no counterpart at all is a new bulletin.
    if counts["revised"]:
        decision = "UPDATE"
        target = max(revisions)[2].payload["source_file"]
    elif counts["unchanged"] and not counts["new"]:
        decision = "NO_CHANGE"
        target = matches[0]["source_file"] if matches else None
    else:
        decision = "NEW_BULLETIN"
        target = None

    return {
        "decision": decision,
        "target_file": target,
        "top_score": matches[0]["score"] if matches else 0.0,
        "threshold": UPDATE_THRESHOLD,
        "matches": matches,
        "sections": {
            "total": len(sections),
            "unchanged": counts["unchanged"],
            "revised": counts["revised"],
            "new": counts["new"],
        },
    }