"""Core logic: ingest documents into Qdrant, and route an incoming document."""

import os
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

EMBED_MODEL = "intfloat/multilingual-e5-base"
COLLECTION = "oir_documents"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
SEPARATORS = ["\n\n", "\n", "。", "!", "?", ";", ". ", "! ", "? ", "; ", " ", ""]

# Above this score the incoming document is considered a revision of an existing one.
UPDATE_THRESHOLD = 0.85

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
model = SentenceTransformer(EMBED_MODEL)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=SEPARATORS,
    keep_separator="end",
)


def embed_passages(texts):
    return model.encode([f"passage: {t}" for t in texts], normalize_embeddings=True).tolist()


def embed_query(text):
    return model.encode(f"query: {text}", normalize_embeddings=True).tolist()


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


def ingest_text(name, text):
    """Chunk, embed and store one document. Re-ingesting the same name overwrites it."""
    ensure_collection()
    chunks = splitter.split_text(text)
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
    with open(path, encoding="utf-8") as f:
        return ingest_text(os.path.basename(path), f.read())


def route_document(text, limit=3):
    """Compare an incoming document against the knowledge base and decide what to do."""
    ensure_collection()
    hits = client.query_points(
        collection_name=COLLECTION, query=embed_query(text), limit=limit
    ).points

    matches = [
        {
            "source_file": h.payload["source_file"],
            "score": round(h.score, 4),
            "excerpt": h.payload["content"][:300],
        }
        for h in hits
    ]

    if matches and matches[0]["score"] >= UPDATE_THRESHOLD:
        decision, target = "UPDATE", matches[0]["source_file"]
    else:
        decision, target = "NEW_BULLETIN", None

    return {
        "decision": decision,
        "target_file": target,
        "top_score": matches[0]["score"] if matches else 0.0,
        "threshold": UPDATE_THRESHOLD,
        "matches": matches,
    }