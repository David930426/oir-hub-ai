"""The bulletin library, reading input files, and the shared retrieval signals.

Two things meet here:
- the LIBRARY: every OIR bulletin scraped from oir.thu.edu.tw (scrape_oir.py), plus any file
  you added with `check_doc.py --apply`. This is what an incoming document is compared to.
- an INPUT FILE: a PDF, Word, .txt or .md document. In the experiment the input files are the
  bulletins' own attachments (fetch_files.py); in daily use it is whatever you pass to check_doc.

An input file dated D is compared only with library documents dated strictly before D, which is
what the system would actually have known on that day.
"""

import hashlib
import json
import math
import pickle
import re
import zipfile
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent  # core/ -> experiment/ -> repo root
EXP = ROOT / "experiment"
DATA = EXP / "data"
CACHE = DATA / "cache"
FILES = DATA / "files"                      # downloaded test inputs (fetch_files.py)
MANIFEST = DATA / "files_manifest.jsonl"    # which input file belongs to which case
KB_LOG = DATA / "kb_changes.jsonl"          # what check_doc.py --apply added or retired


# ---------------------------------------------------------------- the bulletins

@lru_cache(maxsize=None)
def bulletins():
    """Scraped bulletins sorted by (date, doc_id). Undated posts (0000-00-00) are static pages -
    school profiles, old program pages; with no date they would sort before everything and sit
    in every knowledge base, so they go."""
    docs = [json.loads(l) for l in (DATA / "oir_news.jsonl").open(encoding="utf-8")]
    docs = [d for d in docs if (d["title"] or d["text"]) and not d["date"].startswith("0000")]
    return sorted(docs, key=lambda d: (d["date"], d["doc_id"]))


def doc_text(doc, limit=None):
    s = f"{doc['title']}\n{doc['text']}"
    return s[:limit] if limit else s


# ---------------------------------------------------------------- reading input files

def read_file(path):
    """(title, text) of a PDF, .docx, .odt, .txt or .md file.

    The title is what a person would call the document: the first line of a .txt/.md (a
    leading "# " is dropped), otherwise the file name without its extension."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        import pymupdf
        with pymupdf.open(path) as d:
            text = "\n".join(p.get_text() for p in d)
    elif ext in (".docx", ".odt"):
        # Both are zip files of XML; reading the XML directly needs no extra library.
        member, para = ("word/document.xml", r"</w:p>") if ext == ".docx" else ("content.xml", r"</text:(?:p|h)>")
        xml = zipfile.ZipFile(path).read(member).decode("utf-8")
        text = re.sub(r"<[^>]+>", "", re.sub(para, "\n", xml))
    elif ext in (".txt", ".md"):
        text = path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError(f"cannot read {path.name}: use PDF, .docx, .odt, .txt or .md")
    text = re.sub(r"[ \t]+", " ", text).strip()
    if ext in (".txt", ".md") and text:
        lines = text.splitlines()
        k = next(i for i, l in enumerate(lines) if l.strip())
        title = re.sub(r"^#+\s*", "", lines[k]).strip()
        # The title line is the title, not body text: leaving it in made an exact re-post
        # look different from the bulletin it repeats.
        text = "\n".join(lines[k + 1:]).strip()
    else:
        title = path.stem.strip()
    return title, text


# ---------------------------------------------------------------- normalisation

_TITLE_NOISE = [
    r"【[^】]*】", r"\[[^\]]*\]", r"^[^║|｜]{0,8}[║|｜]",          # 【短期研習】, 公告║
    r"(19|20)\d{2}\s*(年度?|學年度?)?", r"\b1[0-2]\d\s*(年度?|學年度?)",  # 2025年, 115年度
    r"[春夏秋冬](季|學期|季班)?", r"第?[一二三四1-4]期", r"\d+(st|nd|rd|th)?",
    r"(spring|summer|fall|autumn|winter)", r"(更正|延長|延期|截止|最新|重要|公告|活動|徵求|招募|開放|報名)",
    r"[\s\W_]+",
]


def norm_title(title):
    """Title with the cycle stripped out, so "2025年度日本國際交流基金會…" and
    "2026年度日本國際交流基金會…" compare equal. This is the metadata heuristic's key."""
    t = title.lower()
    for pat in _TITLE_NOISE:
        t = re.sub(pat, "", t, flags=re.I)
    return t


def norm_body(text):
    """Body with whitespace and year labels removed, to spot a pure re-post."""
    t = re.sub(r"(19|20)\d{2}|1[0-2]\d(?=\s*年)", "<Y>", text)
    return re.sub(r"\s+", "", t)


# ---------------------------------------------------------------- embeddings (e5)

EMBED_MODEL = "intfloat/multilingual-e5-base"
EMBED_CHARS = 1500  # ~512 e5 tokens for mixed zh/en; the encoder truncates past that anyway


def _encode(texts, prefix):
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBED_MODEL, device=dev)
    out = model.encode([f"{prefix}: {t[:EMBED_CHARS]}" for t in texts], normalize_embeddings=True, batch_size=32)
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()  # hand the VRAM back before Ollama needs it
    return out


def bulletin_vectors():
    """Passage vectors for every bulletin, cached on disk by content hash."""
    docs = bulletins()
    key = hashlib.md5("".join(d["doc_id"] + doc_text(d, EMBED_CHARS) for d in docs).encode()).hexdigest()[:12]
    path = CACHE / f"e5_passages_{key}.npy"
    if path.exists():
        return np.load(path)
    vecs = _encode([doc_text(d) for d in docs], "passage")
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, vecs)
    return vecs


def query_vectors(texts):
    """Query vectors for input documents, cached by text so a re-run does not reload the model."""
    path = CACHE / "e5_queries.pkl"
    cache = pickle.loads(path.read_bytes()) if path.exists() else {}
    keys = [hashlib.md5(t[:EMBED_CHARS].encode()).hexdigest() for t in texts]
    todo = sorted({k: t for k, t in zip(keys, texts) if k not in cache}.items())
    if todo:
        for (k, _), v in zip(todo, _encode([t for _, t in todo], "query")):
            cache[k] = v
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(cache))
    return np.stack([cache[k] for k in keys])


# ---------------------------------------------------------------- BM25

def tokens(text):
    """Segmenter-free: Chinese as overlapping character bigrams, everything else as lowercase
    words. Bigrams are the standard CJK IR baseline and sidestep the jieba-vs-CKIP question
    (jieba's dictionary is Simplified-Chinese and measurably worse on zh-TW)."""
    out = re.findall(r"[a-z0-9]+", text.lower())
    for run in re.findall(r"[㐀-鿿]+", text):
        out.extend(run[j:j + 2] for j in range(max(1, len(run) - 1)))
    return out


class BM25:
    """BM25 over a fixed set of documents, as a sparse matrix so one query is scored against
    every document in a single product. Each distinct query term counts once."""

    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        from scipy.sparse import csr_matrix
        self.k1, self.b = k1, b
        tfs = [Counter(t) for t in docs_tokens]
        lens = np.array([len(t) for t in docs_tokens], dtype=float)
        self.avg = lens.mean() if len(lens) else 1.0
        df = Counter(w for tf in tfs for w in tf)
        n = len(docs_tokens)
        self.vocab = {w: i for i, w in enumerate(df)}
        self.idf = np.array([math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)) for w in df])
        rows, cols, vals = [], [], []
        for d, tf in enumerate(tfs):
            norm = k1 * (1 - b + b * lens[d] / self.avg)
            for w, c in tf.items():
                v = self.vocab[w]
                rows.append(d); cols.append(v); vals.append(self.idf[v] * c * (k1 + 1) / (c + norm))
        self.W = csr_matrix((vals, (rows, cols)), shape=(n, len(df)))

    def scores(self, q_tokens):
        """BM25 of the query against every document, divided by the query's score against itself
        so the value is roughly 0-1 and one threshold works across short and long inputs."""
        q = Counter(q_tokens)
        hits = [self.vocab[w] for w in q if w in self.vocab]
        if not hits:
            return np.zeros(self.W.shape[0])
        x = np.zeros(len(self.vocab)); x[hits] = 1.0
        norm = self.k1 * (1 - self.b + self.b * len(q_tokens) / self.avg)
        self_score = sum(self.idf[self.vocab[w]] * c * (self.k1 + 1) / (c + norm) for w, c in q.items() if w in self.vocab)
        return (self.W @ x) / (self_score or 1.0)


# ---------------------------------------------------------------- the library

class Library:
    """The documents an input is compared against, with their vectors and BM25 index.

    `Library.load()` is the scraped bulletins only - what the experiment uses, so the result
    does not depend on anything you added. `Library.load(with_changes=True)` also applies
    kb_changes.jsonl: files you added become documents, and bulletins they replaced are marked
    `replaced_by` so they are no longer offered as the current version."""

    def __init__(self, docs, vectors):
        self.docs, self.vectors = docs, vectors
        self.index = {d["doc_id"]: i for i, d in enumerate(docs)}
        self.bm25 = BM25([tokens(doc_text(d, 4000)) for d in docs])

    @classmethod
    def load(cls, with_changes=False):
        docs = [dict(d) for d in bulletins()]
        vecs = bulletin_vectors()
        if with_changes and KB_LOG.exists():
            added, replaced = [], {}
            for line in KB_LOG.open(encoding="utf-8"):
                ev = json.loads(line)
                if ev["action"] == "add":
                    added.append(ev)
                elif ev["action"] == "replace":
                    replaced[ev["target"]] = ev["by"]
                elif ev["action"] == "undo":
                    added = [a for a in added if a["batch"] != ev["batch"]]
                    replaced = {t: b for t, b in replaced.items() if b not in ev["doc_ids"]}
            docs += [a["doc"] for a in added]
            if added:
                vecs = np.vstack([vecs, np.array([a["vector"] for a in added], dtype=vecs.dtype)])
            for d in docs:
                if d["doc_id"] in replaced:
                    d["replaced_by"] = replaced[d["doc_id"]]
        return cls(docs, vecs)

    def before(self, date):
        """Indices of the documents dated strictly before `date`. Same-day documents are left
        out: we cannot tell which came first, and seeing a same-day twin makes REDUNDANT trivial."""
        return [i for i, d in enumerate(self.docs) if d["date"] < date]
