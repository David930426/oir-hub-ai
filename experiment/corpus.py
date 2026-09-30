"""Load the two corpora and compute the shared retrieval signals.

Two corpora, kept apart because they are different kinds of document:
- `oir`   : announcement posts scraped from oir.thu.edu.tw (scrape_oir.py). Main dataset.
- `local` : the brochure PDFs in data/, i.e. the attachments. Small pilot set for debugging
            prompts and thresholds. Never reported as a result.

A case's knowledge base is every document of the same corpus dated strictly before it
(chronological replay), which is how the pipeline would have seen them in production.
"""

import hashlib
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiment"
DATA = EXP / "data"
CACHE = DATA / "cache"

# Local pilot files: date = the PDF's creation date (none of them states an issue date in a
# consistent place). The JAL file is a scanned image with no text layer, and the experiment
# design PDF is not an OIR document, so both are left out.
LOCAL_FILES = {
    "2024年春全球姊妹校交換計畫甄選簡章_1012.pdf": "2023-10-12",
    "2024年秋全球姊妹校交換計畫甄選簡章_0314.pdf": "2024-03-14",
    "2025年春全球姊妹校交換計畫甄選簡章_10-01.pdf": "2024-10-01",
    "2025年春學期東海大學赴全球姊妹校交換甄試簡章 .pdf": "2024-10-02",  # byte-identical re-post
    "2025年秋全球姊妹校交換計畫甄選簡章_0319.pdf": "2025-03-19",
    "2027年春學期東海大學赴境外姊妹校交換甄試簡章.pdf": "2026-09-15",
    "2025年長期交換計畫甄選辦法重大更新通知.pdf": "2025-11-21",
    "2025年ICU國際日本暑期課程ISPJ.pdf": "2024-10-28",
    "2026年ICU國際日本暑期課程ISPJ.pdf": "2025-10-30",
    "2024年斯洛伐克高等教育及研發環境簡介.pdf": "2024-10-16",
    "2025年新伊拉斯莫斯聯合碩士學程及教育部歐盟獎學金說明會.pdf": "2025-11-06",
    "2025年立命館大學Ritsumeikan-Data-Science-Program.pdf": "2025-11-18",
    "2026年Young飛全球行動計畫培訓營簡章.pdf": "2025-10-30",
    "2026年教育部人文社會科學學術人才跨國培育計畫甄選簡章.pdf": "2025-10-03",
    "2026年教育部赴捷克高等教育機構短期進修研究獎學金甄選簡章.pdf": "2026-01-20",
    "2026年日本國際交流基金會經費資助計畫.pdf": "2025-10-14",
    "2026年第一期日本台灣交流協會獎學金短期留學生申請簡章.pdf": "2025-10-07",
    "2025年教育部補助學生出國參加國際性學術技藝能競賽作業要點.odt": "2025-01-17",
}


# ---------------------------------------------------------------- loading

def _read_local(path):
    if path.suffix == ".pdf":
        import pymupdf
        with pymupdf.open(path) as d:
            return "\n".join(p.get_text() for p in d)
    if path.suffix == ".odt":
        import zipfile
        xml = zipfile.ZipFile(path).read("content.xml").decode("utf-8")
        xml = re.sub(r"</text:(p|h)>", "\n", xml)
        return re.sub(r"<[^>]+>", "", xml)
    raise ValueError(path)


@lru_cache(maxsize=None)
def load(corpus):
    """Documents sorted by (date, doc_id), each a dict with doc_id/title/date/text."""
    if corpus == "oir":
        docs = [json.loads(l) for l in (DATA / "oir_news.jsonl").open(encoding="utf-8")]
    elif corpus == "local":
        docs = []
        for name, date in LOCAL_FILES.items():
            text = _read_local(ROOT / "data" / name)
            docs.append({
                "doc_id": "local-" + hashlib.md5(name.encode()).hexdigest()[:8],
                "source": "local_file", "title": Path(name).stem.strip(), "date": date,
                "text": re.sub(r"[ \t]+", " ", text).strip(), "file": name,
            })
    else:
        raise ValueError(corpus)
    # Undated posts (0000-00-00) are static pages - school profiles, old program pages. With no
    # date they would sort before everything and sit in every knowledge base, so they go.
    docs = [d for d in docs if (d["title"] or d["text"]) and not d["date"].startswith("0000")]
    return sorted(docs, key=lambda d: (d["date"], d["doc_id"]))


def index(corpus):
    return {d["doc_id"]: i for i, d in enumerate(load(corpus))}


def kb_indices(corpus, doc):
    """Knowledge base for an incoming doc: everything strictly older. Same-day posts are
    excluded too - we cannot tell which came first, and letting a doc see a same-day twin
    would make REDUNDANT trivially easy."""
    return [i for i, d in enumerate(load(corpus)) if d["date"] < doc["date"]]


def doc_text(doc, limit=None):
    s = f"{doc['title']}\n{doc['text']}"
    return s[:limit] if limit else s


# ---------------------------------------------------------------- normalisation

CN_NUM = "〇零一二三四五六七八九十百"
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


def embeddings(corpus):
    """L2-normalised e5 vectors for every doc, cached on disk by content hash."""
    docs = load(corpus)
    key = hashlib.md5("".join(d["doc_id"] + doc_text(d, EMBED_CHARS) for d in docs).encode()).hexdigest()[:12]
    path = CACHE / f"e5_{corpus}_{key}.npy"
    if path.exists():
        return np.load(path)
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBED_MODEL, device=dev)
    # Every doc is embedded as a passage AND compared as a query ("query:" prefix) - the
    # incoming doc plays the query role. Stored separately so both are available.
    texts = [doc_text(d, EMBED_CHARS) for d in docs]
    p = model.encode([f"passage: {t}" for t in texts], normalize_embeddings=True, batch_size=32)
    q = model.encode([f"query: {t}" for t in texts], normalize_embeddings=True, batch_size=32)
    CACHE.mkdir(parents=True, exist_ok=True)
    arr = np.stack([q, p])
    np.save(path, arr)
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return arr


def cosine_scores(corpus, i, kb):
    """Cosine between incoming doc i (as query) and each kb doc (as passage)."""
    q, p = embeddings(corpus)
    return p[kb] @ q[i]


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
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(t) for t in docs_tokens]
        self.len = np.array([len(t) for t in docs_tokens], dtype=float)
        self.avg = self.len.mean() if len(self.len) else 1.0
        df = Counter(w for tf in self.tf for w in tf)
        n = len(docs_tokens)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def score(self, q_tokens, j):
        tf, s = self.tf[j], 0.0
        norm = self.k1 * (1 - self.b + self.b * self.len[j] / self.avg)
        for w, qc in Counter(q_tokens).items():
            if w in tf:
                s += self.idf.get(w, 0.0) * tf[w] * (self.k1 + 1) / (tf[w] + norm)
        return s


@lru_cache(maxsize=None)
def bm25(corpus):
    """One BM25 index over the whole corpus. IDF therefore sees 'future' documents too; with
    ~1,300 posts the effect on term weights is negligible, and it keeps scoring O(1) per case."""
    return BM25([tokens(doc_text(d, 4000)) for d in load(corpus)])


def bm25_scores(corpus, i, kb):
    """BM25 of doc i against each kb doc, divided by doc i's score against itself so the
    value is roughly 0-1 and one threshold can work across queries of different lengths."""
    idx, docs = bm25(corpus), load(corpus)
    q = tokens(doc_text(docs[i], 4000))
    self_score = idx.score(q, i) or 1.0
    return np.array([idx.score(q, j) / self_score for j in kb])
