"""Check your own document against every OIR bulletin.

Give it a PDF, Word, .txt or .md file. It finds the past bulletins related to it and tells you
whether your new post is an UPDATE of one of them, something NEW, or REDUNDANT (already posted).
You still write the new post either way; the related bulletins are what it should link to,
and an UPDATE target is the old post that is now out of date.

    python experiment/check_doc.py "data/2026年日本國際交流基金會經費資助計畫.pdf"
    python experiment/check_doc.py my_notice.md --before 2025-10-01   # pretend it is that date
    python experiment/check_doc.py my_notice.md --no-llm              # related posts only, no AI

How the related bulletins are found: three rankings - meaning (e5 cosine), shared words (BM25)
and title similarity - merged with reciprocal rank fusion, so a bulletin near the top of any one
of them makes the list. The experiment showed each signal fails differently (BM25 scored best,
title rules fall for look-alikes, cosine scores everything 0.92+), so no single one is trusted.
"""

import argparse
import difflib
import io
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))  # for app/extract.py
import corpus as C
from extract import extract_text
from methods import CAND_CHARS, IN_CHARS, SYSTEM, Verdict, call_llm

TOP_RELATED = 10  # shown to you
TOP_LLM = 5       # shown to the AI (it cannot read hundreds of bulletins at once)
RRF_K = 60        # standard reciprocal-rank-fusion constant


def read_document(path):
    """Title and text of the input file. The title is the first heading/line of a .txt/.md,
    or the file name for PDF/Word, which is what OIR staff would name the bulletin after."""
    path = Path(path)
    text = extract_text(path.name, path.read_bytes())
    if not text.strip():
        sys.exit(f"No readable text in {path.name} (a scanned PDF has no text layer).")
    if path.suffix.lower() in (".txt", ".md"):
        first = next(l for l in text.splitlines() if l.strip())
        title = re.sub(r"^#+\s*", "", first).strip()
    else:
        title = path.stem.strip()
    return title, re.sub(r"[ \t]+", " ", text).strip()


def related(title, text, before=None):
    """Rank every bulletin (dated before `before`, if given) by fused similarity."""
    docs = C.load("oir")
    kb = [i for i, d in enumerate(docs) if not before or d["date"] < before]

    # 1. meaning: e5 cosine, the document as a query against stored passage vectors
    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(C.EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    q = model.encode(f"query: {title}\n{text}"[: C.EMBED_CHARS + 7], normalize_embeddings=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # free the GPU for Ollama
    _, passages = C.embeddings("oir")
    cos = passages[kb] @ q

    # 2. shared words: BM25 with character bigrams for Chinese
    index, toks = C.bm25("oir"), C.tokens(f"{title}\n{text}"[:4000])
    bm = np.array([index.score(toks, j) for j in kb])

    # 3. title similarity after removing years, seasons and prefixes
    nt = C.norm_title(title)
    ts = np.array([difflib.SequenceMatcher(None, nt, C.norm_title(docs[j]["title"])).ratio() if nt else 0.0
                   for j in kb])

    fused = np.zeros(len(kb))
    for scores in (cos, bm, ts):
        ranks = np.argsort(np.argsort(-scores))
        fused += 1.0 / (RRF_K + ranks + 1)
    order = np.argsort(-fused)
    return [{"doc": docs[kb[k]], "cosine": float(cos[k]), "bm25": float(bm[k] / (bm.max() or 1)),
             "title": float(ts[k])} for k in order[:TOP_RELATED]]


def judge(title, text, cands):
    """Same prompt and answer format as condition M2 in the experiment."""
    msg = f"INCOMING ANNOUNCEMENT\ntitle: {title}\ndate: (new)\n{text[:IN_CHARS]}"
    blocks = [f"[C{k + 1}]\ntitle: {c['doc']['title']}\ndate: {c['doc']['date']}\n{c['doc']['text'][:CAND_CHARS]}"
              for k, c in enumerate(cands[:TOP_LLM])]
    msg += "\n\nEXISTING DOCUMENTS (most similar first)\n" + "\n\n".join(blocks)
    verdict, cost = call_llm([{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}], Verdict)
    target = None
    if verdict.label != "NEW" and verdict.target != "NONE" and int(verdict.target[1:]) <= len(cands):
        target = cands[int(verdict.target[1:]) - 1]["doc"]
    return verdict, target, cost


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="PDF, .docx, .txt or .md")
    ap.add_argument("--before", help="only search bulletins dated before YYYY-MM-DD")
    ap.add_argument("--no-llm", action="store_true", help="list related bulletins only")
    args = ap.parse_args()
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    title, text = read_document(args.file)
    print(f"\nYOUR DOCUMENT: {title}  ({len(text)} characters)")
    cands = related(title, text, args.before)

    print(f"\nRELATED PAST BULLETINS (top {TOP_RELATED}, most related first)")
    for k, c in enumerate(cands, 1):
        d = c["doc"]
        mark = f"C{k}" if k <= TOP_LLM else "  "
        print(f" {k:2d}. [{mark}] {d['date']}  {d['doc_id']:9s} {d['title'][:60]}")
        print(f"            meaning {c['cosine']:.2f} · words {c['bm25']:.2f} · title {c['title']:.2f} · {d['url']}")

    if args.no_llm:
        return
    verdict, target, cost = judge(title, text, cands)
    print(f"\nDECISION: {verdict.label}   (AI confidence {verdict.confidence:.2f}, {cost['latency_s']}s)")
    print(f"Reason:   {verdict.rationale}")
    if verdict.label == "UPDATE" and target:
        print(f"Your new post replaces: {target['date']} {target['doc_id']} {target['title']}\n"
              f"  -> link to it from the new post and mark it as outdated: {target['url']}")
    elif verdict.label == "REDUNDANT" and target:
        print(f"Already posted as: {target['date']} {target['doc_id']} {target['title']}\n"
              f"  -> a new post may not be needed; if you post anyway, link to: {target['url']}")
    else:
        print("No earlier post covers this; the new post stands alone (related list above is for reference).")


if __name__ == "__main__":
    main()
