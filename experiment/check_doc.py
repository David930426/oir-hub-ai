"""Check your own document against every OIR bulletin, and optionally update the library.

Give it a PDF, Word (.docx/.odt), .txt or .md file. It lists the past bulletins related to it
and decides whether your new post is an UPDATE of one of them, something NEW, or REDUNDANT
(already posted). It uses the same code as the experiment (run.py), so the accuracy measured
there is the accuracy of this tool.

    python experiment/check_doc.py my_notice.pdf                     # look only, change nothing
    python experiment/check_doc.py my_notice.pdf --apply             # act on the AI's decision
    python experiment/check_doc.py my_notice.md --apply --decision UPDATE --target oir-1100
    python experiment/check_doc.py my_notice.md --date 2025-10-01    # pretend it is that day
    python experiment/check_doc.py --history                         # what --apply has changed
    python experiment/check_doc.py --undo                            # take back the last --apply

What --apply does to the library (data/kb_changes.jsonl; the scraped bulletins are never edited):
    UPDATE     your file is added, and the target bulletin is marked "replaced by" it - it stops
               being offered as the current version, but stays in the history
    NEW        your file is added
    REDUNDANT  nothing is added; the existing bulletin already says it
"""

import argparse
import datetime as dt
import hashlib
import json
import sys

import numpy as np

import corpus as C
from methods import LLM, TOP_LLM, signals

TOP_RELATED = 10
RRF_K = 60  # standard reciprocal-rank-fusion constant


def related(sig, lib):
    """Top related documents for display: meaning, words and title rankings merged by
    reciprocal rank fusion, so a document near the top of any one of them is listed."""
    ids = list(sig["cos"])
    fused = dict.fromkeys(ids, 0.0)
    for scores in (sig["cos"], sig["bm25"], {d: sig["title_sim"].get(d, 0.0) for d in ids}):
        for rank, d in enumerate(sorted(ids, key=scores.get, reverse=True)):
            fused[d] += 1.0 / (RRF_K + rank + 1)
    top = sorted(ids, key=fused.get, reverse=True)[:TOP_RELATED]
    return [lib.docs[lib.index[d]] for d in top]


def log_events(events):
    C.KB_LOG.parent.mkdir(parents=True, exist_ok=True)
    with C.KB_LOG.open("a", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def read_log():
    return [json.loads(l) for l in C.KB_LOG.open(encoding="utf-8")] if C.KB_LOG.exists() else []


def apply(decision, target, query, path, lib):
    if decision == "REDUNDANT":
        print("\nNothing added: the library already has this (REDUNDANT).")
        return
    doc_id = "file-" + hashlib.md5(f"{query['title']}\n{query['text']}".encode()).hexdigest()[:8]
    if doc_id in lib.index:
        print(f"\nNothing added: this exact file is already in the library as {doc_id}.")
        return
    batch = dt.datetime.now().isoformat(timespec="seconds")
    vector = C._encode([f"{query['title']}\n{query['text']}"], "passage")[0]
    doc = {"doc_id": doc_id, "source": "your_file", "title": query["title"], "date": query["date"],
           "text": query["text"], "url": str(path), "attachments": [], "links": []}
    events = [{"action": "add", "batch": batch, "doc": doc, "vector": np.round(vector, 6).tolist()}]
    if decision == "UPDATE" and target:
        events.append({"action": "replace", "batch": batch, "target": target, "by": doc_id})
    log_events(events)
    print(f"\nLIBRARY UPDATED (batch {batch}):")
    print(f"  added    {doc_id}  {query['title']}")
    if decision == "UPDATE" and target:
        t = lib.docs[lib.index[target]]
        print(f"  replaced {target}  {t['title']}  -> now marked 'replaced by {doc_id}'")
    print("  Undo with: python experiment/check_doc.py --undo")


def undo():
    log = read_log()
    undone = {e["batch"] for e in log if e["action"] == "undo"}
    batches = [e["batch"] for e in log if e["action"] == "add" and e["batch"] not in undone]
    if not batches:
        print("Nothing to undo.")
        return
    last = batches[-1]
    ids = [e["doc"]["doc_id"] for e in log if e["action"] == "add" and e["batch"] == last]
    log_events([{"action": "undo", "batch": last, "doc_ids": ids}])
    print(f"Undid batch {last}: removed {', '.join(ids)} and restored what it had replaced.")


def history():
    log = read_log()
    if not log:
        print("No changes yet - the library is the scraped bulletins only.")
        return
    for e in log:
        if e["action"] == "add":
            print(f"{e['batch']}  ADD      {e['doc']['doc_id']}  {e['doc']['title']}")
        elif e["action"] == "replace":
            print(f"{e['batch']}  REPLACE  {e['target']}  -> replaced by {e['by']}")
        else:
            print(f"{e['batch']}  UNDO     {', '.join(e['doc_ids'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", help="PDF, .docx, .odt, .txt or .md")
    ap.add_argument("--date", default=dt.date.today().isoformat(),
                    help="the document's date (default today); only older documents are searched")
    ap.add_argument("--no-llm", action="store_true", help="list related documents only")
    ap.add_argument("--apply", action="store_true", help="add the file to the library / retire the old one")
    ap.add_argument("--decision", choices=["UPDATE", "NEW", "REDUNDANT"], help="override the AI's decision")
    ap.add_argument("--target", help="override which document is replaced (a doc_id from the list)")
    ap.add_argument("--undo", action="store_true", help="take back the last --apply")
    ap.add_argument("--history", action="store_true", help="list the changes --apply has made")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.undo:
        return undo()
    if args.history:
        return history()
    if not args.file:
        ap.error("give a file, or --history / --undo")

    title, text = C.read_file(args.file)
    if not text:
        sys.exit(f"No readable text in {args.file} (a scanned PDF has no text layer).")
    query = {"title": title, "text": text, "date": args.date}
    lib = C.Library.load(with_changes=True)
    case = {"case_id": "input", "query": query}
    sig = signals([case], lib, rerank=False, cache_name=None)["input"]
    if not sig.get("kb_size"):
        sys.exit(f"No documents dated before {args.date} to compare with.")

    print(f"\nYOUR DOCUMENT: {title}  ({len(text)} characters, dated {args.date})")
    print(f"\nRELATED PAST DOCUMENTS (top {TOP_RELATED} of {sig['kb_size']}, most related first;"
          f" C1-C{TOP_LLM} are what the AI reads)")
    cands = sig["by_cos"][:TOP_LLM]
    for k, d in enumerate(related(sig, lib), 1):
        mark = f"C{cands.index(d['doc_id']) + 1}" if d["doc_id"] in cands else "  "
        mine = "  [your file]" if d.get("source") == "your_file" else ""
        print(f" {k:2d}. [{mark}] {d['date']}  {d['doc_id']:13s} {d['title'][:58]}{mine}")
        print(f"       meaning {sig['cos'][d['doc_id']]:.2f} · words {sig['bm25'][d['doc_id']]:.2f}"
              f" · title {sig['title_sim'].get(d['doc_id'], 0):.2f} · {d.get('url', '')}")

    decision, target = args.decision, args.target
    if not args.no_llm and not decision:
        label, target_ai, info = LLM("M2 LLM + candidates").predict(case, sig)
        if "error" in info:
            sys.exit(f"\nThe AI could not answer ({info['error']}). Is Ollama running?")
        decision, target = label, target or target_ai
        print(f"\nAI DECISION: {decision}   (confidence {info['confidence']:.2f}, {info['latency_s']}s)")
        print(f"Reason:      {info['rationale']}")
    elif decision:
        print(f"\nYOUR DECISION: {decision}")

    if decision in ("UPDATE", "REDUNDANT"):
        if not target or target not in lib.index:
            sys.exit(f"{decision} needs a target - add --target <doc_id> from the list above.")
        t = lib.docs[lib.index[target]]
        verb = "replaces" if decision == "UPDATE" else "repeats"
        print(f"Your document {verb}: {t['date']} {target} {t['title']}\n  {t.get('url', '')}")
    elif decision == "NEW":
        print("No earlier document covers this; the related list above is for reference only.")

    if args.apply and decision:
        apply(decision, target, query, args.file, lib)
    elif decision and decision != "REDUNDANT":
        print("\n(Nothing changed. Add --apply to put this into the library.)")


if __name__ == "__main__":
    main()
