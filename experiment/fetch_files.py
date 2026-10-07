"""Turn every test case into an input FILE, the way the system is used in practice.

For each case in data/cases_oir.jsonl:
- if its bulletin has a PDF or Word attachment, download it (that file is the input);
- otherwise write the bulletin itself as a .md file, which stands in for a notice you type.

Files go to data/files/<case_id>/<name>.<ext>. The file name is the attachment's own label
(e.g. "2026年度日本國際交流基金會經費資助計畫簡章.pdf"), because a file's name is its title.
The answer key does not change: the attachment of a bulletin relates to the older bulletins
exactly as the bulletin itself does.

    python experiment/fetch_files.py

Downloads are cached and rate-limited to one per second. A PDF with no text layer (a scan)
falls back to the .md, and the manifest records which kind each case got.
"""

import json
import re
import sys
import time

import requests

import corpus as C

PREFERENCE = {"pdf": 0, "docx": 1, "odt": 2}   # .doc (old binary Word) cannot be read
MIN_CHARS = 200                                  # below this a PDF is a scan or a poster
DELAY = 1.0


def safe_name(label, ext):
    name = re.sub(r'[\\/:*?"<>|\s]+', "_", label).strip("._")[:80] or "attachment"
    return f"{name}.{ext}"


def pick_attachment(bulletin):
    """The main readable attachment: PDF before Word, and a 簡章/辦法/要點 (the actual call)
    before posters and schedules."""
    files = [a for a in bulletin.get("attachments", []) if a["ext"] in PREFERENCE]
    main = lambda a: not re.search(r"簡章|辦法|要點|說明|guideline|brochure", a["label"], re.I)
    return min(files, key=lambda a: (main(a), PREFERENCE[a["ext"]]), default=None)


def write_md(folder, bulletin):
    path = folder / safe_name(bulletin["title"], "md")
    path.write_text(f"# {bulletin['title']}\n\n{bulletin['text']}\n", encoding="utf-8")
    return path


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    by_id = {d["doc_id"]: d for d in C.bulletins()}
    cases = [json.loads(l) for l in (C.DATA / "cases_oir.jsonl").open(encoding="utf-8")]
    rows = []
    for n, c in enumerate(cases, 1):
        b = by_id[c["doc_id"]]
        folder = C.FILES / c["case_id"]
        folder.mkdir(parents=True, exist_ok=True)
        att, path, kind, note = pick_attachment(b), None, None, ""
        if att:
            target = folder / safe_name(att["label"], att["ext"])
            try:
                if not target.exists():
                    r = requests.get(att["url"], timeout=60)
                    r.raise_for_status()
                    if not r.content:  # older attachments (before ~2022) answer 200 with 0 bytes
                        raise ValueError("server returned an empty file (attachment no longer hosted)")
                    target.write_bytes(r.content)
                    time.sleep(DELAY)
                _, text = C.read_file(target)
                if len(text) >= MIN_CHARS:
                    path, kind = target, f"attachment-{att['ext']}"
                else:
                    note = f"{att['ext']} has only {len(text)} characters of text (scan?)"
            except Exception as e:
                note = f"download/read failed: {e!r}"[:200]
        if path is None:
            path, kind = write_md(folder, b), "bulletin-md"
        _, text = C.read_file(path)
        rows.append({"case_id": c["case_id"], "doc_id": c["doc_id"], "date": c["date"],
                     "file": path.relative_to(C.DATA).as_posix(), "kind": kind,
                     "chars": len(text), "note": note})
        if n % 20 == 0:
            print(f"{n}/{len(cases)}", flush=True)

    with C.MANIFEST.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"wrote {C.MANIFEST.name}:", dict(Counter(r["kind"] for r in rows)))
    for r in rows:
        if r["note"]:
            print(f"  {r['case_id']}: {r['note']}")


if __name__ == "__main__":
    main()
