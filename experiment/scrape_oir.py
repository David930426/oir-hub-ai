"""Scrape the public OIR news feed (oir.thu.edu.tw) into experiment/data/oir_news.jsonl.

Posts are addressed by a sequential `Sn` id. Raw HTML is cached under data/raw/, so a re-run
only fetches pages it has not seen, and parsing can be changed without hitting the site again.
One request per second - this is a university server, not a stress test.

    python experiment/scrape_oir.py            # Sn 1..1350
    python experiment/scrape_oir.py 1200 1350  # a range
"""

import html
import json
import re
import sys
import time
from pathlib import Path

import requests

BASE = "https://oir.thu.edu.tw/redirect.php?ID=news&Sn={}"
DATA = Path(__file__).parent / "data"
RAW = DATA / "raw"
OUT = DATA / "oir_news.jsonl"
DELAY = 1.0

HEADER = re.compile(
    r'<h3 class="page-title">(.*?)</h3>.*?title="日期"></i>\s*(\d{4}-\d{2}-\d{2})'
    r'.*?title="修改人"></i>\s*([^<]*?)\s*<',
    re.S,
)
BODY = re.compile(r'<div class="page-content">(.*?)<ul class="list-group">', re.S)
LINK = re.compile(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.S)


def fetch(sn):
    path = RAW / f"{sn}.html"
    if path.exists():
        return path.read_text(encoding="utf-8"), False
    r = requests.get(BASE.format(sn), timeout=30)
    r.encoding = "utf-8"
    r.raise_for_status()
    path.write_text(r.text, encoding="utf-8")
    return r.text, True


def to_text(fragment):
    fragment = re.sub(r"<(script|style).*?</\1>", "", fragment, flags=re.S)
    fragment = re.sub(r"<br\s*/?>|</p>|</li>|</tr>|</h\d>", "\n", fragment)
    text = html.unescape(re.sub(r"<[^>]+>", " ", fragment)).replace("\xa0", " ")
    lines = (re.sub(r"[ \t]+", " ", l).strip() for l in text.splitlines())
    return "\n".join(l for l in lines if l)


def parse(sn, page):
    head = HEADER.search(page)
    if not head:
        return None  # ids past the end, and deleted posts, render an empty shell
    body = BODY.search(page)
    body_html = body.group(1) if body else ""
    return {
        "doc_id": f"oir-{sn}",
        "source": "oir_news",
        "sn": sn,
        "url": BASE.format(sn),
        "title": to_text(head.group(1)),
        "date": head.group(2),
        "author": head.group(3).strip(),
        "text": to_text(body_html),
        "links": [
            {"href": h, "label": to_text(t)}
            for h, t in LINK.findall(body_html)
            if not h.startswith(("mailto:", "javascript:"))
        ],
    }


def main(lo=1, hi=1350):
    RAW.mkdir(parents=True, exist_ok=True)
    posts, misses = [], 0
    for sn in range(lo, hi + 1):
        try:
            page, fetched = fetch(sn)
        except requests.RequestException as e:
            print(f"Sn={sn}: {e}", file=sys.stderr)
            time.sleep(DELAY * 5)
            continue
        post = parse(sn, page)
        if post:
            posts.append(post)
        else:
            misses += 1
        if sn % 50 == 0:
            print(f"Sn={sn}: {len(posts)} posts, {misses} empty", flush=True)
        if fetched:
            time.sleep(DELAY)

    with OUT.open("w", encoding="utf-8") as f:
        for p in posts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(posts)} posts to {OUT} ({misses} empty ids)")


if __name__ == "__main__":
    main(*map(int, sys.argv[1:3]))
