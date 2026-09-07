"""Load the 2024 bulletins, then route the 2025 ones as if a user had uploaded them."""

import json
from pathlib import Path

from extract import extract_text
from pipeline import ensure_collection, ingest_file, route_document

DATA_DIR = Path(__file__).parent / "data"
pdfs = sorted(DATA_DIR.glob("*.pdf"))

ensure_collection(reset=True)

for path in [p for p in pdfs if "2024" in p.name]:
    print(f"{path.name}: {ingest_file(path)} chunks")

for path in [p for p in pdfs if "2025" in p.name]:
    result = route_document(extract_text(path.name, path.read_bytes()))
    print(f"\n=== {path.name} ===")
    print(json.dumps(result, ensure_ascii=False, indent=2)[:900])