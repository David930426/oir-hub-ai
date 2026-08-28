import json
from pipeline import ensure_collection, ingest_file, route_document

ensure_collection(reset=True)

for path in ["mock-data/tunghai_academic_guide.txt", "mock-data/tunghai_life_guide.txt"]:
    print(f"{path}: {ingest_file(path)} chunks")

for path in ["mock-data/mock-arc.txt", "mock-data/mock-dorm.txt", "mock-data/mock-scooter.txt"]:
    text = open(path, encoding="utf-8").read()
    result = route_document(text)
    print(f"\n=== {path} ===")
    print(json.dumps(result, ensure_ascii=False, indent=2)[:600])