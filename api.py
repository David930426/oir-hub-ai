from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pipeline import ingest_text, route_document, ensure_collection, COLLECTION, client

app = FastAPI(title="OIR Document Router")


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.post("/api/check")
async def check(file: UploadFile = File(...)):
    """Upload an incoming document and get the routing decision."""
    text = (await file.read()).decode("utf-8", errors="ignore")
    result = route_document(text)
    result["filename"] = file.filename
    return result


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)):
    """Add a document to the knowledge base."""
    text = (await file.read()).decode("utf-8", errors="ignore")
    chunks = ingest_text(file.filename, text)
    return {"filename": file.filename, "chunks": chunks}


@app.get("/api/stats")
def stats():
    ensure_collection()
    info = client.get_collection(COLLECTION)
    return {"collection": COLLECTION, "points": info.points_count}