from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pipeline import ingest_text, route_document, ensure_collection, COLLECTION, client
from extract import extract_text

app = FastAPI(title="OIR Document Router")


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.post("/api/check")
async def check(file: UploadFile = File(...)):
    """Upload an incoming document and get the routing decision."""
    text = extract_text(file.filename, await file.read())
    if not text:
        raise HTTPException(400, "No readable text found in that file.")
    result = route_document(text)
    result["filename"] = file.filename
    result["characters"] = len(text)
    return result


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)):
    """Add a document to the knowledge base."""
    text = extract_text(file.filename, await file.read())
    if not text:
        raise HTTPException(400, "No readable text found in that file.")
    chunks = ingest_text(file.filename, text)
    return {"filename": file.filename, "chunks": chunks}


@app.get("/api/stats")
def stats():
    ensure_collection()
    info = client.get_collection(COLLECTION)
    return {"collection": COLLECTION, "points": info.points_count}