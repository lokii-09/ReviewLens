"""
api.py

Wraps the existing ReviewLens system (from end_to_end.py) as a small HTTP API,
so a frontend can call it instead of everything running only in the terminal.

Run with:
    pip install fastapi uvicorn
    uvicorn src.api:app --reload --port 8000

The heavy setup in end_to_end.py (loading Chroma, the cross-encoder, BM25)
runs ONCE when this file is imported at server startup — not per request.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Make sure this file's own folder (src/) is on the import path — without
# this, "from end_to_end import ..." fails when uvicorn loads this as
# "src.api" rather than running it directly, since Python doesn't put
# src/ itself on sys.path in that case.
sys.path.insert(0, os.path.dirname(__file__))

from end_to_end import process_user_query

app = FastAPI(title="ReviewLens API")

# Allow the local React dev server (Vite's default port) to call this API.
# In a real production deploy you'd restrict this to your actual frontend's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    query: str
    intent: str          # "local" or "global"
    engine: str           # human-readable label for the frontend badge
    answer: str


@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = process_user_query(request.question.strip())

    engine_label = "Hybrid RAG" if result["intent"] == "local" else "GraphRAG"

    return QueryResponse(
        query=result["query"],
        intent=result["intent"],
        engine=engine_label,
        answer=result["answer"],
    )


@app.get("/api/health")
def health():
    return {"status": "ok"}