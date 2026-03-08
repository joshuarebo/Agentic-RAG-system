import hashlib
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.models import (
    QueryRequest,
    QueryResponse,
    DocumentInfo,
    HealthResponse,
    ModelUsage,
)
from app.ingestion import parse_document, chunk_text
from app.vectorstore import VectorStore
from app.router import ModelRouter
from app.retriever import Retriever
from app.agent import DecisionAgent
from app.config import get_settings

router = APIRouter(prefix="/api")

# Shared singleton instances (lazy-initialized)
_vector_store = None
_model_router = None
_retriever = None
_agent = None
_query_count = 0


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        settings = get_settings()
        _vector_store = VectorStore(persist_dir=settings.chroma_persist_dir)
    return _vector_store


def get_model_router() -> ModelRouter:
    global _model_router
    if _model_router is None:
        _model_router = ModelRouter()
    return _model_router


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever(get_vector_store(), get_model_router())
    return _retriever


def get_agent() -> DecisionAgent:
    global _agent
    if _agent is None:
        _agent = DecisionAgent(get_retriever(), get_model_router())
    return _agent


def reset_instances():
    """Reset all singletons (used by tests)."""
    global _vector_store, _model_router, _retriever, _agent
    _vector_store = None
    _model_router = None
    _retriever = None
    _agent = None


@router.post("/documents/upload", response_model=DocumentInfo)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document (PDF, TXT, MD) for indexing."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    ext = file.filename.lower().rsplit(".", 1)[-1] if "." in file.filename else ""
    if ext not in ("pdf", "txt", "md", "markdown"):
        raise HTTPException(
            400, f"Unsupported file type: .{ext}. Supported: pdf, txt, md"
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(400, "Empty file")

    try:
        text = parse_document(contents, file.filename)
    except Exception as e:
        raise HTTPException(400, f"Failed to parse document: {str(e)}")

    if not text.strip():
        raise HTTPException(400, "Document contains no extractable text")

    settings = get_settings()
    chunks = chunk_text(
        text,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    if not chunks:
        raise HTTPException(400, "Document produced no valid chunks")

    doc_id = hashlib.md5(
        f"{file.filename}_{datetime.now(timezone.utc).isoformat()}".encode()
    ).hexdigest()[:12]

    vs = get_vector_store()
    chunk_count = vs.add_document(doc_id, file.filename, chunks)

    return DocumentInfo(
        doc_id=doc_id,
        filename=file.filename,
        chunk_count=chunk_count,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all indexed documents."""
    vs = get_vector_store()
    docs = vs.list_documents()
    return [
        DocumentInfo(
            doc_id=d["doc_id"],
            filename=d["filename"],
            chunk_count=d["chunk_count"],
            created_at="",
        )
        for d in docs
    ]


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from the index."""
    vs = get_vector_store()
    vs.delete_document(doc_id)
    return {"status": "deleted", "doc_id": doc_id}


@router.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """Submit a question for the AI Decision Agent."""
    global _query_count
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    agent = get_agent()

    try:
        response = await agent.analyze(
            question=request.question,
            model_preference=request.model_preference,
        )
        _query_count += 1
        return response
    except Exception as e:
        raise HTTPException(500, f"Agent error: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    vs = get_vector_store()
    mr = get_model_router()

    return HealthResponse(
        status="healthy",
        documents_count=vs.get_document_count(),
        models_available=mr.get_available_models(),
    )


@router.get("/logs", response_model=List[ModelUsage])
async def get_logs(limit: int = 50):
    """Get recent model usage logs."""
    mr = get_model_router()
    return mr.get_logs(limit=limit)


@router.get("/dashboard/data")
async def dashboard_data():
    """Aggregated stats for the dashboard."""
    vs = get_vector_store()
    mr = get_model_router()

    logs = mr.get_logs(limit=200)

    total_in = sum(l.tokens_input for l in logs)
    total_out = sum(l.tokens_output for l in logs)
    avg_latency = (sum(l.latency_ms for l in logs) / len(logs)) if logs else 0

    models_breakdown = {}
    for l in logs:
        entry = models_breakdown.setdefault(l.model, {"calls": 0, "tokens": 0})
        entry["calls"] += 1
        entry["tokens"] += l.tokens_input + l.tokens_output

    return {
        "documents": vs.get_document_count(),
        "chunks": vs.get_total_chunks(),
        "total_queries": _query_count,
        "api_calls": len(logs),
        "total_tokens": total_in + total_out,
        "avg_latency_ms": round(avg_latency, 1),
        "models": models_breakdown,
        "recent_logs": [
            {
                "model": l.model,
                "tokens_in": l.tokens_input,
                "tokens_out": l.tokens_output,
                "latency_ms": round(l.latency_ms, 1),
                "timestamp": l.timestamp,
            }
            for l in list(reversed(logs))[:10]
        ],
    }
