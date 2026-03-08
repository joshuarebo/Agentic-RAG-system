from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from datetime import datetime, timezone


class DecisionEnum(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NEEDS_INFO = "NEEDS_INFO"


class ChunkReference(BaseModel):
    document_source: str
    chunk_index: int
    content: str
    relevance_score: float


class ReasoningStep(BaseModel):
    step_number: int
    action: str
    detail: str
    result: Optional[str] = None


class DecisionResult(BaseModel):
    decision: DecisionEnum
    reasons: List[str]
    evidence: List[ChunkReference]
    confidence: float = Field(ge=0, le=1)
    reasoning_steps: List[ReasoningStep]


class ModelUsage(BaseModel):
    model: str
    tokens_input: int
    tokens_output: int
    latency_ms: float
    cached_tokens: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class QueryRequest(BaseModel):
    question: str
    model_preference: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    decision: DecisionResult
    model_usage: ModelUsage


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    created_at: str


class HealthResponse(BaseModel):
    status: str
    documents_count: int
    models_available: List[str]
