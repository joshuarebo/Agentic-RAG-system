# System Guide: How Everything Works

A complete walkthrough of the Policy-Aware AI Decision Agent — every component, every step, every decision explained.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [What Happens When a User Asks a Question](#2-what-happens-when-a-user-asks-a-question)
3. [Document Ingestion Pipeline](#3-document-ingestion-pipeline)
4. [The RAG Pipeline](#4-the-rag-pipeline)
5. [The AI Decision Agent](#5-the-ai-decision-agent)
6. [Multi-Model Routing](#6-multi-model-routing)
7. [Governance & Safety](#7-governance--safety)
8. [The Vector Database](#8-the-vector-database)
9. [The API Layer](#9-the-api-layer)
10. [The UI & Dashboard](#10-the-ui--dashboard)
11. [Testing Strategy](#11-testing-strategy)
12. [Deployment](#12-deployment)
13. [File-by-File Breakdown](#13-file-by-file-breakdown)

---

## 1. The Big Picture

The system is a **Policy-Aware AI Decision Agent**. Think of it as an AI compliance officer:

1. You give it documents (policies, invoices, contracts, offers)
2. You ask it a question ("Is this invoice compliant with the payment policy?")
3. It retrieves the relevant sections from your documents
4. It reasons over the evidence
5. It gives you a structured decision: **PASS**, **FAIL**, or **NEEDS_INFO**

Every decision comes with:
- **Reasons** — why it decided what it decided
- **Evidence** — the exact document chunks it based its decision on, with source citations
- **Confidence** — a 0-100% score
- **Reasoning steps** — a 5-step trace showing its thought process

The philosophy follows **AI-Pass**: Reason → Evaluate → Execute → Explain.

---

## 2. What Happens When a User Asks a Question

Here's the complete flow, step by step:

```
User types: "Is this invoice compliant with the payment policy?"
                            |
                            v
                    ┌───────────────┐
                    │  FastAPI API   │  POST /api/query
                    └───────┬───────┘
                            |
                            v
                    ┌───────────────┐
          LLM Call 1│ Query Expansion│  Llama 3.3 70B (fast, cheap)
                    └───────┬───────┘
                            |
          "invoice compliance payment policy PO number vendor
           address terms net 30 approval threshold verification"
                            |
                            v
                    ┌───────────────┐
                    │  Dual Search   │  ChromaDB vector similarity
                    └───────┬───────┘
                            |
                   Search 1: expanded query (top 14 candidates)
                   Search 2: original query (top 7 candidates)
                   Merge + deduplicate + sort by relevance
                   Take top 7 chunks
                            |
                            v
                    ┌───────────────┐
                    │Evidence Assembly│  Format chunks with citations
                    └───────┬───────┘
                            |
                   [Chunk 0] Source: payment_policy.md | Relevance: 0.8934
                   "All invoices must include a valid Purchase Order..."
                   ---
                   [Chunk 1] Source: sample_invoice.txt | Relevance: 0.8721
                   "Invoice #INV-2024-0042, PO: PO-2024-7891..."
                   ...
                            |
                            v
                    ┌───────────────┐
          LLM Call 2│ Decision Agent │  Gemini 2.0 Flash (powerful)
                    └───────┬───────┘
                            |
                   Raw JSON response from LLM
                            |
                            v
                    ┌───────────────┐
                    │  JSON Parsing  │  Try direct parse, fallback regex
                    └───────┬───────┘
                            |
                            v
                    ┌───────────────┐
                    │  Governance    │  Deterministic safety checks
                    └───────┬───────┘
                            |
                            v
                    ┌───────────────┐
                    │  Structured    │  PASS / FAIL / NEEDS_INFO
                    │  Response      │  + reasons, evidence, confidence,
                    └───────────────┘    reasoning steps, model usage
```

**Total: 2 LLM calls per question** (one cheap, one powerful).

---

## 3. Document Ingestion Pipeline

**File: `app/ingestion.py`**

When you upload a document, three things happen:

### Step 1: Parse the file

The system reads the raw bytes and extracts text based on file type:

- **PDF** → `PyPDF2` reads each page and concatenates text
- **TXT / MD** → decode as UTF-8

Unsupported formats (`.docx`, `.xlsx`, etc.) are rejected with a 400 error.

### Step 2: Chunk the text (LangChain)

The extracted text gets split into smaller pieces using **LangChain's `RecursiveCharacterTextSplitter`**:

```
Settings:
  chunk_size = 500 characters
  chunk_overlap = 50 characters
  separators = ["\n\n", "\n", ". ", " ", ""]
```

How it works:
1. First tries to split on `\n\n` (paragraph boundaries) — this keeps paragraphs intact
2. If a chunk is still too big, splits on `\n` (line breaks)
3. If still too big, splits on `. ` (sentence boundaries)
4. If still too big, splits on ` ` (words)
5. Last resort: splits on any character

After splitting, adjacent small pieces get merged back together as long as they fit within `chunk_size`. The `chunk_overlap` of 50 characters means the end of one chunk overlaps with the start of the next, so context isn't lost at boundaries.

Any chunk shorter than 20 characters is filtered out (not useful for retrieval).

### Step 3: Store in vector database

Each chunk gets:
- A unique ID: `{doc_id}_chunk_{index}`
- Metadata: `doc_id`, `filename`, `chunk_index`
- An embedding vector (computed automatically by ChromaDB using `all-MiniLM-L6-v2`)

The chunk text + embedding + metadata go into ChromaDB for later retrieval.

---

## 4. The RAG Pipeline

**File: `app/retriever.py`**

RAG = **Retrieval-Augmented Generation**. Instead of asking the LLM to answer from memory (which leads to hallucination), we first retrieve relevant documents and feed them as context.

### Step 1: Query Expansion

The user's question might be short or use different terms than the documents. To fix this:

```
User question: "Is this invoice compliant?"

→ Sent to Llama 3.3 70B with prompt:
  "You are a query expansion assistant. Generate an expanded version
   that includes relevant synonyms and related terms."

→ Expanded: "invoice compliance verification payment policy purchase order
   vendor address terms conditions net 30 approval threshold regulatory"
```

This broader query catches more relevant chunks during search.

### Step 2: Dual Vector Search

We search ChromaDB **twice**:

1. **Expanded query** → top `top_k * 2` = 14 candidates (casts a wide net)
2. **Original query** → top `top_k` = 7 candidates (catches what expansion might miss)

Results are merged, deduplicated (by content), sorted by relevance score, and trimmed to the top 7.

### Why dual search?

Query expansion sometimes drifts from the original intent. Searching with both versions gives us the best of both worlds:
- Expanded query finds chunks with different terminology
- Original query keeps focus on what was actually asked

### Step 3: Evidence Assembly

The top 7 chunks become `ChunkReference` objects:

```python
ChunkReference(
    document_source="payment_policy.md",   # which file
    chunk_index=3,                          # which chunk in that file
    content="All invoices must include...", # the actual text
    relevance_score=0.8934                  # how similar (0-1)
)
```

These get passed to the Decision Agent as formatted evidence.

---

## 5. The AI Decision Agent

**File: `app/agent.py`**

This is the brain. It follows a strict 5-step reasoning pipeline:

### Step 1: Identify Intent
```
"What is the user actually asking? They want to know if an invoice
 meets their payment policy requirements."
```

### Step 2: Retrieve Documents
Calls the retriever (Section 4 above). Gets back 7 evidence chunks from potentially multiple documents.

### Step 3: Evaluate Evidence
Checks: did we get any chunks? What's the average relevance score? If no chunks were found, immediately return `NEEDS_INFO` without calling the LLM (saves money).

### Step 4: Analyze Compliance
The evidence chunks get formatted and sent to the primary LLM (Gemini 2.0 Flash) using a **LangChain `ChatPromptTemplate`**:

```python
DECISION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", DECISION_SYSTEM_PROMPT),    # "You are a Policy-Aware AI Decision Agent..."
    ("human", "## Evidence\n\n{context}\n\n## Question\n\n{question}\n\nProduce your decision as JSON.")
])
```

The template is filled with the actual evidence and question, converted to messages, and sent to the LLM.

The system prompt instructs the LLM to respond with strict JSON:

```json
{
    "decision": "PASS",
    "reasons": ["PO number PO-2024-7891 is present", "Vendor address included"],
    "evidence_indices": [0, 1, 3],
    "confidence": 0.88,
    "answer": "The invoice is compliant with the payment policy."
}
```

### Step 5: Generate Decision

The raw LLM response goes through two stages:

**JSON Parsing** (with fallback):
1. Try `json.loads()` directly
2. If that fails, use regex to find `{...}` in the response (handles markdown code blocks)
3. If both fail, return `NEEDS_INFO` with confidence 0.3

**Governance Enforcement** (deterministic, not LLM-dependent):
- See Section 7 below

---

## 6. Multi-Model Routing

**File: `app/router.py`**

The system uses **two different LLMs** for different tasks, accessed through **LangChain's `ChatOpenAI`** via **OpenRouter** (a unified API that provides access to many models).

### Model Selection

| Task | Model | Why |
|------|-------|-----|
| Query expansion | Llama 3.3 70B (secondary) | Fast, cheap, good at simple text tasks |
| Decision generation | Gemini 2.0 Flash (primary) | More capable, better at structured reasoning |

Selection logic:
```python
if complexity == "simple":
    return secondary_model   # Llama
return primary_model         # Gemini
```

### How LangChain ChatOpenAI works here

```python
# Create a ChatOpenAI instance pointing at OpenRouter
llm = ChatOpenAI(
    model="google/gemini-2.0-flash-001",
    api_key="sk-or-v1-...",
    base_url="https://openrouter.ai/api/v1",  # OpenRouter is OpenAI-compatible
)

# Convert our message dicts to LangChain message objects
messages = [SystemMessage(content="..."), HumanMessage(content="...")]

# Call the LLM asynchronously
response = await llm.ainvoke(messages)
content = response.content        # The text response
tokens = response.usage_metadata  # Token counts
```

### Response Caching

A **TTL cache** (5-minute expiry, 100 entries) deduplicates identical requests:
- Same model + same messages + same temperature = cache hit
- Cache hits return instantly with `latency_ms: 0` and `cached_tokens` logged

### Prompt Caching (for Claude models)

When a Claude model is selected, system prompts get wrapped with `cache_control: ephemeral`. This tells Anthropic to cache the prefilled context, reducing costs by up to 90% on subsequent calls with the same system prompt. Currently dormant since we use Gemini/Llama, but ready if models are switched.

### Usage Logging

Every LLM call logs:
- Model name
- Input/output token counts
- Latency in milliseconds
- Cached token count
- Timestamp

These logs power the dashboard and the `/api/logs` endpoint.

---

## 7. Governance & Safety

**File: `app/agent.py` → `_enforce_governance()`**

This is the **deterministic validation layer** — it runs AFTER the LLM responds and enforces rules that the LLM cannot override:

### Rule 1: Confidence Threshold
```
IF confidence < 60% → force decision to NEEDS_INFO
                     → append reason: "Confidence (45%) is below required threshold (60%)"
```
This prevents the system from making decisions it's not sure about.

### Rule 2: Evidence Required
```
IF no evidence chunks → force decision to NEEDS_INFO
                      → cap confidence at 30%
                      → append reason: "No supporting evidence found"
```
No evidence = no decision. Period.

### Rule 3: Valid Decision Values
```
IF decision not in (PASS, FAIL, NEEDS_INFO) → default to NEEDS_INFO
```
If the LLM outputs something weird like "MAYBE" or "PARTIALLY_COMPLIANT", it gets caught.

### Rule 4: Confidence Clamping
```
confidence = max(0.0, min(1.0, confidence))
```
Always between 0 and 1, no matter what the LLM returns.

### Why this matters

LLMs are probabilistic — they might hallucinate, be overconfident, or return malformed output. The governance layer is **deterministic code** that guarantees the output meets quality standards regardless of what the LLM does.

---

## 8. The Vector Database

**File: `app/vectorstore.py`**

### ChromaDB

ChromaDB is an embedded vector database — it runs in-process (no separate server needed) and persists to disk.

### How embeddings work

When you add a document chunk to ChromaDB:
1. ChromaDB passes the text through `all-MiniLM-L6-v2` (a local embedding model, ~23M parameters)
2. The model converts the text to a 384-dimensional vector
3. The vector is stored alongside the text and metadata

When you search:
1. Your query text gets embedded the same way
2. ChromaDB finds the stored vectors closest to your query vector
3. Distance is measured using **cosine similarity**

### Cosine similarity conversion

ChromaDB returns **cosine distance** (0 = identical, 2 = opposite). We convert to similarity:

```python
similarity = 1 - (distance / 2)
# distance=0.0 → similarity=1.0 (perfect match)
# distance=0.5 → similarity=0.75 (good match)
# distance=1.0 → similarity=0.5 (neutral)
# distance=2.0 → similarity=0.0 (opposite)
```

### HNSW Index

ChromaDB uses HNSW (Hierarchical Navigable Small World) for approximate nearest neighbor search. This makes searches fast even with thousands of chunks — it doesn't compare against every single vector, it navigates a graph structure to find close matches quickly.

---

## 9. The API Layer

**File: `app/api.py`**

### Singleton Pattern

The system uses lazy-initialized singletons for shared resources:

```python
_vector_store = None  # ChromaDB instance
_model_router = None  # LangChain ChatOpenAI wrapper
_retriever = None     # Query expansion + search
_agent = None         # Decision pipeline

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(persist_dir=settings.chroma_persist_dir)
    return _vector_store
```

This means:
- Resources are only created when first needed (not at import time)
- The same instances are reused across requests (efficient)
- `reset_instances()` clears everything for test isolation

### Endpoints

| Endpoint | What it does |
|----------|-------------|
| `POST /api/documents/upload` | Parse file → chunk text → store in ChromaDB |
| `GET /api/documents` | List all documents with chunk counts |
| `DELETE /api/documents/{id}` | Remove a document and all its chunks |
| `POST /api/query` | Full pipeline: expand → retrieve → decide → respond |
| `GET /api/health` | Document count + available models |
| `GET /api/logs` | Last 50 model usage entries |
| `GET /api/dashboard/data` | Aggregated stats for the dashboard |

### Validation

FastAPI + Pydantic handle input validation automatically:
- Empty question → 400 "Question cannot be empty"
- Unsupported file type → 400 "Unsupported file type"
- Empty file → 400 "Empty file"
- No extractable text → 400 "Document contains no extractable text"

---

## 10. The UI & Dashboard

### Web UI (`/ui`)

**File: `app/main.py` → `APP_UI_HTML`**

A single-page app served as inline HTML by FastAPI. No build step, no npm, no static files.

Features:
- **Document sidebar**: upload via drag-and-drop or click, see file names + chunk counts, delete
- **Query input**: textarea with Enter-to-submit
- **Decision display**: color-coded badge (green/red/orange), confidence bar, full answer, reasons list, evidence cards with citations, 5-step reasoning trace, model usage stats
- **Responsive**: works on mobile

All API calls go to the same server (`/api/*`) via `fetch()`.

### Dashboard (`/dashboard`)

**File: `app/main.py` → `DASHBOARD_HTML`**

System metrics page that auto-refreshes every 30 seconds:
- Document count, chunk count, query count
- Total API calls, tokens consumed, average latency
- Per-model breakdown (calls + tokens)
- Last 10 API calls with timestamps

---

## 11. Testing Strategy

### Unit Tests (58 tests)

| File | Tests | What's covered |
|------|-------|---------------|
| `tests/test_ingestion.py` | 10 | Text parsing, chunking edge cases, file format validation |
| `tests/test_vectorstore.py` | 9 | Add/search/delete/list, filtering, relevance ordering |
| `tests/test_router.py` | 9 | Model selection, prompt caching logic, log management |
| `tests/test_agent.py` | 13 | Context building, JSON parsing, governance rules, async analyze |
| `tests/test_api.py` | 12 | All endpoints via TestClient, validation, CRUD operations |

Key testing patterns:
- `conftest.py` sets fake env vars before any imports (so no real API key needed)
- Agent tests mock the router and retriever with `AsyncMock`
- API tests use `reset_instances()` to isolate state between tests
- Router tests only test pure logic (no LLM calls)

### E2E Tests (39 checks)

**File: `test_e2e.py`**

Runs against a live server with a real API key. Tests the complete flow:

1. Health check (3 checks)
2. Upload 3 example documents (8 checks)
3. Input validation — empty question, bad format, empty file (3 checks)
4. Compliance query — full decision validation (13 checks)
5. Threshold query — cross-document evidence (4 checks)
6. Multi-model routing — verifies 2+ models used (3 checks)
7. Document deletion (2 checks)

The e2e test validates everything from HTTP status codes to evidence structure to cross-document retrieval.

---

## 12. Deployment

### Render (Production)

```
GitHub push → Render detects change → Docker build → Deploy
```

- `Dockerfile`: Python 3.11-slim, pre-downloads the embedding model at build time, uses `$PORT` env var
- `render.yaml`: Defines the web service configuration
- ChromaDB persists to disk (survives restarts)
- Environment variable: `OPENROUTER_API_KEY`

### Vercel (Alternative)

- `vercel.json` + `api/index.py`: Serverless function entry point
- No persistent storage — ChromaDB data is lost on cold starts
- Good for quick demos, not production

### Docker

```bash
docker build -t policy-agent .
docker run -p 8000:8000 -e OPENROUTER_API_KEY=your-key policy-agent
```

---

## 13. File-by-File Breakdown

```
Agentic-RAG-system/
├── app/
│   ├── config.py          # Settings: models, thresholds, chunk sizes. Reads .env
│   ├── models.py          # Pydantic models: DecisionResult, QueryResponse, etc.
│   ├── ingestion.py       # File parsing (PDF/TXT/MD) + LangChain text splitting
│   ├── vectorstore.py     # ChromaDB wrapper: add, search, delete, list
│   ├── router.py          # LangChain ChatOpenAI, model selection, caching, logging
│   ├── retriever.py       # Query expansion + dual vector search
│   ├── agent.py           # 5-step decision pipeline + governance enforcement
│   ├── api.py             # FastAPI routes + lazy singletons
│   └── main.py            # App factory, CORS, UI HTML, dashboard HTML
├── tests/
│   ├── conftest.py        # Test env vars (fake API key, temp ChromaDB dir)
│   ├── test_ingestion.py  # 10 tests
│   ├── test_vectorstore.py# 9 tests
│   ├── test_router.py     # 9 tests
│   ├── test_agent.py      # 13 tests
│   └── test_api.py        # 12 tests
├── example_documents/
│   ├── payment_policy.md  # Company payment policy
│   ├── sample_invoice.txt # Invoice from TechSupply Corp
│   └── supplier_offer.txt # $75,000 offer from GlobalTech Solutions
├── frontend/
│   └── app.py             # Streamlit UI (optional, runs separately)
├── api/
│   └── index.py           # Vercel serverless entry point
├── test_e2e.py            # 39-check end-to-end test script
├── requirements.txt       # Production dependencies
├── requirements-dev.txt   # Test dependencies (pytest, etc.)
├── Dockerfile             # Docker build config
├── render.yaml            # Render deployment config
├── vercel.json            # Vercel deployment config
├── .env                   # API key (gitignored)
└── CLAUDE.md              # AI assistant context file
```

---

## Summary of the Two LLM Calls

For every user question, exactly **2 LLM calls** are made:

| # | Model | Purpose | Typical Tokens | Typical Latency |
|---|-------|---------|---------------|-----------------|
| 1 | Llama 3.3 70B | Query expansion | ~100 in / ~30 out | ~500ms |
| 2 | Gemini 2.0 Flash | Decision generation | ~1000 in / ~200 out | ~1500ms |

Total cost per query: fractions of a cent via OpenRouter's pay-per-token pricing.

The first call makes retrieval better. The second call makes the decision. Everything else is deterministic code.
