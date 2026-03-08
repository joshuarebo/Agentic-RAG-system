# Policy-Aware AI Decision Agent

A mini AI-Pass agent system demonstrating Retrieval-Augmented Generation (RAG), AI agent orchestration, deterministic validation, and explainable results.

## Architecture

```
User Question
     |
[Query Expansion] -- LLM call 1 (secondary model, fast)
     |
[Vector Retrieval] -- ChromaDB similarity search (dual query)
     |
[Evidence Assembly] -- top-K chunks with citations
     |
[Decision Agent] -- LLM call 2 (primary model)
     |
Structured Decision (PASS / FAIL / NEEDS_INFO)
  + reasons, evidence, confidence, reasoning steps
```

### Core Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| API | FastAPI | REST endpoints for documents and queries |
| Vector DB | ChromaDB | Open-source embedded vector storage |
| Embeddings | all-MiniLM-L6-v2 | Local embeddings via ChromaDB (no API cost) |
| LLM Routing | OpenRouter | Multi-model access (Claude, Gemini, etc.) |
| Frontend | Streamlit | Simple demo UI |

### RAG Pipeline

1. **Document Ingestion** -- PDF/TXT/MD parsed and split into semantic chunks respecting paragraph boundaries
2. **Embedding** -- Chunks embedded via all-MiniLM-L6-v2 (local, runs on CPU)
3. **Query Expansion** -- User query expanded by a fast LLM for better retrieval coverage
4. **Dual Retrieval** -- Both expanded and original queries searched against ChromaDB
5. **Evidence Assembly** -- Top chunks ranked by cosine similarity with full source citations

### Decision Logic

The agent follows a **Reason -> Evaluate -> Execute -> Explain** flow:

1. **Identify Intent** -- Understand what the user is asking
2. **Retrieve Documents** -- Vector search with query expansion
3. **Evaluate Evidence** -- Assess relevance quality
4. **Analyze Compliance** -- Compare evidence against requirements
5. **Generate Decision** -- Structured JSON with governance enforcement

**Governance rules:**
- Confidence below 60% automatically becomes `NEEDS_INFO`
- Decisions without evidence are rejected
- All claims must cite retrieved chunks
- Invalid decision values default to `NEEDS_INFO`

### Model Routing

Requests are routed to different models based on task complexity:

- **Primary model** (Gemini 2.0 Flash via OpenRouter): Decision generation, complex analysis
- **Secondary model** (Llama 3.3 70B via OpenRouter): Query expansion, simple tasks
- **Prompt caching**: When Claude models are selected, requests use `cache_control` on system prompts to reduce costs by up to 90%. A TTL response cache also deduplicates identical queries across all providers.

All calls log: model used, input/output tokens, latency, cached tokens.

### Decision Output Format

```json
{
  "decision": "PASS | FAIL | NEEDS_INFO",
  "reasons": ["..."],
  "evidence": [{"document_source": "...", "chunk_index": 0, "content": "...", "relevance_score": 0.92}],
  "confidence": 0.85,
  "reasoning_steps": [{"step_number": 1, "action": "identify_intent", "detail": "...", "result": "..."}]
}
```

## Setup

### Prerequisites

- Python 3.11+
- An [OpenRouter](https://openrouter.ai/) API key

### Local Development

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/Agentic-RAG-system.git
cd Agentic-RAG-system

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY

# Run backend
uvicorn app.main:app --reload

# (Optional) Run frontend in another terminal
pip install streamlit
streamlit run frontend/app.py
```

API docs available at: `http://localhost:8000/docs`

### Docker

```bash
docker build -t policy-agent .
docker run -p 8000:8000 -e OPENROUTER_API_KEY=your-key policy-agent
```

### Deploy to Render (Recommended)

1. Push to GitHub
2. Go to [render.com](https://render.com) and create a new **Web Service**
3. Connect your GitHub repo
4. Select **Docker** as the environment
5. Add environment variable: `OPENROUTER_API_KEY`
6. Deploy -- the `render.yaml` handles the rest

### Deploy to Vercel

Vercel config is included (`vercel.json` + `api/index.py`). Note: Vercel serverless has no persistent storage, so uploaded documents are lost on cold starts. Best for demo purposes.

```bash
npm i -g vercel
vercel --prod
# Set OPENROUTER_API_KEY in Vercel dashboard -> Settings -> Environment Variables
```

### Running Tests

```bash
# Unit tests (58 tests)
pip install -r requirements-dev.txt
pytest tests/ -v

# End-to-end test (39 checks, requires running server + API key)
uvicorn app.main:app &
python test_e2e.py
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/documents/upload` | Upload PDF/TXT/MD for indexing |
| GET | `/api/documents` | List indexed documents |
| DELETE | `/api/documents/{doc_id}` | Remove a document |
| POST | `/api/query` | Submit a question for analysis |
| GET | `/api/health` | Health check |
| GET | `/api/logs` | Model usage logs |

## Example Usage

1. Upload the example documents from `example_documents/`
2. Ask: "Is this invoice compliant with the payment policy?"
3. The agent retrieves relevant policy sections and invoice details, then produces a structured decision

## Limitations

- **Embedding model**: all-MiniLM-L6-v2 is a general-purpose model; domain-specific fine-tuning would improve accuracy
- **Single vector store**: No hybrid search (BM25 + vector) -- adding keyword search would improve retrieval
- **No cross-encoder re-ranking**: Using score-based ranking; a cross-encoder would improve precision
- **Context window**: Large documents may lose information during chunking
- **No authentication**: API endpoints are open (add auth for production)
- **In-memory logs**: Model usage logs are not persisted across restarts

## Scaling Considerations

- **Vector DB**: Migrate from embedded ChromaDB to managed Qdrant/Pinecone for horizontal scaling
- **Embeddings**: Use GPU-accelerated embedding service or API-based embeddings
- **Caching**: Add Redis for response caching and session management
- **Re-ranking**: Add cross-encoder re-ranking stage for better precision
- **Hybrid search**: Combine vector search with BM25 keyword search
- **Observability**: Add Langfuse/Datadog for prompt tracing and drift detection
- **Auth**: Add API key authentication and rate limiting
- **Queue**: Use Celery/Redis for async document ingestion at scale
- **Evaluation**: Add automated RAG evaluation suite (relevance, faithfulness, answer quality)
