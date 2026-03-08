# Policy-Aware AI Decision Agent

## What This Is
A RAG-powered AI decision agent that ingests documents, retrieves relevant evidence, and produces structured PASS/FAIL/NEEDS_INFO decisions with citations, confidence scores, and reasoning traces. Built for the AI-Pass 24-hour engineering challenge.

## Architecture

```
User Question → Query Expansion (Llama 3.3 70B) → Dual Vector Search (ChromaDB)
    → Evidence Assembly → Decision Agent (Gemini 2.0 Flash) → Structured Decision
```

## Tech Stack
- **Framework**: FastAPI (Python 3.11+)
- **LLM**: LangChain `ChatOpenAI` via OpenRouter (multi-model)
- **Text Splitting**: LangChain `RecursiveCharacterTextSplitter`
- **Prompts**: LangChain `ChatPromptTemplate`
- **Vector DB**: ChromaDB (embedded, cosine similarity, `all-MiniLM-L6-v2` embeddings)
- **Validation**: Pydantic v2 models for all inputs/outputs
- **Tests**: pytest + pytest-asyncio (58 unit) + custom e2e script (39 checks)

## Key Files

| File | Purpose |
|------|---------|
| `app/config.py` | Pydantic settings with `@lru_cache`, reads `.env` |
| `app/models.py` | All Pydantic models: `DecisionResult`, `QueryResponse`, `ModelUsage`, etc. |
| `app/ingestion.py` | PDF/TXT/MD parsing + LangChain `RecursiveCharacterTextSplitter` chunking |
| `app/vectorstore.py` | ChromaDB wrapper: add, search, delete, list. Cosine similarity = `1 - (distance / 2)` |
| `app/router.py` | LangChain `ChatOpenAI` via OpenRouter, model selection, TTL cache, prompt caching, usage logging |
| `app/retriever.py` | Query expansion (secondary model) + dual vector search (expanded + original) |
| `app/agent.py` | 5-step decision pipeline with `ChatPromptTemplate`, JSON parsing, governance enforcement |
| `app/api.py` | FastAPI routes. Lazy singletons. `reset_instances()` for test isolation |
| `app/main.py` | App factory, CORS, HTML UI at `/ui`, dashboard at `/dashboard` |
| `test_e2e.py` | Full integration test (39 checks) against live server |
| `tests/` | 58 unit tests across 5 test files |

## Models (via OpenRouter)
- **Primary** (`google/gemini-2.0-flash-001`): Decision generation, complex analysis
- **Secondary** (`meta-llama/llama-3.3-70b-instruct`): Query expansion, simple tasks
- Free `:free` models on OpenRouter require privacy opt-in at openrouter.ai/settings/privacy

## Governance Rules (deterministic, not LLM-dependent)
- Confidence < 60% → force `NEEDS_INFO`
- No evidence → force `NEEDS_INFO`, cap confidence at 30%
- Invalid decision values → default to `NEEDS_INFO`
- Confidence always clamped to [0, 1]

## Configuration
- All settings in `app/config.py` via pydantic-settings, overridable by `.env`
- Key settings: `chunk_size=500`, `chunk_overlap=50`, `top_k_retrieval=10`, `top_k_final=7`, `confidence_threshold=0.60`

## Deployment
- **Render** (primary): Docker-based, `render.yaml` + `Dockerfile`, deploy hook available
- **Vercel**: `vercel.json` + `api/index.py` (no persistent storage — ephemeral)
- **GitHub**: https://github.com/joshuarebo/Agentic-RAG-system
- **Live URL**: https://agentic-rag-system-ilwr.onrender.com

## Testing Notes
- Python 3.13 on Windows 11 (bash shell via Git Bash)
- Use `datetime.now(timezone.utc)` not `datetime.utcnow()` (deprecated in 3.12+)
- pytest-asyncio in STRICT mode — all async tests need `@pytest.mark.asyncio`
- Mock router must return real `ModelUsage` instances (Pydantic validates, MagicMock fails)
- `ChatPromptTemplate` system prompts with literal `{` `}` must be escaped as `{{` `}}`
- E2e test "Evidence cites payment policy" is non-deterministic — bumped `top_k_final` from 5→7 to fix
- Clear `chroma_data/` before clean e2e runs to avoid stale data interference

## Don't
- Don't commit `.env` (contains OpenRouter API key)
- Don't use `httpx` directly for LLM calls — use LangChain `ChatOpenAI` via `router.py`
- Don't change the `complete()` return format `{"content": str, "model_usage": ModelUsage}` — everything depends on it
- Don't remove `_apply_prompt_caching` from router — it's tested independently and ready for Claude models
