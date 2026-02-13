# Clarividex V2 Enhancements

This document describes the 6 enhancements introduced in V2 of the Clarividex platform.

---

## 1. Prompt Versioning

**Location:** `backend/app/prompts/`

YAML-based prompt template system with version control, allowing A/B testing and easy iteration on system prompts without code changes.

### How It Works

- Prompts are stored as YAML files in `backend/app/prompts/templates/`
- Each template has a `status` field: `active`, `testing`, or `archived`
- The prompt registry (`registry.py`) loads templates with `@lru_cache` for performance
- `prediction_engine.py` and `offline_model.py` use a `system_prompt` property that loads the active prompt from the registry, falling back to a hardcoded default

### Templates

| Template | Purpose | Status |
|----------|---------|--------|
| `prediction_v1.0.yaml` | Main prediction engine prompt | active |
| `prediction_v1.1.yaml` | Enhanced prediction prompt with improved calibration | testing |
| `offline_v1.0.yaml` | Offline/fallback model prompt | active |
| `chat_v1.0.yaml` | Chat assistant prompt | active |

### API Endpoints

- `GET /api/v1/prompts` — List all prompt templates
- `GET /api/v1/prompts/{name}/compare` — Compare two versions of a prompt

---

## 2. RAG-Powered Chat

**Location:** `backend/app/rag/`

Retrieval-Augmented Generation (RAG) pipeline that grounds the floating chatbot's answers in Clarividex's own documentation.

### Architecture

```
User Question → Methodology Keyword Detection → ChromaDB Vector Search → Context Injection → Claude/Ollama Response
```

### Components

- **Embeddings** (`embeddings.py`): Singleton HuggingFace `all-MiniLM-L6-v2` model
- **Ingestion** (`ingest.py`): Loads `*.md` files from `docs/`, chunks at 500 chars with 100-char overlap
- **Vector Store** (`vectorstore.py`): ChromaDB with persistent storage at `data/chroma/`
- **Service** (`service.py`): `RAGService` singleton with `ensure_indexed()` (called at startup) and `query()`

### Indexed Documents

- `docs/PREDICTION_ENGINE.md`
- `docs/METHODOLOGY.md`
- `docs/TECHNICAL_INDICATORS.md`
- `docs/README.md`

### Behavior

When a user asks a methodology-related question in the chat (detected by keyword matching), the system:
1. Queries ChromaDB for the top-3 most relevant document chunks
2. Injects them as context into the system prompt
3. The LLM generates a grounded answer citing the documentation

---

## 3. Output Guardrails

**Location:** `backend/app/guardrails/`

Four output guards that validate and sanitize every prediction and chat response before it reaches the user.

### Guards

| Guard | Description |
|-------|-------------|
| **Response Quality** | Ensures responses meet minimum length and contain substantive content |
| **PII Redaction** | Regex-based detection and redaction of emails, phone numbers, SSNs, and credit card numbers |
| **Financial Advice Detection** | Flags language that could be construed as personalized financial advice; injects disclaimer |
| **Probability Bounds** | Clamps prediction probabilities to 15-85% range (0.15-0.85 on the 0-1 scale) |

### Integration Points

- **`/predict` endpoint**: Guardrails run after prediction generation, before response
- **`/predict/stream` endpoint**: Guardrails run before emitting the `done` SSE event
- **`/chat` endpoint**: Guardrails run on both Claude and Ollama responses

---

## 4. SSE Streaming

**Location:** `backend/app/services/stream_service.py`, `frontend/src/lib/api.ts`

Server-Sent Events (SSE) streaming for real-time prediction progress updates.

### Backend

`PredictionStreamService.stream_prediction()` is an async generator that emits SSE events as each analysis phase completes:

| Event | Description |
|-------|-------------|
| `data_fetch` | Stock data retrieved from Yahoo Finance |
| `technical_analysis` | Technical indicators calculated |
| `sentiment_analysis` | News sentiment analyzed |
| `social_analysis` | Social media sentiment processed |
| `market_conditions` | Market conditions evaluated |
| `ai_reasoning` | Claude AI analysis complete |
| `probability_calculation` | Final probability computed |
| `done` | Full prediction response attached |
| `error` | Error occurred during processing |

### Frontend

- `api.streamPrediction()` uses `fetch()` + `ReadableStream` to parse SSE events
- `LoadingSkeleton` component maps SSE events to UI steps via `SSE_EVENT_TO_STEP`
- Falls back to timer-based animation when SSE events are unavailable

### API Endpoint

- `POST /api/v1/predict/stream` — Streams prediction progress as SSE events

---

## 5. Singleton Caching Fix

**Location:** `backend/app/api/routes.py`

Fixed the `/chat` endpoint to reuse the existing `prediction_engine.claude_client` singleton instead of creating a new `Anthropic()` client instance on every request.

### Before
```python
client = Anthropic(api_key=settings.anthropic_api_key)
```

### After
```python
client = prediction_engine.claude_client
```

This eliminates redundant client instantiation and ensures consistent API key management.

---

## 6. Evaluation Suite

**Location:** `backend/app/evals/`

Automated evaluation framework for measuring prediction quality, RAG accuracy, and guardrail effectiveness.

### Components

- **Golden Dataset** (`golden_dataset.py`): 18 test cases across 4 categories (prediction, RAG, guardrail, edge case)
- **Metrics** (`metrics.py`): 6 metrics — prediction quality, response completeness, keyword recall, latency, guardrail accuracy, probability bounds
- **Runner** (`runner.py`): Executes evaluations and produces reports
- **Experiment Tracker** (`experiment_tracker.py`): Persists eval results as JSON for comparison across runs

### Running Evaluations

```bash
# Via CLI
python -m backend.app.evals.runner

# Via API
GET /api/v1/eval/run
```

### Test Categories

| Category | Count | Description |
|----------|-------|-------------|
| prediction | 5 | Core prediction accuracy tests |
| rag | 4 | RAG retrieval and grounding tests |
| guardrail | 4 | Output guardrail enforcement tests |
| edge_case | 5 | Edge case handling tests |

---

## Infrastructure Changes

### PostgreSQL

Docker Compose now uses PostgreSQL 16 (Alpine) instead of SQLite:
- Healthcheck ensures the database is ready before the backend starts
- Persistent volume `postgres_data` for data durability

### New Dependencies

```
chromadb, langchain-chroma, langchain-huggingface, langchain-text-splitters,
langchain-core, langchain-community, sentence-transformers, pyyaml
```

### Docker

- `g++` added to Dockerfile for sentence-transformers C++ compilation
- `docs/` volume mounted for RAG document access
- `data/chroma/` directory created for ChromaDB persistence
