# Medhā

AI-powered video-to-interactive learning platform.

This repository implements:
- **Phase 1 — Core Pipeline**: YouTube URL → real transcript → cleaning → semantic chunking with timestamp preservation.
- **Phase 2 — Embeddings & FAISS Retrieval**: Dense embeddings (`sentence-transformers/all-MiniLM-L6-v2`) → FAISS vector index (`IndexFlatIP`) → semantic search with timestamps.
- **Phase 3 — Topic & Concept Extraction**: Discourse cue detection & TF-IDF keyphrase analysis → concept-tagged chunks → temporal topic groups.
- **Phase 4 — Qwen Educational Content Generation**: Grounded generation via local `qwen/qwen3-8b` (LM Studio/Ollama) → Summary, Explain Simply, Real-world Analogy, Key Takeaways with persistent disk caching.
- **Phase 5 — Interactive Learning Dashboard**: Full student-facing dashboard with chronological topic navigation (`Topic X of Y`), direct YouTube timestamp jump links (`⏱ Jump to MM:SS`), grounded learning content cards, and focused topic transcript segments.
- **Phase 6 — Context-Aware AI Tutor / Grounded RAG**: Lecture-grounded question answering via `all-MiniLM-L6-v2` dense retrieval + `qwen/qwen3-8b` with strict anti-hallucination guardrails, source attribution, out-of-context fallback, and interactive conversational UI.
- **Phase 8 — Concept Intelligence & Knowledge Graph**: Canonical concept extraction, evidence-backed relationship discovery (`prerequisite`, `related_to`, `used_with`, `depends_on`, `follows`), native Streamlit Graphviz concept map, prerequisite inspector, and RAG context enrichment.

Later phases add quiz/MCQ generation, flashcards, and Whisper video upload.

## Setup

```powershell
cd C:\Users\divya\Documents\PROJECTS\Medha
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

## Local LLM Setup (Phase 4–6)

Ensure LM Studio (or Ollama) is running with `qwen/qwen3-8b` or compatible model served on `http://127.0.0.1:1234/v1`.

## Running the Application

Terminal 1 — API:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2 — UI:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run ui/streamlit_app.py
```

## API Usage

### 1. Process a lecture:

```powershell
curl -X POST http://127.0.0.1:8000/api/process -H "Content-Type: application/json" -d "{\"youtube_url\":\"https://www.youtube.com/watch?v=kqtD5dpn9C8\"}"
```

### 2. Semantic search across a lecture session:

```powershell
curl -X POST http://127.0.0.1:8000/api/search -H "Content-Type: application/json" -d "{\"session_id\":\"3ec404e4f350\",\"query\":\"Where does the instructor explain loops?\",\"top_k\":3}"
```

### 3. Topic extraction & retrieval:

```powershell
curl http://127.0.0.1:8000/api/sessions/3ec404e4f350/topics
```

### 4. Check LLM runtime status:

```powershell
curl http://127.0.0.1:8000/api/llm/status
```

### 5. Generate educational content for a topic:

```powershell
curl -X POST http://127.0.0.1:8000/api/topics/generate -H "Content-Type: application/json" -d "{\"session_id\":\"3ec404e4f350\",\"topic_id\":\"topic-003\"}"
```

### 6. Ask Medhā AI Tutor (Lecture-Grounded RAG):

```powershell
curl -X POST http://127.0.0.1:8000/api/chat -H "Content-Type: application/json" -d "{\"session_id\":\"3ec404e4f350\",\"question\":\"Why do we use variables in Python?\",\"top_k\":5}"
```

### 7. Retrieve Concept Intelligence & Knowledge Graph:

```powershell
curl http://127.0.0.1:8000/api/sessions/3ec404e4f350/concept-graph
```

## Tests

Run the complete test suite (Phase 1 through Phase 8 — 85 tests):

```powershell
pytest -v
```
