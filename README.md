
# Medhā — AI-Powered Video-to-Knowledge Learning Platform

Medhā is an AI-assisted learning platform designed to transform educational videos into structured learning material. It aims to help learners understand concepts, revise important facts, explore relationships between ideas, and ask questions grounded in source content.

> **Status note:** Verify all features, commands, dependencies, and test results against the current repository before treating them as confirmed.

## Features

- **Transcript processing:** Retrieve available YouTube captions, clean text, and split transcripts into manageable chunks.
- **AI-generated learning content:** Generate summaries, notes, and explanations from educational material.
- **Ask Medhā:** Answer questions using relevant source content and retrieval-augmented generation (RAG).
- **Knowledge graph:** Extract concepts and their relationships and display them visually.
- **Topic-based flashcards:** Generate revision questions about concepts, facts, rules, mechanisms, and examples taught in the video.
- **Multilingual learning:** Support non-English source material with English as the default output language.
- **Source traceability:** Connect generated content to transcript chunks and genuine timestamps when available.
- **Caching:** Reuse processed content and model outputs to reduce repeated computation.

Feature availability should be checked in the current codebase.

## Technology Stack

The project plan has referenced the following technologies. Retain only those actually used in this version.

| Component | Technology |
|---|---|
| User interface | Streamlit |
| Local LLM inference | Qwen through LM Studio, if still configured |
| Embeddings | Sentence Transformers, including `all-MiniLM-L6-v2` |
| Semantic search | FAISS |
| Transcript extraction | `youtube-transcript-api` or the implemented provider |
| Language | Python |
| Graph data | JSON/session-based storage, if applicable |
| Audio processing | FFmpeg/Whisper, if enabled |

## Architecture

1. **Input:** Receive a supported video URL or other implemented input.
2. **Transcript acquisition:** Retrieve captions or use the configured transcription path.
3. **Normalization:** Clean the transcript, preserve genuine timestamps, and handle missing timestamps safely.
4. **Chunking:** Split content by sentences, paragraphs, semantic boundaries, and size limits.
5. **Language normalization:** Translate source chunks into English when required, retaining the original transcript.
6. **Indexing:** Generate embeddings and index transcript chunks for semantic retrieval.
7. **Concept extraction:** Identify concepts, definitions, rules, examples, and relationships.
8. **Learning generation:** Generate summaries, flashcards, graph details, and question answers.
9. **Validation:** Validate generated outputs and show source evidence where possible.

## Installation

### Prerequisites

- A compatible Python version.
- Git.
- LM Studio and a compatible Qwen model if local inference is enabled.
- Additional dependencies documented by the repository, such as FFmpeg if required.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/medha.git
cd medha
```

Replace the URL with the actual repository URL.

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt**

```bat
py -m venv .venv
.venv\Scripts\activate.bat
```

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

If the repository contains `requirements.txt`:

```bash
pip install -r requirements.txt
```

If it uses another dependency manager, follow the existing project configuration.

### 4. Configure environment variables

Create a local `.env` file only if the application uses environment variables. Use `.env.example` as a template.

Example configuration — **verify the actual variable names in the code**:

```dotenv
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_MODEL=qwen/qwen3-8b
LLM_TIMEOUT_SECONDS=180
LLM_TEMPERATURE=0.2
CONTENT_CACHE_DIR=./content_cache
OUTPUT_LANGUAGE=en
```

These are illustrative settings, not a guarantee of the current application's configuration. Use the exact names and model identifier expected by the repository.

Never commit `.env` files, API keys, passwords, tokens, or private data.

### 5. Configure LM Studio

If Qwen through LM Studio is enabled:

1. Open LM Studio.
2. Load a compatible Qwen model.
3. Start the local inference server.
4. Verify the base URL, model identifier, context length, and other configuration.
5. Keep the server running while using Medhā.

### 6. Run the application

Find the actual Streamlit entry point in the repository. If the file is `app.py`:

```bash
streamlit run app.py
```

Replace `app.py` if the project uses a different entry-point filename.

## Testing

If the project uses pytest:

```bash
python -m pytest -q
```

Run the tests that actually exist in the repository. Record failures honestly; do not claim that tests pass without running them.

### Recommended regression tests

- Timestamped English lecture.
- Another supported non-English lecture.
- One-shot video without timestamps.
- Long transcript.
- Transcript containing noisy or repetitive text.
- Question unrelated to the video.
- Flashcards validated against source evidence.
- Knowledge-graph node selection and details display.
- Cache reuse and cache invalidation.

## Known Issues and Planned Improvements

The following items are part of the current improvement plan. Update this section as each issue is verified and resolved.

1. **Timestamp-free videos:** Prevent missing timestamps from causing an incorrect default `Introduction` topic.
2. **English learning output:** Translate supported non-English source content into English by default while retaining the original transcript.
3. **Topic-grounded flashcards:** Generate questions about actual lesson content rather than generic questions about the video or speaker.
4. **Interactive knowledge graph:** Allow users to click graph nodes and inspect concept descriptions, relationships, and source evidence.
5. **Qwen optimization:** Reduce unnecessary context, repeated model calls, and token usage through bounded prompts, caching, and structured output.
6. **Regression testing:** Verify that improvements work across all learning features and transcript types.

## Security and Repository Hygiene

- Never commit secrets or credentials.
- Include `.env.example` with placeholders only.
- Exclude virtual environments, local model weights, temporary files, and regenerable caches where appropriate.
- Review `git status` before committing.
- Do not expose local inference servers to untrusted networks without suitable safeguards.

## Development Workflow

1. Inspect the existing implementation before modifying it.
2. Work on a separate feature branch.
3. Make focused changes and preserve working functionality.
4. Add or update tests.
5. Run relevant tests and record actual results.
6. Update documentation when configuration or behavior changes.
7. Submit a pull request with the changes and known limitations.

## License

No license is specified here. Add one after the project owner chooses the appropriate license.
