"""Context-Aware AI Tutor Orchestration Service for Medhā (Phase 6).

Coordinates query embedding, FAISS semantic retrieval, out-of-context relevance
thresholding, RAG prompt construction, and local Qwen inference with source attribution.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatSource,
    SearchResultChunk,
)
from app.services.llm_client import LLMClientError, call_llm_chat
from app.services.process_video import load_session
from app.services.rag_service import FALLBACK_UNAVAILABLE_MSG, build_tutor_prompt
from app.services.search_service import SearchServiceError, SessionNotFoundError, search_session

logger = logging.getLogger(__name__)

# Minimum cosine similarity score for retrieved chunks (all-MiniLM-L6-v2 + IndexFlatIP).
# Queries scoring below this are deemed out-of-context for the lecture.
RELEVANCE_THRESHOLD: float = 0.20


class TutorServiceError(Exception):
    """Base exception for AI tutor operations."""


def ask_tutor(
    session_id: str,
    question: str,
    top_k: int = 5,
    chat_history: list[ChatMessage] | None = None,
    language: str = "en",
) -> ChatResponse:
    """Orchestrate lecture-grounded RAG query answering for a student.
    
    1. Validates session and non-empty query.
    2. Performs dense vector search against session's FAISS index.
    3. Evaluates retrieval relevance threshold.
    4. Constructs strictly bounded prompt with lecture context.
    5. Queries local Qwen3-8B model.
    6. Returns grounded response with timestamped source attribution.
    """
    clean_question = question.strip()
    if not clean_question:
        raise TutorServiceError("Question cannot be empty.")

    session = load_session(session_id)
    if session is None:
        raise SessionNotFoundError(f"Session '{session_id}' not found.")

    # 1. Retrieve relevant transcript chunks via FAISS
    try:
        search_res = search_session(session_id=session_id, query=clean_question, top_k=top_k)
    except SessionNotFoundError:
        raise
    except SearchServiceError as exc:
        raise TutorServiceError(f"Search retrieval failed: {exc}") from exc

    results = search_res.results

    # 2. Out-of-context / low-relevance guardrail
    if not results or results[0].score < RELEVANCE_THRESHOLD:
        logger.info(
            "Query '%s' for session %s scored below threshold (top score: %s). Returning fallback.",
            clean_question,
            session_id,
            results[0].score if results else "None",
        )
        return ChatResponse(
            session_id=session_id,
            question=clean_question,
            answer=FALLBACK_UNAVAILABLE_MSG,
            sources=[],
            primary_topic=None,
            primary_timestamp_range=None,
            primary_start_time=None,
        )

    # 3. Format sources
    sources: list[ChatSource] = [
        ChatSource(
            chunk_id=c.chunk_id,
            topic=c.topic,
            timestamp=c.timestamp,
            start_time=c.start_time,
            end_time=c.end_time,
            text=c.text,
            score=c.score,
        )
        for c in results
    ]

    top_chunk = results[0]
    primary_topic = top_chunk.topic
    primary_timestamp_range = top_chunk.timestamp
    primary_start_time = top_chunk.start_time

    # 4. Optional Concept Knowledge Graph context enrichment
    graph_context: Optional[str] = None
    try:
        from app.services.concept_graph_service import load_concept_graph
        graph = load_concept_graph(session_id)
        if graph and graph.relationships:
            relevant_concepts = {c.topic.lower() for c in results if c.topic}
            graph_lines = []
            for r in graph.relationships:
                if r.relation.value != "follows" and (
                    r.source.lower() in relevant_concepts or r.target.lower() in relevant_concepts
                ):
                    graph_lines.append(f"- Concept '{r.source}' {r.relation.value} '{r.target}' (Evidence: {r.evidence})")
            if graph_lines:
                graph_context = "\n".join(graph_lines[:5])
    except Exception:
        pass

    # 5. Build grounded RAG prompt
    messages = build_tutor_prompt(
        question=clean_question,
        chunks=results,
        chat_history=chat_history,
        graph_context=graph_context,
        language=language,
    )

    # 5. Invoke Qwen3-8B LLM
    try:
        raw_answer = call_llm_chat(messages, max_tokens=600)
    except LLMClientError as exc:
        raise TutorServiceError(
            f"Medhā's AI tutor is currently unavailable because the local language model is not running. ({exc})"
        ) from exc

    answer = raw_answer.strip()

    # If Qwen itself determined the context lacks sufficient information, omit sources
    if FALLBACK_UNAVAILABLE_MSG.lower() in answer.lower():
        sources = []
        primary_topic = None
        primary_timestamp_range = None
        primary_start_time = None

    return ChatResponse(
        session_id=session_id,
        question=clean_question,
        answer=answer,
        sources=sources,
        primary_topic=primary_topic,
        primary_timestamp_range=primary_timestamp_range,
        primary_start_time=primary_start_time,
    )
