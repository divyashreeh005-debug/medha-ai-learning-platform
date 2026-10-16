"""Semantic search service connecting query embedding to FAISS retrieval (Phase 2)."""

from __future__ import annotations

import logging

from app.pipeline.embeddings import get_embedding_service
from app.pipeline.vector_store import get_vector_store
from app.schemas import ProcessResponse, SearchResponse, SearchResultChunk
from app.services.process_video import load_session

logger = logging.getLogger(__name__)


class SearchServiceError(Exception):
    """Base error for search service operations."""


class SessionNotFoundError(SearchServiceError):
    """Raised when a requested session is not found."""


def index_session(session: ProcessResponse) -> None:
    """Build and persist the FAISS index for all chunks in a session."""
    if not session.chunks:
        logger.warning("Session %s has no chunks to index.", session.session_id)
        return

    embedding_service = get_embedding_service()
    vector_store = get_vector_store()

    texts = [chunk.text for chunk in session.chunks]
    embeddings = embedding_service.embed_texts(texts)

    vector_store.build_and_save_index(
        session_id=session.session_id,
        video_id=session.video_id,
        chunks=session.chunks,
        embeddings=embeddings,
    )
    logger.info("Indexed %d chunks for session %s", len(session.chunks), session.session_id)


def search_session(session_id: str, query: str, top_k: int = 5) -> SearchResponse:
    """Perform semantic search against a lecture session using dense embeddings and FAISS.

    If the session exists but has not yet been indexed into FAISS (e.g. from Phase 1),
    the index is automatically constructed on demand.
    """
    clean_query = query.strip()
    if not clean_query:
        raise SearchServiceError("Query cannot be empty.")

    session = load_session(session_id)
    if session is None:
        raise SessionNotFoundError(f"Session '{session_id}' not found.")

    vector_store = get_vector_store()

    # Automatically build index if not yet present
    if not vector_store.index_exists(session_id):
        logger.info("Index for session %s does not exist on disk. Building now...", session_id)
        index_session(session)

    embedding_service = get_embedding_service()
    query_vector = embedding_service.embed_query(clean_query)

    results = vector_store.search(
        session_id=session_id,
        query_vector=query_vector,
        top_k=top_k,
    )

    # Enrich topic and concepts from session metadata if missing from older index metadata
    chunk_topic_map: dict[str, str] = {}
    chunk_concepts_map: dict[str, list[str]] = {}
    for t in session.topics:
        for cid in t.chunk_ids:
            chunk_topic_map[cid] = t.name
            chunk_concepts_map[cid] = t.key_concepts

    for r in results:
        if not r.topic and r.chunk_id in chunk_topic_map:
            r.topic = chunk_topic_map[r.chunk_id]
        if not r.concepts and r.chunk_id in chunk_concepts_map:
            r.concepts = chunk_concepts_map[r.chunk_id]

    return SearchResponse(
        session_id=session_id,
        query=clean_query,
        total_results=len(results),
        results=results,
    )
