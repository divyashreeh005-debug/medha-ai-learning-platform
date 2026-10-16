"""Unit and integration tests for Phase 2: Embeddings, FAISS Vector Store, and Semantic Search."""

from __future__ import annotations

import numpy as np
import pytest

from app.pipeline.embeddings import EmbeddingService, get_embedding_service
from app.pipeline.vector_store import FAISSVectorStore, get_vector_store
from app.schemas import Chunk
from app.services.process_video import load_session
from app.services.search_service import index_session, search_session


@pytest.fixture(scope="module")
def embedding_svc() -> EmbeddingService:
    return get_embedding_service()


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="chunk-001",
            topic="Introduction",
            text="Welcome to Python programming. In this tutorial we cover basics, data types, and syntax.",
            start_timestamp=0.0,
            end_timestamp=60.0,
            start_label="00:00",
            end_label="01:00",
            word_count=13,
        ),
        Chunk(
            chunk_id="chunk-002",
            topic="Variables",
            text="Variables are used to store data in computer memory, such as numbers, strings, and booleans.",
            start_timestamp=65.0,
            end_timestamp=130.0,
            start_label="01:05",
            end_label="02:10",
            word_count=14,
        ),
        Chunk(
            chunk_id="chunk-003",
            topic="While and For Loops",
            text="Loops allow repeated execution of code. We use while loops and for loops to iterate over items.",
            start_timestamp=140.0,
            end_timestamp=220.0,
            start_label="02:20",
            end_label="03:40",
            word_count=16,
        ),
        Chunk(
            chunk_id="chunk-004",
            topic="Functions",
            text="Functions are reusable blocks of code that take parameters and return results using the def keyword.",
            start_timestamp=230.0,
            end_timestamp=310.0,
            start_label="03:50",
            end_label="05:10",
            word_count=16,
        ),
    ]


def test_1_embedding_model_loading(embedding_svc: EmbeddingService) -> None:
    """1. Verify embedding model loads successfully and returns expected dimension."""
    assert embedding_svc.model is not None
    assert embedding_svc.embedding_dim == 384


def test_2_embedding_generation(embedding_svc: EmbeddingService) -> None:
    """2. Verify embedding generation shape, dtype, and L2 normalization."""
    texts = ["Python tutorial", "Machine learning concepts"]
    embeddings = embedding_svc.embed_texts(texts)
    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == np.float32

    # Check L2 normalization (norm should be ~1.0)
    for vec in embeddings:
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-4

    # Single query vector
    query_vec = embedding_svc.embed_query("How do loops work?")
    assert query_vec.shape == (384,)
    assert query_vec.dtype == np.float32
    assert abs(np.linalg.norm(query_vec) - 1.0) < 1e-4


def test_3_faiss_index_creation(
    tmp_path, embedding_svc: EmbeddingService, sample_chunks: list[Chunk]
) -> None:
    """3. Verify FAISS index creation from chunks and embeddings."""
    store = FAISSVectorStore(base_dir=tmp_path)
    texts = [c.text for c in sample_chunks]
    embeddings = embedding_svc.embed_texts(texts)

    store.build_and_save_index(
        session_id="test_sess_01",
        video_id="vid_test_01",
        chunks=sample_chunks,
        embeddings=embeddings,
    )
    assert store.index_exists("test_sess_01")


def test_4_faiss_index_persistence(
    tmp_path, embedding_svc: EmbeddingService, sample_chunks: list[Chunk]
) -> None:
    """4. Verify FAISS index persistence to disk and reloading."""
    store = FAISSVectorStore(base_dir=tmp_path)
    texts = [c.text for c in sample_chunks]
    embeddings = embedding_svc.embed_texts(texts)

    session_id = "test_persist"
    store.build_and_save_index(session_id, "vid_persist", sample_chunks, embeddings)

    # Verify physical files exist
    session_dir = tmp_path / session_id
    assert (session_dir / "index.faiss").exists()
    assert (session_dir / "metadata.json").exists()

    # Create a fresh store instance without cache to verify loading from disk
    fresh_store = FAISSVectorStore(base_dir=tmp_path)
    assert fresh_store.index_exists(session_id)

    index, metadata = fresh_store.load_index(session_id)
    assert index.ntotal == len(sample_chunks)
    assert len(metadata) == len(sample_chunks)
    assert metadata[2]["chunk_id"] == "chunk-003"


def test_5_semantic_search(
    tmp_path, embedding_svc: EmbeddingService, sample_chunks: list[Chunk]
) -> None:
    """5. Verify semantic search retrieves relevant chunks without exact keyword match."""
    store = FAISSVectorStore(base_dir=tmp_path)
    texts = [c.text for c in sample_chunks]
    embeddings = embedding_svc.embed_texts(texts)
    session_id = "test_search"
    store.build_and_save_index(session_id, "vid_search", sample_chunks, embeddings)

    # Query about repetition/iteration (does not mention "Loops" directly in query)
    query = "How can I repeat instructions multiple times in code?"
    q_vec = embedding_svc.embed_query(query)
    results = store.search(session_id, q_vec, top_k=2)

    assert len(results) == 2
    # The loops chunk should be the top match
    assert results[0].chunk_id == "chunk-003"
    assert "repeated execution" in results[0].text
    assert results[0].score > results[1].score


def test_6_metadata_mapping_and_timestamp_preservation(
    tmp_path, embedding_svc: EmbeddingService, sample_chunks: list[Chunk]
) -> None:
    """6 & 7. Verify metadata mapping connecting vectors back to chunk IDs and timestamps."""
    store = FAISSVectorStore(base_dir=tmp_path)
    texts = [c.text for c in sample_chunks]
    embeddings = embedding_svc.embed_texts(texts)
    session_id = "test_meta"
    store.build_and_save_index(session_id, "vid_meta", sample_chunks, embeddings)

    q_vec = embedding_svc.embed_query("saving numbers in memory")
    results = store.search(session_id, q_vec, top_k=1)

    top = results[0]
    assert top.chunk_id == "chunk-002"
    assert top.topic == "Variables"
    assert top.start_time == 65.0
    assert top.end_time == 130.0
    assert top.timestamp == "01:05 – 02:10"
    assert top.score > 0.0


def test_8_session_isolation(
    tmp_path, embedding_svc: EmbeddingService
) -> None:
    """8. Verify session isolation: search on Session A never returns Session B chunks."""
    store = FAISSVectorStore(base_dir=tmp_path)

    chunks_a = [
        Chunk(
            chunk_id="chunk-a1",
            text="Java is an object oriented statically typed language.",
            start_timestamp=0.0,
            end_timestamp=10.0,
            start_label="00:00",
            end_label="00:10",
            word_count=8,
        )
    ]
    chunks_b = [
        Chunk(
            chunk_id="chunk-b1",
            text="Python is a dynamically typed interpreted programming language.",
            start_timestamp=0.0,
            end_timestamp=10.0,
            start_label="00:00",
            end_label="00:10",
            word_count=8,
        )
    ]

    emb_a = embedding_svc.embed_texts([c.text for c in chunks_a])
    emb_b = embedding_svc.embed_texts([c.text for c in chunks_b])

    store.build_and_save_index("session_java", "vid_java", chunks_a, emb_a)
    store.build_and_save_index("session_python", "vid_python", chunks_b, emb_b)

    q_vec = embedding_svc.embed_query("programming language features")

    results_a = store.search("session_java", q_vec, top_k=5)
    assert all(r.chunk_id.startswith("chunk-a") for r in results_a)
    assert not any("Python" in r.text for r in results_a)

    results_b = store.search("session_python", q_vec, top_k=5)
    assert all(r.chunk_id.startswith("chunk-b") for r in results_b)
    assert not any("Java" in r.text for r in results_b)


def test_9_integration_existing_phase1_session() -> None:
    """Integration Test: Load real Phase 1 session, build/search FAISS, verify retrieval & timestamps."""
    session_id = "3ec404e4f350"
    session = load_session(session_id)
    assert session is not None
    assert session.chunk_count == 48

    # Search for loops in the real Python tutorial
    query = "Where does the instructor explain while loops and iteration?"
    resp = search_session(session_id=session_id, query=query, top_k=3)

    assert resp.session_id == session_id
    assert resp.total_results == 3
    assert len(resp.results) == 3

    # Verify top result contains relevant loop explanation
    top_result = resp.results[0]
    assert top_result.chunk_id
    assert top_result.score > 0.3
    assert top_result.start_time > 0.0
    assert top_result.end_time > top_result.start_time
    assert "–" in top_result.timestamp
    assert "loop" in top_result.text.lower() or "while" in top_result.text.lower()
