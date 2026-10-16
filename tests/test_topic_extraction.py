"""Unit and integration tests for Phase 3: Topic & Concept Extraction."""

from __future__ import annotations

import pytest

from app.pipeline.embeddings import get_embedding_service
from app.pipeline.topic_extractor import (
    clean_topic_phrase,
    extract_concepts_per_chunk,
    extract_topics_and_concepts,
)
from app.pipeline.vector_store import FAISSVectorStore
from app.schemas import Chunk
from app.services.process_video import load_session


@pytest.fixture
def sample_educational_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="chunk-001",
            topic=None,
            text="In this tutorial we're going to learn Python programming from scratch.",
            start_timestamp=0.0,
            end_timestamp=60.0,
            start_label="00:00",
            end_label="01:00",
            word_count=10,
        ),
        Chunk(
            chunk_id="chunk-002",
            topic=None,
            text="Now let's talk about variables. We use variables to temporarily store data in computer memory.",
            start_timestamp=65.0,
            end_timestamp=130.0,
            start_label="01:05",
            end_label="02:10",
            word_count=15,
        ),
        Chunk(
            chunk_id="chunk-003",
            topic=None,
            text="To declare a variable in Python, assign a label like age or price to a value in memory.",
            start_timestamp=131.0,
            end_timestamp=190.0,
            start_label="02:11",
            end_label="03:10",
            word_count=18,
        ),
        Chunk(
            chunk_id="chunk-004",
            topic=None,
            text="In this tutorial, we're going to talk about while loops in Python. We use while loops to repeat code.",
            start_timestamp=200.0,
            end_timestamp=270.0,
            start_label="03:20",
            end_label="04:30",
            word_count=18,
        ),
        Chunk(
            chunk_id="chunk-005",
            topic=None,
            text="As long as the loop condition remains true, the while block continues iterating repeatedly.",
            start_timestamp=271.0,
            end_timestamp=340.0,
            start_label="04:31",
            end_label="05:40",
            word_count=15,
        ),
        Chunk(
            chunk_id="chunk-006",
            topic=None,
            text="So, this is what we call a for loop. With this for loop we iterate over sequences and lists.",
            start_timestamp=350.0,
            end_timestamp=420.0,
            start_label="05:50",
            end_label="07:00",
            word_count=18,
        ),
    ]


def test_1_clean_topic_phrase() -> None:
    """Verify topic phrase normalization rules."""
    assert clean_topic_phrase("while loops in python") == "While Loops"
    assert clean_topic_phrase("for loop") == "For Loops"
    assert clean_topic_phrase("if statements in python") == "If Statements"
    assert clean_topic_phrase("the arithmetic operators that we have in python") == "Arithmetic Operators"
    assert clean_topic_phrase("lists in python") == "Lists"


def test_2_concept_extraction(sample_educational_chunks: list[Chunk]) -> None:
    """Verify TF-IDF keyphrase concept extraction per chunk."""
    concepts_list = extract_concepts_per_chunk(sample_educational_chunks)
    assert len(concepts_list) == len(sample_educational_chunks)

    # Variables chunk (chunk-002) should contain relevant concept terms
    var_concepts = concepts_list[1]
    assert any("variable" in c or "memory" in c or "data" in c for c in var_concepts)

    # Loop chunk (chunk-004) should contain loop-related terms
    loop_concepts = concepts_list[3]
    assert any("loop" in c or "repeat" in c or "while" in c for c in loop_concepts)


def test_3_topic_extraction_and_chunk_mapping(sample_educational_chunks: list[Chunk]) -> None:
    """Verify topic extraction associates chunks with topics and creates groups."""
    topics, enriched = extract_topics_and_concepts(sample_educational_chunks)

    assert len(enriched) == len(sample_educational_chunks)
    assert all(c.topic is not None for c in enriched)
    assert all(isinstance(c.concepts, list) for c in enriched)

    # Check that chunk-002 and chunk-003 both map to "Variables"
    assert enriched[1].topic == "Variables"
    assert enriched[2].topic == "Variables"

    # Check that chunk-004 and chunk-005 map to "While Loops"
    assert enriched[3].topic == "While Loops"
    assert enriched[4].topic == "While Loops"

    # Check that chunk-006 maps to "For Loops"
    assert enriched[5].topic == "For Loops"

    # Verify original fields preserved
    assert enriched[0].chunk_id == "chunk-001"
    assert enriched[0].start_timestamp == 0.0
    assert enriched[0].end_timestamp == 60.0
    assert enriched[0].start_label == "00:00"
    assert enriched[0].end_label == "01:00"


def test_4_timestamp_preservation_in_topic_groups(sample_educational_chunks: list[Chunk]) -> None:
    """Verify topic groups aggregate accurate start/end timestamps spanning member chunks."""
    topics, _ = extract_topics_and_concepts(sample_educational_chunks)

    # Find the "Variables" topic group
    var_group = next(t for t in topics if t.name == "Variables")
    assert var_group.chunk_ids == ["chunk-002", "chunk-003"]
    assert var_group.start_time == 65.0
    assert var_group.end_time == 190.0
    assert var_group.start_label == "01:05"
    assert var_group.end_label == "03:10"
    assert var_group.timestamp_range == "01:05 – 03:10"
    assert len(var_group.key_concepts) > 0

    # Find "While Loops" topic group
    while_group = next(t for t in topics if t.name == "While Loops")
    assert while_group.chunk_ids == ["chunk-004", "chunk-005"]
    assert while_group.start_time == 200.0
    assert while_group.end_time == 340.0
    assert while_group.timestamp_range == "03:20 – 05:40"


def test_5_compatibility_with_faiss(
    tmp_path, sample_educational_chunks: list[Chunk]
) -> None:
    """Verify that chunks with topics and concepts are indexed into FAISS and retrievable."""
    embedding_svc = get_embedding_service()
    topics, enriched = extract_topics_and_concepts(sample_educational_chunks)

    store = FAISSVectorStore(base_dir=tmp_path)
    texts = [c.text for c in enriched]
    embeddings = embedding_svc.embed_texts(texts)

    session_id = "test_phase3_faiss"
    store.build_and_save_index(session_id, "test_vid", enriched, embeddings)

    # Search for loops
    q_vec = embedding_svc.embed_query("How to iterate repeatedly?")
    results = store.search(session_id, q_vec, top_k=2)

    assert len(results) == 2
    top = results[0]
    assert top.topic in ("While Loops", "For Loops")
    assert isinstance(top.concepts, list)
    assert top.start_time > 0
    assert top.timestamp != ""
    assert top.score > 0.0


def test_6_session_isolation_with_topics() -> None:
    """Verify that topic extraction on one session does not interfere with another."""
    chunks_python = [
        Chunk(
            chunk_id="chunk-p1",
            text="In this tutorial we talk about while loops in Python.",
            start_timestamp=0.0,
            end_timestamp=10.0,
            start_label="00:00",
            end_label="00:10",
            word_count=9,
        )
    ]
    chunks_sql = [
        Chunk(
            chunk_id="chunk-s1",
            text="In this tutorial we talk about select statements in SQL databases.",
            start_timestamp=0.0,
            end_timestamp=10.0,
            start_label="00:00",
            end_label="00:10",
            word_count=10,
        )
    ]

    topics_p, _ = extract_topics_and_concepts(chunks_python)
    topics_s, _ = extract_topics_and_concepts(chunks_sql)

    assert any("Loop" in t.name for t in topics_p)
    assert not any("Select" in t.name for t in topics_p)

    assert any("Select" in t.name or "Sql" in t.name for t in topics_s)
    assert not any("Loop" in t.name for t in topics_s)


def test_7_integration_real_48_chunk_session() -> None:
    """Integration test: Load real session 3ec404e4f350, verify full topic extraction."""
    session = load_session("3ec404e4f350")
    assert session is not None
    assert session.chunk_count == 48

    # Verify topics were populated
    assert len(session.topics) >= 10
    topic_names = [t.name for t in session.topics]

    # Verify major lecture topics are present from transcript
    assert any("Variable" in name for name in topic_names)
    assert any("While Loop" in name for name in topic_names)
    assert any("List" in name for name in topic_names)
    assert any("For Loop" in name for name in topic_names)
    assert any("If Statement" in name for name in topic_names)

    # Verify all 48 chunks are accounted for in topic groups
    all_chunk_ids_in_topics = []
    for t in session.topics:
        assert t.start_time >= 0.0
        assert t.end_time >= t.start_time
        assert "–" in t.timestamp_range
        all_chunk_ids_in_topics.extend(t.chunk_ids)

    assert len(all_chunk_ids_in_topics) == 48
    # Chunks are preserved
    assert all(c.topic is not None for c in session.chunks)
    assert all(isinstance(c.concepts, list) for c in session.chunks)
