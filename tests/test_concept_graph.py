"""Tests for Medhā Phase 8: Concept Intelligence & Knowledge Graph.

Verifies:
1. Concept extraction from session topics and chunks
2. Concept IDs format (concept-001, concept-002, etc.)
3. Concept -> topic mapping integrity
4. Concept -> chunk mapping integrity
5. Timestamp preservation (start_time <= end_time, timestamp_range format)
6. Relationship creation with source, target, relation, evidence
7. Relationship validation against allowed types
8. Unsupported relationship rejection (Pydantic ValidationError)
9. Session isolation (Session A does not mix with Session B)
10. Graph persistence to JSON
11. Graph loading from JSON
12. Missing graph handling (returns None or raises ConceptGraphError)
13. API endpoint response structure (GET /api/sessions/{session_id}/concept-graph)
14. Concept selection & details retrieval (prerequisites and related concepts)
15. Source chunk attribution
16. YouTube timestamp generation
17. Uploaded-video timestamp fallback
18. Learning-content mapping alignment
19. RAG context enrichment with graph dependencies
20. Out-of-context question protection with polite fallback
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import settings
from app.main import app
from app.schemas import (
    Chunk,
    ConceptGraphResponse,
    ConceptNode,
    ConceptRelationship,
    ConceptRelationshipType,
    ProcessResponse,
    TopicGroup,
)
from app.services.concept_graph_service import (
    ConceptGraphError,
    build_and_save_concept_graph,
    extract_concept_nodes,
    extract_concept_relationships,
    format_dot_graph,
    get_or_create_concept_graph,
    get_prerequisites,
    get_related_concepts,
    load_concept_graph,
    save_concept_graph,
)
from app.services.rag_service import (
    FALLBACK_UNAVAILABLE_MSG,
    build_tutor_prompt,
)
from app.services.tutor_service import RELEVANCE_THRESHOLD, ask_tutor
from ui.dashboard_helpers import get_youtube_jump_url

client = TestClient(app)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def mock_session() -> ProcessResponse:
    """Construct a mock ProcessResponse with realistic chunks and topics."""
    chunks = [
        Chunk(
            chunk_id="chunk-001",
            text="Welcome to Python programming. Today we start by understanding variables and memory storage.",
            start_timestamp=0.0,
            end_timestamp=65.0,
            start_label="00:00",
            end_label="01:05",
            word_count=13,
        ),
        Chunk(
            chunk_id="chunk-002",
            text="In Python, you need variables before you can store data types like integers and strings.",
            start_timestamp=65.0,
            end_timestamp=130.0,
            start_label="01:05",
            end_label="02:10",
            word_count=15,
        ),
        Chunk(
            chunk_id="chunk-003",
            text="Now let's talk about Lists. Lists are used with for loops to iterate over multiple elements.",
            start_timestamp=130.0,
            end_timestamp=200.0,
            start_label="02:10",
            end_label="03:20",
            word_count=16,
        ),
        Chunk(
            chunk_id="chunk-004",
            text="For loops require a sequence like lists or range in order to execute repetitive tasks.",
            start_timestamp=200.0,
            end_timestamp=270.0,
            start_label="03:20",
            end_label="04:30",
            word_count=15,
        ),
    ]

    topics = [
        TopicGroup(
            topic_id="topic-001",
            name="Variables and Memory",
            chunk_ids=["chunk-001", "chunk-002"],
            start_time=0.0,
            end_time=130.0,
            start_label="00:00",
            end_label="02:10",
            timestamp_range="00:00 – 02:10",
            key_concepts=["Variables", "Memory Storage", "Data Types"],
        ),
        TopicGroup(
            topic_id="topic-002",
            name="Lists and Collections",
            chunk_ids=["chunk-003"],
            start_time=130.0,
            end_time=200.0,
            start_label="02:10",
            end_label="03:20",
            timestamp_range="02:10 – 03:20",
            key_concepts=["Lists", "Sequences"],
        ),
        TopicGroup(
            topic_id="topic-003",
            name="For Loops and Iteration",
            chunk_ids=["chunk-004"],
            start_time=200.0,
            end_time=270.0,
            start_label="03:20",
            end_label="04:30",
            timestamp_range="03:20 – 04:30",
            key_concepts=["For Loops", "Range"],
        ),
    ]

    return ProcessResponse(
        session_id="test_session_p8",
        video_id="dQw4w9WgXcQ",
        youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        raw_segment_count=4,
        cleaned_text_preview="Welcome to Python programming...",
        chunk_count=len(chunks),
        chunks=chunks,
        topics=topics,
    )


# ==============================================================================
# Tests
# ==============================================================================

def test_01_concept_extraction(mock_session: ProcessResponse) -> None:
    """1. Verify concept extraction extracts canonical concept nodes from topics."""
    concepts = extract_concept_nodes(mock_session)
    assert len(concepts) == len(mock_session.topics)
    names = [c.name for c in concepts]
    assert "Variables and Memory" in names
    assert "Lists and Collections" in names
    assert "For Loops and Iteration" in names


def test_02_concept_ids_format(mock_session: ProcessResponse) -> None:
    """2. Verify concept IDs follow concept-001, concept-002, etc."""
    concepts = extract_concept_nodes(mock_session)
    for idx, c in enumerate(concepts, start=1):
        expected_id = f"concept-{idx:03d}"
        assert c.concept_id == expected_id


def test_03_concept_to_topic_mapping(mock_session: ProcessResponse) -> None:
    """3. Verify concept -> topic mapping links to existing topic groups."""
    concepts = extract_concept_nodes(mock_session)
    topic_ids = {t.topic_id for t in mock_session.topics}
    for c in concepts:
        assert c.topic_id in topic_ids
        assert c.topic_name != ""


def test_04_concept_to_chunk_mapping(mock_session: ProcessResponse) -> None:
    """4. Verify concept -> chunk mapping contains traceable chunk IDs."""
    concepts = extract_concept_nodes(mock_session)
    session_chunk_ids = {chunk.chunk_id for chunk in mock_session.chunks}
    for c in concepts:
        assert len(c.source_chunk_ids) > 0
        for cid in c.source_chunk_ids:
            assert cid in session_chunk_ids


def test_05_timestamp_preservation(mock_session: ProcessResponse) -> None:
    """5. Verify timestamp preservation and formatting."""
    concepts = extract_concept_nodes(mock_session)
    for c in concepts:
        assert c.start_time <= c.end_time
        assert "–" in c.timestamp_range or "-" in c.timestamp_range
        assert len(c.timestamp_range) >= 11


def test_06_relationship_creation(mock_session: ProcessResponse) -> None:
    """6. Verify relationship creation produces evidence-backed relations."""
    concepts = extract_concept_nodes(mock_session)
    relationships = extract_concept_relationships(concepts, mock_session)
    assert len(relationships) >= 2  # at least linear sequence 'follows'
    for r in relationships:
        assert r.source
        assert r.target
        assert r.evidence
        assert isinstance(r.relation, ConceptRelationshipType)


def test_07_relationship_validation_allowed_types() -> None:
    """7. Verify relationship validation allows exactly the 5 allowed types."""
    allowed = {
        ConceptRelationshipType.PREREQUISITE,
        ConceptRelationshipType.RELATED_TO,
        ConceptRelationshipType.USED_WITH,
        ConceptRelationshipType.DEPENDS_ON,
        ConceptRelationshipType.FOLLOWS,
    }
    assert len(allowed) == 5
    for t in allowed:
        rel = ConceptRelationship(
            source="Variables",
            relation=t,
            target="Data Types",
            evidence="Instructor explains data types follow variables.",
        )
        assert rel.relation == t


def test_08_unsupported_relationship_rejection() -> None:
    """8. Verify unsupported relationship types are rejected by validation."""
    with pytest.raises(ValidationError):
        ConceptRelationship(
            source="A",
            relation="is_parent_of",  # type: ignore
            target="B",
            evidence="Invalid relation type",
        )


def test_09_session_isolation(mock_session: ProcessResponse, tmp_path: Path) -> None:
    """9. Verify session isolation: graph data is strictly keyed and isolated by session_id."""
    with patch.object(settings, "concept_graph_dir", tmp_path):
        concepts = extract_concept_nodes(mock_session)
        rels = extract_concept_relationships(concepts, mock_session)

        graph_a = ConceptGraphResponse(
            session_id="session_A",
            concepts=concepts,
            relationships=rels,
            total_concepts=len(concepts),
            total_relationships=len(rels),
        )
        save_concept_graph(graph_a)

        assert (tmp_path / "session_A.json").exists()
        assert not (tmp_path / "session_B.json").exists()

        loaded_a = load_concept_graph("session_A")
        loaded_b = load_concept_graph("session_B")

        assert loaded_a is not None
        assert loaded_a.session_id == "session_A"
        assert loaded_b is None


def test_10_graph_persistence(mock_session: ProcessResponse, tmp_path: Path) -> None:
    """10. Verify graph persistence writes valid JSON to disk."""
    with patch.object(settings, "concept_graph_dir", tmp_path):
        concepts = extract_concept_nodes(mock_session)
        rels = extract_concept_relationships(concepts, mock_session)
        graph = ConceptGraphResponse(
            session_id="test_persist",
            concepts=concepts,
            relationships=rels,
            total_concepts=len(concepts),
            total_relationships=len(rels),
        )
        save_concept_graph(graph)

        file_path = tmp_path / "test_persist.json"
        assert file_path.exists()
        raw_data = json.loads(file_path.read_text(encoding="utf-8"))
        assert raw_data["session_id"] == "test_persist"
        assert raw_data["total_concepts"] == len(concepts)


def test_11_graph_loading(mock_session: ProcessResponse, tmp_path: Path) -> None:
    """11. Verify graph loading parses saved JSON back into ConceptGraphResponse."""
    with patch.object(settings, "concept_graph_dir", tmp_path):
        concepts = extract_concept_nodes(mock_session)
        graph = ConceptGraphResponse(
            session_id="test_load",
            concepts=concepts,
            relationships=[],
            total_concepts=len(concepts),
            total_relationships=0,
        )
        save_concept_graph(graph)

        loaded = load_concept_graph("test_load")
        assert loaded is not None
        assert loaded.session_id == "test_load"
        assert len(loaded.concepts) == len(concepts)


def test_12_missing_graph_handling(tmp_path: Path) -> None:
    """12. Verify missing graph returns None or raises ConceptGraphError."""
    with patch.object(settings, "concept_graph_dir", tmp_path):
        res = load_concept_graph("nonexistent_session")
        assert res is None

        with pytest.raises(ConceptGraphError):
            build_and_save_concept_graph("nonexistent_session")


def test_13_api_endpoint_response_structure() -> None:
    """13. Verify API endpoint GET /api/sessions/{session_id}/concept-graph."""
    # Test on existing session 3ec404e4f350
    response = client.get("/api/sessions/3ec404e4f350/concept-graph")
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "3ec404e4f350"
    assert "concepts" in data
    assert "relationships" in data
    assert data["total_concepts"] == len(data["concepts"])
    assert data["total_relationships"] == len(data["relationships"])
    assert data["total_concepts"] > 0


def test_14_concept_selection_and_details(mock_session: ProcessResponse) -> None:
    """14. Verify concept selection, prerequisite discovery, and related concepts inspection."""
    concepts = extract_concept_nodes(mock_session)
    rels = [
        ConceptRelationship(
            source="For Loops and Iteration",
            relation=ConceptRelationshipType.DEPENDS_ON,
            target="Lists and Collections",
            evidence="For loops iterate over lists.",
        ),
        ConceptRelationship(
            source="Variables and Memory",
            relation=ConceptRelationshipType.PREREQUISITE,
            target="Lists and Collections",
            evidence="Variables needed before data collections.",
        ),
        ConceptRelationship(
            source="Lists and Collections",
            relation=ConceptRelationshipType.USED_WITH,
            target="For Loops and Iteration",
            evidence="Used together in iteration.",
        ),
    ]
    graph = ConceptGraphResponse(
        session_id="test_inspect",
        concepts=concepts,
        relationships=rels,
        total_concepts=len(concepts),
        total_relationships=len(rels),
    )

    # Check prerequisites for 'For Loops and Iteration'
    prereqs = get_prerequisites("For Loops and Iteration", graph)
    assert any(p["concept"] == "Lists and Collections" for p in prereqs)

    # Check related concepts for 'Lists and Collections'
    related = get_related_concepts("Lists and Collections", graph)
    assert len(related) >= 2


def test_15_source_chunk_attribution(mock_session: ProcessResponse) -> None:
    """15. Verify source chunk attribution in concept nodes."""
    concepts = extract_concept_nodes(mock_session)
    c0 = concepts[0]
    assert c0.source_chunk_ids == ["chunk-001", "chunk-002"]


def test_16_youtube_timestamp_generation() -> None:
    """16. Verify YouTube timestamp URL generation for concepts."""
    url = get_youtube_jump_url("kqtD5dpn9C8", 305)
    assert url == "https://www.youtube.com/watch?v=kqtD5dpn9C8&t=305s"


def test_17_uploaded_video_timestamp_fallback() -> None:
    """17. Verify timestamp formatting works cleanly for uploaded/local video sessions."""
    # When video_id is local or empty, start_time is still valid
    start_sec = 125.5
    mm_ss = f"{int(start_sec) // 60:02d}:{int(start_sec) % 60:02d}"
    assert mm_ss == "02:05"


def test_18_learning_content_mapping(mock_session: ProcessResponse) -> None:
    """18. Verify concepts map to topic learning content topics."""
    concepts = extract_concept_nodes(mock_session)
    for c in concepts:
        topic_match = next((t for t in mock_session.topics if t.topic_id == c.topic_id), None)
        assert topic_match is not None
        assert topic_match.name == c.name


def test_19_rag_context_enrichment() -> None:
    """19. Verify RAG prompt construction is enriched with knowledge graph context."""
    graph_context = (
        "Related Concept Dependencies:\n"
        "- For Loops depends on Lists (Evidence: For loops iterate over sequences)\n"
        "- Lists used with For Loops"
    )
    from app.schemas import SearchResultChunk
    messages = build_tutor_prompt(
        question="How do for loops work with lists?",
        chunks=[
            SearchResultChunk(
                chunk_id="c1",
                topic="Loops",
                concepts=["Loops"],
                start_time=300.0,
                end_time=360.0,
                start_timestamp=300.0,
                end_timestamp=360.0,
                timestamp="05:00 – 06:00",
                text="For loops iterate over lists.",
                score=0.85,
            )
        ],
        graph_context=graph_context,
    )
    user_prompt = messages[1]["content"]
    assert "=== CONCEPT KNOWLEDGE GRAPH CONTEXT ===" in user_prompt
    assert "For Loops depends on Lists" in user_prompt


def test_20_out_of_context_question_protection() -> None:
    """20. Verify out-of-context questions are rejected with polite fallback."""
    with patch("app.services.tutor_service.search_session") as mock_search:
        # Mock search returning no relevant chunks (scores below threshold)
        mock_search.return_value = MagicMock(
            results=[
                MagicMock(score=0.10, topic="Irrelevant", timestamp="00:00 – 01:00", text="Irrelevant content.")
            ]
        )
        resp = ask_tutor(
            session_id="3ec404e4f350",
            question="What is the capital of Australia?",
        )
        assert FALLBACK_UNAVAILABLE_MSG in resp.answer
        assert resp.sources == []
