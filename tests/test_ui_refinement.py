"""Tests for Medhā Phase 8.1: UI/UX Refinement & Multilingual Learning Experience.

Verifies:
1. Topic navigation bounds and next/previous state sync helpers
2. Lecture title formatting with real title, fallback, and legacy handling
3. YouTube title fetching via public oEmbed
4. Legacy session backward compatibility (migration on load)
5. Knowledge Graph DOT formatting removes timestamp clutter from node labels
6. Concept Main Idea generation produces clean, informative descriptions
7. Interactive revision memory card generation and schema compliance
8. Memory card navigation indices and clamping
9. Multilingual enum and display names completeness
10. Source transcript chunk immutability during translation
11. Memory cards API endpoint (GET /api/sessions/{session_id}/memory-cards)
12. Multilingual concept graph and content endpoints
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.pipeline.youtube import fetch_youtube_title
from app.schemas import (
    LANGUAGE_DISPLAY_NAMES,
    Chunk,
    ConceptGraphResponse,
    ConceptNode,
    ConceptRelationship,
    ConceptRelationshipType,
    MemoryCard,
    MemoryCardCollection,
    ProcessResponse,
    SupportedLanguage,
    TopicGroup,
)
from app.services.concept_graph_service import (
    extract_concept_nodes,
    format_dot_graph,
)
from app.services.memory_card_service import (
    generate_memory_cards,
    load_memory_cards,
    save_memory_cards,
)
from app.services.process_video import (
    format_duration,
    load_session,
)
from app.services.translation_service import (
    get_localized_concept_graph,
    translate_educational_content,
)
from ui.dashboard_helpers import (
    format_lecture_title,
    get_next_card_index,
    get_next_topic_index,
    get_previous_card_index,
    get_previous_topic_index,
    load_cached_learning_content,
)

client = TestClient(app)

SAMPLE_SESSION_ID = "3ec404e4f350"


# ==============================================================================
# 1. Navigation State Synchronization & Bounds
# ==============================================================================


def test_topic_navigation_bounds() -> None:
    """Ensure topic navigation clamps within valid boundaries."""
    assert get_previous_topic_index(0) is None
    assert get_previous_topic_index(3) == 2
    assert get_next_topic_index(0, 5) == 1
    assert get_next_topic_index(4, 5) is None
    assert get_next_topic_index(5, 5) is None
    assert get_next_topic_index(0, 0) is None


def test_memory_card_navigation_bounds() -> None:
    """Ensure card navigation clamps within card list bounds."""
    assert get_previous_card_index(0, 10) == 0
    assert get_previous_card_index(5, 10) == 4
    assert get_next_card_index(9, 10) == 9
    assert get_next_card_index(4, 10) == 5
    assert get_next_card_index(0, 0) == 0
    assert get_previous_card_index(0, 0) == 0


# ==============================================================================
# 2. Lecture Title Formatting & Metadata
# ==============================================================================


def test_format_lecture_title() -> None:
    """Verify format_lecture_title returns user-friendly lecture title."""
    assert (
        format_lecture_title("Python for Beginners - Full Course", "kqtD5dpn9C8", "3ec404e4f350")
        == "Python for Beginners - Full Course"
    )
    assert format_lecture_title("", "kqtD5dpn9C8", "3ec404e4f350") == "Lecture (kqtD5dpn9C8)"
    assert format_lecture_title(None, "local_12345", "3ec404e4f350") == "Lecture Session 3ec404e4"
    assert format_lecture_title(None, None, "3ec404e4f350") == "Lecture Session 3ec404e4"
    assert format_lecture_title(None, None, None) == "Untitled Lecture"


def test_format_duration() -> None:
    """Verify format_duration formats seconds into clean string."""
    assert format_duration(45) == "0m 45s"
    assert format_duration(3600) == "1h 00m"
    assert format_duration(3725) == "1h 02m"
    assert format_duration(0) == "N/A"
    assert format_duration(-5) == "N/A"


# ==============================================================================
# 3. YouTube Title Resolution via Public oEmbed
# ==============================================================================


def test_fetch_youtube_title_oembed() -> None:
    """Test fetch_youtube_title handles successful response and fallback."""
    with patch("urllib.request.urlopen") as mock_url:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({"title": "Intro to Computer Science"}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_url.return_value = mock_resp

        title = fetch_youtube_title("abc12345")
        assert title == "Intro to Computer Science"

    # Fallback on network error
    with patch("urllib.request.urlopen", side_effect=Exception("Network Error")):
        fallback_title = fetch_youtube_title("kqtD5dpn9C8")
        assert fallback_title == "YouTube Lecture (kqtD5dpn9C8)"


# ==============================================================================
# 4. Legacy Session Backwards Compatibility
# ==============================================================================


def test_legacy_session_backwards_compatibility() -> None:
    """Verify existing session 3ec404e4f350 is loaded with title, source_type, and duration."""
    session = load_session(SAMPLE_SESSION_ID)
    assert session is not None
    assert session.session_id == SAMPLE_SESSION_ID
    assert session.title is not None
    assert len(session.title) > 0
    assert session.source_type in ["youtube", "upload"]
    assert session.detected_language == "en"
    assert session.duration_seconds is not None
    assert session.formatted_duration is not None


# ==============================================================================
# 5. Concept Graph DOT Formatting (Node Semantics: Concepts, Not Timestamps)
# ==============================================================================


def test_concept_node_labels_no_timestamps() -> None:
    """Verify Graphviz DOT nodes display clean concept names without timestamp clutter."""
    c1 = ConceptNode(
        concept_id="concept-001",
        name="Variables and Assignment",
        description="Variables store mutable values in computer memory.",
        topic_id="topic-001",
        topic_name="Variables",
        source_chunk_ids=["chunk-001"],
        start_time=10.0,
        end_time=30.0,
        timestamp_range="00:10 – 00:30",
    )
    c2 = ConceptNode(
        concept_id="concept-002",
        name="Data Types",
        description="Data types define how data is represented and manipulated.",
        topic_id="topic-001",
        topic_name="Variables",
        source_chunk_ids=["chunk-002"],
        start_time=35.0,
        end_time=60.0,
        timestamp_range="00:35 – 01:00",
    )
    rel = ConceptRelationship(
        source="Variables and Assignment",
        target="Data Types",
        relation=ConceptRelationshipType.PREREQUISITE,
        evidence="Variables must be understood before data types are assigned.",
    )
    graph = ConceptGraphResponse(
        session_id="test_session",
        concepts=[c1, c2],
        relationships=[rel],
        total_concepts=2,
        total_relationships=1,
    )

    dot_str = format_dot_graph(graph)
    # The label must contain the concept name
    assert 'label="Variables and Assignment"' in dot_str
    assert 'label="Data Types"' in dot_str
    # The label must NOT contain the timestamp range
    assert "00:10 – 00:30" not in dot_str
    assert "00:35 – 01:00" not in dot_str


# ==============================================================================
# 6. Concept Main Idea Generation Quality
# ==============================================================================


def test_concept_main_idea_generation() -> None:
    """Verify extracted concept nodes have informative descriptions without filler phrases."""
    topic = TopicGroup(
        topic_id="topic-001",
        name="Python Variables",
        start_time=0.0,
        end_time=60.0,
        start_label="00:00",
        end_label="01:00",
        timestamp_range="00:00 – 01:00",
        chunk_ids=["chunk-001"],
        key_concepts=["Variables", "Memory Storage"],
    )
    chunk = Chunk(
        chunk_id="chunk-001",
        start_timestamp=0.0,
        end_timestamp=60.0,
        start_label="00:00",
        end_label="01:00",
        word_count=18,
        text="In this video we will learn variables. A variable is a container for storing data values in Python.",
    )
    mock_session = ProcessResponse(
        session_id="mock_session_123",
        video_id="mock_vid",
        youtube_url="",
        raw_segment_count=1,
        cleaned_text_preview="In this video we will learn variables...",
        chunk_count=1,
        chunks=[chunk],
        topics=[topic],
        title="Python Basics",
        source_type="youtube",
        detected_language="en",
        duration_seconds=60.0,
        formatted_duration="1m 00s",
    )
    nodes = extract_concept_nodes(mock_session)
    assert len(nodes) >= 1
    for node in nodes:
        assert node.description is not None
        assert len(node.description) > 10
        # Filler removal verification
        assert not node.description.lower().startswith("in this video")
        assert not node.description.lower().startswith("in this lecture")


# ==============================================================================
# 7. Revision Memory Cards Generation & Structure
# ==============================================================================


def test_generate_memory_cards() -> None:
    """Verify revision memory cards are generated and have valid schema fields."""
    cards_coll = generate_memory_cards(SAMPLE_SESSION_ID, language="en")
    assert isinstance(cards_coll, MemoryCardCollection)
    assert cards_coll.session_id == SAMPLE_SESSION_ID
    assert cards_coll.language == "en"
    assert cards_coll.total_cards > 0
    assert len(cards_coll.cards) == cards_coll.total_cards

    card = cards_coll.cards[0]
    assert card.card_id.startswith("card-")
    assert card.concept_name
    assert card.front_prompt
    assert card.back_answer
    assert card.timestamp_range
    assert card.start_time >= 0.0


# ==============================================================================
# 8. Multilingual Support & Schema Verification
# ==============================================================================


def test_multilingual_supported_languages() -> None:
    """Verify all 6 requested languages are supported with display names."""
    expected_codes = ["en", "hi", "kn", "te", "ta", "ml"]
    for code in expected_codes:
        assert SupportedLanguage(code) is not None
        assert code in LANGUAGE_DISPLAY_NAMES
        assert len(LANGUAGE_DISPLAY_NAMES[code]) > 0


def test_transcript_chunks_immutability() -> None:
    """Verify source transcript chunks remain in English/original lecture language."""
    session = load_session(SAMPLE_SESSION_ID)
    assert session is not None
    original_chunk_texts = [c.text for c in session.chunks]

    # Generating memory cards or localized content should NOT touch session chunks
    generate_memory_cards(SAMPLE_SESSION_ID, language="hi")
    reloaded_session = load_session(SAMPLE_SESSION_ID)
    assert reloaded_session is not None
    reloaded_chunk_texts = [c.text for c in reloaded_session.chunks]

    assert original_chunk_texts == reloaded_chunk_texts


# ==============================================================================
# 9. API Endpoints for Phase 8.1
# ==============================================================================


def test_api_health_endpoint() -> None:
    """Verify health endpoint reports Phase 8.1."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "8.1" in data["phase"]


def test_api_memory_cards_endpoint() -> None:
    """Verify GET /api/sessions/{session_id}/memory-cards returns valid collection."""
    resp = client.get(f"/api/sessions/{SAMPLE_SESSION_ID}/memory-cards?language=en")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == SAMPLE_SESSION_ID
    assert data["total_cards"] > 0
    assert len(data["cards"]) == data["total_cards"]
    assert "front_prompt" in data["cards"][0]
    assert "back_answer" in data["cards"][0]


def test_api_concept_graph_localized_endpoint() -> None:
    """Verify GET /api/sessions/{session_id}/concept-graph accepts language param."""
    resp = client.get(f"/api/sessions/{SAMPLE_SESSION_ID}/concept-graph?language=en")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == SAMPLE_SESSION_ID
    assert data["total_concepts"] > 0
