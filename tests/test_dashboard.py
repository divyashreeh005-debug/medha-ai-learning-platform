"""Unit and functional tests for Medhā Phase 5 Interactive Learning Dashboard.

Verifies:
1. Chronological topic ordering (never alphabetical)
2. Previous topic navigation logic
3. Next topic navigation logic
4. Boundary condition: First topic has no previous topic
5. Boundary condition: Last topic has no next topic
6. Selected topic maps accurately to topic_id
7. Selected topic maps to correct transcript chunks
8. Timestamp preservation and YouTube deep-link construction
9. Learning content binds to correct topic
10. Session isolation across lectures
11. Re-use of cached learning content without LLM re-invocation
12. Graceful handling of missing learning content
13. Concept badges formatting and deduplication
14. Topic progress calculations
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.schemas import Chunk, ProcessResponse, TopicGroup, TopicLearningContent
from app.services.process_video import load_session
from ui.dashboard_helpers import (
    calculate_topic_progress,
    filter_chunks_for_topic,
    format_concept_badges,
    get_chronological_topics,
    get_next_topic_index,
    get_previous_topic_index,
    get_youtube_jump_url,
    load_cached_learning_content,
)


@pytest.fixture
def mock_topics() -> list[dict[str, Any]]:
    """Sample topics out of order to verify sorting and navigation."""
    return [
        {
            "topic_id": "topic-002",
            "name": "Variables",
            "start_time": 306.0,
            "end_time": 537.0,
            "timestamp_range": "05:06 – 08:57",
            "key_concepts": ["variable", "memory", "assignment"],
            "chunk_ids": ["chunk-003", "chunk-004"],
        },
        {
            "topic_id": "topic-001",
            "name": "Introduction",
            "start_time": 2.0,
            "end_time": 149.0,
            "timestamp_range": "00:02 – 02:29",
            "key_concepts": ["python", "course", "basics"],
            "chunk_ids": ["chunk-001", "chunk-002"],
        },
        {
            "topic_id": "topic-003",
            "name": "While Loops",
            "start_time": 2507.0,
            "end_time": 2711.0,
            "timestamp_range": "41:47 – 45:11",
            "key_concepts": ["loop", "iteration", "condition"],
            "chunk_ids": ["chunk-005", "chunk-006"],
        },
    ]


@pytest.fixture
def mock_chunks() -> list[dict[str, Any]]:
    """Sample transcript chunks corresponding to mock topics."""
    return [
        {"chunk_id": "chunk-001", "text": "Welcome to python tutorial.", "start_label": "00:02", "end_label": "01:15", "start_time": 2.0},
        {"chunk_id": "chunk-002", "text": "We will cover variables.", "start_label": "01:15", "end_label": "02:29", "start_time": 75.0},
        {"chunk_id": "chunk-003", "text": "Variables store values in memory.", "start_label": "05:06", "end_label": "07:00", "start_time": 306.0},
        {"chunk_id": "chunk-004", "text": "We can change variables dynamically.", "start_label": "07:00", "end_label": "08:57", "start_time": 420.0},
        {"chunk_id": "chunk-005", "text": "While loops execute repeatedly.", "start_label": "41:47", "end_label": "43:30", "start_time": 2507.0},
        {"chunk_id": "chunk-006", "text": "Always increment the counter.", "start_label": "43:30", "end_label": "45:11", "start_time": 2610.0},
    ]


def test_1_topics_chronological_order(mock_topics: list[dict[str, Any]]) -> None:
    """1. Topics must be strictly ordered chronologically by start_time, never alphabetically."""
    sorted_topics = get_chronological_topics(mock_topics)
    assert len(sorted_topics) == 3
    assert sorted_topics[0]["topic_id"] == "topic-001"
    assert sorted_topics[1]["topic_id"] == "topic-002"
    assert sorted_topics[2]["topic_id"] == "topic-003"
    assert sorted_topics[0]["start_time"] < sorted_topics[1]["start_time"] < sorted_topics[2]["start_time"]


def test_2_previous_topic_navigation(mock_topics: list[dict[str, Any]]) -> None:
    """2. Previous topic navigation returns the preceding index."""
    sorted_topics = get_chronological_topics(mock_topics)
    assert get_previous_topic_index(2) == 1
    assert get_previous_topic_index(1) == 0


def test_3_next_topic_navigation(mock_topics: list[dict[str, Any]]) -> None:
    """3. Next topic navigation returns the succeeding index."""
    total = len(mock_topics)
    assert get_next_topic_index(0, total) == 1
    assert get_next_topic_index(1, total) == 2


def test_4_first_topic_has_no_previous(mock_topics: list[dict[str, Any]]) -> None:
    """4. Boundary condition: First topic has no previous topic."""
    assert get_previous_topic_index(0) is None


def test_5_last_topic_has_no_next(mock_topics: list[dict[str, Any]]) -> None:
    """5. Boundary condition: Last topic has no next topic."""
    total = len(mock_topics)
    assert get_next_topic_index(total - 1, total) is None


def test_6_selected_topic_maps_to_correct_topic_id(mock_topics: list[dict[str, Any]]) -> None:
    """6. Selected topic maps cleanly to its unique topic_id."""
    sorted_topics = get_chronological_topics(mock_topics)
    idx = 1
    selected = sorted_topics[idx]
    assert selected["topic_id"] == "topic-002"
    assert selected["name"] == "Variables"


def test_7_selected_topic_maps_to_correct_transcript_chunks(
    mock_topics: list[dict[str, Any]],
    mock_chunks: list[dict[str, Any]],
) -> None:
    """7. Selected topic filters only its associated chunks without leakage."""
    sorted_topics = get_chronological_topics(mock_topics)
    var_topic = sorted_topics[1]  # topic-002: chunk-003, chunk-004

    filtered = filter_chunks_for_topic(var_topic, mock_chunks)
    assert len(filtered) == 2
    assert [c["chunk_id"] for c in filtered] == ["chunk-003", "chunk-004"]
    assert "Variables store values in memory." in filtered[0]["text"]
    assert "While loops execute repeatedly." not in [c["text"] for c in filtered]


def test_8_timestamp_preservation_and_youtube_link(mock_topics: list[dict[str, Any]]) -> None:
    """8. Preserves start_time, end_time, timestamp_range, and builds correct YouTube deep link."""
    sorted_topics = get_chronological_topics(mock_topics)
    topic = sorted_topics[1]
    assert topic["start_time"] == 306.0
    assert topic["end_time"] == 537.0
    assert topic["timestamp_range"] == "05:06 – 08:57"

    url = get_youtube_jump_url("kqtD5dpn9C8", topic["start_time"])
    assert url == "https://www.youtube.com/watch?v=kqtD5dpn9C8&t=306s"


def test_9_learning_content_mapping() -> None:
    """9. Learning content fields map accurately to the target topic schema."""
    content = TopicLearningContent(
        topic_id="topic-002",
        topic_name="Variables",
        timestamp_range="05:06 – 08:57",
        summary="Instructor explains variables and memory allocation.",
        simple_explanation="A variable is like a named box in memory.",
        analogy="Storage lockers labeled with names.",
        key_takeaways=["Store values", "Case sensitive", "Reassignable"],
        source_chunk_ids=["chunk-003", "chunk-004"],
    )

    topic = TopicGroup(
        topic_id="topic-002",
        name="Variables",
        start_time=306.0,
        end_time=537.0,
        start_label="05:06",
        end_label="08:57",
        timestamp_range="05:06 – 08:57",
        chunk_ids=["chunk-003", "chunk-004"],
        key_concepts=["variables", "memory"],
        learning_content=content,
    )

    assert topic.learning_content is not None
    assert topic.learning_content.topic_id == "topic-002"
    assert len(topic.learning_content.key_takeaways) == 3
    assert topic.learning_content.source_chunk_ids == ["chunk-003", "chunk-004"]


def test_10_session_isolation(tmp_path: Path) -> None:
    """10. Content and topics from one session do not leak into another session."""
    session_a = "session_alpha"
    session_b = "session_beta"

    # Save cache for session A only
    cache_a_dir = tmp_path / session_a
    cache_a_dir.mkdir(parents=True)
    with open(cache_a_dir / "topic-001.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "topic_id": "topic-001",
                "summary": "Alpha summary",
                "simple_explanation": "Alpha simple",
                "analogy": "Alpha analogy",
                "key_takeaways": ["Alpha 1"],
            },
            f,
        )

    # Load cache for session A
    loaded_a = load_cached_learning_content(session_a, "topic-001", cache_dir=tmp_path)
    assert loaded_a is not None
    assert loaded_a["summary"] == "Alpha summary"

    # Attempt to load same topic for session B
    loaded_b = load_cached_learning_content(session_b, "topic-001", cache_dir=tmp_path)
    assert loaded_b is None


def test_11_cached_learning_content_reuse(tmp_path: Path) -> None:
    """11. Cached content is loaded directly from disk without requiring LLM generation."""
    session_id = "test_reuse_session"
    topic_id = "topic-test"

    cache_dir = tmp_path / session_id
    cache_dir.mkdir(parents=True)
    content_payload = {
        "topic_id": topic_id,
        "summary": "Existing cached summary",
        "simple_explanation": "Existing cached simple explanation",
        "analogy": "Existing analogy",
        "key_takeaways": ["Point 1", "Point 2"],
    }
    with open(cache_dir / f"{topic_id}.json", "w", encoding="utf-8") as f:
        json.dump(content_payload, f)

    cached = load_cached_learning_content(session_id, topic_id, cache_dir=tmp_path)
    assert cached is not None
    assert cached["summary"] == "Existing cached summary"
    assert cached["key_takeaways"] == ["Point 1", "Point 2"]


def test_12_missing_learning_content_graceful_handling(tmp_path: Path) -> None:
    """12. Missing learning content returns None and does not crash or throw exceptions."""
    missing = load_cached_learning_content("unknown_session", "topic-999", cache_dir=tmp_path)
    assert missing is None

    # TopicGroup schema without learning_content
    topic = TopicGroup(
        topic_id="topic-001",
        name="Intro",
        start_time=0.0,
        end_time=60.0,
        start_label="00:00",
        end_label="01:00",
        timestamp_range="00:00 – 01:00",
        chunk_ids=["chunk-001"],
        key_concepts=[],
        learning_content=None,
    )
    assert topic.learning_content is None


def test_13_concept_badges_formatting() -> None:
    """13. Concept badges are properly cleaned, whitespace-stripped, and deduplicated."""
    raw = ["memory", "  memory  ", "Variable", "variable", "COMPUTER MEMORY", ""]
    cleaned = format_concept_badges(raw)
    assert cleaned == ["memory", "Variable", "COMPUTER MEMORY"]


def test_14_topic_progress_calculation() -> None:
    """14. Progress calculation returns accurate ratio and formatted text label."""
    ratio, label = calculate_topic_progress(0, 10)
    assert ratio == 0.1
    assert label == "Topic 1 of 10"

    ratio, label = calculate_topic_progress(2, 17)
    assert round(ratio, 4) == round(3 / 17, 4)
    assert label == "Topic 3 of 17"

    ratio, label = calculate_topic_progress(16, 17)
    assert ratio == 1.0
    assert label == "Topic 17 of 17"


def test_15_integration_real_session_topics_and_chunks() -> None:
    """15. Integration test on real session 3ec404e4f350: chronological order, chunk mapping, and cache."""
    session = load_session("3ec404e4f350")
    assert session is not None

    topics = get_chronological_topics([t.model_dump() for t in session.topics])
    assert len(topics) == 17

    # Verify chronological ordering
    for i in range(len(topics) - 1):
        assert topics[i]["start_time"] <= topics[i + 1]["start_time"]

    # Verify topic-003 (Variables)
    topic_003 = next((t for t in topics if t["topic_id"] == "topic-003"), None)
    assert topic_003 is not None
    assert topic_003["name"] == "Variables"
    assert topic_003["timestamp_range"] == "05:06 – 08:57"
    assert topic_003["chunk_ids"] == ["chunk-005", "chunk-006", "chunk-007"]

    # Filter chunks
    chunks_dict = [c.model_dump() for c in session.chunks]
    var_chunks = filter_chunks_for_topic(topic_003, chunks_dict)
    assert len(var_chunks) == 3
    assert [c["chunk_id"] for c in var_chunks] == ["chunk-005", "chunk-006", "chunk-007"]

    # Verify cached content for topic-003 exists and reloads
    cached = load_cached_learning_content("3ec404e4f350", "topic-003")
    assert cached is not None
    assert cached["topic_id"] == "topic-003"
    assert cached["topic_name"] == "Variables"
    assert "summary" in cached
    assert "simple_explanation" in cached
    assert "analogy" in cached
    assert len(cached["key_takeaways"]) >= 3
