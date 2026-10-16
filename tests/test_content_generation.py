"""Unit and integration tests for Phase 4: Educational Content Generation with Qwen3."""

from __future__ import annotations

import json
from unittest.mock import patch
import pytest

from app.schemas import Chunk, TopicGroup, TopicLearningContent
from app.services.content_generator import (
    ContentGenerationError,
    build_topic_prompt,
    generate_topic_content,
    load_cached_content,
    parse_structured_json,
    save_cached_content,
)
from app.services.llm_client import LLMClientError, check_llm_runtime


@pytest.fixture
def sample_topic_and_chunks() -> tuple[TopicGroup, list[Chunk]]:
    chunks = [
        Chunk(
            chunk_id="chunk-101",
            topic="Variables",
            concepts=["variables", "memory location"],
            text="In this section we discuss variables. Variables store values in memory.",
            start_timestamp=300.0,
            end_timestamp=360.0,
            start_label="05:00",
            end_label="06:00",
            word_count=11,
        ),
        Chunk(
            chunk_id="chunk-102",
            topic="Variables",
            concepts=["declare variable", "age"],
            text="For example, age = 20 stores the integer 20 with the label age.",
            start_timestamp=361.0,
            end_timestamp=420.0,
            start_label="06:01",
            end_label="07:00",
            word_count=13,
        ),
        Chunk(
            chunk_id="chunk-103",
            topic="Functions",
            concepts=["functions", "def"],
            text="Unrelated chunk about functions and parameters.",
            start_timestamp=500.0,
            end_timestamp=560.0,
            start_label="08:20",
            end_label="09:20",
            word_count=7,
        ),
    ]

    topic = TopicGroup(
        topic_id="topic-test-01",
        name="Variables",
        start_time=300.0,
        end_time=420.0,
        start_label="05:00",
        end_label="07:00",
        timestamp_range="05:00 – 07:00",
        key_concepts=["variables", "memory location", "age"],
        chunk_ids=["chunk-101", "chunk-102"],
    )

    return topic, chunks


def test_1_runtime_availability_handling() -> None:
    """1. Verify runtime detection returns structured metadata without failing."""
    info = check_llm_runtime()
    assert isinstance(info, dict)
    assert "available" in info
    assert "runtime" in info
    assert "base_url" in info
    assert "model" in info
    assert "details" in info


def test_2_prompt_construction(sample_topic_and_chunks: tuple[TopicGroup, list[Chunk]]) -> None:
    """2. Verify grounded prompt construction with strict constraints."""
    topic, chunks = sample_topic_and_chunks
    messages = build_topic_prompt(topic, chunks)

    assert len(messages) == 2
    system_msg = messages[0]["content"]
    user_msg = messages[1]["content"]

    # Grounding instructions must be explicit
    assert "Use ONLY the provided lecture context" in system_msg
    assert "Do not invent facts" in system_msg
    assert "summary" in system_msg
    assert "simple_explanation" in system_msg
    assert "analogy" in system_msg
    assert "key_takeaways" in system_msg

    # User message contains topic metadata and chunk text
    assert "Topic: Variables" in user_msg
    assert "05:00 – 07:00" in user_msg
    assert "chunk-101" in user_msg
    assert "chunk-102" in user_msg


def test_3_topic_context_selection(sample_topic_and_chunks: tuple[TopicGroup, list[Chunk]]) -> None:
    """3. Verify only chunks matching the topic are included in context."""
    topic, chunks = sample_topic_and_chunks
    messages = build_topic_prompt(topic, chunks)
    user_msg = messages[1]["content"]

    # Chunks 101 and 102 must be included
    assert "chunk-101" in user_msg
    assert "chunk-102" in user_msg

    # Unrelated chunk 103 must NOT be included
    assert "chunk-103" not in user_msg
    assert "Unrelated chunk about functions" not in user_msg


def test_4_structured_response_parsing() -> None:
    """4. Verify structured JSON parsing with fences, plain text, and error handling."""
    # Plain valid JSON
    valid_json = json.dumps({
        "summary": "The instructor explains how to store data in variables.",
        "simple_explanation": "Variables are named containers for data.",
        "analogy": "Like labeling a storage box.",
        "key_takeaways": ["Stores data in memory", "Has a name and value", "Can be updated"],
    })
    parsed = parse_structured_json(valid_json)
    assert parsed["summary"].startswith("The instructor explains")
    assert len(parsed["key_takeaways"]) == 3

    # Wrapped in markdown codeblock
    fenced_json = f"```json\n{valid_json}\n```"
    parsed_fenced = parse_structured_json(fenced_json)
    assert parsed_fenced["summary"] == parsed["summary"]

    # Missing required key should raise ContentGenerationError
    incomplete_json = json.dumps({"summary": "Incomplete"})
    with pytest.raises(ContentGenerationError, match="missing required keys"):
        parse_structured_json(incomplete_json)

    # Malformed JSON should raise ContentGenerationError
    with pytest.raises(ContentGenerationError, match="Failed to parse LLM output"):
        parse_structured_json("This is not JSON at all.")


def test_5_source_chunk_and_timestamp_preservation() -> None:
    """5 & 6. Verify source chunk IDs and timestamps are retained in TopicLearningContent."""
    content = TopicLearningContent(
        topic_id="topic-001",
        topic_name="Variables",
        timestamp_range="05:06 – 08:57",
        summary="A variable stores information.",
        simple_explanation="A named box for data.",
        analogy="A labeled kitchen jar.",
        key_takeaways=["Point 1", "Point 2", "Point 3"],
        source_chunk_ids=["chunk-005", "chunk-006", "chunk-007"],
    )

    assert content.topic_id == "topic-001"
    assert content.timestamp_range == "05:06 – 08:57"
    assert content.source_chunk_ids == ["chunk-005", "chunk-006", "chunk-007"]
    assert len(content.key_takeaways) == 3


def test_7_cached_generation_and_persistence(tmp_path) -> None:
    """7. Verify caching saves to disk and avoids repeated model invocation."""
    content = TopicLearningContent(
        topic_id="topic-c1",
        topic_name="While Loops",
        timestamp_range="41:47 – 45:11",
        summary="Explains while loops.",
        simple_explanation="Loops repeat code while a condition is true.",
        analogy="A conveyor belt.",
        key_takeaways=["Repeats code", "Needs increment"],
        source_chunk_ids=["chunk-033", "chunk-034"],
    )

    with patch("app.services.content_generator.settings.content_cache_dir", tmp_path):
        session_id = "test_cache_session"
        topic_id = "topic-c1"

        # Initially no cache
        assert load_cached_content(session_id, topic_id) is None

        # Save to cache
        save_cached_content(session_id, topic_id, content)

        # Retrieve from cache
        reloaded = load_cached_content(session_id, topic_id)
        assert reloaded is not None
        assert reloaded.topic_name == "While Loops"
        assert reloaded.timestamp_range == "41:47 – 45:11"
        assert reloaded.source_chunk_ids == ["chunk-033", "chunk-034"]


def test_8_failure_handling_when_model_unavailable() -> None:
    """8. Verify graceful failure handling when the LLM service cannot be reached."""
    with patch(
        "app.services.content_generator.call_llm_chat",
        side_effect=LLMClientError("Connection refused on port 1234"),
    ):
        with pytest.raises(ContentGenerationError, match="Connection refused on port 1234"):
            generate_topic_content(
                session_id="3ec404e4f350",
                topic_id="topic-003",
                force_regenerate=True,
            )


def test_9_integration_topic_content_generation() -> None:
    """Integration test: Generate educational content for a real topic from session 3ec404e4f350."""
    session_id = "3ec404e4f350"
    topic_id = "topic-003"  # Variables (05:06 – 08:57)

    content = generate_topic_content(session_id=session_id, topic_id=topic_id)

    assert content.topic_id == topic_id
    assert content.topic_name == "Variables"
    assert content.timestamp_range == "05:06 – 08:57"
    assert len(content.summary) > 15
    assert len(content.simple_explanation) > 15
    assert len(content.analogy) > 10
    assert len(content.key_takeaways) >= 3
    assert len(content.source_chunk_ids) > 0

    # Ensure cached on subsequent retrieval
    cached = load_cached_content(session_id, topic_id)
    assert cached is not None
    assert cached.summary == content.summary
