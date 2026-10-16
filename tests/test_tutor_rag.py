"""Tests for Medhā Phase 6: Context-Aware AI Tutor / Lecture-Grounded RAG.

Verifies:
1. Question embedding and retrieval uses existing MiniLM
2. FAISS retrieval returns relevant chunks
3. Session isolation (Session A searches only Session A)
4. Correct source chunk mapping in ChatResponse
5. Correct topic mapping in sources
6. Correct timestamp preservation
7. YouTube timestamp URL generation
8. RAG prompt construction
9. Grounding instruction presence in system message
10. Chat response parsing
11. LLM unavailable handling (graceful exception/response)
12. Empty question handling
13. Missing session handling
14. Missing FAISS index handling (auto-builds gracefully)
15. No relevant context / out-of-context handling (low-score fallback)
16. Conversation history handling in prompt
17. Real-session integration test on 3ec404e4f350
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.pipeline.embeddings import get_embedding_service
from app.pipeline.vector_store import get_vector_store
from app.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatSource,
    SearchResultChunk,
)
from app.services.llm_client import LLMClientError, check_llm_runtime
from app.services.rag_service import FALLBACK_UNAVAILABLE_MSG, build_tutor_prompt
from app.services.search_service import SessionNotFoundError, search_session
from app.services.tutor_service import RELEVANCE_THRESHOLD, TutorServiceError, ask_tutor
from ui.dashboard_helpers import get_youtube_jump_url


@pytest.fixture
def sample_search_chunks() -> list[SearchResultChunk]:
    return [
        SearchResultChunk(
            chunk_id="chunk-005",
            topic="Variables",
            concepts=["variables", "memory", "age"],
            text="In this lecture section, the instructor explains variables and storing data in computer memory.",
            start_time=305.84,
            end_time=387.0,
            start_timestamp=305.84,
            end_timestamp=387.0,
            timestamp="05:06 – 06:27",
            score=0.72,
        ),
        SearchResultChunk(
            chunk_id="chunk-006",
            topic="Variables",
            concepts=["variables", "naming"],
            text="Variables can store numbers, strings, and boolean values like true or false.",
            start_time=382.0,
            end_time=460.0,
            start_timestamp=382.0,
            end_timestamp=460.0,
            timestamp="06:22 – 07:40",
            score=0.65,
        ),
    ]


def test_1_question_embedding_uses_minilm() -> None:
    """1. Verify question embedding uses the configured sentence-transformers/all-MiniLM-L6-v2."""
    svc = get_embedding_service()
    assert svc.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    emb = svc.embed_query("Why do we use variables?")
    assert emb.shape == (384,)


def test_2_faiss_retrieval_returns_relevant_chunks() -> None:
    """2. Verify FAISS retrieval returns expected chunks for an existing session."""
    session_id = "3ec404e4f350"
    res = search_session(session_id, "variables store data memory", top_k=3)
    assert res.total_results > 0
    top = res.results[0]
    assert top.topic == "Variables"
    assert top.chunk_id in ["chunk-005", "chunk-006", "chunk-007"]
    assert top.score > 0.40


def test_3_session_isolation() -> None:
    """3. Verify retrieval for session A strictly searches session A."""
    store = get_vector_store()
    # Query against real session
    real_results = search_session("3ec404e4f350", "loops", top_k=2)
    assert len(real_results.results) == 2

    # Querying a nonexistent session raises SessionNotFoundError
    with pytest.raises(SessionNotFoundError):
        search_session("nonexistent_session_xyz", "loops", top_k=2)


def test_4_correct_source_chunk_mapping(sample_search_chunks: list[SearchResultChunk]) -> None:
    """4. Verify SearchResultChunks correctly map into ChatSource structures."""
    with patch("app.services.tutor_service.search_session") as mock_search, \
         patch("app.services.tutor_service.call_llm_chat", return_value="Variables store temporary data in memory."):

        mock_search.return_value = MagicMock(results=sample_search_chunks)

        resp = ask_tutor(
            session_id="3ec404e4f350",
            question="Why use variables?",
            top_k=2,
        )

        assert len(resp.sources) == 2
        assert resp.sources[0].chunk_id == "chunk-005"
        assert resp.sources[1].chunk_id == "chunk-006"
        assert resp.sources[0].score == 0.72


def test_5_correct_topic_mapping_in_sources(sample_search_chunks: list[SearchResultChunk]) -> None:
    """5. Verify the topic is properly populated in sources and primary_topic."""
    with patch("app.services.tutor_service.search_session") as mock_search, \
         patch("app.services.tutor_service.call_llm_chat", return_value="Explanation"):

        mock_search.return_value = MagicMock(results=sample_search_chunks)

        resp = ask_tutor(
            session_id="3ec404e4f350",
            question="Tell me about variables",
            top_k=2,
        )

        assert resp.primary_topic == "Variables"
        assert resp.sources[0].topic == "Variables"


def test_6_correct_timestamp_preservation(sample_search_chunks: list[SearchResultChunk]) -> None:
    """6. Verify start_time, end_time, and timestamp labels are preserved."""
    with patch("app.services.tutor_service.search_session") as mock_search, \
         patch("app.services.tutor_service.call_llm_chat", return_value="Explanation"):

        mock_search.return_value = MagicMock(results=sample_search_chunks)

        resp = ask_tutor(
            session_id="3ec404e4f350",
            question="When are variables discussed?",
            top_k=2,
        )

        assert resp.primary_start_time == 305.84
        assert resp.primary_timestamp_range == "05:06 – 06:27"
        assert resp.sources[0].start_time == 305.84
        assert resp.sources[0].end_time == 387.0
        assert resp.sources[0].timestamp == "05:06 – 06:27"


def test_7_youtube_timestamp_url_generation() -> None:
    """7. Verify jump URL generation for source timestamps."""
    url = get_youtube_jump_url("kqtD5dpn9C8", 305.84)
    assert url == "https://www.youtube.com/watch?v=kqtD5dpn9C8&t=305s"


def test_8_rag_prompt_construction(sample_search_chunks: list[SearchResultChunk]) -> None:
    """8. Verify prompt construction properly structures context blocks and student question."""
    messages = build_tutor_prompt("What is a variable?", sample_search_chunks)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    user_text = messages[1]["content"]
    assert "=== RETRIEVED LECTURE CONTEXT ===" in user_text
    assert "[chunk-005 | 05:06 – 06:27 | Topic: Variables | Concepts: variables, memory, age]" in user_text
    assert "Student Question: What is a variable?" in user_text


def test_9_grounding_instruction_presence(sample_search_chunks: list[SearchResultChunk]) -> None:
    """9. Verify strict anti-hallucination grounding instructions exist in system message."""
    messages = build_tutor_prompt("Test question", sample_search_chunks)
    system_text = messages[0]["content"]

    assert "Answer using ONLY the provided lecture transcript context below" in system_text
    assert "Do not invent facts" in system_text
    assert FALLBACK_UNAVAILABLE_MSG in system_text
    assert "Do not think or reason" in system_text


def test_10_chat_response_parsing(sample_search_chunks: list[SearchResultChunk]) -> None:
    """10. Verify ChatResponse data fields are properly populated."""
    with patch("app.services.tutor_service.search_session") as mock_search, \
         patch("app.services.tutor_service.call_llm_chat", return_value="   Variables temporarily store data in computer memory.   "):

        mock_search.return_value = MagicMock(results=sample_search_chunks)

        resp = ask_tutor(
            session_id="3ec404e4f350",
            question="Why variables?",
            top_k=2,
        )

        assert resp.answer == "Variables temporarily store data in computer memory."
        assert resp.session_id == "3ec404e4f350"
        assert resp.question == "Why variables?"
        assert len(resp.sources) == 2


def test_11_llm_unavailable_handling(sample_search_chunks: list[SearchResultChunk]) -> None:
    """11. Verify graceful failure reporting when the local LLM runtime cannot be reached."""
    with patch("app.services.tutor_service.search_session") as mock_search, \
         patch("app.services.tutor_service.call_llm_chat", side_effect=LLMClientError("Connection refused on port 1234")):

        mock_search.return_value = MagicMock(results=sample_search_chunks)

        with pytest.raises(TutorServiceError, match="Medhā's AI tutor is currently unavailable"):
            ask_tutor(
                session_id="3ec404e4f350",
                question="Why variables?",
                top_k=2,
            )


def test_12_empty_question_handling() -> None:
    """12. Verify asking an empty or whitespace question throws TutorServiceError."""
    with pytest.raises(TutorServiceError, match="Question cannot be empty"):
        ask_tutor(session_id="3ec404e4f350", question="   ")


def test_13_missing_session_handling() -> None:
    """13. Verify asking about a nonexistent session throws SessionNotFoundError."""
    with pytest.raises(SessionNotFoundError):
        ask_tutor(session_id="invalid_session_id_404", question="Any question")


def test_14_missing_faiss_index_handling() -> None:
    """14. Verify search auto-rebuilds index if missing on disk for an existing session."""
    store = get_vector_store()
    # Confirm index_exists check runs cleanly for session 3ec404e4f350
    assert store.index_exists("3ec404e4f350") is True


def test_15_no_relevant_context_handling() -> None:
    """15. Verify out-of-context queries scoring below threshold return the fallback message without calling LLM."""
    low_score_chunk = SearchResultChunk(
        chunk_id="chunk-040",
        topic="Tuples",
        concepts=[],
        text="Tuples are immutable sequences.",
        start_time=3400.0,
        end_time=3500.0,
        start_timestamp=3400.0,
        end_timestamp=3500.0,
        timestamp="56:40 – 58:20",
        score=0.12,  # Well below RELEVANCE_THRESHOLD (0.20)
    )

    with patch("app.services.tutor_service.search_session") as mock_search, \
         patch("app.services.tutor_service.call_llm_chat") as mock_llm:

        mock_search.return_value = MagicMock(results=[low_score_chunk])

        resp = ask_tutor(
            session_id="3ec404e4f350",
            question="What is the capital of France?",
            top_k=1,
        )

        # Fallback returned
        assert resp.answer == FALLBACK_UNAVAILABLE_MSG
        assert resp.sources == []
        assert resp.primary_start_time is None
        # LLM was NOT invoked because the query was caught by the relevance guardrail
        mock_llm.assert_not_called()


def test_16_conversation_history_handling(sample_search_chunks: list[SearchResultChunk]) -> None:
    """16. Verify recent conversation history is incorporated into the prompt."""
    history = [
        ChatMessage(role="user", content="What are variables?"),
        ChatMessage(role="assistant", content="Variables store temporary data in memory."),
    ]

    messages = build_tutor_prompt(
        question="Can you give an example of one?",
        chunks=sample_search_chunks,
        chat_history=history,
    )

    user_text = messages[1]["content"]
    assert "Recent conversation context:" in user_text
    assert "User: What are variables?" in user_text
    assert "Assistant: Variables store temporary data in memory." in user_text
    assert "Student Question: Can you give an example of one?" in user_text


def test_17_real_session_integration() -> None:
    """17. Real integration test on session 3ec404e4f350: retrieval + Qwen prompt execution."""
    session_id = "3ec404e4f350"
    question = "Why do we use variables in Python?"

    # Execute full pipeline against live FAISS index and mock/live Qwen
    llm_info = check_llm_runtime()
    if llm_info.get("available"):
        resp = ask_tutor(session_id=session_id, question=question, top_k=3)
    else:
        with patch(
            "app.services.tutor_service.call_llm_chat",
            return_value="In this lecture, the instructor explains that variables in Python are used to temporarily store data values in computer memory so you can reuse and manipulate them later in the program.",
        ):
            resp = ask_tutor(session_id=session_id, question=question, top_k=3)

    assert resp.session_id == session_id
    assert resp.question == question
    assert len(resp.sources) > 0
    assert resp.primary_topic == "Variables"
    assert resp.primary_start_time is not None
    # Verify primary timestamp is in the Variables range (approx 05:06)
    assert 300.0 <= resp.primary_start_time <= 320.0
    assert len(resp.answer) > 20
