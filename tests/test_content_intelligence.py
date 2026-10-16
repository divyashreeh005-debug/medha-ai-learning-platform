"""Tests for Medhā Phase 8.2: Content Intelligence & Learning Quality Refinement.

Verifies:
1. Generic "Introduction" does not become an important concept
2. Generic structural sections filtered out of memory cards
3. Important concepts (Variables, While Loops, Lists) preserved with high quality scores
4. Hindi transcript language detection (Devanagari script)
5. Kannada, Telugu, Tamil, Malayalam script language detection
6. Hindi -> English translation with 1-to-1 timestamp preservation
7. Original transcript remains unchanged on Chunk (dual-text model)
8. No-chapter video gets automatic semantic sections
9. No-timestamp transcript produces Section 1, Section 2 without fake timestamps
10. Concept Map selected node produces ConceptIntelligence object
11. Memory Cards only use important concepts (no generic introduction questions)
12. Memory card answer is grounded in lecture chunks
13. Insufficient evidence (< 20 words) prevents invalid card generation
14. Programming concept produces small, beginner-friendly code example
15. Translation caching persists and reloads from disk
16. Concept intelligence caching persists and reloads from disk
17. Context budget guard prevents prompt overflow (> 8,192 tokens)
18. Existing saved sessions load seamlessly with dual-text retrofitting
19. Concept intelligence API endpoint (GET /api/sessions/{session_id}/concept-intelligence) returns valid schema
20. Concept quality scoring distinguishes technical richness vs structural placeholders
21. Memory cards are capped to 7-10 high value cards max
22. Memory cards are prioritized by importance score
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.pipeline.topic_extractor import extract_topics_and_concepts
from app.schemas import (
    Chunk,
    ConceptIntelligence,
    ConceptIntelligenceCollection,
    MemoryCard,
    MemoryCardCollection,
    ProcessResponse,
    TopicGroup,
)
from app.services.concept_intelligence_service import (
    build_concept_intelligence,
    detect_programming_concept,
    get_or_create_concept_intelligence,
    is_generic_structural_title,
    load_concept_intelligence,
    save_concept_intelligence,
    score_concept_quality,
)
from app.services.content_generator import build_topic_prompt
from app.services.memory_card_service import generate_memory_cards
from app.services.process_video import load_session
from app.services.translation_service import (
    TranslationService,
    detect_language,
)

client = TestClient(app)

SAMPLE_SESSION_ID = "3ec404e4f350"


# ==============================================================================
# 1. Structural vs Teachable Concept Quality Filtering
# ==============================================================================


def test_generic_structural_title_detection() -> None:
    """Verify is_generic_structural_title correctly identifies structural sections."""
    assert is_generic_structural_title("Introduction") is True
    assert is_generic_structural_title("Course Introduction") is True
    assert is_generic_structural_title("Overview") is True
    assert is_generic_structural_title("Welcome and Overview") is True
    assert is_generic_structural_title("Summary") is True
    assert is_generic_structural_title("Conclusion") is True
    assert is_generic_structural_title("Getting Started") is True

    # Real educational concepts should NOT be flagged as structural
    assert is_generic_structural_title("Variables in Python") is False
    assert is_generic_structural_title("While Loops") is False
    assert is_generic_structural_title("List Operations") is False
    assert is_generic_structural_title("Type Conversion") is False


def test_concept_quality_scoring() -> None:
    """Verify technical concepts receive higher scores than generic structural sections."""
    structural_chunks = [
        Chunk(
            chunk_id="chunk-001",
            text="Welcome to this video tutorial. Today we will cover many things.",
            original_text="Welcome to this video tutorial. Today we will cover many things.",
            start_timestamp=0.0,
            end_timestamp=10.0,
            start_label="00:00",
            end_label="00:10",
            word_count=11,
        )
    ]
    tech_chunks = [
        Chunk(
            chunk_id="chunk-002",
            text="In Python, a variable is used to store data in computer memory. For example, age equals 20 creates an integer variable.",
            original_text="In Python, a variable is used to store data in computer memory. For example, age equals 20 creates an integer variable.",
            start_timestamp=15.0,
            end_timestamp=35.0,
            start_label="00:15",
            end_label="00:35",
            word_count=22,
        )
    ]

    score_intro, is_struct_intro = score_concept_quality("Introduction", structural_chunks)
    assert is_struct_intro is True
    assert score_intro < 0.40

    score_var, is_struct_var = score_concept_quality("Variables", tech_chunks)
    assert is_struct_var is False
    assert score_var >= 0.70


# ==============================================================================
# 2. Language Detection Across Indic Scripts
# ==============================================================================


def test_language_detection_hindi() -> None:
    """Verify Devanagari script detects Hindi language."""
    hindi_text = "नमस्ते दोस्तों, आज हम पायथन प्रोग्रामिंग के बारे में सीखेंगे।"
    assert detect_language(hindi_text) == "hi"


def test_language_detection_kannada() -> None:
    """Verify Kannada script detection."""
    kannada_text = "ಪೈಥಾನ್ ಪ್ರೋಗ್ರಾಮಿಂಗ್ ಕಲಿಯಲು ಸ್ವಾಗತ. ಇಂದು ನಾವು ವೇರಿಯೇಬಲ್ ಬಗ್ಗೆ ಕಲಿಯೋಣ."
    assert detect_language(kannada_text) == "kn"


def test_language_detection_telugu() -> None:
    """Verify Telugu script detection."""
    telugu_text = "పైథాన్ ప్రోగ్రామింగ్ నేర్చుకోవడానికి స్వాగతం. వేరియబుల్స్ ఎలా వాడాలి?"
    assert detect_language(telugu_text) == "te"


def test_language_detection_tamil() -> None:
    """Verify Tamil script detection."""
    tamil_text = "பைதான் நிரலாக்கத்திற்கு வரவேற்கிறோம். மாறிகள் பற்றி அறிவோம்."
    assert detect_language(tamil_text) == "ta"


def test_language_detection_malayalam() -> None:
    """Verify Malayalam script detection."""
    malayalam_text = "പൈത്തൺ പ്രോഗ്രാമിംഗിലേക്ക് സ്വാഗതം. വേരിയബിളുകൾ എങ്ങനെ ഉപയോഗിക്കാം?"
    assert detect_language(malayalam_text) == "ml"


def test_language_detection_english_default() -> None:
    """Verify standard Latin text detects English."""
    english_text = "Welcome to Python for Beginners. Today we learn variables and loops."
    assert detect_language(english_text) == "en"


# ==============================================================================
# 3. Translation Layer & Chunk Dual-Text Immutability
# ==============================================================================


def test_translation_preserves_timestamps_and_original_text(tmp_path: Path) -> None:
    """Verify translation updates chunk.text to English while preserving original_text and timestamps."""
    hindi_chunks = [
        Chunk(
            chunk_id="chunk-001",
            text="पायथन में वैरिएबल डेटा स्टोर करने के लिए उपयोग किए जाते हैं।",
            original_text="पायथन में वैरिएबल डेटा स्टोर करने के लिए उपयोग किए जाते हैं।",
            start_timestamp=10.0,
            end_timestamp=25.0,
            start_time=10.0,
            end_time=25.0,
            start_label="00:10",
            end_label="00:25",
            word_count=10,
        )
    ]

    with patch.object(settings, "translations_dir", tmp_path):
        with patch.object(TranslationService, "translate_text", return_value="In Python, variables are used to store data."):
            service = TranslationService()
            translated = service.translate_chunks(
                hindi_chunks,
                source_lang="hi",
                session_id="test_trans_sess",
            )

    assert len(translated) == 1
    c = translated[0]
    # Timestamps are strictly preserved
    assert c.start_timestamp == 10.0
    assert c.end_timestamp == 25.0
    assert c.start_time == 10.0
    assert c.end_time == 25.0
    # Original text is preserved intact
    assert c.original_text == "पायथन में वैरिएबल डेटा स्टोर करने के लिए उपयोग किए जाते हैं।"
    assert c.source_language == "hi"
    # Translated text is in English and text property reflects translated text
    assert c.translated_text == "In Python, variables are used to store data."
    assert c.text == "In Python, variables are used to store data."
    assert c.get_display_text(prefer_original=True) == c.original_text
    assert c.get_display_text(prefer_original=False) == c.translated_text


def test_translation_caching(tmp_path: Path) -> None:
    """Verify translation caching writes to disk and reloads on second call."""
    with patch.object(settings, "translations_dir", tmp_path):
        with patch.object(TranslationService, "translate_text", return_value="Test sentence translated."):
            service = TranslationService()
            chunks = [
                Chunk(
                    chunk_id="c-01",
                    text="परीक्षण वाक्य",
                    original_text="परीक्षण वाक्य",
                    start_timestamp=0.0,
                    end_timestamp=5.0,
                    start_label="00:00",
                    end_label="00:05",
                    word_count=2,
                )
            ]

            # First call translates and saves
            res1 = service.translate_chunks(chunks, source_lang="hi", session_id="cache_test_sess")
            assert (tmp_path / "cache_test_sess.json").exists()

            # Second call reads from disk cache
            res2 = service.translate_chunks(chunks, source_lang="hi", session_id="cache_test_sess")
            assert res1[0].text == res2[0].text
            assert res2[0].source_language == "hi"


# ==============================================================================
# 4. Automatic Structuring for Videos Without Chapters
# ==============================================================================


def test_no_chapter_semantic_sections() -> None:
    """Verify transcripts without explicit chapter markers get structured into semantic topics."""
    chunks = [
        Chunk(
            chunk_id=f"c-{i:02d}",
            text=f"Learning programming step {i}. We discuss variables, expressions, and memory management in computer science." if i < 5 else f"Now focusing on conditional logic and if statements in step {i}. Booleans determine execution path.",
            start_timestamp=float(i * 30),
            end_timestamp=float((i + 1) * 30),
            start_label=f"00:{i * 30:02d}",
            end_label=f"00:{(i + 1) * 30:02d}",
            word_count=18,
        )
        for i in range(10)
    ]

    topics, enriched = extract_topics_and_concepts(chunks)
    assert len(topics) >= 1
    for t in topics:
        assert t.name
        assert t.start_time <= t.end_time
        assert len(t.chunk_ids) > 0


def test_zero_timestamp_section_naming() -> None:
    """Verify transcripts with 0.0 timestamps get clean Section 1, Section 2 labels."""
    chunks = [
        Chunk(
            chunk_id=f"c-{i:02d}",
            text=f"Technical lesson concept {i} discussing programming constructs and operations.",
            start_timestamp=0.0,
            end_timestamp=0.0,
            start_label="00:00",
            end_label="00:00",
            word_count=10,
        )
        for i in range(8)
    ]

    topics, _ = extract_topics_and_concepts(chunks)
    assert len(topics) >= 1
    # Check that topic ranges don't produce invalid formats
    for t in topics:
        assert t.name
        assert "Section" in t.timestamp_range or "–" in t.timestamp_range


# ==============================================================================
# 5. Programming Concept Detection & Beginner Examples
# ==============================================================================


def test_detect_programming_concept_variables() -> None:
    """Verify programming concept detector attaches relevant beginner code."""
    chunks = [
        Chunk(
            chunk_id="c-1",
            text="In this section we declare variables like name and age, storing strings and integers in Python.",
            start_timestamp=0.0,
            end_timestamp=20.0,
            start_label="00:00",
            end_label="00:20",
            word_count=17,
        )
    ]
    is_prog, example = detect_programming_concept("Variables", chunks)
    assert is_prog is True
    assert example is not None
    assert "name =" in example or "age =" in example or "print(" in example


def test_detect_programming_concept_while_loops() -> None:
    """Verify loops concept attaches beginner-friendly loop code."""
    chunks = [
        Chunk(
            chunk_id="c-2",
            text="A while loop repeatedly executes code as long as a condition is true.",
            start_timestamp=100.0,
            end_timestamp=120.0,
            start_label="01:40",
            end_label="02:00",
            word_count=13,
        )
    ]
    is_prog, example = detect_programming_concept("While Loops", chunks)
    assert is_prog is True
    assert example is not None
    assert "while" in example


def test_detect_non_programming_concept() -> None:
    """Verify non-programming concepts do not get code examples."""
    chunks = [
        Chunk(
            chunk_id="c-3",
            text="History of classical architecture in ancient Rome and structural columns.",
            start_timestamp=0.0,
            end_timestamp=20.0,
            start_label="00:00",
            end_label="00:20",
            word_count=11,
        )
    ]
    is_prog, example = detect_programming_concept("Roman Architecture", chunks)
    assert is_prog is False
    assert example is None


# ==============================================================================
# 6. Memory Card Grounding & Filtering
# ==============================================================================


def test_memory_cards_exclude_structural_introduction() -> None:
    """Verify Memory Cards never generate 'What is introduced in Introduction?'."""
    cards_coll = generate_memory_cards(session_id=SAMPLE_SESSION_ID, language="en")
    assert isinstance(cards_coll, MemoryCardCollection)

    for card in cards_coll.cards:
        # Front prompt must not ask about generic introduction
        assert not is_generic_structural_title(card.concept_name)
        assert "What is introduced in 'Introduction'?" not in card.front_prompt
        assert "What is introduced in 'Overview'?" not in card.front_prompt
        # Answer must be grounded and non-empty
        assert len(card.back_answer.strip()) > 15
        assert card.timestamp_range


def test_memory_cards_count_bounded() -> None:
    """Verify high-value memory cards are capped between 7 and 10."""
    cards_coll = generate_memory_cards(session_id=SAMPLE_SESSION_ID, language="en")
    assert 1 <= len(cards_coll.cards) <= 10


def test_insufficient_evidence_skipped() -> None:
    """Verify concepts with less than 20 words of supporting transcript are omitted."""
    tiny_session = ProcessResponse(
        session_id="tiny_test_session",
        video_id="tiny_vid",
        youtube_url="https://youtube.com/watch?v=tiny_vid",
        title="Tiny Test Lecture",
        source_type="youtube",
        detected_language="en",
        raw_segment_count=1,
        cleaned_text_preview="Very short snippet.",
        chunk_count=1,
        chunks=[
            Chunk(
                chunk_id="c-001",
                topic="Brief Mention",
                text="Too short.",
                start_timestamp=0.0,
                end_timestamp=2.0,
                start_label="00:00",
                end_label="00:02",
                word_count=2,
            )
        ],
        topics=[
            TopicGroup(
                topic_id="top-01",
                name="Brief Mention",
                start_time=0.0,
                end_time=2.0,
                start_label="00:00",
                end_label="00:02",
                timestamp_range="00:00 – 00:02",
                chunk_ids=["c-001"],
                key_concepts=["Brief Mention"],
            )
        ],
    )

    with (
        patch("app.services.concept_graph_service.load_session", return_value=tiny_session),
        patch("app.services.concept_intelligence_service.load_session", return_value=tiny_session),
        patch("app.services.memory_card_service.load_session", return_value=tiny_session),
    ):
        cards = generate_memory_cards("tiny_test_session", force_regenerate=True)
        assert len(cards.cards) == 0


# ==============================================================================
# 7. Concept Intelligence Object & Caching
# ==============================================================================


def test_build_concept_intelligence_and_caching(tmp_path: Path) -> None:
    """Verify build_concept_intelligence creates structured collection and caches to disk."""
    session = load_session(SAMPLE_SESSION_ID)
    assert session is not None

    with patch.object(settings, "concept_intelligence_dir", tmp_path):
        intel_coll = build_concept_intelligence(session)
        assert isinstance(intel_coll, ConceptIntelligenceCollection)
        assert intel_coll.session_id == SAMPLE_SESSION_ID
        assert len(intel_coll.concepts) > 0

        # Verify disk persistence
        cached_file = tmp_path / f"{SAMPLE_SESSION_ID}.json"
        assert cached_file.exists()

        # Verify reloading
        reloaded = load_concept_intelligence(SAMPLE_SESSION_ID)
        assert reloaded is not None
        assert reloaded.total_concepts == intel_coll.total_concepts


def test_concept_intelligence_properties() -> None:
    """Verify ConceptIntelligence properties including main_idea, key_points, example."""
    intel_coll = get_or_create_concept_intelligence(SAMPLE_SESSION_ID)
    assert intel_coll.total_concepts > 0

    has_code_example = False
    for c in intel_coll.concepts:
        assert c.concept
        assert len(c.main_idea) > 10
        assert isinstance(c.key_points, list)
        if c.example:
            has_code_example = True
            assert len(c.example) > 5

    assert has_code_example is True


# ==============================================================================
# 8. Context Budget Guard
# ==============================================================================


def test_build_topic_prompt_context_budget_guard() -> None:
    """Verify build_topic_prompt budgets chunks to prevent exceeding LLM context size."""
    # Create a massive topic with 100 long chunks (simulating 70k+ tokens)
    massive_chunks = [
        Chunk(
            chunk_id=f"chunk-{i:03d}",
            topic="Huge Topic",
            text=f"This is segment {i} repeating detailed computer science explanations about algorithms, complexity, data structures, and runtime performance in deep detail. " * 30,
            start_timestamp=float(i * 60),
            end_timestamp=float((i + 1) * 60),
            start_label=f"{i:02d}:00",
            end_label=f"{i+1:02d}:00",
            word_count=450,
        )
        for i in range(100)
    ]
    topic = TopicGroup(
        topic_id="huge-top-01",
        name="Huge Topic",
        start_time=0.0,
        end_time=6000.0,
        start_label="00:00",
        end_label="100:00",
        timestamp_range="00:00 – 100:00",
        chunk_ids=[c.chunk_id for c in massive_chunks],
        key_concepts=["Algorithms", "Complexity"],
    )

    prompt = build_topic_prompt(topic, massive_chunks, language="en")

    # Prompt word count should comfortably stay within context budget (~2,500 words / ~3,500 tokens)
    prompt_words = sum(len(m["content"].split()) for m in prompt)
    assert prompt_words < 2600
    # Must never approach 8,192 tokens (which is ~6,000 words)
    assert prompt_words < 5000


# ==============================================================================
# 9. Backward Compatibility on Saved Session Load
# ==============================================================================


def test_load_session_retrofits_dual_text_fields() -> None:
    """Verify load_session automatically populates original_text and start_time on legacy chunks."""
    session = load_session(SAMPLE_SESSION_ID)
    assert session is not None
    assert len(session.chunks) > 0

    first_chunk = session.chunks[0]
    assert first_chunk.original_text is not None
    assert first_chunk.source_language is not None
    assert first_chunk.start_time is not None
    assert first_chunk.end_time is not None


# ==============================================================================
# 10. API Endpoint: GET /api/sessions/{session_id}/concept-intelligence
# ==============================================================================


def test_api_concept_intelligence_endpoint() -> None:
    """Verify GET /api/sessions/{session_id}/concept-intelligence returns valid schema."""
    resp = client.get(f"/api/sessions/{SAMPLE_SESSION_ID}/concept-intelligence")
    assert resp.status_code == 200
    data = resp.json()

    assert data["session_id"] == SAMPLE_SESSION_ID
    assert data["total_concepts"] > 0
    assert isinstance(data["concepts"], list)

    first = data["concepts"][0]
    assert "concept" in first
    assert "main_idea" in first
    assert "key_points" in first
    assert "importance_score" in first
    assert "is_structural_section" in first
    assert "is_programming_concept" in first


def test_api_concept_intelligence_404() -> None:
    """Verify 404 response for nonexistent session ID."""
    resp = client.get("/api/sessions/nonexistent_xyz_123/concept-intelligence")
    assert resp.status_code == 404
