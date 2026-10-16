"""Tests for Phase 8.2: Local Gemma 4 E2B Translation and Context Guarding."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.schemas import Chunk, TopicGroup
from app.services.content_generator import build_topic_prompt
from app.services.gemma_translation_service import (
    GemmaTranslationService,
    GemmaUnavailableError,
    get_gemma_translation_service,
)
from app.services.llm_client import check_llm_runtime, resolve_model_id
from app.services.translation_service import detect_language


def test_01_language_detection() -> None:
    """1. Test script-based detection for Indic languages and English fallback."""
    # Hindi (Devanagari)
    hindi_text = "पायथन में वैरिएबल्स का उपयोग डेटा स्टोर करने के लिए किया जाता है।"
    assert detect_language(hindi_text) == "hi"

    # Kannada
    kannada_text = "ಪೈಥಾನ್‌ನಲ್ಲಿ ವೇರಿಯೇಬಲ್‌ಗಳನ್ನು ಡೇಟಾ ಸಂಗ್ರಹಿಸಲು ಬಳಸಲಾಗುತ್ತದೆ."
    assert detect_language(kannada_text) == "kn"

    # Telugu
    telugu_text = "పైథాన్‌లో వేరియబుల్స్ డేటాను నిల్వ చేయడానికి ఉపయోగిస్తారు."
    assert detect_language(telugu_text) == "te"

    # Tamil
    tamil_text = "பைத்தானில் மாறிகள் தரவை சேமிக்க பயன்படுகின்றன."
    assert detect_language(tamil_text) == "ta"

    # Malayalam
    malayalam_text = "പൈത്തണിൽ ഡാറ്റ സംഭരിക്കുന്നതിന് വേരിയബിളുകൾ ഉപയോഗിക്കുന്നു."
    assert detect_language(malayalam_text) == "ml"

    # English
    english_text = "In Python, variables are used to store data in computer memory."
    assert detect_language(english_text) == "en"

    # Hint fallback
    assert detect_language("Short text", hint="hi") == "hi"
    assert detect_language("", hint="kn") == "kn"


def test_02_gemma_english_bypass() -> None:
    """2. Test English transcripts bypass LLM calls completely."""
    chunks = [
        Chunk(
            chunk_id="chunk-001",
            topic="Introduction",
            concepts=["variables"],
            text="Variables store data in memory.",
            start_timestamp=0.0,
            end_timestamp=30.0,
            start_label="00:00",
            end_label="00:30",
            word_count=5,
        )
    ]

    service = GemmaTranslationService()
    with patch("app.services.gemma_translation_service.call_llm_chat") as mock_call:
        result = service.translate_chunks(chunks, source_lang="en", session_id="test_sess_bypass")
        assert not mock_call.called
        assert len(result) == 1
        assert result[0].original_text == "Variables store data in memory."
        assert result[0].translated_text == "Variables store data in memory."
        assert result[0].text == "Variables store data in memory."
        assert result[0].source_language == "en"


def test_03_gemma_segment_translation() -> None:
    """3. Test translating a non-English segment using Gemma with strict prompt."""
    service = GemmaTranslationService(model_id="gemma-4-e2b-it-qat")

    with patch("app.services.gemma_translation_service.call_llm_chat") as mock_call:
        mock_call.return_value = "In Python, we use variables to store data."

        translated = service.translate_segment(
            "पायथन में वैरिएबल्स का उपयोग डेटा स्टोर करने के लिए किया जाता है।",
            source_lang="hi",
        )

        assert translated == "In Python, we use variables to store data."
        assert mock_call.called
        # Verify model='gemma-4-e2b-it-qat' was requested
        _, kwargs = mock_call.call_args
        assert kwargs.get("model") == "gemma-4-e2b-it-qat"


def test_04_gemma_preserves_timestamps_and_chunk_ids() -> None:
    """4. Test that chunk IDs, timestamps, labels, and original texts are strictly preserved."""
    chunks = [
        Chunk(
            chunk_id="chunk-042",
            topic="Variables",
            concepts=["memory", "identifiers"],
            text="पायथन में डेटा सुरक्षित रखने के लिए वैरिएबल बनते हैं।",
            start_timestamp=125.5,
            end_timestamp=165.0,
            start_label="02:05",
            end_label="02:45",
            word_count=9,
        )
    ]

    service = GemmaTranslationService(model_id="gemma-4-e2b-it-qat")
    with patch.object(service, "translate_segment", return_value="In Python, variables are created to keep data safe."):
        translated_chunks = service.translate_chunks(chunks, source_lang="hi", session_id="test_sess_preserve")

        assert len(translated_chunks) == 1
        tc = translated_chunks[0]
        assert tc.chunk_id == "chunk-042"
        assert tc.start_timestamp == 125.5
        assert tc.end_timestamp == 165.0
        assert tc.start_label == "02:05"
        assert tc.end_label == "02:45"
        assert tc.original_text == "पायथन में डेटा सुरक्षित रखने के लिए वैरिएबल बनते हैं।"
        assert tc.translated_text == "In Python, variables are created to keep data safe."
        assert tc.text == "In Python, variables are created to keep data safe."
        assert tc.source_language == "hi"


def test_05_translation_disk_caching(tmp_path: Path) -> None:
    """5. Test that translated chunks are cached to disk and loaded on subsequent calls."""
    chunks = [
        Chunk(
            chunk_id="chunk-001",
            topic="Lists",
            concepts=["arrays"],
            text="सूचियाँ कई आइटम संग्रहीत करती हैं।",
            start_timestamp=0.0,
            end_timestamp=10.0,
            start_label="00:00",
            end_label="00:10",
            word_count=5,
        )
    ]

    session_id = "test_cache_sess_001"
    cache_file = tmp_path / f"{session_id}.json"

    service = GemmaTranslationService(model_id="gemma-4-e2b-it-qat")
    with patch.object(service, "_cache_path", return_value=cache_file), \
         patch.object(service, "translate_segment", return_value="Lists store multiple items.") as mock_trans:

        # First run: translates and writes cache
        res1 = service.translate_chunks(chunks, source_lang="hi", session_id=session_id)
        assert mock_trans.call_count == 1
        assert cache_file.exists()
        assert res1[0].translated_text == "Lists store multiple items."

        # Second run: reads from cache without translating
        mock_trans.reset_mock()
        res2 = service.translate_chunks(chunks, source_lang="hi", session_id=session_id)
        assert mock_trans.call_count == 0
        assert len(res2) == 1
        assert res2[0].translated_text == "Lists store multiple items."


def test_06_gemma_unavailable_error() -> None:
    """6. Test that if Gemma is not loaded, GemmaUnavailableError is raised."""
    service = GemmaTranslationService()
    with patch("app.services.gemma_translation_service.resolve_model_id", return_value=None):
        with pytest.raises(GemmaUnavailableError):
            service.translate_segment("नमस्ते", source_lang="hi")


def test_07_context_budget_guard() -> None:
    """7. Test build_topic_prompt bounds context words and caps character length at 5000 chars."""
    topic = TopicGroup(
        topic_id="topic-001",
        name="Variables & Storage",
        timestamp_range="00:00 – 10:00",
        start_time=0.0,
        end_time=600.0,
        start_label="00:00",
        end_label="10:00",
        chunk_ids=[f"chunk-{i:03d}" for i in range(50)],
        key_concepts=["variables", "memory", "types"],
    )

    # 50 chunks with ~50 words each = 2500 words total
    chunks = [
        Chunk(
            chunk_id=f"chunk-{i:03d}",
            topic="Variables & Storage",
            concepts=["variables"],
            text=f"This is chunk {i} discussing in detail how Python allocates memory and stores variables " * 5,
            start_timestamp=float(i * 10),
            end_timestamp=float((i + 1) * 10),
            start_label=f"00:{i:02d}",
            end_label=f"00:{i+1:02d}",
            word_count=50,
        )
        for i in range(50)
    ]

    messages = build_topic_prompt(topic, chunks, language="en")
    user_prompt = messages[1]["content"]

    # Must contain context block header and end markers
    assert "=== LECTURE TRANSCRIPT CONTEXT ===" in user_prompt
    assert "=== END OF CONTEXT ===" in user_prompt

    # Total user prompt must be well bounded (prompt words must be under ~1200 words)
    prompt_words = len(user_prompt.split())
    assert prompt_words < 1200

    # Ensure total prompt character length is strictly safe (well under 6500 chars)
    assert len(user_prompt) < 6500


def test_08_resolve_model_id() -> None:
    """8. Test resolve_model_id maps aliases and IDs for both Qwen and Gemma."""
    available = ["gemma-4-e2b-it-qat", "qwen/qwen3-8b", "text-embedding-nomic-embed-text-v1.5"]

    assert resolve_model_id("gemma", available_models=available) == "gemma-4-e2b-it-qat"
    assert resolve_model_id("qwen", available_models=available) == "qwen/qwen3-8b"
    assert resolve_model_id("unknown", available_models=available) is None
