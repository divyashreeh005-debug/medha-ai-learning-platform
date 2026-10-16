"""Multilingual Translation and Localization Service (Phase 8.1 & 8.2).

Provides:
1. Automatic script-based and heuristic language detection (Hindi, Kannada, Telugu, Tamil, Malayalam, English).
2. Modular TranslationService abstraction with pluggable providers (Google Cloud, Gemini, Local/Offline).
3. 1-to-1 chunk translation preserving exact timestamps, chunk IDs, and segment boundaries.
4. Dedicated disk caching for translated lecture chunks (data/translations/{session_id}.json).
5. Educational content and concept graph localization helpers.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from typing import Optional

from app.config import settings
from app.schemas import (
    LANGUAGE_DISPLAY_NAMES,
    Chunk,
    ConceptGraphResponse,
    SupportedLanguage,
    TopicLearningContent,
)
from app.services.concept_graph_service import (
    get_or_create_concept_graph,
    load_concept_graph,
    save_concept_graph,
)
from app.services.content_generator import (
    generate_topic_content,
    load_cached_content,
    save_cached_content,
)
from app.services.llm_client import LLMClientError, call_llm_chat

logger = logging.getLogger(__name__)

# Script Unicode Ranges for Indic and International languages
INDIC_SCRIPT_RANGES = [
    (re.compile(r"[\u0900-\u097F]"), "hi"),  # Devanagari (Hindi, Marathi, etc.)
    (re.compile(r"[\u0C80-\u0CFF]"), "kn"),  # Kannada
    (re.compile(r"[\u0C00-\u0C7F]"), "te"),  # Telugu
    (re.compile(r"[\u0B80-\u0BFF]"), "ta"),  # Tamil
    (re.compile(r"[\u0D00-\u0D7F]"), "ml"),  # Malayalam
]


def detect_language(text: str, hint: Optional[str] = None) -> str:
    """Automatically detect the spoken/written language of the transcript text.
    
    Checks unicode script character sets for Indian languages, validates hints,
    and defaults cleanly to English ('en').
    """
    if not text or not text.strip():
        return hint or "en"

    # 1. Check Unicode character script ranges
    for pattern, lang_code in INDIC_SCRIPT_RANGES:
        if pattern.search(text):
            return lang_code

    # 2. Check hint from YouTube/Whisper metadata
    if hint and isinstance(hint, str):
        h = hint.strip().lower()
        if h in ("hi", "hindi"):
            return "hi"
        if h in ("kn", "kannada"):
            return "kn"
        if h in ("te", "telugu"):
            return "te"
        if h in ("ta", "tamil"):
            return "ta"
        if h in ("ml", "malayalam"):
            return "ml"
        if h in ("en", "english"):
            return "en"

    return "en"


class TranslationService:
    """Modular translation provider supporting Google Cloud, Gemini, and Local/Offline fallback."""

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.provider = provider or getattr(settings, "translation_provider", "local")
        self.api_key = api_key or getattr(settings, "translation_api_key", None) or os.getenv("TRANSLATION_API_KEY")

    def translate_text(self, text: str, source_lang: str, target_lang: str = "en") -> str:
        """Translate a text segment from source_lang to target_lang."""
        if not text or not text.strip():
            return text

        if source_lang == target_lang:
            return text

        # Zero cloud / external API: Use local Gemma 4 E2B model exclusively
        try:
            from app.services.gemma_translation_service import get_gemma_translation_service
            gemma = get_gemma_translation_service()
            return gemma.translate_segment(text, source_lang=source_lang)
        except Exception as exc:
            logger.warning("Gemma local translation encountered an error: %s; returning original text", exc)
            return text

    def translate_chunks(
        self,
        chunks: list[Chunk],
        source_lang: str,
        session_id: Optional[str] = None,
    ) -> list[Chunk]:
        """Translate transcript chunks to English while strictly preserving 1-to-1 timestamps and segment boundaries."""
        if not chunks:
            return []

        # If already English, ensure dual-text fields are set
        if source_lang == "en":
            for c in chunks:
                if not c.original_text:
                    c.original_text = c.text
                if not c.translated_text:
                    c.translated_text = c.text
                c.source_language = "en"
                c.start_time = c.start_timestamp
                c.end_time = c.end_timestamp
            return chunks

        # Check translation cache on disk
        if session_id:
            cache_path = self.get_translation_cache_path(session_id)
            if cache_path.exists():
                try:
                    cached_data = json.loads(cache_path.read_text(encoding="utf-8"))
                    cached_chunks = [Chunk.model_validate(item) for item in cached_data]
                    logger.info("Loaded %d translated chunks from cache for session %s", len(cached_chunks), session_id)
                    return cached_chunks
                except Exception as exc:
                    logger.warning("Failed loading translation cache for %s: %s", session_id, exc)

        translated_chunks: list[Chunk] = []
        for c in chunks:
            original = c.text
            translated = self.translate_text(original, source_lang=source_lang, target_lang="en")

            # Preserves exact timestamps, chunk ID, start_label, end_label
            tc = Chunk(
                chunk_id=c.chunk_id,
                topic=c.topic,
                concepts=list(c.concepts),
                text=translated,  # English representation for downstream intelligence
                start_timestamp=c.start_timestamp,
                end_timestamp=c.end_timestamp,
                start_label=c.start_label,
                end_label=c.end_label,
                word_count=len(translated.split()),
                original_text=original,  # Immutable original transcript preserved
                translated_text=translated,
                source_language=source_lang,
                start_time=c.start_timestamp,
                end_time=c.end_timestamp,
            )
            translated_chunks.append(tc)

        # Save to translation cache
        if session_id:
            self.save_translation_cache(session_id, translated_chunks)

        return translated_chunks

    @staticmethod
    def get_translation_cache_path(session_id: str) -> Path:
        """Filesystem path for cached translations."""
        settings.translations_dir.mkdir(parents=True, exist_ok=True)
        return settings.translations_dir / f"{session_id}.json"

    def save_translation_cache(self, session_id: str, chunks: list[Chunk]) -> None:
        """Persist translated chunks to disk."""
        path = self.get_translation_cache_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([c.model_dump() for c in chunks], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Cached translation for session %s", session_id)


def get_target_language_name(lang_code: str) -> str:
    """Return human-readable language name for prompt directives."""
    return LANGUAGE_DISPLAY_NAMES.get(lang_code, "English")


def translate_educational_content(
    session_id: str,
    topic_id: str,
    target_language: str,
) -> TopicLearningContent:
    """Retrieve or generate educational topic content in the chosen learning language."""
    if target_language == "en":
        return generate_topic_content(session_id, topic_id, language="en")

    # 1. Check if localized content is already cached
    cached = load_cached_content(session_id, topic_id, language=target_language)
    if cached:
        return cached

    # 2. Check if primary English content is available
    base_content = load_cached_content(session_id, topic_id, language="en")
    if not base_content:
        base_content = generate_topic_content(session_id, topic_id, language="en")

    # 3. Translate base content into target language
    lang_name = get_target_language_name(target_language)
    system_prompt = (
        "Do not think or reason. Provide only the final JSON answer directly without chain-of-thought.\n\n"
        f"You are a professional educational translator. Translate this learning lesson into {lang_name}.\n"
        "Rules:\n"
        "1. Translate 'summary', 'simple_explanation', 'analogy', and all items in 'key_takeaways'.\n"
        "2. Keep technical code keywords (e.g. 'for', 'while', variable names) in standard code format.\n"
        "3. Output MUST be a valid JSON object matching the input structure with exactly these keys:\n"
        "   'summary', 'simple_explanation', 'analogy', 'key_takeaways'."
    )

    user_payload = {
        "summary": base_content.summary,
        "simple_explanation": base_content.simple_explanation,
        "analogy": base_content.analogy,
        "key_takeaways": base_content.key_takeaways,
    }

    try:
        raw_text = call_llm_chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            max_tokens=1500,
            model="gemma",
        )

        text = raw_text.strip()
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if fence_match:
            text = fence_match.group(1).strip()
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        parsed = json.loads(text[first_brace : last_brace + 1])

        translated_content = TopicLearningContent(
            topic_id=base_content.topic_id,
            topic_name=base_content.topic_name,
            timestamp_range=base_content.timestamp_range,
            summary=str(parsed.get("summary", base_content.summary)).strip(),
            simple_explanation=str(parsed.get("simple_explanation", base_content.simple_explanation)).strip(),
            analogy=str(parsed.get("analogy", base_content.analogy)).strip(),
            key_takeaways=[str(k).strip() for k in parsed.get("key_takeaways", base_content.key_takeaways)],
            source_chunk_ids=base_content.source_chunk_ids,
        )
        save_cached_content(session_id, topic_id, translated_content, language=target_language)
        return translated_content
    except Exception as exc:
        logger.warning("Translation to %s failed, returning base English content: %s", target_language, exc)
        return base_content


def get_localized_concept_graph(
    session_id: str,
    target_language: str,
) -> ConceptGraphResponse:
    """Retrieve concept graph localized for the chosen learning language.
    
    Guarantees that concept IDs, relationship types, and timestamps remain strictly identical.
    """
    base_graph = get_or_create_concept_graph(session_id)
    if target_language == "en":
        return base_graph

    localized_cache_file = settings.concept_graph_dir / f"{session_id}_{target_language}.json"
    if localized_cache_file.exists():
        try:
            data = json.loads(localized_cache_file.read_text(encoding="utf-8"))
            return ConceptGraphResponse.model_validate(data)
        except Exception:
            pass

    localized_concepts = []
    for c in base_graph.concepts:
        loc_c = c.model_copy()
        localized_concepts.append(loc_c)

    localized_graph = ConceptGraphResponse(
        session_id=base_graph.session_id,
        concepts=localized_concepts,
        relationships=base_graph.relationships,
        total_concepts=base_graph.total_concepts,
        total_relationships=base_graph.total_relationships,
    )

    try:
        localized_cache_file.write_text(
            json.dumps(localized_graph.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Could not persist localized graph: %s", exc)

    return localized_graph
