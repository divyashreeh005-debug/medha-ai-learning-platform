"""Local Gemma 4 E2B Translation Service for Medhā (Phase 8.2).

Provides dedicated local translation of non-English lecture transcripts into canonical English
using google/gemma-4-E2B-it-qat-q4_0-gguf via local LM Studio.

Guarantees:
1. 100% local execution — ZERO cloud/external APIs.
2. English transcripts bypass Gemma completely (zero latency impact).
3. 1-to-1 timestamp and chunk ID preservation.
4. Immutable original_text preservation.
5. Strict translation prompt prevents explanatory drift or summarization.
6. Persistent session translation caching on disk.
7. Controlled error when Gemma is unavailable or unmapped in LM Studio.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from app.config import settings
from app.schemas import Chunk
from app.services.llm_client import LLMClientError, call_llm_chat, resolve_model_id

logger = logging.getLogger(__name__)


class GemmaUnavailableError(Exception):
    """Raised when Gemma translation model is not loaded in LM Studio."""


class GemmaTranslationError(Exception):
    """Raised when translation processing fails."""


GEMMA_SYSTEM_PROMPT = (
    "You are a transcript translation engine.\n\n"
    "Translate the provided educational transcript into natural English.\n\n"
    "Rules:\n"
    "1. Translate only.\n"
    "2. Do not summarize.\n"
    "3. Do not explain.\n"
    "4. Do not add information.\n"
    "5. Do not remove information.\n"
    "6. Preserve technical terminology.\n"
    "7. Preserve programming keywords.\n"
    "8. Preserve code exactly where possible.\n"
    "9. Preserve numbers and proper nouns.\n"
    "10. Preserve the original meaning.\n"
    "11. Return only the translated text.\n"
    "12. Do not add commentary."
)


class GemmaTranslationService:
    """Orchestrates local Gemma 4 E2B translation with timestamp preservation and caching."""

    def __init__(self, model_id: Optional[str] = None) -> None:
        self._model_id = model_id

    def get_active_model_id(self) -> str:
        """Resolve the active Gemma model ID from LM Studio."""
        if self._model_id:
            return self._model_id
        resolved = resolve_model_id("gemma")
        if not resolved:
            raise GemmaUnavailableError(
                "Gemma translation model is unavailable. Load the Gemma model in LM Studio."
            )
        self._model_id = resolved
        return self._model_id

    def is_available(self) -> bool:
        """Check whether Gemma is loaded and available in LM Studio."""
        try:
            self.get_active_model_id()
            return True
        except GemmaUnavailableError:
            return False

    def translate_segment(self, text: str, source_lang: str) -> str:
        """Translate a single transcript text segment using Gemma."""
        clean = text.strip()
        if not clean or source_lang == "en":
            return clean

        model_id = self.get_active_model_id()

        user_content = (
            f"Source Language: {source_lang}\n"
            f"Transcript Segment:\n{clean}\n\n"
            "English Translation:"
        )

        try:
            translated = call_llm_chat(
                messages=[
                    {"role": "system", "content": GEMMA_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=400,
                temperature=0.1,
                model=model_id,
            ).strip()

            # Clean output if wrapped in quotes or conversational artifacts
            if translated.startswith('"') and translated.endswith('"') and len(translated) > 2:
                translated = translated[1:-1].strip()

            return translated or clean
        except LLMClientError as exc:
            raise GemmaTranslationError(f"Gemma translation failed: {exc}") from exc

    def translate_chunks(
        self,
        chunks: list[Chunk],
        source_lang: str,
        session_id: Optional[str] = None,
        force_regenerate: bool = False,
    ) -> list[Chunk]:
        """Translate transcript chunks to English while strictly preserving timestamps, chunk IDs, and boundaries."""
        if not chunks:
            return []

        # If already English, bypass Gemma completely (Rule 6: Do not translate English)
        if source_lang == "en":
            for c in chunks:
                if not getattr(c, "original_text", None):
                    c.original_text = c.text
                if not getattr(c, "translated_text", None):
                    c.translated_text = c.text
                c.source_language = "en"
            return chunks

        # Check persistent disk cache
        if session_id and not force_regenerate:
            cached_chunks = self._load_cached_chunks(session_id, chunks)
            if cached_chunks:
                logger.info("Reusing cached Gemma translation for session %s", session_id)
                return cached_chunks

        # Check model availability before attempting translation
        model_id = self.get_active_model_id()
        logger.info("Translating %d chunks from %s to en using Gemma model %s", len(chunks), source_lang, model_id)

        translated_chunks: list[Chunk] = []
        for c in chunks:
            orig = getattr(c, "original_text", None) or c.text
            s_time = getattr(c, "start_time", None) if getattr(c, "start_time", None) is not None else c.start_timestamp
            e_time = getattr(c, "end_time", None) if getattr(c, "end_time", None) is not None else c.end_timestamp

            # Translate text through Gemma
            trans_text = self.translate_segment(orig, source_lang)

            updated = c.model_copy(
                update={
                    "original_text": orig,
                    "translated_text": trans_text,
                    "text": trans_text,  # Primary downstream text becomes English
                    "source_language": source_lang,
                    "start_time": s_time,
                    "end_time": e_time,
                    "start_timestamp": c.start_timestamp,
                    "end_timestamp": c.end_timestamp,
                    "start_label": c.start_label,
                    "end_label": c.end_label,
                    "chunk_id": c.chunk_id,
                    "word_count": len(trans_text.split()),
                }
            )
            translated_chunks.append(updated)

        # Save to persistent cache
        if session_id:
            self._save_cache(session_id, source_lang, translated_chunks)

        return translated_chunks

    def _cache_path(self, session_id: str) -> Path:
        settings.translations_dir.mkdir(parents=True, exist_ok=True)
        return settings.translations_dir / f"{session_id}.json"

    def _load_cached_chunks(self, session_id: str, original_chunks: list[Chunk]) -> Optional[list[Chunk]]:
        path = self._cache_path(session_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            chunk_data_map = {item["chunk_id"]: item for item in data.get("chunks", [])}
            
            reconstructed: list[Chunk] = []
            for c in original_chunks:
                if c.chunk_id in chunk_data_map:
                    cached_item = chunk_data_map[c.chunk_id]
                    updated = c.model_copy(
                        update={
                            "original_text": cached_item.get("original_text", c.text),
                            "translated_text": cached_item.get("translated_text", c.text),
                            "text": cached_item.get("translated_text", c.text),
                            "source_language": cached_item.get("source_language", data.get("source_language", "en")),
                            "start_time": cached_item.get("start_time", c.start_timestamp),
                            "end_time": cached_item.get("end_time", c.end_timestamp),
                        }
                    )
                    reconstructed.append(updated)
                else:
                    reconstructed.append(c)
            return reconstructed
        except Exception as exc:
            logger.warning("Could not read translation cache for %s: %s", session_id, exc)
            return None

    def _save_cache(self, session_id: str, source_lang: str, chunks: list[Chunk]) -> None:
        path = self._cache_path(session_id)
        try:
            payload = {
                "session_id": session_id,
                "source_language": source_lang,
                "target_language": "en",
                "model": self._model_id or "gemma",
                "chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "original_text": c.original_text,
                        "translated_text": c.translated_text,
                        "source_language": c.source_language,
                        "start_time": c.start_time,
                        "end_time": c.end_time,
                    }
                    for c in chunks
                ],
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not persist translation cache for %s: %s", session_id, exc)


_gemma_service_instance: Optional[GemmaTranslationService] = None


def get_gemma_translation_service() -> GemmaTranslationService:
    """Return singleton instance of GemmaTranslationService."""
    global _gemma_service_instance
    if _gemma_service_instance is None:
        _gemma_service_instance = GemmaTranslationService()
    return _gemma_service_instance
