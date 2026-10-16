from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.pipeline.chunker import chunk_sentences
from app.pipeline.cleaner import sentences_from_segments
from app.pipeline.youtube import TranscriptError, fetch_transcript
from app.schemas import Chunk, ProcessResponse, SessionSummary


class ProcessingError(Exception):
    pass


def format_duration(seconds: float | None) -> str:
    """Format total seconds into a readable string like '1h 04m' or '12m 30s'."""
    if seconds is None or seconds <= 0:
        return "N/A"
    sec = int(seconds)
    hours = sec // 3600
    minutes = (sec % 3600) // 60
    rem_sec = sec % 60
    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {rem_sec:02d}s"


def process_youtube_url(youtube_url: str) -> ProcessResponse:
    try:
        transcript = fetch_transcript(youtube_url)
    except TranscriptError as exc:
        raise ProcessingError(str(exc)) from exc

    sentences, cleaned_text = sentences_from_segments(transcript.segments)
    if not sentences:
        raise ProcessingError("Transcript cleaning produced no usable sentences.")

    chunks = chunk_sentences(sentences)
    if not chunks:
        raise ProcessingError("Chunking produced no segments.")

    # Phase 8.2: Automatic Language Detection & Local Gemma 4 E2B Translation Layer
    session_id = uuid.uuid4().hex[:12]
    from app.services.translation_service import detect_language
    detected_lang = detect_language(cleaned_text, hint=transcript.language)

    if detected_lang != "en":
        from app.services.gemma_translation_service import (
            GemmaTranslationError,
            GemmaTranslationService,
            GemmaUnavailableError,
        )
        try:
            gemma_svc = GemmaTranslationService()
            chunks = gemma_svc.translate_chunks(chunks, source_lang=detected_lang, session_id=session_id)
        except (GemmaUnavailableError, GemmaTranslationError) as exc:
            import logging
            logging.getLogger(__name__).warning("Gemma translation issue (%s); keeping original chunks", exc)


    from app.pipeline.topic_extractor import extract_topics_and_concepts

    topics, enriched_chunks = extract_topics_and_concepts(chunks)

    canonical_url = f"https://www.youtube.com/watch?v={transcript.video_id}"
    duration_sec = enriched_chunks[-1].end_timestamp if enriched_chunks else None
    title = getattr(transcript, "title", f"YouTube Lecture ({transcript.video_id})")

    response = ProcessResponse(
        session_id=session_id,
        video_id=transcript.video_id,
        youtube_url=canonical_url,
        title=title,
        source_type="youtube",
        detected_language=detected_lang,
        duration_seconds=duration_sec,
        formatted_duration=format_duration(duration_sec),
        language=transcript.language,
        is_generated_transcript=transcript.is_generated,
        raw_segment_count=len(transcript.segments),
        cleaned_text_preview=_preview(cleaned_text),
        chunk_count=len(enriched_chunks),
        chunks=enriched_chunks,
        topics=topics,
    )
    _save_session(response)
    try:
        from app.services.search_service import index_session
        index_session(response)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Automatic indexing failed for %s: %s", session_id, exc)

    try:
        from app.services.concept_intelligence_service import get_or_create_concept_intelligence
        get_or_create_concept_intelligence(session_id)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Automatic concept intelligence caching failed for %s: %s", session_id, exc)

    return response


def load_session(session_id: str) -> ProcessResponse | None:
    path = _session_path(session_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))

    modified = False

    # Backwards compatibility: Populate title and metadata for legacy sessions
    if not data.get("title") or data.get("title") == "Untitled Lecture":
        vid = data.get("video_id", "")
        if vid == "kqtD5dpn9C8":
            data["title"] = "Python for Beginners - Learn Coding with Python in 1 Hour"
        elif vid:
            from app.pipeline.youtube import fetch_youtube_title
            data["title"] = fetch_youtube_title(vid)
        else:
            data["title"] = f"Lecture ({session_id})"
        modified = True

    if "source_type" not in data:
        data["source_type"] = "uploaded" if data.get("video_id", "").startswith("local_") else "youtube"
        modified = True

    if "detected_language" not in data:
        data["detected_language"] = data.get("language") or "en"
        modified = True

    if not data.get("formatted_duration") and data.get("chunks"):
        max_t = max((c.get("end_timestamp", 0) for c in data["chunks"]), default=0)
        data["duration_seconds"] = max_t
        data["formatted_duration"] = format_duration(max_t)
        modified = True

    # Phase 8.2: Ensure legacy chunks have dual-text and timestamp fields
    if data.get("chunks"):
        chunk_modified = False
        for c in data["chunks"]:
            if "original_text" not in c or not c["original_text"]:
                c["original_text"] = c.get("text", "")
                chunk_modified = True
            if "source_language" not in c or not c["source_language"]:
                c["source_language"] = data.get("detected_language") or data.get("language") or "en"
                chunk_modified = True
            if "start_time" not in c or c.get("start_time") is None:
                c["start_time"] = c.get("start_timestamp", 0.0)
                chunk_modified = True
            if "end_time" not in c or c.get("end_time") is None:
                c["end_time"] = c.get("end_timestamp", 0.0)
                chunk_modified = True
        if chunk_modified:
            modified = True

    # If session was created before Phase 3, extract and attach topics/concepts seamlessly
    if not data.get("topics") and data.get("chunks"):
        from app.pipeline.topic_extractor import extract_topics_and_concepts

        raw_chunks = [Chunk.model_validate(c) for c in data["chunks"]]
        topics, enriched_chunks = extract_topics_and_concepts(raw_chunks)
        data["topics"] = [t.model_dump() for t in topics]
        data["chunks"] = [c.model_dump() for c in enriched_chunks]
        modified = True

    if modified:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return ProcessResponse.model_validate(data)


def list_sessions() -> list[SessionSummary]:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[SessionSummary] = []
    for path in sorted(settings.data_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        data = json.loads(path.read_text(encoding="utf-8"))
        title = data.get("title")
        if not title or title == "Untitled Lecture":
            if data.get("video_id") == "kqtD5dpn9C8":
                title = "Python for Beginners - Learn Coding with Python in 1 Hour"
            else:
                title = f"Lecture ({data.get('video_id', data['session_id'])})"

        source_type = data.get("source_type", "youtube")
        dur_fmt = data.get("formatted_duration")
        if not dur_fmt and data.get("chunks"):
            max_t = max((c.get("end_timestamp", 0) for c in data["chunks"]), default=0)
            dur_fmt = format_duration(max_t)

        summaries.append(
            SessionSummary(
                session_id=data["session_id"],
                video_id=data["video_id"],
                youtube_url=data["youtube_url"],
                title=title,
                source_type=source_type,
                formatted_duration=dur_fmt,
                chunk_count=data["chunk_count"],
            )
        )
    return summaries


def _save_session(response: ProcessResponse) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    payload = response.model_dump()
    payload["saved_at"] = datetime.now(timezone.utc).isoformat()
    _session_path(response.session_id).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _session_path(session_id: str) -> Path:
    return settings.data_dir / f"{session_id}.json"


def _preview(text: str, limit: int = 400) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."
