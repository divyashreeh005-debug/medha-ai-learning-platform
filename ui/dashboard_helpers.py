"""Dashboard helper utilities for Medhā (Phase 5).

Provides pure, testable functions for topic navigation, YouTube timestamp
deep-linking, transcript chunk filtering, concept badge formatting, and cache loading.
"""

import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings


def get_chronological_topics(topics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return topics strictly sorted by chronological start_time.
    
    Guarantees that topics follow the lecture flow and are never alphabetically sorted.
    """
    return sorted(topics, key=lambda t: float(t.get("start_time", 0.0)))


def get_next_topic_index(current_idx: int, total_topics: int) -> int | None:
    """Return the next topic index, or None if already at the last topic."""
    if total_topics <= 0 or current_idx >= total_topics - 1:
        return None
    return current_idx + 1


def get_previous_topic_index(current_idx: int) -> int | None:
    """Return the previous topic index, or None if already at the first topic."""
    if current_idx <= 0:
        return None
    return current_idx - 1


def get_youtube_jump_url(video_id: str, start_time: float | int) -> str:
    """Generate a direct YouTube link starting at the topic's start timestamp."""
    seconds = max(0, int(start_time))
    return f"https://www.youtube.com/watch?v={video_id}&t={seconds}s"


def filter_chunks_for_topic(
    topic: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter and return only the transcript chunks associated with the selected topic.
    
    Preserves exact chunk order, chunk IDs, and timestamp metadata.
    """
    topic_chunk_ids = set(topic.get("chunk_ids", []))
    return [c for c in chunks if c.get("chunk_id") in topic_chunk_ids]


def format_concept_badges(concepts: list[str] | None) -> list[str]:
    """Clean and deduplicate technical concepts for badge display."""
    if not concepts:
        return []
    seen = set()
    cleaned = []
    for c in concepts:
        s = c.strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            cleaned.append(s)
    return cleaned


def calculate_topic_progress(current_idx: int, total_topics: int) -> tuple[float, str]:
    """Calculate progress percentage and human-readable indicator string (e.g. 'Topic 3 of 17')."""
    if total_topics <= 0:
        return 0.0, "Topic 0 of 0"
    clamped_idx = max(0, min(current_idx, total_topics - 1))
    ratio = (clamped_idx + 1) / total_topics
    return ratio, f"Topic {clamped_idx + 1} of {total_topics}"


def format_lecture_title(title: str | None, video_id: str | None = None, session_id: str | None = None) -> str:
    """Return a human-readable title, prioritizing real lecture title over raw IDs."""
    if title and title.strip() and not title.lower().startswith("youtube lecture"):
        return title.strip()
    if title and title.strip():
        return title.strip()
    if video_id and not video_id.startswith("local_"):
        return f"Lecture ({video_id})"
    if session_id:
        return f"Lecture Session {session_id[:8]}"
    return "Untitled Lecture"


def load_cached_learning_content(
    session_id: str,
    topic_id: str,
    cache_dir: Path | str | None = None,
    language: str = "en",
) -> dict[str, Any] | None:
    """Load cached learning content for a topic from disk without invoking the LLM.
    
    Supports localized cache files (e.g., topic-001_hi.json) when language != 'en'.
    """
    base_dir = Path(cache_dir) if cache_dir else settings.content_cache_dir
    filename = f"{topic_id}_{language}.json" if language and language != "en" else f"{topic_id}.json"
    cache_file = base_dir / session_id / filename
    if not cache_file.exists():
        # Fallback to English if localized doesn't exist
        if language != "en":
            fallback_file = base_dir / session_id / f"{topic_id}.json"
            if fallback_file.exists():
                try:
                    with open(fallback_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict) and "summary" in data:
                            return data
                except Exception:
                    pass
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "summary" in data and "simple_explanation" in data:
                return data
    except Exception:
        return None
    return None


def get_next_card_index(current_idx: int, total_cards: int) -> int:
    """Return the next card index, looping or clamping safely."""
    if total_cards <= 0:
        return 0
    return min(current_idx + 1, total_cards - 1)


def get_previous_card_index(current_idx: int, total_cards: int) -> int:
    """Return the previous card index, clamping safely."""
    if total_cards <= 0:
        return 0
    return max(current_idx - 1, 0)
