"""YouTube transcript acquisition (Phase 1)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
    YouTubeTranscriptApiException,
)

YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
WATCH_RE = re.compile(r"(?:v=|/v/|/embed/|/shorts/|/live/)([A-Za-z0-9_-]{11})")
SHORT_RE = re.compile(r"youtu\.be/([A-Za-z0-9_-]{11})")


class TranscriptError(Exception):
    """Raised when a real transcript cannot be obtained."""


@dataclass
class RawSegment:
    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


def fetch_youtube_title(video_id: str) -> str:
    """Fetch the real YouTube video title via YouTube's public oEmbed API.
    
    Requires no API keys. Gracefully falls back to a clean label if offline or unavailable.
    """
    import json
    import urllib.request
    try:
        url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Medha-Educational-App)"})
        with urllib.request.urlopen(req, timeout=4) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                title = data.get("title")
                if title and isinstance(title, str) and title.strip():
                    return title.strip()
    except Exception:
        pass
    return f"YouTube Lecture ({video_id})"


@dataclass
class TranscriptResult:
    video_id: str
    language: str | None
    is_generated: bool
    segments: list[RawSegment]
    title: str = "Untitled Lecture"


def extract_video_id(url_or_id: str) -> str:
    value = url_or_id.strip()
    if YOUTUBE_ID_RE.match(value):
        return value
    match = WATCH_RE.search(value) or SHORT_RE.search(value)
    if not match:
        raise TranscriptError(
            "Could not parse a YouTube video ID. Use a watch URL, youtu.be link, or 11-character ID."
        )
    return match.group(1)


def fetch_transcript(youtube_url: str) -> TranscriptResult:
    video_id = extract_video_id(youtube_url)
    api = YouTubeTranscriptApi()

    try:
        listing = api.list(video_id)
    except TranscriptsDisabled as exc:
        raise TranscriptError("Captions are disabled for this video.") from exc
    except VideoUnavailable as exc:
        raise TranscriptError("This YouTube video is unavailable.") from exc
    except YouTubeTranscriptApiException as exc:
        raise TranscriptError(f"YouTube transcript request failed: {exc}") from exc

    fetched, language, is_generated = _choose_and_fetch(listing)
    segments = [_to_segment(item) for item in fetched]
    segments = [s for s in segments if s.text.strip()]
    if not segments:
        raise TranscriptError("The transcript was empty after download.")

    title = fetch_youtube_title(video_id)

    return TranscriptResult(
        video_id=video_id,
        language=language,
        is_generated=is_generated,
        segments=segments,
        title=title,
    )


def _choose_and_fetch(listing: Any) -> tuple[Any, str | None, bool]:
    preferred = ["en", "en-US", "en-GB"]
    try:
        transcript = listing.find_manually_created_transcript(preferred)
        fetched = transcript.fetch()
        return fetched, transcript.language_code, False
    except (NoTranscriptFound, YouTubeTranscriptApiException):
        pass

    try:
        transcript = listing.find_generated_transcript(preferred)
        fetched = transcript.fetch()
        return fetched, transcript.language_code, True
    except (NoTranscriptFound, YouTubeTranscriptApiException):
        pass

    try:
        transcript = next(iter(listing))
    except StopIteration as exc:
        raise TranscriptError("No captions are available for this video.") from exc

    fetched = transcript.fetch()
    is_generated = bool(getattr(transcript, "is_generated", False))
    return fetched, getattr(transcript, "language_code", None), is_generated


def _to_segment(item: Any) -> RawSegment:
    if hasattr(item, "text") and hasattr(item, "start"):
        duration = float(getattr(item, "duration", 0.0) or 0.0)
        return RawSegment(text=str(item.text), start=float(item.start), duration=duration)
    return RawSegment(
        text=str(item["text"]),
        start=float(item["start"]),
        duration=float(item.get("duration", 0.0) or 0.0),
    )
