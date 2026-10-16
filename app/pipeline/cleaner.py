"""Transcript cleaning and sentence segmentation with timestamp preservation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.pipeline.youtube import RawSegment

FILLER_RE = re.compile(
    r"\b(?:um+|uh+|erm+|huh|hmm+|you know|i mean|kind of|sort of)\b",
    flags=re.IGNORECASE,
)
BRACKET_NOISE_RE = re.compile(r"\[(?:music|applause|laughter|silence|inaudible).*?\]", re.IGNORECASE)
MULTISPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass
class Sentence:
    text: str
    start: float
    end: float


def clean_caption_text(text: str) -> str:
    value = text.replace("\n", " ")
    value = value.replace("&#39;", "'").replace("&amp;", "&").replace("&quot;", '"')
    value = value.replace("&nbsp;", " ")
    value = BRACKET_NOISE_RE.sub(" ", value)
    value = FILLER_RE.sub(" ", value)
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    value = MULTISPACE_RE.sub(" ", value).strip()
    return value


def sentences_from_segments(segments: list[RawSegment]) -> tuple[list[Sentence], str]:
    cleaned: list[RawSegment] = []
    for segment in segments:
        text = clean_caption_text(segment.text)
        if text:
            cleaned.append(RawSegment(text=text, start=segment.start, duration=segment.duration))

    if not cleaned:
        return [], ""

    spans: list[tuple[int, int, float, float]] = []
    parts: list[str] = []
    cursor = 0
    for index, segment in enumerate(cleaned):
        if index:
            cursor += 1
        start_char = cursor
        parts.append(segment.text)
        cursor += len(segment.text)
        spans.append((start_char, cursor, segment.start, segment.end))

    full_text = " ".join(parts)
    pieces = _split_sentence_texts(full_text)
    sentences: list[Sentence] = []
    search_from = 0
    for piece in pieces:
        pos = full_text.find(piece, search_from)
        if pos < 0:
            pos = search_from
        end_pos = pos + len(piece)
        start_t, end_t = _times_for_span(spans, pos, end_pos)
        sentences.append(Sentence(text=piece.strip(), start=start_t, end=end_t))
        search_from = end_pos

    return sentences, full_text


def _split_sentence_texts(text: str) -> list[str]:
    parts = SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _times_for_span(
    spans: list[tuple[int, int, float, float]], start_char: int, end_char: int
) -> tuple[float, float]:
    start_t: float | None = None
    end_t: float | None = None
    for s_char, e_char, s_time, e_time in spans:
        if e_char <= start_char:
            continue
        if s_char >= end_char:
            break
        if start_t is None:
            start_t = s_time
        end_t = e_time
    if start_t is None or end_t is None:
        return spans[0][2], spans[-1][3]
    return start_t, end_t
