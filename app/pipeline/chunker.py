"""Semantic chunking that keeps complete sentences and timestamps."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.pipeline.cleaner import Sentence
from app.schemas import Chunk


def format_timestamp(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@dataclass
class ChunkDraft:
    sentences: list[Sentence]

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.sentences).strip()

    @property
    def start(self) -> float:
        return self.sentences[0].start

    @property
    def end(self) -> float:
        return self.sentences[-1].end

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def chunk_sentences(sentences: list[Sentence]) -> list[Chunk]:
    if not sentences:
        return []

    drafts: list[ChunkDraft] = []
    current: list[Sentence] = []

    def current_words() -> int:
        return len(" ".join(s.text for s in current).split()) if current else 0

    for sentence in sentences:
        if not current:
            current = [sentence]
            continue

        gap = sentence.start - current[-1].end
        words_if_added = current_words() + len(sentence.text.split())
        topic_break = gap >= settings.topic_gap_seconds and current_words() >= settings.chunk_min_words
        too_long = words_if_added > settings.chunk_max_words and current_words() >= settings.chunk_min_words
        target_reached = (
            current_words() >= settings.chunk_target_words
            and gap >= 1.5
        )

        if topic_break or too_long or target_reached:
            drafts.append(ChunkDraft(sentences=current))
            current = [sentence]
        else:
            current.append(sentence)

    if current:
        drafts.append(ChunkDraft(sentences=current))

    drafts = _merge_tiny_tails(drafts)
    return [_to_chunk(index, draft) for index, draft in enumerate(drafts, start=1)]


def _merge_tiny_tails(drafts: list[ChunkDraft]) -> list[ChunkDraft]:
    if len(drafts) < 2:
        return drafts
    merged: list[ChunkDraft] = []
    for draft in drafts:
        if (
            merged
            and draft.word_count < settings.chunk_min_words
            and merged[-1].word_count + draft.word_count <= settings.chunk_max_words
        ):
            merged[-1] = ChunkDraft(sentences=merged[-1].sentences + draft.sentences)
        else:
            merged.append(draft)
    return merged


def _to_chunk(index: int, draft: ChunkDraft) -> Chunk:
    return Chunk(
        chunk_id=f"chunk-{index:03d}",
        topic=None,
        text=draft.text,
        original_text=draft.text,
        start_timestamp=round(draft.start, 2),
        end_timestamp=round(draft.end, 2),
        start_time=round(draft.start, 2),
        end_time=round(draft.end, 2),
        start_label=format_timestamp(draft.start),
        end_label=format_timestamp(draft.end),
        word_count=draft.word_count,
    )

