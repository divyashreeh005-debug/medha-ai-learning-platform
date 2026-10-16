"""Topic and Concept Extraction service (Phase 3).

Uses discourse markers and TF-IDF keyphrase analysis to extract meaningful
topics, concept tags, and temporal topic groups directly from lecture chunks.
Does not depend on external LLMs.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.schemas import Chunk, TopicGroup

logger = logging.getLogger(__name__)

DISCOURSE_PATTERNS = [
    re.compile(
        r"(?:in this (?:tutorial|video|lecture)|today|now)\s*,?\s*(?:we(?:'re| are)|I(?:'m| am)|we will|we'll|let's)\s+(?:going to\s+)?(?:talk about|cover|learn|look at|explore|discuss|show you)\s+(?:a bunch of (?:cool )?things you can do with\s+)?([a-zA-Z0-9_ -]{3,50})",
        re.IGNORECASE,
    ),
    re.compile(r"(?:this is what we call|we call this)\s+(?:a |an )?([a-zA-Z0-9_ -]{3,30})", re.IGNORECASE),
    re.compile(r"(?:now|all right,? now|next,?) let's (?:talk about|look at|cover|discuss|explore|start with)\s+([a-zA-Z0-9_ -]{3,50})", re.IGNORECASE),
    re.compile(r"(?:another set of operators called|set of operators called|operators called)\s+([a-zA-Z0-9_ -]{3,40})", re.IGNORECASE),
    re.compile(r"(?:we're going to talk about|talk about)\s+([a-zA-Z0-9_ -]{3,40})", re.IGNORECASE),
    re.compile(r"write a\s+([a-zA-Z0-9_ -]{3,30}\s+program)", re.IGNORECASE),
]

DISALLOWED_TOPICS = {
    "it", "them", "this", "something", "string", "few examples", "few more examples",
    "another example", "next example", "simple example", "here", "what", "next",
    "shortcut", "shortcut associated", "shortcut associated with this command",
}

CUSTOM_CONCEPT_STOPS = {
    "tutorial", "video", "python", "going", "want", "let", "code", "program",
    "just", "like", "know", "really", "simply", "use", "using", "look", "see",
    "need", "say", "tell", "talk", "make", "sure", "come", "think", "good",
    "right", "okay", "alright", "actually", "mean", "lot", "things", "way",
    "run", "shows", "shown", "click", "double", "page", "window", "folder",
    "line", "lines", "type", "press", "enter", "print", "screen", "terminal",
}


def normalize_text(text: str) -> str:
    """Normalize curly quotes and whitespace."""
    return text.replace("’", "'").replace("“", '"').replace("”", '"')


def clean_topic_phrase(phrase: str) -> str:
    """Normalize extracted raw topic phrases into clean titles."""
    cleaned = phrase.strip()
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\b(?:that we have in python|that we have in math|that we have|in python|in programming|built into python|in this tutorial|over here|on this page)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+\b(?:in|with|as|so)\b.*$", "", cleaned, flags=re.IGNORECASE).strip()
    t = cleaned.title()

    # Standardize common programming construct names
    lower = t.lower()
    if lower in ("for loop", "a for loop"):
        return "For Loops"
    if lower in ("while loop", "a while loop"):
        return "While Loops"
    if lower in ("if statement", "an if statement", "if statements"):
        return "If Statements"
    if lower in ("list", "lists in python"):
        return "Lists"
    if lower in ("tuple", "tuples in python"):
        return "Tuples"
    if lower in ("how to receive input from the user", "receiving input"):
        return "Receiving User Input"

    return t


def extract_concepts_per_chunk(chunks: list[Chunk]) -> list[list[str]]:
    """Extract distinctive technical terms and n-grams for each chunk using TF-IDF."""
    if not chunks:
        return []

    corpus = [c.text for c in chunks]

    # Handle single or tiny chunk lists safely
    min_df = 1
    max_df = 1.0 if len(chunks) <= 2 else 0.85

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_df=max_df,
            min_df=min_df,
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
        feature_names = np.array(vectorizer.get_feature_names_out())

        chunk_concepts: list[list[str]] = []
        for i in range(len(chunks)):
            row = tfidf_matrix[i].toarray()[0]
            top_indices = np.argsort(row)[::-1]
            concepts: list[str] = []
            for idx in top_indices:
                if row[idx] <= 0:
                    break
                feat = str(feature_names[idx])
                parts = feat.split()
                if any(p in CUSTOM_CONCEPT_STOPS for p in parts):
                    continue
                if len(feat) <= 2:
                    continue
                concepts.append(feat)
                if len(concepts) >= 4:
                    break
            chunk_concepts.append(concepts)
        return chunk_concepts
    except Exception as exc:
        logger.warning("TF-IDF concept extraction fallback due to: %s", exc)
        return [[] for _ in chunks]


def extract_topics_and_concepts(chunks: list[Chunk]) -> tuple[list[TopicGroup], list[Chunk]]:
    """Assign topics and concepts to each chunk, then aggregate contiguous topic groups.
    
    If video has no discourse markers or chapters, automatically detects semantic topic shifts
    based on TF-IDF keyphrase clusters. If timestamps are genuinely missing, assigns clean Section labels.
    """
    if not chunks:
        return [], []

    # 1. Extract concepts per chunk
    concepts_list = extract_concepts_per_chunk(chunks)

    # 2. Extract discourse or semantic topic per chunk
    current_topic = "Introduction"
    assigned_topics: list[str] = []
    consecutive_same_topic = 0

    for i, chunk in enumerate(chunks):
        text = normalize_text(chunk.text)
        matched_topic: str | None = None

        # A. Check discourse marker patterns
        for pattern in DISCOURSE_PATTERNS:
            match = pattern.search(text)
            if match:
                candidate = clean_topic_phrase(match.group(1))
                if (
                    len(candidate) >= 3
                    and candidate.lower() not in DISALLOWED_TOPICS
                ):
                    matched_topic = candidate
                    break

        # B. Fallback: Semantic shift detection if no discourse markers for 6+ chunks
        chunk_concepts = concepts_list[i] if i < len(concepts_list) else []
        if not matched_topic and consecutive_same_topic >= 6 and chunk_concepts:
            # Pick the most prominent technical concept in this chunk
            candidate_concept = chunk_concepts[0].title()
            if len(candidate_concept) >= 3 and candidate_concept.lower() not in DISALLOWED_TOPICS:
                matched_topic = candidate_concept

        if matched_topic and matched_topic != current_topic:
            current_topic = matched_topic
            consecutive_same_topic = 1
        else:
            if i == 0 and not matched_topic:
                # If first chunk has strong concepts, prefer naming after top concept over generic Introduction
                if chunk_concepts and len(chunk_concepts[0]) >= 4:
                    current_topic = chunk_concepts[0].title()
                else:
                    current_topic = "Introduction"
            consecutive_same_topic += 1

        assigned_topics.append(current_topic)

    # Check if transcript genuinely contains no timestamps (all start and end are 0.0)
    has_real_timestamps = any(c.start_timestamp > 0.0 or c.end_timestamp > 0.0 for c in chunks)

    # 3. Create enriched chunks
    enriched_chunks: list[Chunk] = []
    for idx, (chunk, topic_name, concepts) in enumerate(zip(chunks, assigned_topics, concepts_list)):
        s_lbl = chunk.start_label if has_real_timestamps else f"Sec {idx + 1}"
        e_lbl = chunk.end_label if has_real_timestamps else f"Sec {idx + 1}"
        enriched_chunks.append(
            Chunk(
                chunk_id=chunk.chunk_id,
                topic=topic_name,
                concepts=concepts,
                text=chunk.text,
                start_timestamp=chunk.start_timestamp,
                end_timestamp=chunk.end_timestamp,
                start_label=s_lbl,
                end_label=e_lbl,
                word_count=chunk.word_count,
                original_text=chunk.original_text or chunk.text,
                translated_text=chunk.translated_text or chunk.text,
                source_language=chunk.source_language or "en",
                start_time=chunk.start_timestamp,
                end_time=chunk.end_timestamp,
            )
        )

    # 4. Group contiguous chunks into TopicGroup structures
    topic_groups: list[TopicGroup] = []
    for chunk in enriched_chunks:
        assert chunk.topic is not None
        if not topic_groups or topic_groups[-1].name != chunk.topic:
            group_id = f"topic-{len(topic_groups) + 1:03d}"
            sec_num = len(topic_groups) + 1
            t_range = f"{chunk.start_label} – {chunk.end_label}" if has_real_timestamps else f"Section {sec_num}"
            topic_groups.append(
                TopicGroup(
                    topic_id=group_id,
                    name=chunk.topic,
                    start_time=chunk.start_timestamp,
                    end_time=chunk.end_timestamp,
                    start_label=chunk.start_label if has_real_timestamps else f"Section {sec_num}",
                    end_label=chunk.end_label if has_real_timestamps else f"Section {sec_num}",
                    timestamp_range=t_range,
                    key_concepts=list(chunk.concepts),
                    chunk_ids=[chunk.chunk_id],
                )
            )
        else:
            current_group = topic_groups[-1]
            current_group.end_time = chunk.end_timestamp
            if has_real_timestamps:
                current_group.end_label = chunk.end_label
                current_group.timestamp_range = f"{current_group.start_label} – {chunk.end_label}"
            current_group.chunk_ids.append(chunk.chunk_id)
            for c in chunk.concepts:
                if c not in current_group.key_concepts and len(current_group.key_concepts) < 8:
                    current_group.key_concepts.append(c)

    return topic_groups, enriched_chunks
