"""Memory Cards / Revision Cards Service (Phase 8.1).

Generates, caches, and manages interactive revision flip cards derived from
lecture concepts and topic learning content with grounded source timestamps.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from app.config import settings
from app.schemas import (
    LANGUAGE_DISPLAY_NAMES,
    MemoryCard,
    MemoryCardCollection,
    ProcessResponse,
)
from app.services.concept_graph_service import extract_concept_nodes
from app.services.process_video import load_session

logger = logging.getLogger(__name__)


class MemoryCardError(Exception):
    """Raised when memory cards generation or loading fails."""


def get_memory_cards_path(session_id: str, language: str = "en") -> Path:
    """Return filesystem path for cached memory cards collection."""
    settings.memory_cards_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{language}" if language and language != "en" else ""
    return settings.memory_cards_dir / f"{session_id}{suffix}.json"


def load_memory_cards(session_id: str, language: str = "en") -> Optional[MemoryCardCollection]:
    """Load cached memory cards from disk if available."""
    path = get_memory_cards_path(session_id, language=language)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return MemoryCardCollection.model_validate(data)
    except Exception as exc:
        logger.warning("Could not read memory cards cache for %s (%s): %s", session_id, language, exc)
        return None


def save_memory_cards(collection: MemoryCardCollection) -> None:
    """Persist memory cards collection to disk."""
    path = get_memory_cards_path(collection.session_id, language=collection.language)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(collection.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved %d memory cards for session %s (%s)", collection.total_cards, collection.session_id, collection.language)


def generate_memory_cards(
    session_id: str,
    language: str = "en",
    force_regenerate: bool = False,
) -> MemoryCardCollection:
    """Generate or retrieve high-value grounded revision cards for a session (Phase 8.2).
    
    Excludes generic structural sections ('Introduction', 'Overview', 'Conclusion').
    Curates 7–10 important concepts with grounded answers and programming code examples.
    """
    from app.services.concept_intelligence_service import (
        get_or_create_concept_intelligence,
        is_generic_structural_title,
    )

    if not force_regenerate:
        cached = load_memory_cards(session_id, language=language)
        if (
            cached
            and cached.cards
            and len(cached.cards) <= 10
            and not any(is_generic_structural_title(c.concept_name) for c in cached.cards)
        ):
            return cached

    session = load_session(session_id)
    if session is None:
        raise MemoryCardError(f"Session '{session_id}' not found.")


    intel_collection = get_or_create_concept_intelligence(session_id)
    chunk_map = {c.chunk_id: c for c in session.chunks}

    # 1. Filter out structural sections and weak concepts
    eligible_concepts = []
    for c in intel_collection.concepts:
        if c.is_structural_section or is_generic_structural_title(c.concept):
            continue
        if c.importance_score < 0.35:
            continue

        # Check for sufficient transcript evidence (at least 20 words of supporting text)
        supporting_text = " ".join(
            chunk_map[cid].text for cid in c.supporting_chunks if cid in chunk_map
        )
        if len(supporting_text.split()) < 20:
            continue

        eligible_concepts.append(c)

    # 2. Sort by importance score descending and curate top 7–10 high-value concepts
    eligible_concepts.sort(key=lambda item: item.importance_score, reverse=True)
    curated = eligible_concepts[:10]

    cards: list[MemoryCard] = []
    for idx, c in enumerate(curated, start=1):
        card_id = f"card-{idx:03d}"
        concept_lower = c.concept.lower()

        # Build focused, natural revision question
        if "variable" in concept_lower:
            front = "What is a variable used for in Python?"
        elif "loop" in concept_lower:
            front = f"What is a {c.concept} and when do we use it?"
        elif "if statement" in concept_lower or "condition" in concept_lower:
            front = "How do conditional statements control program execution?"
        elif "list" in concept_lower:
            front = f"What is a {c.concept} in Python and how do you store items in it?"
        elif "function" in concept_lower:
            front = "Why do we define functions in programming?"
        elif "input" in concept_lower:
            front = "How do you receive and process user input in Python?"
        elif "operator" in concept_lower:
            front = f"How do {c.concept} evaluate expressions in code?"
        else:
            front = f"What is the core purpose of '{c.concept}' in this lecture?"

        # Back answer: strictly grounded explanation
        back = c.main_idea
        if not back:
            back = f"{c.concept} is a fundamental topic covered in the lecture between {c.timestamp}."

        card = MemoryCard(
            card_id=card_id,
            concept_id=f"concept-{idx:03d}",
            concept_name=c.concept,
            topic_name=c.concept,
            front_prompt=front,
            back_answer=back,
            timestamp_range=c.timestamp,
            start_time=c.start_time,
            source_chunk_ids=c.supporting_chunks,
            example=c.example,
            importance_score=c.importance_score,
        )
        cards.append(card)

    collection = MemoryCardCollection(
        session_id=session_id,
        language=language,
        total_cards=len(cards),
        cards=cards,
    )

    # If non-English language requested, localize card prompts and answers
    if language != "en":
        collection = _localize_memory_cards(collection, language)

    save_memory_cards(collection)
    return collection


def _localize_memory_cards(collection: MemoryCardCollection, language: str) -> MemoryCardCollection:
    """Translate front prompts and back answers into the target learning language."""
    from app.services.llm_client import LLMClientError, call_llm_chat

    lang_name = LANGUAGE_DISPLAY_NAMES.get(language, language)
    system_prompt = (
        "Do not think or reason. Provide only the final JSON array directly without chain-of-thought.\n\n"
        f"You are a professional educational translator. Translate these revision cards into {lang_name}.\n"
        "Keep programming keywords, function names, and syntax in standard Latin code format where appropriate.\n"
        "Output MUST be a JSON array of objects with keys: 'card_id', 'front_prompt', 'back_answer'."
    )

    items_to_translate = [
        {"card_id": c.card_id, "front_prompt": c.front_prompt, "back_answer": c.back_answer}
        for c in collection.cards
    ]

    try:
        raw = call_llm_chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(items_to_translate, ensure_ascii=False)},
            ],
            max_tokens=2000,
        )
        import re
        text = raw.strip()
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if fence_match:
            text = fence_match.group(1).strip()
        first_bracket = text.find("[")
        last_bracket = text.rfind("]")
        if first_bracket != -1 and last_bracket != -1:
            translated_list = json.loads(text[first_bracket : last_bracket + 1])
            trans_map = {item["card_id"]: item for item in translated_list if "card_id" in item}
            for c in collection.cards:
                if c.card_id in trans_map:
                    c.front_prompt = trans_map[c.card_id].get("front_prompt", c.front_prompt)
                    c.back_answer = trans_map[c.card_id].get("back_answer", c.back_answer)
    except Exception as exc:
        logger.warning("Memory card translation to %s failed, keeping base language: %s", language, exc)

    return collection
