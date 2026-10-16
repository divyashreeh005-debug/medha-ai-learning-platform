"""Educational content generator using Qwen3-8B-Instruct (Phase 4).

Generates grounded Summary, Simple Explanation, Analogy, and Key Takeaways
for specific topics using strictly bounded lecture context, with persistent caching.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from app.config import settings
from app.schemas import (
    LANGUAGE_DISPLAY_NAMES,
    Chunk,
    ProcessResponse,
    TopicGroup,
    TopicLearningContent,
)
from app.services.llm_client import LLMClientError, call_llm_chat
from app.services.process_video import load_session

logger = logging.getLogger(__name__)


class ContentGenerationError(Exception):
    """Raised when generating educational learning content fails."""


def build_topic_prompt(
    topic: TopicGroup,
    chunks: list[Chunk],
    language: str = "en",
) -> list[dict[str, str]]:
    """Construct grounded system and user messages containing only the topic's relevant chunks."""
    lang_name = LANGUAGE_DISPLAY_NAMES.get(language, "English")
    lang_directive = (
        f"Language Requirement: Generate ALL text values (summary, simple_explanation, analogy, key_takeaways) "
        f"in {lang_name} language. Technical syntax or code terms (e.g. variable names, 'for', 'while') "
        f"may remain in standard code syntax.\n\n"
    ) if language != "en" else ""

    system_message = (
        "Do not think or reason. Provide only the final JSON answer directly without chain-of-thought.\n\n"
        "You are Medhā (मेधा), an AI learning companion built to transform educational video lectures "
        "into clear, structured understanding for students.\n\n"
        f"{lang_directive}"
        "Grounding Rules:\n"
        "1. Use ONLY the provided lecture context below.\n"
        "2. Do not invent facts, features, or syntax that are not mentioned in or supported by the context.\n"
        "3. If the context does not provide enough detail, state clearly what the instructor covered.\n"
        "4. Do NOT generate quizzes, MCQs, or follow-up questions.\n"
        "5. Output MUST be a single, valid JSON object with EXACTLY these four keys:\n"
        "   - \"summary\": A well-structured educational summary (approx 2 to 4 informative paragraphs) detailing: "
        "what the concept is, why it matters, how it works in the lecture, and practical examples demonstrated by the instructor.\n"
        "   - \"simple_explanation\": A genuinely beginner-friendly explanation breaking down the core concept in plain, intuitive everyday words.\n"
        "   - \"analogy\": One understandable, intuitive real-world analogy to help the student remember.\n"
        "   - \"key_takeaways\": A JSON array of 4 to 7 important, distinct revision points as strings.\n\n"
        "Output ONLY the JSON object. Do not include markdown code block markers or conversational preamble."
    )

    # Filter chunks belonging to this topic with strict context window budgeting
    chunk_map = {c.chunk_id: c for c in chunks}
    topic_chunks = [chunk_map[cid] for cid in topic.chunk_ids if cid in chunk_map]

    # Enforce safe context budget (max 1000 words / ~1300 tokens) to prevent LLM context overflow
    MAX_CONTEXT_WORDS = 1000
    selected_chunks: list[Chunk] = []
    accumulated_words = 0

    def chunk_word_count(c: Chunk) -> int:
        t = c.get_display_text(prefer_original=False) if hasattr(c, "get_display_text") else (c.translated_text or c.text)
        return len(t.split())

    if topic_chunks:
        # Always prioritize first 3 chunks (introduction of topic)
        for c in topic_chunks[:3]:
            selected_chunks.append(c)
            accumulated_words += chunk_word_count(c)

        # If more chunks exist, select evenly across remainder without exceeding budget
        remainder = topic_chunks[3:]
        if remainder and accumulated_words < MAX_CONTEXT_WORDS:
            step = max(1, len(remainder) // 6)
            for c in remainder[::step]:
                c_words = chunk_word_count(c)
                if accumulated_words + c_words > MAX_CONTEXT_WORDS:
                    break
                selected_chunks.append(c)
                accumulated_words += c_words

    # Sort selected chunks chronologically
    selected_chunks.sort(key=lambda c: c.start_timestamp)

    context_blocks: list[str] = []
    for c in selected_chunks:
        # Use translated/English text for learning generation
        text_content = c.get_display_text(prefer_original=False) if hasattr(c, "get_display_text") else (c.translated_text or c.text)
        context_blocks.append(f"[{c.chunk_id} | {c.start_label} – {c.end_label}]:\n{text_content.strip()}")

    lecture_context = "\n\n".join(context_blocks)
    # Hard ceiling on characters to ensure 100% safety
    if len(lecture_context) > 5000:
        lecture_context = lecture_context[:5000] + "\n[Context capped for length]"

    concepts_str = ", ".join(topic.key_concepts) if topic.key_concepts else "N/A"

    user_message = (
        f"Topic: {topic.name}\n"
        f"Timestamp Range: {topic.timestamp_range}\n"
        f"Key Concepts: {concepts_str}\n\n"
        f"=== LECTURE TRANSCRIPT CONTEXT ===\n"
        f"{lecture_context}\n"
        f"=== END OF CONTEXT ===\n\n"
        f"Generate the educational learning content JSON for '{topic.name}' based strictly on the context above."
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def parse_structured_json(raw_text: str) -> dict[str, Any]:
    """Parse and clean JSON output from LLM, stripping markdown codeblocks if present."""
    text = raw_text.strip()

    # Strip ```json ... ``` code fence
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    # Find the outermost JSON object braces
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1:
        text = text[first_brace : last_brace + 1]

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ContentGenerationError(
            f"Failed to parse LLM output into valid JSON: {exc}\nRaw content:\n{raw_text}"
        ) from exc

    required_keys = ["summary", "simple_explanation", "analogy", "key_takeaways"]
    missing = [k for k in required_keys if k not in parsed]
    if missing:
        raise ContentGenerationError(f"Generated JSON missing required keys: {missing}")

    if not isinstance(parsed["key_takeaways"], list):
        if isinstance(parsed["key_takeaways"], str):
            parsed["key_takeaways"] = [parsed["key_takeaways"]]
        else:
            parsed["key_takeaways"] = []

    return parsed


def get_cache_path(session_id: str, topic_id: str, language: str = "en") -> Path:
    """Path to cached topic content JSON file, supporting language variants."""
    suffix = f"_{language}" if language and language != "en" else ""
    return settings.content_cache_dir / session_id / f"{topic_id}{suffix}.json"


def load_cached_content(
    session_id: str, topic_id: str, language: str = "en"
) -> Optional[TopicLearningContent]:
    """Read cached topic learning content if available."""
    path = get_cache_path(session_id, topic_id, language=language)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return TopicLearningContent.model_validate(data)
    except Exception as exc:
        logger.warning("Could not read cache for %s/%s (%s): %s", session_id, topic_id, language, exc)
        return None


def save_cached_content(
    session_id: str, topic_id: str, content: TopicLearningContent, language: str = "en"
) -> None:
    """Persist generated topic learning content to disk cache."""
    path = get_cache_path(session_id, topic_id, language=language)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(content.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def generate_topic_content(
    session_id: str,
    topic_id: str,
    force_regenerate: bool = False,
    language: str = "en",
) -> TopicLearningContent:
    """Generate or retrieve cached educational content for a topic using Qwen3-8B-Instruct."""
    # 1. Return cached content if already generated
    if not force_regenerate:
        cached = load_cached_content(session_id, topic_id, language=language)
        if cached:
            logger.info("Returning cached learning content for %s/%s (%s)", session_id, topic_id, language)
            return cached

    # 2. Load session and topic details
    session = load_session(session_id)
    if session is None:
        raise ContentGenerationError(f"Session '{session_id}' not found.")

    target_topic = next((t for t in session.topics if t.topic_id == topic_id), None)
    if target_topic is None:
        raise ContentGenerationError(
            f"Topic '{topic_id}' not found in session '{session_id}'."
        )

    # 3. Construct bounded prompt
    messages = build_topic_prompt(target_topic, session.chunks, language=language)

    # 4. Invoke local Qwen3-8B model
    try:
        raw_output = call_llm_chat(messages, max_tokens=1200, model="qwen")
    except LLMClientError as exc:
        raise ContentGenerationError(str(exc)) from exc

    # 5. Parse structured output
    parsed = parse_structured_json(raw_output)

    content = TopicLearningContent(
        topic_id=target_topic.topic_id,
        topic_name=target_topic.name,
        timestamp_range=target_topic.timestamp_range,
        summary=str(parsed["summary"]).strip(),
        simple_explanation=str(parsed["simple_explanation"]).strip(),
        analogy=str(parsed["analogy"]).strip(),
        key_takeaways=[str(p).strip() for p in parsed["key_takeaways"] if str(p).strip()],
        source_chunk_ids=list(target_topic.chunk_ids),
    )

    # 6. Save to cache
    save_cached_content(session_id, topic_id, content, language=language)

    # 7. Attach to session and update session file (for primary English content)
    if language == "en":
        _attach_content_to_session(session, topic_id, content)

    return content


def _attach_content_to_session(
    session: ProcessResponse, topic_id: str, content: TopicLearningContent
) -> None:
    """Update session data so learning content is embedded directly in session.topics."""
    for t in session.topics:
        if t.topic_id == topic_id:
            t.learning_content = content
            break

    session_path = settings.data_dir / f"{session.session_id}.json"
    if session_path.exists():
        try:
            session_data = session.model_dump()
            session_path.write_text(
                json.dumps(session_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to update session with learning content: %s", exc)
