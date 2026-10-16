"""RAG prompt builder for lecture-grounded AI Tutor (Phase 6).

Constructs strictly bounded, anti-hallucination prompts for Qwen3-8B-Instruct
containing retrieved lecture chunks, associated topic/concept context,
optional recent conversation history, and student questions.
"""

from __future__ import annotations

from typing import Any

from app.schemas import LANGUAGE_DISPLAY_NAMES, ChatMessage, SearchResultChunk

FALLBACK_UNAVAILABLE_MSG = "I couldn't find enough information about this in the lecture."


def build_tutor_prompt(
    question: str,
    chunks: list[SearchResultChunk],
    chat_history: list[ChatMessage] | None = None,
    graph_context: str | None = None,
    language: str = "en",
) -> list[dict[str, str]]:
    """Build grounded system and conversation messages for Qwen3-8B.
    
    Instructs the model to rely solely on the retrieved lecture transcript chunks
    and concept graph relationships, and never answer from external pre-trained knowledge.
    """
    lang_name = LANGUAGE_DISPLAY_NAMES.get(language, "English")
    lang_rule = f"\n6. Respond in {lang_name} while keeping code syntax, keywords, and variable names in standard format." if language != "en" else ""

    system_instruction = (
        "Do not think or reason. Provide only the direct educational answer without chain-of-thought.\n\n"
        "You are Medhā (मेधा), an AI lecture tutor helping a student understand this video lecture.\n\n"
        "Grounding Rules:\n"
        "1. Answer using ONLY the provided lecture transcript context below.\n"
        "2. Do not invent facts, features, or syntax that are not supported by the context.\n"
        "3. If the lecture context does not contain enough information to answer the question, "
        f"clearly state: \"{FALLBACK_UNAVAILABLE_MSG}\" Do not fabricate an answer or use general knowledge.\n"
        "4. Be concise, educational, and friendly. Mention what the instructor explained in the video.\n"
        f"5. Do NOT generate quizzes, MCQs, or conversational filler.{lang_rule}"
    )

    # Format retrieved transcript chunks with their IDs, timestamps, and topics
    context_sections: list[str] = []
    for c in chunks:
        topic_info = f" | Topic: {c.topic}" if c.topic else ""
        concepts_info = f" | Concepts: {', '.join(c.concepts)}" if c.concepts else ""
        header = f"[{c.chunk_id} | {c.timestamp}{topic_info}{concepts_info}]"
        context_sections.append(f"{header}\n{c.text.strip()}")

    formatted_context = "\n\n".join(context_sections) if context_sections else "No lecture chunks retrieved."

    # Format graph context if provided
    graph_text = ""
    if graph_context:
        graph_text = f"=== CONCEPT KNOWLEDGE GRAPH CONTEXT ===\n{graph_context.strip()}\n=== END OF CONCEPT GRAPH CONTEXT ===\n\n"

    # Format limited recent chat history (up to last 3 turns)
    history_text = ""
    if chat_history:
        recent_turns = chat_history[-3:]
        history_lines = [f"{msg.role.capitalize()}: {msg.content}" for msg in recent_turns]
        history_text = "Recent conversation context:\n" + "\n".join(history_lines) + "\n\n"

    user_content = (
        f"=== RETRIEVED LECTURE CONTEXT ===\n"
        f"{formatted_context}\n"
        f"=== END OF LECTURE CONTEXT ===\n\n"
        f"{graph_text}"
        f"{history_text}"
        f"Student Question: {question.strip()}\n\n"
        f"Answer the student's question based strictly on the lecture context above:"
    )

    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_content},
    ]
