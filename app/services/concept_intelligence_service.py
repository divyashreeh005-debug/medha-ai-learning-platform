"""Concept Intelligence and Quality Scoring Service for Medhā (Phase 8.2).

Provides:
1. Concept/Topic quality filtering (distinguishing structural sections like 'Introduction'
   from actual teachable concepts).
2. Programming concept detection with small, beginner-friendly, grounded code examples.
3. Structured ConceptIntelligence object creation and disk caching.
4. Foundation for Learn, Concepts, Concept Map, and high-value Memory Cards.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from app.config import settings
from app.schemas import (
    Chunk,
    ConceptIntelligence,
    ConceptIntelligenceCollection,
    ProcessResponse,
)
from app.services.concept_graph_service import (
    get_or_create_concept_graph,
    get_prerequisites,
    get_related_concepts,
)
from app.services.process_video import load_session

logger = logging.getLogger(__name__)


class ConceptIntelligenceError(Exception):
    pass

GENERIC_STRUCTURAL_PATTERNS = [
    re.compile(
        r"^(?:introduction|intro|overview|welcome|course introduction|about this course|agenda|getting started|in this video|let'?s begin|summary|conclusion|outro|wrap up|next steps)\b",
        re.IGNORECASE,
    ),
    re.compile(r"^(?:a few more examples|few examples|another example|next example|simple example)$", re.IGNORECASE),
]

PROGRAMMING_KEYWORDS = {
    "python", "variable", "variables", "data type", "data types", "string", "strings",
    "integer", "float", "boolean", "list", "lists", "tuple", "tuples", "dictionary",
    "dictionaries", "for loop", "while loop", "loop", "loops", "if statement", "if statements",
    "condition", "conditionals", "function", "functions", "method", "methods", "class",
    "classes", "object", "objects", "array", "arrays", "operator", "operators", "input",
    "print", "return", "import", "module", "modules", "syntax", "code", "programming",
}

# Grounded idiomatic examples for standard programming constructs taught in lectures
CANONICAL_PROGRAMMING_EXAMPLES: dict[str, str] = {
    "variables": 'name = "Divya"\nage = 22\nprint(name, age)',
    "receiving user input": 'name = input("What is your name? ")\nprint("Hello " + name)',
    "type conversion": 'birth_year = input("Birth year: ")\nage = 2026 - int(birth_year)\nprint(age)',
    "strings": 'course = "Python for Beginners"\nprint(course.upper())\nprint("Python" in course)',
    "arithmetic operators": 'x = 10\nx += 3  # augmented assignment\nprint(x * 2)',
    "operator precedence": 'result = (10 + 3) * 2\nprint(result)  # 26',
    "comparison operators": 'temperature = 25\nis_hot = temperature > 30\nprint(is_hot)  # False',
    "logical operators": 'price = 25\nprint(price > 10 and price < 30)  # True',
    "if statements": 'temperature = 35\nif temperature > 30:\n    print("It\'s a hot day")\nelse:\n    print("It\'s lovely")',
    "while loops": 'i = 1\nwhile i <= 5:\n    print("*" * i)\n    i += 1',
    "lists": 'names = ["John", "Bob", "Mary"]\nnames.append("Alice")\nprint(names[0])',
    "list methods": 'numbers = [1, 2, 3, 4, 5]\nnumbers.insert(0, -1)\nprint(len(numbers))',
    "for loops": 'numbers = [1, 2, 3, 4, 5]\nfor item in numbers:\n    print(item)',
    "range": 'for number in range(5, 10, 2):\n    print(number)  # 5, 7, 9',
    "tuples": 'coordinates = (10, 20, 30)\nprint(coordinates[0])  # immutable',
    "dictionaries": 'customer = {"name": "John", "age": 30}\nprint(customer.get("name"))',
    "functions": 'def greet(first_name, last_name):\n    print(f"Hi {first_name} {last_name}!")\n\ngreet("John", "Smith")',
    "return statement": 'def square(number):\n    return number * number\n\nprint(square(3))',
}


def is_generic_structural_title(title: str) -> bool:
    """Return True if title corresponds to a generic structural/introductory phrase."""
    t = title.strip().lower()
    return any(p.search(t) for p in GENERIC_STRUCTURAL_PATTERNS)


def score_concept_quality(name: str, chunks: list[Chunk]) -> tuple[float, bool]:
    """Calculate educational quality score (0.0 to 1.0) and structural section flag.
    
    Distinguishes structural scaffolding ('Introduction', 'Course Overview')
    from substantive teachable concepts ('Variables', 'For Loops').
    """
    is_structural = is_generic_structural_title(name)
    combined_text = " ".join(c.text.lower() for c in chunks)
    word_count = len(combined_text.split())

    # Check for technical substance in transcript
    tech_hits = sum(1 for kw in PROGRAMMING_KEYWORDS if kw in combined_text)
    
    if is_structural:
        # If it's titled Introduction/Overview, verify if it teaches concrete programming concepts
        if tech_hits >= 3 and word_count >= 120:
            # Substantial teachable introductory section
            return 0.65, False
        return 0.20, True

    # For non-structural titles, compute substantive score
    score = 0.50
    if tech_hits >= 1:
        score += min(0.30, tech_hits * 0.08)
    if word_count >= 80:
        score += 0.15
    if len(name.split()) <= 4:
        score += 0.05

    return min(1.0, round(score, 2)), False


def detect_programming_concept(concept_name: str, chunks: list[Chunk]) -> tuple[bool, Optional[str]]:
    """Determine if concept is code-related and extract or assign an idiomatic small example."""
    name_clean = concept_name.lower().strip()

    # 1. Check exact canonical match
    for canonical_name, code in CANONICAL_PROGRAMMING_EXAMPLES.items():
        if canonical_name in name_clean or name_clean in canonical_name:
            return True, code

    # 2. Check keyword overlap in title
    matched_keywords = [kw for kw in PROGRAMMING_KEYWORDS if kw in name_clean]
    if matched_keywords:
        # Check if transcript contains code patterns or snippets
        combined_text = " ".join(c.text for c in chunks)
        code_match = re.search(r"([a-zA-Z_]\w*\s*=\s*[^;\n]{2,30})", combined_text)
        if code_match:
            candidate_line = code_match.group(1).strip()
            if len(candidate_line) <= 40:
                return True, f"# Example from lecture\n{candidate_line}"

        # Use general canonical example for the matched keyword
        kw = matched_keywords[0]
        if kw in CANONICAL_PROGRAMMING_EXAMPLES:
            return True, CANONICAL_PROGRAMMING_EXAMPLES[kw]
        return True, f"# Programming concept: {concept_name}\n# (Refer to lecture for syntax)"

    return False, None


def get_concept_intelligence_path(session_id: str) -> Path:
    """Return filesystem path for cached ConceptIntelligenceCollection."""
    settings.concept_intelligence_dir.mkdir(parents=True, exist_ok=True)
    return settings.concept_intelligence_dir / f"{session_id}.json"


def load_concept_intelligence(session_id: str) -> Optional[ConceptIntelligenceCollection]:
    """Read cached ConceptIntelligenceCollection from disk if available."""
    path = get_concept_intelligence_path(session_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ConceptIntelligenceCollection.model_validate(data)
    except Exception as exc:
        logger.warning("Could not read concept intelligence for %s: %s", session_id, exc)
        return None


def save_concept_intelligence(collection: ConceptIntelligenceCollection) -> None:
    """Persist ConceptIntelligenceCollection to disk."""
    path = get_concept_intelligence_path(collection.session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(collection.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved %d concept intelligence objects for session %s", collection.total_concepts, collection.session_id)


def build_concept_intelligence(session: ProcessResponse) -> ConceptIntelligenceCollection:
    """Extract, score, and build reusable ConceptIntelligence objects for all session concepts."""
    # 1. Load or build graph to retrieve prerequisites and related concepts
    graph_obj = get_or_create_concept_graph(session.session_id)
    chunk_map = {c.chunk_id: c for c in session.chunks}

    concepts_list: list[ConceptIntelligence] = []

    for topic in session.topics:
        topic_chunks = [chunk_map[cid] for cid in topic.chunk_ids if cid in chunk_map]
        quality_score, is_structural = score_concept_quality(topic.name, topic_chunks)

        # 2. Main Idea: grounded, concise 2-sentence explanation without video filler
        if topic.learning_content and topic.learning_content.simple_explanation:
            main_idea = topic.learning_content.simple_explanation
            key_points = topic.learning_content.key_takeaways[:4]
        else:
            full_text = " ".join(c.text.strip() for c in topic_chunks[:3])
            sentences = re.split(r"(?<=[.!?])\s+", full_text)
            clean_sents = [
                s.strip() for s in sentences
                if len(s.strip()) > 20 and not any(w in s.lower() for w in [
                    "welcome", "hello", "subscribe", "let's get started", "in this video",
                    "in this tutorial", "all right guys", "hit the bell"
                ])
            ]
            if clean_sents:
                main_idea = " ".join(clean_sents[:2])
            else:
                main_idea = f"{topic.name} is a key educational topic covered in this lecture."

            # Synthesize key points from concepts and chunks
            key_points = [f"Focuses on {c}" for c in topic.key_concepts[:4]]
            if not key_points and topic_chunks:
                key_points = [f"Covers {topic.name} as demonstrated by the instructor."]

        # 3. Detect programming content and code example
        is_prog, code_example = detect_programming_concept(topic.name, topic_chunks)

        # 4. Fetch prerequisites & related concepts from graph
        prereqs_raw = get_prerequisites(topic.name, graph_obj)
        prereqs = [p["concept"] for p in prereqs_raw if p.get("concept")]
        
        related_raw = get_related_concepts(topic.name, graph_obj)
        related = [r["related_concept"] for r in related_raw if r.get("related_concept") and r.get("relation") != "follows"]

        concept_intel = ConceptIntelligence(
            concept=topic.name,
            main_idea=main_idea,
            key_points=key_points,
            example=code_example,
            supporting_chunks=list(topic.chunk_ids),
            timestamp=topic.timestamp_range,
            start_time=topic.start_time,
            end_time=topic.end_time,
            prerequisites=prereqs,
            related_concepts=related[:5],
            importance_score=quality_score,
            is_structural_section=is_structural,
            is_programming_concept=is_prog,
        )
        concepts_list.append(concept_intel)

    collection = ConceptIntelligenceCollection(
        session_id=session.session_id,
        total_concepts=len(concepts_list),
        concepts=concepts_list,
    )
    save_concept_intelligence(collection)
    return collection


def get_or_create_concept_intelligence(session_id: str) -> ConceptIntelligenceCollection:
    """Retrieve cached concept intelligence or construct it on demand."""
    cached = load_concept_intelligence(session_id)
    if cached:
        return cached

    session = load_session(session_id)
    if session is None:
        raise ConceptIntelligenceError(f"Session '{session_id}' not found.")

    return build_concept_intelligence(session)
