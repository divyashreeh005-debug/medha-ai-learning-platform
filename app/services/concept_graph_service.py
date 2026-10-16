"""Concept Intelligence & Knowledge Graph Service for Medhā (Phase 8).

Extracts concept nodes, identifies evidence-backed relationships
(prerequisite, related_to, used_with, depends_on, follows), persists graphs
to disk, and formats graph visualizations for Streamlit.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from app.config import settings
from app.schemas import (
    ConceptGraphResponse,
    ConceptNode,
    ConceptRelationship,
    ConceptRelationshipType,
    ProcessResponse,
    TopicGroup,
)
from app.services.process_video import load_session

logger = logging.getLogger(__name__)


class ConceptGraphError(Exception):
    """Raised when concept graph processing or loading fails."""


def extract_concept_nodes(session: ProcessResponse) -> list[ConceptNode]:
    """Extract canonical concept nodes from session topics and chunks with timestamp preservation."""
    nodes: list[ConceptNode] = []
    chunk_map = {c.chunk_id: c for c in session.chunks}

    for idx, topic in enumerate(session.topics):
        concept_id = f"concept-{idx + 1:03d}"
        
        # Calculate time boundaries from associated chunks
        topic_chunks = [chunk_map[cid] for cid in topic.chunk_ids if cid in chunk_map]
        if topic_chunks:
            start_t = min(c.start_timestamp for c in topic_chunks)
            end_t = max(c.end_timestamp for c in topic_chunks)
            t_range = f"{topic_chunks[0].start_label} – {topic_chunks[-1].end_label}"
            
            # Meaningful 2-sentence educational Main Idea:
            if topic.learning_content and topic.learning_content.simple_explanation:
                desc = topic.learning_content.simple_explanation
            else:
                full_text = " ".join(c.text.strip() for c in topic_chunks[:3])
                sentences = re.split(r"(?<=[.!?])\s+", full_text)
                clean_sents = [
                    s.strip() for s in sentences
                    if len(s.strip()) > 25 and not any(w in s.lower() for w in [
                        "welcome", "hello", "hi everyone", "hi guys", "subscribe", 
                        "let's get started", "all right guys", "in this video"
                    ])
                ]
                if clean_sents:
                    desc = " ".join(clean_sents[:2])
                else:
                    desc = f"{topic.name} is an essential programming concept introduced in this lecture."
        else:
            start_t = topic.start_time
            end_t = topic.end_time
            t_range = topic.timestamp_range
            desc = f"{topic.name} is an essential programming concept introduced in this lecture."

        node = ConceptNode(
            concept_id=concept_id,
            name=topic.name,
            description=desc,
            topic_id=topic.topic_id,
            topic_name=topic.name,
            source_chunk_ids=list(topic.chunk_ids),
            start_time=start_t,
            end_time=end_t,
            timestamp_range=t_range,
        )
        nodes.append(node)

    return nodes


def extract_concept_relationships(
    concepts: list[ConceptNode],
    session: ProcessResponse,
) -> list[ConceptRelationship]:
    """Identify strictly evidence-backed relationships between concepts.
    
    Supported types:
    - follows: Chronological lecture topic succession
    - depends_on / prerequisite: Explicit functional dependency expressed in lecture
    - used_with: Synergistic concepts frequently combined in code
    - related_to: Conceptual analogy or direct comparison
    """
    relationships: list[ConceptRelationship] = []
    concept_names = {c.name.lower(): c.name for c in concepts}
    chunk_map = {c.chunk_id: c for c in session.chunks}

    # 1. Sequential progression: Every concept 'follows' its predecessor in lecture order
    for i in range(len(concepts) - 1):
        c1 = concepts[i]
        c2 = concepts[i + 1]
        relationships.append(
            ConceptRelationship(
                source=c2.name,
                relation=ConceptRelationshipType.FOLLOWS,
                target=c1.name,
                evidence=f"In the lecture, '{c2.name}' ({c2.timestamp_range}) is taught sequentially after '{c1.name}' ({c1.timestamp_range}).",
            )
        )

    # 2. Textual & discourse relationship rules based on transcript evidence
    # Search each concept's chunks for mentions of other concepts
    for concept in concepts:
        # Aggregate text from this concept's source chunks
        concept_text = " ".join(
            chunk_map[cid].text for cid in concept.source_chunk_ids if cid in chunk_map
        )
        concept_lower = concept_text.lower()

        for other in concepts:
            if other.name == concept.name:
                continue

            other_name_lower = other.name.lower()
            # If the concept text explicitly discusses the other concept
            if other_name_lower in concept_lower or (
                len(other_name_lower) > 4 and other_name_lower.rstrip("s") in concept_lower
            ):
                # Classify relationship based on context cues
                rel_type, evidence_snippet = _classify_relationship(
                    concept.name, other.name, concept_text
                )
                if rel_type:
                    relationships.append(
                        ConceptRelationship(
                            source=concept.name,
                            relation=rel_type,
                            target=other.name,
                            evidence=evidence_snippet,
                        )
                    )

    # Deduplicate relationships by (source, relation, target)
    unique_rels: list[ConceptRelationship] = []
    seen = set()
    for r in relationships:
        key = (r.source, r.relation.value, r.target)
        if key not in seen:
            seen.add(key)
            unique_rels.append(r)

    return unique_rels


def _classify_relationship(
    source_name: str, target_name: str, text: str
) -> tuple[ConceptRelationshipType | None, str]:
    """Helper to detect semantic dependency cues between concepts in transcript text."""
    target_lower = target_name.lower()
    sentences = re.split(r"[.!?]\s+", text)

    for sentence in sentences:
        s_lower = sentence.lower()
        if target_lower not in s_lower and target_lower.rstrip("s") not in s_lower:
            continue

        # Prerequisite / Depends On cues
        if any(w in s_lower for w in ["need", "require", "before", "in order to", "first", "based on", "uses", "using"]):
            return (
                ConceptRelationshipType.DEPENDS_ON,
                f"Transcript evidence in '{source_name}': \"{sentence.strip()}\"",
            )

        # Comparison / Analogy cues
        if any(w in s_lower for w in ["like", "similar to", "unlike", "compared to", "difference between", "instead of"]):
            return (
                ConceptRelationshipType.RELATED_TO,
                f"Transcript comparative evidence: \"{sentence.strip()}\"",
            )

        # Co-usage cues
        if any(w in s_lower for w in ["with", "combine", "together", "iterate over", "loop through"]):
            return (
                ConceptRelationshipType.USED_WITH,
                f"Transcript co-usage evidence: \"{sentence.strip()}\"",
            )

    return (
        ConceptRelationshipType.RELATED_TO,
        f"Concept '{target_name}' is referenced and explained within the context of '{source_name}'.",
    )


def build_and_save_concept_graph(session_id: str) -> ConceptGraphResponse:
    """Extract and persist the concept graph for a session."""
    session = load_session(session_id)
    if session is None:
        raise ConceptGraphError(f"Session '{session_id}' not found.")

    concepts = extract_concept_nodes(session)
    relationships = extract_concept_relationships(concepts, session)

    graph_res = ConceptGraphResponse(
        session_id=session_id,
        concepts=concepts,
        relationships=relationships,
        total_concepts=len(concepts),
        total_relationships=len(relationships),
    )

    save_concept_graph(graph_res)
    return graph_res


def save_concept_graph(graph: ConceptGraphResponse) -> None:
    """Save the concept graph JSON into data/concept_graphs/{session_id}.json."""
    settings.concept_graph_dir.mkdir(parents=True, exist_ok=True)
    out_file = settings.concept_graph_dir / f"{graph.session_id}.json"
    out_file.write_text(
        json.dumps(graph.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Persisted concept graph for session %s to %s", graph.session_id, out_file)


def load_concept_graph(session_id: str) -> ConceptGraphResponse | None:
    """Load cached concept graph from disk if available."""
    settings.concept_graph_dir.mkdir(parents=True, exist_ok=True)
    graph_file = settings.concept_graph_dir / f"{session_id}.json"
    if not graph_file.exists():
        return None
    try:
        data = json.loads(graph_file.read_text(encoding="utf-8"))
        return ConceptGraphResponse.model_validate(data)
    except Exception as exc:
        logger.warning("Failed to load concept graph for session %s: %s", session_id, exc)
        return None


def get_or_create_concept_graph(session_id: str) -> ConceptGraphResponse:
    """Retrieve existing cached concept graph or build and persist on demand."""
    cached = load_concept_graph(session_id)
    if cached is not None:
        return cached
    return build_and_save_concept_graph(session_id)


def get_prerequisites(
    concept_name: str,
    graph: ConceptGraphResponse,
) -> list[dict[str, str]]:
    """Return all concepts that the given concept depends on, requires, or immediately follows."""
    prereqs = []
    c_lower = concept_name.lower()

    for rel in graph.relationships:
        # e.g., For Loops -> depends_on -> Lists
        if rel.source.lower() == c_lower and rel.relation in (
            ConceptRelationshipType.DEPENDS_ON,
            ConceptRelationshipType.PREREQUISITE,
        ):
            prereqs.append({
                "concept": rel.target,
                "relation": rel.relation.value,
                "evidence": rel.evidence,
            })
        # e.g., Variables -> prerequisite -> Basic Calculator Program
        elif rel.target.lower() == c_lower and rel.relation == ConceptRelationshipType.PREREQUISITE:
            prereqs.append({
                "concept": rel.source,
                "relation": "prerequisite",
                "evidence": rel.evidence,
            })
        elif rel.source.lower() == c_lower and rel.relation == ConceptRelationshipType.FOLLOWS:
            prereqs.append({
                "concept": rel.target,
                "relation": "preceded_by",
                "evidence": rel.evidence,
            })

    return prereqs


def get_related_concepts(
    concept_name: str,
    graph: ConceptGraphResponse,
) -> list[dict[str, str]]:
    """Return all outgoing and incoming relationships for a specific concept."""
    related = []
    c_lower = concept_name.lower()

    for rel in graph.relationships:
        if rel.relation == ConceptRelationshipType.FOLLOWS:
            continue
        if rel.source.lower() == c_lower:
            related.append({
                "related_concept": rel.target,
                "direction": "outgoing",
                "relation": rel.relation.value,
                "evidence": rel.evidence,
            })
        elif rel.target.lower() == c_lower:
            related.append({
                "related_concept": rel.source,
                "direction": "incoming",
                "relation": rel.relation.value,
                "evidence": rel.evidence,
            })

    return related


def format_dot_graph(
    graph: ConceptGraphResponse,
    selected_concept: Optional[str] = None,
) -> str:
    """Generate a clean DOT graph string for Streamlit's st.graphviz_chart."""
    lines = [
        "digraph MedhaConceptGraph {",
        "  rankdir=TB;",
        "  graph [bgcolor=transparent, pad=\"0.2\", nodesep=\"0.4\", ranksep=\"0.6\"];",
        "  node [shape=box, style=\"rounded,filled\", fontname=\"Helvetica\", fontsize=10, margin=\"0.2,0.1\"];",
        "  edge [fontname=\"Helvetica\", fontsize=8, color=\"#6c757d\"];",
    ]

    selected_lower = selected_concept.lower() if selected_concept else None

    # Render Concept-Centric Nodes (NO timestamps cluttering the node visual)
    for c in graph.concepts:
        safe_name = c.name.replace('"', '\\"')
        is_selected = (selected_lower is not None and c.name.lower() == selected_lower)
        if is_selected:
            lines.append(
                f'  "{safe_name}" [label="{safe_name}", fillcolor="#0d6efd", fontcolor="#ffffff", color="#0b5ed7", penwidth=2.5];'
            )
        else:
            lines.append(
                f'  "{safe_name}" [label="{safe_name}", fillcolor="#f8f9fa", fontcolor="#212529", color="#ced4da", penwidth=1.2];'
            )

    # Render Non-Follows Edges (to avoid visual clutter from linear sequence)
    # Highlight meaningful dependency, co-usage, and conceptual relationship edges
    meaningful_relations = [
        r for r in graph.relationships if r.relation != ConceptRelationshipType.FOLLOWS
    ]

    # If no non-follows relationships, fallback to rendering follows edges
    display_rels = meaningful_relations if meaningful_relations else graph.relationships

    # Edge styling by relation type
    style_map = {
        ConceptRelationshipType.PREREQUISITE: 'color="#dc3545", penwidth=1.8, style="solid"',
        ConceptRelationshipType.DEPENDS_ON: 'color="#fd7e14", penwidth=1.5, style="solid"',
        ConceptRelationshipType.USED_WITH: 'color="#198754", penwidth=1.2, style="dashed"',
        ConceptRelationshipType.RELATED_TO: 'color="#6f42c1", penwidth=1.0, style="dotted"',
        ConceptRelationshipType.FOLLOWS: 'color="#adb5bd", penwidth=0.8, style="solid"',
    }

    for r in display_rels:
        src = r.source.replace('"', '\\"')
        tgt = r.target.replace('"', '\\"')
        edge_style = style_map.get(r.relation, 'color="#6c757d"')
        label = r.relation.value.replace("_", " ")
        lines.append(f'  "{src}" -> "{tgt}" [label="{label}", {edge_style}];')

    lines.append("}")
    return "\n".join(lines)
