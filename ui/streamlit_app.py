"""Medhā — AI-Powered Video-to-Interactive Learning Platform.

Phase 8.1: UI/UX Refinement & Multilingual Learning Experience
Information Architecture:
  🏠 Overview · 📚 Learn · 🧠 Concepts · 🕸 Concept Map · 📝 Memory Cards · 🤖 Ask Medhā · 📜 Sources
"""

import os
import random
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
import streamlit as st

import importlib
try:
    import app.schemas
    importlib.reload(app.schemas)
    from app.schemas import (
        LANGUAGE_DISPLAY_NAMES,
        ConceptGraphResponse,
        ConceptIntelligenceCollection,
        MemoryCardCollection,
        SupportedLanguage,
    )
except (ImportError, AttributeError):
    from app.schemas import ConceptGraphResponse
    try:
        from app.schemas import MemoryCardCollection
    except ImportError:
        MemoryCardCollection = None
    try:
        from app.schemas import SupportedLanguage
    except ImportError:
        from enum import Enum
        class SupportedLanguage(str, Enum):
            ENGLISH = "en"
            HINDI = "hi"
            KANNADA = "kn"
            TELUGU = "te"
            TAMIL = "ta"
            MALAYALAM = "ml"
    LANGUAGE_DISPLAY_NAMES = {
        "en": "English",
        "hi": "\u0939\u093f\u0928\u094d\u0926\u0940 (Hindi)",
        "kn": "\u0c95\u0ca8\u0ccd\u0ca8\u0ca1 (Kannada)",
        "te": "\u0c24\u0c46\u0c32\u0c41\u0c17\u0c41 (Telugu)",
        "ta": "\u0ba4\u0bae\u0bbf\u0bb4\u0bcd (Tamil)",
        "ml": "\u0d2e\u0d32\u0d2f\u0d3e\u0d33\u0d02 (Malayalam)",
    }
from app.services.concept_graph_service import (
    format_dot_graph,
    get_prerequisites,
    get_related_concepts,
)
try:
    import ui.dashboard_helpers
    importlib.reload(ui.dashboard_helpers)
except Exception:
    pass

from ui.dashboard_helpers import (
    calculate_topic_progress,
    filter_chunks_for_topic,
    format_concept_badges,
    format_lecture_title,
    get_chronological_topics,
    get_next_card_index,
    get_next_topic_index,
    get_previous_card_index,
    get_previous_topic_index,
    get_youtube_jump_url,
    load_cached_learning_content,
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def fetch_sessions() -> list[dict[str, Any]]:
    """Fetch saved sessions list from the FastAPI backend."""
    try:
        resp = requests.get(f"{API_BASE_URL}/api/sessions", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


def fetch_llm_status() -> dict[str, Any]:
    """Check availability of local Qwen LLM runtime."""
    try:
        resp = requests.get(f"{API_BASE_URL}/api/llm/status", timeout=2)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {"available": False, "runtime": "Unavailable", "details": "Backend or LLM offline"}


# ==============================================================================
# UI Component Renderers
# ==============================================================================


def render_header() -> None:
    """Render application header and educational mission subtitle."""
    st.title("Medhā (मेधा)")
    st.caption("AI-Powered Video-to-Interactive Learning Platform · Watch → Understand → Explore → Navigate")


def render_sidebar() -> None:
    """Render sidebar with lecture language info, AI model status, and saved session management."""
    with st.sidebar:
        result = st.session_state.get("result")
        if result:
            st.header("📖 Lecture Info")
            src_lang = result.get("detected_language") or result.get("language") or "en"
            lang_display = LANGUAGE_DISPLAY_NAMES.get(src_lang, src_lang.upper())
            if src_lang != "en":
                st.info(f"🌐 **Spoken Language**: {lang_display}\n\n*Medhā automatically translated this lecture to English for unified concept intelligence.*")
            else:
                st.caption(f"🌐 **Spoken Language**: {lang_display}")
            st.divider()

        st.header("🤖 Local AI Engine")
        llm_info = fetch_llm_status()
        if llm_info.get("available"):
            st.success(f"🟢 **{llm_info.get('runtime', 'LM Studio')}** Online")
        else:
            st.warning(f"⚪ **LLM Server Offline**\n\n{llm_info.get('details', '')}")

        gemma_dot = "🟢" if llm_info.get("gemma_available") else "⚪"
        qwen_dot = "🟢" if llm_info.get("qwen_available") else "⚪"
        gemma_name = llm_info.get("gemma_model") or "gemma-4-e2b-it-qat"
        qwen_name = llm_info.get("qwen_model") or "qwen3-8b"

        st.markdown(
            f"""
- {gemma_dot} **Translation**: `{gemma_name}`
- {qwen_dot} **Reasoning & RAG**: `{qwen_name}`
- 🟢 **Embeddings**: `all-MiniLM-L6-v2` (Local)
- 🟢 **Speech-to-Text**: `Whisper base` (Local)
            """
        )
        st.divider()
        st.header("Saved Lectures")
        sessions = fetch_sessions()
        if sessions:
            session_options = {}
            for s in sessions:
                display_name = s.get("title") or f"Lecture ({s.get('video_id', 'unknown')})"
                label = f"{display_name[:40]}... ({s['chunk_count']} chunks)" if len(display_name) > 40 else f"{display_name} ({s['chunk_count']} chunks)"
                session_options[label] = s["session_id"]

            selected_label = st.selectbox(
                "Load previous lecture",
                options=list(session_options.keys()),
                key="sidebar_session_select",
            )
            if st.button("Load Session", use_container_width=True, type="secondary"):
                sid = session_options[selected_label]
                with st.spinner("Loading session..."):
                    try:
                        resp = requests.get(f"{API_BASE_URL}/api/sessions/{sid}", timeout=10)
                        if resp.status_code == 200:
                            st.session_state["result"] = resp.json()
                            st.session_state["selected_topic_idx"] = 0
                            if "topic_dropdown_selector" in st.session_state:
                                st.session_state["topic_dropdown_selector"] = 0
                            st.session_state["search_results"] = None
                            st.session_state["card_idx"] = 0
                            st.session_state["card_revealed"] = False
                            st.rerun()
                        else:
                            st.error("Failed to load session from server.")
                    except requests.RequestException as exc:
                        st.error(f"Could not connect to server: {exc}")
        else:
            st.info("No saved sessions found. Process a YouTube video to begin.")


def render_video_input() -> None:
    """Render YouTube URL input form and processing trigger."""
    st.subheader("Process a Lecture")
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=kqtD5dpn9C8",
            label_visibility="collapsed",
        )
    with col_btn:
        process = st.button("Process Video", type="primary", use_container_width=True)

    if process:
        if not url.strip():
            st.warning("Please enter a valid YouTube video URL.")
        else:
            with st.spinner("Fetching lecture transcript, chunking, and extracting topics..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/api/process",
                        json={"youtube_url": url.strip()},
                        timeout=180,
                    )
                except requests.RequestException as exc:
                    st.error(f"Cannot reach backend at {API_BASE_URL}. Ensure the FastAPI server is running. ({exc})")
                else:
                    if response.status_code >= 400:
                        detail = (
                            response.json().get("detail")
                            if response.headers.get("content-type", "").startswith("application/json")
                            else response.text
                        )
                        st.error(f"Processing failed: {detail}")
                    else:
                        st.session_state["result"] = response.json()
                        st.session_state["selected_topic_idx"] = 0
                        if "topic_dropdown_selector" in st.session_state:
                            st.session_state["topic_dropdown_selector"] = 0
                        st.session_state["search_results"] = None
                        st.session_state["card_idx"] = 0
                        st.session_state["card_revealed"] = False
                        st.rerun()


# ==============================================================================
# TAB 1: OVERVIEW
# ==============================================================================


def render_tab_overview(
    result: dict[str, Any],
    topics: list[dict[str, Any]],
    current_idx: int,
) -> None:
    """Render high-level lecture overview, metadata, and topic table of contents."""
    title = format_lecture_title(result.get("title"), result.get("video_id"), result.get("session_id"))
    st.markdown(f"## 🎬 {title}")

    # Badges row
    col1, col2, col3, col4, col5 = st.columns(5)
    source_type = result.get("source_type", "youtube").capitalize()
    duration_str = result.get("formatted_duration") or "—"
    det_lang = result.get("detected_language") or "en"
    
    col1.metric("Source", source_type)
    col2.metric("Duration", duration_str)
    col3.metric("Lecture Language", det_lang.upper())
    col4.metric("Topics Identified", len(topics))
    col5.metric("Transcript Chunks", result.get("chunk_count", 0))

    if result.get("youtube_url"):
        st.markdown(f"🔗 **Original Video:** [{result['youtube_url']}]({result['youtube_url']})")

    st.divider()
    st.markdown("### 📋 Lecture Table of Contents")
    st.caption("Chronological progression of learning modules identified in this lecture:")

    for i, t in enumerate(topics):
        is_active = (i == current_idx)
        with st.container(border=True):
            col_t_title, col_t_time, col_t_btn = st.columns([5, 2, 2])
            with col_t_title:
                prefix = "👉 **" if is_active else ""
                suffix = "** (Current)" if is_active else ""
                st.markdown(f"{prefix}{i + 1:02d}. {t['name']}{suffix}")
                raw_concepts = t.get("key_concepts", [])
                cleaned_concepts = format_concept_badges(raw_concepts)
                if cleaned_concepts:
                    st.caption("Concepts: " + ", ".join(cleaned_concepts[:5]))
            with col_t_time:
                st.markdown(f"⏱️ `{t['timestamp_range']}`")
            with col_t_btn:
                if st.button(f"Study Topic {i + 1}", key=f"btn_study_{i}"):
                    st.session_state["selected_topic_idx"] = i
                    st.session_state["topic_dropdown_selector"] = i
                    st.rerun()


# ==============================================================================
# TAB 2: LEARN (Structured Educational Content)
# ==============================================================================


def _set_topic(idx: int) -> None:
    """Helper to set selected topic index and synchronize dropdown state."""
    st.session_state["selected_topic_idx"] = idx
    st.session_state["topic_dropdown_selector"] = idx


def render_tab_learn(
    session_id: str,
    video_id: str,
    topics: list[dict[str, Any]],
    current_idx: int,
) -> None:
    """Render topic-by-topic interactive learning experience."""
    total_topics = len(topics)
    current_idx = max(0, min(current_idx, total_topics - 1))
    topic = topics[current_idx]
    topic_id = topic["topic_id"]

    # Top Navigation & Progress Header
    ratio, label = calculate_topic_progress(current_idx, total_topics)
    col_prog_lbl, col_prog_bar = st.columns([1, 4])
    with col_prog_lbl:
        st.markdown(f"**{label}**")
    with col_prog_bar:
        st.progress(ratio)

    # Topic Dropdown (Safe Two-Way Synchronization)
    topic_options = list(range(total_topics))
    if "topic_dropdown_selector" not in st.session_state or st.session_state["topic_dropdown_selector"] != current_idx:
        st.session_state["topic_dropdown_selector"] = current_idx

    chosen_idx = st.selectbox(
        "Jump to Topic",
        options=topic_options,
        index=current_idx,
        format_func=lambda i: f"{i + 1:02d} — {topics[i]['name']} ({topics[i]['timestamp_range']})",
        key="topic_dropdown_selector",
        on_change=lambda: st.session_state.update({"selected_topic_idx": st.session_state["topic_dropdown_selector"]}),
    )

    # Active Topic Card Header
    with st.container(border=True):
        col_name, col_yt = st.columns([3, 1])
        with col_name:
            st.markdown(f"## {topic['name']}")
            st.markdown(f"⏱️ **Timestamp Range:** `{topic['timestamp_range']}`")
        with col_yt:
            jump_url = get_youtube_jump_url(video_id, topic.get("start_time", 0))
            if video_id and not video_id.startswith("local_"):
                st.link_button(
                    f"⏱ Jump to {topic['start_label']}",
                    jump_url,
                    type="primary",
                    use_container_width=True,
                )
            else:
                st.caption(f"Starts at {topic['start_label']}")

        raw_concepts = topic.get("key_concepts", [])
        cleaned_concepts = format_concept_badges(raw_concepts)
        if cleaned_concepts:
            badges = " ".join(f"`{c}`" for c in cleaned_concepts)
            st.markdown(f"**Key Concepts:** {badges}")

    # Learning Content Section (Summary, Explain Simply, Analogy, Takeaways)
    lang = st.session_state.get("learning_language", "en")
    cache_key = f"content_{session_id}_{topic_id}_{lang}"
    content = st.session_state.get(cache_key)

    if not content:
        # Check disk cache
        loaded = load_cached_learning_content(session_id, topic_id, language=lang)
        if loaded:
            content = loaded
            st.session_state[cache_key] = content

    if content:
        st.markdown("### 📖 Comprehensive Summary")
        summary_text = content.get("summary", "")
        # Render clean paragraphs
        for para in summary_text.split("\n\n"):
            if para.strip():
                st.write(para.strip())

        st.markdown("### 💡 Explain Simply")
        st.info(content.get("simple_explanation", "No explanation available."))

        st.markdown("### 🌍 Real-world Analogy")
        st.success(f"💡 {content.get('analogy', 'No analogy available.')}")

        st.markdown("### 🔑 Key Takeaways")
        takeaways = content.get("key_takeaways", [])
        if takeaways:
            for item in takeaways:
                st.markdown(f"• {item}")
        else:
            st.write("No key takeaways recorded.")

        # Content Regeneration Option
        with st.expander("⚙️ Options"):
            lang_label = LANGUAGE_DISPLAY_NAMES.get(lang, lang)
            if st.button(f"Regenerate Content in {lang_label}", key=f"btn_regen_{topic_id}_{lang}"):
                _trigger_content_generation_multi_stage(session_id, topic_id, lang=lang, force=True)
    else:
        lang_label = LANGUAGE_DISPLAY_NAMES.get(lang, lang)
        st.info(f"Educational learning content in **{lang_label}** is ready to be generated for this topic.")
        if st.button(
            f"✨ Generate Learning Content ({lang_label})",
            type="primary",
            key=f"btn_gen_{topic_id}_{lang}",
        ):
            _trigger_content_generation_multi_stage(session_id, topic_id, lang=lang, force=False)

    # Bottom Previous / Next Topic Navigation Buttons
    st.divider()
    col_prev, col_center, col_next = st.columns([3, 4, 3])
    prev_idx = get_previous_topic_index(current_idx)
    next_idx = get_next_topic_index(current_idx, total_topics)

    with col_prev:
        prev_disabled = prev_idx is None
        prev_label = f"← Previous: {topics[prev_idx]['name'][:20]}..." if prev_idx is not None else "← Previous Topic"
        if st.button(prev_label, disabled=prev_disabled, use_container_width=True, key="btn_nav_prev"):
            _set_topic(prev_idx)
            st.rerun()

    with col_center:
        st.markdown(
            f"<div style='text-align: center; padding-top: 8px; color: gray;'>"
            f"Topic {current_idx + 1} of {total_topics}</div>",
            unsafe_allow_html=True,
        )

    with col_next:
        next_disabled = next_idx is None
        next_label = f"Next: {topics[next_idx]['name'][:20]}... →" if next_idx is not None else "Next Topic →"
        if st.button(next_label, disabled=next_disabled, use_container_width=True, key="btn_nav_next"):
            _set_topic(next_idx)
            st.rerun()


def _trigger_content_generation_multi_stage(
    session_id: str, topic_id: str, lang: str = "en", force: bool = False
) -> None:
    """Generate educational content with genuine staged progress feedback."""
    stage_box = st.empty()
    progress_bar = st.progress(0.1)

    stage_box.info("Step 1 of 4: Gathering topic transcript chunks & contextual grounding...")
    time.sleep(0.3)
    progress_bar.progress(0.3)

    stage_box.info("Step 2 of 4: Synthesizing educational summary with local Qwen3-8B...")
    progress_bar.progress(0.5)

    try:
        resp = requests.post(
            f"{API_BASE_URL}/api/topics/generate",
            json={
                "session_id": session_id,
                "topic_id": topic_id,
                "force_regenerate": force,
                "language": lang,
            },
            timeout=240,
        )
        if resp.status_code == 200:
            stage_box.info("Step 3 of 4: Formulating intuition, beginner explanation & analogy...")
            progress_bar.progress(0.8)
            time.sleep(0.2)

            stage_box.info("Step 4 of 4: Distilling actionable key takeaways...")
            progress_bar.progress(1.0)
            time.sleep(0.2)

            content = resp.json()
            st.session_state[f"content_{session_id}_{topic_id}_{lang}"] = content
            stage_box.success("Educational content successfully prepared!")
            st.rerun()
        else:
            progress_bar.empty()
            detail = resp.json().get("detail", resp.text)
            stage_box.error(f"Generation failed: {detail}")
    except requests.RequestException as exc:
        progress_bar.empty()
        stage_box.error(f"Could not reach backend LLM service: {exc}")


# ==============================================================================
# TAB 3: CONCEPTS (Clean Concept Cards & Details)
# ==============================================================================


def render_tab_concepts(session_id: str, video_id: str) -> None:
    """Render concept index with clean cards, main ideas, and prerequisite details."""
    lang = st.session_state.get("learning_language", "en")
    st.markdown("## 🧠 Concept Intelligence")
    st.caption("Individual concepts, techniques, and ideas extracted from the lecture with clear main ideas and dependencies.")

    try:
        resp = requests.get(
            f"{API_BASE_URL}/api/sessions/{session_id}/concept-graph",
            params={"language": lang},
            timeout=15,
        )
        if resp.status_code != 200:
            st.info("Concept intelligence is not yet available for this session.")
            return
        graph_obj = ConceptGraphResponse.model_validate(resp.json())
    except Exception as exc:
        st.warning(f"Could not load concepts: {exc}")
        return

    if not graph_obj.concepts:
        st.info("No concepts extracted for this lecture.")
        return

    # Concept Selection
    concept_names = [c.name for c in graph_obj.concepts]
    selected_name = st.selectbox(
        "Select a Concept to inspect:",
        options=concept_names,
        key=f"tab_concept_select_{session_id}_{lang}",
    )

    selected_concept = next((c for c in graph_obj.concepts if c.name == selected_name), graph_obj.concepts[0])

    # Concept Card View
    with st.container(border=True):
        col_info, col_time = st.columns([3, 1])
        with col_info:
            st.markdown(f"### 📌 {selected_concept.name}")
            st.markdown(f"**Topic Association:** `{selected_concept.topic_name or 'General'}`")
            if selected_concept.description:
                st.markdown(f"**Main Idea:**\n\n{selected_concept.description}")
            else:
                st.caption("Key conceptual technique extracted from lecture discourse.")
        with col_time:
            st.markdown(f"⏱️ **Timestamp:** `{selected_concept.timestamp_range}`")
            if video_id and not video_id.startswith("local_"):
                jump_url = get_youtube_jump_url(video_id, int(selected_concept.start_time))
                st.link_button(
                    f"⏱ Jump to {selected_concept.timestamp_range.split('–')[0].strip()}",
                    jump_url,
                    key=f"concept_tab_jump_{selected_concept.concept_id}",
                    type="primary",
                    use_container_width=True,
                )

    # Dependencies & Prerequisites
    st.markdown("#### 🧭 What should I learn before this concept?")
    prereqs = get_prerequisites(selected_concept.name, graph_obj)
    if prereqs:
        for p in prereqs:
            rel = p["relation"].replace("_", " ").upper()
            st.markdown(f"- **{p['concept']}** (`{rel}`)")
            if p.get("evidence"):
                st.caption(f"  *Discourse Evidence:* {p['evidence']}")
    else:
        st.success("✨ Foundational concept introduced early in the lecture. No earlier prerequisites required!")

    # Connected Concepts
    st.markdown("#### 🔗 Connected Concepts")
    related = get_related_concepts(selected_concept.name, graph_obj)
    meaningful = [r for r in related if r["relation"] != "follows"]
    if meaningful:
        for r in meaningful:
            rel = r["relation"].replace("_", " ").title()
            arrow = "➔" if r["direction"] == "outgoing" else "⬅"
            st.markdown(f"- **{selected_concept.name}** {arrow} *{rel}* {arrow} **{r['related_concept']}**")
    else:
        st.info("No specific cross-concept dependencies identified.")


# ==============================================================================
# TAB 4: CONCEPT MAP (Visual Knowledge Graph)
# ==============================================================================


def render_tab_concept_map(session_id: str, video_id: str = "") -> None:
    """Render visual concept-centric knowledge graph."""
    lang = st.session_state.get("learning_language", "en")
    st.markdown("## 🕸️ Lecture Concept Map")
    st.caption("Visual representation of concepts and their semantic dependencies. Node labels represent clean concepts; timestamps remain secondary metadata.")

    try:
        resp = requests.get(
            f"{API_BASE_URL}/api/sessions/{session_id}/concept-graph",
            params={"language": lang},
            timeout=15,
        )
        if resp.status_code != 200:
            st.info("Concept graph is not yet available for this session.")
            return
        graph_obj = ConceptGraphResponse.model_validate(resp.json())
    except Exception as exc:
        st.warning(f"Could not load concept graph: {exc}")
        return

    if not graph_obj.concepts:
        st.info("No concepts extracted for this lecture.")
        return

    # Metrics Row
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Concepts", graph_obj.total_concepts)
    m2.metric("Evidence-Backed Relations", graph_obj.total_relationships)
    meaningful_count = len([r for r in graph_obj.relationships if r.relation.value != "follows"])
    m3.metric("Semantic Connections", meaningful_count)

    concept_names = [c.name for c in graph_obj.concepts]
    highlight_concept = st.selectbox(
        "Highlight Concept in Graph:",
        options=["(None)"] + concept_names,
        key=f"map_highlight_{session_id}_{lang}",
    )
    sel = None if highlight_concept == "(None)" else highlight_concept

    dot_str = format_dot_graph(graph_obj, selected_concept=sel)
    st.graphviz_chart(dot_str, use_container_width=True)

    # Phase 8.2: Rich Concept Intelligence Inspection Card
    if sel:
        intel_map: dict[str, Any] = {}
        try:
            ci_resp = requests.get(f"{API_BASE_URL}/api/sessions/{session_id}/concept-intelligence", timeout=10)
            if ci_resp.status_code == 200 and ConceptIntelligenceCollection is not None:
                ci_data = ConceptIntelligenceCollection.model_validate(ci_resp.json())
                for item in ci_data.concepts:
                    intel_map[item.concept.lower()] = item
        except Exception:
            pass

        item = intel_map.get(sel.lower())
        st.markdown(f"### 💡 Concept Intelligence: **{sel}**")
        with st.container(border=True):
            if item:
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**Main Idea**: {item.main_idea}")
                with c2:
                    if item.timestamp:
                        st.caption(f"⏱️ **Timestamp**: `{item.timestamp}`")
                    if item.start_time is not None and video_id and not video_id.startswith("local_"):
                        jump_url = get_youtube_jump_url(video_id, int(item.start_time))
                        t_lbl = item.timestamp.split('–')[0].strip() if item.timestamp else ""
                        st.link_button(
                            f"▶️ Jump to Lecture ({t_lbl})",
                            jump_url,
                            key=f"btn_jump_concept_{sel}",
                        )

                if item.key_points:
                    st.markdown("**Key Takeaways:**")
                    for pt in item.key_points:
                        st.markdown(f"- {pt}")

                if getattr(item, "example", None):
                    st.markdown("**Beginner-Friendly Code Example:**")
                    st.code(item.example, language="python")

                r_cols = st.columns(2)
                with r_cols[0]:
                    if item.prerequisites:
                        st.markdown(f"**Prerequisites:** {', '.join(item.prerequisites)}")
                    else:
                        st.markdown("**Prerequisites:** *Foundational concept*")
                with r_cols[1]:
                    if item.related_concepts:
                        st.markdown(f"**Related Concepts:** {', '.join(item.related_concepts)}")
                    else:
                        st.markdown("**Related Concepts:** *None identified*")
            else:
                st.markdown(f"*{sel}* is an identified topic concept in this lecture.")



# ==============================================================================
# TAB 5: MEMORY CARDS (Interactive Revision Cards)
# ==============================================================================


def render_tab_memory_cards(session_id: str, video_id: str) -> None:
    """Render interactive revision flashcards (Question front, Grounded answer back)."""
    lang = st.session_state.get("learning_language", "en")
    st.markdown("## 📝 Revision Memory Cards")
    st.caption("Active recall flashcards grounded in the lecture. Test your understanding, reveal the answer, and jump to the exact lecture timestamp.")

    try:
        resp = requests.get(
            f"{API_BASE_URL}/api/sessions/{session_id}/memory-cards",
            params={"language": lang},
            timeout=20,
        )
        if resp.status_code != 200:
            st.info("Memory cards are not yet available for this session.")
            return
        card_collection = MemoryCardCollection.model_validate(resp.json())
    except Exception as exc:
        st.warning(f"Could not load memory cards: {exc}")
        return

    cards = card_collection.cards
    if not cards:
        st.info("No memory cards generated for this lecture.")
        return

    # Track order and card index in session state
    order_key = f"card_order_{session_id}_{lang}"
    if order_key not in st.session_state or len(st.session_state[order_key]) != len(cards):
        st.session_state[order_key] = list(range(len(cards)))

    order = st.session_state[order_key]
    total_cards = len(order)

    if "card_idx" not in st.session_state:
        st.session_state["card_idx"] = 0
    if "card_revealed" not in st.session_state:
        st.session_state["card_revealed"] = False

    current_card_pos = max(0, min(st.session_state["card_idx"], total_cards - 1))
    st.session_state["card_idx"] = current_card_pos
    actual_card_idx = order[current_card_pos]
    card = cards[actual_card_idx]

    # Controls Header
    col_counter, col_shuffle = st.columns([4, 1])
    with col_counter:
        st.markdown(f"**Card {current_card_pos + 1} of {total_cards}** · Concept: `{card.concept_name}`")
    with col_shuffle:
        if st.button("🔀 Shuffle", use_container_width=True):
            random.shuffle(st.session_state[order_key])
            st.session_state["card_idx"] = 0
            st.session_state["card_revealed"] = False
            st.rerun()

    # Flashcard Visual Card Container
    with st.container(border=True):
        st.markdown(f"### ❓ {card.front_prompt}")
        st.caption(f"Topic: {getattr(card, 'topic_name', None) or 'General'} · `{card.timestamp_range}`")

        st.divider()

        revealed = st.session_state["card_revealed"]
        if revealed:
            st.markdown("#### 💡 Explanation & Answer")
            st.success(card.back_answer)
            if getattr(card, "example", None):
                st.markdown("**Beginner-Friendly Code Example:**")
                st.code(card.example, language="python")
            if video_id and not video_id.startswith("local_"):
                jump_url = get_youtube_jump_url(video_id, int(card.start_time))
                st.link_button(
                    f"⏱ Review lecture at {card.timestamp_range.split('–')[0].strip()}",
                    jump_url,
                    key=f"card_jump_{card.card_id}",
                )
            if st.button("Hide Answer", key=f"btn_hide_{card.card_id}"):
                st.session_state["card_revealed"] = False
                st.rerun()
        else:
            if st.button("👀 Reveal Answer", type="primary", key=f"btn_reveal_{card.card_id}"):
                st.session_state["card_revealed"] = True
                st.rerun()

    # Navigation Controls
    col_cprev, col_spacer, col_cnext = st.columns([2, 4, 2])
    with col_cprev:
        if st.button("← Previous Card", disabled=(current_card_pos <= 0), use_container_width=True):
            st.session_state["card_idx"] = get_previous_card_index(current_card_pos, total_cards)
            st.session_state["card_revealed"] = False
            st.rerun()
    with col_cnext:
        if st.button("Next Card →", disabled=(current_card_pos >= total_cards - 1), use_container_width=True):
            st.session_state["card_idx"] = get_next_card_index(current_card_pos, total_cards)
            st.session_state["card_revealed"] = False
            st.rerun()


# ==============================================================================
# TAB 6: ASK MEDHĀ (AI Tutor Q&A)
# ==============================================================================


def render_tab_tutor(session_id: str, video_id: str) -> None:
    """Render lecture-grounded AI Tutor conversation section."""
    lang = st.session_state.get("learning_language", "en")
    lang_name = LANGUAGE_DISPLAY_NAMES.get(lang, lang)
    st.markdown("## 🤖 Ask Medhā — Lecture-Grounded AI Tutor")
    st.caption(f"Ask any question about this lecture. Medhā answers strictly using retrieved lecture moments in **{lang_name}**.")

    chat_history_key = f"chat_history_{session_id}"
    if chat_history_key not in st.session_state:
        st.session_state[chat_history_key] = []

    history = st.session_state[chat_history_key]

    # Question Input
    q_col, btn_col = st.columns([5, 1])
    with q_col:
        question = st.text_input(
            "Ask Medhā",
            placeholder="e.g., Why do we use variables in Python?",
            key=f"tutor_q_{session_id}",
            label_visibility="collapsed",
        )
    with btn_col:
        ask_clicked = st.button("Ask Medhā", type="primary", use_container_width=True, key=f"btn_ask_{session_id}")

    if ask_clicked:
        if not question.strip():
            st.warning("Please enter a question to ask Medhā.")
        else:
            with st.spinner(f"Retrieving lecture moments & formulating grounded answer in {lang_name}..."):
                try:
                    payload = {
                        "session_id": session_id,
                        "question": question.strip(),
                        "top_k": 5,
                        "chat_history": [
                            {"role": turn["role"], "content": turn["content"]}
                            for turn in history[-4:]
                        ],
                        "language": lang,
                    }
                    resp = requests.post(f"{API_BASE_URL}/api/chat", json=payload, timeout=200)
                    if resp.status_code == 200:
                        chat_resp = resp.json()
                        history.append({"role": "user", "content": question.strip()})
                        history.append({
                            "role": "assistant",
                            "content": chat_resp["answer"],
                            "sources": chat_resp.get("sources", []),
                            "primary_topic": chat_resp.get("primary_topic"),
                            "primary_timestamp_range": chat_resp.get("primary_timestamp_range"),
                            "primary_start_time": chat_resp.get("primary_start_time"),
                        })
                        st.session_state[chat_history_key] = history
                        st.rerun()
                    else:
                        detail = resp.json().get("detail", resp.text)
                        st.error(f"AI Tutor error: {detail}")
                except requests.RequestException as exc:
                    st.error(f"Cannot reach AI Tutor backend: {exc}")

    # Conversation Flow
    if history:
        st.markdown("### Conversation")
        for idx, turn in enumerate(history):
            if turn["role"] == "user":
                with st.chat_message("user"):
                    st.write(turn["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(turn["content"])
                    if turn.get("primary_timestamp_range") and turn.get("primary_start_time") is not None:
                        start_sec = int(turn["primary_start_time"])
                        yt_link = get_youtube_jump_url(video_id, start_sec)
                        time_badge = turn["primary_timestamp_range"]
                        topic_label = f" ({turn['primary_topic']})" if turn.get("primary_topic") else ""
                        st.markdown(f"📍 **Related lecture section{topic_label}:** `{time_badge}`")
                        if video_id and not video_id.startswith("local_"):
                            st.link_button(
                                f"⏱ Jump to {time_badge.split('–')[0].strip()}",
                                yt_link,
                                key=f"tutor_jump_{idx}_{start_sec}",
                            )

                    sources = turn.get("sources", [])
                    if sources:
                        with st.expander(f"📚 Sources ({len(sources)} transcript chunks)"):
                            for s in sources:
                                s_start = int(s.get("start_time", 0))
                                s_yt = get_youtube_jump_url(video_id, s_start)
                                st.markdown(f"**Chunk `{s['chunk_id']}`** · [`{s['timestamp']}`]({s_yt}) · Relevance: `{s['score']:.4f}`")
                                st.caption(s["text"])
                                st.divider()


# ==============================================================================
# TAB 7: SOURCES (Verbatim Chunks & Semantic Search)
# ==============================================================================


def render_tab_sources(
    session_id: str,
    video_id: str,
    chunks: list[dict[str, Any]],
    topics: list[dict[str, Any]],
    current_idx: int,
) -> None:
    """Render original lecture transcript chunks and semantic search."""
    st.markdown("## 📜 Lecture Source Transcript & Search")
    st.caption("Verbatim transcript chunks in their original spoken lecture language for complete transparency and verification.")

    # FAISS Semantic Search inside Sources Tab
    with st.expander("🔍 Search Lecture Transcripts (FAISS Semantic Retrieval)", expanded=False):
        search_col1, search_col2 = st.columns([4, 1])
        with search_col1:
            query = st.text_input(
                "Search Query",
                placeholder="e.g., Where does the instructor explain data types?",
                key="tab_faiss_search_query",
            )
        with search_col2:
            top_k = st.number_input("Top Chunks", min_value=1, max_value=20, value=3, step=1, key="tab_faiss_top_k")

        if st.button("Search Transcripts", type="secondary", key="tab_faiss_search_btn"):
            if not query.strip():
                st.warning("Please enter a question or query.")
            else:
                with st.spinner("Searching FAISS vector index..."):
                    try:
                        search_resp = requests.post(
                            f"{API_BASE_URL}/api/search",
                            json={
                                "session_id": session_id,
                                "query": query.strip(),
                                "top_k": int(top_k),
                            },
                            timeout=30,
                        )
                        if search_resp.status_code == 200:
                            st.session_state["search_results"] = search_resp.json()
                        else:
                            detail = search_resp.json().get("detail", search_resp.text)
                            st.error(f"Search error: {detail}")
                    except requests.RequestException as exc:
                        st.error(f"Search request failed: {exc}")

        search_data = st.session_state.get("search_results")
        if search_data:
            st.markdown(f"**Results for:** *\"{search_data['query']}\"*")
            for item in search_data.get("results", []):
                chunk_id = item["chunk_id"]
                timestamp_label = item["timestamp"]
                score = item["score"]
                start_sec = int(item["start_time"])
                yt_link = get_youtube_jump_url(video_id, start_sec)

                with st.container(border=True):
                    head_col1, head_col2 = st.columns([3, 1])
                    with head_col1:
                        st.markdown(
                            f"**{chunk_id}** · `{timestamp_label}` · [Jump to {timestamp_label}]({yt_link})"
                        )
                    with head_col2:
                        st.markdown(f"**Similarity:** `{score:.4f}`")
                    st.write(item["text"])

    st.divider()

    # Topic Chunk Filter
    topic_options = ["(All Topics)"] + [f"{i + 1:02d}. {t['name']}" for i, t in enumerate(topics)]
    filter_choice = st.selectbox("Filter chunks by Topic:", options=topic_options, index=current_idx + 1)

    if filter_choice == "(All Topics)":
        display_chunks = chunks
    else:
        chosen_topic_idx = int(filter_choice.split(".")[0]) - 1
        display_chunks = filter_chunks_for_topic(topics[chosen_topic_idx], chunks)

    st.markdown(f"**Displaying {len(display_chunks)} transcript segments:**")
    for c in display_chunks:
        start_sec = int(c.get("start_time", 0))
        chunk_yt_link = get_youtube_jump_url(video_id, start_sec)
        src_lang = c.get("source_language", "en")
        disp_text = c.get("original_text") or c.get("text", "")
        with st.container(border=True):
            col_id, col_time = st.columns([3, 1])
            with col_id:
                lang_tag = f" · `{src_lang.upper()}`" if src_lang != "en" else ""
                st.markdown(f"**Chunk `{c.get('chunk_id')}`**{lang_tag}")
            with col_time:
                st.markdown(f"[`{c.get('start_label')} – {c.get('end_label')}`]({chunk_yt_link})")
            st.write(disp_text.strip())
            if c.get("translated_text"):
                with st.expander("🌐 English Translation"):
                    st.write(c.get("translated_text", "").strip())



# ==============================================================================
# MAIN APPLICATION CONTROLLER
# ==============================================================================


def main() -> None:
    st.set_page_config(page_title="Medhā — Interactive Learning Dashboard", layout="wide")

    render_header()
    render_sidebar()
    render_video_input()

    result = st.session_state.get("result")
    if not result:
        return

    raw_topics = result.get("topics", [])
    if not raw_topics:
        st.warning("No structured topics found for this lecture.")
        return

    # Ensure chronological order
    topics = get_chronological_topics(raw_topics)
    total_topics = len(topics)

    # Validate and clamp current topic index
    current_idx = st.session_state.get("selected_topic_idx", 0)
    current_idx = max(0, min(current_idx, total_topics - 1))
    st.session_state["selected_topic_idx"] = current_idx

    video_id = result.get("video_id", "")
    session_id = result.get("session_id", "")
    chunks = result.get("chunks", [])

    st.divider()

    # Organized 7-Tab Workspace
    tab_overview, tab_learn, tab_concepts, tab_map, tab_cards, tab_tutor, tab_sources = st.tabs([
        "🏠 Overview",
        "📚 Learn",
        "🧠 Concepts",
        "🕸 Concept Map",
        "📝 Memory Cards",
        "🤖 Ask Medhā",
        "📜 Sources",
    ])

    with tab_overview:
        render_tab_overview(result, topics, current_idx)

    with tab_learn:
        render_tab_learn(session_id, video_id, topics, current_idx)

    with tab_concepts:
        render_tab_concepts(session_id, video_id)

    with tab_map:
        render_tab_concept_map(session_id, video_id)

    with tab_cards:
        render_tab_memory_cards(session_id, video_id)

    with tab_tutor:
        render_tab_tutor(session_id, video_id)

    with tab_sources:
        render_tab_sources(session_id, video_id, chunks, topics, current_idx)


if __name__ == "__main__":
    main()

