from fastapi import APIRouter, HTTPException

from app.schemas import (
    ChatRequest,
    ChatResponse,
    ConceptGraphResponse,
    ConceptIntelligenceCollection,
    GenerateContentRequest,
    HealthResponse,
    LLMStatusResponse,
    MemoryCardCollection,
    ProcessRequest,
    ProcessResponse,
    SearchRequest,
    SearchResponse,
    SessionSummary,
    TopicGroup,
    TopicLearningContent,
)
from app.services.concept_graph_service import ConceptGraphError, get_or_create_concept_graph
from app.services.concept_intelligence_service import (
    ConceptIntelligenceError,
    get_or_create_concept_intelligence,
)
from app.services.content_generator import ContentGenerationError, generate_topic_content
from app.services.llm_client import check_llm_runtime
from app.services.memory_card_service import MemoryCardError, generate_memory_cards
from app.services.process_video import ProcessingError, list_sessions, load_session, process_youtube_url
from app.services.search_service import SearchServiceError, SessionNotFoundError, search_session
from app.services.translation_service import get_localized_concept_graph, translate_educational_content
from app.services.tutor_service import TutorServiceError, ask_tutor

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", phase="8.2-content-intelligence-quality (includes 8.1)")



@router.get("/llm/status", response_model=LLMStatusResponse)
def llm_status() -> LLMStatusResponse:
    info = check_llm_runtime()
    return LLMStatusResponse(**info)


@router.post("/process", response_model=ProcessResponse)
def process_lecture(payload: ProcessRequest) -> ProcessResponse:
    try:
        return process_youtube_url(payload.youtube_url)
    except ProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/sessions", response_model=list[SessionSummary])
def sessions() -> list[SessionSummary]:
    return list_sessions()


@router.get("/sessions/{session_id}", response_model=ProcessResponse)
def session_detail(session_id: str) -> ProcessResponse:
    found = load_session(session_id)
    if found is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return found


@router.get("/sessions/{session_id}/topics", response_model=list[TopicGroup])
def session_topics(session_id: str) -> list[TopicGroup]:
    found = load_session(session_id)
    if found is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return found.topics


@router.post("/search", response_model=SearchResponse)
def search_lecture(payload: SearchRequest) -> SearchResponse:
    try:
        return search_session(
            session_id=payload.session_id,
            query=payload.query,
            top_k=payload.top_k,
        )
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SearchServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc


@router.post("/topics/generate", response_model=TopicLearningContent)
def generate_content_for_topic(payload: GenerateContentRequest) -> TopicLearningContent:
    try:
        lang = payload.language or "en"
        if lang != "en":
            return translate_educational_content(
                session_id=payload.session_id,
                topic_id=payload.topic_id,
                target_language=lang,
            )
        return generate_topic_content(
            session_id=payload.session_id,
            topic_id=payload.topic_id,
            force_regenerate=payload.force_regenerate,
            language=lang,
        )
    except ContentGenerationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate content: {exc}") from exc


@router.post("/topics/{topic_id}/generate", response_model=TopicLearningContent)
def generate_content_by_path(
    topic_id: str, session_id: str, force_regenerate: bool = False, language: str = "en"
) -> TopicLearningContent:
    try:
        if language != "en":
            return translate_educational_content(
                session_id=session_id,
                topic_id=topic_id,
                target_language=language,
            )
        return generate_topic_content(
            session_id=session_id,
            topic_id=topic_id,
            force_regenerate=force_regenerate,
            language=language,
        )
    except ContentGenerationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate content: {exc}") from exc


@router.post("/chat", response_model=ChatResponse)
def chat_with_tutor(payload: ChatRequest) -> ChatResponse:
    """Lecture-grounded AI Tutor Q&A endpoint (Phase 6 & 8.1).
    
    Retrieves relevant lecture transcript chunks via FAISS and invokes Qwen3-8B
    with strict anti-hallucination grounding constraints and optional localized output.
    """
    try:
        return ask_tutor(
            session_id=payload.session_id,
            question=payload.question,
            top_k=payload.top_k,
            chat_history=payload.chat_history,
            language=payload.language or "en",
        )
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TutorServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"AI Tutor error: {exc}") from exc


@router.get("/sessions/{session_id}/concept-graph", response_model=ConceptGraphResponse)
def session_concept_graph(session_id: str, language: str = "en") -> ConceptGraphResponse:
    """Retrieve or build the concept intelligence and knowledge graph for a session (Phase 8 & 8.1)."""
    try:
        if language != "en":
            return get_localized_concept_graph(session_id, target_language=language)
        return get_or_create_concept_graph(session_id)
    except ConceptGraphError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load concept graph: {exc}") from exc


@router.get("/sessions/{session_id}/memory-cards", response_model=MemoryCardCollection)
def session_memory_cards(session_id: str, language: str = "en") -> MemoryCardCollection:
    """Retrieve or generate interactive memory cards for active revision (Phase 8.1)."""
    try:
        return generate_memory_cards(session_id=session_id, language=language)
    except MemoryCardError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load memory cards: {exc}") from exc


@router.get("/sessions/{session_id}/concept-intelligence", response_model=ConceptIntelligenceCollection)
def session_concept_intelligence(session_id: str) -> ConceptIntelligenceCollection:
    """Retrieve structured, quality-scored concept intelligence for a session (Phase 8.2)."""
    try:
        return get_or_create_concept_intelligence(session_id)
    except ConceptIntelligenceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load concept intelligence: {exc}") from exc


