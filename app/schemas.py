from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ProcessRequest(BaseModel):
    youtube_url: str = Field(..., min_length=8, description="YouTube watch, short, or youtu.be URL")


class TranscriptSegment(BaseModel):
    text: str
    start: float
    duration: float
    end: float


class Chunk(BaseModel):
    chunk_id: str
    topic: Optional[str] = None
    concepts: list[str] = Field(default_factory=list)
    text: str
    start_timestamp: float
    end_timestamp: float
    start_label: str
    end_label: str
    word_count: int

    # Phase 8.2: Dual-text preservation & source tracking
    original_text: Optional[str] = None
    translated_text: Optional[str] = None
    source_language: str = "en"
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def get_display_text(self, prefer_original: bool = False) -> str:
        """Return original or translated text based on preference, falling back gracefully."""
        if prefer_original and self.original_text:
            return self.original_text
        return self.translated_text or self.text


class TopicLearningContent(BaseModel):
    topic_id: str
    topic_name: str
    timestamp_range: str
    summary: str
    simple_explanation: str
    analogy: str
    key_takeaways: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)


class TopicGroup(BaseModel):
    topic_id: str
    name: str
    start_time: float
    end_time: float
    start_label: str
    end_label: str
    timestamp_range: str
    key_concepts: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    learning_content: Optional[TopicLearningContent] = None


class SupportedLanguage(str, Enum):
    ENGLISH = "en"
    HINDI = "hi"
    KANNADA = "kn"
    TELUGU = "te"
    TAMIL = "ta"
    MALAYALAM = "ml"


LANGUAGE_DISPLAY_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "\u0939\u093f\u0928\u094d\u0926\u0940 (Hindi)",
    "kn": "\u0c95\u0ca8\u0ccd\u0ca8\u0ca1 (Kannada)",
    "te": "\u0c24\u0c46\u0c32\u0c41\u0c17\u0c41 (Telugu)",
    "ta": "\u0ba4\u0bae\u0bbf\u0bb4\u0bcd (Tamil)",
    "ml": "\u0d2e\u0d32\u0d2f\u0d3e\u0d33\u0d02 (Malayalam)",
}


class GenerateContentRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Session ID")
    topic_id: str = Field(..., min_length=1, description="Topic ID (e.g. topic-001)")
    force_regenerate: bool = Field(False, description="Whether to bypass cache and regenerate")
    language: str = Field("en", description="Target learning language code, e.g. 'en', 'hi', 'kn'")


class LLMStatusResponse(BaseModel):
    available: bool
    runtime: str
    base_url: str
    model: str
    details: Optional[str] = None
    qwen_available: bool = False
    gemma_available: bool = False
    qwen_model: Optional[str] = None
    gemma_model: Optional[str] = None
    available_models: list[str] = Field(default_factory=list)



class ProcessResponse(BaseModel):
    session_id: str
    video_id: str
    youtube_url: str
    title: str = Field("Untitled Lecture", description="Human-readable lecture title")
    source_type: str = Field("youtube", description="Lecture source: 'youtube' or 'uploaded'")
    detected_language: str = Field("en", description="Detected language of original lecture")
    duration_seconds: Optional[float] = Field(None, description="Total video duration in seconds")
    formatted_duration: Optional[str] = Field(None, description="Formatted video duration, e.g. 1h 04m")
    language: Optional[str] = None
    is_generated_transcript: bool = False
    raw_segment_count: int
    cleaned_text_preview: str
    chunk_count: int
    chunks: list[Chunk]
    topics: list[TopicGroup] = Field(default_factory=list)


class SessionSummary(BaseModel):
    session_id: str
    video_id: str
    youtube_url: str
    title: str = Field("Untitled Lecture", description="Lecture title")
    source_type: str = Field("youtube", description="Source type")
    formatted_duration: Optional[str] = None
    chunk_count: int


class HealthResponse(BaseModel):
    status: str
    phase: str


class SearchRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Session ID to search within")
    query: str = Field(..., min_length=1, description="Student search query or doubt")
    top_k: int = Field(5, ge=1, le=50, description="Number of most relevant chunks to retrieve")


class SearchResultChunk(BaseModel):
    chunk_id: str
    topic: Optional[str] = None
    concepts: list[str] = Field(default_factory=list)
    text: str
    start_time: float
    end_time: float
    start_timestamp: float
    end_timestamp: float
    timestamp: str
    score: float


class SearchResponse(BaseModel):
    session_id: str
    query: str
    total_results: int
    results: list[SearchResultChunk]


# AI Tutor / Lecture-Grounded RAG (Phase 6)
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatSource(BaseModel):
    chunk_id: str
    topic: Optional[str] = None
    timestamp: str
    start_time: float
    end_time: float
    text: str
    score: float


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Target session ID")
    question: str = Field(..., min_length=1, description="Student question regarding the lecture")
    top_k: int = Field(5, ge=1, le=20, description="Number of relevant chunks to retrieve")
    chat_history: list[ChatMessage] = Field(default_factory=list, description="Recent conversation turns")
    language: str = Field("en", description="Target learning language code, e.g. 'en', 'hi', 'kn'")


class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    sources: list[ChatSource] = Field(default_factory=list)
    primary_topic: Optional[str] = None
    primary_timestamp_range: Optional[str] = None
    primary_start_time: Optional[float] = None


# Concept Intelligence & Knowledge Graph (Phase 8)
class ConceptRelationshipType(str, Enum):
    PREREQUISITE = "prerequisite"
    RELATED_TO = "related_to"
    USED_WITH = "used_with"
    DEPENDS_ON = "depends_on"
    FOLLOWS = "follows"


class ConceptNode(BaseModel):
    concept_id: str = Field(..., description="Unique concept ID, e.g. concept-001")
    name: str = Field(..., description="Canonical concept name, e.g. 'Variables'")
    description: Optional[str] = Field(None, description="Concise educational description")
    topic_id: str = Field(..., description="Associated parent topic ID, e.g. topic-003")
    topic_name: str = Field(..., description="Associated parent topic name")
    source_chunk_ids: list[str] = Field(default_factory=list, description="Traceable chunk IDs")
    start_time: float = Field(..., description="Earliest mention timestamp in seconds")
    end_time: float = Field(..., description="Latest mention timestamp in seconds")
    timestamp_range: str = Field(..., description="Formatted timestamp label, e.g. 05:06 – 08:57")


class ConceptRelationship(BaseModel):
    source: str = Field(..., description="Source concept name")
    relation: ConceptRelationshipType = Field(..., description="Strictly bounded relationship type")
    target: str = Field(..., description="Target concept name")
    evidence: str = Field(..., description="Transcript discourse cue or contextual justification")


class ConceptGraphResponse(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    concepts: list[ConceptNode] = Field(default_factory=list)
    relationships: list[ConceptRelationship] = Field(default_factory=list)
    total_concepts: int = Field(0, description="Total extracted concepts count")
    total_relationships: int = Field(0, description="Total validated relationships count")


# Memory Cards / Revision Cards (Phase 8.1 & 8.2)
class MemoryCard(BaseModel):
    card_id: str = Field(..., description="Unique card ID, e.g. card-001")
    concept_id: str = Field(..., description="Associated concept ID")
    concept_name: str = Field(..., description="Canonical concept name")
    topic_name: Optional[str] = Field(None, description="Associated topic name")
    front_prompt: str = Field(..., description="Revision question/prompt on front of card")
    back_answer: str = Field(..., description="Grounded educational explanation on back")
    timestamp_range: str = Field(..., description="Lecture timestamp range")
    start_time: float = Field(..., description="Start time in seconds")
    source_chunk_ids: list[str] = Field(default_factory=list, description="Source chunk references")
    example: Optional[str] = Field(None, description="Small, relevant code or practical example")
    importance_score: float = Field(1.0, description="Concept importance score")


class MemoryCardCollection(BaseModel):
    session_id: str
    language: str = "en"
    total_cards: int = 0
    cards: list[MemoryCard] = Field(default_factory=list)


# Structured Concept Intelligence Object (Phase 8.2)
class ConceptIntelligence(BaseModel):
    concept: str = Field(..., description="Canonical concept name, e.g. 'Variables'")
    main_idea: str = Field(..., description="2-3 sentence grounded educational explanation")
    key_points: list[str] = Field(default_factory=list, description="Core technical takeaways")
    example: Optional[str] = Field(None, description="Small, relevant, beginner-friendly code or practical example")
    supporting_chunks: list[str] = Field(default_factory=list, description="IDs of supporting lecture chunks")
    timestamp: str = Field(..., description="Timestamp range in lecture, e.g. '05:06 – 08:57'")
    start_time: float = Field(0.0, description="Start timestamp in seconds")
    end_time: float = Field(0.0, description="End timestamp in seconds")
    prerequisites: list[str] = Field(default_factory=list, description="Concepts to understand first")
    related_concepts: list[str] = Field(default_factory=list, description="Connected or co-used concepts")
    importance_score: float = Field(1.0, description="Educational quality/importance score")
    is_structural_section: bool = Field(False, description="Whether this is a structural section like Introduction")
    is_programming_concept: bool = Field(False, description="Whether this involves programming/software code")


class ConceptIntelligenceCollection(BaseModel):
    session_id: str
    total_concepts: int = 0
    concepts: list[ConceptIntelligence] = Field(default_factory=list)


