from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_host: str = "127.0.0.1"
    app_port: int = 8000
    api_base_url: str = "http://127.0.0.1:8000"

    data_dir: Path = PROJECT_ROOT / "data" / "sessions"
    vector_store_dir: Path = PROJECT_ROOT / "data" / "vector_store"

    # Embedding & Retrieval (Phase 2)
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Generative Educational Content & Dual Local LLMs (Phase 8.2)
    llm_base_url: str = "http://127.0.0.1:1234/v1"
    llm_model: str = "qwen/qwen3-8b"
    qwen_model: str = "qwen/qwen3-8b"
    gemma_model: str = "google/gemma-4-E2B-it-qat-q4_0-gguf"
    gemma_model_alias: str = "gemma-4-e2b-it-qat"
    llm_timeout_seconds: float = 180.0
    llm_temperature: float = 0.2
    content_cache_dir: Path = PROJECT_ROOT / "data" / "content_cache"
    concept_graph_dir: Path = PROJECT_ROOT / "data" / "concept_graphs"
    memory_cards_dir: Path = PROJECT_ROOT / "data" / "memory_cards"
    concept_intelligence_dir: Path = PROJECT_ROOT / "data" / "concept_intelligence"
    translations_dir: Path = PROJECT_ROOT / "data" / "translations"


    chunk_target_words: int = 140
    chunk_max_words: int = 220
    chunk_min_words: int = 40
    topic_gap_seconds: float = 8.0


settings = Settings()
