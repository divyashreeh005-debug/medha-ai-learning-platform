"""Embedding service using sentence-transformers (Phase 2)."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_embedding_service_instance: Optional["EmbeddingService"] = None


class EmbeddingService:
    """Manages loading and inference for sentence-transformers embedding models.

    Avoids re-loading the model on every request by caching the model instance.
    Generates normalized embeddings suitable for cosine similarity via inner product.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model_name
        self._model = None

    @property
    def model(self):
        """Lazy-load the sentence transformer model upon first use."""
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def embedding_dim(self) -> int:
        if hasattr(self.model, "get_embedding_dimension"):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode a list of text strings into L2-normalized numpy vectors."""
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single query string into an L2-normalized 1D numpy vector."""
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        vector = self.model.encode(
            [query.strip()],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        return vector.astype(np.float32)


def get_embedding_service(model_name: str | None = None) -> EmbeddingService:
    """Singleton provider for EmbeddingService to prevent reloading models."""
    global _embedding_service_instance
    if _embedding_service_instance is None or (
        model_name and _embedding_service_instance.model_name != model_name
    ):
        _embedding_service_instance = EmbeddingService(model_name=model_name)
    return _embedding_service_instance
