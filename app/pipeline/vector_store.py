"""FAISS vector store and retrieval service (Phase 2)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np

from app.config import settings
from app.schemas import Chunk, SearchResultChunk

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Raised when vector storage or retrieval fails."""


class FAISSVectorStore:
    """Manages per-session FAISS indices and chunk metadata persistence."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or settings.vector_store_dir
        # In-memory cache: session_id -> (faiss_index, metadata_list)
        self._cache: dict[str, tuple[faiss.Index, list[dict[str, Any]]]] = {}

    def get_session_dir(self, session_id: str) -> Path:
        return self.base_dir / session_id

    def index_exists(self, session_id: str) -> bool:
        session_dir = self.get_session_dir(session_id)
        index_file = session_dir / "index.faiss"
        meta_file = session_dir / "metadata.json"
        return index_file.exists() and meta_file.exists()

    def build_and_save_index(
        self,
        session_id: str,
        video_id: str,
        chunks: list[Chunk],
        embeddings: np.ndarray,
    ) -> None:
        """Create a FAISS IndexFlatIP index, populate it with embeddings, and persist to disk."""
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Chunk count ({len(chunks)}) does not match embedding count ({len(embeddings)})."
            )

        if len(chunks) == 0:
            raise VectorStoreError("Cannot build an index from empty chunks.")

        dimension = embeddings.shape[1]
        # Since embeddings are L2 normalized, IndexFlatIP provides exact cosine similarity
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype(np.float32))

        # Prepare metadata connecting each vector index to original chunk details
        metadata: list[dict[str, Any]] = []
        for chunk in chunks:
            timestamp_str = f"{chunk.start_label} – {chunk.end_label}"
            metadata.append(
                {
                    "session_id": session_id,
                    "video_id": video_id,
                    "chunk_id": chunk.chunk_id,
                    "topic": chunk.topic,
                    "concepts": getattr(chunk, "concepts", []) or [],
                    "text": chunk.text,
                    "start_timestamp": chunk.start_timestamp,
                    "end_timestamp": chunk.end_timestamp,
                    "start_time": chunk.start_timestamp,
                    "end_time": chunk.end_timestamp,
                    "start_label": chunk.start_label,
                    "end_label": chunk.end_label,
                    "timestamp": timestamp_str,
                    "word_count": chunk.word_count,
                }
            )

        # Persist to disk
        session_dir = self.get_session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        index_file = session_dir / "index.faiss"
        meta_file = session_dir / "metadata.json"

        faiss.write_index(index, str(index_file))
        meta_file.write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "video_id": video_id,
                    "count": len(chunks),
                    "dimension": dimension,
                    "chunks": metadata,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        # Update cache
        self._cache[session_id] = (index, metadata)
        logger.info("Persisted FAISS index and metadata for session %s at %s", session_id, session_dir)

    def load_index(self, session_id: str) -> tuple[faiss.Index, list[dict[str, Any]]]:
        """Load the FAISS index and metadata for a session from memory or disk."""
        if session_id in self._cache:
            return self._cache[session_id]

        session_dir = self.get_session_dir(session_id)
        index_file = session_dir / "index.faiss"
        meta_file = session_dir / "metadata.json"

        if not index_file.exists() or not meta_file.exists():
            raise VectorStoreError(f"No FAISS index found for session '{session_id}'.")

        try:
            index = faiss.read_index(str(index_file))
            meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
            chunks_meta = meta_data.get("chunks", [])
            self._cache[session_id] = (index, chunks_meta)
            return index, chunks_meta
        except Exception as exc:
            raise VectorStoreError(f"Failed to load FAISS index for '{session_id}': {exc}") from exc

    def search(
        self,
        session_id: str,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[SearchResultChunk]:
        """Search the session's FAISS index for the most relevant transcript chunks."""
        index, metadata = self.load_index(session_id)

        if not metadata:
            return []

        # Bound top_k by the number of indexed elements
        effective_k = min(max(1, top_k), len(metadata))

        # Ensure query vector is shape (1, dim) and float32
        query_mat = query_vector.reshape(1, -1).astype(np.float32)

        scores, indices = index.search(query_mat, effective_k)

        results: list[SearchResultChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(metadata):
                continue
            item = metadata[idx]
            results.append(
                SearchResultChunk(
                    chunk_id=item["chunk_id"],
                    topic=item.get("topic"),
                    concepts=item.get("concepts", []),
                    text=item["text"],
                    start_time=item["start_time"],
                    end_time=item["end_time"],
                    start_timestamp=item["start_timestamp"],
                    end_timestamp=item["end_timestamp"],
                    timestamp=item["timestamp"],
                    score=round(float(score), 4),
                )
            )

        return results


_vector_store_instance: Optional[FAISSVectorStore] = None


def get_vector_store() -> FAISSVectorStore:
    """Singleton provider for FAISSVectorStore."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = FAISSVectorStore()
    return _vector_store_instance
