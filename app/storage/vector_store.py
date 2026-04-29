from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    faiss = None


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: str
    doc_id: str
    title: str
    url: str
    text: str
    order: int


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: StoredChunk
    score: float


class VectorStore:
    def __init__(self, *, vector_dir: Path) -> None:
        self._vector_dir = vector_dir
        self._index_path = vector_dir / "index.faiss"
        self._matrix_path = vector_dir / "index.npy"
        self._meta_path = vector_dir / "chunks.json"
        self._chunks: List[StoredChunk] = []
        self._index = None
        self._matrix: np.ndarray | None = None
        self._dimension: int | None = None
        self._load()

    def rebuild(self, *, chunks: Sequence[StoredChunk], embeddings: Sequence[Sequence[float]]) -> None:
        vectors = np.array(embeddings, dtype="float32")
        if vectors.size == 0:
            self._chunks = []
            self._index = None
            self._dimension = None
            self._persist()
            return

        self._dimension = int(vectors.shape[1])
        vectors = self._normalize(vectors)
        if faiss is not None:
            self._index = faiss.IndexFlatIP(self._dimension)
            self._index.add(vectors)
        self._matrix = vectors
        self._chunks = list(chunks)
        self._persist()

    def search(self, query_embedding: Sequence[float], top_k: int) -> List[RetrievedChunk]:
        if self._index is None or not self._chunks:
            if self._matrix is None:
                return []
        query = np.array([query_embedding], dtype="float32")
        query = self._normalize(query)
        if self._index is not None and faiss is not None:
            scores, indices = self._index.search(query, top_k)
        else:
            similarities = np.dot(self._matrix, query[0])  # type: ignore[arg-type]
            ranked = np.argsort(-similarities)[:top_k]
            scores = np.array([[float(similarities[i]) for i in ranked]], dtype="float32")
            indices = np.array([ranked], dtype="int64")
        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            results.append(RetrievedChunk(chunk=self._chunks[idx], score=float(score)))
        return results

    def chunk_count(self) -> int:
        return len(self._chunks)

    def _persist(self) -> None:
        self._vector_dir.mkdir(parents=True, exist_ok=True)
        if self._index is None and self._matrix is None:
            if self._index_path.exists():
                self._index_path.unlink()
            if self._matrix_path.exists():
                self._matrix_path.unlink()
            self._meta_path.write_text("[]", encoding="utf-8")
            return
        if self._index is not None and faiss is not None:
            faiss.write_index(self._index, str(self._index_path))
            if self._matrix_path.exists():
                self._matrix_path.unlink()
        elif self._index_path.exists():
            self._index_path.unlink()
        if self._matrix is not None and faiss is None:
            np.save(self._matrix_path, self._matrix)
        payload = [asdict(chunk) for chunk in self._chunks]
        self._meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if faiss is not None and self._index_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            self._dimension = self._index.d
        elif self._matrix_path.exists():
            self._matrix = np.load(self._matrix_path)
            self._dimension = int(self._matrix.shape[1]) if self._matrix.size else None
        if self._meta_path.exists():
            items = json.loads(self._meta_path.read_text(encoding="utf-8"))
            self._chunks = [StoredChunk(**item) for item in items]

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

