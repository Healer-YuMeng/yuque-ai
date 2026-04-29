from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    doc_id: str
    title: str
    url: str
    text: str
    order: int


class RecursiveTextSplitter:
    def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def split_document(self, *, doc_id: str, title: str, url: str, text: str) -> List[TextChunk]:
        normalized = self._normalize(text)
        if not normalized:
            return []

        chunks: List[TextChunk] = []
        step = self._chunk_size - self._chunk_overlap
        start = 0
        order = 0
        while start < len(normalized):
            end = min(len(normalized), start + self._chunk_size)
            chunk_text = normalized[start:end].strip()
            if chunk_text:
                chunks.append(
                    TextChunk(
                        chunk_id=f"{doc_id}:{order}",
                        doc_id=doc_id,
                        title=title,
                        url=url,
                        text=chunk_text,
                        order=order,
                    )
                )
                order += 1
            start += step
        return chunks

    @staticmethod
    def _normalize(text: str) -> str:
        lines = [line.strip() for line in (text or "").splitlines()]
        return "\n".join(line for line in lines if line)

