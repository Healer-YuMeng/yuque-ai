from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from openai import AsyncOpenAI


class Embedder(ABC):
    @abstractmethod
    async def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError

    async def embed_query(self, text: str) -> List[float]:
        vectors = await self.embed_texts([text])
        return vectors[0]


class OpenAIEmbedder(Embedder):
    def __init__(self, *, model: str, api_key: str, base_url: str = "") -> None:
        if not api_key:
            raise ValueError("缺少 embedding API key。")
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
        self._model = model

    async def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(model=self._model, input=list(texts))
        return [list(item.embedding) for item in response.data]


class BGESmallEmbedder(Embedder):
    async def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError("MVP 默认未实现本地 bge-small，请切换到 OpenAI embedding。")

