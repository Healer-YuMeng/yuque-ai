from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


class YuqueLoaderError(RuntimeError):
    pass


def build_yuque_doc_url(raw: str, *, scope: str = "") -> str:
    """把语雀返回的链接/slug 规整成可点击的绝对地址。

    语雀 toc/doc 接口经常只返回文档 slug（如 ``wbop09b3zygg9erg``）或 ``/owner/repo/slug``
    形式的相对路径；仅有 slug 时需要补上知识库作用域 ``owner/repo`` 才能正常打开。
    """
    value = (raw or "").strip()
    if not value:
        return ""
    if value.startswith(("http://", "https://")):
        return value
    value = value.lstrip("/")
    if not value:
        return ""
    sc = (scope or "").strip().strip("/")
    # 形如 owner/repo/slug 的相对路径：直接拼域名
    if "/" in value:
        return f"https://www.yuque.com/{value}"
    # 仅有文档 slug：补上 owner/repo 作用域
    if sc:
        return f"https://www.yuque.com/{sc}/{value}"
    return f"https://www.yuque.com/{value}"


@dataclass(frozen=True)
class YuqueDocument:
    doc_id: str
    title: str
    url: str
    body: str


@dataclass(frozen=True)
class YuqueSearchHit:
    title: str
    url: str
    summary: str
    book_id: Optional[int]
    doc_id: Optional[int]
    slug: Optional[str]


@dataclass(frozen=True)
class YuqueTocNode:
    uuid: str
    type: str
    title: str
    url: str
    doc_id: Optional[int]
    level: int
    parent_uuid: str


@dataclass(frozen=True)
class YuqueDocMeta:
    id: int
    slug: str
    title: str
    url: str
    updated_at: str


class YuqueLoader:
    def __init__(self, *, token: str, base_url: str, timeout_s: float, scope: str = "") -> None:
        self._scope = scope.strip()
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._token = (token or "").strip()
        self._client: Optional[httpx.AsyncClient] = None
        if self._token:
            self._client = self._build_client()

    @property
    def scope(self) -> str:
        """当前知识库作用域，形如 login/repo。"""
        return self._scope

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    def _build_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout_s,
            headers={
                "X-Auth-Token": self._token,
                "User-Agent": "enterprise-rag-mvp/0.1",
                "Accept": "application/json",
            },
        )

    def _require_client(self) -> httpx.AsyncClient:
        if not self._token:
            raise YuqueLoaderError("缺少 YUQUE_TOKEN，无法访问语雀。")
        if self._client is None:
            self._client = self._build_client()
        return self._client

    async def _request(self, method: str, path: str, *, params: Optional[Dict[str, Any]] = None) -> Any:
        client = self._require_client()
        try:
            response = await client.request(method, path, params=params)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise YuqueLoaderError(f"语雀接口错误: HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise YuqueLoaderError(f"语雀网络错误: {exc}") from exc

        try:
            return response.json()
        except ValueError as exc:
            raise YuqueLoaderError("语雀返回了非法 JSON。") from exc

    async def search_docs(self, query: str, *, page: int = 1) -> List[YuqueSearchHit]:
        if not query.strip():
            return []
        params: Dict[str, Any] = {"q": query.strip(), "type": "doc", "page": page}
        if self._scope:
            params["scope"] = self._scope
        payload = await self._request("GET", "/search", params=params)
        items = (payload or {}).get("data") or []
        hits: List[YuqueSearchHit] = []
        for item in items:
            target = dict(item.get("target") or {})
            hits.append(
                YuqueSearchHit(
                    title=str(item.get("title") or ""),
                    url=self._normalize_url(str(item.get("url") or "")),
                    summary=str(item.get("summary") or ""),
                    book_id=int(target["book_id"]) if target.get("book_id") is not None else None,
                    doc_id=int(target["id"]) if target.get("id") is not None else None,
                    slug=str(target.get("slug")) if target.get("slug") is not None else None,
                )
            )
        return hits

    async def get_doc(self, *, book: str | int, id_or_slug: str) -> YuqueDocument:
        key = str(int(book)) if isinstance(book, int) else str(book).strip().strip("/")
        payload = await self._request("GET", f"/repos/{key}/docs/{id_or_slug}")
        data = (payload or {}).get("data") or {}
        return YuqueDocument(
            doc_id=str(data.get("id") or id_or_slug),
            title=str(data.get("title") or ""),
            url=self._normalize_url(str(data.get("url") or data.get("slug") or "")),
            body=str(data.get("body") or ""),
        )

    async def fetch_documents_for_bootstrap(self, *, query: str, limit: int = 10) -> List[YuqueDocument]:
        hits = await self.search_docs(query)
        documents: List[YuqueDocument] = []
        for hit in hits[:limit]:
            if hit.book_id is None:
                continue
            identifier = str(hit.doc_id or hit.slug or "")
            if not identifier:
                continue
            documents.append(await self.get_doc(book=hit.book_id, id_or_slug=identifier))
        return documents

    async def get_book_toc(self, *, book: str | int) -> List[YuqueTocNode]:
        key = str(int(book)) if isinstance(book, int) else str(book).strip().strip("/")
        if not key:
            return []
        payload = await self._request("GET", f"/repos/{key}/toc")
        items = (payload or {}).get("data") or []
        nodes: List[YuqueTocNode] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            raw_doc_id = item.get("doc_id")
            try:
                doc_id = int(raw_doc_id) if raw_doc_id is not None else None
            except Exception:
                doc_id = None
            nodes.append(
                YuqueTocNode(
                    uuid=str(item.get("uuid") or ""),
                    type=str(item.get("type") or ""),
                    title=str(item.get("title") or ""),
                    url=self._normalize_url(str(item.get("url") or item.get("slug") or "")),
                    doc_id=doc_id,
                    level=int(item.get("level") or item.get("depth") or 0),
                    parent_uuid=str(item.get("parent_uuid") or ""),
                )
            )
        return nodes

    async def fetch_self_login(self) -> str:
        """GET /user：当前 Token 对应的语雀用户 login（与知识库路径里的 login 可能不同）。"""
        payload = await self._request("GET", "/user")
        if not isinstance(payload, dict):
            return ""
        data = payload.get("data")
        if isinstance(data, dict):
            for key in ("login", "slug"):
                v = str(data.get(key) or "").strip()
                if v:
                    return v
        for key in ("login", "slug"):
            v = str(payload.get(key) or "").strip()
            if v:
                return v
        return ""

    async def list_docs(self, *, book: str | int, offset: int = 0, limit: int = 50) -> List[YuqueDocMeta]:
        key = str(int(book)) if isinstance(book, int) else str(book).strip().strip("/")
        if not key:
            return []
        # 语雀 OpenAPI：limit 超过 100 会返回 422
        safe_limit = max(1, min(int(limit), 100))
        safe_offset = max(0, int(offset))
        payload = await self._request(
            "GET", f"/repos/{key}/docs", params={"offset": safe_offset, "limit": safe_limit}
        )
        items = (payload or {}).get("data") or []
        docs: List[YuqueDocMeta] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            docs.append(
                YuqueDocMeta(
                    id=int(item.get("id") or 0),
                    slug=str(item.get("slug") or ""),
                    title=str(item.get("title") or ""),
                    url=self._normalize_url(str(item.get("url") or item.get("slug") or "")),
                    updated_at=str(item.get("updated_at") or ""),
                )
            )
        return docs

    def _normalize_url(self, url: str) -> str:
        return build_yuque_doc_url(url, scope=self._scope)

