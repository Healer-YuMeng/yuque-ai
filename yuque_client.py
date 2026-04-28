from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import httpx


class YuqueAuthError(RuntimeError):
    pass


class YuqueApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class YuqueSearchHit:
    id: int
    type: str
    title: str
    summary: str
    url: str
    info: str
    target: Dict[str, Any]

    @property
    def book_id(self) -> Optional[int]:
        v = self.target.get("book_id")
        return int(v) if v is not None else None

    @property
    def doc_id(self) -> Optional[int]:
        v = self.target.get("id")
        return int(v) if v is not None else None

    @property
    def slug(self) -> Optional[str]:
        v = self.target.get("slug")
        return str(v) if v is not None else None


@dataclass(frozen=True)
class YuqueDoc:
    id: int
    slug: str
    title: str
    url: str
    format: str
    body: str
    book_id: int


@dataclass(frozen=True)
class YuqueTocNode:
    uuid: str
    type: str
    title: str
    url: str
    doc_id: Optional[int]
    level: int
    parent_uuid: str
    child_uuid: str
    sibling_uuid: str


class YuqueClient:
    def __init__(
        self,
        token: str,
        base_url: str = "https://www.yuque.com/api/v2",
        timeout_s: float = 30.0,
    ) -> None:
        token = (token or "").strip()
        if not token:
            raise YuqueAuthError(
                "缺少语雀 Token。请设置环境变量 YUQUE_TOKEN（请求头 X-Auth-Token）。"
            )

        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self._base_url,
            headers={
                "X-Auth-Token": token,
                "User-Agent": "yuque-rag-demo/0.1",
                "Accept": "application/json",
            },
            timeout=timeout_s,
        )

    def close(self) -> None:
        self._client.close()

    def _normalize_url(self, url: str) -> str:
        url = (url or "").strip()
        if not url:
            return ""
        if url.startswith("http://") or url.startswith("https://"):
            return url
        if url.startswith("/"):
            return f"https://www.yuque.com{url}"
        return url

    def _request(self, method: str, path: str, *, params: Optional[Dict[str, Any]] = None) -> Any:
        try:
            resp = self._client.request(method, path, params=params)
        except httpx.HTTPError as e:
            raise YuqueApiError(f"网络错误：{e}") from e

        if resp.status_code == 401:
            raise YuqueAuthError("语雀鉴权失败（401）。请检查 Token 是否有效。")

        if resp.status_code >= 400:
            raise YuqueApiError(f"语雀 API 错误：HTTP {resp.status_code}，body={resp.text[:500]}")

        try:
            return resp.json()
        except ValueError as e:
            raise YuqueApiError(f"响应不是合法 JSON：{resp.text[:500]}") from e

    def hello(self) -> str:
        data = self._request("GET", "/hello")
        msg = ((data or {}).get("data") or {}).get("message")
        return str(msg) if msg is not None else ""

    def search_docs(
        self,
        *,
        q: str,
        scope: Optional[str] = None,
        page: int = 1,
    ) -> List[YuqueSearchHit]:
        q = (q or "").strip()
        if not q:
            return []

        params: Dict[str, Any] = {"q": q, "type": "doc", "page": page}
        if scope:
            params["scope"] = scope

        payload = self._request("GET", "/search", params=params)
        items = (payload or {}).get("data") or []
        hits: List[YuqueSearchHit] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            hits.append(
                YuqueSearchHit(
                    id=int(it.get("id") or 0),
                    type=str(it.get("type") or ""),
                    title=str(it.get("title") or ""),
                    summary=str(it.get("summary") or ""),
                    url=str(it.get("url") or ""),
                    info=str(it.get("info") or ""),
                    target=dict(it.get("target") or {}),
                )
            )
        return hits

    def get_doc(self, *, book_id: int, id_or_slug: str) -> YuqueDoc:
        payload = self._request("GET", f"/repos/{int(book_id)}/docs/{id_or_slug}")
        data = (payload or {}).get("data") or {}
        return YuqueDoc(
            id=int(data.get("id") or 0),
            slug=str(data.get("slug") or ""),
            title=str(data.get("title") or ""),
            url=self._normalize_url(str(data.get("url") or "")),
            format=str(data.get("format") or ""),
            body=str(data.get("body") or ""),
            book_id=int(data.get("book_id") or book_id),
        )

    def get_book_toc(self, *, book: Union[int, str]) -> List[YuqueTocNode]:
        """
        获取知识库目录（TOC）。
        book 支持：
        - int: book_id（例如 78343688）
        - str: namespace（例如 "group_login/book_slug"）
        """
        key = str(int(book)) if isinstance(book, int) else str(book).strip().strip("/")
        payload = self._request("GET", f"/repos/{key}/toc")
        items = (payload or {}).get("data") or []
        out: List[YuqueTocNode] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            doc_id_raw = it.get("doc_id", None)
            try:
                doc_id = int(doc_id_raw) if doc_id_raw is not None else None
            except Exception:
                doc_id = None
            out.append(
                YuqueTocNode(
                    uuid=str(it.get("uuid") or ""),
                    type=str(it.get("type") or ""),
                    title=str(it.get("title") or ""),
                    url=self._normalize_url(str(it.get("url") or it.get("slug") or "")),
                    doc_id=doc_id,
                    level=int(it.get("level") or it.get("depth") or 0),
                    parent_uuid=str(it.get("parent_uuid") or ""),
                    child_uuid=str(it.get("child_uuid") or ""),
                    sibling_uuid=str(it.get("sibling_uuid") or ""),
                )
            )
        return out

