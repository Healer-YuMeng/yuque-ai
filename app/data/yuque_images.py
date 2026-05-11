from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import List
from urllib.parse import urlparse

# 语雀正文插图常见 CDN / 主站（防 SSRF 白名单）
_ALLOWED_IMAGE_HOST_SUFFIXES: tuple[str, ...] = (
    "yuque.com",
    "yuque.net",
    "nlark.com",
    "larkusercontent.com",
    "alicdn.com",
    "qpic.cn",
)


@dataclass(frozen=True)
class YuqueImageRef:
    """从语雀文档 body 中解析出的一张图。"""

    src: str
    alt: str = ""


_IMG_TAG = re.compile(r"<img\s[^>]*>", re.IGNORECASE)
_SRC_ATTR = re.compile(r"""src\s*=\s*(?P<q>['"])(?P<src>.*?)(?P=q)""", re.IGNORECASE | re.DOTALL)
_DATA_SRC_ATTR = re.compile(r"""data-src\s*=\s*(?P<q>['"])(?P<src>.*?)(?P=q)""", re.IGNORECASE | re.DOTALL)
_ALT_ATTR = re.compile(r"""alt\s*=\s*(?P<q>['"])(?P<alt>.*?)(?P=q)""", re.IGNORECASE | re.DOTALL)
_MD_IMG = re.compile(r"!\[[^\]]*]\(\s*(https?://[^)\s]+)\s*\)")
# Lake / JSON 里裸链（含无后缀、仅 OSS 处理参数等）
_ANY_HTTP = re.compile(r"https?://[^\s\"'<>\\]+", re.IGNORECASE)


def is_allowed_yuque_image_url(url: str) -> bool:
    """仅允许常见语雀/阿里 CDN 主机，防止 asset 代理被用作开放代理。"""
    raw = (url or "").strip()
    if not raw.lower().startswith(("http://", "https://")):
        return False
    try:
        host = (urlparse(raw).hostname or "").lower()
    except ValueError:
        return False
    if not host:
        return False
    return any(host == s or host.endswith("." + s) for s in _ALLOWED_IMAGE_HOST_SUFFIXES)


def _looks_like_image_url(url: str) -> bool:
    """Lake/长链常见形态：未必以 .png 结尾。"""
    u = (url or "").lower()
    if re.search(r"\.(png|jpe?g|gif|webp|svg)(\?|#|$)", url, re.I):
        return True
    if "imageview" in u or "x-oss-process" in u or "x-ocs-process" in u:
        return True
    if "cdn.nlark.com/yuque" in u and re.search(r"/(png|jpe?g|jpeg|jpg|gif|webp)(/|\?|$)", u):
        return True
    if "larkusercontent.com" in u and ("/image/" in u or "image" in u):
        return True
    return False


def extract_image_refs_from_body(body: str) -> List[YuqueImageRef]:
    """
    从语雀文档 HTML / 混排正文中提取插图 URL。
    同时识别 Markdown 图片语法中的 http(s) 链接。
    """
    if not (body or "").strip():
        return []

    # Lake JSON 中常见 Unicode 转义，先归一化便于匹配
    text = body.replace("\\u002F", "/").replace("\\/", "/")

    seen: set[str] = set()
    out: List[YuqueImageRef] = []

    def _push(src: str, alt: str = "") -> None:
        s = (src or "").strip().rstrip(".,;)]}\"'")
        if not s or not is_allowed_yuque_image_url(s):
            return
        if s in seen:
            return
        seen.add(s)
        out.append(YuqueImageRef(src=s, alt=(alt or "").strip()[:200]))

    for m in _IMG_TAG.finditer(text):
        tag = m.group(0) or ""
        sm = _SRC_ATTR.search(tag) or _DATA_SRC_ATTR.search(tag)
        if not sm:
            continue
        src = sm.group("src") or ""
        am = _ALT_ATTR.search(tag)
        alt = am.group("alt") if am else ""
        _push(src, alt)

    for m in _MD_IMG.finditer(text):
        _push(m.group(1), "")

    # Lake / JSON 中裸 URL（含仅有扩展名在路径分段中的 nlark 地址）
    if text.strip().startswith("{") or '"src"' in text or '"url"' in text or "cdn.nlark.com" in text:
        for m in re.finditer(r"https?://[^\s\"'<>\\]+\.(?:png|jpe?g|gif|webp|svg)(?:\?[^\s\"'<>\\]*)?", text, re.I):
            _push(m.group(0), "")

    # 无后缀但明显为图床处理链 / 语雀资源路径
    for m in _ANY_HTTP.finditer(text):
        raw = m.group(0).rstrip(".,;)]}\"'")
        if raw in seen:
            continue
        if not is_allowed_yuque_image_url(raw):
            continue
        if _looks_like_image_url(raw):
            _push(raw, "")

    return out


def encode_image_proxy_token(url: str) -> str:
    """将原始图片 URL 编码为 `/yuque/asset?t=` 查询参数（urlsafe，无 padding）。"""
    raw = (url or "").strip().encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_image_proxy_token(token: str) -> str:
    t = (token or "").strip()
    pad = "=" * (-len(t) % 4)
    return base64.urlsafe_b64decode(t + pad).decode("utf-8")
