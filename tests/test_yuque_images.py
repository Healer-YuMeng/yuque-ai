from __future__ import annotations

from app.data.yuque_images import (
    decode_image_proxy_token,
    encode_image_proxy_token,
    extract_image_refs_from_body,
    is_allowed_yuque_image_url,
)


def test_is_allowed_yuque_image_url_accepts_nlark() -> None:
    assert is_allowed_yuque_image_url("https://cdn.nlark.com/yuque/0/2024/png/1/2.png")


def test_is_allowed_yuque_image_url_rejects_random_host() -> None:
    assert not is_allowed_yuque_image_url("https://evil.example.com/a.png")


def test_extract_image_refs_from_html_img() -> None:
    body = '<p>x</p><img src="https://cdn.nlark.com/yuque/0/a.png" alt="人工智能素养" />'
    refs = extract_image_refs_from_body(body)
    assert len(refs) == 1
    assert refs[0].src.endswith("a.png")
    assert "人工智能" in refs[0].alt


def test_extract_image_refs_markdown() -> None:
    body = "正文\n![](https://cdn.nlark.com/yuque/0/b.jpg)\n"
    refs = extract_image_refs_from_body(body)
    assert len(refs) == 1
    assert refs[0].src.endswith("b.jpg")


def test_encode_decode_image_proxy_token_roundtrip() -> None:
    u = "https://cdn.nlark.com/yuque/0/x.png?token=abc"
    t = encode_image_proxy_token(u)
    assert decode_image_proxy_token(t) == u


def test_extract_dedupes_same_src() -> None:
    u = "https://cdn.nlark.com/yuque/0/c.png"
    body = f'<img src="{u}"/><img src="{u}"/>'
    assert len(extract_image_refs_from_body(body)) == 1


def test_extract_img_prefers_data_src_when_present() -> None:
    body = '<img data-src="https://cdn.nlark.com/yuque/0/ds.png" alt="lake" />'
    refs = extract_image_refs_from_body(body)
    assert len(refs) == 1
    assert refs[0].src.endswith("ds.png")
    assert refs[0].alt == "lake"


def test_extract_nlark_jpeg_segment_without_file_suffix() -> None:
    u = "https://cdn.nlark.com/yuque/0/2024/jpeg/abc123def456"
    body = f'{{"type":"image","url":"{u}"}}'
    refs = extract_image_refs_from_body(body)
    assert len(refs) == 1
    assert refs[0].src == u


def test_extract_unescapes_lake_json_slashes() -> None:
    u = "https://cdn.nlark.com/yuque/0/2024/png/x/y.png"
    # 语雀 Lake 常见：JSON 内用 \u002f 表示 /
    body = '{"src":"https:\u002f\u002fcdn.nlark.com\u002fyuque\u002f0\u002f2024\u002fpng\u002fx\u002fy.png"}'
    refs = extract_image_refs_from_body(body)
    assert len(refs) == 1
    assert refs[0].src == u
