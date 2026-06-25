from __future__ import annotations

import pytest

from app.conversation.user_info_extractor import UserInfoStructuredExtractor


@pytest.mark.asyncio
async def test_user_info_structured_extractor_fallback_splits_sentence_fields() -> None:
    extractor = UserInfoStructuredExtractor(client=None)

    info = await extractor.extract("我的名字是 zjt，我在腾讯上班，手机号 13813655304，邮箱是 ZJT@Test.COM")

    assert info.display_name == "zjt"
    assert info.org_name == "腾讯"
    assert info.contact == "13813655304"
    assert info.email == "zjt@test.com"


@pytest.mark.asyncio
async def test_user_info_structured_extractor_keeps_partial_email_for_prefill() -> None:
    extractor = UserInfoStructuredExtractor(client=None)

    info = await extractor.extract("我是王校长，邮箱是 ziy")

    assert info.display_name == "王校长"
    assert info.email == "ziy"


@pytest.mark.asyncio
async def test_user_info_structured_extractor_strips_employment_suffix() -> None:
    extractor = UserInfoStructuredExtractor(client=None)

    info = await extractor.extract("我在有为中学就职")

    assert info.org_name == "有为中学"
