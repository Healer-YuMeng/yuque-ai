from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.data.yuque_loader import YuqueLoader


@pytest.mark.asyncio
async def test_fetch_self_login_reads_data_login() -> None:
    ld = YuqueLoader(
        token="dummy",
        base_url="https://www.yuque.com/api/v2",
        timeout_s=5.0,
        scope="",
    )
    ld._request = AsyncMock(return_value={"data": {"login": "suesun-yb1bi", "name": "Test"}})  # type: ignore[method-assign]
    try:
        assert await ld.fetch_self_login() == "suesun-yb1bi"
    finally:
        await ld.close()


@pytest.mark.asyncio
async def test_fetch_self_login_falls_back_to_slug_in_data() -> None:
    ld = YuqueLoader(
        token="dummy",
        base_url="https://www.yuque.com/api/v2",
        timeout_s=5.0,
        scope="",
    )
    ld._request = AsyncMock(return_value={"data": {"slug": "suesun-yb1bi"}})  # type: ignore[method-assign]
    try:
        assert await ld.fetch_self_login() == "suesun-yb1bi"
    finally:
        await ld.close()


@pytest.mark.asyncio
async def test_fetch_self_login_empty_when_no_data() -> None:
    ld = YuqueLoader(
        token="dummy",
        base_url="https://www.yuque.com/api/v2",
        timeout_s=5.0,
        scope="",
    )
    ld._request = AsyncMock(return_value={})  # type: ignore[method-assign]
    try:
        assert await ld.fetch_self_login() == ""
    finally:
        await ld.close()
