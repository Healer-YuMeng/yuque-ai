from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.data.yuque_loader import YuqueLoader


@pytest.mark.asyncio
async def test_list_docs_clamps_limit_to_100() -> None:
    loader = YuqueLoader(token="t", base_url="https://www.yuque.com/api/v2", timeout_s=5.0)
    with patch.object(loader, "_request", new_callable=AsyncMock) as req:
        req.return_value = {"data": []}
        await loader.list_docs(book="org/repo", offset=0, limit=120)
        assert req.call_count == 1
        _args, kwargs = req.call_args
        assert kwargs["params"]["limit"] == 100
        assert kwargs["params"]["offset"] == 0
    await loader.close()
