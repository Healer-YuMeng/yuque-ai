from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.chat import ChatRequest, SelectedYuqueDocRef


def test_chat_request_selected_yuque_docs_roundtrip() -> None:
    body = ChatRequest(
        question="你好",
        selected_yuque_docs=[SelectedYuqueDocRef(doc_id=1001, slug="my-doc", title="标题")],
    )
    dumped = body.model_dump()
    assert dumped["selected_yuque_docs"][0]["doc_id"] == 1001
    assert dumped["selected_yuque_docs"][0]["slug"] == "my-doc"


def test_selected_yuque_doc_rejects_non_positive_id() -> None:
    with pytest.raises(ValidationError):
        SelectedYuqueDocRef(doc_id=0, title="x")
