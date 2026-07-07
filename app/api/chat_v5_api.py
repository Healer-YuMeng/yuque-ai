from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

from app.api.chat_api import _chat_stream_error_message, get_qa_service
from app.core.config import settings
from app.core.logger import get_logger
from app.data.yuque_loader import YuqueLoaderError
from app.rag.friend_v5_generator import FriendV5WebSourcesMissing
from app.rag.generator import GeneratorConfigError
from app.schemas.chat_v5 import ChatV5CapabilitiesResponse, ChatV5Request
from app.service.qa_service import QAService

logger = get_logger(__name__)
router = APIRouter()


@router.get("/chat/v5/capabilities", response_model=ChatV5CapabilitiesResponse)
async def chat_v5_capabilities(qa_service: QAService = Depends(get_qa_service)) -> ChatV5CapabilitiesResponse:
    if hasattr(qa_service, "chat_v5_capabilities"):
        data = qa_service.chat_v5_capabilities()
    else:
        data = {
            "enabled": bool(settings.chat_v5_enabled),
            "model": settings.chat_v5_model,
            "require_web_sources": bool(settings.chat_v5_require_web_sources),
        }
    return ChatV5CapabilitiesResponse(**data)


@router.post("/chat/v5/stream")
async def chat_v5_stream(request: ChatV5Request, qa_service: QAService = Depends(get_qa_service)) -> StreamingResponse:
    async def event_generator():
        if not settings.chat_v5_enabled:
            payload = {"message": "V5 链路未开启，请先设置 CHAT_V5_ENABLED=true。"}
            yield _sse("error", payload)
            return
        try:
            async for item in qa_service.chat_v5_stream(
                request.question,
                model=request.model,
                owner=request.owner,
                token_profile=request.token_profile,
                chat_mode=request.chat_mode,
                session_id=request.session_id,
                scene=request.scene,
                trigger_type=request.trigger_type,
            ):
                yield _sse(str(item["event"]), item["data"])
        except (GeneratorConfigError, YuqueLoaderError, FriendV5WebSourcesMissing) as exc:
            yield _sse("error", {"message": str(exc)})
        except (APIConnectionError, APITimeoutError, APIError, RateLimitError) as exc:
            logger.warning("chat_v5_stream_llm_client_error err=%s", exc)
            yield _sse("error", {"message": _chat_stream_error_message(exc)})
        except Exception:
            logger.exception("chat_v5_stream_unhandled")
            yield _sse("error", {"message": "这次回复暂时没有成功发出，请稍后重试。"})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"
