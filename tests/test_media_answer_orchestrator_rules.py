from app.schemas.chat import ChatMediaBundle, MediaItem
from app.service.media_answer_orchestrator import MediaAnswerOrchestrator
from app.conversation.lead_nudge_policy import LeadNudgePolicy


class _FakeMCP:
    enabled = True


class _FakeGenerator:
    async def generate(self, *, question: str, contexts, sources, visitor_sales: bool = False):
        return "ok"


class _FakeLeadRepo:
    async def has_lead_for_session(self, *, session_id: str) -> bool:
        return False


class _FakeSessionRepo:
    async def list_recent_messages(self, *, session_id: str, limit: int):
        return []


def _mk_orch() -> MediaAnswerOrchestrator:
    return MediaAnswerOrchestrator(
        mcp_client=_FakeMCP(),
        generator=_FakeGenerator(),
        lead_policy=LeadNudgePolicy(rounds_threshold=5, stay_seconds_threshold=120),
        lead_capture_repository=_FakeLeadRepo(),
        chat_session_repository=_FakeSessionRepo(),
        max_images=3,
        max_videos=1,
        max_docs=6,
        image_rerank_mode="text_rerank",
    )


def test_build_search_queries_adds_core_keywords() -> None:
    queries = MediaAnswerOrchestrator._build_search_queries("请介绍平台核心功能，并给我相关图片或视频")
    assert len(queries) >= 2
    # 含精简后的关键词查询，避免只用整句检索
    assert any("介绍平台核心功能" in q or "介绍" in q for q in queries)


def test_generation_question_without_media_avoids_forced_media_text() -> None:
    q = MediaAnswerOrchestrator._build_generation_question(
        question="介绍核心功能",
        media=ChatMediaBundle(images=[], videos=[]),
        intent="video",
        nudge_text="",
        skill_instruction="",
    )
    assert "不要主动强调" in q
    assert "可用视频数量" not in q


def test_generation_question_with_media_uses_intent_hint() -> None:
    q = MediaAnswerOrchestrator._build_generation_question(
        question="有没有演示视频",
        media=ChatMediaBundle(images=[MediaItem(url="https://a.png")], videos=[MediaItem(url="https://a.mp4")]),
        intent="video",
        nudge_text="",
        skill_instruction="",
    )
    assert "用户偏好视频" in q


def test_normalize_doc_url_keeps_absolute() -> None:
    got = MediaAnswerOrchestrator._normalize_doc_url("https://www.yuque.com/a/b/c")
    assert got == "https://www.yuque.com/a/b/c"


def test_normalize_doc_url_builds_from_slug_path() -> None:
    got = MediaAnswerOrchestrator._normalize_doc_url("tuv9fxvc39knpt7c")
    assert got == "https://www.yuque.com/tuv9fxvc39knpt7c"


def test_text_rerank_adds_context_relevance_score() -> None:
    orch = MediaAnswerOrchestrator(
        mcp_client=_FakeMCP(),
        generator=_FakeGenerator(),
        lead_policy=LeadNudgePolicy(rounds_threshold=5, stay_seconds_threshold=120),
        lead_capture_repository=_FakeLeadRepo(),
        chat_session_repository=_FakeSessionRepo(),
        max_images=3,
        max_videos=1,
        max_docs=6,
        image_rerank_mode="text_rerank",
    )
    item = MediaItem(url="https://a.png", title="", doc_title="平台介绍", doc_id="1")
    s_with_ctx = orch._media_score(
        item=item,
        keywords=["平台"],
        context="这张图展示平台核心功能模块",
        question="平台核心功能",
        media_kind="image",
        intent="text",
    )
    s_without_ctx = orch._media_score(
        item=item,
        keywords=["平台"],
        context="这是一段无关描述",
        question="平台核心功能",
        media_kind="image",
        intent="text",
    )
    assert s_with_ctx > s_without_ctx


def test_pick_guide_titles_prefers_directory_modules() -> None:
    titles = ["课程产品矩阵", "使用指南-教师端", "平台介绍", "优秀案例库"]
    picked = MediaAnswerOrchestrator._pick_guide_titles(titles)
    assert picked[0] == "平台介绍"
    assert any("使用指南" in x for x in picked)
    assert any("案例" in x for x in picked)


def test_has_reliable_doc_evidence_requires_body_length() -> None:
    # 使用具备属性的最小对象模拟 _DocContext
    class _D:
        def __init__(self, body: str) -> None:
            self.body = body

    assert MediaAnswerOrchestrator._has_reliable_doc_evidence([_D("短内容")]) is False
    assert MediaAnswerOrchestrator._has_reliable_doc_evidence([_D("a" * 100)]) is True


def test_collect_media_video_intent_returns_only_videos() -> None:
    orch = _mk_orch()
    docs = [
        type(
            "_D",
            (),
            {
                "title": "IDEAS-PBL",
                "snippet": "",
                "body": "![图1](https://cdn.example.com/a.png)\nhttps://cdn.example.com/demo.mp4",
                "doc_id": "1",
                "url": "",
            },
        )()
    ]
    media = orch._collect_media(docs, question="给我视频", intent="video")
    assert len(media.videos) == 1
    assert media.images == []


def test_collect_media_image_intent_returns_only_images() -> None:
    orch = _mk_orch()
    docs = [
        type(
            "_D",
            (),
            {
                "title": "IDEAS-PBL",
                "snippet": "",
                "body": "![图1](https://cdn.example.com/a.png)\nhttps://cdn.example.com/demo.mp4",
                "doc_id": "1",
                "url": "",
            },
        )()
    ]
    media = orch._collect_media(docs, question="给我图片", intent="image")
    assert len(media.images) == 1
    assert media.videos == []
