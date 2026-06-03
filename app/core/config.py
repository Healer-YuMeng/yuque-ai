from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or "").strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name, str(default))
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env(name, str(default))
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, "true" if default else "false").lower()
    return raw in ("1", "true", "yes", "y", "on")


@dataclass(frozen=True)
class Settings:
    app_name: str = _env("APP_NAME", "Enterprise RAG MVP")
    app_env: str = _env("APP_ENV", "dev")
    log_level: str = _env("LOG_LEVEL", "INFO")

    host: str = _env("HOST", "0.0.0.0")
    port: int = _env_int("PORT", 8000)

    data_dir: Path = BASE_DIR / "data_runtime"
    vector_dir: Path = BASE_DIR / "data_runtime" / "vector_store"
    sqlite_path: Path = BASE_DIR / "data_runtime" / "rag_mvp.db"

    yuque_token: str = _env("YUQUE_TOKEN")
    yuque_token_secondary: str = _env("YUQUE_TOKEN_SECONDARY", "")
    yuque_scope: str = _env("YUQUE_SCOPE")
    yuque_scope_secondary: str = _env("YUQUE_SCOPE_SECONDARY", "")
    yuque_base_url: str = _env("YUQUE_BASE_URL", "https://www.yuque.com/api/v2")
    yuque_timeout_s: float = _env_float("YUQUE_TIMEOUT_S", 30.0)

    embedding_provider: str = _env("EMBEDDING_PROVIDER", "openai")
    embedding_model: str = _env("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_base_url: str = _env("EMBEDDING_BASE_URL", _env("OPENAI_BASE_URL"))
    embedding_api_key: str = _env(
        "EMBEDDING_API_KEY",
        _env("OPENAI_API_KEY", _env("DEEPSEEK_API_KEY")),
    )

    # 为了支持前端按模型动态选择并给出“缺少对应 API Key”的明确错误
    # deepseek key 兼容历史写法：
    # - 新写法：DEEPSEEK_API_KEY
    # - 旧写法：只配置 LLM_API_KEY（此时深度模型可复用）
    # openai key 仍保持严格：gpt-* 模型必须配置 OPENAI_API_KEY
    openai_api_key: str = _env("OPENAI_API_KEY")
    deepseek_api_key: str = _env("DEEPSEEK_API_KEY") or _env("LLM_API_KEY")
    dashscope_api_key: str = _env("DASHSCOPE_API_KEY")
    dashscope_base_url: str = _env("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    llm_provider: str = _env("LLM_PROVIDER", "openai")
    llm_model: str = _env("LLM_MODEL", "qwen3.6-plus")
    openai_base_url: str = _env("OPENAI_BASE_URL")
    deepseek_base_url: str = _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    llm_base_url: str = _env(
        "LLM_BASE_URL",
        _env(
            "DASHSCOPE_BASE_URL",
            _env("DEEPSEEK_BASE_URL", _env("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")),
        ),
    )
    llm_api_key: str = _env(
        "LLM_API_KEY",
        _env("DASHSCOPE_API_KEY", _env("DEEPSEEK_API_KEY", _env("OPENAI_API_KEY"))),
    )

    top_k: int = _env_int("TOP_K", 4)
    chunk_size: int = _env_int("CHUNK_SIZE", 800)
    chunk_overlap: int = _env_int("CHUNK_OVERLAP", 120)
    retrieval_score_threshold: float = _env_float("RETRIEVAL_SCORE_THRESHOLD", 0.35)

    mcp_server_command: str = _env("YUQUE_MCP_COMMAND")
    mcp_server_args: str = _env("YUQUE_MCP_ARGS")
    # 语雀官方 MCP 注册名为 yuque_*；旧版示例曾用 search/get_doc，会导致 Unknown tool
    mcp_search_tool: str = _env("YUQUE_MCP_SEARCH_TOOL", "yuque_search")
    mcp_get_doc_tool: str = _env("YUQUE_MCP_GET_DOC_TOOL", "yuque_get_doc")
    mcp_timeout_s: float = _env_float("YUQUE_MCP_TIMEOUT_S", 20.0)
    force_mcp_fallback: bool = _env_bool("FORCE_MCP_FALLBACK", False)
    auto_mcp_tool_router: bool = _env_bool("AUTO_MCP_TOOL_ROUTER", False)
    intent_llm_enabled: bool = _env_bool("INTENT_LLM_ENABLED", False)
    intent_llm_model: str = _env("INTENT_LLM_MODEL", _env("LLM_MODEL", "qwen3.6-plus"))

    # 元问题：正则未命中时，可用一次 LLM 判断是否「只问助手自身」，减少无限扩写关键词（见 README / doc）
    assistant_meta_llm_router: bool = _env_bool("ASSISTANT_META_LLM_ROUTER", False)
    assistant_meta_router_max_chars: int = _env_int("ASSISTANT_META_ROUTER_MAX_CHARS", 120)
    assistant_meta_router_model: str = _env(
        "ASSISTANT_META_ROUTER_MODEL",
        _env("INTENT_LLM_MODEL", _env("LLM_MODEL", "qwen3.6-plus")),
    )

    web_dir: Path = BASE_DIR / "web"
    frontend_dir: Path = BASE_DIR / "frontend"
    frontend_dist_dir: Path = BASE_DIR / "frontend" / "dist"
    frontend_public_port: int = _env_int("FRONTEND_PORT", 18080)
    expose_source_urls: bool = _env_bool("EXPOSE_SOURCE_URLS", False)

    # 语雀插图：不启用多模态时，仍可将命中文档内的图片以 Markdown（/yuque/asset 代理）追加进上下文，供主模型原样插入回答
    doc_images_markdown_in_context: bool = _env_bool("DOC_IMAGES_MARKDOWN_IN_CONTEXT", True)
    # 为 true 时：对每篇命中再拉全文抽图（易带入「整篇所有插图」）；默认 false，仅从检索片段 contexts 抽图
    doc_images_full_document_fallback: bool = _env_bool("DOC_IMAGES_FULL_DOCUMENT_FALLBACK", False)

    # 语雀多媒体：独立多模态识读（OpenAI 兼容）+ 主模型写回答；可接阿里百炼视觉模型
    vision_enabled: bool = _env_bool("VISION_ENABLED", False)
    vision_model: str = _env("VISION_MODEL", "qwen3.6-plus")
    vision_max_images: int = _env_int("VISION_MAX_IMAGES", 4)
    vision_max_videos: int = _env_int("VISION_MAX_VIDEOS", 1)
    vision_video_fps: int = _env_int("VISION_VIDEO_FPS", 2)
    vision_max_bytes: int = _env_int("VISION_MAX_BYTES", 4_000_000)
    vision_base_url: str = _env("VISION_BASE_URL", _env("DASHSCOPE_BASE_URL", _env("OPENAI_BASE_URL")))
    vision_api_key: str = _env("VISION_API_KEY", _env("DASHSCOPE_API_KEY", _env("OPENAI_API_KEY")))

    # V1.5 多媒体优先链路（默认关闭，避免影响旧链路）
    chat_v15_enabled: bool = _env_bool("CHAT_V15_ENABLED", False)
    chat_v15_max_images: int = _env_int("CHAT_V15_MAX_IMAGES", 3)
    chat_v15_max_videos: int = _env_int("CHAT_V15_MAX_VIDEOS", 1)
    chat_v15_max_docs: int = _env_int("CHAT_V15_MAX_DOCS", 10)
    chat_v15_image_rerank_mode: str = _env("CHAT_V15_IMAGE_RERANK_MODE", "text_rerank")
    chat_v15_lead_nudge_rounds: int = _env_int("CHAT_V15_LEAD_NUDGE_ROUNDS", 8)
    chat_v15_lead_nudge_stay_s: int = _env_int("CHAT_V15_LEAD_NUDGE_STAY_S", 120)
    chat_v15_guide_refresh_s: int = _env_int("CHAT_V15_GUIDE_REFRESH_S", 300)

    # V3：会话画像 + 兴趣驱动引导（默认关闭，旁路接入）
    chat_v3_enabled: bool = _env_bool("CHAT_V3_ENABLED", False)

    # V4：目录状态机 + 目录内关联检索（默认关闭，旁路接入）
    chat_v4_enabled: bool = _env_bool("CHAT_V4_ENABLED", False)
    # V4 开发者追踪：SSE done.debug.turn_trace（生产建议 false）
    expose_turn_trace: bool = _env_bool("EXPOSE_TURN_TRACE", True)
    # V4 试用账号池 JSON：[{"username":"demo","password":"***","label":"教师端"}]
    trial_accounts_json: str = _env("TRIAL_ACCOUNTS_JSON", "[]")

    @staticmethod
    def _normalized_model_name(model: str) -> str:
        return (model or "").strip().lower()

    @classmethod
    def is_openai_model(cls, model: str) -> bool:
        return cls._normalized_model_name(model).startswith("gpt-")

    @classmethod
    def is_deepseek_model(cls, model: str) -> bool:
        return cls._normalized_model_name(model).startswith("deepseek")

    @classmethod
    def is_dashscope_model(cls, model: str) -> bool:
        lower = cls._normalized_model_name(model)
        return lower.startswith("qwen") or lower.startswith("qwq")

    def resolve_model_endpoint(self, model: str) -> tuple[str, str]:
        normalized = (model or "").strip()
        if not normalized:
            return "", ""
        if self.is_openai_model(normalized):
            return self.openai_api_key, self.openai_base_url
        if self.is_deepseek_model(normalized):
            return self.deepseek_api_key, self.deepseek_base_url
        if self.is_dashscope_model(normalized):
            return (
                self.dashscope_api_key or self.llm_api_key,
                self.dashscope_base_url or self.llm_base_url,
            )
        return self.llm_api_key, self.llm_base_url

    def ensure_runtime_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.vector_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_runtime_dirs()
