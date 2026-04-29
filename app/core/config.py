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
    yuque_scope: str = _env("YUQUE_SCOPE")
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

    llm_provider: str = _env("LLM_PROVIDER", "deepseek")
    llm_model: str = _env("LLM_MODEL", "deepseek-chat")
    openai_base_url: str = _env("OPENAI_BASE_URL")
    deepseek_base_url: str = _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    llm_base_url: str = _env(
        "LLM_BASE_URL",
        _env("DEEPSEEK_BASE_URL", _env("OPENAI_BASE_URL", "https://api.deepseek.com")),
    )
    llm_api_key: str = _env(
        "LLM_API_KEY",
        _env("DEEPSEEK_API_KEY", _env("OPENAI_API_KEY")),
    )

    top_k: int = _env_int("TOP_K", 4)
    chunk_size: int = _env_int("CHUNK_SIZE", 800)
    chunk_overlap: int = _env_int("CHUNK_OVERLAP", 120)
    retrieval_score_threshold: float = _env_float("RETRIEVAL_SCORE_THRESHOLD", 0.35)

    mcp_server_command: str = _env("YUQUE_MCP_COMMAND")
    mcp_server_args: str = _env("YUQUE_MCP_ARGS")
    mcp_search_tool: str = _env("YUQUE_MCP_SEARCH_TOOL", "search")
    mcp_get_doc_tool: str = _env("YUQUE_MCP_GET_DOC_TOOL", "get_doc")
    mcp_timeout_s: float = _env_float("YUQUE_MCP_TIMEOUT_S", 20.0)
    force_mcp_fallback: bool = _env_bool("FORCE_MCP_FALLBACK", False)
    auto_mcp_tool_router: bool = _env_bool("AUTO_MCP_TOOL_ROUTER", False)
    intent_llm_enabled: bool = _env_bool("INTENT_LLM_ENABLED", False)
    intent_llm_model: str = _env("INTENT_LLM_MODEL", _env("LLM_MODEL", "deepseek-chat"))

    web_dir: Path = BASE_DIR / "web"
    frontend_dir: Path = BASE_DIR / "frontend"
    frontend_dist_dir: Path = BASE_DIR / "frontend" / "dist"
    expose_source_urls: bool = _env_bool("EXPOSE_SOURCE_URLS", False)

    def ensure_runtime_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.vector_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_runtime_dirs()

