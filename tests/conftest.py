from __future__ import annotations

import sys
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 让测试不依赖开发者本机的 .env
# 这些开关会触发额外的 MCP/意图路由逻辑，单元测试里使用的是 Fake 对象。
os.environ.setdefault("FORCE_MCP_FALLBACK", "false")
os.environ.setdefault("AUTO_MCP_TOOL_ROUTER", "false")
os.environ.setdefault("INTENT_LLM_ENABLED", "false")
os.environ.setdefault("ASSISTANT_META_LLM_ROUTER", "false")
os.environ.setdefault("VISION_ENABLED", "false")

