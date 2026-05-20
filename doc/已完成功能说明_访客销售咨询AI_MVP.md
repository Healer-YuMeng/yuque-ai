# 已完成功能说明：访客销售咨询 AI（MVP）

本文档用于沉淀「基于语雀知识库的销售 AI 客服机器人」第一版 MVP 在本仓库中的**已完成功能**与**实际实现形态**，对应：

- `doc/基于语雀知识库的销售ai客服机器人/需求文档V1.0.md`
- `doc/基于语雀知识库的销售ai客服机器人/开发文档V1.0.md`
- 根目录 `SPEC.md`（工程可执行规格，含会话记忆持久化策略）

> 说明：本文是“已完成能力”的工程交付文档，不替代需求原文；若需求变更，请先更新 `SPEC.md` 再实现。

---

## 1. 项目目标（MVP）

面向未登录访客的网页聊天 AI，围绕「有为人工智能教育平台」提供销售式咨询：

- **主动欢迎**（新会话首条欢迎语）
- **基于语雀知识库答疑**（产品介绍/使用方式/案例/试用购买等）
- **顾问式表达与引导**（回答后自然追问、合适时机引导留资）
- **联系方式收集**（电话/微信其一即可，写入 SQLite）
- **访客模式不展示参考来源**（对外隐藏 sources）
- **服务端持久化会话记忆**（最近 10 轮上下文用于承接对话；默认保留 7 天）

---

## 2. 已完成的核心功能清单

### 2.1 访客聊天与欢迎语

- **无需登录即可聊天**：前端页面直接发起 `/chat` 或 `/chat/stream`。
- **新会话自动出现欢迎语**：前端在新会话创建时插入首条 AI 欢迎语（品牌为「有为」）。
- **快捷问题**：提供若干常见问题按钮，降低冷启动成本。

相关实现位置：

- `frontend/src/visitorSales.ts`
- `frontend/src/App.tsx`

### 2.2 访客身份与意向的轻量识别（规则）

在访客模式下，对用户输入做轻量规则识别，用于调节语气与留资节奏：

- 访客倾向：机构/老师/学生/家长等（unknown 则不强行贴标签）
- 意向线索：购买/价格/合作、试用/演示/体验

相关实现位置：

- `app/conversation/visitor_profile.py`
- `app/conversation/visitor_intent.py`
- `app/conversation/visitor_prompt.py`

### 2.3 销售顾问式生成（visitor_sales prompt）

访客模式生成策略与知识库问答不同点：

- 语气更像真人顾问
- **不强制输出“## 回答 / ## 参考来源”固定结构**
- 不向访客展示内部资料编号/文档标题作为“来源”
- 若用户已留联系方式，温和确认并说明后续可由顾问联系

相关实现位置：

- `app/rag/generator.py`（访客 system prompt 与 visitor_sales 分支）
- `app/service/qa_service.py`（基于 `chat_mode` 切换 visitor_sales）

### 2.4 语雀知识库检索（RAG / 直连 / MCP 回退）

系统按运行时条件走不同检索模式（以 `.env`/索引与配置为准）：

- **向量检索（FAISS）**：embedding 可用时优先
- **语雀直连**：embedding 不可用或作用域不一致等场景
- **MCP 回退（可选）**：当启用 MCP 时，支持目录/文档列表/搜索/拉全文等只读能力

实现要点：

- 检索与生成分离：检索使用用户原问题；生成可拼接访客内部提示与历史对话块
- MCP 工具名按语雀官方注册名默认：`yuque_search` / `yuque_get_doc`

相关实现位置：

- `app/rag/retriever.py`
- `app/rag/pipeline.py`
- `app/data/yuque_loader.py`
- `app/data/mcp_client.py`
- `app/core/config.py`（MCP 工具名默认值）

### 2.5 访客模式隐藏参考来源（sources mask）

访客模式下，对客户端返回进行处理：

- `sources` 强制置空（避免前端展示参考来源）
- `debug` 中保留 `visitor_sales` 字段用于调试（含是否检测到联系方式、是否保存成功等）

相关实现位置：

- `app/service/qa_service.py`（`_apply_visitor_sales_client_mask`）

### 2.6 留资识别与落库（SQLite）

- **识别电话**：大陆 11 位手机号
- **识别微信**：常见 “微信/微信号/wx/wechat: xxx” 等表达
- **落库表**：`lead_captures`
- **去重**：同一 `session_id + contact_type + contact_value` 唯一索引

相关实现位置：

- `app/conversation/contact_extractor.py`
- `app/db/models.py`
- `app/db/repositories.py`（`LeadCaptureRepository`）

### 2.7 无互动留资提醒（90 秒）

访客在已发送至少一条消息后，超过 90 秒无输入：

- 前端插入一次温和提醒文案，引导留下电话/微信
- 同会话不重复打扰（本地 session state 记录）

相关实现位置：

- `frontend/src/visitorSales.ts`（`INACTIVITY_MS = 90_000`）
- `frontend/src/App.tsx`（无互动定时逻辑与去重）

### 2.8 服务端会话记忆持久化（最近 10 轮，保留 7 天）

为解决“顾问没有记忆、重复确认身份”的问题，新增服务端持久化：

- **会话表**：`chat_sessions`
  - 关键字段：`session_id`、`chat_mode`、`advisor_role`（预留）、`visitor_type`（结构化状态）、时间戳
- **消息表**：`chat_messages`
  - 存储 `user/assistant` 文本消息与时间戳
- **生成侧使用历史**：
  - 每轮生成时，从服务端取最近 **20 条消息（≈10 轮）**拼为 history block（仅用于承接上下文，不用于向量检索）
  - 每条消息截断，避免 prompt 无限膨胀
- **保留策略**：
  - 启动时清理一次默认 **7 天**过期数据（后续可演进为定时任务/后台清理）

相关实现位置：

- `app/db/models.py`（`chat_sessions` / `chat_messages` DDL）
- `app/db/repositories.py`（`ChatSessionRepository`）
- `app/service/qa_service.py`（写入 user/assistant 消息、拼接 history block）

---

## 3. 已完成的 API（对外契约）

### 3.1 聊天接口

- `POST /chat`
  - Body：`{ question, chat_mode, session_id, owner, model, token_profile, selected_yuque_docs }`
  - 返回：`ChatResponse`（访客模式 sources 为空）

- `POST /chat/stream`（SSE）
  - 同上，返回 SSE events（含 stage/token/done/error）

相关实现位置：`app/api/chat_api.py`

### 3.2 会话历史读取（用于刷新恢复）

- `GET /chat/history?session_id=...&limit=...`
  - 返回：`{ session_id, messages: [{ role, text, created_at }] }`

相关实现位置：

- `app/api/chat_api.py`
- `app/schemas/chat.py`

### 3.3 语雀图片同源代理

- `GET /yuque/asset?t=...`
  - 同源代理语雀图片（白名单校验），供回答中 Markdown 插图展示

相关实现位置：`app/api/chat_api.py`

---

## 4. 数据与存储（SQLite）

数据库默认路径（可配置）：`data_runtime/rag_mvp.db`

已落地表：

- `documents` / `chunks`：索引与切片元数据
- `qa_logs`：问答日志
- `lead_captures`：留资记录（带唯一索引）
- `chat_sessions` / `chat_messages`：会话与消息持久化（用于记忆）

DDL 定义位置：`app/db/models.py`

---

## 5. 配置与运行要点（与本期相关）

### 5.1 必要配置

- 语雀：`YUQUE_TOKEN`、`YUQUE_SCOPE`
- LLM：`DEEPSEEK_API_KEY`（或兼容 `OPENAI_API_KEY` 体系，取决于模型选择）

### 5.2 MCP（可选）

若启用语雀 MCP，默认工具名为：

- `YUQUE_MCP_SEARCH_TOOL=yuque_search`
- `YUQUE_MCP_GET_DOC_TOOL=yuque_get_doc`

模板见：`.env.example`

---

## 6. 验收与回归（本仓库已通过）

### 6.1 后端测试

```bash
make test
```

### 6.2 前端质量

```bash
cd frontend
npm run lint
npm run build
```

### 6.3 手动检查清单（建议）

- 新会话首条欢迎语出现（品牌“有为”）
- 连续两轮对话中，第二轮不重复追问身份（同 `session_id`）
- 刷新页面后可通过 `/chat/history` 恢复服务端记录（在本地会话无用户消息时会自动 hydration）
- 留资：输入手机号/微信，SQLite `lead_captures` 有记录且同会话不重复
- 访客模式 sources 不展示（响应中 sources 为空）

---

## 7. 已明确不做（仍符合 MVP 边界）

本期仍不包含：

- 登录/注册/多角色权限
- CRM、线索分配、销售后台、人工接管
- 微信/企微/短信等外部主动触达
- 访客模式展示知识库参考来源

