# Spec: 有为人工智能教育平台 · 访客销售咨询 AI（MVP）

本文档将 `doc/基于语雀知识库的销售ai客服机器人/需求文档V1.0.md` 与 `doc/基于语雀知识库的销售ai客服机器人/开发文档V1.0.md` 中的第一版范围，收敛为可验收的规格，并与当前仓库实现对照。**实施前请人工确认「开放问题」一节。**

**已决议（相对初版 SPEC）**：**会话记忆须服务端持久化**（不以「仅前端每次携带 history」为唯一手段），详见下文「会话与记忆（持久化）」。

---

## 假设（请纠正后再冻结规格）

1. **主交付形态**：网页聊天；后端 FastAPI，前端以 `frontend/`（React + TS + Vite）为准，不扩展遗留 `web/` 静态页为本期目标。
2. **知识来源**：产品事实仍以语雀知识库为权威；通过现有 RAG（向量 / 语雀直连 / MCP 等，取决于 `.env` 与索引）检索，**非**「全库预装进模型权重」。
3. **第一版角色**：仅访客；不做登录、CRM、人工接管、外呼触达。
4. **品牌**：对外文案与访客欢迎语使用「有为」人工智能教育平台（与 `frontend/src/visitorSales.ts` 一致）。
5. **会话记忆**：顾问应具备**跨轮上下文**；**权威状态在后端持久化**（推荐 SQLite，与现有 `data_runtime` 一致），以 `session_id` 关联；换设备时只要客户端持有同一 `session_id`（或未来登录后绑定）即可恢复对话连续性（实现分期落地）。

---

## Objective

### 要解决什么问题

在现有 Enterprise RAG MVP 上，增加**访客向销售咨询**体验：像顾问一样介绍「有为」AI 教育平台，基于语雀知识库作答，并在合适时机**自然引导留资**（电话或微信其一即可），且**不向访客展示知识库参考来源**。

**延伸目标（与需求对齐）**：顾问应能减少重复确认身份、承接前文；**服务端持久化**多轮对话（及可选的衍生状态，如已识别的访客倾向），使检索与生成在每轮可利用**历史上下文**，而非仅依赖本轮单句 `question`。

### 目标用户

未登录访客：机构/学校负责人、老师、家长、学生及对 AI 教育产品感兴趣的浏览者。

### 核心能力（验收口径）


| #   | 能力        | 验收要点                                         | 与当前实现关系                                                         |
| --- | --------- | -------------------------------------------- | --------------------------------------------------------------- |
| 1   | 访客开箱聊天    | 无需登录即可发消息并得到流式/非流式回复                         | ✅ `/chat`、`/chat/stream`                                        |
| 2   | 新会话欢迎     | 进入新会话即出现 AI 欢迎语；含产品介绍范围 + 轻量身份询问             | ✅ 前端静态欢迎语（`visitorSales.ts` / `App.tsx`）                        |
| 3   | 基于知识库答疑   | 产品、场景、案例、使用方式等可从检索摘录组织回答；无摘录时坦诚说明            | ✅ `RAGPipeline` + 访客 `generator` 分支                             |
| 4   | 顾问式表达     | 非「文档检索机」口吻；不强制 `## 回答` / `## 参考来源` 结构        | ✅ `_VISITOR_SALES_SYSTEM`                                       |
| 5   | 轻量身份 / 意向 | 从用户话中识别机构/老师/家长/学生等倾向；购买、试用、演示等意向用于调节语气与留资节奏 | ✅ `visitor_profile`、`visitor_intent` + `visitor_prompt` 注入生成用问题 |
| 6   | 留资识别与落库   | 识别大陆 11 位手机、常见「微信/wx」表述；同会话同联系方式去重写入 SQLite  | ✅ `lead_captures` + `LeadCaptureRepository.try_insert_lead`     |
| 7   | 访客不展示来源   | 响应中不向客户端返回可展示的 `sources`                     | ✅ `_apply_visitor_sales_client_mask` 清空 `sources`               |
| 8   | 无互动提醒     | 用户发过至少一条消息后，长时间无输入则插入一条温和留资提醒；同会话不重复骚扰       | ✅ 前端 90s（`INACTIVITY_MS`）+ 状态控制                                 |
| 9   | 快捷入口      | 提供若干预设问题降低冷启动                                | ✅ `VISITOR_QUICK_QUESTIONS`                                     |
| 10  | 会话关联      | 留资与调试信息能关联到会话                                | ✅ `session_id` + `chat_mode`                                    |
| 11  | **会话记忆（持久化）** | 后端按 `session_id` 存储并可加载多轮对话；生成侧使用历史，避免每轮像「首次见面」；身份/意向可在会话内延续 | ⏳ **待实现**（当前 API 仅传单句 `question`，无服务端 transcript） |


### 会话与记忆（持久化）— 规格说明

**目标**：用户不必在每轮重复「我是老师」；模型与规则能在**服务端**读到本会话此前内容（及可选摘要字段）。

**本期冻结参数（默认策略）**

- **历史窗口 K**：每轮生成时带入最近 **10 轮**（按消息条数计；建议实现为“最近 10 条 user/assistant 消息对”，或等价的固定条数窗口，具体以实现定义为准）。
- **保留策略**：会话记录默认保留 **7 天**；超过保留期可自动清理（按 `created_at`/`updated_at` 过期）。

**原则**

1. **权威数据源在后端**：浏览器 `localStorage` 可作 UI 缓存，但**以服务端存储的对话为准**（或可定义「合并策略」，默认以后端为准）。
2. **关联键**：沿用请求中的 `session_id`（必填或 strongly recommended for visitor）；无 `session_id` 时不保证跨请求记忆。
3. **存储内容（最低）**：按时间序的多条记录，至少包含 `role`（user/assistant）、`content`（文本）、`created_at`；可选 `token_count` / 截断策略字段。
4. **衍生状态（推荐）**：在同表或侧表存 `visitor_type`（已确认倾向）、`contact_hint` 等，避免仅靠全文回溯。
5. **检索策略**：可为「仍用本轮用户句做向量检索」+「把最近 K 轮拼进生成 prompt」；复杂场景可另立 ADR（二轮检索不在本节强制）。
6. **隐私与留存**：对话可能含电话/微信；须定义**保留天数 / 最大条数**（开放问题）；日志禁止明文打印完整密钥。

**非目标（本期可不做的记忆相关）**：跨用户全局画像、与 CRM 同步、加密静止数据（除非合规要求升级）。

### 本期明确不做（边界）

与需求文档 3.2 节一致：登录/注册、内部销售与付费客户角色、历史客户画像、订单、CRM、销售任务、人工接管、短信/企微等**主动外呼**、访客模式下展示语雀参考来源、销售后台与线索分配。

### 成功标准（可测）

- 默认请求为 `chat_mode=visitor_sales` 时，完成一轮问答且**不**在 JSON 中向客户端暴露 `sources`（或为空列表）。
- 用户消息含合法手机号或约定格式微信时，带有效 `session_id` 的请求在 `lead_captures` 中产生**一条**记录（重复同联系方式不产生重复行）。
- 新会话首条为欢迎语；用户发消息后若 90s 无操作，出现**一条**无互动提醒（同会话不重复，除非产品改为可配置）。
- **记忆（持久化）落地后须满足**：同一 `session_id` 下，用户在前一轮声明身份（如「我是家长」）后，下一轮仅问业务问题时不应被顾问**再次强行追问身份**（除非产品明确要求重置）；刷新页面或新开标签（仍携带同 `session_id` 时）应能继续基于服务端记录对话（实现验收以集成测试或手测清单为准）。
- `make test` 通过；若改前端则 `npm run lint` 与 `npm run build` 通过（见 `AGENTS.md`）。

---

## Tech Stack


| 层级  | 技术                                                                                                   |
| --- | ---------------------------------------------------------------------------------------------------- |
| 后端  | Python 3.11+、FastAPI、Pydantic、SQLite、httpx；RAG：自研 pipeline + FAISS 向量存储；可选 OpenAI 兼容 LLM（如 DeepSeek）；**会话持久化目标与现有 SQLite 同栈** |
| 前端  | React 19、TypeScript、Vite 8；`react-markdown` + `remark-gfm`                                           |
| 外部  | 语雀 Open API；可选 Yuque MCP（stdio）；向量依赖 embedding 配置                                                    |
| 配置  | 根目录 `.env`（模板 `.env.example`）                                                                        |


---

## Commands

在仓库根目录（详见 `AGENTS.md`）：

```bash
# 虚拟环境
source scripts/activate.sh

# Python 依赖
python -m pip install -r requirements.txt

# 后端测试
make test

# 仅后端
make run

# 前后端联调
make dev

# 前端（在 frontend/）
cd frontend && npm install && npm run dev
cd frontend && npm run lint
cd frontend && npm run build

# 向量索引（按需）
python scripts/build_index.py
```

---

## Project Structure（与本规格相关）

```text
app/
  api/chat_api.py          # HTTP：chat / stream
  service/qa_service.py    # 编排：模式、留资 mask、RAG 调用
  rag/pipeline.py          # 检索 → 生成
  rag/retriever.py         # 向量 / 语雀 / MCP
  rag/generator.py         # 含 visitor_sales 分支
  conversation/            # 访客：身份、意向、联系方式、生成用问题拼接
  db/models.py             # 含 lead_captures DDL；（规划）chat_sessions / chat_messages
  db/repositories.py       # LeadCaptureRepository；（规划）会话读写
  schemas/chat.py          # ChatRequest：chat_mode、session_id 等；（规划）与持久化一致的契约
frontend/src/
  App.tsx                  # 访客 UI、session、无互动、请求体
  visitorSales.ts          # 欢迎语、快捷问、90s、联系方式启发式（前端）
doc/基于语雀知识库的销售ai客服机器人/
  需求文档V1.0.md
  开发文档V1.0.md
tests/                     # pytest
```

---

## Code Style

与仓库现有风格一致（见 `AGENTS.md`）：分层清晰、密钥不进代码、类型注解、`from __future__ import annotations`；前端遵循 ESLint flat 配置。

**示例（访客生成用问题拼接，检索仍用用户原句）：**

```python
# app/conversation/visitor_prompt.py — 仅示意
def build_visitor_generation_question(user_question: str) -> str:
    ...
    return "【内部分析…】\n...\n\n用户原话：\n" + q
```

---

## Testing Strategy


| 层级   | 工具                             | 范围                                       |
| ---- | ------------------------------ | ---------------------------------------- |
| 单元测试 | pytest，`tests/`                | Schema、contact 抽取、pipeline 行为、API Fake 等 |
| 前端   | `npm run lint`、`npm run build` | 类型与构建门禁                                  |
| 手动   | 浏览器 + 配置好的 `.env`              | 流式体验、留资、无互动提醒、**同 session 刷新后续聊** |


覆盖率：无强制百分比；**改访客/留资/聊天契约须带回归测试或明确写明仅手测原因**。

---

## Boundaries

### Always（始终）

- 运行与本次改动相关的 `pytest` 后再合并。
- 密钥仅通过 `.env` 或环境变量注入。
- 访客模式下不向前端返回可用于「语雀文档溯源」的 `sources`。
- 产品事实陈述须受检索摘录约束（已在访客 system prompt 中要求）。
- **会话持久化实现后**：写入对话须遵守既定保留策略；禁止在日志中输出完整对话中的敏感联系方式（必要时脱敏）。

### Ask first（先问再动）

- 新增依赖、CI 工作流、大范围表结构变更。
- 删除或弱化 `rag` 模式、改动 `/chat` 公共契约。
- 新增对外管理 API（如线索列表）或认证方案。
- **新增或变更聊天记录表结构、默认保留天数、是否对客户开放「导出会话」**。

### Never（禁止）

- 将 `YUQUE_TOKEN`、`DEEPSEEK_API_KEY` 等写入仓库或 SPEC 示例值。
- 在访客 UI 默认暴露内部 skill 与参考来源（与第一版产品边界冲突时）。
- 无审批删除测试或绕过失败用例合并。

---

## 开放问题（需产品 / 维护者确认）

1. **双模式保留**：`chat_mode=rag` 是否长期保留供内部调试，还是第一版起完全隐藏？
2. **线索运营**：是否需要在 MVP 内增加**只读 HTTP API** 或简单管理页导出 `lead_captures`，还是接受「直连 SQLite」？
3. **无互动时长**：固定 90s 是否写入产品冻结，还是改为可配置（`.env`）？
4. **欢迎语文案**：以代码中「有为」版为准，还是必须与需求文档中的「我们的人工智能教育平台」逐字对齐？
5. **二轮检索**：是否在后续版本将「模型输出的受控检索词 → 二次检索」纳入规格（当前为单轮 retrieve → generate）？
6. **会话记忆（持久化）— 待细化**：
   - **保留策略**：默认保留 **7 天（已冻结）**；是否还需要单会话最大消息条数或总字符上限；
   - **访客 session_id**：是否强制每个访客会话必须生成 UUID（前端已有则对齐后端校验）；
   - **加载策略**：每轮带入模型的最近 **10 轮（已冻结）** 全文 vs 滑动窗口 + 摘要（摘要是否允许二次 LLM 调用）；
   - **目录/知识增强**：是否在首轮或按需注入压缩 TOC（与「顾问心中有目录」需求一并规划）。

---

## 文档维护

- 产品细节以 `doc/基于语雀知识库的销售ai客服机器人/需求文档V1.0.md` 为业务原文；本 SPEC 为工程可执行摘要。
- 决策变更时先更新本文件再改代码（见 spec-driven-development 技能「Keeping the Spec Alive」）。

---

## 人类确认清单（进入大规模开发前）

- 已阅读并同意「假设」与「开放问题」的处理方式  
- Success Criteria 与 Boundaries 可接受  
- 无新增阻塞项

**确认后**：可将本 SPEC 与 PR 关联，并按技能 Phase 2 拆实施计划（若需要另建 `PLAN.md` 可再开任务）。