# AGENTS.md

面向在本仓库中工作的 **AI Coding Agent** 的操作说明：命令、架构边界、测试与安全要求。详细产品行为仍以根目录 [README.md](README.md) 为准。

---

## 1. 项目概述

**Enterprise RAG MVP**：基于语雀知识库的企业级 RAG 问答演示——后端用 **FastAPI** 提供异步 API、RAG 流水线、SQLite 元数据与日志、FAISS 向量检索（可选 MCP fallback）；前端为 **React 19 + TypeScript + Vite**，构建产物可由 FastAPI 托管。仓库为「**Python 应用 + 独立 frontend 目录**」的双端结构（**不是** Java/Spring monorepo）。根目录另有遗留 CLI/脚本（如 `main.py`、`self_check.py`）与 `web/` 静态回退，主 Web 路径以 `app/` + `frontend/` 为准。

---

## 2. 快速命令

### 2.1 速查表

| 目的 | 命令 | 说明 |
|------|------|------|
| 激活虚拟环境 | `source scripts/activate.sh` | 默认 venv：`./yuqueai`；可 `export VENV_DIR=其他目录` 后执行 |
| 安装 Python 依赖 | `python -m pip install -r requirements.txt` | 需在已激活的 venv 中执行 |
| 一键前后端联调 | `make dev` 或 `./scripts/dev_up.sh` | FastAPI `:8000` + Vite `:5173` |
| 仅后端 | `make run` | 使用 `yuqueai/bin/python -m uvicorn`，无需先 `source` |
| 停止联调占用端口 | `make down` | 释放 8000/5173 等 |
| 查看端口状态 | `make status` | |
| 构建前端产物 | `make build-ui` 或 `./scripts/build_frontend.sh` | 输出到 `frontend/dist`，供 FastAPI 托管 |
| 前端开发依赖 | `cd frontend && npm install` | |
| 前端本地开发 | `cd frontend && npm run dev` | 需后端已起；`/chat`、`/mcp`、`/docs` 由 Vite 代理到 8000 |
| 前端 Lint | `cd frontend && npm run lint` | |
| 后端测试 | `make test` 或 `source scripts/activate.sh && pytest` | |
| 构建向量索引 | `python scripts/build_index.py` | 可选 `BOOTSTRAP_QUERY=...` |
| 语雀连通性自检（独立脚本） | `python self_check.py` | 依赖根目录 `.env` 或环境变量中的 `YUQUE_TOKEN` 等 |

### 2.2 环境变量配置与优先级

- **主配置方式**：复制 [.env.example](.env.example) 为仓库根目录 **`.env`**，填入密钥与业务参数。应用通过 `app/core/config.py` 中的 `load_dotenv(BASE_DIR / ".env")` 加载。
- **优先级（与 `python-dotenv` 默认行为一致）**：已在进程环境中设置的变量 **不会被** `.env` 覆盖；仅对「尚未设置」的变量，由 `.env` 补全。
- **CI / 一次性覆盖**：在命令前导出变量，或在平台密钥管理中注入环境变量，无需提交 `.env`。
- **未使用** `~/<project>_env` 类全局文件；若需个人机器级覆盖，请使用 shell profile、`export` 或各 IDE Run Configuration 中的环境变量。

---

## 3. 后端架构

### 3.1 包结构（ASCII）

```text
app/
├── main.py              # FastAPI 入口：生命周期、CORS、静态资源挂载、路由注册
├── api/
│   └── chat_api.py      # HTTP 路由：chat、stream、health、index/sync、docs/suggest 等
├── core/
│   ├── config.py        # Settings：从环境变量/.env 读取，路径与默认值
│   └── logger.py        # 日志初始化
├── data/
│   ├── yuque_loader.py  # 语雀 API 拉取与错误类型
│   ├── splitter.py      # 文本切片
│   └── mcp_client.py  # MCP fallback 客户端
├── db/
│   ├── models.py        # SQLite DDL 字符串与表结构定义
│   ├── session.py       # 数据库会话工厂
│   └── repositories.py # 文档与问答日志持久化
├── rag/
│   ├── pipeline.py      # RAG 编排
│   ├── retriever.py     # 检索
│   ├── embedder.py      # 向量嵌入
│   ├── generator.py     # LLM 生成
│   └── skill_router.py  # 技能/意图路由相关
├── schemas/
│   ├── chat.py          # 聊天相关 Pydantic 模型
│   └── docs.py          # 文档联想等请求/响应模型
├── service/
│   └── qa_service.py    # 领域服务：组合 loader、vector、repo、generator
└── storage/
    └── vector_store.py  # FAISS 向量存储封装
```

### 3.2 核心子系统（简要）

- **QAService**：对外问答、流式输出、索引重建、运行时模式等与 HTTP 的桥梁。
- **YuqueLoader**：语雀 HTTP 调用与错误处理。
- **VectorStore + Retriever + Pipeline**：召回与生成链路；无 embedding 时可走降级路径（见 README）。
- **Repositories + SQLite**：文档元数据与问答日志。

**更细的产品与 API 列表** → [README.md](README.md) 中「当前能力」「启动后可访问」章节。

---

## 4. 前端架构

- **技术栈**：React 19、TypeScript、Vite 8；`react-markdown` + `remark-gfm` 用于展示。
- **开发与生产**：开发时 `vite.config.ts` 将 `/chat`、`/mcp`、`/docs` **代理**到 `http://127.0.0.1:8000`；生产构建后由 FastAPI 同域提供，API 使用 **相对路径**（如 `/chat/stream`）。
- **路由**：当前为单页应用，无独立 React Router 分层说明文件；以 `frontend/src/App.tsx` 为准。
- **组件与 API 约定**：通过 `fetch` 调用后端；新增接口时同步更新 Vite `server.proxy`（开发态）并保证生产态同路径在 FastAPI 上可用。
- **质量**：`npm run lint`（ESLint flat config + typescript-eslint）；构建 `npm run build`。

**设计文档**：仓库内暂无单独的 `docs/design-docs/frontend-architecture.md`；以前端源码与 [README.md](README.md) 为准。

---

## 5. 关键约定

1. **变更范围**：只改与任务相关的文件；避免无关格式化、大段重排或与需求无关的重构。
2. **配置与密钥**：禁止将 `YUQUE_TOKEN`、`DEEPSEEK_API_KEY`、`OPENAI_API_KEY`、`EMBEDDING_API_KEY` 等写入代码或提交到 git；仅通过 `.env`（本地）或 CI 密钥注入。
3. **后端分层**：路由层（`app/api`）薄封装；业务逻辑放在 `app/service`；数据访问在 `app/db`；外部系统访问在 `app/data`；避免在路由函数内堆叠复杂业务（与现有 `qa_service` 风格一致）。
4. **错误与 HTTP**：对外 API 使用 `HTTPException` 或已存在的异常类型（如 `YuqueLoaderError`、`GeneratorConfigError`）并在路由中映射为合适状态码；不要随意吞掉异常。
5. **类型与风格**：Python 侧与现有模块一致，优先使用 `from __future__ import annotations`、类型注解；中文注释仅在与仓库现有风格一致时使用。
6. **前端**：遵循现有 ESLint 规则；保持与当前 UI 的间距与文案风格一致。
7. **CORS**：开发中 `allow_origins=["*"]` 仅为便利；若延伸到生产部署，应收紧来源并配合 HTTPS。

**语雀侧写作规范（提升 RAG 效果）** → [AUTHORING_GUIDE.md](AUTHORING_GUIDE.md)。  
**遗留静态前端** → [web/README_LEGACY.md](web/README_LEGACY.md)。

---

## 6. 本地开发及验证流程

推荐闭环：**改代码 →（前端若涉及）`npm run build` 或 `npm run dev` → 启动后端 → 浏览器/ curl 验证**。

1. `source scripts/activate.sh`（若手动跑 pytest/脚本）。
2. `make run` 或 `make dev`。
3. 浏览器打开 `http://127.0.0.1:8000/`（不要依赖 `http://0.0.0.0:8000` 作为浏览器地址）。
4. **curl 模板示例**：
   - `curl -sS http://127.0.0.1:8000/health`
   - `curl -sS http://127.0.0.1:8000/runtime-mode`
   - 流式聊天需按 `POST /chat/stream` 的 SSE 协议构造请求，可直接以 UI 验证为主。
5. **Token**：在 `.env` 中配置 `YUQUE_TOKEN`（及 `YUQUE_SCOPE` 等）；语雀 Token 从语雀开放平台/账户设置获取，勿粘贴到 issue 或聊天日志。
6. **日志与数据**：运行时 SQLite 与向量目录默认在 `data_runtime/`（见 `Settings`）；勿提交含敏感内容的数据库备份。

**独立语雀 API 自检（非 FastAPI）**：配置好环境后运行 `python self_check.py`（见该文件说明）。

---

## 7. 质量检查

| 检查项 | 命令 | 备注 |
|--------|------|------|
| 后端单元测试 | `make test` | `tests/conftest.py` 会设置若干默认 env，避免测试依赖本机 `.env` 中的 MCP/意图开关 |
| 前端 Lint | `cd frontend && npm run lint` | |
| 前端构建 | `cd frontend && npm run build` | 与 `make build-ui` 一致（脚本内 npm 路径以本机为准） |
| 架构/格式自动化门禁 | 无 | 当前仓库 **未** 配置 `ruff`/`black`/`mypy`/`pre-commit`；新增此类工具需与维护者约定后再改 CI |

提交前至少应：**相关 pytest 通过**；若改前端，**`npm run build` 或 `npm run lint` 无新增错误**。

---

## 8. 参考项目约定

- 本仓库为自包含 MVP，**无**强制对齐的外部 monorepo 模板。
- 引入新模式时：**优先模仿本仓库 `app/service`、`app/api` 既有写法**；第三方集成（语雀、OpenAI 兼容接口、MCP）以官方文档与 [README.md](README.md) 环境变量说明为准。

---

## 9. 文档导航

| 文档 | 内容 |
|------|------|
| [README.md](README.md) | 能力说明、目录结构、环境变量、索引构建、启动命令、API 列表 |
| [AUTHORING_GUIDE.md](AUTHORING_GUIDE.md) | 语雀知识库写作规范（非代码） |
| [web/README_LEGACY.md](web/README_LEGACY.md) | 无 `frontend/dist` 时的静态回退说明 |
| [.env.example](.env.example) | 环境变量模板 |
| [doc/开发计划.md](doc/开发计划.md) | 新人向开发计划：准确性、可用性与 RAG/Skill 优先级（非演示向） |
| [doc/产品使用文档.md](doc/产品使用文档.md) | 业务配置与使用说明；含检索/生成链路与 SSE 的技术说明 |
| [doc/前后端启动说明.md](doc/前后端启动说明.md) | 启动后端、前端的命令整理与常见问题 |

当前仓库 **无** `docs/architecture.md` 或 `docs/design-docs/*`；若后续补充，请在本表追加一行索引。
