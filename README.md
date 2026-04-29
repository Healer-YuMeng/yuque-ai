## Enterprise RAG MVP

基于语雀知识库的企业级 RAG 问答系统 MVP。

### 当前能力
- FastAPI 异步后端
- 手写 RAG pipeline（无 LangChain）
- 语雀文档拉取 + 文本切片 + embedding + FAISS 检索
- 向量召回不足时可通过 `yuque-mcp-server` fallback
- 极简 Web 问答页面
- SQLite 记录问答日志与文档元数据

### 项目结构
```text
app/
  api/chat_api.py
  core/config.py
  core/logger.py
  data/mcp_client.py
  data/splitter.py
  data/yuque_loader.py
  db/models.py
  db/repositories.py
  db/session.py
  rag/embedder.py
  rag/generator.py
  rag/pipeline.py
  rag/retriever.py
  schemas/chat.py
  service/qa_service.py
  storage/vector_store.py
  main.py
web/
  README_LEGACY.md
frontend/
scripts/
tests/
```

### 环境准备
```bash
source scripts/activate.sh
python -m pip install -r requirements.txt
cp .env.example .env
```

然后填写 `.env`：
- `YUQUE_TOKEN`
- `YUQUE_SCOPE`
- `DEEPSEEK_API_KEY` 或 `LLM_API_KEY`
- `LLM_BASE_URL=https://api.deepseek.com`

如果要启用向量检索，再额外配置：
- `EMBEDDING_API_KEY`
- `EMBEDDING_BASE_URL`

如果没有 embedding key，系统会自动退化为“直接搜索语雀正文后交给 DeepSeek 回答”的非向量模式。

### 构建索引
```bash
python scripts/build_index.py
```

可选：
```bash
BOOTSTRAP_QUERY=退款 python scripts/build_index.py
```

### 启动服务
```bash
uvicorn app.main:app --reload
```

### 前端（React）开发与发布
阶段二起默认使用 `frontend/`（React + Vite）。

一键本地联调（推荐）：
```bash
./scripts/dev_up.sh
```

启动后：
- FastAPI: `http://127.0.0.1:8000`
- Vite: `http://127.0.0.1:5173`

开发模式（前后端分离）：
```bash
cd frontend
npm install
npm run dev
```

生产构建（由 FastAPI 托管）：
```bash
./scripts/build_frontend.sh
uvicorn app.main:app --reload
```

说明：
- 若存在 `frontend/dist`，FastAPI 会优先返回 React 构建产物。
- 若不存在 `frontend/dist`，会回退到 `web/` 静态页面。
- `web/` 现为 legacy 回退前端，见 `web/README_LEGACY.md`。

### 常用 Make 命令
```bash
make dev       # 一键启动 FastAPI + Vite（前后端联调）
make down      # 一键停止 8000/5173 端口进程
make status    # 查看 8000/5173 端口监听状态
make build-ui  # 构建 React 前端产物到 frontend/dist
make run       # 仅启动 FastAPI
make test      # 运行 pytest
```

启动后可访问：
- `http://127.0.0.1:8000/`
- `POST /chat`
- `POST /index/rebuild`
- `POST /sync/yuque`
- `GET /health`

### 运行测试
```bash
pytest
```

### MCP fallback 配置
如果你本地已经安装并可启动 `yuque-mcp-server`，在 `.env` 中补充：

```bash
YUQUE_MCP_COMMAND=yuque-mcp-server
YUQUE_MCP_ARGS=
```

当向量检索命中为空或低于阈值时，系统会自动尝试调用 MCP 检索实时结果。

说明：
- 项目代码已预留 `YuqueMCPClient` 接入点。
- 如果你的运行环境暂时无法安装 MCP SDK，也不影响主链路运行；只是 fallback 不会生效。

