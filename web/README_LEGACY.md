# Legacy Web UI

`web/` 目录是旧版静态前端（HTML/CSS/JS），仅作为回退方案保留。

当前默认前端为 `frontend/`（React + Vite）：

- 开发：`cd frontend && npm run dev`
- 生产构建：`./scripts/build_frontend.sh`
- FastAPI 会优先托管 `frontend/dist`，不存在时才回退到 `web/`
