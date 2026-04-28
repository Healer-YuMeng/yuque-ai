## Yuque-RAG Demo（语雀知识库可被智能体引用）

### 你将得到什么
- 用户在命令行提问
- 智能体自动调用语雀 OpenAPI：
  - `GET /api/v2/search` 搜索文档
  - `GET /api/v2/repos/:book_id/docs/:id` 拉取正文
- 最终输出：**答案 + 引用（语雀文档链接）**

### 前置条件
- Python 3.9+
- 一个可用的语雀 Token（请求头 `X-Auth-Token`）

### 安装依赖
在本目录执行：

```bash
python -m pip install -r requirements.txt
```

### Windows 中文乱码（建议）
PowerShell 里先执行：

```powershell
chcp 65001
```

### 配置环境变量
PowerShell（注意 `setx` 需要重新打开终端生效）：

```powershell
setx OPENAI_API_KEY "你的OpenAI Key"
setx YUQUE_TOKEN "你的语雀Token"
setx YUQUE_SCOPE "团队login/知识库slug"
```

- `YUQUE_SCOPE` 用于把搜索限定在某个团队/知识库内，例如：`group_a/book_x`

### 自检（验证语雀 API 可被调用）

```bash
python self_check.py
```

成功时你会看到：`/hello`、`/search`、以及拉取正文成功的输出。

### 运行智能体问答

```bash
python main.py 退款多久到账？
```

输出会包含“引用”列表（title + url）。

### 给内容同事的写作规范
见 `AUTHORING_GUIDE.md`。

