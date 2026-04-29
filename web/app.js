const questionEl = document.getElementById("question");
const submitEl = document.getElementById("submit");
const stopEl = document.getElementById("stop");
const chatListEl = document.getElementById("chat-list");
const mcpMetaEl = document.getElementById("mcp-meta");
const mcpToolsEl = document.getElementById("mcp-tools");
let currentController = null;
let activeAssistantBubble = null;
let activeDebugBlock = null;

function appendMessage(role, text) {
  const row = document.createElement("div");
  row.className = `msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  row.appendChild(bubble);
  chatListEl.appendChild(row);
  chatListEl.scrollTop = chatListEl.scrollHeight;
  return bubble;
}

function appendAssistantFrame() {
  const row = document.createElement("div");
  row.className = "msg assistant";
  const wrapper = document.createElement("div");
  wrapper.style.width = "100%";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const debug = document.createElement("pre");
  debug.className = "debug";
  debug.textContent = "等待调试信息...";
  wrapper.appendChild(bubble);
  wrapper.appendChild(debug);
  row.appendChild(wrapper);
  chatListEl.appendChild(row);
  chatListEl.scrollTop = chatListEl.scrollHeight;
  return { bubble, debug };
}

function parseSseEvent(block) {
  const lines = block.split("\n");
  let event = "message";
  let data = "";
  for (const line of lines) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      data += line.slice(5).trim();
    }
  }
  return { event, data };
}

async function loadMcpCapabilities() {
  try {
    const response = await fetch("/mcp/capabilities");
    const data = await response.json();
    mcpMetaEl.textContent = `状态：${data.enabled ? "已启用" : "未启用"} | 作用域：${data.repo_scope || "未配置"}`;
    mcpToolsEl.innerHTML = "";
    (data.tools || []).forEach((tool) => {
      const li = document.createElement("li");
      const tag = tool.status === "integrated" ? "已接入" : "可接入";
      li.textContent = `[${tag}] ${tool.name}（${tool.category}）- ${tool.description}`;
      mcpToolsEl.appendChild(li);
    });
  } catch (_error) {
    mcpMetaEl.textContent = "MCP 能力读取失败。";
  }
}

async function askQuestion() {
  const question = questionEl.value.trim();
  if (!question) {
    return;
  }
  appendMessage("user", question);
  questionEl.value = "";
  const { bubble, debug } = appendAssistantFrame();
  activeAssistantBubble = bubble;
  activeDebugBlock = debug;

  submitEl.disabled = true;
  stopEl.disabled = false;
  activeAssistantBubble.textContent = "";
  activeDebugBlock.textContent = "正在获取调试信息...";
  currentController = new AbortController();

  try {
    const response = await fetch("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      signal: currentController.signal,
    });
    if (!response.ok || !response.body) {
      throw new Error("stream_unavailable");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let doneReceived = false;

    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split("\n\n");
      buffer = blocks.pop() || "";
      for (const block of blocks) {
        if (!block.trim()) {
          continue;
        }
        const parsed = parseSseEvent(block);
        if (parsed.event === "token") {
          try {
            const payload = JSON.parse(parsed.data);
            activeAssistantBubble.textContent += payload.token || "";
            chatListEl.scrollTop = chatListEl.scrollHeight;
          } catch (_err) {
            // ignore malformed chunk
          }
        } else if (parsed.event === "done") {
          doneReceived = true;
          const payload = JSON.parse(parsed.data);
          activeAssistantBubble.textContent =
            payload.answer || activeAssistantBubble.textContent || "没有返回回答。";
          activeDebugBlock.textContent = payload.debug
            ? JSON.stringify(payload.debug, null, 2)
            : "本次未返回 MCP 调试信息（可能未走 fallback）。";
        } else if (parsed.event === "error") {
          const payload = JSON.parse(parsed.data || "{}");
          throw new Error(payload.message || "请求失败");
        }
      }
    }

    if (!doneReceived) {
      activeDebugBlock.textContent = "流式输出中断，未收到完成事件。";
      if (!activeAssistantBubble.textContent) {
        activeAssistantBubble.textContent = "请求失败，请稍后重试。";
      }
    }
  } catch (error) {
    if (error?.name === "AbortError") {
      if (!activeAssistantBubble.textContent) {
        activeAssistantBubble.textContent = "已停止生成。";
      }
      activeDebugBlock.textContent = "已手动停止流式输出。";
    } else {
      activeAssistantBubble.textContent = "请求失败，请稍后重试。";
      activeDebugBlock.textContent = "请求失败，无法获取调试信息。";
    }
  } finally {
    currentController = null;
    submitEl.disabled = false;
    stopEl.disabled = true;
  }
}

submitEl.addEventListener("click", askQuestion);
stopEl.addEventListener("click", () => {
  if (currentController) {
    currentController.abort();
  }
});
questionEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    askQuestion();
  }
});
loadMcpCapabilities();

