import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { normalizeMarkdownAutolinks } from "./markdownAutolink";
import "./floatingWidget.css";

type Role = "user" | "assistant";

type Message = {
  id: string;
  role: Role;
  text: string;
};

type MCPCapabilitiesResponse = {
  repo_scope?: string;
};

const STREAM_READ_IDLE_MS = 120_000;
const STREAM_CONNECT_MS = 45_000;
const STREAM_FLUSH_MS = 48;
const STORAGE_DEBOUNCE_MS = 800;
const STORAGE_MESSAGES_KEY = "floating_widget_messages_v1";
const STORAGE_OPEN_KEY = "floating_widget_open_v1";

function createSessionId(): string {
  const uuid =
    typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `fw-${uuid}`;
}

function createMessageId(prefix: "u" | "a"): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
}

function parseSseEvent(block: string): { event: string; data: string } {
  const lines = block.split("\n");
  let event = "message";
  let data = "";
  for (const line of lines) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  return { event, data };
}

async function readStreamChunkWithIdle(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  idleMs: number,
  outerAbort: AbortController,
): Promise<ReadableStreamReadResult<Uint8Array>> {
  return new Promise((resolve, reject) => {
    let settled = false;
    const tid = window.setTimeout(() => {
      if (settled) return;
      settled = true;
      outerAbort.abort();
      void reader.cancel("idle_timeout").catch(() => {});
      reject(new Error("stream_idle_timeout"));
    }, idleMs);

    reader
      .read()
      .then((chunk) => {
        if (settled) return;
        settled = true;
        window.clearTimeout(tid);
        resolve(chunk);
      })
      .catch((err) => {
        if (settled) return;
        settled = true;
        window.clearTimeout(tid);
        reject(err);
      });
  });
}

function welcomeMessage(): Message {
  return {
    id: createMessageId("a"),
    role: "assistant",
    text: "您好，我是语雀智能销售助手。可以为您介绍产品能力、试用流程与报价方案。",
  };
}

function readStoredMessages(): Message[] {
  try {
    const raw = localStorage.getItem(STORAGE_MESSAGES_KEY);
    if (!raw) return [welcomeMessage()];
    const parsed = JSON.parse(raw) as Message[];
    if (!Array.isArray(parsed) || parsed.length === 0) return [welcomeMessage()];
    return parsed;
  } catch {
    return [welcomeMessage()];
  }
}

function readStoredOpen(): boolean {
  try {
    const raw = localStorage.getItem(STORAGE_OPEN_KEY);
    return raw === "0" ? false : true;
  } catch {
    return true;
  }
}

function FloatingWidgetApp() {
  const [open, setOpen] = useState<boolean>(() => readStoredOpen());
  const [messages, setMessages] = useState<Message[]>(() => readStoredMessages());
  const [question, setQuestion] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [owner, setOwner] = useState("");
  const [hint, setHint] = useState("");
  const [activeTab, setActiveTab] = useState<"human" | "ai">("ai");
  const chatListRef = useRef<HTMLDivElement | null>(null);
  const sessionIdRef = useRef(createSessionId());
  const controllerRef = useRef<AbortController | null>(null);
  const userStopRef = useRef(false);

  useEffect(() => {
    const tid = window.setTimeout(() => {
      localStorage.setItem(STORAGE_MESSAGES_KEY, JSON.stringify(messages));
    }, STORAGE_DEBOUNCE_MS);
    return () => window.clearTimeout(tid);
  }, [messages]);

  useEffect(() => {
    localStorage.setItem(STORAGE_OPEN_KEY, open ? "1" : "0");
  }, [open]);

  useEffect(() => {
    chatListRef.current?.scrollTo({ top: chatListRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, open]);

  useEffect(() => {
    const loadOwner = async () => {
      try {
        const response = await fetch("/mcp/capabilities");
        if (!response.ok) return;
        const data = (await response.json()) as MCPCapabilitiesResponse;
        const scope = (data.repo_scope || "").trim();
        const detectedOwner = scope.split("/")[0]?.trim() || "";
        setOwner(detectedOwner);
      } catch {
        // ignore
      }
    };
    void loadOwner();
  }, []);

  const canSend = useMemo(() => question.trim().length > 0 && !isStreaming, [question, isStreaming]);

  const stopStreaming = () => {
    userStopRef.current = true;
    controllerRef.current?.abort();
  };

  const handleSend = async () => {
    const text = question.trim();
    if (!text || isStreaming) return;
    const userMsg: Message = { id: createMessageId("u"), role: "user", text };
    const assistantMsg: Message = { id: createMessageId("a"), role: "assistant", text: "" };
    setQuestion("");
    setHint("");
    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setIsStreaming(true);
    userStopRef.current = false;

    const controller = new AbortController();
    controllerRef.current = controller;
    let connectTimer: number | null = null;
    let streamTextBuf = "";
    let streamFlushTimer: number | null = null;

    const flushStreamText = () => {
      if (!streamTextBuf) return;
      const chunk = streamTextBuf;
      streamTextBuf = "";
      setMessages((prev) =>
        prev.map((msg) => (msg.id === assistantMsg.id ? { ...msg, text: msg.text + chunk } : msg)),
      );
    };

    const scheduleStreamFlush = () => {
      if (streamFlushTimer != null) return;
      streamFlushTimer = window.setTimeout(() => {
        streamFlushTimer = null;
        flushStreamText();
      }, STREAM_FLUSH_MS);
    };

    try {
      connectTimer = window.setTimeout(() => controller.abort(), STREAM_CONNECT_MS);
      const response = await fetch("/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: text,
          model: "deepseek-chat",
          owner,
          token_profile: "primary",
          chat_mode: "visitor_sales",
          session_id: sessionIdRef.current,
          selected_yuque_docs: [],
        }),
        signal: controller.signal,
      });
      if (connectTimer != null) {
        window.clearTimeout(connectTimer);
        connectTimer = null;
      }
      if (!response.ok || !response.body) throw new Error("stream_unavailable");

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      let doneReceived = false;

      while (true) {
        const { value, done } = await readStreamChunkWithIdle(reader, STREAM_READ_IDLE_MS, controller);
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() || "";
        for (const block of blocks) {
          if (!block.trim()) continue;
          const parsed = parseSseEvent(block);
          if (parsed.event === "token") {
            const payload = JSON.parse(parsed.data || "{}") as { token?: string };
            const token = payload.token || "";
            if (!token) continue;
            streamTextBuf += token;
            scheduleStreamFlush();
          } else if (parsed.event === "done") {
            doneReceived = true;
            const payload = JSON.parse(parsed.data || "{}") as { answer?: string };
            const finalText = (payload.answer || "").trim();
            if (!finalText) continue;
            setMessages((prev) => prev.map((msg) => (msg.id === assistantMsg.id ? { ...msg, text: finalText } : msg)));
          } else if (parsed.event === "error") {
            const payload = JSON.parse(parsed.data || "{}") as { message?: string };
            throw new Error(payload.message || "请求失败，请稍后重试。");
          }
        }
      }

      if (!doneReceived) {
        setMessages((prev) =>
          prev.map((msg) => (msg.id === assistantMsg.id ? { ...msg, text: msg.text || "请求失败，请稍后重试。" } : msg)),
        );
      }
    } catch (error) {
      const err = error as { name?: string; message?: string };
      let fallback = "请求失败，请稍后重试。";
      if (err?.message === "stream_idle_timeout") {
        fallback = "等待超时：服务长时间未返回新内容，请稍后重试。";
      } else if (err?.name === "AbortError" && userStopRef.current) {
        fallback = "已停止生成。";
      } else if (err?.message) {
        fallback = err.message;
      }
      setHint(fallback);
      setMessages((prev) =>
        prev.map((msg) => (msg.id === assistantMsg.id ? { ...msg, text: msg.text || fallback } : msg)),
      );
    } finally {
      if (connectTimer != null) window.clearTimeout(connectTimer);
      if (streamFlushTimer != null) {
        window.clearTimeout(streamFlushTimer);
        streamFlushTimer = null;
      }
      flushStreamText();
      controllerRef.current = null;
      userStopRef.current = false;
      setIsStreaming(false);
    }
  };

  return (
    <div className="fw-page">
      <div className="fw-demo-tip">悬浮小窗模式（URL：`?ui=floating`）</div>
      {open ? (
        <section className="fw-widget" aria-label="客服聊天悬浮窗">
          <header className="fw-header">
            <div className="fw-header-title">客服聊天</div>
            <div className="fw-header-actions">
              <button type="button" className="fw-icon-btn" aria-label="语言切换">
                EN
              </button>
              <button type="button" className="fw-icon-btn" aria-label="语音">
                ♪
              </button>
              <button type="button" className="fw-icon-btn" aria-label="关闭" onClick={() => setOpen(false)}>
                ×
              </button>
            </div>
          </header>

          <div className="fw-tabs">
            <button
              type="button"
              className={`fw-tab${activeTab === "human" ? " active" : ""}`}
              onClick={() => {
                setActiveTab("human");
                setHint("当前版本先开放 AI 客服，人工客服入口稍后接入。");
              }}
            >
              人工客服
            </button>
            <button
              type="button"
              className={`fw-tab${activeTab === "ai" ? " active" : ""}`}
              onClick={() => {
                setActiveTab("ai");
                setHint("");
              }}
            >
              AI 客服
            </button>
          </div>

          <div className="fw-chat-list" ref={chatListRef}>
            {messages.map((msg) => (
              <div key={msg.id} className={`fw-msg ${msg.role}`}>
                <div className="fw-bubble">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{normalizeMarkdownAutolinks(msg.text)}</ReactMarkdown>
                </div>
              </div>
            ))}
          </div>

          <footer className="fw-composer">
            {hint ? <div className="fw-hint">{hint}</div> : null}
            <div className="fw-input-row">
              <button type="button" className="fw-lite-btn" aria-label="附件">
                📎
              </button>
              <input
                className="fw-input"
                placeholder="输入消息"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    if (isStreaming) stopStreaming();
                    else void handleSend();
                  }
                }}
              />
              <select className="fw-select" defaultValue="draw" aria-label="模式">
                <option value="chat">聊天</option>
                <option value="draw">绘画</option>
              </select>
              <button
                type="button"
                className={`fw-send-btn${isStreaming ? " stop" : ""}`}
                onClick={() => {
                  if (isStreaming) stopStreaming();
                  else void handleSend();
                }}
                disabled={!isStreaming && !canSend}
                title={isStreaming ? "停止生成" : "发送消息"}
              >
                {isStreaming ? "停止" : "↑"}
              </button>
            </div>
          </footer>
        </section>
      ) : null}

      <button
        type="button"
        className={`fw-launcher${open ? " fw-launcher-hidden" : ""}`}
        onClick={() => setOpen(true)}
        aria-label="打开客服聊天"
      >
        💬
      </button>
    </div>
  );
}

export default FloatingWidgetApp;
