import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Role = "user" | "assistant";
type ChatItem = {
  id: string;
  role: Role;
  text: string;
  debug?: string;
};
type SessionState = {
  id: string;
  title: string;
  updatedAt: number;
  messages: ChatItem[];
};

type MCPToolItem = {
  name: string;
  category: string;
  status: "integrated" | "available";
  description: string;
};

type MCPCapabilitiesResponse = {
  enabled: boolean;
  repo_scope: string;
  tools: MCPToolItem[];
};

type DocMeta = {
  id?: number | null;
  slug?: string | null;
  title: string;
  url?: string | null;
  updated_at?: string | null;
};

type DocSuggestResponse = {
  docs: DocMeta[];
};

type SkillId =
  | "reading-digest"
  | "daily-capture"
  | "note-refine"
  | "knowledge-connect"
  | "style-extract"
  | "smart-search"
  | "smart-summary"
  | "stale-detector";

type SkillDef = {
  id: SkillId;
  name: string;
  description: string;
  triggers: string[];
};

type CapabilityDetail = {
  kind: "mcp" | "skill";
  name: string;
  description: string;
  triggers?: string[];
};

const skillDefs: SkillDef[] = [
  {
    id: "reading-digest",
    name: "reading-digest",
    description: "阅读后提取核心观点/金句/行动项，生成结构化阅读笔记（只读）。",
    triggers: ["阅读笔记", "金句", "行动项", "核心观点", "digest"],
  },
  {
    id: "daily-capture",
    name: "daily-capture",
    description: "把碎片想法/待办整理成结构化条目：标题/正文/标签/待办（只读）。",
    triggers: ["碎片", "捕获", "记录", "待办", "想法收集", "daily", "capture"],
  },
  {
    id: "note-refine",
    name: "note-refine",
    description: "润色打磨笔记：提升结构与表达（只读）。",
    triggers: ["润色", "打磨", "refine", "优化表达", "改写", "提高质量", "note-refine"],
  },
  {
    id: "knowledge-connect",
    name: "knowledge-connect",
    description: "分析多文档关联，输出主题簇与关联点（只读）。",
    triggers: ["关联", "联系", "聚类", "主题", "知识网络", "connect", "关联发现"],
  },
  {
    id: "style-extract",
    name: "style-extract",
    description: "从样本文档提炼写作风格画像：语气/句式/结构等（只读）。",
    triggers: ["风格", "用词", "句式", "表达习惯", "style", "画像", "style-extract"],
  },
  {
    id: "smart-search",
    name: "smart-search",
    description: "把检索到的候选上下文组织成可读搜索回答：候选标题 + 摘要（只读）。",
    triggers: ["搜索", "找", "在哪里", "文档在哪", "smart-search", "查找"],
  },
  {
    id: "smart-summary",
    name: "smart-summary",
    description: "按粒度生成摘要/概述：要点、详细段落；可根据“约100字”控制长度（只读）。",
    triggers: ["总结", "摘要", "概述", "要点", "大概100字", "约100字", "一句话", "详细总结", "smart-summary"],
  },
  {
    id: "stale-detector",
    name: "stale-detector",
    description: "过期检测：基于文档 updated_at 列出疑似过期候选，并给出更新建议（只读）。",
    triggers: ["过期", "陈旧", "检测", "stale", "更新建议", "健康度", "过期检测"],
  },
];

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

function buildWelcomeMessages(sessionId: string): ChatItem[] {
  return [
    {
      id: `welcome-${sessionId}`,
      role: "assistant",
      text: "你好，我是你的企业知识助手。请输入问题开始对话。",
      debug: "等待调试信息...",
    },
  ];
}

function touchSession(session: SessionState): SessionState {
  return { ...session, updatedAt: Date.now() };
}

function App() {
  const [question, setQuestion] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [selectedModel, setSelectedModel] = useState("deepseek-chat");
  const [selectedOwner, setSelectedOwner] = useState("fenyuansaki");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [activeSessionId, setActiveSessionId] = useState("default");
  const [sessions, setSessions] = useState<SessionState[]>([]);
  const [mcpData, setMcpData] = useState<MCPCapabilitiesResponse | null>(null);
  const [mcpError, setMcpError] = useState("");
  const [capabilityDetail, setCapabilityDetail] = useState<CapabilityDetail | null>(null);
  const [abilitiesCollapsed, setAbilitiesCollapsed] = useState(true);
  const [docSuggestOpen, setDocSuggestOpen] = useState(false);
  const [docSuggestDocs, setDocSuggestDocs] = useState<DocMeta[]>([]);
  const [docSuggestActiveIndex, setDocSuggestActiveIndex] = useState(0);
  const [selectedDocTitles, setSelectedDocTitles] = useState<string[]>([]);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const controllerRef = useRef<AbortController | null>(null);
  const chatListRef = useRef<HTMLDivElement | null>(null);
  const activeSessionRef = useRef(activeSessionId);
  const suggestDebounceTimerRef = useRef<number | null>(null);
  const latestSuggestReqIdRef = useRef(0);
  const docTokenRangeRef = useRef<{ start: number; end: number } | null>(null);

  const copyTextToClipboard = async (text: string) => {
    const safeText = text ?? "";
    if (!safeText) return;
    try {
      await navigator.clipboard.writeText(safeText);
      return;
    } catch (_e) {
      // 兼容：部分环境可能不支持 navigator.clipboard
      const ta = document.createElement("textarea");
      ta.value = safeText;
      ta.style.position = "fixed";
      ta.style.left = "-9999px";
      ta.style.top = "-9999px";
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    }
  };

  const copyMessage = async (messageId: string, text: string) => {
    await copyTextToClipboard(text);
    setCopiedMessageId(messageId);
    window.setTimeout(() => setCopiedMessageId((prev) => (prev === messageId ? null : prev)), 1200);
  };

  useEffect(() => {
    const saved = localStorage.getItem("rag_frontend_sessions_v1");
    if (!saved) {
      setSessions([
        { id: "default", title: "默认会话", updatedAt: Date.now(), messages: buildWelcomeMessages("default") },
      ]);
      setActiveSessionId("default");
      return;
    }
    try {
      const parsed = JSON.parse(saved) as { activeSessionId: string; sessions: SessionState[] };
      if (!parsed?.sessions?.length) throw new Error("empty");
      setSessions(parsed.sessions);
      setActiveSessionId(parsed.activeSessionId || parsed.sessions[0].id);
    } catch (_error) {
      setSessions([
        { id: "default", title: "默认会话", updatedAt: Date.now(), messages: buildWelcomeMessages("default") },
      ]);
      setActiveSessionId("default");
    }
  }, []);

  useEffect(() => {
    activeSessionRef.current = activeSessionId;
  }, [activeSessionId]);

  useEffect(() => {
    if (!sessions.length) return;
    localStorage.setItem(
      "rag_frontend_sessions_v1",
      JSON.stringify({ activeSessionId, sessions })
    );
  }, [sessions, activeSessionId]);

  useEffect(() => {
    const loadMcpCapabilities = async () => {
      try {
        const response = await fetch("/mcp/capabilities");
        const data: MCPCapabilitiesResponse = await response.json();
        setMcpData(data);
      } catch (_error) {
        setMcpError("MCP 能力读取失败");
      }
    };
    loadMcpCapabilities();
  }, []);

  const activeSession = useMemo(
    () => sessions.find((item) => item.id === activeSessionId) || null,
    [sessions, activeSessionId]
  );
  const chatItems = activeSession?.messages || [];

  useEffect(() => {
    chatListRef.current?.scrollTo({ top: chatListRef.current.scrollHeight, behavior: "smooth" });
  }, [chatItems, activeSessionId]);

  const mcpMeta = useMemo(() => {
    if (mcpError) return mcpError;
    if (!mcpData) return "加载中...";
    return `状态：${mcpData.enabled ? "已启用" : "未启用"} | 作用域：${mcpData.repo_scope || "未配置"}`;
  }, [mcpData, mcpError]);

  const ownerOptions = useMemo(() => {
    const fromScope = (mcpData?.repo_scope || "").split("/")[0];
    const values = [fromScope, selectedOwner].filter(Boolean);
    return Array.from(new Set(values));
  }, [mcpData, selectedOwner]);

  const closeDocSuggest = () => {
    setDocSuggestOpen(false);
    setDocSuggestDocs([]);
    docTokenRangeRef.current = null;
    setDocSuggestActiveIndex(0);
  };

  const removeSelectedDocTitle = (title: string) => {
    setSelectedDocTitles((prev) => prev.filter((t) => t !== title));
  };

  const extractAtToken = (text: string, cursorPos: number) => {
    const before = text.slice(0, cursorPos);
    // 匹配 1) 光标末尾仅为 '@'
    const m1 = before.match(/(^|\s)@$/);
    if (m1) {
      const wholeIndex = m1.index ?? 0;
      const leading = m1[1]; // '' 或 一个空白字符
      const atPos = leading ? wholeIndex + leading.length : wholeIndex;
      return { start: atPos, end: cursorPos, term: "" };
    }

    // 匹配 2) 光标末尾为 “行首/空白 + @ + 非空白字符” 的 @ 前缀
    const m2 = before.match(/(^|\s)@([^\s@]{1,80})$/);
    if (!m2) return null;
    const wholeIndex = m2.index ?? 0;
    const leading = m2[1]; // '' 或 一个空白字符
    const atPos = leading ? wholeIndex + leading.length : wholeIndex;
    const term = m2[2];
    return { start: atPos, end: cursorPos, term };
  };

  const handleQuestionChange = (event: any) => {
    const nextValue = event.target.value as string;
    const cursorPos = event.currentTarget.selectionStart ?? nextValue.length;
    setQuestion(nextValue);

    const at = extractAtToken(nextValue, cursorPos);
    if (!at) {
      closeDocSuggest();
      return;
    }

    docTokenRangeRef.current = { start: at.start, end: at.end };

    if (suggestDebounceTimerRef.current) window.clearTimeout(suggestDebounceTimerRef.current);
    const reqId = ++latestSuggestReqIdRef.current;

    suggestDebounceTimerRef.current = window.setTimeout(async () => {
      try {
        const resp = await fetch("/docs/suggest", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: at.term, owner: selectedOwner }),
        });
        if (!resp.ok) throw new Error("suggest_failed");
        const data = (await resp.json()) as DocSuggestResponse;
        if (latestSuggestReqIdRef.current !== reqId) return;
        const docs = data?.docs || [];
        setDocSuggestDocs(docs);
        setDocSuggestOpen(docs.length > 0);
        setDocSuggestActiveIndex(0);
      } catch (_e) {
        closeDocSuggest();
      }
    }, 250);
  };

  const stopStreaming = () => {
    controllerRef.current?.abort();
  };

  const createSession = () => {
    const id = `s-${Date.now()}`;
    const title = `新会话 ${sessions.length + 1}`;
    setSessions((prev) => [{ id, title, updatedAt: Date.now(), messages: buildWelcomeMessages(id) }, ...prev]);
    setActiveSessionId(id);
  };

  const renameSession = (sessionId: string) => {
    const current = sessions.find((s) => s.id === sessionId);
    if (!current) return;
    const next = window.prompt("请输入新的会话名称：", current.title)?.trim();
    if (!next) return;
    setSessions((prev) =>
      prev.map((session) => (session.id === sessionId ? { ...session, title: next, updatedAt: Date.now() } : session))
    );
  };

  const removeSession = (sessionId: string) => {
    if (sessions.length <= 1) return;
    if (!window.confirm("确认删除该会话吗？")) return;
    const nextSessions = sessions.filter((s) => s.id !== sessionId);
    setSessions(nextSessions);
    if (activeSessionId === sessionId) {
      setActiveSessionId(nextSessions[0].id);
    }
  };

  const groupedTools = useMemo(() => {
    const map = new Map<string, MCPToolItem[]>();
    for (const tool of mcpData?.tools || []) {
      const current = map.get(tool.category) || [];
      current.push(tool);
      map.set(tool.category, current);
    }
    return Array.from(map.entries());
  }, [mcpData]);

  const askQuestion = async () => {
    const text = question.trim();
    if (!text || isStreaming) return;
    // 允许用户使用 @ 选文档：后端基于文档标题做匹配时不需要这个前缀符号
    const payloadQuestion = text.replace(/(^|\s)@/g, "$1");
    const sessionId = activeSessionRef.current;
    const userId = `u-${Date.now()}`;
    const assistantId = `a-${Date.now()}`;
    setQuestion("");
    setSelectedDocTitles([]);
    closeDocSuggest();
    setIsStreaming(true);
    setSessions((prev) =>
      prev.map((session) =>
        session.id === sessionId
          ? {
                  ...touchSession(session),
              messages: [
                ...session.messages,
                { id: userId, role: "user", text },
                { id: assistantId, role: "assistant", text: "", debug: "正在获取调试信息..." },
              ],
            }
          : session
      )
    );

    const controller = new AbortController();
    controllerRef.current = controller;

    try {
      const response = await fetch("/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: payloadQuestion,
          model: selectedModel,
          owner: selectedOwner,
        }),
        signal: controller.signal,
      });
      if (!response.ok || !response.body) throw new Error("stream_unavailable");

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      let doneReceived = false;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() || "";
        for (const block of blocks) {
          if (!block.trim()) continue;
          const parsed = parseSseEvent(block);
          if (parsed.event === "token") {
            const payload = JSON.parse(parsed.data || "{}");
            const token = payload.token || "";
            setSessions((prev) =>
              prev.map((session) =>
                session.id === sessionId
                  ? {
                      ...touchSession(session),
                      messages: session.messages.map((item) =>
                        item.id === assistantId ? { ...item, text: item.text + token } : item
                      ),
                    }
                  : session
              )
            );
          } else if (parsed.event === "done") {
            doneReceived = true;
            const payload = JSON.parse(parsed.data || "{}");
            setSessions((prev) =>
              prev.map((session) =>
                session.id === sessionId
                  ? {
                      ...touchSession(session),
                  messages: session.messages.map((item) =>
                        item.id === assistantId
                          ? {
                              ...item,
                              text: payload.answer || item.text || "没有返回回答。",
                              debug: payload.debug
                                ? JSON.stringify(payload.debug, null, 2)
                            : "本次未返回 MCP 调试信息（可能未走 fallback）。",
                            }
                          : item
                      ),
                    }
                  : session
              )
            );
          } else if (parsed.event === "error") {
            const payload = JSON.parse(parsed.data || "{}");
            throw new Error(payload.message || "请求失败");
          }
        }
      }

      if (!doneReceived) {
        setSessions((prev) =>
          prev.map((session) =>
            session.id === sessionId
              ? {
                  ...touchSession(session),
                  messages: session.messages.map((item) =>
                    item.id === assistantId
                      ? {
                          ...item,
                          text: item.text || "请求失败，请稍后重试。",
                          debug: "流式输出中断，未收到完成事件。",
                        }
                      : item
                  ),
                }
              : session
          )
        );
      }
    } catch (error: any) {
      const errMsg = typeof error?.message === "string" ? error.message : "";
      setSessions((prev) =>
        prev.map((session) =>
          session.id === sessionId
            ? {
                ...touchSession(session),
                messages: session.messages.map((item) =>
                  item.id === assistantId
                    ? {
                        ...item,
                        text:
                          item.text ||
                          (error?.name === "AbortError"
                            ? "已停止生成。"
                            : errMsg || "请求失败，请稍后重试。"),
                        debug:
                          error?.name === "AbortError"
                            ? "已手动停止流式输出。"
                            : errMsg
                            ? `错误: ${errMsg}`
                            : "请求失败，无法获取调试信息。",
                      }
                    : item
                ),
              }
            : session
        )
      );
    } finally {
      controllerRef.current = null;
      setIsStreaming(false);
    }
  };

  return (
    <main className={`layout ${sidebarCollapsed ? "layout-collapsed" : ""}`}>
      <aside className={`sidebar ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
        <div className="sidebar-top">
          <div>
            <div className="brand">RAG 助手</div>
            {!sidebarCollapsed && <div className="sub">语雀 + MCP</div>}
          </div>
          <button
            type="button"
            className="ghost-btn"
            onClick={() => setSidebarCollapsed((prev) => !prev)}
            title={sidebarCollapsed ? "展开侧栏" : "收起侧栏"}
          >
            {sidebarCollapsed ? ">" : "<"}
          </button>
        </div>

        {!sidebarCollapsed && (
          <>
            <section className="panel">
              <div className="session-head">
                <h3>历史会话</h3>
                <button type="button" className="ghost-btn" onClick={createSession}>
                  +
                </button>
              </div>
              <ul className="tool-list">
                {[...sessions]
                  .sort((a, b) => b.updatedAt - a.updatedAt)
                  .map((session) => (
                  <li key={session.id}>
                    <div className="session-row">
                      <button
                        type="button"
                        className={`session-btn ${session.id === activeSessionId ? "active" : ""}`}
                        onClick={() => setActiveSessionId(session.id)}
                      >
                        {session.title}
                      </button>
                      <button type="button" className="ghost-btn small" onClick={() => renameSession(session.id)}>
                        改
                      </button>
                      <button
                        type="button"
                        className="ghost-btn small danger"
                        onClick={() => removeSession(session.id)}
                        disabled={sessions.length <= 1}
                      >
                        删
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            </section>

            <section className="panel">
              <div className="session-head">
                <h3>MCP & Skills 能力</h3>
                <button
                  type="button"
                  className="ghost-btn"
                  onClick={() => setAbilitiesCollapsed((prev) => !prev)}
                  title={abilitiesCollapsed ? "展开" : "收起"}
                >
                  {abilitiesCollapsed ? "展开" : "收起"}
                </button>
              </div>

              {!abilitiesCollapsed && (
                <>
                  <section className="panel abilities-subpanel">
                    <h3 style={{ marginTop: 0 }}>MCP 能力</h3>
                    <div className="meta">{mcpMeta}</div>
                    {groupedTools.map(([group, tools]) => (
                      <div key={group} className="tool-group">
                        <div className="tool-group-title">{group}</div>
                        <ul className="tool-list">
                          {tools.map((tool) => (
                            <li key={tool.name}>
                              <button
                                type="button"
                                className="capability-btn"
                                onClick={() =>
                                  setCapabilityDetail({
                                    kind: "mcp",
                                    name: tool.name,
                                    description: tool.description,
                                  })
                                }
                              >
                                <span
                                  className={`status-dot ${
                                    tool.status === "integrated" ? "status-dot-green" : "status-dot-red"
                                  }`}
                                  aria-hidden="true"
                                />
                                <span className="capability-btn-text">{tool.name}</span>
                              </button>
                            </li>
                          ))}
                        </ul>
                      </div>
                    ))}
                  </section>

                  <section className="panel abilities-subpanel">
                    <h3 style={{ marginTop: 0 }}>Skills 能力</h3>
                    <ul className="tool-list">
                      {skillDefs.map((skill) => (
                        <li key={skill.id}>
                          <button
                            type="button"
                            className="capability-btn"
                            onClick={() =>
                              setCapabilityDetail({
                                kind: "skill",
                                name: skill.name,
                                description: skill.description,
                                triggers: skill.triggers,
                              })
                            }
                          >
                            <span
                              className="status-dot status-dot-green"
                              aria-hidden="true"
                            />
                            <span className="capability-btn-text">{skill.name}</span>
                          </button>
                        </li>
                      ))}
                    </ul>
                  </section>
                </>
              )}
            </section>

            <section className="panel">
              <h3>能力说明</h3>
              {capabilityDetail ? (
                <div className="capability-detail">
                  <div className="capability-detail-title">{capabilityDetail.kind === "mcp" ? "MCP" : "Skill"}：{capabilityDetail.name}</div>
                  <div className="capability-detail-desc">{capabilityDetail.description}</div>
                  {capabilityDetail.triggers && capabilityDetail.triggers.length > 0 && (
                    <div className="capability-detail-triggers">
                      <div className="capability-detail-subtitle">触发关键词</div>
                      <div className="capability-detail-tags">
                        {capabilityDetail.triggers.map((t) => (
                          <span className="tag" key={t}>
                            {t}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="meta">点击左侧的 MCP 工具或 Skill 条目查看能力。</div>
              )}
            </section>
          </>
        )}
      </aside>

      <section className="chat-shell">
        <header className="topbar">企业级 RAG 问答系统</header>
        <div className="chat-list" ref={chatListRef}>
          {chatItems.map((item) => (
            <div className={`msg ${item.role}`} key={item.id}>
              <div style={{ width: "100%" }}>
                <div className="bubble">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{item.text}</ReactMarkdown>
                </div>
                <div className="msg-footer">
                  <button
                    type="button"
                    className={`copy-button ${copiedMessageId === item.id ? "copied" : ""}`}
                    onClick={() => void copyMessage(item.id, item.text)}
                    aria-label="复制对话内容"
                    title="复制"
                  >
                    <span className="copy-icon" aria-hidden="true">
                      ⧉
                    </span>
                  </button>
                </div>
                {item.role === "assistant" && <pre className="debug">{item.debug || "等待调试信息..."}</pre>}
              </div>
            </div>
          ))}
        </div>

        <section className="composer">
          <div className="composer-toolbar">
            <label className="field">
              <span>模型</span>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                disabled={isStreaming}
              >
                <option value="deepseek-chat">deepseek-chat</option>
                <option value="deepseek-reasoner">deepseek-reasoner</option>
                <option value="gpt-4o-mini">gpt-4o-mini</option>
              </select>
            </label>
            <label className="field">
              <span>知识库所有者</span>
              <select
                value={selectedOwner}
                onChange={(e) => setSelectedOwner(e.target.value)}
                disabled={isStreaming}
              >
                {ownerOptions.map((owner) => (
                  <option key={owner} value={owner}>
                    {owner}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="composer-input-row">
            <div className="doc-suggest-wrap">
              {docSuggestOpen && (
                <div className="doc-suggest-box">
                  {docSuggestDocs.map((doc, idx) => (
                    <button
                      key={doc.id ?? doc.slug ?? doc.title + idx}
                      type="button"
                      className={`doc-suggest-item ${idx === docSuggestActiveIndex ? "active" : ""}`}
                      onClick={() => {
                        const range = docTokenRangeRef.current;
                        if (!range) return;
                        const insert = doc.title;
                        const after = question.slice(range.end);
                        const needsSpace = after && !/^\s/.test(after);
                        const next =
                          question.slice(0, range.start) + insert + (needsSpace ? " " : "") + after;
                        setQuestion(next);
                        setSelectedDocTitles((prev) => (prev.includes(insert) ? prev : [...prev, insert]));
                        closeDocSuggest();
                      }}
                    >
                      {doc.title}
                    </button>
                  ))}
                </div>
              )}
              <textarea
                className="composer-textarea"
                rows={3}
                placeholder="输入问题，Shift+Enter 换行，Enter 发送（支持 @ 选择文档）"
                value={question}
                onChange={handleQuestionChange}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    void askQuestion();
                  }
                  if (event.key === "Escape") {
                    closeDocSuggest();
                  }
                }}
              />
              <button
                type="button"
                className={`send-button ${isStreaming ? "send-button-stop" : "send-button-send"}`}
                onClick={() => {
                  if (isStreaming) stopStreaming();
                  else void askQuestion();
                }}
                disabled={false}
                title={isStreaming ? "停止生成" : "发送"}
                aria-label={isStreaming ? "停止生成" : "发送"}
              >
                <span className={isStreaming ? "stop-icon" : "send-icon"} aria-hidden="true" />
              </button>
            </div>
          </div>
          {selectedDocTitles.length > 0 && (
            <div className="selected-docs">
              <div className="selected-docs-label">已选择文档：</div>
              <div className="selected-docs-chips">
                {selectedDocTitles.map((title) => (
                  <span className="doc-chip" key={title}>
                    {title}
                    <button
                      type="button"
                      className="doc-chip-x"
                      onClick={() => removeSelectedDocTitle(title)}
                      aria-label={`移除 ${title}`}
                    >
                      ×
                    </button>
                  </span>
                ))}
              </div>
            </div>
          )}
        </section>
      </section>
    </main>
  )
}

export default App
