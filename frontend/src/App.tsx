import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ChangeEvent } from "react";
import { createPortal } from "react-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { normalizeMarkdownAutolinks } from "./markdownAutolink";
import {
  INACTIVITY_MS,
  INACTIVITY_REMINDER_TEXT,
  VISITOR_WELCOME_TEXT,
  VISITOR_QUICK_QUESTIONS,
  looksLikeContactInUserMessage,
  looksLikeDeclineFollowup,
  visitorWelcomeMessages,
} from "./visitorSales";

/** 建立连接后首包 / 两次数据之间的最大等待（毫秒），超时则中止并提示 */
const STREAM_READ_IDLE_MS = 120_000;
/** 从发起请求到收到响应头的最长等待 */
const STREAM_CONNECT_MS = 45_000;
/** 超过该秒数后显示「可能较慢」说明 */
const STREAM_SLOW_HINT_SEC = 20;
const CHAT_V2_ENV_ENABLED = String(import.meta.env.VITE_CHAT_V2_ENABLED ?? "false").toLowerCase() === "true";
const CHAT_V3_ENV_ENABLED = String(import.meta.env.VITE_CHAT_V3_ENABLED ?? "false").toLowerCase() === "true";
const CHAT_V4_ENV_ENABLED = String(import.meta.env.VITE_CHAT_V4_ENABLED ?? "false").toLowerCase() === "true";
/** 是否为访客专用入口：/visitor 开头的路径 */
const IS_VISITOR_ROUTE =
  typeof window !== "undefined" && window.location && window.location.pathname.startsWith("/visitor");
const FOCUS_SCENE_ITEMS = ["人工智能通识教育", "智能招生", "跨学科项目化学习", "学校AI场景定制"] as const;
/** 开发者侧栏（运行追踪）；生产构建建议 VITE_SHOW_DEV_PANEL=false
 * 访客入口（/visitor）下强制关闭开发者面板。
 */
const SHOW_DEV_PANEL =
  !IS_VISITOR_ROUTE &&
  String(import.meta.env.VITE_SHOW_DEV_PANEL ?? "true").toLowerCase() === "true";

type TurnTraceMcpCall = {
  tool: string;
  query?: string;
  doc_id?: string;
  title?: string;
  hit_count?: number;
  body_chars?: number;
};

type TurnTraceSkill = {
  skill_id: string;
  reason?: string;
};

type TurnTraceDocument = {
  doc_id?: string;
  title?: string;
  role?: string;
  source_type?: string;
  snippet?: string;
};

type TurnTracePayload = {
  pipeline?: string;
  catalog_path?: string[];
  dialog_level?: number;
  mcp_calls?: TurnTraceMcpCall[];
  skills?: TurnTraceSkill[];
  documents?: TurnTraceDocument[];
};

function parseTurnTrace(debug: Record<string, unknown> | null | undefined): TurnTracePayload | null {
  const raw = debug?.turn_trace;
  if (!raw || typeof raw !== "object") return null;
  return raw as TurnTracePayload;
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

type Role = "user" | "assistant";

type ChatItem = {
  id: string;
  role: Role;
  text: string;
  media?: ChatMediaBundle;
  debug?: string;
  /** 流式阶段提示（来自 SSE event: stage），完成或出错后清除 */
  streamStage?: string;
  /** 本轮流式已等待秒数（前端计时，完成或出错后清除） */
  streamElapsedSec?: number;
  /** V4：是否展示「申请测试账号」按钮 */
  trialApplyAvailable?: boolean;
  trialCredentialsShown?: boolean;
};

type ChatMediaItem = {
  url: string;
  title?: string;
  doc_title?: string;
  doc_id?: string | null;
};

type ChatMediaBundle = {
  images: ChatMediaItem[];
  videos: ChatMediaItem[];
};

type SessionState = {
  id: string;
  title: string;
  updatedAt: number;
  messages: ChatItem[];
  /** 侧栏置顶排序用；旧版 localStorage 无此字段时视为未置顶 */
  pinned?: boolean;
  /** 访客销售：已识别到联系方式 */
  contactCollected?: boolean;
  /** 访客销售：用户明确拒绝后续联系 */
  declinedContact?: boolean;
  /** 访客销售：本会话已插入无互动留资提醒 */
  inactivityPromptSent?: boolean;
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
  /** 主 Token 调语雀 /user 得到的 login；侧栏展示优先用它，与 YUQUE_SCOPE 第一段可不同 */
  yuque_token_primary_login?: string;
  tools: MCPToolItem[];
};

type DocMeta = {
  id?: number | null;
  slug?: string | null;
  title: string;
  url?: string | null;
  updated_at?: string | null;
  toc_uuid?: string | null;
  toc_parent_uuid?: string | null;
  toc_level?: number | null;
  toc_kind?: string | null;
  toc_selectable?: boolean | null;
};

function docMetaSelectable(doc: DocMeta): boolean {
  return doc.toc_selectable !== false;
}

function normalizeTocLevel(level: number | null | undefined): number {
  return Math.max(1, Math.min(3, level ?? 1));
}

function firstSelectableDocIndex(docs: DocMeta[]): number {
  const i = docs.findIndex(docMetaSelectable);
  return i >= 0 ? i : 0;
}

function selectableDocIndices(docs: DocMeta[]): number[] {
  return docs.map((d, i) => (docMetaSelectable(d) ? i : -1)).filter((i) => i >= 0);
}

type DocSuggestResponse = {
  docs: DocMeta[];
};

/** 已选语雀文档：含 doc_id 供 POST /chat/stream 锚定；docId 为 0 表示仅有标题（不落 selected_yuque_docs） */
type SelectedYuqueDocLocal = {
  docId: number;
  title: string;
  slug?: string | null;
};

function upsertSelectedYuqueDoc(prev: SelectedYuqueDocLocal[], doc: DocMeta): SelectedYuqueDocLocal[] {
  if (!docMetaSelectable(doc)) return prev;
  const id = doc.id;
  if (typeof id === "number" && id >= 1) {
    if (prev.some((p) => p.docId === id)) return prev;
    return [...prev, { docId: id, title: doc.title, slug: doc.slug }];
  }
  if (prev.some((p) => p.title === doc.title)) return prev;
  return [...prev, { docId: 0, title: doc.title, slug: doc.slug }];
}

/** 检索阶段统一展示为产品文案；生成阶段仍用服务端 detail */
function formatStreamStageLine(stage: string, detail: string): string {
  if (stage === "retrieving") return "正在搜索知识库资料…";
  if (stage === "vision") return detail.trim() || "正在识读文档插图…";
  return detail.trim() || stage || "处理中…";
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

function parseChatMedia(input: unknown): ChatMediaBundle | undefined {
  if (!input || typeof input !== "object") return undefined;
  const raw = input as { images?: unknown; videos?: unknown };
  const normalize = (value: unknown): ChatMediaItem[] => {
    if (!Array.isArray(value)) return [];
    return value
      .map((item) => {
        if (!item || typeof item !== "object") return null;
        const x = item as Record<string, unknown>;
        const url = typeof x.url === "string" ? x.url.trim() : "";
        if (!url) return null;
        return {
          url,
          title: typeof x.title === "string" ? x.title : "",
          doc_title: typeof x.doc_title === "string" ? x.doc_title : "",
          doc_id: typeof x.doc_id === "string" ? x.doc_id : null,
        } as ChatMediaItem;
      })
      .filter((x): x is ChatMediaItem => Boolean(x));
  };
  const images = normalize(raw.images);
  const videos = normalize(raw.videos);
  if (images.length === 0 && videos.length === 0) return undefined;
  return { images, videos };
}

/** 访客销售：新会话首条为 AI 欢迎语 */
function emptySessionMessages(welcomeId: string): ChatItem[] {
  return visitorWelcomeMessages(welcomeId);
}

function formatRelativeTimeCN(ts: number): string {
  const diff = Date.now() - ts;
  if (diff < 60_000) return "刚刚";
  if (diff < 3600_000) return `${Math.floor(diff / 60_000)} 分钟前`;
  if (diff < 86400_000) return `${Math.floor(diff / 3600_000)} 小时前`;
  if (diff < 7 * 86400_000) return `${Math.floor(diff / 86400_000)} 天前`;
  return new Date(ts).toLocaleDateString("zh-CN", { month: "short", day: "numeric" });
}

function touchSession(session: SessionState): SessionState {
  return { ...session, updatedAt: Date.now() };
}

function generateSessionId(): string {
  const uuid =
    typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `s-${uuid}`;
}

const SESSIONS_STORAGE_KEY = "rag_frontend_sessions_v2";
const SESSION_TITLE_SEQ_KEY = "rag_frontend_session_title_seq_v1";
/** localStorage 最多保留会话数，避免长期堆积占满浏览器配额 */
const MAX_STORED_SESSIONS = 40;
/** 单会话最多持久化消息条数 */
const MAX_MESSAGES_PER_SESSION = 100;
/** 流式 token 批量刷 UI 间隔（毫秒），避免逐字 setState */
const STREAM_FLUSH_MS = 48;
/** 会话写入 localStorage 防抖（毫秒） */
const STORAGE_DEBOUNCE_MS = 800;

function sanitizeMessageForStorage(m: ChatItem): ChatItem {
  const copy = { ...m };
  delete copy.streamStage;
  delete copy.streamElapsedSec;
  return copy;
}

function sanitizeSessionsForStorage(
  sessions: SessionState[],
  activeSessionId: string,
): { activeSessionId: string; sessions: SessionState[] } {
  const byRecent = [...sessions].sort((a, b) => b.updatedAt - a.updatedAt);
  let trimmed = byRecent.slice(0, MAX_STORED_SESSIONS);
  if (activeSessionId && !trimmed.some((s) => s.id === activeSessionId)) {
    const active = sessions.find((s) => s.id === activeSessionId);
    if (active) {
      trimmed = [active, ...trimmed.slice(0, MAX_STORED_SESSIONS - 1)];
    }
  }
  return {
    activeSessionId,
    sessions: trimmed.map((s) => ({
      ...s,
      messages: s.messages.slice(-MAX_MESSAGES_PER_SESSION).map(sanitizeMessageForStorage),
    })),
  };
}

function persistSessionsToStorage(sessions: SessionState[], activeSessionId: string): void {
  if (!sessions.length) return;
  const payload = sanitizeSessionsForStorage(sessions, activeSessionId);
  try {
    localStorage.setItem(SESSIONS_STORAGE_KEY, JSON.stringify(payload));
  } catch {
    // 配额不足时再裁一轮
    const emergency = {
      activeSessionId: payload.activeSessionId,
      sessions: payload.sessions.slice(0, Math.max(5, Math.floor(MAX_STORED_SESSIONS / 2))).map((s) => ({
        ...s,
        messages: s.messages.slice(-30).map(sanitizeMessageForStorage),
      })),
    };
    try {
      localStorage.setItem(SESSIONS_STORAGE_KEY, JSON.stringify(emergency));
    } catch {
      /* 仍失败则放弃写入，避免拖垮页面 */
    }
  }
}

function normalizeSessionForVisitor(s: SessionState): SessionState {
  const flags = {
    contactCollected: s.contactCollected ?? false,
    declinedContact: s.declinedContact ?? false,
    inactivityPromptSent: s.inactivityPromptSent ?? false,
  };
  if (s.messages && s.messages.length > 0) {
    return { ...s, ...flags };
  }
  const wid = `welcome-${s.id}`;
  return { ...s, messages: emptySessionMessages(wid), ...flags };
}

function formatUnknownDebug(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "string" || typeof v === "number" || typeof v === "boolean") return String(v);
  try {
    const s = JSON.stringify(v);
    return s.length > 240 ? `${s.slice(0, 240)}…` : s;
  } catch {
    return String(v);
  }
}

function formatSessionTranscript(session: SessionState): string {
  const title = (session.title || "会话").trim();
  const lines: string[] = [
    `「${title}」· 有为销售顾问对话记录`,
    `导出时间：${new Date().toLocaleString("zh-CN", { hour12: false })}`,
    "",
  ];
  let hasContent = false;
  for (const item of session.messages) {
    const text = (item.text || "").trim();
    const images = item.media?.images ?? [];
    const videos = item.media?.videos ?? [];
    if (!text && images.length === 0 && videos.length === 0) {
      continue;
    }
    hasContent = true;
    const roleLabel = item.role === "user" ? "访客" : "顾问";
    lines.push(`${roleLabel}：`);
    if (text) {
      lines.push(text);
    }
    if (images.length > 0) {
      lines.push(`[相关图片 ${images.length} 张]`);
      images.forEach((img, idx) => {
        const label = (img.title || img.doc_title || "图片").trim();
        lines.push(`  ${idx + 1}. ${label}: ${img.url}`);
      });
    }
    if (videos.length > 0) {
      lines.push(`[相关视频 ${videos.length} 个]`);
      videos.forEach((video, idx) => {
        const label = (video.title || video.doc_title || "视频").trim();
        lines.push(`  ${idx + 1}. ${label}: ${video.url}`);
      });
    }
    lines.push("");
  }
  if (!hasContent) {
    return "";
  }
  return lines.join("\n").trimEnd();
}

function readStoredSessionState(): { sessions: SessionState[]; activeSessionId: string } {
  const fallback = (): { sessions: SessionState[]; activeSessionId: string } => ({
    sessions: [
      {
        id: "default",
        title: "默认会话",
        updatedAt: Date.now(),
        messages: emptySessionMessages("welcome-default"),
        ...{ contactCollected: false, declinedContact: false, inactivityPromptSent: false },
      },
    ],
    activeSessionId: "default",
  });
  try {
    let raw = localStorage.getItem(SESSIONS_STORAGE_KEY);
    if (!raw) {
      const legacy = localStorage.getItem("rag_frontend_sessions_v1");
      if (legacy) {
        raw = legacy;
      }
    }
    if (!raw) return fallback();
    const parsed = JSON.parse(raw) as { activeSessionId?: string; sessions?: SessionState[] };
    if (!parsed?.sessions?.length) return fallback();
    // 迁移：旧版本会话 id 可能是 s-<timestamp>，容易碰撞并串到服务端历史。
    // 对“未真正开始聊天”的会话（无 user 消息），自动换新 id，确保新会话一定从零开始。
    const idMap = new Map<string, string>();
    const sessions = parsed.sessions.map((x) => {
      const normalized = normalizeSessionForVisitor(x);
      const hasUser = normalized.messages.some((m) => m.role === "user" && (m.text || "").trim());
      if (hasUser) return normalized;
      const oldId = normalized.id;
      if (oldId === "default") return normalized;
      const newId = generateSessionId();
      idMap.set(oldId, newId);
      return {
        ...normalized,
        id: newId,
        messages: emptySessionMessages(`welcome-${newId}`),
      };
    });
    const mappedActive = idMap.get(parsed.activeSessionId || "") || parsed.activeSessionId;
    return {
      sessions,
      activeSessionId: mappedActive || sessions[0].id,
    };
  } catch {
    return fallback();
  }
}

function App() {
  const [chatV2Enabled, setChatV2Enabled] = useState(CHAT_V2_ENV_ENABLED);
  const [chatV3Enabled, setChatV3Enabled] = useState(CHAT_V3_ENV_ENABLED);
  const [chatV4Enabled, setChatV4Enabled] = useState(CHAT_V4_ENV_ENABLED);
  const [question, setQuestion] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [historySessionPicked, setHistorySessionPicked] = useState(false);
  const [selectedModel, setSelectedModel] = useState("deepseek-chat");
  /** 左侧开发者面板收起（仅保留展开条） */
  const [devSidebarCollapsed, setDevSidebarCollapsed] = useState(false);
  const [activeSessionId, setActiveSessionId] = useState(() => readStoredSessionState().activeSessionId);
  const [sessions, setSessions] = useState<SessionState[]>(() => readStoredSessionState().sessions);
  /** 历史会话行「⋯」溢出菜单：同时只展开一行 */
  const [openSessionMenuId, setOpenSessionMenuId] = useState<string | null>(null);
  const [mcpData, setMcpData] = useState<MCPCapabilitiesResponse | null>(null);
  const [docSuggestOpen, setDocSuggestOpen] = useState(false);
  const [docSuggestDocs, setDocSuggestDocs] = useState<DocMeta[]>([]);
  const [docSuggestActiveIndex, setDocSuggestActiveIndex] = useState(0);
  const [selectedYuqueDocs, setSelectedYuqueDocs] = useState<SelectedYuqueDocLocal[]>([]);
  const [kbPanelLoading, setKbPanelLoading] = useState(false);
  const [kbPanelError, setKbPanelError] = useState("");
  const [kbPanelDocs, setKbPanelDocs] = useState<DocMeta[]>([]);
  const [collapsedKbNodeIds, setCollapsedKbNodeIds] = useState<Set<string>>(new Set());
  /** 最近一次完成的 RAG / 检索 debug（SSE done） */
  const [lastPipelineDebug, setLastPipelineDebug] = useState<Record<string, unknown> | null>(null);
  /** 最近一次请求携带的模型、作用域等（便于与 debug 对照） */
  const [lastRequestMeta, setLastRequestMeta] = useState<{
    model: string;
    owner: string;
    chat_mode: string;
    token_profile: string;
    stream_path: string;
  } | null>(null);
  const [composerDocGateHint, setComposerDocGateHint] = useState("");
  const [activeFocusScene, setActiveFocusScene] = useState<(typeof FOCUS_SCENE_ITEMS)[number]>(FOCUS_SCENE_ITEMS[0]);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [sessionCopied, setSessionCopied] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);
  /** 用户点击「停止」触发的 abort，与超时/空闲 abort 区分 */
  const userStreamStopRef = useRef(false);
  const chatListRef = useRef<HTMLDivElement | null>(null);
  const [streamingAssistantId, setStreamingAssistantId] = useState<string | null>(null);
  const activeSessionRef = useRef(activeSessionId);
  const questionRef = useRef(question);
  const isStreamingRef = useRef(isStreaming);
  const historyHydratedRef = useRef<Set<string>>(new Set());
  const resetPendingRef = useRef<Map<string, Promise<void>>>(new Map());
  const sessionTitleSeqRef = useRef<number | null>(null);
  const [inactivityEpoch, setInactivityEpoch] = useState(0);
  const suggestDebounceTimerRef = useRef<number | null>(null);
  const latestSuggestReqIdRef = useRef(0);
  const docTokenRangeRef = useRef<{ start: number; end: number } | null>(null);
  const messageIdSeqRef = useRef(0);
  const floatingInitDoneRef = useRef(false);
  const composerInputWrapRef = useRef<HTMLDivElement | null>(null);
  const composerTextareaRef = useRef<HTMLTextAreaElement | null>(null);
  const docFloatRootRef = useRef<HTMLDivElement | null>(null);
  type DocFloatRect = { bottom: number; left: number; width: number; maxHeight: number };
  const [docFloatRect, setDocFloatRect] = useState<DocFloatRect | null>(null);

  const copyTextToClipboard = async (text: string) => {
    const safeText = text ?? "";
    if (!safeText) return;
    try {
      await navigator.clipboard.writeText(safeText);
      return;
    } catch {
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
    activeSessionRef.current = activeSessionId;
  }, [activeSessionId]);

  const setActiveSession = useCallback((sid: string) => {
    activeSessionRef.current = sid;
    setActiveSessionId(sid);
    // 前端兜底：切换到“未开始聊天”的会话时，强制只显示欢迎语，避免 UI 残留造成“看起来复制了其它会话历史”。
    setSessions((prev) =>
      prev.map((s) => {
        if (s.id !== sid) return s;
        const hasUser = s.messages.some((m) => m.role === "user" && (m.text || "").trim());
        if (hasUser) return s;
        return { ...touchSession(s), messages: emptySessionMessages(`welcome-${sid}`) };
      }),
    );
  }, []);

  useEffect(() => {
    questionRef.current = question;
  }, [question]);

  useEffect(() => {
    isStreamingRef.current = isStreaming;
  }, [isStreaming]);

  useEffect(() => {
    const sid = activeSessionId;
    const timer = window.setTimeout(() => {
      if (isStreamingRef.current) return;
      if (questionRef.current.trim()) return;
      setSessions((prev) => {
        const s = prev.find((x) => x.id === sid);
        if (!s) return prev;
        if (s.inactivityPromptSent || s.contactCollected || s.declinedContact) return prev;
        if (!s.messages.some((m) => m.role === "user")) return prev;
        const aid = `inact-${Date.now()}`;
        return prev.map((sess) =>
          sess.id !== sid
            ? sess
            : {
                ...touchSession(sess),
                inactivityPromptSent: true,
                messages: [
                  ...sess.messages,
                  { id: aid, role: "assistant", text: INACTIVITY_REMINDER_TEXT },
                ],
              },
        );
      });
    }, INACTIVITY_MS);
    return () => window.clearTimeout(timer);
  }, [inactivityEpoch, activeSessionId]);

  useEffect(() => {
    if (!sessions.length) return;
    if (isStreaming) return;
    const tid = window.setTimeout(() => {
      persistSessionsToStorage(sessions, activeSessionId);
    }, STORAGE_DEBOUNCE_MS);
    return () => window.clearTimeout(tid);
  }, [sessions, activeSessionId, isStreaming]);

  useEffect(() => {
    const sid = activeSessionId;
    if (!sid) return;
    if (isStreamingRef.current) return;
    if (historyHydratedRef.current.has(sid)) return;
    const session = sessions.find((s) => s.id === sid);
    if (!session) return;
    const hasUser = session.messages.some((m) => m.role === "user");
    if (hasUser) {
      historyHydratedRef.current.add(sid);
      return;
    }

    const run = async () => {
      try {
        const resp = await fetch(`/chat/history?session_id=${encodeURIComponent(sid)}&limit=40`);
        if (!resp.ok) return;
        const json = (await resp.json()) as {
          session_id?: string;
          messages?: { role: "user" | "assistant"; text: string; created_at: string }[];
        };
        const msgs = Array.isArray(json?.messages) ? json.messages : [];
        if (!msgs.length) return;
        const restored: ChatItem[] = msgs.map((m, idx) => ({
          id: `srv-${m.created_at || "t"}-${idx}`,
          role: m.role,
          text: m.text || "",
        }));
        setSessions((prev) => prev.map((s) => (s.id === sid ? { ...touchSession(s), messages: restored } : s)));
      } catch {
        // ignore
      } finally {
        historyHydratedRef.current.add(sid);
      }
    };
    void run();
  }, [activeSessionId, sessions]);

  useEffect(() => {
    const loadMcpCapabilities = async () => {
      try {
        const response = await fetch("/mcp/capabilities");
        const data: MCPCapabilitiesResponse = await response.json();
        setMcpData(data);
      } catch {
        /* ignore */
      }
    };
    void loadMcpCapabilities();
    const onVis = () => {
      if (document.visibilityState === "visible") void loadMcpCapabilities();
    };
    document.addEventListener("visibilitychange", onVis);
    return () => document.removeEventListener("visibilitychange", onVis);
  }, []);

  useEffect(() => {
    let cancelled = false;
    const detectV2Availability = async () => {
      try {
        const resp = await fetch("/chat/v2/guide-titles");
        if (!resp.ok) {
          // 后端未开启或不存在该路由时，回退到环境变量控制
          if (!cancelled) setChatV2Enabled(CHAT_V2_ENV_ENABLED);
          return;
        }
        const data = (await resp.json()) as { v15_enabled?: boolean };
        if (cancelled) return;
        // 只要后端确认 V1.5 已开启，前端优先走 /chat/v2/stream
        if (data?.v15_enabled === true) {
          setChatV2Enabled(true);
          return;
        }
        setChatV2Enabled(CHAT_V2_ENV_ENABLED);
      } catch {
        if (!cancelled) setChatV2Enabled(CHAT_V2_ENV_ENABLED);
      }
    };
    void detectV2Availability();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const detectV3Availability = async () => {
      try {
        const resp = await fetch("/chat/v3/capabilities");
        if (!resp.ok) {
          if (!cancelled) setChatV3Enabled(CHAT_V3_ENV_ENABLED);
          return;
        }
        const data = (await resp.json()) as { enabled?: boolean };
        if (cancelled) return;
        if (data?.enabled === true) {
          setChatV3Enabled(true);
          return;
        }
        setChatV3Enabled(CHAT_V3_ENV_ENABLED);
      } catch {
        if (!cancelled) setChatV3Enabled(CHAT_V3_ENV_ENABLED);
      }
    };
    void detectV3Availability();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const detectV4Availability = async () => {
      try {
        const resp = await fetch("/chat/v4/capabilities");
        if (!resp.ok) {
          if (!cancelled) setChatV4Enabled(CHAT_V4_ENV_ENABLED);
          return;
        }
        const data = (await resp.json()) as { enabled?: boolean };
        if (cancelled) return;
        if (data?.enabled === true) {
          setChatV4Enabled(true);
          return;
        }
        setChatV4Enabled(CHAT_V4_ENV_ENABLED);
      } catch {
        if (!cancelled) setChatV4Enabled(CHAT_V4_ENV_ENABLED);
      }
    };
    void detectV4Availability();
    return () => {
      cancelled = true;
    };
  }, []);

  const activeSession = useMemo(
    () => sessions.find((item) => item.id === activeSessionId) || null,
    [sessions, activeSessionId]
  );
  const chatItems = useMemo(() => activeSession?.messages ?? [], [activeSession]);
  const sessionTranscript = useMemo(
    () => (activeSession ? formatSessionTranscript(activeSession) : ""),
    [activeSession],
  );
  const copySessionTranscript = useCallback(async () => {
    if (!sessionTranscript) return;
    await copyTextToClipboard(sessionTranscript);
    setSessionCopied(true);
    window.setTimeout(() => setSessionCopied(false), 1500);
  }, [sessionTranscript]);

  const requestTrialCredentials = useCallback(async (assistantId: string) => {
    const sid = activeSessionRef.current;
    if (!sid) return;
    try {
      const resp = await fetch("/chat/v4/trial-credentials", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sid }),
      });
      const data = (await resp.json()) as {
        ok?: boolean;
        username?: string;
        password?: string;
        label?: string;
        message?: string;
      };
      const block = data.ok
        ? `\n\n【试用账号】\n账号：${data.username || ""}\n密码：${data.password || ""}${
            data.label ? `\n说明：${data.label}` : ""
          }`
        : `\n\n${data.message || "暂时无法获取试用账号，请稍后再试。"}`;
      setSessions((prev) =>
        prev.map((session) =>
          session.id === sid
            ? {
                ...touchSession(session),
                messages: session.messages.map((item) =>
                  item.id === assistantId
                    ? {
                        ...item,
                        text: (item.text || "") + block,
                        trialApplyAvailable: false,
                        trialCredentialsShown: true,
                      }
                    : item
                ),
              }
            : session
        )
      );
    } catch {
      setSessions((prev) =>
        prev.map((session) =>
          session.id === sid
            ? {
                ...touchSession(session),
                messages: session.messages.map((item) =>
                  item.id === assistantId
                    ? {
                        ...item,
                        text: `${item.text || ""}\n\n获取试用账号失败，请稍后重试。`,
                      }
                    : item
                ),
              }
            : session
        )
      );
    }
  }, []);
  /** 默认优先展示“新对话欢迎屏”；只有选择历史会话后，才按历史消息渲染 */
  const showWelcomeHero = useMemo(() => {
    if (historySessionPicked) return false;
    return !chatItems.some((item) => item.role === "user" && (item.text || "").trim());
  }, [chatItems, historySessionPicked]);

  const newChatShortcutLabel = useMemo(
    () =>
      typeof navigator !== "undefined" && /Mac|iPhone|iPod|iPad/i.test(navigator.platform || "")
        ? "⌘K"
        : "Ctrl+K",
    []
  );

  useEffect(() => {
    chatListRef.current?.scrollTo({ top: chatListRef.current.scrollHeight, behavior: "smooth" });
  }, [chatItems, activeSessionId]);

  const activeRepoScope = useMemo(() => (mcpData?.repo_scope || "").trim(), [mcpData]);

  const yuqueOwnerForApi = useMemo(() => {
    const fromScope = (activeRepoScope || "").split("/")[0]?.trim() || "";
    return fromScope.trim();
  }, [activeRepoScope]);

  /** 侧栏头像/名称：优先当前 Token 对应语雀账号（/user），与请求里 owner（来自作用域）可分离 */
  const yuqueSidebarLabel = useMemo(() => {
    const fromUser = (mcpData?.yuque_token_primary_login || "").trim();
    if (fromUser) return fromUser;
    return yuqueOwnerForApi;
  }, [mcpData, yuqueOwnerForApi]);

  const loadKbToc = useCallback(async () => {
    setKbPanelLoading(true);
    setKbPanelError("");
    const controller = new AbortController();
    const tid = window.setTimeout(() => controller.abort(), 45_000);
    try {
      const qs = new URLSearchParams();
      qs.set("owner", yuqueOwnerForApi);
      qs.set("token_profile", "primary");
      const resp = await fetch(`/docs/toc?${qs.toString()}`, {
        signal: controller.signal,
      });
      if (!resp.ok) {
        let detail = "";
        try {
          const j = (await resp.json()) as { detail?: string };
          if (typeof j?.detail === "string") detail = j.detail;
        } catch {
          /* ignore */
        }
        setKbPanelDocs([]);
        setKbPanelError(
          detail
            ? `目录加载失败：${detail}`
            : `目录加载失败（HTTP ${resp.status}）。请检查 Token 与知识库所有者。`,
        );
        return;
      }
      const data = (await resp.json()) as DocSuggestResponse;
      setKbPanelDocs(data?.docs || []);
    } catch (e) {
      setKbPanelDocs([]);
      if ((e as Error)?.name === "AbortError") {
        setKbPanelError("目录加载超时，请检查网络或语雀服务。");
      } else {
        setKbPanelError("目录请求异常，请稍后重试。");
      }
    } finally {
      window.clearTimeout(tid);
      setKbPanelLoading(false);
    }
  }, [yuqueOwnerForApi]);

  useEffect(() => {
    if (!yuqueOwnerForApi) return;
    const tid = window.setTimeout(() => {
      void loadKbToc();
    }, 0);
    return () => window.clearTimeout(tid);
  }, [yuqueOwnerForApi, loadKbToc]);

  const closeDocSuggest = useCallback(() => {
    setDocSuggestOpen(false);
    setDocSuggestDocs([]);
    docTokenRangeRef.current = null;
    setDocSuggestActiveIndex(0);
  }, []);

  const updateDocFloatRect = useCallback(() => {
    const showSuggest = docSuggestOpen && docSuggestDocs.length > 0;
    if (!showSuggest) {
      setDocFloatRect(null);
      return;
    }
    const anchorEl = composerInputWrapRef.current;
    if (!anchorEl) {
      setDocFloatRect(null);
      return;
    }
    const r = anchorEl.getBoundingClientRect();
    const margin = 8;
    const w = Math.max(r.width, 280);
    let left = r.left;
    if (left + w > window.innerWidth - margin) left = Math.max(margin, window.innerWidth - margin - w);
    if (left < margin) left = margin;
    /** 悬浮层贴在锚点上方：用 fixed + bottom，使列表底部与输入区顶边留出 margin */
    const bottom = window.innerHeight - r.top + margin;
    const maxHeight = Math.min(360, Math.max(120, Math.floor(r.top - margin - 8)));
    setDocFloatRect({ bottom, left, width: w, maxHeight });
  }, [docSuggestOpen, docSuggestDocs.length]);

  /* Floating 层位置依赖 DOM 测量；此处同步 setState 为既有交互模式 */
  /* eslint-disable react-hooks/set-state-in-effect */
  useLayoutEffect(() => {
    const open = docSuggestOpen && docSuggestDocs.length > 0;
    if (!open) {
      const tid = window.setTimeout(() => setDocFloatRect(null), 0);
      return () => window.clearTimeout(tid);
    }
    updateDocFloatRect();
    const ro = new ResizeObserver(() => {
      updateDocFloatRect();
    });
    const watch = [composerInputWrapRef.current].filter(Boolean);
    watch.forEach((el) => ro.observe(el!));
    window.addEventListener("resize", updateDocFloatRect);
    window.addEventListener("scroll", updateDocFloatRect, true);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", updateDocFloatRect);
      window.removeEventListener("scroll", updateDocFloatRect, true);
    };
  }, [docSuggestOpen, docSuggestDocs.length, updateDocFloatRect]);
  /* eslint-enable react-hooks/set-state-in-effect */

  /** 输入框自适应高度：默认 44px（与发送按钮同高），随内容自动撑高至 max-height（CSS），超出后内部滚动 */
  useLayoutEffect(() => {
    const el = composerTextareaRef.current;
    if (!el) return;
    el.style.height = "44px";
    const next = Math.max(44, el.scrollHeight);
    el.style.height = `${next}px`;
  }, [question]);

  useEffect(() => {
    const open = docSuggestOpen && docSuggestDocs.length > 0;
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      const t = e.target as Node | null;
      if (!t) return;
      if (docFloatRootRef.current?.contains(t)) return;
      if (composerInputWrapRef.current?.contains(t)) return;
      closeDocSuggest();
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [docSuggestOpen, docSuggestDocs.length, closeDocSuggest]);

  useEffect(() => {
    const open = docSuggestOpen && docSuggestDocs.length > 0;
    if (!open) return;
    const onEsc = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      closeDocSuggest();
    };
    document.addEventListener("keydown", onEsc);
    return () => document.removeEventListener("keydown", onEsc);
  }, [docSuggestOpen, docSuggestDocs.length, closeDocSuggest]);

  const applyDocSuggestPick = (doc: DocMeta) => {
    if (!docMetaSelectable(doc)) return;
    const range = docTokenRangeRef.current;
    if (!range) return;
    const insert = doc.title;
    const after = question.slice(range.end);
    const needsSpace = after && !/^\s/.test(after);
    const next = question.slice(0, range.start) + insert + (needsSpace ? " " : "") + after;
    setQuestion(next);
    setSelectedYuqueDocs((prev) => upsertSelectedYuqueDoc(prev, doc));
    setComposerDocGateHint("");
    closeDocSuggest();
  };

  const removeSelectedYuqueDoc = (docId: number, title: string) => {
    setSelectedYuqueDocs((prev) => prev.filter((p) => !(p.docId === docId && p.title === title)));
  };

  const pickDocFromKbPanel = (doc: DocMeta) => {
    if (!docMetaSelectable(doc)) return;
    setSelectedYuqueDocs((prev) => upsertSelectedYuqueDoc(prev, doc));
    setComposerDocGateHint("");
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

  const handleQuestionChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
    setComposerDocGateHint("");
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
          body: JSON.stringify({ query: at.term, owner: yuqueOwnerForApi, token_profile: "primary" }),
        });
        if (!resp.ok) {
          let detail = "";
          try {
            const j = (await resp.json()) as { detail?: string };
            if (typeof j?.detail === "string") detail = j.detail;
          } catch {
            /* ignore */
          }
          if (latestSuggestReqIdRef.current !== reqId) return;
          setComposerDocGateHint(
            detail
              ? `文档联想失败：${detail}`
              : `文档联想失败（HTTP ${resp.status}）。请检查 Token、YUQUE_SCOPE 与「知识库所有者」是否与语雀一致。`,
          );
          closeDocSuggest();
          return;
        }
        const data = (await resp.json()) as DocSuggestResponse;
        if (latestSuggestReqIdRef.current !== reqId) return;
        const docs = data?.docs || [];
        setDocSuggestDocs(docs);
        setDocSuggestOpen(docs.length > 0);
        setDocSuggestActiveIndex(firstSelectableDocIndex(docs));
      } catch {
        if (latestSuggestReqIdRef.current !== reqId) return;
        setComposerDocGateHint("文档联想请求异常，请检查网络或稍后重试。");
        closeDocSuggest();
      }
    }, 250);
  };

  const stopStreaming = () => {
    userStreamStopRef.current = true;
    controllerRef.current?.abort();
  };

  const nextSessionTitle = useCallback((): string => {
    if (sessionTitleSeqRef.current === null) {
      let raw = "0";
      try {
        raw = localStorage.getItem(SESSION_TITLE_SEQ_KEY) || "0";
      } catch {
        /* ignore */
      }
      const n = Number.parseInt(raw, 10);
      sessionTitleSeqRef.current = Number.isFinite(n) && n >= 0 ? n : 0;
    }
    const next = (sessionTitleSeqRef.current || 0) + 1;
    sessionTitleSeqRef.current = next;
    try {
      localStorage.setItem(SESSION_TITLE_SEQ_KEY, String(next));
    } catch {
      /* ignore */
    }
    return `新会话 #${next}`;
  }, []);

  const createSession = useCallback(() => {
    setHistorySessionPicked(false);
    const id = generateSessionId();
    // 新建会话必须“从零开始”：禁止用服务端历史覆盖该会话的欢迎语
    historyHydratedRef.current.add(id);
    setSessions((prev) => {
      const title = nextSessionTitle();
      return [{ id, title, updatedAt: Date.now(), messages: emptySessionMessages(`welcome-${id}`) }, ...prev];
    });
    setActiveSession(id);
    // 服务端兜底：强制重置该 session，避免任何串话（即便曾经碰撞/复用过）
    const p = fetch("/chat/session/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: id, chat_mode: "visitor_sales" }),
    })
      .then(() => {})
      .catch(() => {})
      .finally(() => {
        resetPendingRef.current.delete(id);
      });
    resetPendingRef.current.set(id, p);
  }, [nextSessionTitle, setActiveSession]);

  useEffect(() => {
    if (floatingInitDoneRef.current) return;
    if (!activeSession) return;
    floatingInitDoneRef.current = true;
    const hasUser = activeSession.messages.some((m) => m.role === "user" && (m.text || "").trim());
    if (hasUser) {
      const tid = window.setTimeout(() => {
        createSession();
      }, 0);
      return () => window.clearTimeout(tid);
    }
  }, [activeSession, createSession]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        createSession();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [createSession]);

  useEffect(() => {
    if (!openSessionMenuId) return;
    const close = (e: MouseEvent) => {
      const t = e.target as HTMLElement | null;
      if (t?.closest?.("[data-session-menu-root]")) return;
      setOpenSessionMenuId(null);
    };
    const onEsc = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpenSessionMenuId(null);
    };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", onEsc);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", onEsc);
    };
  }, [openSessionMenuId]);

  const orderedSessions = useMemo(
    () =>
      [...sessions].sort((a, b) => {
        const ap = a.pinned ? 1 : 0;
        const bp = b.pinned ? 1 : 0;
        if (ap !== bp) return bp - ap;
        return b.updatedAt - a.updatedAt;
      }),
    [sessions]
  );

  const shareSession = async (session: SessionState) => {
    const origin = typeof window !== "undefined" ? window.location.origin : "";
    const path = typeof window !== "undefined" ? window.location.pathname || "/" : "/";
    const text = `「${session.title}」· 有为销售顾问会话（仅本地存储）\n${origin}${path}`;
    try {
      if (typeof navigator !== "undefined" && navigator.share) {
        await navigator.share({ title: session.title, text });
        return;
      }
    } catch {
      /* 用户取消分享面板或不可用 */
    }
    await copyTextToClipboard(text);
  };

  const togglePinSession = (sessionId: string) => {
    setSessions((prev) =>
      prev.map((s) => (s.id === sessionId ? { ...s, pinned: !s.pinned, updatedAt: s.updatedAt } : s))
    );
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
    setOpenSessionMenuId((id) => (id === sessionId ? null : id));
    if (activeSessionId === sessionId) {
      setActiveSessionId(nextSessions[0].id);
    }
  };

  const askQuestion = async (presetQuestion?: string) => {
    const text = (presetQuestion ?? question).trim();
    if (!text || isStreaming) return;
    setComposerDocGateHint("");
    // 允许用户使用 @ 选文档：后端基于文档标题做匹配时不需要这个前缀符号
    const payloadQuestion = text.replace(/(^|\s)@/g, "$1");
    const sessionId = activeSessionRef.current;
    const mid = ++messageIdSeqRef.current;
    const userId = `u-${mid}`;
    const assistantId = `a-${mid}`;
    const declineHit = looksLikeDeclineFollowup(text);
    const contactHit = looksLikeContactInUserMessage(text);
    setQuestion("");
    const docsForRequest = selectedYuqueDocs.filter((d) => d.docId >= 1);
    setSelectedYuqueDocs([]);
    closeDocSuggest();
    const streamPathForMeta = chatV4Enabled
      ? "/chat/v4/stream"
      : chatV3Enabled
        ? "/chat/v3/stream"
        : chatV2Enabled
          ? "/chat/v2/stream"
          : "/chat/stream";
    setLastRequestMeta({
      model: selectedModel,
      owner: yuqueOwnerForApi,
      chat_mode: "visitor_sales",
      token_profile: "primary",
      stream_path: streamPathForMeta,
    });
    userStreamStopRef.current = false;
    setStreamingAssistantId(assistantId);
    setIsStreaming(true);
    setSessions((prev) =>
      prev.map((session) =>
        session.id === sessionId
          ? {
              ...touchSession(session),
              declinedContact: session.declinedContact || declineHit,
              contactCollected: session.contactCollected || contactHit,
              messages: [
                ...session.messages,
                { id: userId, role: "user", text },
                {
                  id: assistantId,
                  role: "assistant",
                  text: "",
                  streamStage: "正在搜索知识库资料…",
                  streamElapsedSec: 0,
                },
              ],
            }
          : session
      )
    );

    const controller = new AbortController();
    controllerRef.current = controller;

    let connectTimer: number | null = null;
    let elapsedTimer: number | null = null;
    let elapsedSec = 0;
    let streamTextBuf = "";
    let streamFlushTimer: number | null = null;
    let pendingDonePayload: Record<string, unknown> | null = null;
    let receivedAnyToken = false;

    const flushStreamText = () => {
      if (!streamTextBuf) return;
      const chunk = streamTextBuf;
      streamTextBuf = "";
      appendTokenToMessage(chunk);
    };

    const scheduleStreamFlush = () => {
      if (streamFlushTimer != null) return;
      streamFlushTimer = window.setTimeout(() => {
        streamFlushTimer = null;
        flushStreamText();
        if (pendingDonePayload && !streamTextBuf) {
          const payload = pendingDonePayload;
          pendingDonePayload = null;
          applyDonePayload(payload);
        }
      }, STREAM_FLUSH_MS);
    };

    const flushStreamTextNow = () => {
      if (streamFlushTimer != null) {
        window.clearTimeout(streamFlushTimer);
        streamFlushTimer = null;
      }
      flushStreamText();
    };

    const appendTokenToMessage = (token: string) => {
      if (!token) return;
      setSessions((prev) =>
        prev.map((session) =>
          session.id === sessionId
            ? {
                ...session,
                messages: session.messages.map((item) =>
                  item.id === assistantId
                    ? {
                        ...item,
                        text: item.text + token,
                        ...(token.trim() ? { streamStage: undefined } : {}),
                      }
                    : item
                ),
              }
            : session
        )
      );
    };

    const applyDonePayload = (payload: Record<string, unknown>) => {
      const dbg = payload.debug as Record<string, unknown> | undefined;
      const media = parseChatMedia(payload.media);
      if (dbg && typeof dbg === "object") {
        setLastPipelineDebug({ ...dbg });
      }
      const vs = dbg?.visitor_sales as Record<string, unknown> | undefined;
      const serverContact = Boolean(
        vs && vs.contact_detected === true
      ) || Boolean(dbg && dbg.contact_detected === true);
      const trialApplyAvailable = payload.trial_apply_available === true;
      setSessions((prev) =>
        prev.map((session) =>
          session.id === sessionId
            ? {
                ...touchSession(session),
                contactCollected: Boolean(session.contactCollected || serverContact),
                messages: session.messages.map((item) =>
                  item.id === assistantId
                    ? {
                        ...item,
                        media,
                        streamStage: undefined,
                        streamElapsedSec: undefined,
                        trialApplyAvailable,
                        text:
                          item.text ||
                          (typeof payload.answer === "string" && payload.answer) ||
                          "没有返回回答。",
                      }
                    : item
                ),
              }
            : session
        )
      );
    };

    try {
      connectTimer = window.setTimeout(() => {
        controller.abort();
      }, STREAM_CONNECT_MS);

      // 新建会话后会触发服务端 reset：在首条消息发出前，尽量等待 reset 完成，避免串历史。
      const pendingReset = resetPendingRef.current.get(sessionId);
      if (pendingReset) {
        await Promise.race([
          pendingReset,
          new Promise<void>((resolve) => window.setTimeout(() => resolve(), 1200)),
        ]);
      }

      const streamPath = chatV4Enabled
        ? "/chat/v4/stream"
        : chatV3Enabled
          ? "/chat/v3/stream"
          : chatV2Enabled
            ? "/chat/v2/stream"
            : "/chat/stream";
      const response = await fetch(streamPath, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: payloadQuestion,
          model: selectedModel,
          owner: yuqueOwnerForApi,
          token_profile: "primary",
          chat_mode: "visitor_sales",
          session_id: sessionId,
          selected_yuque_docs: docsForRequest.map((d) => ({
            doc_id: d.docId,
            slug: d.slug || null,
            title: d.title,
          })),
        }),
        signal: controller.signal,
      });
      if (connectTimer != null) {
        window.clearTimeout(connectTimer);
        connectTimer = null;
      }
      if (!response.ok || !response.body) throw new Error("stream_unavailable");

      elapsedTimer = window.setInterval(() => {
        elapsedSec += 1;
        setSessions((prev) =>
          prev.map((session) =>
            session.id === sessionId
              ? {
                  ...session,
                  messages: session.messages.map((item) =>
                    item.id === assistantId ? { ...item, streamElapsedSec: elapsedSec } : item
                  ),
                }
              : session
          )
        );
      }, 2000);

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
            const payload = JSON.parse(parsed.data || "{}");
            const token = String(payload.token || "");
            if (token) {
              receivedAnyToken = true;
              streamTextBuf += token;
              scheduleStreamFlush();
            }
          } else if (parsed.event === "stage") {
            const payload = JSON.parse(parsed.data || "{}") as Record<string, unknown>;
            const detail = typeof payload.detail === "string" ? payload.detail : "";
            const stage = typeof payload.stage === "string" ? payload.stage : "";
            const line = formatStreamStageLine(stage, detail);
            setSessions((prev) =>
              prev.map((session) =>
                session.id === sessionId
                  ? {
                      ...touchSession(session),
                      messages: session.messages.map((item) =>
                        item.id === assistantId ? { ...item, streamStage: line } : item
                      ),
                    }
                  : session
              )
            );
          } else if (parsed.event === "done") {
            doneReceived = true;
            const payload = JSON.parse(parsed.data || "{}") as Record<string, unknown>;
            if (streamTextBuf || streamFlushTimer != null) {
              pendingDonePayload = payload;
            } else {
              applyDonePayload(payload);
            }
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
                          streamStage: undefined,
                          streamElapsedSec: undefined,
                          text: item.text || "请求失败，请稍后重试。",
                        }
                      : item
                  ),
                }
              : session
          )
        );
      }
    } catch (error: unknown) {
      const err = error as { name?: string; message?: string };
      const errMsg = typeof err?.message === "string" ? err.message : "";
      const idleStop = errMsg === "stream_idle_timeout";
      const userStop = err?.name === "AbortError" && userStreamStopRef.current;
      const connectAbort =
        err?.name === "AbortError" && !userStreamStopRef.current && elapsedSec === 0 && !idleStop;

      let fallbackText = "";
      if (idleStop) {
        fallbackText =
          "等待超时：长时间未收到服务器新数据，已自动停止。若问题包含多图识读、语雀拉取或 MCP 多步调用，后台可能较慢；可稍后重试、缩短问题或关闭部分能力后再试。";
      } else if (userStop) {
        fallbackText = "已停止生成。";
      } else if (connectAbort) {
        fallbackText =
          "连接超时：在限定时间内未收到服务器响应，请检查网络、后端是否已启动，或稍后重试。";
      } else if (err?.name === "AbortError") {
        fallbackText = "请求已中断。";
      } else if (errMsg === "stream_unavailable") {
        fallbackText = "流式服务不可用（未收到有效响应），请确认后端已启动或稍后重试。";
      } else {
        fallbackText = errMsg || "请求失败，请稍后重试。";
      }

      // 兼容后端重启后的旧会话：若本次一个 token 都没收到且直接失败，
      // 自动重置服务端该 session，减少“必须手动新建会话”。
      if (!receivedAnyToken && sessionId) {
        try {
          await fetch("/chat/session/reset", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId, chat_mode: "visitor_sales" }),
          });
          fallbackText =
            `${fallbackText}\n\n检测到会话状态异常，已自动重置当前会话上下文。请直接重试这条问题。`;
        } catch {
          // 忽略重置失败，保留原始错误提示
        }
      }

      setSessions((prev) =>
        prev.map((session) =>
          session.id === sessionId
            ? {
                ...touchSession(session),
                messages: session.messages.map((item) =>
                  item.id === assistantId
                    ? {
                        ...item,
                        streamStage: undefined,
                        streamElapsedSec: undefined,
                        text: item.text || fallbackText,
                      }
                    : item
                ),
              }
            : session
        )
      );
    } finally {
      if (connectTimer != null) {
        window.clearTimeout(connectTimer);
      }
      if (elapsedTimer != null) {
        window.clearInterval(elapsedTimer);
      }
      flushStreamTextNow();
      if (pendingDonePayload) {
        const payload = pendingDonePayload;
        pendingDonePayload = null;
        applyDonePayload(payload);
      }
      controllerRef.current = null;
      setStreamingAssistantId(null);
      userStreamStopRef.current = false;
      setIsStreaming(false);
      setInactivityEpoch((e) => e + 1);
    }
  };

  const showDocSuggest = docSuggestOpen && docSuggestDocs.length > 0;
  const docFloatPortalOpen = showDocSuggest;
  const handleFocusSceneShortcut = (scene: (typeof FOCUS_SCENE_ITEMS)[number]) => {
    if (isStreaming) return;
    setActiveFocusScene(scene);
    void askQuestion(`我想要咨询${scene}的内容，请帮我解答。`);
  };

  const turnTrace = parseTurnTrace(lastPipelineDebug);
  const pipelineMode = typeof lastPipelineDebug?.mode === "string" ? lastPipelineDebug.mode : turnTrace?.pipeline;
  const kbNodeId = useCallback((doc: DocMeta, idx: number) => {
    return doc.toc_uuid || `${doc.toc_kind || "doc"}-${doc.id ?? doc.slug ?? doc.title}-${idx}`;
  }, []);
  const kbTreeRows = useMemo(() => {
    const rows: Array<{
      doc: DocMeta;
      idx: number;
      level: number;
      nodeId: string;
      hasChildren: boolean;
      isCollapsed: boolean;
    }> = [];
    const byUuid = new Map<string, DocMeta>();
    kbPanelDocs.forEach((doc) => {
      const id = (doc.toc_uuid || "").trim();
      if (id) byUuid.set(id, doc);
    });
    const levelMemo = new Map<string, number>();
    const computeLevel = (doc: DocMeta, visiting: Set<string> = new Set()): number => {
      const selfId = (doc.toc_uuid || "").trim();
      if (selfId && levelMemo.has(selfId)) return levelMemo.get(selfId)!;
      const parentId = (doc.toc_parent_uuid || "").trim();
      const fallback = normalizeTocLevel(doc.toc_level);
      if (!parentId || !byUuid.has(parentId)) {
        if (selfId) levelMemo.set(selfId, fallback);
        return fallback;
      }
      if (selfId && visiting.has(selfId)) return fallback;
      const nextVisiting = new Set(visiting);
      if (selfId) nextVisiting.add(selfId);
      const parent = byUuid.get(parentId)!;
      const level = Math.max(1, Math.min(3, computeLevel(parent, nextVisiting) + 1));
      if (selfId) levelMemo.set(selfId, level);
      return level;
    };
    const childrenCount = new Map<string, number>();
    kbPanelDocs.forEach((doc) => {
      const parentId = (doc.toc_parent_uuid || "").trim();
      if (!parentId) return;
      childrenCount.set(parentId, (childrenCount.get(parentId) || 0) + 1);
    });
    const hiddenAncestorLevels: number[] = [];
    for (let idx = 0; idx < kbPanelDocs.length; idx += 1) {
      const doc = kbPanelDocs[idx]!;
      const level = computeLevel(doc);
      while (
        hiddenAncestorLevels.length > 0 &&
        level <= hiddenAncestorLevels[hiddenAncestorLevels.length - 1]!
      ) {
        hiddenAncestorLevels.pop();
      }
      const nodeId = kbNodeId(doc, idx);
      const selfId = (doc.toc_uuid || "").trim();
      const hasChildren = Boolean(selfId && childrenCount.get(selfId));
      const isCollapsed = collapsedKbNodeIds.has(nodeId);
      const hidden = hiddenAncestorLevels.length > 0;
      if (!hidden) {
        rows.push({ doc, idx, level, nodeId, hasChildren, isCollapsed });
      }
      if (hasChildren && isCollapsed) {
        hiddenAncestorLevels.push(level);
      }
    }
    return rows;
  }, [kbPanelDocs, kbNodeId, collapsedKbNodeIds]);

  useEffect(() => {
    setCollapsedKbNodeIds((prev) => {
      if (!prev.size) return prev;
      const validIds = new Set(kbPanelDocs.map((doc, idx) => kbNodeId(doc, idx)));
      let changed = false;
      const next = new Set<string>();
      prev.forEach((id) => {
        if (validIds.has(id)) next.add(id);
        else changed = true;
      });
      return changed ? next : prev;
    });
  }, [kbPanelDocs, kbNodeId]);

  return (
    <>
      <div
        className={`app-shell${
          !SHOW_DEV_PANEL
            ? " app-shell--no-dev"
            : devSidebarCollapsed
              ? " app-shell--dev-collapsed"
              : ""
        }`}
      >
        <header className="app-top-nav">
          <div className="app-top-nav-inner">
            <div className="consult-top-brand">
              <img className="consult-top-brand-image" src="/youwei-logo.png" alt="有为 Logo" />
            </div>
            <span className="consult-top-page-title">预约方案咨询</span>
            <button type="button" className="consult-top-ghost-btn" aria-label="咨询工作台">
              咨询工作台
            </button>
          </div>
        </header>

        <div className="app-body">
          <aside className="focus-scene-sidebar" aria-label="你最关注的场景">
            <div className="focus-scene-title">你最关注的场景</div>
            <div className="focus-scene-list">
              {FOCUS_SCENE_ITEMS.map((scene) => (
                <button
                  key={scene}
                  type="button"
                  className={`focus-scene-btn${activeFocusScene === scene ? " focus-scene-btn--active" : ""}`}
                  onClick={() => handleFocusSceneShortcut(scene)}
                  disabled={isStreaming}
                >
                  {scene}
                </button>
              ))}
            </div>
            <button type="button" className="focus-scene-consult-btn" disabled={isStreaming}>
              咨询
            </button>
            <p className="focus-scene-consult-hint">快速选择下方重点问题，不中断当天进度。</p>
          </aside>
          {SHOW_DEV_PANEL ? (
          <aside className={`dev-sidebar${devSidebarCollapsed ? " dev-sidebar--collapsed" : ""}`}>
            {devSidebarCollapsed ? (
              <div className="dev-sidebar-collapsed-stack">
                <button
                  type="button"
                  className="dev-sidebar-icon-btn"
                  onClick={() => setDevSidebarCollapsed(false)}
                  title="展开开发者面板"
                  aria-label="展开开发者面板"
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
                    <path d="M9 18l6-6-6-6" />
                  </svg>
                </button>
                <button type="button" className="dev-sidebar-icon-btn" onClick={createSession} title="新对话" aria-label="新对话">
                  +
                </button>
              </div>
            ) : (
              <>
                <div className="dev-sidebar-header">
                  <div>
                    <div className="dev-sidebar-title">开发者面板</div>
                    <div className="dev-sidebar-sub">运行追踪 · MCP · Skill · 文档</div>
                  </div>
                  <button
                    type="button"
                    className="dev-sidebar-collapse"
                    onClick={() => setDevSidebarCollapsed(true)}
                    title="收起"
                    aria-label="收起开发者面板"
                  >
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
                      <path d="M15 18l-6-6 6-6" />
                    </svg>
                  </button>
                </div>

                <div className="dev-sidebar-scroll">
                  <section className="dev-panel-section">
                    <div className="dev-panel-section-title">请求与模型</div>
                    <label className="dev-field">
                      <span className="dev-field-label">大语言模型</span>
                      <select
                        className="dev-select"
                        value={selectedModel}
                        onChange={(e) => setSelectedModel(e.target.value)}
                        disabled={isStreaming}
                      >
                        <option value="deepseek-chat">deepseek-chat</option>
                        <option value="deepseek-reasoner">deepseek-reasoner</option>
                        <option value="gpt-4o-mini">gpt-4o-mini</option>
                      </select>
                    </label>
                    <dl className="dev-kv">
                      <div>
                        <dt>chat_mode</dt>
                        <dd>{lastRequestMeta?.chat_mode ?? "visitor_sales"}</dd>
                      </div>
                      <div>
                        <dt>owner</dt>
                        <dd>{lastRequestMeta?.owner || yuqueOwnerForApi || "—"}</dd>
                      </div>
                      <div>
                        <dt>token_profile</dt>
                        <dd>{lastRequestMeta?.token_profile ?? "primary"}</dd>
                      </div>
                      <div>
                        <dt>上次请求 model</dt>
                        <dd>{lastRequestMeta?.model ?? "—"}</dd>
                      </div>
                      <div>
                        <dt>API 路径</dt>
                        <dd className="dev-mono">{lastRequestMeta?.stream_path ?? "—"}</dd>
                      </div>
                      <div>
                        <dt>pipeline</dt>
                        <dd>{formatUnknownDebug(pipelineMode ?? "—")}</dd>
                      </div>
                    </dl>
                  </section>

                  <section className="dev-panel-section">
                    <div className="dev-panel-section-title">本轮 MCP 调用</div>
                    {!lastPipelineDebug ? (
                      <p className="dev-muted">完成一次对话后显示本轮实际 MCP 调用。</p>
                    ) : turnTrace?.mcp_calls && turnTrace.mcp_calls.length > 0 ? (
                      <table className="dev-trace-table">
                        <thead>
                          <tr>
                            <th>工具</th>
                            <th>参数</th>
                            <th>结果</th>
                          </tr>
                        </thead>
                        <tbody>
                          {turnTrace.mcp_calls.map((row, idx) => (
                            <tr key={`${row.tool}-${idx}`}>
                              <td className="dev-mono">{row.tool}</td>
                              <td>
                                {row.tool === "yuque_search"
                                  ? row.query || "—"
                                  : row.doc_id
                                    ? `${row.doc_id}${row.title ? ` · ${row.title}` : ""}`
                                    : "—"}
                              </td>
                              <td>
                                {row.tool === "yuque_search"
                                  ? `命中 ${row.hit_count ?? 0}`
                                  : `${row.body_chars ?? 0} 字`}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    ) : (
                      <p className="dev-muted">
                        {pipelineMode === "v4_guide" || turnTrace?.pipeline === "v4_guide"
                          ? "本层为目录引导，未调用 MCP。"
                          : "本轮无 MCP 记录（可能未开启 EXPOSE_TURN_TRACE）。"}
                      </p>
                    )}
                  </section>

                  <section className="dev-panel-section">
                    <div className="dev-panel-section-title">本轮 Skill</div>
                    {!lastPipelineDebug ? (
                      <p className="dev-muted">完成一次深度讲解后显示动态选中的 Skill。</p>
                    ) : turnTrace?.skills && turnTrace.skills.length > 0 ? (
                      <ul className="dev-tool-list">
                        {turnTrace.skills.map((s) => (
                          <li key={s.skill_id}>
                            <span className="dev-mono">{s.skill_id}</span>
                            <span className="dev-tool-meta">{s.reason || "—"}</span>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="dev-muted">
                        {pipelineMode === "v4_guide" || (turnTrace?.dialog_level ?? 0) <= 1
                          ? "本层为目录引导，未启用 Skill。"
                          : "本轮未选中 Skill。"}
                      </p>
                    )}
                  </section>

                  <section className="dev-panel-section">
                    <div className="dev-panel-section-title">本轮文档</div>
                    {!lastPipelineDebug ? (
                      <p className="dev-muted">检索到的语雀文档将列于此。</p>
                    ) : turnTrace?.documents && turnTrace.documents.length > 0 ? (
                      <table className="dev-trace-table">
                        <thead>
                          <tr>
                            <th>标题</th>
                            <th>角色</th>
                            <th>预览</th>
                          </tr>
                        </thead>
                        <tbody>
                          {turnTrace.documents.map((d, idx) => (
                            <tr key={`${d.doc_id || d.title}-${idx}`}>
                              <td>
                                <div>{d.title || "—"}</div>
                                {d.doc_id ? <div className="dev-mono dev-small">{d.doc_id}</div> : null}
                              </td>
                              <td>{d.role || "related"}</td>
                              <td className="dev-trace-snippet">{d.snippet || "—"}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    ) : (
                      <p className="dev-muted">本轮未拉取文档（引导/澄清轮为空）。</p>
                    )}
                  </section>

                  <section className="dev-panel-section">
                    <div className="dev-panel-section-title">MCP 服务配置</div>
                    {!mcpData ? (
                      <p className="dev-muted">加载中或未配置…</p>
                    ) : (
                      <>
                        <dl className="dev-kv">
                          <div>
                            <dt>enabled</dt>
                            <dd>{mcpData.enabled ? "是" : "否"}</dd>
                          </div>
                          <div>
                            <dt>repo_scope</dt>
                            <dd className="dev-mono">{mcpData.repo_scope || "—"}</dd>
                          </div>
                        </dl>
                        <ul className="dev-tool-list">
                          {(mcpData.tools || []).map((t) => (
                            <li key={t.name}>
                              <span className="dev-mono">{t.name}</span>
                              <span className="dev-tool-meta">{t.category}</span>
                            </li>
                          ))}
                        </ul>
                      </>
                    )}
                  </section>

                  {lastPipelineDebug ? (
                    <section className="dev-panel-section">
                      <details className="dev-json-details">
                        <summary>完整 debug JSON</summary>
                        <pre className="dev-json-pre">{JSON.stringify(lastPipelineDebug, null, 2)}</pre>
                      </details>
                    </section>
                  ) : null}

                  <section className="dev-panel-section">
                    <div className="dev-panel-section-head">
                      <span className="dev-panel-section-title">知识库目录</span>
                      <button type="button" className="dev-refresh-btn" onClick={() => void loadKbToc()} disabled={kbPanelLoading}>
                        刷新
                      </button>
                    </div>
                    {kbPanelLoading ? <p className="dev-muted">正在加载…</p> : null}
                    {kbPanelError ? (
                      <p className="dev-error" role="alert">
                        {kbPanelError}
                      </p>
                    ) : null}
                    <div className="dev-kb-scroll">
                      {kbTreeRows.map(({ doc, level, nodeId, hasChildren, isCollapsed }) => {
                        const levelClass = `dev-kb-row--l${level}`;
                        const selectable = docMetaSelectable(doc);
                        const isTitle = !selectable;
                        const expandable = hasChildren;
                        return (
                          <button
                            key={nodeId}
                            type="button"
                            className={`dev-kb-row ${selectable ? "dev-kb-row--doc" : "dev-kb-row--title"} ${levelClass}${
                              expandable ? " dev-kb-row--expandable" : ""
                            }`}
                            onClick={() => {
                              if (expandable) {
                                setCollapsedKbNodeIds((prev) => {
                                  const next = new Set(prev);
                                  if (next.has(nodeId)) next.delete(nodeId);
                                  else next.add(nodeId);
                                  return next;
                                });
                                return;
                              }
                              if (selectable) {
                                pickDocFromKbPanel(doc);
                              }
                            }}
                            aria-expanded={expandable ? !isCollapsed : undefined}
                          >
                            {expandable ? (
                              <span className={`dev-kb-toggle${isCollapsed ? " dev-kb-toggle--collapsed" : ""}`} aria-hidden>
                                ▾
                              </span>
                            ) : (
                              <span className="dev-kb-toggle dev-kb-toggle--spacer" aria-hidden />
                            )}
                            <span className={`dev-kb-node-label${isTitle ? " dev-kb-node-label--title" : ""}`}>
                              {doc.title}
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  </section>

                  <section className="dev-panel-section">
                    <div className="dev-panel-section-title">快捷问法</div>
                    <div className="dev-quick-grid">
                      {VISITOR_QUICK_QUESTIONS.map((q) => (
                        <button
                          key={q.label}
                          type="button"
                          className="dev-quick-btn"
                          disabled={isStreaming}
                          onClick={() => setQuestion((prev) => (prev.trim() ? prev : q.text))}
                        >
                          {q.label}
                        </button>
                      ))}
                    </div>
                  </section>

                  <section className="dev-panel-section">
                    <div className="sidebar-section-head">
                      <span className="sidebar-section-title">历史会话</span>
                      <button type="button" className="sidebar-section-add" onClick={createSession} title="新建会话" aria-label="新建会话">
                        +
                      </button>
                    </div>
                    <ul className="session-list">
                      {orderedSessions.map((session) => (
                        <li key={session.id} className="session-item">
                          <div className={`session-row${session.id === activeSessionId ? " session-row--active" : ""}`}>
                            <button
                              type="button"
                              className="session-main"
                              onClick={() => {
                                setOpenSessionMenuId(null);
                              setHistorySessionPicked(true);
                                setActiveSession(session.id);
                              }}
                            >
                              <span className="session-title">{session.title}</span>
                              <span className="session-time">{formatRelativeTimeCN(session.updatedAt)}</span>
                            </button>
                            <div className="session-item-menu-wrap" data-session-menu-root>
                              <button
                                type="button"
                                className={`session-menu-trigger${openSessionMenuId === session.id ? " session-menu-trigger--open" : ""}`}
                                aria-label="更多操作"
                                aria-haspopup="menu"
                                aria-expanded={openSessionMenuId === session.id}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setOpenSessionMenuId((id) => (id === session.id ? null : session.id));
                                }}
                              >
                                <span className="session-menu-trigger-dots" aria-hidden>
                                  ···
                                </span>
                              </button>
                              {openSessionMenuId === session.id ? (
                                <div className="session-overflow-menu" role="menu">
                                  <button
                                    type="button"
                                    className="session-overflow-item"
                                    role="menuitem"
                                    onClick={() => {
                                      setOpenSessionMenuId(null);
                                      void shareSession(session);
                                    }}
                                  >
                                    <span className="session-overflow-icon" aria-hidden>
                                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="M4 12v8a2 2 0 002 2h12a2 2 0 002-2v-8" />
                                        <polyline points="16 6 12 2 8 6" />
                                        <line x1="12" y1="2" x2="12" y2="15" />
                                      </svg>
                                    </span>
                                    <span>分享</span>
                                  </button>
                                  <div className="session-overflow-divider" role="separator" />
                                  <button
                                    type="button"
                                    className="session-overflow-item"
                                    role="menuitem"
                                    onClick={() => {
                                      setOpenSessionMenuId(null);
                                      renameSession(session.id);
                                    }}
                                  >
                                    <span className="session-overflow-icon" aria-hidden>
                                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="M12 20h9M16.5 3.5a2.121 2.121 0 013 3L7 19l-4 1 1-4L16.5 3.5z" />
                                      </svg>
                                    </span>
                                    <span>重命名</span>
                                  </button>
                                  <div className="session-overflow-divider" role="separator" />
                                  <button
                                    type="button"
                                    className="session-overflow-item"
                                    role="menuitem"
                                    onClick={() => {
                                      setOpenSessionMenuId(null);
                                      togglePinSession(session.id);
                                    }}
                                  >
                                    <span className="session-overflow-icon" aria-hidden>
                                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                        <path d="M12 22s-4-5.5-4-9a4 4 0 118 0c0 3.5-4 9-4 9z" />
                                        <circle cx="12" cy="9" r="1.5" />
                                      </svg>
                                    </span>
                                    <span>{session.pinned ? "取消置顶" : "置顶聊天"}</span>
                                  </button>
                                  <div className="session-overflow-divider" role="separator" />
                                  <button
                                    type="button"
                                    className="session-overflow-item session-overflow-item--danger"
                                    role="menuitem"
                                    disabled={sessions.length <= 1}
                                    onClick={() => {
                                      setOpenSessionMenuId(null);
                                      removeSession(session.id);
                                    }}
                                  >
                                    <span className="session-overflow-icon" aria-hidden>
                                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <polyline points="3 6 5 6 21 6" />
                                        <path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2" />
                                      </svg>
                                    </span>
                                    <span>删除</span>
                                  </button>
                                </div>
                              ) : null}
                            </div>
                          </div>
                        </li>
                      ))}
                    </ul>
                  </section>

                  <section className="dev-panel-section dev-panel-section--account">
                    <div className="dev-account-row">
                      <div className="dev-account-avatar" aria-hidden>
                        {(yuqueSidebarLabel || "?").slice(0, 1).toUpperCase()}
                      </div>
                      <div className="dev-account-meta">
                        <div className="dev-account-name">{yuqueSidebarLabel || "—"}</div>
                        <div className="dev-account-hint">语雀 Token / 知识库</div>
                      </div>
                    </div>
                  </section>
                </div>
              </>
            )}
          </aside>
          ) : null}

          <div className="app-main chat-shell">
            <header className="chat-subbar">
              <div className="chat-subbar-title-wrap">
                <div className="chat-subbar-brand">
                  <span className="chat-subbar-brand-logo" aria-hidden>
                    AI
                  </span>
                  <div className="chat-subbar-brand-meta">
                    <span className="chat-subbar-brand-en">YOUWEI AI CONSULTANT</span>
                    <span className="chat-subbar-title">有为 AI 方案顾问</span>
                  </div>
                </div>
                <span className="chat-subbar-status">在线沟通</span>
              </div>
              <div className="chat-subbar-actions">
                <button
                  type="button"
                  className={`chat-subbar-btn${sessionCopied ? " chat-subbar-btn--copied" : ""}`}
                  onClick={() => void copySessionTranscript()}
                  disabled={!sessionTranscript || isStreaming}
                  title="复制当前会话全部对话"
                  aria-label="复制当前会话全部对话"
                >
                  {sessionCopied ? "已复制" : "复制会话"}
                </button>
                <button type="button" className="chat-subbar-new" onClick={createSession}>
                  新对话 <span className="chat-subbar-kbd">{newChatShortcutLabel}</span>
                </button>
              </div>
            </header>
            <div className="chat-consulting-direction" aria-label="当前咨询方向">
              <span className="chat-consulting-direction-label">当前咨询方向</span>
              <span className="chat-consulting-direction-tag">综合咨询</span>
            </div>
            <div className="chat-main">
              <div className={`chat-content-inner chat-body-inner${showWelcomeHero ? "" : " chat-body-inner--scroll"}`}>
                {showWelcomeHero ? (
                  <div className="welcome-hero">
                    <div className="welcome-sparkle" aria-hidden>
                      ✨
                    </div>
                    <h1 className="welcome-title">有为人工智能教育平台</h1>
                    <p className="welcome-sub">{VISITOR_WELCOME_TEXT}</p>
                    <div className="welcome-cards">
                      <button
                        type="button"
                        className="welcome-card"
                        onClick={() => setQuestion((q) => (q.trim() ? q : "你们平台是做什么的？"))}
                      >
                        <span className="welcome-card-icon welcome-card-icon--book" aria-hidden />
                        <div className="welcome-card-text">
                          <div className="welcome-card-name">了解平台</div>
                          <div className="welcome-card-desc">产品定位与价值</div>
                        </div>
                      </button>
                      <button
                        type="button"
                        className="welcome-card"
                        onClick={() => setQuestion((q) => (q.trim() ? q : "可以申请试用吗？"))}
                      >
                        <span className="welcome-card-icon welcome-card-icon--doc" aria-hidden />
                        <div className="welcome-card-text">
                          <div className="welcome-card-name">试用与演示</div>
                          <div className="welcome-card-desc">预约产品顾问</div>
                        </div>
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="chat-list chat-list--thread" ref={chatListRef} key={activeSessionId}>
                    {chatItems.map((item) => (
                      <div className={`msg ${item.role}`} key={item.id}>
                        <div style={{ width: "100%" }}>
                          {item.role === "assistant" && (item.streamStage || (item.streamElapsedSec ?? 0) > 0) ? (
                            <div className="stream-stage-block">
                              {item.streamStage ? (
                                <div className="stream-stage stream-stage--pending">{item.streamStage}</div>
                              ) : null}
                              {(item.streamElapsedSec ?? 0) > 0 ? (
                                <div className="stream-elapsed">已等待 {item.streamElapsedSec} 秒</div>
                              ) : null}
                              {(item.streamElapsedSec ?? 0) >= STREAM_SLOW_HINT_SEC ? (
                                <div className="stream-slow-hint">
                                  若进度长时间不变，可能是检索、配图识读或 MCP 较慢；超过{" "}
                                  {Math.floor(STREAM_READ_IDLE_MS / 1000)} 秒无新数据将自动中断本次请求。
                                </div>
                              ) : null}
                            </div>
                          ) : null}
                          <div className="bubble">
                            {item.text.trim() ? (
                              isStreaming && item.role === "assistant" && item.id === streamingAssistantId ? (
                                <div className="bubble-stream-text">{item.text}</div>
                              ) : (
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                  {normalizeMarkdownAutolinks(item.text)}
                                </ReactMarkdown>
                              )
                            ) : null}
                            {item.role === "assistant" && item.media && (item.media.videos.length > 0 || item.media.images.length > 0) ? (
                              <div className="msg-media">
                                {item.media.videos.length > 0 ? (
                                  <div className="msg-media-block">
                                    <div className="msg-media-title">相关视频</div>
                                    {item.media.videos.map((video, idx) => (
                                      <div key={`${video.url}-${idx}`} className="msg-video-card">
                                        <video className="msg-video-player" controls playsInline preload="metadata" src={video.url} />
                                        <a href={video.url} target="_blank" rel="noreferrer" className="msg-video-link">
                                          {video.title || video.doc_title || "查看视频链接"}
                                        </a>
                                      </div>
                                    ))}
                                  </div>
                                ) : null}
                                {item.media.images.length > 0 ? (
                                  <div className="msg-media-block">
                                    <div className="msg-media-title">相关图片</div>
                                    <div className="msg-image-grid">
                                      {item.media.images.map((image, idx) => (
                                        <a
                                          key={`${image.url}-${idx}`}
                                          className="msg-image-card"
                                          href={image.url}
                                          target="_blank"
                                          rel="noreferrer"
                                        >
                                          <img src={image.url} alt={image.title || image.doc_title || "相关图片"} loading="lazy" />
                                          <span>{image.title || image.doc_title || "查看原图"}</span>
                                        </a>
                                      ))}
                                    </div>
                                  </div>
                                ) : null}
                              </div>
                            ) : null}
                          </div>
                          <div className="msg-footer">
                            {item.role === "assistant" &&
                            item.trialApplyAvailable &&
                            !item.trialCredentialsShown ? (
                              <button
                                type="button"
                                className="trial-apply-button"
                                onClick={() => void requestTrialCredentials(item.id)}
                              >
                                申请测试账号
                              </button>
                            ) : null}
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
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              <footer className="composer-simple">
                <div className="composer-simple-inner">
                  {composerDocGateHint ? (
                    <div className="composer-doc-gate-hint" role="alert">
                      {composerDocGateHint}
                    </div>
                  ) : null}
                  {selectedYuqueDocs.length > 0 ? (
                    <div className="selected-docs selected-docs--compact">
                      <div className="selected-docs-label">已选语雀文档（将随本次发送锚定）</div>
                      <div className="selected-docs-chips">
                        {selectedYuqueDocs.map((d) => (
                          <span className="doc-chip" key={`${d.docId}-${d.title}`}>
                            {d.title}
                            <button
                              type="button"
                              className="doc-chip-x"
                              onClick={() => removeSelectedYuqueDoc(d.docId, d.title)}
                              aria-label={`移除 ${d.title}`}
                            >
                              ×
                            </button>
                          </span>
                        ))}
                      </div>
                    </div>
                  ) : null}
                  <div className="composer-simple-row">
                    {!IS_VISITOR_ROUTE ? (
                      <button
                        type="button"
                        className="composer-simple-upload"
                        aria-label="上传附件"
                        title="上传附件（待接入）"
                      >
                        📎
                      </button>
                    ) : null}
                    <div ref={composerInputWrapRef} className="composer-simple-field doc-suggest-wrap">
                      <textarea
                        ref={composerTextareaRef}
                        className="composer-simple-textarea"
                        rows={1}
                        placeholder="输入您想了解的内容；Enter 发送，Shift+Enter 换行。输入 @ 可联想语雀文档。"
                        value={question}
                        onChange={handleQuestionChange}
                        onKeyDown={(event) => {
                          if (docSuggestOpen && docSuggestDocs.length > 0) {
                            const idxs = selectableDocIndices(docSuggestDocs);
                            if (event.key === "ArrowDown" && idxs.length) {
                              event.preventDefault();
                              const cur = docSuggestActiveIndex;
                              const pos = idxs.indexOf(cur);
                              const nextPos = pos < 0 ? 0 : Math.min(pos + 1, idxs.length - 1);
                              setDocSuggestActiveIndex(idxs[nextPos]!);
                              return;
                            }
                            if (event.key === "ArrowUp" && idxs.length) {
                              event.preventDefault();
                              const cur = docSuggestActiveIndex;
                              const pos = idxs.indexOf(cur);
                              const nextPos = pos <= 0 ? 0 : pos - 1;
                              setDocSuggestActiveIndex(idxs[nextPos]!);
                              return;
                            }
                            if (event.key === "Enter" && !event.shiftKey) {
                              const doc = docSuggestDocs[docSuggestActiveIndex];
                              if (doc && docMetaSelectable(doc)) {
                                event.preventDefault();
                                applyDocSuggestPick(doc);
                                return;
                              }
                            }
                          }
                          if (event.key === "Enter" && !event.shiftKey) {
                            event.preventDefault();
                            void askQuestion();
                          }
                          if (event.key === "Escape") {
                            closeDocSuggest();
                          }
                        }}
                      />
                    </div>
                    <button
                      type="button"
                      className={`composer-simple-send${isStreaming ? " composer-simple-send--stop" : ""}`}
                      onClick={() => {
                        if (isStreaming) stopStreaming();
                        else void askQuestion();
                      }}
                      disabled={!isStreaming && !question.trim()}
                      title={isStreaming ? "停止生成" : "提交"}
                    >
                      {isStreaming ? "停止" : "↑"}
                    </button>
                  </div>
                </div>
              </footer>
            </div>
          </div>
        </div>
      </div>
    {docFloatPortalOpen && docFloatRect
      ? createPortal(
          <div
            ref={docFloatRootRef}
            className="doc-suggest-box doc-suggest-box--portal"
            style={{
              bottom: docFloatRect.bottom,
              left: docFloatRect.left,
              width: docFloatRect.width,
              maxHeight: docFloatRect.maxHeight,
              top: "auto",
            }}
            role="listbox"
            aria-label="@ 文档联想"
          >
            {docSuggestDocs.map((doc, idx) => {
              const depth = Math.max(0, (doc.toc_level ?? 1) - 1);
              const pad = 10 + depth * 14;
              const active = idx === docSuggestActiveIndex;
              const rowClass = `doc-suggest-item ${active ? "active" : ""} ${
                doc.toc_kind === "title" ? "doc-suggest-item--toc-title" : ""
              }`.trim();
              if (!docMetaSelectable(doc)) {
                return (
                  <div
                    key={doc.toc_uuid ?? `h-${doc.title}-${idx}`}
                    className={rowClass}
                    style={{ paddingLeft: pad }}
                    aria-hidden
                  >
                    <span className="doc-suggest-toc-title-label">{doc.title}</span>
                  </div>
                );
              }
              return (
                <button
                  key={doc.toc_uuid ?? doc.id ?? doc.slug ?? `${doc.title}-${idx}`}
                  type="button"
                  className={rowClass}
                  style={{ paddingLeft: pad }}
                  onClick={() => applyDocSuggestPick(doc)}
                >
                  {doc.title}
                </button>
              );
            })}
          </div>,
          document.body,
        )
      : null}
    </>
  );
}

export default App
