import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ChangeEvent } from "react";
import { createPortal } from "react-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { normalizeMarkdownAutolinks } from "./markdownAutolink";
import { matchRoutedSkillId } from "./matchRoutedSkill";

/** 建立连接后首包 / 两次数据之间的最大等待（毫秒），超时则中止并提示 */
const STREAM_READ_IDLE_MS = 120_000;
/** 从发起请求到收到响应头的最长等待 */
const STREAM_CONNECT_MS = 45_000;
/** 超过该秒数后显示「可能较慢」说明 */
const STREAM_SLOW_HINT_SEC = 20;

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

type SourceLinkItem = {
  title: string;
  url: string | null;
};

type ChatItem = {
  id: string;
  role: Role;
  text: string;
  debug?: string;
  /** 流式阶段提示（来自 SSE event: stage），完成或出错后清除 */
  streamStage?: string;
  /** 本轮流式已等待秒数（前端计时，完成或出错后清除） */
  streamElapsedSec?: number;
  /** 本轮回答完成后，后端返回的参考来源（含语雀链接，若服务端配置为展示） */
  sourceLinks?: SourceLinkItem[];
};
type SessionState = {
  id: string;
  title: string;
  updatedAt: number;
  messages: ChatItem[];
  /** 侧栏置顶排序用；旧版 localStorage 无此字段时视为未置顶 */
  pinned?: boolean;
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
  toc_level?: number | null;
  toc_kind?: string | null;
  toc_selectable?: boolean | null;
};

function docMetaSelectable(doc: DocMeta): boolean {
  return doc.toc_selectable !== false;
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

type SkillId =
  | "reading-digest"
  | "daily-capture"
  | "note-refine"
  | "knowledge-connect"
  | "style-extract"
  | "smart-search"
  | "smart-summary"
  | "stale-detector";

type DocumentAnchorPolicy = "yes" | "optional" | "no";

type SkillDef = {
  id: SkillId;
  name: string;
  description: string;
  triggers: string[];
  /** yes：必须先 @ 选中文档（进入「已选择文档」）再发送或使用该 Skill 快捷 */
  documentRequired: DocumentAnchorPolicy;
  /** 输入区快捷填入；文案内含与后端 app/rag/skill_router 一致的触发词 */
  quickFill: { label: string; text: string };
};

/** 侧栏外 fixed 浮层（避免 sidebar overflow 裁切） */
type AbilityHoverTip = {
  top: number;
  left: number;
  maxWidth: number;
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
    documentRequired: "yes",
    quickFill: {
      label: "阅读笔记",
      text: "请结合检索到的上下文做阅读笔记：输出核心观点、金句与行动项。",
    },
  },
  {
    id: "daily-capture",
    name: "daily-capture",
    description: "把碎片想法/待办整理成结构化条目：标题/正文/标签/待办（只读）。",
    triggers: ["碎片", "捕获", "记录", "待办", "想法收集", "daily", "capture"],
    documentRequired: "yes",
    quickFill: {
      label: "碎片整理",
      text: "我想做碎片捕获，请把下面想法整理成标题、正文、标签和待办。",
    },
  },
  {
    id: "note-refine",
    name: "note-refine",
    description: "润色打磨笔记：提升结构与表达（只读）。",
    triggers: ["润色", "打磨", "refine", "优化表达", "改写", "提高质量", "note-refine"],
    documentRequired: "yes",
    quickFill: {
      label: "润色笔记",
      text: "请润色下面这段内容，优化表达与结构，保持原意。",
    },
  },
  {
    id: "knowledge-connect",
    name: "knowledge-connect",
    description: "分析多文档关联，输出主题簇与关联点（只读）。",
    triggers: ["关联", "联系", "聚类", "主题", "知识网络", "connect", "关联发现"],
    documentRequired: "no",
    quickFill: {
      label: "主题关联",
      text: "请基于知识库做多文档主题聚类，说明文档之间的联系与关联点。",
    },
  },
  {
    id: "style-extract",
    name: "style-extract",
    description: "从样本文档提炼写作风格画像：语气/句式/结构等（只读）。",
    triggers: ["风格", "用词", "句式", "表达习惯", "style", "画像", "style-extract"],
    documentRequired: "yes",
    quickFill: {
      label: "风格画像",
      text: "请从上下文中提炼写作风格画像：语气、用词与句式习惯。",
    },
  },
  {
    id: "smart-search",
    name: "smart-search",
    description: "把检索到的候选上下文组织成可读搜索回答：候选标题 + 摘要（只读）。",
    triggers: ["搜索", "找", "在哪里", "文档在哪", "smart-search", "查找"],
    documentRequired: "no",
    quickFill: {
      label: "文档搜索",
      text: "请帮我在知识库里搜索并列出最相关的文档及摘要说明。",
    },
  },
  {
    id: "smart-summary",
    name: "smart-summary",
    description: "按粒度生成摘要/概述：要点、详细段落；可根据“约100字”控制长度（只读）。",
    triggers: ["总结", "摘要", "概述", "要点", "大概100字", "约100字", "一句话", "详细总结", "smart-summary"],
    documentRequired: "yes",
    quickFill: {
      label: "摘要概述",
      text: "请概述要点，并给出简要总结（约100字）。",
    },
  },
  {
    id: "stale-detector",
    name: "stale-detector",
    description: "过期检测：基于文档 updated_at 列出疑似过期候选，并给出更新建议（只读）。",
    triggers: ["过期", "陈旧", "检测", "stale", "更新建议", "健康度", "过期检测"],
    documentRequired: "no",
    quickFill: {
      label: "过期检测",
      text: "请做过期检测，从健康度角度列出需要关注的文档并给出更新建议。",
    },
  },
];

/** 检索阶段统一展示为产品文案；生成阶段仍用服务端 detail */
function formatStreamStageLine(stage: string, detail: string): string {
  if (stage === "retrieving") return "正在搜索知识库资料…";
  if (stage === "vision") return detail.trim() || "正在识读文档插图…";
  return detail.trim() || stage || "处理中…";
}

function parseDoneSourceLinks(payload: Record<string, unknown>): SourceLinkItem[] {
  const raw = payload.sources;
  if (!Array.isArray(raw)) return [];
  return raw.map((entry) => {
    const o = entry as Record<string, unknown>;
    const title = typeof o.title === "string" && o.title.trim() ? o.title.trim() : "未命名资料";
    const u = o.url;
    const url = typeof u === "string" && u.trim() ? u.trim() : null;
    return { title, url };
  });
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

/** 新会话无占位消息；空态由主区欢迎屏展示（与产品原型一致） */
function emptySessionMessages(): ChatItem[] {
  return [];
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

function readStoredSessionState(): { sessions: SessionState[]; activeSessionId: string } {
  const fallback = (): { sessions: SessionState[]; activeSessionId: string } => ({
    sessions: [{ id: "default", title: "默认会话", updatedAt: Date.now(), messages: emptySessionMessages() }],
    activeSessionId: "default",
  });
  try {
    const saved = localStorage.getItem("rag_frontend_sessions_v1");
    if (!saved) return fallback();
    const parsed = JSON.parse(saved) as { activeSessionId?: string; sessions?: SessionState[] };
    if (!parsed?.sessions?.length) return fallback();
    return {
      sessions: parsed.sessions,
      activeSessionId: parsed.activeSessionId || parsed.sessions[0].id,
    };
  } catch {
    return fallback();
  }
}

/** 将已选语雀文档标题写入快捷问法前缀，便于检索锚定（多篇用书名号串联） */
function buildSkillQuickBody(skill: SkillDef, docTitles: readonly string[]): string {
  const scope =
    docTitles.length === 0
      ? ""
      : docTitles.length === 1
        ? `针对知识库文档《${docTitles[0]}》，`
        : `针对知识库文档《${docTitles.join("》《")}》，`;
  return `${scope}${skill.quickFill.text}`;
}

function App() {
  const [question, setQuestion] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [selectedModel, setSelectedModel] = useState("deepseek-chat");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [activeSessionId, setActiveSessionId] = useState(() => readStoredSessionState().activeSessionId);
  const [sessions, setSessions] = useState<SessionState[]>(() => readStoredSessionState().sessions);
  /** 历史会话行「⋯」溢出菜单：同时只展开一行 */
  const [openSessionMenuId, setOpenSessionMenuId] = useState<string | null>(null);
  const [mcpData, setMcpData] = useState<MCPCapabilitiesResponse | null>(null);
  const [mcpError, setMcpError] = useState("");
  const [abilityHover, setAbilityHover] = useState<AbilityHoverTip | null>(null);
  const abilityHoverHideTimerRef = useRef<number | null>(null);
  const [abilitiesCollapsed, setAbilitiesCollapsed] = useState(false);
  const [docSuggestOpen, setDocSuggestOpen] = useState(false);
  const [docSuggestDocs, setDocSuggestDocs] = useState<DocMeta[]>([]);
  const [docSuggestActiveIndex, setDocSuggestActiveIndex] = useState(0);
  const [selectedYuqueDocs, setSelectedYuqueDocs] = useState<SelectedYuqueDocLocal[]>([]);
  const [kbPanelOpen, setKbPanelOpen] = useState(false);
  const [kbPanelLoading, setKbPanelLoading] = useState(false);
  const [kbPanelError, setKbPanelError] = useState("");
  const [kbPanelDocs, setKbPanelDocs] = useState<DocMeta[]>([]);
  const [composerDocGateHint, setComposerDocGateHint] = useState("");
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const controllerRef = useRef<AbortController | null>(null);
  /** 用户点击「停止」触发的 abort，与超时/空闲 abort 区分 */
  const userStreamStopRef = useRef(false);
  const chatListRef = useRef<HTMLDivElement | null>(null);
  const activeSessionRef = useRef(activeSessionId);
  const suggestDebounceTimerRef = useRef<number | null>(null);
  const latestSuggestReqIdRef = useRef(0);
  const docTokenRangeRef = useRef<{ start: number; end: number } | null>(null);
  /** 上一次通过 Skill 快捷填入的完整文案，用于切换快捷时先删掉上一段 */
  const lastSkillQuickTextRef = useRef<string | null>(null);
  const messageIdSeqRef = useRef(0);
  const composerInputWrapRef = useRef<HTMLDivElement | null>(null);
  const kbListPillRef = useRef<HTMLButtonElement | null>(null);
  const docFloatRootRef = useRef<HTMLDivElement | null>(null);
  type DocFloatRect = { bottom: number; left: number; width: number; maxHeight: number };
  const [docFloatRect, setDocFloatRect] = useState<DocFloatRect | null>(null);

  const cancelAbilityHoverHide = () => {
    if (abilityHoverHideTimerRef.current != null) {
      window.clearTimeout(abilityHoverHideTimerRef.current);
      abilityHoverHideTimerRef.current = null;
    }
  };

  const scheduleAbilityHoverHide = () => {
    cancelAbilityHoverHide();
    abilityHoverHideTimerRef.current = window.setTimeout(() => {
      setAbilityHover(null);
      abilityHoverHideTimerRef.current = null;
    }, 160);
  };

  const showAbilityHover = (
    anchor: HTMLElement,
    payload: Pick<AbilityHoverTip, "kind" | "name" | "description" | "triggers">,
  ) => {
    cancelAbilityHoverHide();
    const r = anchor.getBoundingClientRect();
    const maxW = Math.min(280, Math.max(160, window.innerWidth - r.right - 24));
    setAbilityHover({
      top: r.top,
      left: r.right + 8,
      maxWidth: maxW,
      ...payload,
    });
  };

  useEffect(() => {
    return () => {
      if (abilityHoverHideTimerRef.current != null) {
        window.clearTimeout(abilityHoverHideTimerRef.current);
      }
    };
  }, []);

  const stripPreviousSkillQuick = (prev: string, last: string | null): string => {
    if (!last) return prev;
    const p = prev.replace(/\r\n/g, "\n");
    const L = last.replace(/\r\n/g, "\n");
    if (p.trim() === L.trim()) return "";
    const suffix = `\n\n${L}`;
    if (p.endsWith(suffix)) return p.slice(0, -suffix.length).replace(/\s+$/, "");
    if (p === L) return "";
    return p;
  };

  const applyQuickSkillShortcut = (skill: SkillDef) => {
    if (skill.documentRequired === "yes" && selectedYuqueDocs.length === 0) {
      setComposerDocGateHint(
        `使用「${skill.quickFill.label}」前请先输入 @，并从列表中选择至少一篇语雀文档（将出现在下方「已选择文档」）。`,
      );
      return;
    }
    setComposerDocGateHint("");
    const text = buildSkillQuickBody(skill, selectedDocTitles);
    const previous = lastSkillQuickTextRef.current;
    setQuestion((prev) => {
      const without = stripPreviousSkillQuick(prev, previous);
      const base = without.trim();
      return base ? `${base}\n\n${text}` : text;
    });
    lastSkillQuickTextRef.current = text;
  };

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
      } catch {
        setMcpError("MCP 能力读取失败");
      }
    };
    void loadMcpCapabilities();
    const onVis = () => {
      if (document.visibilityState === "visible") void loadMcpCapabilities();
    };
    document.addEventListener("visibilitychange", onVis);
    return () => document.removeEventListener("visibilitychange", onVis);
  }, []);

  const activeSession = useMemo(
    () => sessions.find((item) => item.id === activeSessionId) || null,
    [sessions, activeSessionId]
  );
  const chatItems = useMemo(() => activeSession?.messages ?? [], [activeSession]);
  /** 尚无用户消息时展示居中欢迎屏（与原型空态一致） */
  const showWelcomeHero = useMemo(() => !chatItems.some((m) => m.role === "user"), [chatItems]);

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

  const mcpMeta = useMemo(() => {
    if (mcpError) return mcpError;
    if (!mcpData) return "加载中...";
    return `状态：${mcpData.enabled ? "已启用" : "未启用"} | 作用域：${mcpData.repo_scope || "未配置"}`;
  }, [mcpData, mcpError]);

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

  const selectedDocTitles = useMemo(() => selectedYuqueDocs.map((d) => d.title), [selectedYuqueDocs]);

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
    if (!kbPanelOpen) return;
    void loadKbToc();
  }, [kbPanelOpen, yuqueOwnerForApi, loadKbToc]);

  const closeDocSuggest = useCallback(() => {
    setDocSuggestOpen(false);
    setDocSuggestDocs([]);
    docTokenRangeRef.current = null;
    setDocSuggestActiveIndex(0);
  }, []);

  const toggleKbPanel = useCallback(() => {
    setKbPanelOpen((open) => {
      const next = !open;
      if (next) closeDocSuggest();
      return next;
    });
  }, [closeDocSuggest]);

  const updateDocFloatRect = useCallback(() => {
    const showSuggest = docSuggestOpen && docSuggestDocs.length > 0;
    const open = showSuggest || kbPanelOpen;
    if (!open) {
      setDocFloatRect(null);
      return;
    }
    const anchorEl = showSuggest
      ? composerInputWrapRef.current
      : kbPanelOpen
        ? kbListPillRef.current
        : null;
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
  }, [docSuggestOpen, docSuggestDocs.length, kbPanelOpen]);

  useLayoutEffect(() => {
    const open = (docSuggestOpen && docSuggestDocs.length > 0) || kbPanelOpen;
    if (!open) {
      setDocFloatRect(null);
      return;
    }
    updateDocFloatRect();
    const ro = new ResizeObserver(() => {
      updateDocFloatRect();
    });
    const watch = [composerInputWrapRef.current, kbListPillRef.current].filter(Boolean);
    watch.forEach((el) => ro.observe(el!));
    window.addEventListener("resize", updateDocFloatRect);
    window.addEventListener("scroll", updateDocFloatRect, true);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", updateDocFloatRect);
      window.removeEventListener("scroll", updateDocFloatRect, true);
    };
  }, [docSuggestOpen, docSuggestDocs.length, kbPanelOpen, updateDocFloatRect]);

  useEffect(() => {
    const open = (docSuggestOpen && docSuggestDocs.length > 0) || kbPanelOpen;
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      const t = e.target as Node | null;
      if (!t) return;
      if (docFloatRootRef.current?.contains(t)) return;
      if (composerInputWrapRef.current?.contains(t)) return;
      if (kbListPillRef.current?.contains(t)) return;
      closeDocSuggest();
      setKbPanelOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [docSuggestOpen, docSuggestDocs.length, kbPanelOpen, closeDocSuggest]);

  useEffect(() => {
    const open = (docSuggestOpen && docSuggestDocs.length > 0) || kbPanelOpen;
    if (!open) return;
    const onEsc = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      closeDocSuggest();
      setKbPanelOpen(false);
    };
    document.addEventListener("keydown", onEsc);
    return () => document.removeEventListener("keydown", onEsc);
  }, [docSuggestOpen, docSuggestDocs.length, kbPanelOpen, closeDocSuggest]);

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
    setKbPanelOpen(false);
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

    setKbPanelOpen(false);
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

  const createSession = useCallback(() => {
    const id = `s-${Date.now()}`;
    setSessions((prev) => {
      const title = `新会话 ${prev.length + 1}`;
      return [{ id, title, updatedAt: Date.now(), messages: emptySessionMessages() }, ...prev];
    });
    setActiveSessionId(id);
  }, []);

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
    const text = `「${session.title}」· 语雀 AI 会话摘要（仅本地存储，无分享链接）\n${origin}${path}`;
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
    const routedId = matchRoutedSkillId(text);
    const routedDef = routedId ? skillDefs.find((s) => s.id === routedId) : undefined;
    if (routedDef?.documentRequired === "yes" && selectedYuqueDocs.length === 0) {
      setComposerDocGateHint(
        `当前问题会触发「${routedDef.quickFill.label}」能力，请先输入 @ 并选择至少一篇文档后再发送。`,
      );
      return;
    }
    setComposerDocGateHint("");
    // 允许用户使用 @ 选文档：后端基于文档标题做匹配时不需要这个前缀符号
    const payloadQuestion = text.replace(/(^|\s)@/g, "$1");
    const sessionId = activeSessionRef.current;
    const mid = ++messageIdSeqRef.current;
    const userId = `u-${mid}`;
    const assistantId = `a-${mid}`;
    setQuestion("");
    const docsForRequest = selectedYuqueDocs.filter((d) => d.docId >= 1);
    setSelectedYuqueDocs([]);
    closeDocSuggest();
    userStreamStopRef.current = false;
    setIsStreaming(true);
    setSessions((prev) =>
      prev.map((session) =>
        session.id === sessionId
          ? {
                  ...touchSession(session),
              messages: [
                ...session.messages,
                { id: userId, role: "user", text },
                {
                  id: assistantId,
                  role: "assistant",
                  text: "",
                  streamStage: "正在搜索知识库资料…",
                  streamElapsedSec: 0,
                  debug: "详细调试信息将在本轮结束后显示。",
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

    try {
      connectTimer = window.setTimeout(() => {
        controller.abort();
      }, STREAM_CONNECT_MS);

      const response = await fetch("/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: payloadQuestion,
          model: selectedModel,
          owner: yuqueOwnerForApi,
          token_profile: "primary",
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
                  ...touchSession(session),
                  messages: session.messages.map((item) =>
                    item.id === assistantId ? { ...item, streamElapsedSec: elapsedSec } : item
                  ),
                }
              : session
          )
        );
      }, 1000);

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
            const token = payload.token || "";
            setSessions((prev) =>
              prev.map((session) =>
                session.id === sessionId
                  ? {
                      ...touchSession(session),
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
            const sourceLinks = parseDoneSourceLinks(payload);
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
                              text:
                                (typeof payload.answer === "string" && payload.answer) ||
                                item.text ||
                                "没有返回回答。",
                              sourceLinks: sourceLinks.length > 0 ? sourceLinks : undefined,
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
                          streamStage: undefined,
                          streamElapsedSec: undefined,
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
    } catch (error: unknown) {
      const err = error as { name?: string; message?: string };
      const errMsg = typeof err?.message === "string" ? err.message : "";
      const idleStop = errMsg === "stream_idle_timeout";
      const userStop = err?.name === "AbortError" && userStreamStopRef.current;
      const connectAbort =
        err?.name === "AbortError" && !userStreamStopRef.current && elapsedSec === 0 && !idleStop;

      let fallbackText = "";
      let fallbackDebug = "";
      if (idleStop) {
        fallbackText =
          "等待超时：长时间未收到服务器新数据，已自动停止。若问题包含多图识读、语雀拉取或 MCP 多步调用，后台可能较慢；可稍后重试、缩短问题或关闭部分能力后再试。";
        fallbackDebug = `流式读取空闲超时（>${Math.floor(STREAM_READ_IDLE_MS / 1000)}s 无数据）。`;
      } else if (userStop) {
        fallbackText = "已停止生成。";
        fallbackDebug = "已手动停止流式输出。";
      } else if (connectAbort) {
        fallbackText =
          "连接超时：在限定时间内未收到服务器响应，请检查网络、后端是否已启动，或稍后重试。";
        fallbackDebug = `等待响应头超时（>${Math.floor(STREAM_CONNECT_MS / 1000)}s）。`;
      } else if (err?.name === "AbortError") {
        fallbackText = "请求已中断。";
        fallbackDebug = "连接被中止。";
      } else if (errMsg === "stream_unavailable") {
        fallbackText = "流式服务不可用（未收到有效响应），请确认后端已启动或稍后重试。";
        fallbackDebug = "stream_unavailable";
      } else {
        fallbackDebug = errMsg ? `错误: ${errMsg}` : "请求失败，无法获取调试信息。";
        fallbackText = errMsg || "请求失败，请稍后重试。";
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
                        debug: fallbackDebug,
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
      controllerRef.current = null;
      userStreamStopRef.current = false;
      setIsStreaming(false);
    }
  };

  const showDocSuggest = docSuggestOpen && docSuggestDocs.length > 0;
  const docFloatPortalOpen = showDocSuggest || kbPanelOpen;

  return (
    <>
    <main className={`layout ${sidebarCollapsed ? "layout-collapsed" : ""}`}>
      <aside className={`sidebar ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
        {sidebarCollapsed ? (
          <div className="sidebar-collapsed-stack">
            <button
              type="button"
              className="sidebar-icon-tile"
              onClick={() => setSidebarCollapsed(false)}
              title="展开侧栏"
              aria-label="展开侧栏"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
                <path d="M9 18l6-6-6-6" />
              </svg>
            </button>
            <button type="button" className="sidebar-icon-tile" onClick={createSession} title="新对话" aria-label="新对话">
              +
            </button>
          </div>
        ) : (
          <>
            <div className="sidebar-header-row">
              <div>
                <div className="brand">语雀 AI</div>
                <div className="sub">语雀 + MCP</div>
              </div>
              <button
                type="button"
                className="sidebar-collapse-btn"
                onClick={() => setSidebarCollapsed(true)}
                title="收起侧栏"
                aria-label="收起侧栏"
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
                  <path d="M15 18l-6-6 6-6" />
                </svg>
              </button>
            </div>

            <button
              type="button"
              className="btn-new-chat"
              onClick={createSession}
              title={`新对话（${newChatShortcutLabel}）`}
            >
              <span className="btn-new-chat-plus" aria-hidden>
                +
              </span>
              <span>新对话</span>
            </button>

            <div className="sidebar-scroll">
              <section className="sidebar-section">
                <div className="sidebar-section-head">
                  <span className="sidebar-section-title">历史会话</span>
                  <button type="button" className="sidebar-section-add" onClick={createSession} title="新建会话" aria-label="新建会话">
                    +
                  </button>
                </div>
                <ul className="session-list">
                  {orderedSessions.map((session) => (
                    <li key={session.id} className="session-item">
                      <div
                        className={`session-row${session.id === activeSessionId ? " session-row--active" : ""}`}
                      >
                        <button
                          type="button"
                          className="session-main"
                          onClick={() => {
                            setOpenSessionMenuId(null);
                            setActiveSessionId(session.id);
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

              <section className="sidebar-section sidebar-section--abilities">
                <button
                  type="button"
                  className="abilities-toggle-head"
                  onClick={() => setAbilitiesCollapsed((prev) => !prev)}
                  aria-expanded={!abilitiesCollapsed}
                >
                  <span className="sidebar-section-title">MCP &amp; Skills</span>
                  <span className="abilities-chevron" aria-hidden>
                    {abilitiesCollapsed ? "▼" : "▲"}
                  </span>
                </button>
                {!abilitiesCollapsed && (
                  <div className="abilities-body">
                    <div className="mcp-block">
                      <div className="subblock-head">
                        <span className="subblock-title">MCP 服务器</span>
                        <button type="button" className="link-like" title="由后端与 .env 配置">
                          管理
                        </button>
                      </div>
                      <p className="mcp-subline">{mcpMeta}</p>
                      {groupedTools.map(([group, tools]) => (
                        <div key={group} className="tool-group">
                          <div className="tool-group-title">{group}</div>
                          <ul className="tool-list tool-list--flat">
                            {tools.map((tool) => (
                              <li key={tool.name}>
                                <button
                                  type="button"
                                  className="mcp-server-row"
                                  onMouseEnter={(e) =>
                                    showAbilityHover(e.currentTarget, {
                                      kind: "mcp",
                                      name: tool.name,
                                      description: tool.description,
                                    })
                                  }
                                  onFocus={(e) =>
                                    showAbilityHover(e.currentTarget, {
                                      kind: "mcp",
                                      name: tool.name,
                                      description: tool.description,
                                    })
                                  }
                                  onMouseLeave={scheduleAbilityHoverHide}
                                  onBlur={scheduleAbilityHoverHide}
                                >
                                  <span className="status-dot status-dot-green" aria-hidden="true" />
                                  <span className="mcp-server-row-name">{tool.name}</span>
                                </button>
                              </li>
                            ))}
                          </ul>
                        </div>
                      ))}
                      <button type="button" className="add-ability-link" title="动态接入待产品化">
                        + 添加 MCP 服务器
                      </button>
                    </div>

                    <div className="skills-block">
                      <div className="subblock-head">
                        <span className="subblock-title">Skills 能力</span>
                        <button type="button" className="link-like" title="与后端 skill_router 对齐">
                          管理
                        </button>
                      </div>
                      <div className="tool-group">
                        <div className="tool-group-title">Skills 手册</div>
                        <ul className="tool-list tool-list--flat">
                          {skillDefs.map((skill) => (
                            <li key={skill.id}>
                              <div
                                className="mcp-server-row mcp-server-row--readonly"
                                role="presentation"
                                onMouseEnter={(e) =>
                                  showAbilityHover(e.currentTarget, {
                                    kind: "skill",
                                    name: skill.name,
                                    description: skill.description,
                                    triggers: skill.triggers,
                                  })
                                }
                                onMouseLeave={scheduleAbilityHoverHide}
                              >
                                <span className="status-dot status-dot-green" aria-hidden="true" />
                                <span className="mcp-server-row-name mcp-server-row-name--stack">
                                  <span className="mcp-server-row-label-cn">{skill.quickFill.label}</span>
                                  <span className="mcp-server-row-id">{skill.name}</span>
                                </span>
                              </div>
                            </li>
                          ))}
                        </ul>
                      </div>
                      <button type="button" className="add-ability-link" title="扩展 Skill 待产品化">
                        + 添加 Skill
                      </button>
                    </div>
                  </div>
                )}
              </section>
            </div>

            <div className="sidebar-user">
              <div className="sidebar-user-main">
                <div className="sidebar-user-avatar" aria-hidden>
                  {(yuqueSidebarLabel || "?").slice(0, 1).toUpperCase()}
                </div>
                <div className="sidebar-user-meta">
                  <div className="sidebar-user-name">{yuqueSidebarLabel || "—"}</div>
                  <div className="sidebar-user-hint">知识库上下文</div>
                </div>
              </div>
              <button type="button" className="sidebar-user-settings" title="设置（占位）" aria-label="设置">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
                  <circle cx="12" cy="12" r="3" />
                  <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
                </svg>
              </button>
            </div>
          </>
        )}
      </aside>

      <section className="chat-shell">
        <header className="chat-topbar">
          <div className="chat-content-inner chat-topbar-inner">
            <span className="chat-topbar-title">{activeSession?.title ?? "会话"}</span>
          </div>
        </header>
        <div className="chat-main">
          <div className={`chat-content-inner chat-body-inner${showWelcomeHero ? "" : " chat-body-inner--scroll"}`}>
          {showWelcomeHero ? (
            <div className="welcome-hero">
              <div className="welcome-sparkle" aria-hidden>
                ✨
              </div>
              <h1 className="welcome-title">你好，我是你的企业知识助手</h1>
              <p className="welcome-sub">基于语雀知识库与 MCP 能力，随时为你提供专业、准确的解答</p>
              <div className="welcome-cards">
                <button
                  type="button"
                  className="welcome-card"
                  onClick={() => setQuestion((q) => (q.trim() ? q : "请基于语雀知识库检索并回答："))}
                >
                  <span className="welcome-card-icon welcome-card-icon--book" aria-hidden />
                  <div className="welcome-card-text">
                    <div className="welcome-card-name">知识问答</div>
                    <div className="welcome-card-desc">基于知识库 精准问答</div>
                  </div>
                </button>
                <button
                  type="button"
                  className="welcome-card"
                  onClick={() =>
                    setQuestion((q) => (q.trim() ? q : "请先输入 @ 选择文档，再提炼关键信息与结构要点。"))
                  }
                >
                  <span className="welcome-card-icon welcome-card-icon--doc" aria-hidden />
                  <div className="welcome-card-text">
                    <div className="welcome-card-name">文档理解</div>
                    <div className="welcome-card-desc">深度解析文档 提炼关键信息</div>
                  </div>
                </button>
              </div>
            </div>
          ) : (
            <div className="chat-list chat-list--thread" ref={chatListRef}>
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
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {normalizeMarkdownAutolinks(item.text)}
                    </ReactMarkdown>
                  ) : null}
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
                {item.role === "assistant" && item.sourceLinks && item.sourceLinks.length > 0 ? (
                  <div className="msg-sources" aria-label="本轮参考来源">
                    <div className="msg-sources-title">资料链接</div>
                    <ul className="msg-sources-list">
                      {item.sourceLinks.map((s, idx) => (
                        <li key={`${item.id}-src-${idx}`}>
                          <div className="msg-sources-doc-title">{s.title}</div>
                          {s.url ? (
                            <div className="msg-sources-url-text" title={s.url}>
                              {s.url}
                            </div>
                          ) : null}
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}
                {item.role === "assistant" && (
                  <pre className="debug">{item.debug || "调试信息将在本轮回答完成后显示。"}</pre>
                )}
              </div>
            </div>
          ))}
            </div>
          )}
          </div>
        </div>

        <section className="composer composer--floating">
          <div className="composer-inner">
          {composerDocGateHint ? (
            <div className="composer-doc-gate-hint" role="alert">
              {composerDocGateHint}
            </div>
          ) : null}
          {selectedYuqueDocs.length > 0 ? (
            <div className="selected-docs selected-docs--above-card">
              <div className="selected-docs-label">已选择文档</div>
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
          <div className="composer-row">
            <aside className="composer-shortcut-toolbar" aria-label="Skill 快捷">
              <div className="composer-skill-stack" role="list">
                {skillDefs.map((skill) => {
                  const needsDoc = skill.documentRequired === "yes" && selectedYuqueDocs.length === 0;
                  const shortcutDisabled = isStreaming || needsDoc;
                  const skillTooltipText = shortcutDisabled
                    ? needsDoc
                      ? "需先在输入框输入 @ 并从知识库选择至少一篇文档后再使用此能力"
                      : "请等待当前回答完成后再试"
                    : `${skill.quickFill.label}（${skill.name}）：${skill.description}`;
                  return (
                    <span
                      key={skill.id}
                      className={`composer-skill-wrap${shortcutDisabled ? " composer-skill-wrap--disabled" : ""}`}
                      data-skill-tooltip={skillTooltipText}
                    >
                      <button
                        type="button"
                        className={`composer-skill-btn${shortcutDisabled ? " composer-skill-btn--disabled-hit" : ""}`}
                        disabled={shortcutDisabled}
                        aria-label={
                          shortcutDisabled
                            ? needsDoc
                              ? `${skill.quickFill.label}（需先 @ 选择知识库文档）`
                              : `${skill.quickFill.label}（生成中暂不可用）`
                            : `${skill.quickFill.label}：填入快捷问法`
                        }
                        onMouseEnter={(e) =>
                          showAbilityHover(e.currentTarget, {
                            kind: "skill",
                            name: skill.name,
                            description: skill.description,
                            triggers: skill.triggers,
                          })
                        }
                        onFocus={(e) =>
                          showAbilityHover(e.currentTarget, {
                            kind: "skill",
                            name: skill.name,
                            description: skill.description,
                            triggers: skill.triggers,
                          })
                        }
                        onMouseLeave={scheduleAbilityHoverHide}
                        onBlur={scheduleAbilityHoverHide}
                        onClick={() => applyQuickSkillShortcut(skill)}
                      >
                        {skill.quickFill.label}
                      </button>
                    </span>
                  );
                })}
              </div>
            </aside>
            <div className="composer-card-wrap">
          <div className="composer-card">
            <div ref={composerInputWrapRef} className="composer-input-wrap doc-suggest-wrap">
              <textarea
                className="composer-textarea composer-textarea--card"
                rows={4}
                placeholder="输入问题，Shift+Enter 换行，Enter 发送（支持 @ 选择文档）"
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
                    setKbPanelOpen(false);
                  }
                }}
              />
            </div>
            <div className="composer-card-footer">
              <div className="composer-pills">
                <label className="select-pill-label">
                  <span className="visually-hidden">模型</span>
                  <select
                    className="select-pill"
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    disabled={isStreaming}
                    aria-label="模型"
                  >
                    <option value="deepseek-chat">deepseek-chat</option>
                    <option value="deepseek-reasoner">deepseek-reasoner</option>
                    <option value="gpt-4o-mini">gpt-4o-mini</option>
                  </select>
                </label>
                <button
                  type="button"
                  ref={kbListPillRef}
                  className={`kb-list-pill${kbPanelOpen ? " kb-list-pill--open" : ""}`}
                  onClick={() => toggleKbPanel()}
                  disabled={isStreaming}
                  aria-expanded={kbPanelOpen}
                  aria-controls="kb-toc-panel"
                >
                  知识库列表
                </button>
              </div>
              <div className="composer-card-actions">
                <button type="button" className="mic-btn" disabled title="语音输入即将支持" aria-label="语音输入（即将支持）">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2M12 19v4M8 23h8" />
                  </svg>
                </button>
                <button
                  type="button"
                  className={`send-fab ${isStreaming ? "send-fab--stop" : ""}`}
                  onClick={() => {
                    if (isStreaming) stopStreaming();
                    else void askQuestion();
                  }}
                  title={isStreaming ? "停止生成" : "发送"}
                  aria-label={isStreaming ? "停止生成" : "发送"}
                >
                  <span className={isStreaming ? "stop-icon" : "send-icon"} aria-hidden="true" />
                </button>
              </div>
            </div>
          </div>
            </div>
          </div>
          </div>
        </section>
      </section>
    </main>
    {docFloatPortalOpen && docFloatRect
      ? createPortal(
          <div
            ref={docFloatRootRef}
            id={!showDocSuggest && kbPanelOpen ? "kb-toc-panel" : undefined}
            className="doc-suggest-box doc-suggest-box--portal"
            style={{
              bottom: docFloatRect.bottom,
              left: docFloatRect.left,
              width: docFloatRect.width,
              maxHeight: docFloatRect.maxHeight,
              top: "auto",
            }}
            role={showDocSuggest ? "listbox" : "region"}
            aria-label={showDocSuggest ? "@ 文档联想" : "语雀知识库目录"}
          >
            {showDocSuggest ? (
              docSuggestDocs.map((doc, idx) => {
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
              })
            ) : (
              <>
                {kbPanelLoading ? <div className="kb-toc-panel-status">正在加载目录…</div> : null}
                {kbPanelError ? (
                  <div className="kb-toc-panel-status kb-toc-panel-status--error" role="alert">
                    {kbPanelError}
                  </div>
                ) : null}
                {!kbPanelLoading && !kbPanelError && kbPanelDocs.length === 0 ? (
                  <div className="kb-toc-panel-status">暂无目录数据</div>
                ) : null}
                {!kbPanelLoading && kbPanelDocs.length > 0
                  ? kbPanelDocs.map((doc, idx) => {
                      const depth = Math.max(0, (doc.toc_level ?? 1) - 1);
                      const pad = 10 + depth * 14;
                      const rowClass = `doc-suggest-item kb-toc-item ${
                        doc.toc_kind === "title" ? "doc-suggest-item--toc-title" : ""
                      }`.trim();
                      if (!docMetaSelectable(doc)) {
                        return (
                          <div
                            key={doc.toc_uuid ?? `kb-h-${doc.title}-${idx}`}
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
                          key={doc.toc_uuid ?? doc.id ?? doc.slug ?? `kb-${doc.title}-${idx}`}
                          type="button"
                          className={rowClass}
                          style={{ paddingLeft: pad }}
                          onClick={() => pickDocFromKbPanel(doc)}
                        >
                          {doc.title}
                        </button>
                      );
                    })
                  : null}
              </>
            )}
          </div>,
          document.body,
        )
      : null}
    {abilityHover ? (
      <div
        role="tooltip"
        className="capability-floating-popover"
        style={{
          position: "fixed",
          top: abilityHover.top,
          left: abilityHover.left,
          maxWidth: abilityHover.maxWidth,
        }}
        onMouseEnter={cancelAbilityHoverHide}
        onMouseLeave={scheduleAbilityHoverHide}
      >
        <div className="capability-detail-title">
          {abilityHover.kind === "mcp" ? "MCP" : "Skill"}：{abilityHover.name}
        </div>
        <div className="capability-detail-desc">{abilityHover.description}</div>
        {abilityHover.triggers && abilityHover.triggers.length > 0 ? (
          <div className="capability-detail-triggers">
            <div className="capability-detail-subtitle">触发关键词</div>
            <div className="capability-detail-tags">
              {abilityHover.triggers.map((t) => (
                <span className="tag" key={t}>
                  {t}
                </span>
              ))}
            </div>
          </div>
        ) : null}
      </div>
    ) : null}
    </>
  );
}

export default App
