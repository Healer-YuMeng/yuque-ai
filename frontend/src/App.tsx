import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ChangeEvent } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

// 正文里的 Markdown 链接与图片统一在新标签页打开，避免离开当前对话页
const MARKDOWN_COMPONENTS: Components = {
  a(props) {
    const { node, ...rest } = props;
    void node;
    return <a {...rest} target="_blank" rel="noreferrer noopener" />;
  },
  img(props) {
    const { node, className, ...rest } = props;
    void node;
    const src = typeof rest.src === "string" ? rest.src : undefined;
    return (
      <a
        href={src}
        target="_blank"
        rel="noreferrer noopener"
        className={["msg-image-card", className].filter(Boolean).join(" ")}
      >
        <span className="msg-image-frame">
          <img {...rest} className="msg-image-card-media" />
        </span>
      </a>
    );
  },
};

import youweiLogo from "./assets/youwei-logo.png";
import { normalizeMarkdownAutolinks } from "./markdownAutolink";
import {
  INACTIVITY_MS,
  INACTIVITY_REMINDER_TEXT,
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
const STREAM_COMFORT_INITIAL_SEC = 5;
const STREAM_COMFORT_FOLLOWUP_SEC = 12;
const CHAT_V2_ENV_ENABLED = String(import.meta.env.VITE_CHAT_V2_ENABLED ?? "false").toLowerCase() === "true";
const CHAT_V3_ENV_ENABLED = String(import.meta.env.VITE_CHAT_V3_ENABLED ?? "false").toLowerCase() === "true";
const CHAT_V4_ENV_ENABLED = String(import.meta.env.VITE_CHAT_V4_ENABLED ?? "false").toLowerCase() === "true";
const CHAT_V5_ENV_ENABLED = String(import.meta.env.VITE_CHAT_V5_ENABLED ?? "false").toLowerCase() === "true";
/** 是否为访客专用入口：/visitor 开头的路径（布局等静态判断） */
const IS_VISITOR_ROUTE =
  typeof window !== "undefined" && window.location && window.location.pathname.startsWith("/visitor");
/** 访客端流式等待时统一展示的提示（替代各阶段文案、等待秒数、安抚卡片） */
const VISITOR_STREAM_STAGE_TEXT = "小为正在输入中......";

function isVisitorRoute(): boolean {
  if (typeof window === "undefined") return false;
  return window.location.pathname.startsWith("/visitor");
}

function isInactivityReminderMessage(item: ChatItem): boolean {
  return (
    item.isInactivityReminder === true ||
    item.id.startsWith("inact-") ||
    item.text === INACTIVITY_REMINDER_TEXT
  );
}
const FOCUS_SCENE_ITEMS = ["人工智能通识教育", "智能招生", "跨学科项目化学习", "学校AI场景定制"] as const;
type FocusScene = (typeof FOCUS_SCENE_ITEMS)[number];

const SCENE_TO_TOC_TITLE: Record<FocusScene, string> = {
  人工智能通识教育: "人工智能通识课程",
  跨学科项目化学习: "跨学科项目式学习",
  智能招生: "智能招生",
  学校AI场景定制: "学校AI场景定制",
};

type AdminSceneKey = "general_ai_course" | "project_based_learning" | "smart_enrollment" | "school_ai_custom";

const SCENE_TO_ADMIN_VIDEO_KEY: Record<FocusScene, AdminSceneKey> = {
  人工智能通识教育: "general_ai_course",
  跨学科项目化学习: "project_based_learning",
  智能招生: "smart_enrollment",
  学校AI场景定制: "school_ai_custom",
};

type AdminVideoPreviewAsset = {
  id: number;
  scene_key: AdminSceneKey;
  scene_name?: string;
  title?: string;
  original_filename?: string;
  file_url: string;
};

function resolveSceneForFriendV5Tag(tag: string, fallback: FocusScene): FocusScene {
  const patterns = [
    /^想看看(.+?)的产品的使用指南？$/,
    /^想看看(.+?)的产品的优秀案例库？$/,
    /^想申请测试账号，试一试(.+?)的产品？$/,
    /^想要了解一下(.+?)产品的价格？$/,
    /^想了解一下(.+?)产品？$/,
  ];
  for (const pattern of patterns) {
    const match = (tag || "").trim().match(pattern);
    const product = match?.[1]?.trim();
    if (!product) continue;
    for (const scene of FOCUS_SCENE_ITEMS) {
      const tocTitle = SCENE_TO_TOC_TITLE[scene];
      if (product === scene || product === tocTitle) return scene;
    }
  }
  return fallback;
}

function isFriendV5ExploreProductTag(tag: string): boolean {
  return /^想了解一下(.+?)产品？$/.test((tag || "").trim());
}

function friendV5VideoIntroForScene(scene: FocusScene): string {
  return `可以先看这段${SCENE_TO_TOC_TITLE[scene]}的演示视频，直观感受一下实际效果。`;
}

function isFriendV5TrialApplyTag(tag: string) {
  const normalized = tag.replace(/[\s？?，,。！!、：:；;]/g, "");
  return normalized.includes("申请测试账号") || (normalized.includes("试一试") && normalized.includes("产品"));
}

function friendV5TagDisplayText(tag: string): string {
  const normalized = tag.replace(/[[\s「」『』《》【】\]（）()、，,。！!？?:：;；\-_\\/|]/g, "");
  let label = tag.trim();
  if (normalized.includes("价格") || normalized.includes("报价") || normalized.includes("费用")) {
    label = "想了解一下价格？";
  } else if (normalized.includes("优秀案例库") || normalized.includes("产品案例") || normalized.includes("案例库")) {
    label = "想看看优秀案例库？";
  } else if (normalized.includes("申请测试账号") || (normalized.includes("试一试") && normalized.includes("产品"))) {
    label = "想申请测试账号，试试产品？";
  } else if (normalized.includes("使用指南") || normalized.includes("操作说明")) {
    label = "想看看使用指南？";
  }
  return label;
}

function renderFocusSceneIcon(scene: FocusScene) {
  if (scene === "人工智能通识教育") {
    return (
      <svg viewBox="0 0 24 24" focusable="false">
        <path d="M8.5 4.2c-1.8.4-3 1.8-3 3.5 0 .3 0 .6.1.9-1.2.7-2 1.9-2 3.4 0 1.8 1.2 3.4 2.9 3.9.2 2.1 1.8 3.7 3.9 3.7 1 0 1.8-.4 2.5-1 .7.6 1.5 1 2.5 1 2.1 0 3.7-1.6 3.9-3.7 1.7-.5 2.9-2.1 2.9-3.9 0-1.5-.8-2.8-2-3.4.1-.3.1-.6.1-.9 0-1.7-1.2-3.1-3-3.5-1.2-.3-2.4.1-3.3 1-.9-1-2.1-1.3-3.3-1Z" />
        <path d="M12.9 5.2v13.4M8.2 9.1c1.4.1 2.4 1.1 2.5 2.5M8.1 14.3c1.3-.1 2.3-1 2.6-2.2M17.7 9.1c-1.4.1-2.4 1.1-2.5 2.5M17.8 14.3c-1.3-.1-2.3-1-2.6-2.2" />
      </svg>
    );
  }
  if (scene === "智能招生") {
    return (
      <svg viewBox="0 0 24 24" focusable="false">
        <path d="M8 11a3 3 0 1 0 0-6 3 3 0 0 0 0 6ZM16.5 10a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5ZM3.5 19c.5-3.2 2.5-5 5.1-5s4.6 1.8 5.1 5M13.7 14.2c2.9-.4 5.5 1.3 6 4.8" />
      </svg>
    );
  }
  if (scene === "跨学科项目化学习") {
    return (
      <svg viewBox="0 0 24 24" focusable="false">
        <path d="m12 3.5 8 4.1-8 4.1-8-4.1 8-4.1Z" />
        <path d="m5.4 11.2 6.6 3.4 6.6-3.4M5.4 15.4l6.6 3.4 6.6-3.4" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 24 24" focusable="false">
      <path d="M4 21h16M6 21V9l6-4 6 4v12" />
      <path d="M9 21v-6h6v6M9 11h.1M12 11h.1M15 11h.1" />
      <path d="M12 5V3" />
    </svg>
  );
}

/** 开发者侧栏（运行追踪）；生产构建建议 VITE_SHOW_DEV_PANEL=false
 * 访客入口（/visitor）下强制关闭开发者面板。
 */
const SHOW_DEV_PANEL =
  !IS_VISITOR_ROUTE &&
  String(import.meta.env.VITE_SHOW_DEV_PANEL ?? "true").toLowerCase() === "true";
const MODEL_STORAGE_KEY = "visitor_selected_model_v1";
const DEFAULT_MODEL = "qwen3.7-plus";
const SUPPORTED_MODEL_OPTIONS = ["qwen3.7-max", "qwen3.7-plus", "deepseek-chat", "gpt-4o-mini"] as const;
const WELCOME_HERO_TITLE = "我是小为顾问，欢迎了解有为 AI 教育方案。";
const WELCOME_HERO_TITLE_V5 = "我是小为，咱们把 AI 教育这件事看明白。";
const WELCOME_HERO_SUBTEXT = "";

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

function normalizeSelectedModel(value: string): string {
  const model = (value || "").trim();
  return (SUPPORTED_MODEL_OPTIONS as readonly string[]).includes(model) ? model : DEFAULT_MODEL;
}

type ComfortPhase = "initial" | "followup";

type PendingComfortMessage = {
  text: string;
  phase: ComfortPhase;
};

function resolveComfortMessage(
  _scene: string | null,
  phase: ComfortPhase,
  friendV5 = false,
): PendingComfortMessage {
  const genericInitial = friendV5
    ? "我在按您的场景再看一遍资料，马上接着说。"
    : "我正在按您的情况核对资料，马上给您一个清楚的答复。";
  const genericFollowup = friendV5
    ? "这块内容我再核一下，尽量说得贴近您这边。"
    : "这块我再多确认一下，尽量给您说得更稳一些。";
  return {
    phase,
    text: phase === "initial" ? genericInitial : genericFollowup,
  };
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
  /** 仅用于发起后端请求，不在对话流中展示 */
  hidden?: boolean;
  media?: ChatMediaBundle;
  debug?: string;
  /** 流式阶段提示（来自 SSE event: stage），完成或出错后清除 */
  streamStage?: string;
  /** 本轮流式已等待秒数（前端计时，完成或出错后清除） */
  streamElapsedSec?: number;
  /** 仅前端等待态使用的临时安抚提示，不写入持久化会话 */
  pendingComfortMessage?: PendingComfortMessage;
  /** V4：是否展示「申请测试账号」按钮 */
  trialApplyAvailable?: boolean;
  trialCredentialsShown?: boolean;
  /** V5：推荐追问标签 */
  tags?: string[];
  /** V5：联网搜索与语雀补充阅读来源 */
  sources?: FriendV5SourceItem[];
  /** V5：搜索关键词 */
  searchKeywords?: string[];
  /** V5：部分场景需要先看视频，再读文字 */
  mediaDisplayMode?: "before_answer" | "after_answer";
  mediaIntro?: string;
  isFriendV5?: boolean;
  /** 访客无互动定时提醒，不计入新一轮助手回复 */
  isInactivityReminder?: boolean;
};

type FriendV5SourceItem = {
  source_type?: "web" | "yuque" | string;
  title?: string;
  url?: string | null;
  snippet?: string | null;
  index?: number | null;
  doc_id?: string | null;
};

type ChatMediaItem = {
  url: string;
  title?: string;
  doc_title?: string;
  doc_id?: string | null;
  summary?: string;
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
  /** 本会话已提交过测试账号申请，后续不再展示申请按钮 */
  trialApplicationSubmitted?: boolean;
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
function docMetaAnchorable(doc: DocMeta): boolean {
  return docMetaSelectable(doc) && Number.isInteger(doc.id) && Number(doc.id) >= 1;
}

function normalizeTocLevel(level: number | null | undefined): number {
  return Math.max(1, Math.min(3, level ?? 1));
}

type DocSuggestResponse = {
  docs: DocMeta[];
};

type TrialApplyResponse = {
  ok?: boolean;
  username?: string;
  password?: string;
  label?: string;
  message?: string;
};

type VisitorProfileResponse = {
  ok?: boolean;
  name?: string;
  org_name?: string;
  contact?: string;
  email?: string;
  interested_product?: string;
  concern?: string;
  module_scope?: string;
  trial_account_issued?: boolean;
};

type RuntimeModeResponse = {
  mode: "rag" | "direct_yuque";
  label: string;
  llm_model?: string;
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
          summary: typeof x.summary === "string" ? x.summary : "",
        } as ChatMediaItem;
      })
      .filter((x): x is ChatMediaItem => Boolean(x));
  };
  const images = normalize(raw.images);
  const videos = normalize(raw.videos);
  if (images.length === 0 && videos.length === 0) return undefined;
  return { images, videos };
}

function parseFriendV5MediaDisplay(debug: Record<string, unknown> | undefined): {
  mode?: "before_answer" | "after_answer";
  intro?: string;
} {
  const rawMode = typeof debug?.media_display_mode === "string" ? debug.media_display_mode : "";
  const mode = rawMode === "before_answer" || rawMode === "after_answer" ? rawMode : undefined;
  const intro = typeof debug?.media_intro === "string" ? debug.media_intro.trim() : "";
  return { mode, intro };
}

function parseFriendV5Tags(input: unknown): string[] {
  if (!Array.isArray(input)) return [];
  const tags: string[] = [];
  for (const item of input) {
    const tag = typeof item === "string" ? item.trim() : "";
    if (!tag || tags.includes(tag)) continue;
    tags.push(tag);
    if (tags.length >= 3) break;
  }
  return tags;
}

function parseFriendV5Sources(input: unknown): FriendV5SourceItem[] {
  if (!Array.isArray(input)) return [];
  return input
    .map((item) => {
      if (!item || typeof item !== "object") return null;
      const x = item as Record<string, unknown>;
      const title = typeof x.title === "string" ? x.title.trim() : "";
      const url = normalizeFriendV5SourceUrl(typeof x.url === "string" ? x.url : "");
      if (!title && !url) return null;
      return {
        source_type: typeof x.source_type === "string" ? x.source_type : "",
        title,
        url: url || null,
        snippet: typeof x.snippet === "string" ? x.snippet.trim() : null,
        index: typeof x.index === "number" ? x.index : null,
        doc_id: typeof x.doc_id === "string" ? x.doc_id : null,
      } as FriendV5SourceItem;
    })
    .filter((x): x is FriendV5SourceItem => Boolean(x));
}

function parseFriendV5SearchKeywords(input: unknown): string[] {
  const rawItems = Array.isArray(input) ? input : typeof input === "string" ? [input] : [];
  const out: string[] = [];
  for (const raw of rawItems) {
    if (typeof raw !== "string") continue;
    const parts = raw
      .split(/[\n,，、;；|]+/)
      .map((item) => item.trim().replace(/^["'“”‘’]+|["'“”‘’]+$/g, ""))
      .filter(Boolean);
    for (const part of parts) {
      if (!out.includes(part)) {
        out.push(part);
      }
    }
  }
  return out;
}

function normalizeFriendV5SourceUrl(input: string): string | null {
  const value = (input || "").trim();
  if (!value) return null;
  const yuqueRelative = normalizeYuqueRelativeUrl(value);
  if (yuqueRelative) return yuqueRelative;
  const match = value.match(
    /(?:https?:\/\/|www\.)[^\s\][<>{}"'，。；;：]+|(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,24}(?::\d+)?(?:\/[^\s\][<>{}"'，。；;：]*)?/,
  );
  if (!match) return null;
  const raw = match[0].replace(/[.,;:)\]}>，。、；;：]+$/g, "");
  const href = /^https?:\/\//i.test(raw) ? raw : `https://${raw}`;
  try {
    const url = new URL(href);
    if (url.hostname === "example.com" || url.hostname === "example.org" || url.hostname === "example.net") {
      return null;
    }
    if (url.protocol !== "http:" && url.protocol !== "https:") return null;
    const host = url.hostname.replace(/\.+$/, "");
    if (!host.includes(".")) return null;
    const tld = host.slice(host.lastIndexOf(".") + 1);
    if (tld.length < 2 || !/^[a-z]+$/i.test(tld)) return null;
    return url.href;
  } catch {
    return null;
  }
}

function normalizeYuqueRelativeUrl(input: string): string | null {
  const value = (input || "").trim();
  if (!value) return null;
  const match = value.match(/^\/?([a-zA-Z0-9._-]+\/[a-zA-Z0-9._-]+\/[^\s)\]}>，。、；;："'`]+)/);
  if (!match) return null;
  return `https://www.yuque.com/${match[1]}`;
}

function countFriendV5SearchKeywords(input: string[] | undefined): number {
  return Array.isArray(input) ? input.length : 0;
}

// 小为(v5) 正文里偶尔会泄漏 [SOURCES]/[/SOURCES] 标记与裸链接（来源已在下方“参考资料”里以文档标题超链接展示），正文不再直接显示链接
function stripHiddenMarkerFragments(text: string): string {
  const markerSuffixes = ["[SOURCES]", "[/SOURCES]", "[TAGS]", "[END_TAGS]"];
  let out = text || "";
  out = out.replace(
    new RegExp(
      String.raw`(?:\[\/SOURCES\]|\[SOURCES\]|\[\/SOURCES(?!\])|\[\/(?:TAGS|END_TAGS)(?=\]|$)|\[?(?:TAGS|END_TAGS)\]|(?<!\[)(?:\[/SOURCES\]|SOURCES?\]?|OURCES?\]?|URCES?\]?|RCES?\]?|CES?\]?|ES?\]?|S\]))`,
      "g",
    ),
    "",
  );
  while (out) {
    let stripped = false;
    for (const marker of markerSuffixes) {
      const maxLen = Math.min(out.length, marker.length - 1);
      for (let size = maxLen; size > 0; size -= 1) {
        if (marker.startsWith(out.slice(-size))) {
          out = out.slice(0, -size);
          stripped = true;
          break;
        }
      }
      if (stripped) break;
    }
    if (!stripped) break;
  }
  return out;
}

function sanitizeFriendV5AnswerText(text: string): string {
  const preservedLinks: string[] = [];
  let out = text || "";
  out = out.replace(/\[SOURCES\][\s\S]*?\[\/SOURCES\]/g, "");
  out = out.replace(/\[TAGS\][\s\S]*?\[END_TAGS\]/g, "");
  out = stripHiddenMarkerFragments(out);
  out = out.replace(/\[([^\]]+)\]\(((?:https?:\/\/|www\.)[^)]+|\/?[a-zA-Z0-9._-]+\/[a-zA-Z0-9._-]+\/[^)\s]+)\)/g, (_match, label: string, url: string) => {
    const normalized = normalizeFriendV5SourceUrl(url);
    if (!normalized) return label;
    const key = `__FRIEND_V5_MD_LINK_${preservedLinks.length}__`;
    preservedLinks.push(`[${label}](${normalized})`);
    return key;
  });
  out = out.replace(/(?:https?:\/\/|www\.)[^\s)\]}>"'，。、；;：]+/g, "");
  out = out.replace(/(?<!\()\/[a-zA-Z0-9._-]+\/[a-zA-Z0-9._-]+\/[^\s)\]}>"'，。、；;：]+/g, "");
  preservedLinks.forEach((link, index) => {
    out = out.replace(`__FRIEND_V5_MD_LINK_${index}__`, link);
  });
  out = out.replace(/^[ \t]*:\/\/[^\s]+[ \t]*$/gm, "");
  out = out.replace(/https?:\/\/\s*$/g, "");
  out = out.replace(/:\/\/\s*$/g, "");
  out = out.replace(/^[ \t]*(?::\/\/|https?:\/\/)[ \t]*$/gm, "");
  out = out.replace(/[ \t]*\[\^\d+\][ \t]*/g, "");
  out = out.replace(/更多正文\.{2,}/g, "");
  out = out.replace(/^[ \t]*正文\.{2,}[ \t]*/gm, "");
  out = out
    .split("\n")
    .filter((line) => {
      const s = line.trim();
      return !(s && /^[\s/:：。.、，,;；!！?？\-—_*·•]+$/.test(s));
    })
    .join("\n");
  out = out.replace(/[ \t]+([，。、；：！？）])/g, "$1");
  out = out.replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n");
  out = out.replace(/苹果\s*STAM(?!E)/gi, "苹果 STEAM");
  out = out.replace(/(?<!E)IDAS\s*-?\s*PBL/gi, "IDEAS-PBL");
  out = out.replace(/IDEAS\s+PBL/gi, "IDEAS-PBL");
  return stripTrialAccountDisclosure(out.trim());
}

function stripTrialAccountDisclosure(text: string): string {
  let out = text || "";
  out = out.replace(/信息校验通过[，,]?\s*已为您分配测试账号。?/g, "提交成功，我们会尽快与您联系。");
  out = out.replace(/【测试账号】[^\n]*/g, "");
  out = out.replace(/^[ \t]*(?:账号|密码|说明)[：:][^\n]*$/gm, "");
  out = out.replace(/\n{3,}/g, "\n\n");
  return out.trim();
}

function renderInlineMediaCard(item: ChatMediaItem, kind: "image" | "video", key: string) {
  const label = (item.title || item.doc_title || (kind === "image" ? "相关图片" : "相关视频")).trim();
  if (kind === "video") {
    return (
      <div key={key} className="msg-video-card">
        <video className="msg-video-player" controls playsInline preload="metadata" src={item.url} aria-label={label || "相关视频"} />
      </div>
    );
  }
  return (
    <a key={key} className="msg-image-card" href={item.url} target="_blank" rel="noreferrer" aria-label={label || "查看相关图片"}>
      <span className="msg-image-frame">
        <img className="msg-image-card-media" src={item.url} alt={label} loading="lazy" />
      </span>
    </a>
  );
}

function renderInlineMediaBlock(item: ChatItem, placement: "lead" | "tail") {
  if (!item.media) return null;
  const videos = item.media.videos.map((video, idx) => renderInlineMediaCard(video, "video", `${item.id}-video-${idx}`));
  const images = item.media.images.map((image, idx) => renderInlineMediaCard(image, "image", `${item.id}-img-${idx}`));
  return (
    <div className={`msg-inline-media msg-inline-media--${placement}`}>
      {placement === "lead" ? [...videos, ...images] : [...images, ...videos]}
    </div>
  );
}

function renderFriendV5Sources(sources: FriendV5SourceItem[] | undefined, searchKeywords: string[] | undefined) {
  if (!sources || sources.length === 0) return null;
  const normalizedSources = sources
    .map((item) => ({
      ...item,
      url: normalizeFriendV5SourceUrl(item.url || ""),
    }))
    .filter((item) => {
      const title = (item.title || "").trim();
      return Boolean(title || item.url);
    });
  const sourceItems: FriendV5SourceItem[] = [];
  const sourceSeen = new Set<string>();
  normalizedSources.forEach((item) => {
    const key = (item.url || `${item.source_type || ""}:${(item.title || "").trim()}`).trim().toLowerCase();
    if (!key || sourceSeen.has(key)) return;
    sourceSeen.add(key);
    sourceItems.push(item);
  });
  const totalCount = sourceItems.length;
  if (totalCount === 0) return null;
  const keywordItems = Array.isArray(searchKeywords) ? searchKeywords : [];
  const keywordCount = Math.max(1, countFriendV5SearchKeywords(keywordItems));

  return (
    <details className="friend-v5-sources">
      <summary className="friend-v5-sources-summary">
        <span className="friend-v5-sources-summary-text">{`搜索${keywordCount}个关键词，参考${totalCount}篇资料`}</span>
        <span className="friend-v5-sources-arrow" aria-hidden="true">
          ∨
        </span>
      </summary>
      <div className="friend-v5-sources-body">
        {keywordItems.length > 0 ? (
          <div className="friend-v5-sources-keywords">
            {`“${keywordItems.join("”、“")}”`}
          </div>
        ) : null}
        <ol className="friend-v5-sources-list">
          {sourceItems.map((item, idx) =>
            renderFriendV5SourceListItem(
              item,
              `src-${item.url || item.doc_id || item.title || idx}`,
              item.source_type === "yuque" ? "语雀文档" : "联网来源",
            ),
          )}
        </ol>
      </div>
    </details>
  );
}

function renderFriendV5SourceListItem(item: FriendV5SourceItem, key: string, fallbackTitle: string) {
  const displayTitle = (item.title || "").trim() || `${fallbackTitle}${typeof item.index === "number" ? ` ${item.index}` : ""}`;
  const body = <span className="friend-v5-source-link">{displayTitle}</span>;
  if (!item.url) {
    return (
      <li key={key} className="friend-v5-source-item">
        {body}
      </li>
    );
  }
  return (
    <li key={key} className="friend-v5-source-item">
      <a className="friend-v5-source-link" href={item.url} target="_blank" rel="noreferrer">
        {displayTitle}
      </a>
    </li>
  );
}

function itemTextWithGuideOffer(answer: string, guideTopicHit: boolean, offerLine: string): string {
  const base = (answer || "").trim();
  if (!guideTopicHit) return base;
  if (base.includes(offerLine)) return base;
  return base ? `${base}\n\n${offerLine}` : offerLine;
}

function looksLikeTrialApplyIntent(text: string): boolean {
  const q = (text || "").trim();
  return /(申请|开通|领取|获取).*(测试账号|试用账号)|(测试账号|试用账号).*(申请|开通|领取|获取)/.test(q);
}

function looksLikeAlreadyApplied(text: string): boolean {
  const q = (text || "").trim();
  return /(已经|已).*(申请过|提交过)|申请过了|提交过了|我已经申请/.test(q);
}

const INVALID_TRIAL_APPLY_NAME_PATTERNS = [
  /^(?:低年级|中年级|高年级|低中年级|中高年级)$/,
  /^(?:小学|初中|高中|大学)(?:阶段|年级)?$/,
  /^[一二三四五六七八九十]+年级$/,
  /^[0-9]+年级$/,
  /^(?:软件项目|软件编程|硬件搭建|信息课|社团)$/,
  /^(?:给|带|做|看)(?:小学|初中|高中|低年级|中年级|高年级|低中年级|中高年级|软件项目|软件编程|硬件搭建|社团).*$/,
  /^(?:学校|机构|培训机构|学校里|机构里)?(?:老师|教师|家长|学生|同学|校长|主任|负责人)$/,
];

const INVALID_TRIAL_APPLY_ORG_VALUES = [
  "老师",
  "教师",
  "家长",
  "学生",
  "同学",
  "校长",
  "主任",
  "负责人",
  "先生",
  "女士",
];

function sanitizeTrialApplyNameCandidate(raw: string): string {
  let name = (raw || "").trim().replace(/^[“"'《【（(<]+|[”"'》】）)>]+$/g, "");
  name = name.replace(/^(?:我是|我时|我叫|姓名是|名字是)/, "").trim();
  name = name.replace(/^[，,。；;：:\s]+|[，,。；;：:\s]+$/g, "");
  if (!name) return "";
  if (["老师", "教师", "家长", "学生", "同学", "校长", "先生", "女士"].includes(name)) return "";
  if (INVALID_TRIAL_APPLY_NAME_PATTERNS.some((pattern) => pattern.test(name))) return "";
  return name.slice(0, 24);
}

function sanitizeTrialApplyOrgCandidate(raw: string): string {
  const org = (raw || "")
    .trim()
    .replace(/^(?:我在|我于|我来自|来自)\s*/, "")
    .replace(/\s*(?:上班|工作|任职|就职)$/, "")
    .replace(/^[，,。；;：:\s]+|[，,。；;：:\s]+$/g, "");
  if (!org) return "";
  if (INVALID_TRIAL_APPLY_ORG_VALUES.includes(org)) return "";
  return org.slice(0, 40);
}

function extractTrialApplyEmail(userText: string): string {
  const strict = userText.match(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/i)?.[0] || "";
  if (strict) return strict.toLowerCase();
  return (
    userText.match(/(?:邮箱|邮件|email|e-mail)\s*(?:是|为)?\s*[:：]?\s*([A-Z0-9._%+-@]{2,80})/i)?.[1]?.trim() || ""
  );
}

function extractTrialApplyName(userText: string): string {
  const candidates = [
    userText.match(/(?:我叫|姓名是|名字是)\s*([^，,。；;\n]{1,12})/)?.[1] || "",
    userText.match(/(?:我是|我时)\s*([^，,。；;\n]{1,16}(?:老师|教师|校长|主任|先生|女士|家长|同学))/)?.[1] || "",
  ];
  for (const candidate of candidates) {
    const name = sanitizeTrialApplyNameCandidate(candidate);
    if (name) return name;
  }
  return "";
}

function extractTrialApplyDraft(session: SessionState | null, fallbackProduct: string): VisitorProfileResponse {
  const userText = (session?.messages || [])
    .filter((m) => m.role === "user")
    .map((m) => m.text || "")
    .join("\n");
  const phone = userText.match(/1[3-9]\d{9}/)?.[0] || "";
  const name = extractTrialApplyName(userText);
  const org = sanitizeTrialApplyOrgCandidate(
    userText.match(/(?:单位是|学校是|来自|在|办公地点是|办公地址是|办公单位是)\s*([^，,。；;\n]{2,40}(?:学校|学院|机构|中心|公司|集团|教育局)?)/)?.[1] || "",
  );
  const email = extractTrialApplyEmail(userText);
  const concern =
    [...(session?.messages || [])]
      .reverse()
      .find((m) => m.role === "user" && !looksLikeContactInUserMessage(m.text) && !looksLikeTrialApplyIntent(m.text))
      ?.text.trim()
      .slice(0, 180) || "";
  return {
    ok: true,
    name,
    org_name: org,
    contact: phone,
    email,
    interested_product: fallbackProduct,
    concern,
  };
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

function generateSessionId(chatMode: "visitor_sales" | "friend_v5" = "visitor_sales"): string {
  const uuid =
    typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  if (chatMode === "friend_v5") return `sess_v5_${uuid}`;
  return `s-${uuid}`;
}

function generateTurnMessageIds(sessionId: string, seq: number): { userId: string; assistantId: string } {
  const nonce = `${sessionId}-${Date.now()}-${seq}-${Math.random().toString(16).slice(2, 8)}`;
  return { userId: `u-${nonce}`, assistantId: `a-${nonce}` };
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
  delete copy.pendingComfortMessage;
  return copy;
}

function ensureUniqueMessageIds(messages: ChatItem[], sessionId: string): ChatItem[] {
  const seen = new Set<string>();
  return messages.map((message, idx) => {
    const baseId = String(message.id || `${message.role}-${idx}`);
    if (!seen.has(baseId)) {
      seen.add(baseId);
      return { ...message, id: baseId };
    }
    let nextId = `${baseId}-${sessionId}-${idx}`;
    while (seen.has(nextId)) {
      nextId = `${nextId}-${seen.size}`;
    }
    seen.add(nextId);
    return { ...message, id: nextId };
  });
}

function refreshVisitorWelcomeText(messages: ChatItem[]): ChatItem[] {
  return messages.flatMap((message) => {
    if (
      message.role === "assistant" &&
      (
        message.text.includes("我想先了解一下：您是学校或机构负责人、老师、学生，还是家长呢？") ||
        message.text.includes("您好，欢迎了解有为人工智能教育平台。") ||
        message.text.includes("您好，欢迎了解人工智能教育平台。") ||
        message.text.includes("我可以帮您介绍平台功能、适用场景、使用方式和案例。")
      )
    ) {
      return [];
    }
    return [message];
  });
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
    trialApplicationSubmitted: s.trialApplicationSubmitted ?? false,
  };
  if (s.messages && s.messages.length > 0) {
    return { ...s, messages: refreshVisitorWelcomeText(ensureUniqueMessageIds(s.messages, s.id)), ...flags };
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
  const friendV5Session = session.messages.some((item) => item.isFriendV5);
  const assistantName = friendV5Session ? "小为" : "小为顾问";
  const lines: string[] = [
    `「${title}」· ${assistantName}对话记录`,
    `导出时间：${new Date().toLocaleString("zh-CN", { hour12: false })}`,
    "",
  ];
  let hasContent = false;
  for (const item of session.messages) {
    if (item.hidden) continue;
    const text = (item.text || "").trim();
    const images = item.media?.images ?? [];
    const videos = item.media?.videos ?? [];
    if (!text && images.length === 0 && videos.length === 0) {
      continue;
    }
    hasContent = true;
    const roleLabel = item.role === "user" ? "访客" : assistantName;
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
        ...{ trialApplicationSubmitted: false },
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
      const newId = oldId === "default" ? oldId : generateSessionId();
      if (newId !== oldId) {
        idMap.set(oldId, newId);
      }
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

let initialStoredSessionState: { sessions: SessionState[]; activeSessionId: string } | null = null;

function getInitialStoredSessionState(): { sessions: SessionState[]; activeSessionId: string } {
  if (!initialStoredSessionState) {
    initialStoredSessionState = readStoredSessionState();
  }
  return initialStoredSessionState;
}

function App() {
  const [chatV2Enabled, setChatV2Enabled] = useState(CHAT_V2_ENV_ENABLED);
  const [chatV3Enabled, setChatV3Enabled] = useState(CHAT_V3_ENV_ENABLED);
  const [chatV4Enabled, setChatV4Enabled] = useState(CHAT_V4_ENV_ENABLED);
  const [chatV5Enabled, setChatV5Enabled] = useState(CHAT_V5_ENV_ENABLED);
  const [question, setQuestion] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [historySessionPicked, setHistorySessionPicked] = useState(false);
  const [selectedModel, setSelectedModel] = useState(() => {
    if (typeof window === "undefined") return DEFAULT_MODEL;
    const stored = window.localStorage.getItem(MODEL_STORAGE_KEY) || "";
    return normalizeSelectedModel(stored);
  });
  /** 左侧开发者面板收起（仅保留展开条） */
  const [devSidebarCollapsed, setDevSidebarCollapsed] = useState(false);
  const [activeSessionId, setActiveSessionId] = useState(() => getInitialStoredSessionState().activeSessionId);
  const [sessions, setSessions] = useState<SessionState[]>(() => getInitialStoredSessionState().sessions);
  /** 历史会话行「⋯」溢出菜单：同时只展开一行 */
  const [openSessionMenuId, setOpenSessionMenuId] = useState<string | null>(null);
  const [mcpData, setMcpData] = useState<MCPCapabilitiesResponse | null>(null);
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
  const [activeFocusScene, setActiveFocusScene] = useState<((typeof FOCUS_SCENE_ITEMS)[number]) | null>(null);
  const [hasPickedFocusScene, setHasPickedFocusScene] = useState(false);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [sessionCopied, setSessionCopied] = useState(false);
  const [guideSectionOpen, setGuideSectionOpen] = useState(false);
  const [selectedGuideNodeId, setSelectedGuideNodeId] = useState<string | null>(null);
  const [trialApplyDialogOpen, setTrialApplyDialogOpen] = useState(false);
  const [trialApplySubmitting, setTrialApplySubmitting] = useState(false);
  const [trialApplyError, setTrialApplyError] = useState("");
  const [trialApplyName, setTrialApplyName] = useState("");
  const [trialApplyOrg, setTrialApplyOrg] = useState("");
  const [trialApplyContact, setTrialApplyContact] = useState("");
  const [trialApplyEmail, setTrialApplyEmail] = useState("");
  const [trialApplySuccess, setTrialApplySuccess] = useState(false);
  const [visitorMobileSceneOpen, setVisitorMobileSceneOpen] = useState(false);
  const [welcomeHeroTitle, setWelcomeHeroTitle] = useState("");
  const controllerRef = useRef<AbortController | null>(null);
  /** 用户点击「停止」触发的 abort，与超时/空闲 abort 区分 */
  const userStreamStopRef = useRef(false);
  const chatListRef = useRef<HTMLDivElement | null>(null);
  const [streamingAssistantId, setStreamingAssistantId] = useState<string | null>(null);
  const activeSessionRef = useRef(activeSessionId);
  const questionRef = useRef(question);
  const isStreamingRef = useRef(isStreaming);
  const resetPendingRef = useRef<Map<string, Promise<void>>>(new Map());
  const sessionTitleSeqRef = useRef<number | null>(null);
  const v5PageRefreshInitDoneRef = useRef(false);
  const adminVideoPreviewCacheRef = useRef<Partial<Record<FocusScene, ChatMediaBundle | null>>>({});
  const [inactivityEpoch, setInactivityEpoch] = useState(0);
  const messageIdSeqRef = useRef(0);
  const floatingInitDoneRef = useRef(false);
  const composerTextareaRef = useRef<HTMLTextAreaElement | null>(null);

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
        const sanitizedMessages = s.messages.some((m) => m.pendingComfortMessage)
          ? s.messages.map((m) => (m.pendingComfortMessage ? { ...m, pendingComfortMessage: undefined } : m))
          : s.messages;
        if (s.id !== sid) return sanitizedMessages === s.messages ? s : { ...s, messages: sanitizedMessages };
        const hasUser = s.messages.some((m) => m.role === "user" && (m.text || "").trim());
        if (hasUser) {
          return sanitizedMessages === s.messages ? s : { ...s, messages: sanitizedMessages };
        }
        return { ...touchSession(s), messages: emptySessionMessages(`welcome-${sid}`) };
      }),
    );
  }, []);

  useEffect(() => {
    questionRef.current = question;
  }, [question]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(MODEL_STORAGE_KEY, selectedModel);
  }, [selectedModel]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const onStorage = (event: StorageEvent) => {
      if (event.key !== MODEL_STORAGE_KEY) return;
      const next = normalizeSelectedModel(event.newValue || "");
      if (next && next !== selectedModel) {
        setSelectedModel(next);
      }
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, [selectedModel]);

  useEffect(() => {
    isStreamingRef.current = isStreaming;
  }, [isStreaming]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const stored = (window.localStorage.getItem(MODEL_STORAGE_KEY) || "").trim();
    if (stored) return;
    let cancelled = false;
    const loadRuntimeModel = async () => {
      try {
        const resp = await fetch("/runtime-mode");
        if (!resp.ok) return;
        const data = (await resp.json()) as RuntimeModeResponse;
        const runtimeModel = normalizeSelectedModel(data.llm_model || "");
        if (!cancelled && runtimeModel) {
          setSelectedModel(runtimeModel);
        }
      } catch {
        // ignore
      }
    };
    void loadRuntimeModel();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const s = sessions.find((x) => x.id === activeSessionId);
      if (!s) return;
      const hasUserTurns = s.messages.some((m) => m.role === "user" && (m.text || "").trim());
      if (hasUserTurns) {
        setHasPickedFocusScene(true);
      }
      if (!s.contactCollected && s.messages.some((m) => m.role === "user" && looksLikeContactInUserMessage(m.text))) {
        setSessions((prev) => prev.map((item) => (item.id === activeSessionId ? { ...item, contactCollected: true } : item)));
      }
    }, 0);
    return () => window.clearTimeout(timer);
  }, [activeSessionId, sessions]);

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
                  { id: aid, role: "assistant", text: INACTIVITY_REMINDER_TEXT, isInactivityReminder: true },
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

  useEffect(() => {
    let cancelled = false;
    const detectV5Availability = async () => {
      try {
        const resp = await fetch("/chat/v5/capabilities");
        if (!resp.ok) {
          if (!cancelled) setChatV5Enabled(CHAT_V5_ENV_ENABLED);
          return;
        }
        const data = (await resp.json()) as { enabled?: boolean; model?: string };
        if (cancelled) return;
        if (data?.enabled === true) {
          setChatV5Enabled(true);
          if (data.model) setSelectedModel(normalizeSelectedModel(data.model));
          void fetch("/chat/v2/guide-titles?refresh=true").catch(() => {});
          return;
        }
        setChatV5Enabled(CHAT_V5_ENV_ENABLED);
      } catch {
        if (!cancelled) setChatV5Enabled(CHAT_V5_ENV_ENABLED);
      }
    };
    void detectV5Availability();
    return () => {
      cancelled = true;
    };
  }, []);

  const activeSession = useMemo(
    () => sessions.find((item) => item.id === activeSessionId) || null,
    [sessions, activeSessionId]
  );

  useEffect(() => {
    if (!IS_VISITOR_ROUTE || !chatV5Enabled || v5PageRefreshInitDoneRef.current) return;
    v5PageRefreshInitDoneRef.current = true;
    const id = generateSessionId("friend_v5");
    activeSessionRef.current = id;
    setActiveSessionId(id);
    setHistorySessionPicked(false);
    setActiveFocusScene(null);
    setHasPickedFocusScene(false);
    setSelectedYuqueDocs([]);
    setGuideSectionOpen(false);
    setSessions((prev) => {
      const startedSessions = prev.filter((session) =>
        session.messages.some((message) => message.role === "user" && (message.text || "").trim())
      );
      return [
        {
          id,
          title: "新对话",
          updatedAt: Date.now(),
          messages: emptySessionMessages(`welcome-${id}`),
          contactCollected: false,
          declinedContact: false,
          inactivityPromptSent: false,
          trialApplicationSubmitted: false,
        },
        ...startedSessions,
      ];
    });
  }, [chatV5Enabled]);

  const chatItems = useMemo(() => activeSession?.messages ?? [], [activeSession]);
  // 仅最新一轮助手回复展示推荐标签；无互动定时提醒不算新一轮
  const tagDisplayAssistantId = useMemo(() => {
    for (let i = chatItems.length - 1; i >= 0; i -= 1) {
      const item = chatItems[i];
      if (item.role !== "assistant") continue;
      if (isInactivityReminderMessage(item)) continue;
      if (item.isFriendV5 && item.tags && item.tags.length > 0) return item.id;
      return null;
    }
    return null;
  }, [chatItems]);
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

  /** 默认优先展示“新对话欢迎屏”；只有选择历史会话后，才按历史消息渲染 */
  const showWelcomeHero = useMemo(() => {
    if (historySessionPicked) return false;
    return !chatItems.some(
      (item) =>
        !item.hidden &&
        (
          (item.text || "").trim().length > 0 ||
          Boolean(item.streamStage) ||
          Boolean(item.pendingComfortMessage) ||
          Boolean(item.media && (item.media.images.length > 0 || item.media.videos.length > 0))
        ),
    );
  }, [chatItems, historySessionPicked]);
  const visitorChatVisible = useMemo(() => {
    if (!IS_VISITOR_ROUTE) return true;
    if (historySessionPicked || isStreaming) return true;
    if (!hasPickedFocusScene) return false;
    return chatItems.some(
      (item) =>
        !item.hidden &&
        (
          (item.text || "").trim().length > 0 ||
          Boolean(item.streamStage) ||
          Boolean(item.pendingComfortMessage) ||
          Boolean(item.media && (item.media.images.length > 0 || item.media.videos.length > 0))
        ),
    );
  }, [chatItems, hasPickedFocusScene, historySessionPicked, isStreaming]);
  const chatHasThreadContent = useMemo(() => {
    if (showWelcomeHero) return false;
    if (isStreaming) return true;
    return chatItems.some(
      (item) =>
        !item.hidden &&
        (
          (item.text || "").trim().length > 0 ||
          Boolean(item.streamStage) ||
          Boolean(item.pendingComfortMessage)
        ),
    );
  }, [chatItems, isStreaming, showWelcomeHero]);
  const assistantHasRenderedOutput = useMemo(() => {
    return chatItems.some(
      (item) =>
        item.role === "assistant" &&
        ((((item.text || "").trim().length > 0) ||
          Boolean(item.media && (item.media.images.length || item.media.videos.length))))
    );
  }, [chatItems]);

  useEffect(() => {
    const heroTitle = chatV5Enabled ? WELCOME_HERO_TITLE_V5 : WELCOME_HERO_TITLE;
    let startTimer: number | null = null;
    let typeTimer: number | null = null;
    if (!showWelcomeHero) {
      startTimer = window.setTimeout(() => setWelcomeHeroTitle(heroTitle), 0);
      return () => {
        if (startTimer != null) window.clearTimeout(startTimer);
      };
    }
    startTimer = window.setTimeout(() => {
      setWelcomeHeroTitle("");
      let index = 0;
      typeTimer = window.setInterval(() => {
        index += 1;
        setWelcomeHeroTitle(heroTitle.slice(0, index));
        if (index >= heroTitle.length && typeTimer != null) {
          window.clearInterval(typeTimer);
          typeTimer = null;
        }
      }, 55);
    }, 0);
    return () => {
      if (startTimer != null) window.clearTimeout(startTimer);
      if (typeTimer != null) window.clearInterval(typeTimer);
    };
  }, [chatV5Enabled, showWelcomeHero]);

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

  /** 输入框自适应高度：默认 44px（与发送按钮同高），随内容自动撑高至 max-height（CSS），超出后内部滚动 */
  useLayoutEffect(() => {
    const el = composerTextareaRef.current;
    if (!el) return;
    el.style.height = "44px";
    const next = Math.max(44, el.scrollHeight);
    el.style.height = `${next}px`;
  }, [question]);

  const removeSelectedYuqueDoc = (docId: number, title: string) => {
    setSelectedYuqueDocs((prev) => prev.filter((p) => !(p.docId === docId && p.title === title)));
  };

  const pickDocFromKbPanel = (doc: DocMeta) => {
    if (!docMetaSelectable(doc)) return;
    setSelectedYuqueDocs((prev) => upsertSelectedYuqueDoc(prev, doc));
    setComposerDocGateHint("");
  };

  const handleGuideEntryClick = useCallback(() => {
    setGuideSectionOpen((open) => !open);
    if (kbPanelDocs.length === 0) {
      void loadKbToc();
    }
  }, [kbPanelDocs.length, loadKbToc]);

  const handleTrialApplyEntryClick = useCallback(async () => {
    const sid = activeSessionRef.current;
    const fallback = extractTrialApplyDraft(activeSession, activeFocusScene || "");
    let profile = fallback;
    if (sid) {
      try {
        const resp = await fetch(`/visitor/profile?session_id=${encodeURIComponent(sid)}`);
        if (resp.ok) {
          const data = (await resp.json()) as VisitorProfileResponse;
          profile = {
            ...fallback,
            ...data,
            name: data.name || fallback.name,
            org_name: data.org_name || fallback.org_name,
            contact: data.contact || fallback.contact,
            email: data.email || fallback.email,
            interested_product: data.interested_product || fallback.interested_product,
            concern: data.concern || fallback.concern,
          };
        }
      } catch {
        profile = fallback;
      }
    }
    const safeName = sanitizeTrialApplyNameCandidate(profile.name || "");
    setTrialApplyName(safeName || "");
    setTrialApplyOrg(profile.org_name || "");
    setTrialApplyContact(profile.contact || "");
    setTrialApplyEmail(profile.email || "");
    setTrialApplyError("");
    setTrialApplySuccess(false);
    setTrialApplyDialogOpen(true);
  }, [activeFocusScene, activeSession]);

  const submitTrialApply = useCallback(async () => {
    const sid = activeSessionRef.current;
    if (!sid) return;
    setTrialApplyError("");
    setTrialApplySuccess(false);
    setTrialApplySubmitting(true);
    try {
      const draft = extractTrialApplyDraft(activeSession, activeFocusScene || "");
      const resp = await fetch("/visitor/trial/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sid,
          name: trialApplyName.trim(),
          org_name: trialApplyOrg.trim(),
          contact: trialApplyContact.trim(),
          email: trialApplyEmail.trim(),
          interested_product: draft.interested_product || activeFocusScene || "",
        }),
      });
      const data = (await resp.json()) as TrialApplyResponse;
      if (!resp.ok || !data.ok) {
        setTrialApplyError(data.message || "信息校验未通过，请检查后重试。");
        setLastPipelineDebug({ source: "visitor_trial_apply", error: data.message || "信息校验未通过" });
        return;
      }
      setSessions((prev) =>
        prev.map((session) =>
          session.id === sid
            ? {
                ...touchSession(session),
                trialApplicationSubmitted: true,
                messages: session.messages.map((item) =>
                  item.role === "assistant" ? { ...item, trialApplyAvailable: false } : item
                ),
              }
            : session
        )
      );
      setTrialApplySuccess(true);
      setTrialApplyName("");
      setTrialApplyOrg("");
      setTrialApplyContact("");
      setTrialApplyEmail("");
      window.setTimeout(() => {
        setTrialApplyDialogOpen(false);
        setTrialApplySuccess(false);
      }, 1600);
    } catch {
      const msg = "账号申请请求失败，请稍后重试。";
      setTrialApplyError(msg);
      setLastPipelineDebug({ source: "visitor_trial_apply", error: msg });
    } finally {
      setTrialApplySubmitting(false);
    }
  }, [
    activeFocusScene,
    activeSession,
    trialApplyContact,
    trialApplyEmail,
    trialApplyName,
    trialApplyOrg,
  ]);

  const handleQuestionChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
    setComposerDocGateHint("");
    setQuestion(event.target.value as string);
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
    setActiveFocusScene(null);
    setHasPickedFocusScene(false);
    const chatMode = chatV5Enabled ? "friend_v5" : "visitor_sales";
    const id = generateSessionId(chatMode);
    setSessions((prev) => {
      const title = nextSessionTitle();
      return [{ id, title, updatedAt: Date.now(), messages: emptySessionMessages(`welcome-${id}`) }, ...prev];
    });
    setActiveSession(id);
    // 服务端兜底：强制重置该 session，避免任何串话（即便曾经碰撞/复用过）
    const p = fetch("/chat/session/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: id, chat_mode: chatMode }),
    })
      .then(() => {})
      .catch(() => {})
      .finally(() => {
        resetPendingRef.current.delete(id);
      });
    resetPendingRef.current.set(id, p);
  }, [chatV5Enabled, nextSessionTitle, setActiveSession]);

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
    const friendV5Session = session.messages.some((item) => item.isFriendV5);
    const text = `「${session.title}」· ${friendV5Session ? "有为小为对话" : "有为销售顾问会话"}（仅本地存储）\n${origin}${path}`;
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

  const loadAdminSceneVideoPreview = async (scene: FocusScene): Promise<ChatMediaBundle | null> => {
    const cached = adminVideoPreviewCacheRef.current[scene];
    if (cached !== undefined) return cached;
    const sceneKey = SCENE_TO_ADMIN_VIDEO_KEY[scene];
    try {
      const resp = await fetch(`/admin-api/videos?scene_key=${encodeURIComponent(sceneKey)}`);
      if (!resp.ok) {
        adminVideoPreviewCacheRef.current[scene] = null;
        return null;
      }
      const data = (await resp.json()) as { items?: AdminVideoPreviewAsset[] };
      const video = (data.items || []).find((item) => item.file_url);
      if (!video) {
        adminVideoPreviewCacheRef.current[scene] = null;
        return null;
      }
      const media: ChatMediaBundle = {
        images: [],
        videos: [
          {
            url: video.file_url,
            title: video.title || video.original_filename || "",
            doc_title: video.scene_name || SCENE_TO_TOC_TITLE[scene],
            doc_id: `admin_video:${video.id}`,
            summary: "",
          },
        ],
      };
      adminVideoPreviewCacheRef.current[scene] = media;
      return media;
    } catch {
      adminVideoPreviewCacheRef.current[scene] = null;
      return null;
    }
  };

  const applyImmediateAdminVideoPreview = async (
    scene: FocusScene,
    sessionId: string,
    assistantId: string,
  ) => {
    const media = await loadAdminSceneVideoPreview(scene);
    if (!media) return;
    try {
      setSessions((prev) =>
        prev.map((session) =>
          session.id === sessionId
            ? {
                ...touchSession(session),
                messages: session.messages.map((item) =>
                  item.id === assistantId
                    ? {
                        ...item,
                        media,
                        mediaDisplayMode: "before_answer",
                        mediaIntro: friendV5VideoIntroForScene(scene),
                        isFriendV5: true,
                        pendingComfortMessage: undefined,
                      }
                    : item
                ),
              }
            : session
        )
      );
    } catch {
      // 视频只是首屏体验增强，失败时继续走后端流式回答。
    }
  };

  const askQuestion = async (
    presetQuestion?: string,
    bypassSceneGate = false,
    options?: { hideUserMessage?: boolean; triggerType?: "scene" | "tag" | "manual"; scene?: FocusScene },
  ) => {
    const text = (presetQuestion ?? question).trim();
    if (!text || isStreaming) return;
    if (!bypassSceneGate && !hasPickedFocusScene) return;
    const v5Scene = options?.scene || activeFocusScene;
    const isFriendV5Request = Boolean(chatV5Enabled && v5Scene && (bypassSceneGate || hasPickedFocusScene));
    const friendV5Scene = v5Scene || FOCUS_SCENE_ITEMS[0];
    const friendV5TriggerType: "scene" | "tag" | "manual" = options?.triggerType || "manual";
    const guideTopicHit = text.includes("使用指南") || selectedGuideNodeId !== null;
    const trialApplyIntentHit = looksLikeTrialApplyIntent(text);
    const alreadyAppliedHit = looksLikeAlreadyApplied(text);
    if (text.includes("使用指南")) {
      if (!guideSectionOpen) {
        setGuideSectionOpen(true);
      }
      if (kbPanelDocs.length === 0) {
        void loadKbToc();
      }
    }
    setComposerDocGateHint("");
    const payloadQuestion = text;
    let sessionId = activeSessionRef.current;
    if (isFriendV5Request && !sessionId.startsWith("sess_v5_")) {
      sessionId = generateSessionId("friend_v5");
      activeSessionRef.current = sessionId;
      setActiveSessionId(sessionId);
      setSessions((prev) => [
        {
          id: sessionId,
          title: nextSessionTitle(),
          updatedAt: Date.now(),
          messages: emptySessionMessages(`welcome-${sessionId}`),
          contactCollected: false,
          declinedContact: false,
          inactivityPromptSent: false,
          trialApplicationSubmitted: false,
        },
        ...prev,
      ]);
    }
    const { userId, assistantId } = generateTurnMessageIds(sessionId, ++messageIdSeqRef.current);
    const declineHit = looksLikeDeclineFollowup(text);
    const contactHit = looksLikeContactInUserMessage(text);
    setQuestion("");
    const docsForRequest = selectedYuqueDocs.filter((d) => d.docId >= 1);
    setSelectedYuqueDocs([]);
    const shouldShowImmediateSceneVideo =
      isFriendV5Request &&
      (friendV5TriggerType === "scene" || (friendV5TriggerType === "tag" && isFriendV5ExploreProductTag(text)));
    const cachedSceneVideo =
      shouldShowImmediateSceneVideo && adminVideoPreviewCacheRef.current[friendV5Scene]
        ? adminVideoPreviewCacheRef.current[friendV5Scene] || undefined
        : undefined;
    const streamPathForMeta = isFriendV5Request
      ? "/chat/v5/stream"
      : chatV4Enabled
      ? "/chat/v4/stream"
      : chatV3Enabled
        ? "/chat/v3/stream"
        : chatV2Enabled
          ? "/chat/v2/stream"
          : "/chat/stream";
    setLastRequestMeta({
      model: selectedModel,
      owner: yuqueOwnerForApi,
      chat_mode: isFriendV5Request ? "friend_v5" : "visitor_sales",
      token_profile: "primary",
      stream_path: streamPathForMeta,
    });
    userStreamStopRef.current = false;
    setStreamingAssistantId(assistantId);
    setIsStreaming(true);
    const visitorStreamUi = isVisitorRoute();
    const nextTurnMessages: ChatItem[] = [
      { id: userId, role: "user", text, hidden: options?.hideUserMessage },
      {
        id: assistantId,
        role: "assistant",
        text: "",
        media: cachedSceneVideo,
        mediaDisplayMode: cachedSceneVideo ? "before_answer" : undefined,
        mediaIntro: cachedSceneVideo ? friendV5VideoIntroForScene(friendV5Scene) : undefined,
        isFriendV5: isFriendV5Request,
        streamStage: visitorStreamUi
          ? VISITOR_STREAM_STAGE_TEXT
          : isFriendV5Request
            ? "小为正在看相关资料..."
            : "正在搜索知识库资料…",
        streamElapsedSec: 0,
      },
    ];
    setSessions((prev) => {
      let matched = false;
      const next = prev.map((session) => {
        if (session.id !== sessionId) return session;
        matched = true;
        return {
          ...touchSession(session),
          declinedContact: session.declinedContact || declineHit,
          contactCollected: session.contactCollected || contactHit,
          trialApplicationSubmitted: session.trialApplicationSubmitted || alreadyAppliedHit,
          messages: [...session.messages, ...nextTurnMessages],
        };
      });
      if (matched) return next;
      return [
        {
          id: sessionId,
          title: "默认会话",
          updatedAt: Date.now(),
          messages: [...emptySessionMessages(`welcome-${sessionId}`), ...nextTurnMessages],
          contactCollected: contactHit,
          declinedContact: declineHit,
          inactivityPromptSent: false,
          trialApplicationSubmitted: alreadyAppliedHit,
        },
        ...next,
      ];
    });

    if (shouldShowImmediateSceneVideo && !cachedSceneVideo) {
      void applyImmediateAdminVideoPreview(friendV5Scene, sessionId, assistantId);
    }

    const controller = new AbortController();
    controllerRef.current = controller;

    let connectTimer: number | null = null;
    let elapsedTimer: number | null = null;
    let elapsedSec = 0;
    let streamTextBuf = "";
    let streamFlushTimer: number | null = null;
    let pendingDonePayload: Record<string, unknown> | null = null;
    let receivedAnyToken = false;
    const comfortScene = activeFocusScene;

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
                        pendingComfortMessage: undefined,
                        ...(token.trim() ? { streamStage: undefined } : {}),
                      }
                    : item
                ),
              }
            : session
        )
      );
    };

    const applyMediaPreviewPayload = (payload: Record<string, unknown>) => {
      const media = parseChatMedia(payload.media);
      if (!media) return;
      const mediaDisplay = parseFriendV5MediaDisplay(payload);
      setSessions((prev) =>
        prev.map((session) =>
          session.id === sessionId
            ? {
                ...touchSession(session),
                messages: session.messages.map((item) =>
                  item.id === assistantId
                    ? {
                        ...item,
                        media,
                        mediaDisplayMode: mediaDisplay.mode,
                        mediaIntro: mediaDisplay.intro,
                        isFriendV5: isFriendV5Request,
                        pendingComfortMessage: undefined,
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
      const friendTags = isFriendV5Request ? parseFriendV5Tags(payload.tags) : [];
      const friendSources = isFriendV5Request ? parseFriendV5Sources(payload.sources) : [];
      const friendSearchKw = isFriendV5Request ? parseFriendV5SearchKeywords(payload.search_keywords) : [];
      const friendMediaDisplay = isFriendV5Request ? parseFriendV5MediaDisplay(dbg) : {};
      if (isFriendV5Request && typeof window !== "undefined") {
        console.log("[V5 done]", { sources: payload.sources, parsed: friendSources, search_keywords: friendSearchKw });
      }
      if (dbg && typeof dbg === "object") {
        setLastPipelineDebug(
          isFriendV5Request
            ? { ...dbg, v5_sources: friendSources, v5_search_keywords: friendSearchKw }
            : { ...dbg },
        );
      }
      const vs = dbg?.visitor_sales as Record<string, unknown> | undefined;
      const serverContact = Boolean(
        vs && vs.contact_detected === true
      ) || Boolean(dbg && dbg.contact_detected === true);
      const serverAnswer = typeof payload.answer === "string" ? payload.answer : "";
      const finalText = isFriendV5Request
        ? serverAnswer
        : itemTextWithGuideOffer(
            serverAnswer,
            guideTopicHit,
            "需要我给您提供这个模块的测试账号吗？",
          );
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
                        tags: friendTags,
                        sources: friendSources,
                        searchKeywords: friendSearchKw,
                        mediaDisplayMode: friendMediaDisplay.mode,
                        mediaIntro: friendMediaDisplay.intro,
                        isFriendV5: isFriendV5Request,
                        streamStage: undefined,
                        streamElapsedSec: undefined,
                        pendingComfortMessage: undefined,
                        trialApplyAvailable:
                          !isFriendV5Request &&
                          !session.trialApplicationSubmitted &&
                          (trialApplyIntentHit ||
                            looksLikeTrialApplyIntent(serverAnswer)),
                        text: isFriendV5Request
                          ? finalText || item.text || "没有返回回答。"
                          : item.text || finalText || "没有返回回答。",
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

      const streamPath = isFriendV5Request
        ? "/chat/v5/stream"
        : chatV4Enabled
        ? "/chat/v4/stream"
        : chatV3Enabled
          ? "/chat/v3/stream"
          : chatV2Enabled
            ? "/chat/v2/stream"
            : "/chat/stream";
      const requestBody = isFriendV5Request
        ? {
            question: payloadQuestion,
            model: selectedModel,
            owner: yuqueOwnerForApi,
            token_profile: "primary",
            chat_mode: "friend_v5",
            session_id: sessionId,
            scene: friendV5Scene,
            trigger_type: friendV5TriggerType,
          }
        : {
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
          };
      const response = await fetch(streamPath, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
        signal: controller.signal,
      });
      if (connectTimer != null) {
        window.clearTimeout(connectTimer);
        connectTimer = null;
      }
      if (!response.ok || !response.body) throw new Error("stream_unavailable");

      if (!visitorStreamUi) {
        elapsedTimer = window.setInterval(() => {
          elapsedSec += 1;
          setSessions((prev) =>
            prev.map((session) =>
              session.id === sessionId
                ? {
                    ...session,
                    messages: session.messages.map((item) =>
                      item.id === assistantId
                        ? {
                            ...item,
                            streamElapsedSec: elapsedSec,
                            pendingComfortMessage:
                              item.text.trim().length > 0
                                ? undefined
                                : elapsedSec >= STREAM_COMFORT_FOLLOWUP_SEC
                                  ? resolveComfortMessage(comfortScene, "followup", isFriendV5Request)
                                  : elapsedSec >= STREAM_COMFORT_INITIAL_SEC
                                    ? item.pendingComfortMessage?.phase === "followup"
                                      ? item.pendingComfortMessage
                                      : resolveComfortMessage(comfortScene, "initial", isFriendV5Request)
                                    : item.pendingComfortMessage,
                          }
                        : item
                    ),
                  }
                : session
            )
          );
        }, 1000);
      }

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
            const line = visitorStreamUi
              ? VISITOR_STREAM_STAGE_TEXT
              : formatStreamStageLine(stage, detail);
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
          } else if (parsed.event === "media_preview") {
            const payload = JSON.parse(parsed.data || "{}") as Record<string, unknown>;
            applyMediaPreviewPayload(payload);
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
                        pendingComfortMessage: undefined,
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
            body: JSON.stringify({ session_id: sessionId, chat_mode: isFriendV5Request ? "friend_v5" : "visitor_sales" }),
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
                        pendingComfortMessage: undefined,
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

  const handleFocusSceneShortcut = (scene: (typeof FOCUS_SCENE_ITEMS)[number]) => {
    if (isStreaming) return;
    setVisitorMobileSceneOpen(false);
    setActiveFocusScene(scene);
    if (IS_VISITOR_ROUTE) {
      setHasPickedFocusScene(true);
    } else {
      setHasPickedFocusScene(true);
    }
    setSelectedGuideNodeId(null);
    setSelectedYuqueDocs([]);
    if (IS_VISITOR_ROUTE) void loadAdminSceneVideoPreview(scene);
    if (chatV5Enabled) {
      void askQuestion(scene, true, {
        hideUserMessage: false,
        triggerType: "scene",
        scene,
      });
      return;
    }
    void askQuestion(`我想要咨询${scene}的内容，请帮我解答。`, true, {
      triggerType: "scene",
      scene,
    });
  };

  const turnTrace = parseTurnTrace(lastPipelineDebug);
  const pipelineMode = typeof lastPipelineDebug?.mode === "string" ? lastPipelineDebug.mode : turnTrace?.pipeline;
  const v5DebugSources = parseFriendV5Sources(lastPipelineDebug?.v5_sources);
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

  const guidePanelDocs = useMemo(() => {
    if (!kbPanelDocs.length) return [] as DocMeta[];
    const root = kbPanelDocs.find((doc) => (doc.title || "").trim() === "使用指南");
    if (!root) return [] as DocMeta[];
    const rootUuid = (root.toc_uuid || "").trim();
    if (!rootUuid) return [root];
    const childMap = new Map<string, DocMeta[]>();
    kbPanelDocs.forEach((doc) => {
      const parent = (doc.toc_parent_uuid || "").trim();
      if (!parent) return;
      const list = childMap.get(parent);
      if (list) list.push(doc);
      else childMap.set(parent, [doc]);
    });
    const out: DocMeta[] = [];
    const stack: DocMeta[] = [root];
    const seen = new Set<string>();
    while (stack.length) {
      const node = stack.shift()!;
      const id = (node.toc_uuid || "").trim() || `${node.id ?? node.slug ?? node.title}`;
      if (seen.has(id)) continue;
      seen.add(id);
      out.push(node);
      const children = childMap.get((node.toc_uuid || "").trim()) || [];
      children.forEach((child) => stack.push(child));
    }
    return out;
  }, [kbPanelDocs]);

  const guideTreeRows = useMemo(() => {
    if (!guidePanelDocs.length) return [];
    const allowed = new Set(
      guidePanelDocs.map((doc) => (doc.toc_uuid || "").trim() || `${doc.id ?? doc.slug ?? doc.title}`),
    );
    return kbTreeRows.filter((row) =>
      allowed.has((row.doc.toc_uuid || "").trim() || `${row.doc.id ?? row.doc.slug ?? row.doc.title}`),
    );
  }, [guidePanelDocs, kbTreeRows]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
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
    }, 0);
    return () => window.clearTimeout(timer);
  }, [kbPanelDocs, kbNodeId]);

  return (
    <>
      <div
        className={`app-shell${
          IS_VISITOR_ROUTE
            ? ` app-shell--visitor${visitorChatVisible ? " app-shell--visitor-chat-active" : ""}${visitorMobileSceneOpen ? " app-shell--visitor-scene-open" : ""}`
            : !SHOW_DEV_PANEL
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
              <span className="consult-top-brand-divider" aria-hidden="true" />
              <span className="consult-top-brand-title">人工智能教育平台</span>
            </div>
          </div>
        </header>

        <div className="app-body">
          {IS_VISITOR_ROUTE && visitorMobileSceneOpen ? (
            <button
              type="button"
              className="visitor-scene-overlay-backdrop"
              aria-label="关闭场景选择"
              onClick={() => setVisitorMobileSceneOpen(false)}
            />
          ) : null}
          {IS_VISITOR_ROUTE ? (
            <div
              className={`visitor-scene-column${visitorMobileSceneOpen ? " visitor-scene-column--overlay" : ""}`}
              aria-label="你关注的场景"
            >
              {visitorMobileSceneOpen ? (
                <button
                  type="button"
                  className="visitor-scene-overlay-close"
                  aria-label="关闭场景选择"
                  onClick={() => setVisitorMobileSceneOpen(false)}
                >
                  ×
                </button>
              ) : null}
              <div className="focus-scene-title-card focus-scene-title-card--visitor">
                <div className="focus-scene-title">你关注的场景</div>
              </div>
              <aside className="focus-scene-sidebar focus-scene-sidebar--visitor">
                <div className="focus-scene-list">
                  {FOCUS_SCENE_ITEMS.map((scene, idx) => (
                    <button
                      key={scene}
                      type="button"
                      className={`focus-scene-btn${activeFocusScene === scene ? " focus-scene-btn--active" : ""} focus-scene-btn--visitor-drop`}
                      onClick={() => handleFocusSceneShortcut(scene)}
                      disabled={isStreaming}
                      style={{ animationDelay: `${120 + idx * 90}ms` }}
                    >
                      <span className="focus-scene-btn-icon" aria-hidden="true">
                        {renderFocusSceneIcon(scene)}
                      </span>
                      <span className="focus-scene-btn-label">{scene}</span>
                      <span className="focus-scene-btn-dot" aria-hidden="true" />
                    </button>
                  ))}
                </div>
              </aside>
            </div>
          ) : (
            <aside className="focus-scene-sidebar" aria-label="你关注的场景">
              <div className="focus-scene-title-card">
                <div className="focus-scene-title">你关注的场景</div>
              </div>
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
              <div className="focus-extra-block">
                <button
                  type="button"
                  className="focus-extra-toggle"
                  onClick={handleGuideEntryClick}
                  disabled={isStreaming}
                  aria-expanded={guideSectionOpen}
                >
                  <span className={`focus-extra-arrow${guideSectionOpen ? "" : " focus-extra-arrow--collapsed"}`} aria-hidden>
                    ▾
                  </span>
                  <span>使用指南</span>
                </button>
                {guideSectionOpen ? (
                  <div className="focus-guide-panel">
                    {kbPanelLoading ? (
                      <p className="focus-guide-hint">正在加载指南目录…</p>
                    ) : kbPanelError ? (
                      <p className="focus-guide-error">{kbPanelError}</p>
                    ) : (
                      <div className="focus-guide-list">
                        {guideTreeRows.map(({ doc, level, nodeId, hasChildren, isCollapsed }) => {
                          const selectable = docMetaAnchorable(doc);
                          const expandable = hasChildren;
                          return (
                            <button
                              key={`focus-guide-${nodeId}`}
                              type="button"
                              className={`focus-guide-row${selectable ? " focus-guide-row--doc" : " focus-guide-row--title"}${
                                selectedGuideNodeId === nodeId ? " focus-guide-row--active" : ""
                              }`}
                              style={{ paddingLeft: `${10 + level * 14}px` }}
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
                                  setSelectedGuideNodeId(nodeId);
                                  setSelectedYuqueDocs([{ docId: Number(doc.id), title: doc.title, slug: doc.slug || null }]);
                                  window.setTimeout(() => setSelectedGuideNodeId((prev) => (prev === nodeId ? null : prev)), 0);
                                }
                              }}
                            >
                              {expandable ? (
                                <span className={`focus-guide-arrow${isCollapsed ? " focus-guide-arrow--collapsed" : ""}`} aria-hidden>
                                  ▾
                                </span>
                              ) : (
                                <span className="focus-guide-arrow focus-guide-arrow--spacer" aria-hidden />
                              )}
                              <span>{doc.title}</span>
                            </button>
                          );
                        })}
                      </div>
                    )}
                  </div>
                ) : null}
                {!chatV5Enabled ? (
                  <button
                    type="button"
                    className="focus-extra-apply"
                    onClick={handleTrialApplyEntryClick}
                    disabled={isStreaming}
                  >
                    账号申请
                  </button>
                ) : null}
              </div>
            </aside>
          )}
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
                        <option value="qwen3.7-max">qwen3.7-max</option>
                        <option value="qwen3.7-plus">qwen3.7-plus</option>
                        <option value="deepseek-chat">deepseek-chat</option>
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
                    ) : v5DebugSources.length > 0 ? (
                      <table className="dev-trace-table">
                        <thead>
                          <tr>
                            <th>标题</th>
                            <th>角色</th>
                            <th>预览</th>
                          </tr>
                        </thead>
                        <tbody>
                          {v5DebugSources.map((d, idx) => (
                            <tr key={`${d.doc_id || d.url || d.title}-${idx}`}>
                              <td>
                                <div>{d.title || "—"}</div>
                                {d.doc_id ? <div className="dev-mono dev-small">{d.doc_id}</div> : null}
                              </td>
                              <td>{d.source_type === "yuque" ? "yuque" : "web"}</td>
                              <td className="dev-trace-snippet">{d.snippet || d.url || "—"}</td>
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

          <div className={`app-main chat-shell${IS_VISITOR_ROUTE ? " chat-shell--visitor-compact" : ""}`}>
            {IS_VISITOR_ROUTE ? null : (
              <>
                <header className="chat-subbar">
                  <div className="chat-subbar-title-wrap">
                    <div className="chat-subbar-brand">
                      <span className="chat-subbar-brand-logo" aria-hidden>
                        AI
                      </span>
                      <div className="chat-subbar-brand-meta">
                        <span className="chat-subbar-brand-en">{chatV5Enabled ? "YOUWEI XIAOWEI" : "YOUWEI AI CONSULTANT"}</span>
                        <span className="chat-subbar-title">{chatV5Enabled ? "小为" : "小为顾问"}</span>
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
              </>
            )}
            {!IS_VISITOR_ROUTE ? <div className="chat-consulting-direction" aria-label="当前咨询方向">
              <span className="chat-consulting-direction-label">当前咨询方向</span>
              <span className="chat-consulting-direction-tag">{activeFocusScene || "待选择场景"}</span>
            </div> : null}
            {visitorChatVisible ? (
            <div
              className={`chat-main${IS_VISITOR_ROUTE ? " chat-main--visitor-reveal" : ""}${chatHasThreadContent ? " chat-main--content-fit" : ""}${selectedYuqueDocs.length > 0 ? " chat-main--has-selected-docs" : ""}`}
            >
              {IS_VISITOR_ROUTE && activeFocusScene ? (
                <div className="visitor-mobile-scene-bar">
                  <div className="visitor-mobile-scene-bar-main">
                    <span className="visitor-mobile-scene-bar-label">当前场景</span>
                    <span className="visitor-mobile-scene-bar-scene">{activeFocusScene}</span>
                  </div>
                  <button
                    type="button"
                    className="visitor-mobile-scene-bar-switch"
                    onClick={() => setVisitorMobileSceneOpen(true)}
                    disabled={isStreaming}
                  >
                    换场景
                  </button>
                </div>
              ) : null}
              <div
                className={`chat-content-inner chat-body-inner${showWelcomeHero ? "" : " chat-body-inner--scroll"}${chatHasThreadContent ? " chat-body-inner--thread-active" : ""}${IS_VISITOR_ROUTE && !assistantHasRenderedOutput ? " chat-body-inner--waiting-assistant" : ""}`}
              >
                {showWelcomeHero ? (
                  IS_VISITOR_ROUTE ? <div className="welcome-hero welcome-hero--empty" aria-hidden /> : (
                    <div className="welcome-hero">
                      <div className="welcome-logo-wrap" aria-hidden>
                        <img className="welcome-logo" src={youweiLogo} alt="" />
                      </div>
                      <h1 className="welcome-title">
                        {welcomeHeroTitle}
                        <span className="welcome-title-cursor" aria-hidden>
                          |
                        </span>
                      </h1>
                      {WELCOME_HERO_SUBTEXT ? <p className="welcome-sub">{WELCOME_HERO_SUBTEXT}</p> : null}
                      {!chatV5Enabled ? <div className="welcome-cards">
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
                      </div> : null}
                    </div>
                  )
                ) : (
                  <div className="chat-list chat-list--thread" ref={chatListRef} key={activeSessionId}>
                    {chatItems.map((item) => {
                      const hasInlineMedia = Boolean(
                        item.role === "assistant" &&
                          item.media &&
                          (item.media.videos.length > 0 || item.media.images.length > 0)
                      );
                      let displayText = item.text || "";
                      if (item.role === "assistant") {
                        if (item.isFriendV5) {
                          displayText = sanitizeFriendV5AnswerText(displayText);
                        } else if (IS_VISITOR_ROUTE) {
                          displayText = stripTrialAccountDisclosure(displayText);
                        }
                      }
                      const hasBubbleContent = displayText.trim().length > 0 || hasInlineMedia;
                      const shouldLeadWithMedia =
                        item.role === "assistant" &&
                        item.isFriendV5 &&
                        item.mediaDisplayMode === "before_answer" &&
                        hasInlineMedia;
                      const showInactivityTrialApplyButton =
                        item.role === "assistant" &&
                        isInactivityReminderMessage(item) &&
                        !item.isFriendV5 &&
                        !item.trialCredentialsShown &&
                        !activeSession?.trialApplicationSubmitted;
                      const showRegularTrialApplyButton =
                        item.role === "assistant" &&
                        item.trialApplyAvailable &&
                        !isInactivityReminderMessage(item) &&
                        !item.isFriendV5 &&
                        !item.trialCredentialsShown &&
                        !activeSession?.trialApplicationSubmitted;
                      if (item.hidden) return null;
                      return (
                      <div
                        className={`msg ${item.role}${isInactivityReminderMessage(item) ? " msg--inactivity" : ""}`}
                        key={item.id}
                      >
                        <div style={{ width: "100%" }}>
                          {item.role === "assistant" && item.isFriendV5 ? renderFriendV5Sources(item.sources, item.searchKeywords) : null}
                          {item.role === "assistant" &&
                          (isVisitorRoute()
                            ? !hasBubbleContent && isStreaming && item.id === streamingAssistantId
                            : Boolean(item.streamStage) || (item.streamElapsedSec ?? 0) > 0) ? (
                            <div className="stream-stage-block">
                              {isVisitorRoute() ? (
                                <div className="stream-stage stream-stage--pending">{VISITOR_STREAM_STAGE_TEXT}</div>
                              ) : (
                                <>
                                  {item.streamStage ? (
                                    <div className="stream-stage stream-stage--pending">{item.streamStage}</div>
                                  ) : null}
                                  {(item.streamElapsedSec ?? 0) > 0 ? (
                                    <div className="stream-elapsed">已等待 {item.streamElapsedSec} 秒</div>
                                  ) : null}
                                  {item.pendingComfortMessage ? (
                                    <div className="stream-comfort-card">
                                      <div className="stream-comfort-label">
                                        {item.isFriendV5 ? "小为再看一眼资料" : "小为顾问正在核对资料"}
                                      </div>
                                      <div className="stream-comfort-text">{item.pendingComfortMessage.text}</div>
                                    </div>
                                  ) : null}
                                </>
                              )}
                            </div>
                          ) : null}
                          {hasBubbleContent ? (
                            <div className="bubble">
                              {hasInlineMedia ? (
                                <div className="msg-rich-content">
                                  {shouldLeadWithMedia ? (
                                    <>
                                      {item.mediaIntro ? <div className="msg-media-intro">{item.mediaIntro}</div> : null}
                                      {renderInlineMediaBlock(item, "lead")}
                                      {isStreaming && item.role === "assistant" && item.id === streamingAssistantId ? (
                                        <div className="stream-stage stream-stage--pending msg-media-stream-stage">
                                          {isVisitorRoute() ? VISITOR_STREAM_STAGE_TEXT : item.streamStage || "小为正在看相关资料..."}
                                        </div>
                                      ) : null}
                                    </>
                                  ) : null}
                                  {displayText.trim() ? (
                                    <ReactMarkdown remarkPlugins={[remarkGfm]} components={MARKDOWN_COMPONENTS}>
                                      {normalizeMarkdownAutolinks(displayText)}
                                    </ReactMarkdown>
                                  ) : null}
                                  {shouldLeadWithMedia ? null : renderInlineMediaBlock(item, "tail")}
                                </div>
                              ) : displayText.trim() ? (
                                isStreaming && item.role === "assistant" && item.id === streamingAssistantId ? (
                                  <div className="bubble-stream-text">{displayText}</div>
                                ) : (
                                  <ReactMarkdown remarkPlugins={[remarkGfm]} components={MARKDOWN_COMPONENTS}>
                                    {normalizeMarkdownAutolinks(displayText)}
                                  </ReactMarkdown>
                                )
                              ) : null}
                            </div>
                          ) : null}
                          {showInactivityTrialApplyButton ? (
                            <div className="msg-inline-action">
                              <button
                                type="button"
                                className="trial-apply-button"
                                onClick={() => void handleTrialApplyEntryClick()}
                              >
                                申请测试账号
                              </button>
                            </div>
                          ) : null}
                          {item.role === "assistant" && item.isFriendV5 && item.id === tagDisplayAssistantId && item.tags && item.tags.length > 0 ? (
                            <div className="friend-v5-tags" aria-label="小为推荐的继续了解方向">
                              {item.tags.map((tag) => (
                                <button
                                  key={`${item.id}-tag-${tag}`}
                                  type="button"
                                  className="friend-v5-tag-chip"
                                  onClick={() => {
                                    if (isFriendV5TrialApplyTag(tag)) {
                                      void handleTrialApplyEntryClick();
                                      return;
                                    }
                                    const scene = resolveSceneForFriendV5Tag(
                                      tag,
                                      activeFocusScene || FOCUS_SCENE_ITEMS[0],
                                    );
                                    void askQuestion(tag, true, { triggerType: "tag", scene });
                                  }}
                                  disabled={isStreaming}
                                >
                                  {friendV5TagDisplayText(tag)}
                                </button>
                              ))}
                            </div>
                          ) : null}
                          <div className="msg-footer">
                            {showRegularTrialApplyButton ? (
                              <button
                                type="button"
                                className="trial-apply-button"
                                onClick={() => void handleTrialApplyEntryClick()}
                              >
                                申请测试账号
                              </button>
                            ) : null}
                            {!IS_VISITOR_ROUTE ? (
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
                            ) : null}
                          </div>
                        </div>
                      </div>
                    )})}
                  </div>
                )}
              </div>
              <footer className={`composer-simple${IS_VISITOR_ROUTE ? " composer-simple--visitor" : ""}`}>
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
                    <div className="composer-simple-field">
                      <textarea
                        ref={composerTextareaRef}
                        className="composer-simple-textarea"
                        rows={1}
                        placeholder={
                          IS_VISITOR_ROUTE
                            ? "请输入您的问题，我们会尽快与您对接..."
                            : hasPickedFocusScene
                              ? "输入您想了解的内容或者可以问问左侧使用指南。"
                              : "请现在左侧选择你最想关注的场景。"
                        }
                        value={question}
                        disabled={!hasPickedFocusScene}
                        onChange={handleQuestionChange}
                        onKeyDown={(event) => {
                          if (event.key === "Enter" && !event.shiftKey) {
                            event.preventDefault();
                            void askQuestion();
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
                      disabled={!isStreaming && (!question.trim() || !hasPickedFocusScene)}
                      title={isStreaming ? "停止生成" : "提交"}
                    >
                      {isStreaming ? "停止" : IS_VISITOR_ROUTE ? "发送" : "咨询"}
                    </button>
                  </div>
                </div>
              </footer>
            </div>
            ) : null}
          </div>
        </div>
      </div>
      {trialApplyDialogOpen ? (
        <div className="visitor-dialog-mask" role="dialog" aria-modal="true" aria-label="账号申请">
          <div className="visitor-dialog-card">
            {trialApplySuccess ? (
              <div className="visitor-dialog-success" role="status">
                <div className="visitor-dialog-success-icon" aria-hidden="true">✓</div>
                <div className="visitor-dialog-success-title">提交成功</div>
                <p className="visitor-dialog-success-text">我们已收到您的申请，会尽快与您联系。</p>
              </div>
            ) : (
              <>
                <div className="visitor-dialog-title">测试账号申请</div>
                <label className="visitor-dialog-field">
                  <span>姓名</span>
                  <input value={trialApplyName} onChange={(e) => setTrialApplyName(e.target.value)} placeholder="请输入姓名" disabled={trialApplySubmitting} />
                </label>
                <label className="visitor-dialog-field">
                  <span>单位</span>
                  <input value={trialApplyOrg} onChange={(e) => setTrialApplyOrg(e.target.value)} placeholder="请输入单位" disabled={trialApplySubmitting} />
                </label>
                <label className="visitor-dialog-field">
                  <span>联系方式</span>
                  <input
                    value={trialApplyContact}
                    onChange={(e) => setTrialApplyContact(e.target.value)}
                    placeholder="手机号或微信号"
                    disabled={trialApplySubmitting}
                  />
                </label>
                <label className="visitor-dialog-field">
                  <span>邮箱（选填）</span>
                  <input
                    type="email"
                    value={trialApplyEmail}
                    onChange={(e) => setTrialApplyEmail(e.target.value)}
                    placeholder="请输入邮箱"
                    disabled={trialApplySubmitting}
                  />
                </label>
                {trialApplyError ? <p className="visitor-dialog-error">{trialApplyError}</p> : null}
                <div className="visitor-dialog-actions">
                  <button type="button" className="visitor-dialog-btn visitor-dialog-btn--ghost" onClick={() => setTrialApplyDialogOpen(false)} disabled={trialApplySubmitting}>
                    取消
                  </button>
                  <button
                    type="button"
                    className="visitor-dialog-btn"
                    onClick={() => void submitTrialApply()}
                    disabled={
                      trialApplySubmitting ||
                      !trialApplyName.trim() ||
                      !trialApplyOrg.trim() ||
                      !trialApplyContact.trim()
                    }
                  >
                    {trialApplySubmitting ? "提交中…" : "提交申请"}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      ) : null}
    </>
  );
}

export default App
