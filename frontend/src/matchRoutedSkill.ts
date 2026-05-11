/**
 * 与后端 app/rag/skill_router.route_skill 判定顺序、关键词保持一致（修改时请同步两边）。
 */
export type RoutedSkillId =
  | "stale-detector"
  | "reading-digest"
  | "daily-capture"
  | "note-refine"
  | "knowledge-connect"
  | "style-extract"
  | "smart-search"
  | "smart-summary";

function containsAny(text: string, needles: readonly string[]): boolean {
  return needles.some((n) => n && text.includes(n));
}

/** 返回本轮会命中的 skill_id；无命中返回 null */
export function matchRoutedSkillId(question: string): RoutedSkillId | null {
  const q = (question || "").trim();
  if (!q) return null;

  if (containsAny(q, ["过期", "陈旧", "检测", "stale", "更新建议", "健康度", "过期检测"])) {
    return "stale-detector";
  }
  if (containsAny(q, ["阅读笔记", "金句", "行动项", "核心观点", "digest"])) {
    return "reading-digest";
  }
  if (containsAny(q, ["碎片", "捕获", "记录", "待办", "想法收集", "daily", "capture"])) {
    return "daily-capture";
  }
  if (containsAny(q, ["润色", "打磨", "refine", "优化表达", "改写", "提高质量", "note-refine"])) {
    return "note-refine";
  }
  if (containsAny(q, ["关联", "联系", "聚类", "主题", "知识网络", "connect", "关联发现"])) {
    return "knowledge-connect";
  }
  if (containsAny(q, ["风格", "用词", "句式", "表达习惯", "style", "画像", "style-extract"])) {
    return "style-extract";
  }
  if (containsAny(q, ["搜索", "找", "在哪里", "文档在哪", "smart-search", "查找"])) {
    return "smart-search";
  }
  if (
    containsAny(q, [
      "总结",
      "摘要",
      "概述",
      "要点",
      "大概100字",
      "约100字",
      "一句话",
      "详细总结",
      "smart-summary",
    ])
  ) {
    return "smart-summary";
  }
  return null;
}
