/** 与 PRD 一致，品牌为「有为」 */
export const VISITOR_WELCOME_TEXT =
  "您好，欢迎了解有为人工智能教育平台。\n我可以帮您介绍平台功能、适用场景、使用方式和案例。请您先在左侧选择您最关注的场景进行咨询。";

export const INACTIVITY_REMINDER_TEXT =
  "如果您现在不方便继续看，也可以留个电话或微信。我让顾问把平台介绍、使用案例和试用方式发您，您有空再慢慢看。";

export const INACTIVITY_MS = 120_000;

export const VISITOR_QUICK_QUESTIONS: { label: string; text: string }[] = [
  { label: "平台是做什么的？", text: "你们平台是做什么的？" },
  { label: "适合老师怎么用？", text: "适合老师怎么用？" },
  { label: "适合家长了解什么？", text: "适合家长了解什么？" },
  { label: "有哪些使用案例？", text: "有哪些使用案例？" },
  { label: "怎么申请试用？", text: "可以申请试用吗？" },
  { label: "怎么购买或合作？", text: "怎么购买或合作？" },
];

export function visitorWelcomeMessages(welcomeId: string): { id: string; role: "assistant"; text: string }[] {
  void welcomeId;
  return [];
}

const DECLINE_PATTERNS =
  /别联系我|不要联系|不用联系|不需要联系|不要打电话|别打电话|不要加微信|不用了谢谢|不要再发|别发我/;

export function looksLikeDeclineFollowup(text: string): boolean {
  return DECLINE_PATTERNS.test((text || "").trim());
}

const PHONE_RE = /(?<!\d)(1[3-9]\d{9})(?!\d)/;

export function looksLikeContactInUserMessage(text: string): boolean {
  const t = (text || "").trim();
  if (!t) return false;
  if (PHONE_RE.test(t)) return true;
  return /(?:微信(?:号)?|wx|wechat)\s*[:：]?\s*[^\s，,。；]{2,}/i.test(t);
}
