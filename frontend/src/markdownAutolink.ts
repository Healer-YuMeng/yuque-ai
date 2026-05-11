/**
 * GFM（remark-gfm）对裸 `http(s)://` 的自动链接在遇全角括号、中文标点时，
 * 容易把「URL 之后直到空格」的正文一并算进链接范围。
 *
 * 将「常见 ASCII 域名 + 可选路径」的裸 URL 预先包成 CommonMark 尖括号形式
 * `<https://host/path>`，以固定链接终点；不破坏 `](https://...)` 与已有 `<https://...>`。
 */
export function normalizeMarkdownAutolinks(source: string): string {
  if (!source) return source;
  const re =
    /(?<!<|\()https?:\/\/[a-zA-Z0-9][-a-zA-Z0-9._]*(?:\.[a-zA-Z0-9][-a-zA-Z0-9._]*)+(?::[0-9]+)?(?:\/[-a-zA-Z0-9._~:/?#[\]@!$&'()*+,;=%]*)?/g;
  return source.replace(re, (url) => `<${url}>`);
}
