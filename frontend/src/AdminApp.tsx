import { useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";

type AdminSection = "dashboard" | "knowledge" | "customers";
type KnowledgeSceneKey = "general_ai_course" | "project_based_learning" | "smart_enrollment" | "school_ai_custom";

type DashboardCard = {
  label: string;
  value: number;
  accent: "blue" | "green" | "orange" | "emerald";
  icon: "users" | "key" | "video" | "database";
};

type KnowledgeSceneItem = {
  key: KnowledgeSceneKey;
  name: string;
};

type AdminVideoAsset = {
  id: number;
  scene_key: KnowledgeSceneKey;
  scene_name: string;
  title: string;
  original_filename: string;
  file_url: string;
  mime_type: string;
  file_size: number;
  created_at: string;
};

type AdminCustomer = {
  session_id: string;
  display_name: string;
  org_name: string;
  contact: string;
  follow_up_status: string;
  trial_account: string;
  updated_at: string;
};

type CustomerSummary = {
  customer_total: number;
  trial_issued_total: number;
};

const KNOWLEDGE_SCENES: KnowledgeSceneItem[] = [
  { key: "general_ai_course", name: "人工智能通识课程" },
  { key: "project_based_learning", name: "跨学科项目式学习" },
  { key: "smart_enrollment", name: "智能招生" },
  { key: "school_ai_custom", name: "学校AI场景定制" },
];

const FOLLOW_UP_OPTIONS = ["待跟进", "跟进中", "已发放测试账号", "已完成"] as const;
const TEST_ACCOUNT_OPTIONS = ["待发放", "已发放"] as const;
const CUSTOMER_PAGE_SIZE = 10;

function AdminStatIcon({ icon }: { icon: DashboardCard["icon"] }) {
  if (icon === "users") {
    return (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M16 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
        <circle cx="10" cy="7" r="3" />
        <path d="M22 21v-2a4 4 0 0 0-3-3.87" />
        <path d="M16 4.13a4 4 0 0 1 0 7.75" />
      </svg>
    );
  }
  if (icon === "key") {
    return (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <circle cx="7.5" cy="15.5" r="3.5" />
        <path d="M11 13l8-8" />
        <path d="M16 5l3 3" />
        <path d="M14 7l3 3" />
      </svg>
    );
  }
  if (icon === "video") {
    return (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <rect x="3" y="7" width="13" height="10" rx="2" />
        <path d="M16 10l5-3v10l-5-3z" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <ellipse cx="12" cy="5" rx="7" ry="2.5" />
      <path d="M5 5v6c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5V5" />
      <path d="M5 11v6c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5v-6" />
    </svg>
  );
}

function AdminApp() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [activeSection, setActiveSection] = useState<AdminSection>("dashboard");
  const [activeSceneKey, setActiveSceneKey] = useState<KnowledgeSceneKey>("general_ai_course");
  const [videosByScene, setVideosByScene] = useState<Partial<Record<KnowledgeSceneKey, AdminVideoAsset[]>>>({});
  const [videosLoading, setVideosLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [deletingVideoIds, setDeletingVideoIds] = useState<Set<number>>(() => new Set());
  const [videoError, setVideoError] = useState("");
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [customerSummary, setCustomerSummary] = useState<CustomerSummary>({ customer_total: 0, trial_issued_total: 0 });
  const [customers, setCustomers] = useState<AdminCustomer[]>([]);
  const [customerQuery, setCustomerQuery] = useState("");
  const [customerPage, setCustomerPage] = useState(1);
  const [customerTotal, setCustomerTotal] = useState(0);
  const [customerTotalPages, setCustomerTotalPages] = useState(0);
  const [customersLoading, setCustomersLoading] = useState(false);
  const [customerError, setCustomerError] = useState("");
  const [customerListVersion, setCustomerListVersion] = useState(0);

  const activeScene = useMemo(
    () => KNOWLEDGE_SCENES.find((scene) => scene.key === activeSceneKey) ?? KNOWLEDGE_SCENES[0],
    [activeSceneKey],
  );
  const activeVideos = videosByScene[activeSceneKey] ?? [];
  const totalVideoCount = Object.values(videosByScene).reduce((sum, items) => sum + (items?.length ?? 0), 0);
  const dashboardCards: DashboardCard[] = useMemo(
    () => [
      { label: "客户总数", value: customerSummary.customer_total, accent: "blue", icon: "users" },
      { label: "已发放账号", value: customerSummary.trial_issued_total, accent: "green", icon: "key" },
      { label: "视频素材", value: totalVideoCount, accent: "orange", icon: "video" },
      { label: "课程场景", value: KNOWLEDGE_SCENES.length, accent: "emerald", icon: "database" },
    ],
    [customerSummary, totalVideoCount],
  );

  const pageMeta = useMemo(() => {
    if (activeSection === "knowledge") {
      return {
        title: "知识库素材管理",
        subtitle: "管理四个 AI 课程场景的视频素材",
      };
    }
    if (activeSection === "customers") {
      return {
        title: "客户管理",
        subtitle: "管理客户信息、跟进进度和聊天记录",
      };
    }
    return {
      title: "工作台",
      subtitle: "欢迎使用人工智能通识课程管理后台",
    };
  }, [activeSection]);

  const selectSection = (section: AdminSection) => {
    setActiveSection(section);
    setMenuOpen(false);
  };

  const refreshCustomerSummary = () => {
    fetch("/admin-api/customers/summary")
      .then(async (resp) => {
        if (!resp.ok) return;
        const data = (await resp.json()) as CustomerSummary;
        setCustomerSummary({
          customer_total: data.customer_total ?? 0,
          trial_issued_total: data.trial_issued_total ?? 0,
        });
      })
      .catch(() => {
        // ignore
      });
  };

  useEffect(() => {
    let cancelled = false;
    fetch("/admin-api/customers/summary")
      .then(async (resp) => {
        if (!resp.ok || cancelled) return;
        const data = (await resp.json()) as CustomerSummary;
        setCustomerSummary({
          customer_total: data.customer_total ?? 0,
          trial_issued_total: data.trial_issued_total ?? 0,
        });
      })
      .catch(() => {
        // 工作台统计失败时保留默认值
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (activeSection !== "knowledge") return;
    let cancelled = false;
    queueMicrotask(() => {
      if (cancelled) return;
      setVideosLoading(true);
      setVideoError("");
    });
    fetch(`/admin-api/videos?scene_key=${encodeURIComponent(activeSceneKey)}`)
      .then(async (resp) => {
        if (!resp.ok) {
          const data = await resp.json().catch(() => ({}));
          throw new Error(data.detail || "视频列表加载失败");
        }
        return resp.json() as Promise<{ items: AdminVideoAsset[] }>;
      })
      .then((data) => {
        if (cancelled) return;
        setVideosByScene((prev) => ({ ...prev, [activeSceneKey]: data.items ?? [] }));
      })
      .catch((err) => {
        if (cancelled) return;
        setVideoError(err instanceof Error ? err.message : "视频列表加载失败");
      })
      .finally(() => {
        if (!cancelled) setVideosLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [activeSection, activeSceneKey]);

  useEffect(() => {
    if (activeSection !== "customers") return;
    let cancelled = false;
    const timer = window.setTimeout(() => {
      queueMicrotask(() => {
        if (cancelled) return;
        setCustomersLoading(true);
        setCustomerError("");
      });
      fetch(
        `/admin-api/customers?q=${encodeURIComponent(customerQuery.trim())}&page=${customerPage}&page_size=${CUSTOMER_PAGE_SIZE}`,
      )
        .then(async (resp) => {
          if (!resp.ok) {
            const data = await resp.json().catch(() => ({}));
            throw new Error(data.detail || "客户列表加载失败");
          }
          return resp.json() as Promise<{
            items: AdminCustomer[];
            total: number;
            page: number;
            total_pages: number;
          }>;
        })
        .then((data) => {
          if (cancelled) return;
          setCustomers(data.items ?? []);
          setCustomerTotal(data.total ?? 0);
          setCustomerTotalPages(data.total_pages ?? 0);
        })
        .catch((err) => {
          if (cancelled) return;
          setCustomerError(err instanceof Error ? err.message : "客户列表加载失败");
        })
        .finally(() => {
          if (!cancelled) setCustomersLoading(false);
        });
    }, 200);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [activeSection, customerQuery, customerPage, customerListVersion]);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleVideoFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    const formData = new FormData();
    formData.append("scene_key", activeSceneKey);
    formData.append("title", file.name.replace(/\.[^.]+$/, ""));
    formData.append("file", file);
    setUploading(true);
    setVideoError("");
    try {
      const resp = await fetch("/admin-api/videos/upload", {
        method: "POST",
        body: formData,
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        throw new Error(data.detail || "视频上传失败");
      }
      const video = data as AdminVideoAsset;
      setVideosByScene((prev) => ({
        ...prev,
        [activeSceneKey]: [video, ...(prev[activeSceneKey] ?? [])],
      }));
    } catch (err) {
      setVideoError(err instanceof Error ? err.message : "视频上传失败");
    } finally {
      setUploading(false);
    }
  };

  const handleDeleteVideo = async (video: AdminVideoAsset) => {
    const label = (video.title || video.original_filename || "该视频").trim();
    if (!window.confirm(`确定删除「${label}」吗？删除后前台将不再展示该视频。`)) return;
    setVideoError("");
    setDeletingVideoIds((prev) => new Set(prev).add(video.id));
    try {
      const resp = await fetch(`/admin-api/videos/${encodeURIComponent(String(video.id))}`, {
        method: "DELETE",
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        throw new Error(data.detail || "视频删除失败");
      }
      setVideosByScene((prev) => ({
        ...prev,
        [video.scene_key]: (prev[video.scene_key] ?? []).filter((item) => item.id !== video.id),
      }));
    } catch (err) {
      setVideoError(err instanceof Error ? err.message : "视频删除失败");
    } finally {
      setDeletingVideoIds((prev) => {
        const next = new Set(prev);
        next.delete(video.id);
        return next;
      });
    }
  };

  const handleFollowUpChange = async (sessionId: string, followUpStatus: string) => {
    setCustomerError("");
    try {
      const resp = await fetch(`/admin-api/customers/${encodeURIComponent(sessionId)}/follow-up`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ follow_up_status: followUpStatus }),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        throw new Error(data.detail || "更新跟进进度失败");
      }
      const updated = data as AdminCustomer;
      setCustomers((prev) => prev.map((item) => (item.session_id === sessionId ? updated : item)));
      refreshCustomerSummary();
    } catch (err) {
      setCustomerError(err instanceof Error ? err.message : "更新跟进进度失败");
    }
  };

  const handleTestAccountChange = async (sessionId: string, testAccountStatus: string) => {
    setCustomerError("");
    try {
      const resp = await fetch(`/admin-api/customers/${encodeURIComponent(sessionId)}/test-account`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ test_account_status: testAccountStatus }),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        throw new Error(data.detail || "更新测试账号状态失败");
      }
      const updated = data as AdminCustomer;
      setCustomers((prev) => prev.map((item) => (item.session_id === sessionId ? updated : item)));
      refreshCustomerSummary();
    } catch (err) {
      setCustomerError(err instanceof Error ? err.message : "更新测试账号状态失败");
    }
  };

  const handleDeleteCustomer = async (sessionId: string, displayName: string) => {
    const label = displayName.trim() || "该客户";
    if (!window.confirm(`确定删除「${label}」吗？删除后不可恢复。`)) return;
    setCustomerError("");
    try {
      const resp = await fetch(`/admin-api/customers/${encodeURIComponent(sessionId)}`, {
        method: "DELETE",
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        throw new Error(data.detail || "删除客户失败");
      }
      if (customers.length <= 1 && customerPage > 1) {
        setCustomerPage((page) => Math.max(1, page - 1));
      } else {
        setCustomerListVersion((version) => version + 1);
      }
      refreshCustomerSummary();
    } catch (err) {
      setCustomerError(err instanceof Error ? err.message : "删除客户失败");
    }
  };

  return (
    <div className="admin-shell">
      <header className="admin-topbar">
        <button
          type="button"
          className="admin-menu-trigger"
          onClick={() => setMenuOpen((open) => !open)}
          aria-label={menuOpen ? "关闭菜单" : "打开菜单"}
          aria-expanded={menuOpen}
        >
          <span />
          <span />
          <span />
        </button>
        <div className="admin-topbar-title">管理后台</div>
      </header>

      <div className="admin-layout">
        <aside className={`admin-sidebar${menuOpen ? " admin-sidebar--open" : ""}`}>
          <div className="admin-sidebar-head">
            <div className="admin-brand-mark" aria-hidden="true">
              <span>⌂</span>
            </div>
            <div className="admin-brand-text">有为AI教育</div>
            <button
              type="button"
              className="admin-sidebar-close"
              onClick={() => setMenuOpen(false)}
              aria-label="关闭菜单"
            >
              ×
            </button>
          </div>

          <nav className="admin-nav" aria-label="管理后台导航">
            <button
              type="button"
              className={`admin-nav-item${activeSection === "dashboard" ? " admin-nav-item--active" : ""}`}
              onClick={() => selectSection("dashboard")}
            >
              <span className="admin-nav-icon" aria-hidden="true">◫</span>
              <span>工作台</span>
            </button>
            <button
              type="button"
              className={`admin-nav-item${activeSection === "knowledge" ? " admin-nav-item--active" : ""}`}
              onClick={() => selectSection("knowledge")}
            >
              <span className="admin-nav-icon" aria-hidden="true">▤</span>
              <span>知识库管理</span>
            </button>
            <button
              type="button"
              className={`admin-nav-item${activeSection === "customers" ? " admin-nav-item--active" : ""}`}
              onClick={() => selectSection("customers")}
            >
              <span className="admin-nav-icon" aria-hidden="true">◎</span>
              <span>客户管理</span>
            </button>
          </nav>
        </aside>

        {menuOpen ? <button type="button" className="admin-sidebar-mask" onClick={() => setMenuOpen(false)} aria-label="关闭菜单遮罩" /> : null}

        <main className="admin-main">
          <section className="admin-page-head">
            <div>
              <h1>{pageMeta.title}</h1>
              <p>{pageMeta.subtitle}</p>
            </div>
          </section>

          {activeSection === "dashboard" ? (
            <section className="admin-cards-grid" aria-label="工作台数据概览">
              {dashboardCards.map((card) => (
                <article key={card.label} className="admin-stat-card">
                  <div className="admin-stat-row">
                    <div className="admin-stat-label">{card.label}</div>
                    <div className={`admin-stat-icon admin-stat-icon--${card.accent}`} aria-hidden="true">
                      <AdminStatIcon icon={card.icon} />
                    </div>
                  </div>
                  <div className="admin-stat-value">{card.value}</div>
                </article>
              ))}
            </section>
          ) : null}

          {activeSection === "knowledge" ? (
            <section className="admin-knowledge-panel">
              <div className="admin-scene-tabs" role="tablist" aria-label="知识库场景">
                {KNOWLEDGE_SCENES.map((scene) => (
                  <button
                    key={scene.key}
                    type="button"
                    role="tab"
                    aria-selected={activeSceneKey === scene.key}
                    className={`admin-scene-tab${activeSceneKey === scene.key ? " admin-scene-tab--active" : ""}`}
                    onClick={() => setActiveSceneKey(scene.key)}
                  >
                    {scene.name}
                  </button>
                ))}
              </div>

              <article className="admin-scene-card">
                <div className="admin-scene-card-head">
                  <h2>{activeScene.name}</h2>
                  <button type="button" className="admin-upload-btn" onClick={handleUploadClick} disabled={uploading}>
                    <span aria-hidden="true">⇪</span>
                    <span>{uploading ? "上传中..." : "上传视频"}</span>
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="video/mp4,video/quicktime,video/webm"
                    className="admin-upload-input"
                    onChange={handleVideoFileChange}
                  />
                </div>

                {videoError ? <div className="admin-video-error" role="alert">{videoError}</div> : null}
                {videosLoading ? (
                  <div className="admin-empty-state">
                    <div className="admin-empty-title">正在加载视频素材</div>
                  </div>
                ) : activeVideos.length > 0 ? (
                  <div className="admin-video-list">
                    {activeVideos.map((video) => (
                      <div className="admin-video-row" key={video.id}>
                        <video className="admin-video-thumb" src={video.file_url} controls preload="metadata" />
                        <div className="admin-video-meta">
                          <div className="admin-video-title">{video.title || video.original_filename}</div>
                          <div className="admin-video-sub">
                            <span>{video.original_filename}</span>
                            <span>{formatFileSize(video.file_size)}</span>
                          </div>
                        </div>
                        <div className="admin-video-actions">
                          <a className="admin-video-link" href={video.file_url} target="_blank" rel="noreferrer">
                            查看
                          </a>
                          <button
                            type="button"
                            className="admin-video-delete-btn"
                            onClick={() => void handleDeleteVideo(video)}
                            disabled={deletingVideoIds.has(video.id)}
                          >
                            {deletingVideoIds.has(video.id) ? "删除中" : "删除"}
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="admin-empty-state">
                    <div className="admin-empty-icon" aria-hidden="true">▣</div>
                    <div className="admin-empty-title">暂无视频素材</div>
                    <div className="admin-empty-subtitle">点击上方按钮上传视频</div>
                  </div>
                )}
              </article>
            </section>
          ) : null}

          {activeSection === "customers" ? (
            <section className="admin-customers-panel" aria-label="客户管理">
              <input
                type="search"
                className="admin-customers-search"
                placeholder="搜索客户名称或单位..."
                value={customerQuery}
                onChange={(event) => {
                  setCustomerQuery(event.target.value);
                  setCustomerPage(1);
                }}
              />
              {customerError ? <div className="admin-customers-error" role="alert">{customerError}</div> : null}
              <div className="admin-customers-table-wrap">
                <table className="admin-customers-table">
                  <thead>
                    <tr>
                      <th>称呼</th>
                      <th>单位</th>
                      <th>联系方式</th>
                      <th>跟进进度</th>
                      <th>测试账号</th>
                      <th className="admin-customers-col-actions">操作</th>
                      <th className="admin-customers-col-actions">删除</th>
                    </tr>
                  </thead>
                  <tbody>
                    {customersLoading ? (
                      <tr>
                        <td colSpan={7} className="admin-customers-empty">正在加载客户数据...</td>
                      </tr>
                    ) : customers.length === 0 ? (
                      <tr>
                        <td colSpan={7} className="admin-customers-empty">暂无客户数据</td>
                      </tr>
                    ) : (
                      customers.map((customer) => (
                        <tr key={customer.session_id}>
                          <td>{customer.display_name || "—"}</td>
                          <td>{customer.org_name || "—"}</td>
                          <td>{customer.contact || "—"}</td>
                          <td>
                            <select
                              className="admin-follow-select"
                              value={customer.follow_up_status}
                              onChange={(event) => void handleFollowUpChange(customer.session_id, event.target.value)}
                            >
                              {FOLLOW_UP_OPTIONS.map((option) => (
                                <option key={option} value={option}>{option}</option>
                              ))}
                            </select>
                          </td>
                          <td>
                            <select
                              className="admin-follow-select"
                              value={customer.trial_account || "待发放"}
                              onChange={(event) => void handleTestAccountChange(customer.session_id, event.target.value)}
                            >
                              {TEST_ACCOUNT_OPTIONS.map((option) => (
                                <option key={option} value={option}>{option}</option>
                              ))}
                            </select>
                          </td>
                          <td className="admin-customers-col-actions">
                            <button
                              type="button"
                              className="admin-customer-action-btn"
                              onClick={() => window.alert(`会话 ID：${customer.session_id}`)}
                            >
                              查看
                            </button>
                          </td>
                          <td className="admin-customers-col-actions">
                            <button
                              type="button"
                              className="admin-customer-delete-btn"
                              onClick={() => void handleDeleteCustomer(customer.session_id, customer.display_name)}
                            >
                              删除
                            </button>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
              {customerTotalPages > 1 ? (
                <div className="admin-customers-pagination" aria-label="客户列表分页">
                  <button
                    type="button"
                    className="admin-pagination-btn"
                    disabled={customerPage <= 1 || customersLoading}
                    onClick={() => setCustomerPage((page) => Math.max(1, page - 1))}
                  >
                    上一页
                  </button>
                  <span className="admin-pagination-info">
                    第 {customerPage} / {customerTotalPages} 页，共 {customerTotal} 条
                  </span>
                  <button
                    type="button"
                    className="admin-pagination-btn"
                    disabled={customerPage >= customerTotalPages || customersLoading}
                    onClick={() => setCustomerPage((page) => Math.min(customerTotalPages, page + 1))}
                  >
                    下一页
                  </button>
                </div>
              ) : null}
            </section>
          ) : null}
        </main>
      </div>
    </div>
  );
}

function formatFileSize(bytes: number) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export default AdminApp;
