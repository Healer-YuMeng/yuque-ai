import { useMemo, useState } from "react";

type AdminSection = "dashboard" | "knowledge";
type KnowledgeScene = "人工智能通识课程" | "跨学科项目式学习" | "智能招生" | "学校AI场景定制";

type DashboardCard = {
  label: string;
  value: number;
  accent: "blue" | "green" | "orange" | "emerald";
  icon: "users" | "key" | "video" | "database";
};

const DASHBOARD_CARDS: DashboardCard[] = [
  { label: "客户总数", value: 0, accent: "blue", icon: "users" },
  { label: "已发放账号", value: 0, accent: "green", icon: "key" },
  { label: "视频素材", value: 0, accent: "orange", icon: "video" },
  { label: "课程场景", value: 4, accent: "emerald", icon: "database" },
];

const KNOWLEDGE_SCENES: KnowledgeScene[] = [
  "人工智能通识课程",
  "跨学科项目式学习",
  "智能招生",
  "学校AI场景定制",
];

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
  const [activeScene, setActiveScene] = useState<KnowledgeScene>("人工智能通识课程");

  const pageMeta = useMemo(() => {
    if (activeSection === "knowledge") {
      return {
        title: "知识库素材管理",
        subtitle: "管理四个 AI 课程场景的视频素材",
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
          </nav>
        </aside>

        {menuOpen ? <button type="button" className="admin-sidebar-mask" onClick={() => setMenuOpen(false)} aria-label="关闭菜单遮罩" /> : null}

        <main className="admin-main">
          <section className="admin-page-head">
            <h1>{pageMeta.title}</h1>
            <p>{pageMeta.subtitle}</p>
          </section>

          {activeSection === "dashboard" ? (
            <section className="admin-cards-grid" aria-label="工作台数据概览">
              {DASHBOARD_CARDS.map((card) => (
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
          ) : (
            <section className="admin-knowledge-panel">
              <div className="admin-scene-tabs" role="tablist" aria-label="知识库场景">
                {KNOWLEDGE_SCENES.map((scene) => (
                  <button
                    key={scene}
                    type="button"
                    role="tab"
                    aria-selected={activeScene === scene}
                    className={`admin-scene-tab${activeScene === scene ? " admin-scene-tab--active" : ""}`}
                    onClick={() => setActiveScene(scene)}
                  >
                    {scene}
                  </button>
                ))}
              </div>

              <article className="admin-scene-card">
                <div className="admin-scene-card-head">
                  <h2>{activeScene}</h2>
                  <button type="button" className="admin-upload-btn">
                    <span aria-hidden="true">⇪</span>
                    <span>上传视频</span>
                  </button>
                </div>

                <div className="admin-empty-state">
                  <div className="admin-empty-icon" aria-hidden="true">▣</div>
                  <div className="admin-empty-title">暂无视频素材</div>
                  <div className="admin-empty-subtitle">点击上方按钮上传视频</div>
                </div>
              </article>
            </section>
          )}
        </main>
      </div>
    </div>
  );
}

export default AdminApp;
