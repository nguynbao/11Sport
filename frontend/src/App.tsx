import {
  BookOpen,
  CloudArrowUp,
  Flask,
  Gift,
  ListBullets,
  ShieldCheck,
  SidebarSimple,
  Waveform,
} from "@phosphor-icons/react";
import { useQuery } from "@tanstack/react-query";
import { Button, Layout, Menu, Tooltip } from "antd";
import { useState } from "react";
import { Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { http } from "./api/client";
import GlossaryPage from "./pages/GlossaryPage";
import JobDetailPage from "./pages/JobDetailPage";
import JobsPage from "./pages/JobsPage";
import NewJobPage from "./pages/NewJobPage";

const { Sider, Content } = Layout;

const ICON = { size: 20, "aria-hidden": true } as const;

export default function App() {
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const [collapsed, setCollapsed] = useState(false);
  const selected = pathname.startsWith("/new") ? "/new" : pathname.startsWith("/glossary") ? "/glossary" : "/";
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: async () => (await http.get<{ provider_mode: string; tts_provider: string }>("/health")).data,
    staleTime: 60_000,
  });

  return (
    <Layout style={{ minHeight: "100vh" }} className={collapsed ? "sider-is-collapsed" : undefined}>
      <Sider
        breakpoint="lg"
        collapsible
        collapsed={collapsed}
        onCollapse={(c) => setCollapsed(c)}
        collapsedWidth={0}
        width={248}
        trigger={null}
        className="app-sider"
        aria-label="Điều hướng chính"
      >
        <div className="brand">
          <div className="brand-mark">
            <Waveform size={22} weight="bold" aria-hidden />
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div className="brand-name">11Sport Dubber</div>
            <div className="brand-sub">Lồng tiếng & phụ đề AI</div>
          </div>
          <Tooltip title="Ẩn menu">
            <Button
              type="text"
              size="small"
              className="sider-collapse-btn"
              icon={<SidebarSimple size={18} aria-hidden />}
              aria-label="Ẩn menu"
              onClick={() => setCollapsed(true)}
            />
          </Tooltip>
        </div>
        <Menu
          className="sider-menu"
          mode="inline"
          selectedKeys={[selected]}
          onClick={(e) => navigate(e.key)}
          items={[
            { key: "/new", icon: <CloudArrowUp {...ICON} />, label: "Tạo video mới" },
            { key: "/", icon: <ListBullets {...ICON} />, label: "Danh sách video" },
            { key: "/glossary", icon: <BookOpen {...ICON} />, label: "Thuật ngữ" },
          ]}
        />
        {health && (
          <div className="sider-footer">
            {health.provider_mode === "fake" ? (
              <div className="mode-pill fake" role="status">
                <Flask size={16} weight="bold" aria-hidden />
                Chế độ thử nghiệm (AI giả lập)
              </div>
            ) : health.provider_mode === "free" ? (
              <div className="mode-pill free" role="status">
                <Gift size={16} weight="bold" aria-hidden />
                Demo miễn phí · Whisper + Edge TTS
              </div>
            ) : (
              <div className="mode-pill real" role="status">
                <ShieldCheck size={16} weight="bold" aria-hidden />
                AI thật · TTS {health.tts_provider === "azure" ? "Azure" : "OpenAI"}
              </div>
            )}
          </div>
        )}
      </Sider>
      <Layout className="app-main">
        {collapsed && (
          <Tooltip title="Hiện menu" placement="right">
            <button
              type="button"
              className="sider-open-trigger"
              aria-label="Hiện menu"
              onClick={() => setCollapsed(false)}
            >
              <SidebarSimple size={20} weight="bold" aria-hidden />
            </button>
          </Tooltip>
        )}
        <Content className="app-content">
          <Routes>
            <Route path="/" element={<JobsPage />} />
            <Route path="/new" element={<NewJobPage />} />
            <Route path="/jobs/:id" element={<JobDetailPage />} />
            <Route path="/glossary" element={<GlossaryPage />} />
          </Routes>
        </Content>
      </Layout>
    </Layout>
  );
}

