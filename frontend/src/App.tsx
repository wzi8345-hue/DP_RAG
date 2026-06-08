import { useCallback, useEffect, useMemo, useState } from "react";
import { ApiClient } from "./lib/api";
import { useSettings } from "./lib/settings";
import type { HealthResponse } from "./lib/types";
import {
  deleteConversation,
  loadConversations,
  type Conversation,
} from "./lib/conversations";
import { ChatView } from "./components/ChatView";
import { KnowledgeBase } from "./components/KnowledgeBase";
import { SkillsManager } from "./components/SkillsManager";
import { SystemStatus } from "./components/SystemStatus";
import { SettingsModal } from "./components/SettingsModal";
import { HealthDot } from "./components/HealthDot";
import { LogViewer } from "./pages/LogViewer";

type Tab = "chat" | "kb" | "skills" | "system" | "logs";

const OTHER_NAV: { id: Tab; label: string; icon: string }[] = [
  { id: "kb", label: "知识库", icon: "📚" },
  { id: "skills", label: "专家技能", icon: "🧩" },
  { id: "system", label: "系统状态", icon: "📊" },
  { id: "logs", label: "检索日志", icon: "📋" },
];

export default function App() {
  const { settings } = useSettings();
  const api = useMemo(() => new ApiClient(settings), [settings]);
  const [tab, setTab] = useState<Tab>("chat");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthErr, setHealthErr] = useState<string | null>(null);

  // 对话留存: 侧栏会话列表 + 当前选中会话
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConvId, setActiveConvId] = useState<string | null>(null);
  const [chatOpen, setChatOpen] = useState(true);

  const reloadConversations = useCallback(() => {
    setConversations(loadConversations());
  }, []);

  useEffect(() => {
    reloadConversations();
  }, [reloadConversations]);

  const selectConv = useCallback((id: string) => {
    setActiveConvId(id);
    setTab("chat");
    setChatOpen(true);
  }, []);

  const newChat = useCallback(() => {
    setActiveConvId(null);
    setTab("chat");
    setChatOpen(true);
  }, []);

  const removeConv = useCallback(
    (id: string) => {
      deleteConversation(id);
      setActiveConvId((cur) => (cur === id ? null : cur));
      reloadConversations();
    },
    [reloadConversations]
  );

  const refreshHealth = useCallback(async () => {
    try {
      const h = await api.health();
      setHealth(h);
      setHealthErr(null);
    } catch (e) {
      setHealth(null);
      setHealthErr(e instanceof Error ? e.message : String(e));
    }
  }, [api]);

  useEffect(() => {
    refreshHealth();
    const t = setInterval(refreshHealth, 15000);
    return () => clearInterval(t);
  }, [refreshHealth]);

  return (
    <div className="flex h-full w-full overflow-hidden">
      {/* Sidebar */}
      <aside className="flex w-60 shrink-0 flex-col border-r border-slate-800 bg-slate-950/60">
        <div className="flex items-center gap-2 px-5 py-5">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 text-lg font-bold">
            DP
          </div>
          <div>
            <div className="text-sm font-semibold leading-tight">DP-RAG</div>
            <div className="text-xs text-slate-400">科研文献问答</div>
          </div>
        </div>

        <nav className="flex flex-1 flex-col gap-1 overflow-y-auto px-3">
          {/* 智能问答 + 历史对话子下拉 */}
          <button
            onClick={() => {
              setTab("chat");
              setChatOpen((o) => (tab === "chat" ? !o : true));
            }}
            className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition ${
              tab === "chat"
                ? "bg-blue-600/20 text-blue-200 ring-1 ring-blue-500/40"
                : "text-slate-300 hover:bg-slate-800/60"
            }`}
          >
            <span className="text-base">💬</span>
            智能问答
            <span className="ml-auto text-xs text-slate-500">
              {chatOpen ? "▲" : "▼"}
            </span>
          </button>

          {chatOpen && (
            <div className="mb-1 ml-3 flex flex-col gap-0.5 border-l border-slate-800 pl-2">
              <button
                onClick={newChat}
                className="flex items-center gap-2 rounded-md px-2.5 py-1.5 text-xs text-slate-400 hover:bg-slate-800/60 hover:text-slate-200"
              >
                <span>＋</span> 新对话
              </button>
              {conversations.length === 0 ? (
                <div className="px-2.5 py-1.5 text-xs text-slate-600">
                  暂无历史对话
                </div>
              ) : (
                conversations.map((c) => (
                  <div
                    key={c.id}
                    className={`group flex items-center rounded-md pr-1 text-xs transition ${
                      activeConvId === c.id
                        ? "bg-slate-800 text-slate-100"
                        : "text-slate-400 hover:bg-slate-800/50"
                    }`}
                  >
                    <button
                      onClick={() => selectConv(c.id)}
                      title={c.title}
                      className="flex min-w-0 flex-1 items-center gap-1.5 px-2.5 py-1.5 text-left"
                    >
                      <span className="shrink-0">
                        {c.expert ? "🎓" : "💬"}
                      </span>
                      <span className="truncate">{c.title}</span>
                    </button>
                    <button
                      onClick={() => removeConv(c.id)}
                      title="删除对话"
                      className="shrink-0 px-1 text-slate-600 opacity-0 hover:text-rose-400 group-hover:opacity-100"
                    >
                      ✕
                    </button>
                  </div>
                ))
              )}
            </div>
          )}

          {OTHER_NAV.map((n) => (
            <button
              key={n.id}
              onClick={() => setTab(n.id)}
              className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition ${
                tab === n.id
                  ? "bg-blue-600/20 text-blue-200 ring-1 ring-blue-500/40"
                  : "text-slate-300 hover:bg-slate-800/60"
              }`}
            >
              <span className="text-base">{n.icon}</span>
              {n.label}
            </button>
          ))}
        </nav>

        <div className="border-t border-slate-800 p-3">
          <button
            onClick={() => setSettingsOpen(true)}
            className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm text-slate-300 transition hover:bg-slate-800/60"
          >
            <span className="text-base">⚙️</span> 设置
          </button>
          <div className="mt-2 flex items-center gap-2 px-3 py-1.5 text-xs text-slate-400">
            <HealthDot
              status={healthErr ? "down" : health?.status ?? "unknown"}
            />
            <span className="truncate">
              {healthErr
                ? "后端未连接"
                : health
                ? `服务 ${health.status}`
                : "检测中…"}
            </span>
          </div>
        </div>
      </aside>

      {/* Main */}
      <main className="flex min-w-0 flex-1 flex-col">
        {tab === "chat" && (
          <ChatView
            api={api}
            conversationId={activeConvId}
            onConversationIdChange={setActiveConvId}
            onConversationsChanged={reloadConversations}
          />
        )}
        {tab === "kb" && <KnowledgeBase api={api} />}
        {tab === "skills" && <SkillsManager api={api} />}
        {tab === "system" && (
          <SystemStatus
            api={api}
            health={health}
            healthErr={healthErr}
            onRefresh={refreshHealth}
          />
        )}
        {tab === "logs" && <LogViewer api={api} />}
      </main>

      {settingsOpen && (
        <SettingsModal
          onClose={() => setSettingsOpen(false)}
          onSaved={refreshHealth}
        />
      )}
    </div>
  );
}
