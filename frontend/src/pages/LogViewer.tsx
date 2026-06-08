import { useEffect, useMemo, useRef, useState } from "react";
import type { ApiClient } from "../lib/api";
import type {
  LogLineEntry,
  LogSessionDetail,
  LogSessionSummary,
} from "../lib/types";

/* ------------------------------------------------------------------ */
/*  Types & Constants                                                  */
/* ------------------------------------------------------------------ */

type LogLevel = "ALL" | "DEBUG" | "INFO" | "WARNING" | "ERROR";

const LEVEL_ORDER: Record<string, number> = {
  DEBUG: 10,
  INFO: 20,
  WARNING: 30,
  ERROR: 40,
};

const LEVEL_OPTIONS: LogLevel[] = ["ALL", "ERROR", "WARNING", "INFO", "DEBUG"];

/* ------------------------------------------------------------------ */
/*  Utility                                                            */
/* ------------------------------------------------------------------ */

function formatTime(ts: number): string {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  return d.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function levelColor(level: string): string {
  switch (level) {
    case "ERROR":
      return "text-rose-400";
    case "WARNING":
      return "text-amber-400";
    case "DEBUG":
      return "text-slate-500";
    default:
      return "text-slate-300";
  }
}

function levelBg(level: string): string {
  switch (level) {
    case "ERROR":
      return "bg-rose-500/15 text-rose-300";
    case "WARNING":
      return "bg-amber-500/15 text-amber-300";
    case "DEBUG":
      return "bg-slate-800 text-slate-500";
    default:
      return "bg-slate-800 text-slate-400";
  }
}

/** 提取 logger 短名: pipeline.retrieval.langgraph_agent → langgraph_agent */
function shortLogger(name: string): string {
  const parts = name.split(".");
  return parts.length > 1 ? parts.slice(1).join(".") : name;
}

/* ------------------------------------------------------------------ */
/*  Main Component                                                     */
/* ------------------------------------------------------------------ */

export function LogViewer({ api }: { api: ApiClient }) {
  const [sessions, setSessions] = useState<LogSessionSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<LogSessionDetail | null>(null);
  const [search, setSearch] = useState("");
  const [levelFilter, setLevelFilter] = useState<LogLevel>("ALL");
  const [live, setLive] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const [backendError, setBackendError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const esRef = useRef<EventSource | null>(null);

  // 加载 session 列表
  const reload = () => {
    api.listLogSessions()
      .then((r) => {
        setSessions(r.sessions);
        setBackendError(null);
      })
      .catch((e) => {
        setBackendError(e instanceof Error ? e.message : String(e));
      });
  };

  useEffect(() => {
    reload();
    const iv = setInterval(reload, 5000);
    return () => clearInterval(iv);
  }, [api]);

  // 选中 session → 加载详情
  useEffect(() => {
    if (!selected) {
      setDetail(null);
      return;
    }
    api.getLogSession(selected)
      .then((d) => {
        setDetail(d);
        setBackendError(null);
      })
      .catch((e) => {
        setBackendError(e instanceof Error ? e.message : String(e));
      });
  }, [selected, api]);

  // 按级别过滤日志行
  const filteredLines = useMemo(() => {
    if (!detail?.lines) return [];
    if (levelFilter === "ALL") return detail.lines;
    const minLevel = LEVEL_ORDER[levelFilter] ?? 0;
    return detail.lines.filter(
      (line) => (LEVEL_ORDER[line.level] ?? 0) >= minLevel
    );
  }, [detail?.lines, levelFilter]);

  // 自动滚动 (只在过滤后行数变化时触发)
  useEffect(() => {
    if (autoScroll && filteredLines.length) {
      scrollRef.current?.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [filteredLines.length, autoScroll]);

  // SSE 实时追踪
  useEffect(() => {
    // 关闭旧连接
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }

    if (!live || !selected) return;

    const es = new EventSource(api.logStreamUrl(selected));
    esRef.current = es;

    es.onmessage = (e) => {
      try {
        const line: LogLineEntry = JSON.parse(e.data);
        setDetail((prev) => {
          if (!prev) return prev;
          // 限制内存: 超过 2000 行时截断
          const lines = [...prev.lines, line];
          return { ...prev, lines: lines.slice(-2000), line_count: prev.line_count + 1 };
        });
      } catch { /* ignore parse errors */ }
    };

    es.onerror = () => {
      es.close();
      esRef.current = null;
      // SSE 断开时关闭 live 状态, 避免按钮显示活跃但实际连接已死
      setLive(false);
    };

    return () => {
      es.close();
      esRef.current = null;
    };
  }, [live, selected, api]);

  // 过滤 session
  const filtered = sessions.filter(
    (s) =>
      !search ||
      s.session_id.includes(search) ||
      s.query.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="flex h-full overflow-hidden text-slate-200">
      {/* ── 左侧: Session 列表 ── */}
      <aside className="flex w-72 shrink-0 flex-col border-r border-slate-800 bg-slate-950/60">
        <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3">
          <h1 className="text-sm font-semibold">📋 检索日志</h1>
          <button
            onClick={reload}
            className="rounded px-2 py-1 text-xs text-slate-400 hover:bg-slate-800 hover:text-slate-200"
          >
            刷新
          </button>
        </div>

        {/* 搜索框 */}
        <div className="border-b border-slate-800 px-3 py-2">
          <input
            type="text"
            placeholder="搜索 session / 查询…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs outline-none focus:border-blue-500"
          />
        </div>

        {/* Session 列表 */}
        <div className="flex-1 overflow-y-auto">
          {backendError && (
            <div className="border-b border-rose-500/20 bg-rose-500/10 px-4 py-2 text-xs text-rose-300">
              ⚠ 后端连接失败: {backendError}
            </div>
          )}
          {filtered.length === 0 && !backendError ? (
            <p className="px-4 py-6 text-center text-xs text-slate-600">
              暂无日志
            </p>
          ) : (
            filtered.map((s) => (
              <button
                key={s.session_id}
                onClick={() => setSelected(s.session_id)}
                className={`w-full border-b border-slate-800/50 px-4 py-2.5 text-left transition hover:bg-slate-800/60 ${
                  selected === s.session_id
                    ? "bg-blue-600/15 border-l-2 border-l-blue-500"
                    : ""
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate text-xs font-mono text-slate-400">
                    {s.session_id}
                  </span>
                  <span className="shrink-0 rounded bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-500">
                    {s.line_count} 行
                  </span>
                </div>
                {s.query && (
                  <p className="mt-0.5 truncate text-xs text-slate-300">
                    {s.query}
                  </p>
                )}
                <p className="mt-0.5 text-[10px] text-slate-600">
                  {formatTime(s.updated_at)}
                </p>
              </button>
            ))
          )}
        </div>
      </aside>

      {/* ── 右侧: 日志详情 ── */}
      <main className="flex min-w-0 flex-1 flex-col">
        {!selected ? (
          <div className="flex flex-1 items-center justify-center text-sm text-slate-600">
            ← 选择左侧 session 查看检索流程日志
          </div>
        ) : (
          <>
            {/* 顶部信息栏 */}
            <header className="flex items-center justify-between border-b border-slate-800 px-5 py-2.5">
              <div className="flex items-center gap-3">
                <span className="font-mono text-xs text-slate-400">
                  {selected}
                </span>
                {detail?.query && (
                  <>
                    <span className="text-slate-600">·</span>
                    <span className="max-w-md truncate text-sm text-slate-300">
                      {detail.query}
                    </span>
                  </>
                )}
                <span className="rounded bg-slate-800 px-2 py-0.5 text-xs text-slate-400">
                  {detail?.line_count ?? 0} 行
                </span>
              </div>
              <div className="flex items-center gap-2">
                {/* 日志级别过滤 */}
                <select
                  value={levelFilter}
                  onChange={(e) => setLevelFilter(e.target.value as LogLevel)}
                  className="rounded-lg border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-300 outline-none focus:border-blue-500"
                  title="按日志级别过滤"
                >
                  {LEVEL_OPTIONS.map((l) => (
                    <option key={l} value={l}>
                      {l === "ALL" ? "全部级别" : l}
                    </option>
                  ))}
                </select>
                <label className="flex cursor-pointer items-center gap-1.5 text-xs text-slate-400">
                  <input
                    type="checkbox"
                    checked={autoScroll}
                    onChange={(e) => setAutoScroll(e.target.checked)}
                    className="accent-blue-500"
                  />
                  自动滚动
                </label>
                <button
                  onClick={() => setLive(!live)}
                  className={`rounded-full px-3 py-1 text-xs transition ${
                    live
                      ? "bg-green-600/25 text-green-200 ring-1 ring-green-500/40"
                      : "bg-slate-800 text-slate-400 hover:text-slate-200"
                  }`}
                >
                  {live ? "● 实时追踪" : "○ 实时追踪"}
                </button>
              </div>
            </header>

            {/* 过滤统计 */}
            {levelFilter !== "ALL" && (
              <div className="flex items-center gap-2 border-b border-slate-800/50 px-5 py-1.5 text-xs text-slate-500">
                <span>
                  显示 {filteredLines.length} / {detail?.line_count ?? 0} 行
                </span>
                <button
                  onClick={() => setLevelFilter("ALL")}
                  className="text-blue-400 hover:text-blue-300"
                >
                  清除过滤
                </button>
              </div>
            )}

            {/* 日志内容 */}
            <div
              ref={scrollRef}
              className="flex-1 overflow-y-auto px-5 py-3 font-mono text-xs leading-5"
            >
              {(!detail?.lines || detail.lines.length === 0) && (
                <p className="py-8 text-center text-slate-600">
                  该 session 暂无日志
                </p>
              )}
              {filteredLines.length === 0 && detail?.lines && detail.lines.length > 0 && (
                <p className="py-8 text-center text-slate-600">
                  当前过滤条件下无匹配日志
                </p>
              )}
              {filteredLines.map((line, i) => (
                <LogRow key={i} line={line} />
              ))}
            </div>
          </>
        )}
      </main>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  LogRow                                                             */
/* ------------------------------------------------------------------ */

function LogRow({ line }: { line: LogLineEntry }) {
  const isTiming = line.message.includes("耗时");
  const isRoute = /route|router|reflect|rerank|policy/i.test(
    line.message.slice(0, 40)
  );

  return (
    <div
      className={`flex gap-2 py-px ${
        isTiming
          ? "bg-blue-500/5 font-semibold"
          : line.level === "ERROR"
          ? "bg-rose-500/5"
          : line.level === "WARNING"
          ? "bg-amber-500/5"
          : ""
      }`}
    >
      {/* 时间戳 */}
      <span className="shrink-0 text-slate-600">{line.timestamp}</span>

      {/* 级别 */}
      <span className={`shrink-0 rounded px-1 text-[10px] ${levelBg(line.level)}`}>
        {line.level}
      </span>

      {/* logger 短名 */}
      <span className="shrink-0 text-slate-600">{shortLogger(line.logger)}</span>

      {/* 消息正文 */}
      <span className={`break-all ${levelColor(line.level)} ${isRoute ? "text-cyan-300" : ""}`}>
        <LogMessage text={line.message} />
      </span>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  LogMessage: 高亮 correlation_id 和关键阶段                         */
/* ------------------------------------------------------------------ */

function LogMessage({ text }: { text: string }) {
  // 高亮 [correlation_id] 标签
  const parts = text.split(/(\[[0-9a-f]{8}\])/g);
  if (parts.length <= 1) return <>{text}</>;

  return (
    <>
      {parts.map((part, i) =>
        /^\[[0-9a-f]{8}\]$/.test(part) ? (
          <span key={i} className="rounded bg-violet-500/15 px-1 text-violet-300">
            {part}
          </span>
        ) : (
          <span key={i}>{part}</span>
        )
      )}
    </>
  );
}
