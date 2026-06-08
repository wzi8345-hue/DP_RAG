import { useCallback, useEffect, useState } from "react";
import type { ApiClient } from "../lib/api";
import type { CollectionInfo, HealthResponse, StatsResponse } from "../lib/types";
import { DEFAULT_COLLECTION } from "../lib/types";
import { HealthDot } from "./HealthDot";

export function SystemStatus({
  api,
  health,
  healthErr,
  onRefresh,
}: {
  api: ApiClient;
  health: HealthResponse | null;
  healthErr: string | null;
  onRefresh: () => void;
}) {
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [statsErr, setStatsErr] = useState<string | null>(null);
  const [collections, setCollections] = useState<CollectionInfo[] | null>(null);
  const [loading, setLoading] = useState(false);

  const loadStats = useCallback(async () => {
    setLoading(true);
    const [s, c] = await Promise.allSettled([api.stats(), api.listCollections()]);
    if (s.status === "fulfilled") {
      setStats(s.value);
      setStatsErr(null);
    } else {
      setStats(null);
      setStatsErr(s.reason instanceof Error ? s.reason.message : String(s.reason));
    }
    setCollections(c.status === "fulfilled" ? c.value.collections : null);
    setLoading(false);
  }, [api]);

  useEffect(() => {
    loadStats();
  }, [loadStats]);

  const deps: { key: keyof HealthResponse; label: string }[] = [
    { key: "milvus", label: "Milvus 向量库" },
    { key: "llm", label: "生成 LLM" },
    { key: "embedding", label: "Embedding" },
    { key: "reranker", label: "Reranker" },
    { key: "reflection", label: "Reflection" },
  ];

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      <header className="flex items-center justify-between border-b border-slate-800 px-6 py-3">
        <span className="text-sm font-semibold">系统状态</span>
        <button
          onClick={() => {
            onRefresh();
            loadStats();
          }}
          className="rounded-lg border border-slate-700 px-3 py-1.5 text-xs text-slate-300 hover:bg-slate-800"
        >
          ↻ 刷新
        </button>
      </header>

      <div className="mx-auto w-full max-w-4xl space-y-6 p-6">
        {/* Overall */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900/30 p-5">
          <div className="flex items-center gap-3">
            <HealthDot status={healthErr ? "down" : health?.status ?? "unknown"} />
            <div>
              <div className="text-base font-semibold">
                {healthErr
                  ? "后端未连接"
                  : health
                  ? `服务状态：${health.status}`
                  : "检测中…"}
              </div>
              {healthErr && (
                <div className="text-xs text-rose-400">{healthErr}</div>
              )}
            </div>
          </div>
        </div>

        {/* Dependencies */}
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {deps.map((d) => {
            const status = health?.[d.key] as string | undefined;
            return (
              <div
                key={d.key}
                className="rounded-xl border border-slate-800 bg-slate-900/40 p-4"
              >
                <div className="text-xs text-slate-400">{d.label}</div>
                <div className="mt-1 flex items-center gap-2">
                  <DepBadge status={status} />
                </div>
              </div>
            );
          })}
        </div>

        {/* 知识库列表 */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900/30 p-5">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-semibold">
              知识库列表
              {collections && (
                <span className="ml-2 text-xs font-normal text-slate-500">
                  共 {collections.length} 个
                </span>
              )}
            </h3>
            {loading && <span className="text-xs text-slate-500">加载中…</span>}
          </div>
          {collections && collections.length > 0 ? (
            <CollectionList
              items={collections}
              defaultDocCount={
                typeof stats?.stats?.doc_count === "number"
                  ? stats.stats.doc_count
                  : undefined
              }
            />
          ) : (
            <p className="text-sm text-slate-500">暂无知识库</p>
          )}
        </div>

        {/* 默认库详细统计 */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900/30 p-5">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-semibold">默认库详细统计</h3>
            {loading && <span className="text-xs text-slate-500">加载中…</span>}
          </div>
          {statsErr ? (
            <div className="text-sm text-rose-400">{statsErr}</div>
          ) : stats && Object.keys(stats.stats).length > 0 ? (
            <CollectionStats raw={stats.stats} />
          ) : (
            <p className="text-sm text-slate-500">暂无统计数据</p>
          )}
        </div>
      </div>
    </div>
  );
}

function CollectionList({
  items,
  defaultDocCount,
}: {
  items: CollectionInfo[];
  defaultDocCount?: number;
}) {
  const sorted = [...items].sort((a, b) => {
    if (a.name === DEFAULT_COLLECTION) return -1;
    if (b.name === DEFAULT_COLLECTION) return 1;
    return b.row_count - a.row_count;
  });
  return (
    <div className="space-y-2">
      {sorted.map((c) => {
        const isDefault = c.name === DEFAULT_COLLECTION;
        const label = c.display_name || c.name.replace(/^kb_/, "");
        const docCount = isDefault
          ? defaultDocCount ?? c.doc_count
          : c.doc_count;
        const hasDocs = docCount != null && docCount > 0;
        return (
          <div
            key={c.name}
            className={`flex items-center justify-between gap-3 rounded-xl border px-4 py-3 ${
              isDefault
                ? "border-blue-500/30 bg-blue-500/5"
                : "border-slate-800 bg-slate-950/40"
            }`}
          >
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="truncate text-sm font-medium text-slate-200">{label}</span>
                {isDefault && (
                  <span className="shrink-0 rounded bg-blue-600/20 px-1.5 py-0.5 text-[10px] text-blue-300">
                    默认
                  </span>
                )}
              </div>
              <code className="text-[11px] text-slate-500">{c.name}</code>
            </div>
            <div className="flex shrink-0 items-center gap-5 text-right">
              <div>
                <div className="text-[11px] text-slate-500">文献</div>
                <div className="text-sm font-semibold text-slate-100">
                  {hasDocs ? docCount!.toLocaleString() : "—"}
                </div>
              </div>
              <div>
                <div className="text-[11px] text-slate-500">文本块</div>
                <div className="text-sm font-semibold text-slate-100">
                  {c.row_count.toLocaleString()}
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function CollectionStats({ raw }: { raw: Record<string, unknown> }) {
  const docCount = typeof raw.doc_count === "number" ? raw.doc_count : undefined;
  const total = typeof raw.total === "number" ? raw.total : undefined;
  const scanned = typeof raw.scanned === "number" ? raw.scanned : undefined;
  const perDoc = (raw.per_doc as Record<string, { type?: Record<string, number> }>) || {};

  // 文献数: 优先用后端 doc_count, 兜底用 per_doc 的键数量
  const docs = docCount ?? Object.keys(perDoc).length;

  // 按类型聚合 chunk 数 (text / table / image …)
  const typeTotals: Record<string, number> = {};
  for (const v of Object.values(perDoc)) {
    for (const [t, n] of Object.entries(v?.type || {})) {
      typeTotals[t] = (typeTotals[t] || 0) + (n as number);
    }
  }
  const partial = total != null && scanned != null && scanned < total;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        <Metric label="文献数量 (doc_id)" value={docs.toLocaleString()} accent />
        {total != null && (
          <Metric label="文本块总数 (chunks)" value={total.toLocaleString()} />
        )}
        {scanned != null && (
          <Metric label="已扫描块" value={scanned.toLocaleString()} />
        )}
      </div>

      {Object.keys(typeTotals).length > 0 && (
        <div className="flex flex-wrap gap-2">
          {Object.entries(typeTotals).map(([t, n]) => (
            <span
              key={t}
              className="rounded-lg bg-slate-950/50 px-3 py-1 text-xs text-slate-300"
            >
              {t}: <span className="font-medium text-slate-100">{n.toLocaleString()}</span>
            </span>
          ))}
        </div>
      )}

      {partial && (
        <p className="text-xs text-amber-400/80">
          注：仅扫描了 {scanned!.toLocaleString()} / {total!.toLocaleString()} 块，
          文献数可能略有偏差。
        </p>
      )}
    </div>
  );
}

function Metric({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: boolean;
}) {
  return (
    <div
      className={`rounded-xl px-4 py-3 ${
        accent ? "bg-blue-600/15 ring-1 ring-blue-500/30" : "bg-slate-950/50"
      }`}
    >
      <div className="text-xs text-slate-400">{label}</div>
      <div
        className={`mt-0.5 truncate text-2xl font-semibold ${
          accent ? "text-blue-200" : "text-slate-100"
        }`}
      >
        {value}
      </div>
    </div>
  );
}

function DepBadge({ status }: { status?: string }) {
  if (!status) return <span className="text-sm text-slate-500">unknown</span>;
  const good = ["ok", "configured", "inherits_generation"].includes(status);
  const neutral = ["disabled"].includes(status);
  const color = good
    ? "text-emerald-300"
    : neutral
    ? "text-slate-400"
    : status.startsWith("error")
    ? "text-rose-400"
    : "text-amber-300";
  return <span className={`text-sm font-medium ${color}`}>{status}</span>;
}
