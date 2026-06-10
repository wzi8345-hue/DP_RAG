import { useEffect, useRef, useState } from "react";
import type { ApiClient } from "../lib/api";
import type {
  CollectionInfo,
  DocumentInfo,
  IngestResult,
  TaskResponse,
} from "../lib/types";

// 灌入任务跑在后端 (TaskStore + 线程池, 保留 24h), 与前端页面无关。
// 把任务持久化到 localStorage, 切换页面/刷新后重新挂载时能恢复进度,
// 避免"切走再回来像是进程停了"的错觉。
const TASKS_STORAGE_KEY = "dp-rag-kb-tasks";

function loadPersistedTasks(): Record<string, TaskResponse> {
  try {
    const raw = localStorage.getItem(TASKS_STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch {
    /* ignore */
  }
  return {};
}

export function KnowledgeBase({ api }: { api: ApiClient }) {
  const [collections, setCollections] = useState<CollectionInfo[]>([]);
  const [tasks, setTasks] = useState<Record<string, TaskResponse>>(loadPersistedTasks);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [newKbName, setNewKbName] = useState("");
  const [showNewKb, setShowNewKb] = useState(false);
  const [targetCollection, setTargetCollection] = useState("");
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [docsLoading, setDocsLoading] = useState(false);
  const pollRef = useRef<number | null>(null);

  // 加载集合列表
  const reloadCollections = () => {
    api.listCollections()
      .then((r) => setCollections(r.collections))
      .catch(() => {});
  };

  // 加载选中知识库的文献清单
  const reloadDocuments = (name: string) => {
    if (!name) {
      setDocs([]);
      return;
    }
    setDocsLoading(true);
    api.listDocuments(name)
      .then((r) => setDocs(r.documents))
      .catch(() => setDocs([]))
      .finally(() => setDocsLoading(false));
  };

  useEffect(() => { reloadCollections(); }, [api]);

  // 切换选中的知识库时, 拉取其文献清单
  useEffect(() => { reloadDocuments(targetCollection); }, [targetCollection, api]);

  // 重新挂载 (切回本页/刷新) 时, 用持久化的任务 ID 向后端拉取最新状态并恢复进度。
  // 后端已无该任务 (TTL 过期或服务重启清空) 则移除。
  useEffect(() => {
    const ids = Object.keys(loadPersistedTasks());
    if (ids.length === 0) return;
    let cancelled = false;
    (async () => {
      for (const id of ids) {
        try {
          const t = await api.getTask(id);
          if (!cancelled) setTasks((prev) => ({ ...prev, [id]: t }));
        } catch {
          if (!cancelled)
            setTasks((prev) => {
              const next = { ...prev };
              delete next[id];
              return next;
            });
        }
      }
      if (!cancelled) reloadCollections();
    })();
    return () => { cancelled = true; };
  }, [api]);

  // 持久化任务表, 供下次挂载恢复
  useEffect(() => {
    try {
      localStorage.setItem(TASKS_STORAGE_KEY, JSON.stringify(tasks));
    } catch {
      /* ignore */
    }
  }, [tasks]);

  // 轮询未完成任务
  useEffect(() => {
    const active = Object.values(tasks).some(
      (t) => t.status === "pending" || t.status === "running"
    );
    if (!active) {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      return;
    }
    if (pollRef.current) return;
    pollRef.current = window.setInterval(async () => {
      const ids = Object.entries(tasks)
        .filter(([, t]) => t.status === "pending" || t.status === "running")
        .map(([id]) => id);
      for (const id of ids) {
        try {
          const t = await api.getTask(id);
          setTasks((prev) => ({ ...prev, [id]: t }));
          // 任务完成时刷新集合列表与当前知识库文献清单
          if (t.status === "done" || t.status === "failed") {
            reloadCollections();
            reloadDocuments(targetCollection);
          }
        } catch {
          /* keep previous */
        }
      }
    }, 2000);
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [tasks, api, targetCollection]);

  const flash = (m: string, isErr = false) => {
    if (isErr) { setErr(m); setMsg(null); }
    else { setMsg(m); setErr(null); }
    window.setTimeout(() => { setMsg(null); setErr(null); }, 4000);
  };

  const createAndSelect = async () => {
    const name = newKbName.trim();
    if (!name) return;
    try {
      const info = await api.createCollection(name);
      setNewKbName("");
      setShowNewKb(false);
      setTargetCollection(info.name);
      reloadCollections();
      flash(`知识库 "${info.display_name || name}" 已创建，可上传 PDF 灌入`);
    } catch (e) {
      flash(e instanceof Error ? e.message : String(e), true);
    }
  };

  const onRebuild = async (name: string, label: string) => {
    if (!confirm(`确定要重建知识库 "${label}" 吗？将清空集合并用本地已存解析产物重新灌入（不重新解析 PDF）。`))
      return;
    try {
      const t = await api.rebuildCollection(name);
      setTasks((prev) => ({ ...prev, [t.id]: t }));
      flash(`已提交重建任务（${t.id}）`);
    } catch (e) {
      flash(e instanceof Error ? e.message : String(e), true);
    }
  };

  const onUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const collection = targetCollection.trim();
    if (!collection) {
      flash("请先选择或创建一个知识库", true);
      return;
    }
    try {
      const t = await api.uploadAndIngest(Array.from(files), collection);
      setTasks((prev) => ({ ...prev, [t.id]: t }));
      flash(`已提交 ${files.length} 个文件到知识库 "${collection}"（任务 ${t.id}）`);
    } catch (e) {
      flash(e instanceof Error ? e.message : String(e), true);
    }
  };

  const onDeleteCollection = async (name: string, label: string) => {
    if (!confirm(`确定要删除知识库 "${label}" 及其所有数据吗？此操作不可撤销。`)) return;
    try {
      await api.deleteCollection(name);
      flash(`已删除知识库 "${label}"`);
      reloadCollections();
      if (targetCollection === name) {
        setTargetCollection("");
      }
    } catch (e) {
      flash(e instanceof Error ? e.message : String(e), true);
    }
  };

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      <header className="border-b border-slate-800 px-6 py-3">
        <span className="text-sm font-semibold">知识库管理</span>
      </header>

      <div className="mx-auto w-full max-w-4xl space-y-6 p-6">
        {(msg || err) && (
          <div
            className={`rounded-lg px-4 py-2.5 text-sm ${
              err
                ? "bg-rose-500/15 text-rose-300"
                : "bg-emerald-500/15 text-emerald-300"
            }`}
          >
            {err || msg}
          </div>
        )}

        {/* 集合列表 */}
        <Section title="知识库列表" desc="每个知识库对应一个独立的 Milvus 集合，数据互不干扰。">
          {collections.length === 0 ? (
            <div className="py-4 text-center text-sm text-slate-500">
              暂无知识库，点击下方按钮创建
            </div>
          ) : (
            <div className="space-y-2">
              {collections.map((c) => {
                const label = c.display_name || c.name.replace(/^kb_/, "");
                const selected = targetCollection === c.name;
                return (
                <div
                  key={c.name}
                  className={`flex items-center justify-between rounded-xl border px-4 py-3 transition ${
                    selected
                      ? "border-blue-500/40 bg-blue-500/5"
                      : "border-slate-800 bg-slate-900/50"
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <span className="text-lg">📚</span>
                    <div>
                      <div className="text-sm font-medium text-slate-200">
                        {label}
                      </div>
                      <div className="text-xs text-slate-500">
                        {c.doc_count ?? 0} 篇文档 · {c.row_count} 条数据块
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setTargetCollection(c.name)}
                      className="rounded-lg px-3 py-1 text-xs text-blue-300 transition hover:bg-blue-500/10"
                    >
                      {selected ? "✓ 已选中" : "选为目标"}
                    </button>
                    <button
                      onClick={() => onRebuild(c.name, label)}
                      disabled={(c.doc_count ?? 0) === 0}
                      className="rounded-lg px-2 py-1 text-xs text-slate-400 transition hover:bg-amber-500/10 hover:text-amber-300 disabled:cursor-not-allowed disabled:opacity-40"
                      title="复用本地解析产物重建集合"
                    >
                      重建
                    </button>
                    <button
                      onClick={() => onDeleteCollection(c.name, label)}
                      className="rounded-lg px-2 py-1 text-xs text-slate-500 transition hover:bg-rose-500/10 hover:text-rose-300"
                      title="删除知识库"
                    >
                      删除
                    </button>
                  </div>
                </div>
                );
              })}
            </div>
          )}

          {showNewKb ? (
            <div className="flex items-center gap-2">
              <input
                value={newKbName}
                onChange={(e) => setNewKbName(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && createAndSelect()}
                placeholder="输入知识库名称 (如: 材料科学)"
                className="input flex-1"
                autoFocus
              />
              <button onClick={createAndSelect} disabled={!newKbName.trim()} className="btn-primary">
                创建
              </button>
              <button onClick={() => setShowNewKb(false)} className="btn-ghost">
                取消
              </button>
            </div>
          ) : (
            <button
              onClick={() => setShowNewKb(true)}
              className="mt-2 flex w-full items-center justify-center gap-1.5 rounded-xl border border-dashed border-slate-700 px-4 py-3 text-sm text-slate-400 transition hover:border-blue-500/40 hover:text-blue-300"
            >
              <span className="text-base">＋</span> 新建知识库
            </button>
          )}
        </Section>

        {/* 上传 + 灌入 */}
        <Section
          title="上传并灌入"
          desc={`目标知识库: ${
            targetCollection
              ? collections.find((c) => c.name === targetCollection)?.display_name ||
                targetCollection.replace(/^kb_/, "")
              : "未选择"
          }。上传 PDF 后自动执行解析→分块→向量化→入库。`}
        >
          {targetCollection ? (
            <UploadZone onFiles={onUpload} />
          ) : (
            <div className="rounded-xl border border-dashed border-slate-700 px-6 py-8 text-center text-sm text-slate-500">
              请先在上方选择或创建一个知识库
            </div>
          )}
        </Section>

        {/* 文献清单: 选中知识库后展示库内已入库文献 */}
        {targetCollection && (
          <Section
            title={`文献清单（${docs.length} 篇）`}
            desc="该知识库中已入库的文献，与列表的篇数一致；上传完成后自动刷新。"
          >
            <div className="mb-2 flex justify-end">
              <button
                onClick={() => reloadDocuments(targetCollection)}
                className="rounded-lg px-2.5 py-1 text-xs text-slate-400 transition hover:bg-slate-800/60 hover:text-slate-200"
              >
                ↻ 刷新
              </button>
            </div>
            {docsLoading ? (
              <p className="py-3 text-center text-sm text-slate-500">加载中…</p>
            ) : docs.length === 0 ? (
              <p className="py-3 text-center text-sm text-slate-500">
                该知识库暂无已入库文献
              </p>
            ) : (
              <div className="max-h-96 space-y-1.5 overflow-y-auto pr-1">
                {docs.map((d) => (
                  <div
                    key={d.doc_id}
                    className="flex items-center justify-between gap-3 rounded-lg border border-slate-800 bg-slate-900/40 px-3 py-2"
                  >
                    <div className="flex min-w-0 items-center gap-2">
                      <span className="shrink-0 text-sm">📄</span>
                      <div className="min-w-0">
                        <div className="truncate text-sm text-slate-200" title={d.doc_name}>
                          {d.doc_name}
                        </div>
                        {d.doc_name !== d.doc_id && (
                          <div className="truncate text-xs text-slate-500" title={d.doc_id}>
                            {d.doc_id}
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="flex shrink-0 items-center gap-2 text-xs text-slate-400">
                      {d.year > 0 && (
                        <span className="rounded bg-slate-800 px-1.5 py-0.5">{d.year}</span>
                      )}
                      <span>{d.chunks} 块</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Section>
        )}

        {/* 任务进度 */}
        <Section title="任务进度" desc="灌入任务在后端运行（保留 24h）；切换页面或刷新都不会中断，回到本页会自动恢复进度。">
          {Object.keys(tasks).length === 0 ? (
            <p className="text-sm text-slate-500">暂无任务</p>
          ) : (
            <div className="space-y-2">
              {Object.values(tasks)
                .sort((a, b) => b.created_at - a.created_at)
                .map((t) => (
                  <TaskRow key={t.id} task={t} />
                ))}
            </div>
          )}
        </Section>
      </div>
    </div>
  );
}

function TaskRow({ task }: { task: TaskResponse }) {
  const color =
    task.status === "done"
      ? "text-emerald-300"
      : task.status === "failed"
      ? "text-rose-300"
      : "text-blue-300";
  const pct = Math.round((task.progress || 0) * 100);
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-3">
      <div className="mb-1.5 flex items-center justify-between text-sm">
        <span className="font-mono text-xs text-slate-400">{task.id}</span>
        <span className={`text-xs font-medium ${color}`}>{task.status}</span>
      </div>
      {(task.status === "running" || task.status === "pending") && (
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
          <div
            className="h-full bg-blue-500 transition-all"
            style={{ width: `${pct}%` }}
          />
        </div>
      )}
      {task.error && (
        <div className="mt-1 text-xs text-rose-400">{task.error}</div>
      )}
      {task.status === "done" && <TaskResultPanel result={task.result} />}
    </div>
  );
}

function ResultChip({
  tone,
  children,
}: {
  tone: "ok" | "skip" | "fail" | "muted";
  children: React.ReactNode;
}) {
  const cls = {
    ok: "bg-emerald-500/15 text-emerald-300",
    skip: "bg-amber-500/15 text-amber-300",
    fail: "bg-rose-500/15 text-rose-300",
    muted: "bg-slate-800 text-slate-300",
  }[tone];
  return <span className={`rounded px-1.5 py-0.5 text-xs ${cls}`}>{children}</span>;
}

// 灌入结果面板: 概要 chips + 逐篇可读日志 (成功 / 已存在跳过 / 失败原因)。
function TaskResultPanel({ result }: { result: unknown }) {
  if (!result || typeof result !== "object") return null;
  const r = result as IngestResult;
  const success = r.success ?? 0;
  const failed = r.failed ?? 0;
  const skipped = r.skipped_existing ?? 0;
  const stored = r.stored_chunks ?? 0;
  const failedMap = new Map(
    (r.failed_reasons || []).map((f) => [f.doc_id || "", f.reason || "未知原因"])
  );
  const hasLog =
    (r.skipped_files && r.skipped_files.length > 0) ||
    (r.details && r.details.length > 0);
  return (
    <div className="mt-2 space-y-2">
      <div className="flex flex-wrap items-center gap-1.5">
        {success > 0 && <ResultChip tone="ok">✓ 成功 {success} 篇</ResultChip>}
        {skipped > 0 && (
          <ResultChip tone="skip">⏭ 已存在跳过 {skipped} 篇</ResultChip>
        )}
        {failed > 0 && <ResultChip tone="fail">✗ 失败 {failed} 篇</ResultChip>}
        <ResultChip tone="muted">入库 {stored} 块</ResultChip>
      </div>
      {hasLog && (
        <div className="max-h-48 space-y-0.5 overflow-y-auto rounded-lg bg-slate-950/50 p-2 text-xs">
          {(r.skipped_files || []).map((name, i) => (
            <div key={`s${i}`} className="text-amber-300/90">
              ⏭ {name} —— 已入库，跳过
            </div>
          ))}
          {(r.details || []).map((d, i) => {
            const id = d.doc_id || "(未知文档)";
            return d.ok ? (
              <div key={`d${i}`} className="text-emerald-300/90">
                ✓ {id} —— 入库 {d.total_chunks ?? 0} 块
              </div>
            ) : (
              <div key={`d${i}`} className="text-rose-300/90">
                ✗ {id} —— {failedMap.get(id) || "灌入失败"}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function UploadZone({ onFiles }: { onFiles: (f: FileList | null) => void }) {
  const ref = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  return (
    <div
      onClick={() => ref.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => { e.preventDefault(); setDrag(false); onFiles(e.dataTransfer.files); }}
      className={`cursor-pointer rounded-xl border-2 border-dashed px-6 py-8 text-center transition ${
        drag
          ? "border-blue-500 bg-blue-500/5"
          : "border-slate-700 hover:border-slate-600"
      }`}
    >
      <div className="text-2xl">📤</div>
      <p className="mt-1 text-sm text-slate-300">
        点击或拖拽 PDF 到此处上传
      </p>
      <p className="mt-0.5 text-xs text-slate-500">支持批量上传多个文件</p>
      <input
        ref={ref}
        type="file"
        accept=".pdf"
        multiple
        className="hidden"
        onChange={(e) => onFiles(e.target.files)}
      />
    </div>
  );
}

function Section({
  title,
  desc,
  children,
}: {
  title: string;
  desc?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-2xl border border-slate-800 bg-slate-900/30 p-5">
      <h3 className="text-sm font-semibold text-slate-100">{title}</h3>
      {desc && <p className="mb-3 mt-0.5 text-xs text-slate-400">{desc}</p>}
      {children}
    </section>
  );
}
