import { useEffect, useState } from "react";
import { useSettings, type Settings } from "../lib/settings";
import { ApiClient } from "../lib/api";
import type { CollectionInfo } from "../lib/types";

export function SettingsModal({
  onClose,
  onSaved,
}: {
  onClose: () => void;
  onSaved: () => void;
}) {
  const { settings, update, reset } = useSettings();
  const [draft, setDraft] = useState<Settings>(settings);

  const save = () => {
    update(draft);
    onSaved();
    onClose();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-lg rounded-2xl border border-slate-700 bg-slate-900 p-6 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="mb-1 text-lg font-semibold">设置</h2>
        <p className="mb-5 text-xs text-slate-400">
          配置后端地址与查询参数，保存在浏览器本地。留空地址将通过开发代理访问
          <code className="mx-1 rounded bg-slate-800 px-1">/api</code>。
        </p>

        <div className="space-y-4">
          <Field label="后端地址 (Base URL)">
            <input
              value={draft.baseUrl}
              onChange={(e) => setDraft({ ...draft, baseUrl: e.target.value })}
              placeholder="留空 = 使用开发代理 (http://localhost:8080)"
              className="input"
            />
          </Field>

          <Field label="API Key (可选)">
            <input
              value={draft.apiKey}
              type="password"
              onChange={(e) => setDraft({ ...draft, apiKey: e.target.value })}
              placeholder="Bearer token，后端未开启鉴权时留空"
              className="input"
            />
          </Field>

          <Field label="默认知识库">
            <SettingsCollectionSelect
              value={draft.collection}
              onChange={(v) => setDraft({ ...draft, collection: v })}
            />
          </Field>

          <div className="grid grid-cols-2 gap-4">
            <Field label="检索模式">
              <select
                value={draft.mode}
                onChange={(e) =>
                  setDraft({ ...draft, mode: e.target.value as Settings["mode"] })
                }
                className="input"
              >
                <option value="auto">auto (后端默认)</option>
                <option value="hybrid">hybrid</option>
                <option value="vector">vector</option>
                <option value="metadata">metadata</option>
              </select>
            </Field>

            <Field label={`Top-K: ${draft.topK}`}>
              <input
                type="range"
                min={1}
                max={20}
                value={draft.topK}
                onChange={(e) =>
                  setDraft({ ...draft, topK: Number(e.target.value) })
                }
                className="w-full accent-blue-500"
              />
            </Field>
          </div>

          <div className="flex gap-6">
            <Toggle
              label="Agentic RAG"
              checked={draft.useAgentic}
              onChange={(v) => setDraft({ ...draft, useAgentic: v })}
            />
            <Toggle
              label="流式输出"
              checked={draft.stream}
              onChange={(v) => setDraft({ ...draft, stream: v })}
            />
          </div>
        </div>

        <div className="mt-6 flex items-center justify-between">
          <button
            onClick={() => setDraft({ ...settings, ...resetDefaults(reset) })}
            className="text-xs text-slate-400 hover:text-slate-200"
          >
            恢复默认
          </button>
          <div className="flex gap-2">
            <button onClick={onClose} className="btn-ghost">
              取消
            </button>
            <button onClick={save} className="btn-primary">
              保存
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/** 子组件: 在 SettingsModal 内获取集合列表并渲染下拉框 */
function SettingsCollectionSelect({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  const { settings } = useSettings();
  const [collections, setCollections] = useState<CollectionInfo[]>([]);

  useEffect(() => {
    let alive = true;
    const api = new ApiClient(settings);
    api.listCollections()
      .then((r) => alive && setCollections(r.collections))
      .catch(() => {});
    return () => { alive = false; };
  }, [settings]);

  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="input"
    >
      <option value="">默认知识库</option>
      {collections.map((c) => (
        <option key={c.name} value={c.name}>
          {c.name.replace(/^kb_/, "")} ({c.row_count})
        </option>
      ))}
    </select>
  );
}

// reset() mutates the store; we also want to reflect defaults in the draft.
function resetDefaults(reset: () => void): Partial<Settings> {
  reset();
  return {
    baseUrl: "",
    apiKey: "",
    useAgentic: true,
    mode: "auto",
    topK: 5,
    stream: true,
    collection: "",
  };
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-xs font-medium text-slate-300">
        {label}
      </span>
      {children}
    </label>
  );
}

function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className="flex items-center gap-2 text-sm text-slate-200"
    >
      <span
        className={`relative h-5 w-9 rounded-full transition ${
          checked ? "bg-blue-500" : "bg-slate-600"
        }`}
      >
        <span
          className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition ${
            checked ? "left-4.5 translate-x-0" : "left-0.5"
          }`}
          style={{ left: checked ? "1.125rem" : "0.125rem" }}
        />
      </span>
      {label}
    </button>
  );
}
