import { useEffect, useRef, useState } from "react";
import type { ApiClient } from "../lib/api";
import type {
  SkillListResponse,
  SkillSpec,
  SkillSummary,
  SkillTemplate,
} from "../lib/types";

interface FormState {
  id: string;
  name: string;
  description: string;
  priority: string;
  triggers: string; // 每行一个
  min_docs: string;
  need_conflict_check: boolean;
  need_quantitative_data: boolean;
  max_rounds: string;
  max_batches: string;
  gap_stall_limit: string;
  guards: string[];
  plan: string;
  policy: string;
  synthesis_system: string;
  synthesis_thinking: string;
  synthesis_user: string;
}

const EMPTY: FormState = {
  id: "",
  name: "",
  description: "",
  priority: "50",
  triggers: "",
  min_docs: "",
  need_conflict_check: false,
  need_quantitative_data: false,
  max_rounds: "",
  max_batches: "",
  gap_stall_limit: "",
  guards: [],
  plan: "",
  policy: "",
  synthesis_system: "",
  synthesis_thinking: "",
  synthesis_user: "",
};

function fromSpec(s: SkillSummary | SkillSpec, overrideId?: string): FormState {
  const suff = s.sufficiency || {};
  const tun = s.tuning || {};
  return {
    id: overrideId ?? s.id,
    name: s.name + (overrideId ? " (副本)" : ""),
    description: s.description || "",
    priority: s.priority != null ? String(s.priority) : "50",
    triggers: (s.triggers || []).join("\n"),
    min_docs: suff.min_docs != null ? String(suff.min_docs) : "",
    need_conflict_check: !!suff.need_conflict_check,
    need_quantitative_data: !!suff.need_quantitative_data,
    max_rounds: tun.max_rounds != null ? String(tun.max_rounds) : "",
    max_batches: tun.max_batches != null ? String(tun.max_batches) : "",
    gap_stall_limit: tun.gap_stall_limit != null ? String(tun.gap_stall_limit) : "",
    guards: s.guards || [],
    plan: s.plan || "",
    policy: s.policy || "",
    synthesis_system: s.synthesis_system || "",
    synthesis_thinking: s.synthesis_thinking || "",
    synthesis_user: s.synthesis_user || "",
  };
}

function toSpec(f: FormState): SkillSpec {
  const num = (v: string) => (v.trim() === "" ? undefined : Number(v));
  const sufficiency: Record<string, unknown> = {};
  if (f.min_docs.trim() !== "") sufficiency.min_docs = Number(f.min_docs);
  if (f.need_conflict_check) sufficiency.need_conflict_check = true;
  if (f.need_quantitative_data) sufficiency.need_quantitative_data = true;
  const tuning: Record<string, unknown> = {};
  if (f.max_rounds.trim() !== "") tuning.max_rounds = Number(f.max_rounds);
  if (f.max_batches.trim() !== "") tuning.max_batches = Number(f.max_batches);
  if (f.gap_stall_limit.trim() !== "") tuning.gap_stall_limit = Number(f.gap_stall_limit);
  return {
    id: f.id.trim(),
    name: f.name.trim(),
    description: f.description.trim(),
    priority: num(f.priority),
    triggers: f.triggers.split("\n").map((t) => t.trim()).filter(Boolean),
    sufficiency,
    tuning,
    guards: f.guards,
    plan: f.plan,
    policy: f.policy,
    synthesis_system: f.synthesis_system,
    synthesis_thinking: f.synthesis_thinking,
    synthesis_user: f.synthesis_user,
  };
}

export function SkillsManager({ api }: { api: ApiClient }) {
  const [data, setData] = useState<SkillListResponse | null>(null);
  const [tpl, setTpl] = useState<SkillTemplate | null>(null);
  const [form, setForm] = useState<FormState>(EMPTY);
  const [mode, setMode] = useState<"new" | "edit">("new");
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const editorRef = useRef<HTMLDivElement>(null);

  const reload = () => {
    api.listSkills().then(setData).catch((e) => flash(String(e.message || e), true));
  };

  useEffect(() => {
    reload();
    api.getSkillTemplate().then(setTpl).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api]);

  const flash = (m: string, isErr = false) => {
    if (isErr) {
      setErr(m);
      setMsg(null);
    } else {
      setMsg(m);
      setErr(null);
    }
    window.setTimeout(() => {
      setMsg(null);
      setErr(null);
    }, 5000);
  };

  const set = <K extends keyof FormState>(k: K, v: FormState[K]) =>
    setForm((p) => ({ ...p, [k]: v }));

  const toggle = (k: "guards", v: string) =>
    setForm((p) => {
      const arr = p[k];
      return { ...p, [k]: arr.includes(v) ? arr.filter((x) => x !== v) : [...arr, v] };
    });

  const startNew = () => {
    setForm(EMPTY);
    setMode("new");
    editorRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const startFromExample = () => {
    if (tpl?.example) {
      setForm(fromSpec(tpl.example));
      setMode("new");
      editorRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  };

  const editSkill = (s: SkillSummary) => {
    setForm(fromSpec(s));
    setMode("edit");
    editorRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const copySkill = (s: SkillSummary) => {
    setForm(fromSpec(s, `${s.id}_copy`));
    setMode("new");
    editorRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const onDelete = async (s: SkillSummary) => {
    if (!confirm(`确定删除技能 "${s.name}" (${s.id})？此操作不可撤销。`)) return;
    try {
      await api.deleteSkill(s.id);
      flash(`已删除技能 "${s.name}"`);
      reload();
    } catch (e) {
      flash(e instanceof Error ? e.message : String(e), true);
    }
  };

  const onSave = async () => {
    if (!form.id.trim() || !form.name.trim()) {
      flash("技能 ID 和名称为必填", true);
      return;
    }
    if (!form.plan.trim() || !form.policy.trim()) {
      flash("规划提示词 (plan) 与策略提示词 (policy) 为必填", true);
      return;
    }
    setSaving(true);
    try {
      const r = await api.saveSkill(toSpec(form));
      flash(`已保存技能 "${r.skill.name}"，下次专家模式提问即生效`);
      reload();
      setMode("edit");
    } catch (e) {
      flash(e instanceof Error ? e.message : String(e), true);
    } finally {
      setSaving(false);
    }
  };

  const disabled = data != null && !data.enabled;
  const guards = tpl?.valid_guards || [];

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      <header className="flex items-center justify-between border-b border-slate-800 px-6 py-3">
        <span className="text-sm font-semibold">专家技能管理</span>
        {data && (
          <span className="text-xs text-slate-500">
            路由模式: <span className="text-slate-300">{data.router_mode}</span>
            {!data.enabled && <span className="ml-2 text-amber-400">（未启用）</span>}
          </span>
        )}
      </header>

      <div className="mx-auto w-full max-w-4xl space-y-6 p-6">
        {(msg || err) && (
          <div
            className={`rounded-lg px-4 py-2.5 text-sm ${
              err ? "bg-rose-500/15 text-rose-300" : "bg-emerald-500/15 text-emerald-300"
            }`}
          >
            {err || msg}
          </div>
        )}

        {disabled && (
          <div className="rounded-lg bg-amber-500/10 px-4 py-2.5 text-sm text-amber-300">
            技能系统未启用（professional.skills.enabled=false），新建技能不会生效。
          </div>
        )}

        {/* 技能列表 */}
        <Section
          title="已加载技能"
          desc="专家模式提问时，会根据触发词/语义自动选择最匹配的技能；选不到时回退通用逻辑。内置技能只读，可“复制为新技能”后修改。"
        >
          {!data || data.skills.length === 0 ? (
            <div className="py-4 text-center text-sm text-slate-500">
              暂无技能，点击下方“新建技能”创建
            </div>
          ) : (
            <div className="space-y-2">
              {data.skills.map((s) => (
                <div
                  key={s.id}
                  className={`rounded-xl border px-4 py-3 transition ${
                    form.id === s.id
                      ? "border-blue-500/40 bg-blue-500/5"
                      : "border-slate-800 bg-slate-900/50"
                  }`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-base">🧩</span>
                        <span className="text-sm font-medium text-slate-200">{s.name}</span>
                        <code className="rounded bg-slate-800 px-1.5 py-0.5 text-[11px] text-slate-400">
                          {s.id}
                        </code>
                        <span
                          className={`rounded px-1.5 py-0.5 text-[11px] ${
                            s.editable
                              ? "bg-emerald-500/15 text-emerald-300"
                              : "bg-slate-700/60 text-slate-400"
                          }`}
                        >
                          {s.editable ? "自定义" : "内置"}
                        </span>
                        <span className="text-[11px] text-slate-500">优先级 {s.priority}</span>
                      </div>
                      {s.description && (
                        <p className="mt-1 line-clamp-2 text-xs text-slate-400">{s.description}</p>
                      )}
                      {s.triggers && s.triggers.length > 0 && (
                        <div className="mt-1.5 flex flex-wrap gap-1">
                          {s.triggers.slice(0, 8).map((t) => (
                            <span
                              key={t}
                              className="rounded bg-slate-800/80 px-1.5 py-0.5 text-[11px] text-slate-400"
                            >
                              {t}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                    <div className="flex shrink-0 items-center gap-1.5">
                      {s.editable ? (
                        <>
                          <button
                            onClick={() => editSkill(s)}
                            className="rounded-lg px-2.5 py-1 text-xs text-blue-300 transition hover:bg-blue-500/10"
                          >
                            编辑
                          </button>
                          <button
                            onClick={() => onDelete(s)}
                            className="rounded-lg px-2 py-1 text-xs text-slate-500 transition hover:bg-rose-500/10 hover:text-rose-300"
                          >
                            删除
                          </button>
                        </>
                      ) : (
                        <button
                          onClick={() => copySkill(s)}
                          className="rounded-lg px-2.5 py-1 text-xs text-slate-300 transition hover:bg-slate-700/60"
                        >
                          复制为新技能
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          <div className="mt-3 flex gap-2">
            <button onClick={startNew} className="btn-primary">
              ＋ 新建技能
            </button>
            <button onClick={startFromExample} className="btn-ghost" disabled={!tpl}>
              用示例模版填充
            </button>
          </div>
        </Section>

        {/* 编辑器 */}
        <section
          ref={editorRef}
          className="scroll-mt-4 rounded-2xl border border-slate-800 bg-slate-900/30 p-5"
        >
          <div className="mb-1 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-100">
              {mode === "edit" ? `编辑技能 · ${form.id}` : "新建技能"}
            </h3>
            <span className="text-xs text-slate-500">填写模版 · 必填项标 *</span>
          </div>
          <p className="mb-4 text-xs text-slate-400">
            一个技能 = 一套“思考方式”。检索流程不变，但 plan/policy 决定如何拆解问题、如何判断收口与引导下一步检索。
          </p>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <Field label="技能 ID *" help="小写字母开头，仅含小写字母/数字/下划线">
              <input
                className="input"
                value={form.id}
                disabled={mode === "edit"}
                placeholder="如 quantitative_extraction"
                onChange={(e) => set("id", e.target.value)}
              />
            </Field>
            <Field label="技能名称 *" help="中文展示名">
              <input
                className="input"
                value={form.name}
                placeholder="如 定量数据抽取"
                onChange={(e) => set("name", e.target.value)}
              />
            </Field>
          </div>

          <Field label="适用场景说明" help="什么样的提问该用这个技能 — 供分类器判断">
            <textarea
              className="input min-h-[60px]"
              value={form.description}
              onChange={(e) => set("description", e.target.value)}
            />
          </Field>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <Field label="优先级" help="触发词同时命中时数值大者优先（默认 50）">
              <input
                type="number"
                className="input"
                value={form.priority}
                onChange={(e) => set("priority", e.target.value)}
              />
            </Field>
            <Field label="触发词" help="命中即倾向选该技能，支持正则。每行一个">
              <textarea
                className="input min-h-[60px] font-mono text-xs"
                value={form.triggers}
                placeholder={"数值\n含量\n速率"}
                onChange={(e) => set("triggers", e.target.value)}
              />
            </Field>
          </div>

          <Field label="收口标准 (sufficiency)" help="达到即可结束检索">
            <div className="flex flex-wrap items-center gap-4">
              <label className="flex items-center gap-2 text-xs text-slate-300">
                至少文献数
                <input
                  type="number"
                  className="input w-20"
                  value={form.min_docs}
                  onChange={(e) => set("min_docs", e.target.value)}
                />
              </label>
              <Toggle
                checked={form.need_conflict_check}
                onChange={(v) => set("need_conflict_check", v)}
                label="需冲突核查"
              />
              <Toggle
                checked={form.need_quantitative_data}
                onChange={(v) => set("need_quantitative_data", v)}
                label="需定量数据"
              />
            </div>
          </Field>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <Field label="最大轮次 max_rounds" help="留空用全局默认">
              <input
                type="number"
                className="input"
                value={form.max_rounds}
                onChange={(e) => set("max_rounds", e.target.value)}
              />
            </Field>
            <Field label="最大批次 max_batches" help="留空用全局默认">
              <input
                type="number"
                className="input"
                value={form.max_batches}
                onChange={(e) => set("max_batches", e.target.value)}
              />
            </Field>
            <Field label="停滞上限 gap_stall_limit" help="留空用全局默认">
              <input
                type="number"
                className="input"
                value={form.gap_stall_limit}
                onChange={(e) => set("gap_stall_limit", e.target.value)}
              />
            </Field>
          </div>

          {guards.length > 0 && (
            <Field label="守卫 (guards)" help="未满足时会提示 policy 继续定向补证据">
              <div className="flex flex-wrap gap-2">
                {guards.map((g) => (
                  <Chip key={g} active={form.guards.includes(g)} onClick={() => toggle("guards", g)}>
                    {g}
                  </Chip>
                ))}
              </div>
            </Field>
          )}

          <Field label="规划提示词 plan *" help="选中后如何拆解 facets 与首轮批次（替换通用拆解段）">
            <textarea
              className="input min-h-[120px] text-sm leading-relaxed"
              value={form.plan}
              onChange={(e) => set("plan", e.target.value)}
            />
          </Field>

          <Field label="策略提示词 policy *" help="每轮检索后如何判断 继续/收口/反问 并引导下一步">
            <textarea
              className="input min-h-[120px] text-sm leading-relaxed"
              value={form.policy}
              onChange={(e) => set("policy", e.target.value)}
            />
          </Field>

          <details className="mt-2 rounded-xl border border-slate-800 bg-slate-950/40 p-3">
            <summary className="cursor-pointer text-xs font-medium text-slate-300">
              综述提示词（可选，留空回退通用模板）
            </summary>
            <div className="mt-3 space-y-3">
              <Field label="综述结构提示词 synthesis_system" help="最终综述的输出结构">
                <textarea
                  className="input min-h-[80px] text-sm"
                  value={form.synthesis_system}
                  onChange={(e) => set("synthesis_system", e.target.value)}
                />
              </Field>
              <Field label="综述分析思路 synthesis_thinking" help="综述前的中文分析思路">
                <textarea
                  className="input min-h-[80px] text-sm"
                  value={form.synthesis_thinking}
                  onChange={(e) => set("synthesis_thinking", e.target.value)}
                />
              </Field>
              <Field label="综述 User 模版 synthesis_user" help="须包含 {context} 占位符">
                <textarea
                  className="input min-h-[80px] text-sm"
                  value={form.synthesis_user}
                  onChange={(e) => set("synthesis_user", e.target.value)}
                />
              </Field>
            </div>
          </details>

          <div className="mt-5 flex items-center gap-3">
            <button onClick={onSave} disabled={saving || disabled} className="btn-primary">
              {saving ? "保存中…" : mode === "edit" ? "保存修改" : "创建技能"}
            </button>
            <button onClick={startNew} className="btn-ghost">
              清空 / 新建
            </button>
          </div>
        </section>
      </div>
    </div>
  );
}

function Field({
  label,
  help,
  children,
}: {
  label: string;
  help?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="mt-3">
      <label className="mb-1 block text-xs font-medium text-slate-300">{label}</label>
      {help && <p className="mb-1.5 text-[11px] text-slate-500">{help}</p>}
      {children}
    </div>
  );
}

function Chip({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-lg px-2.5 py-1 text-xs transition ${
        active
          ? "bg-blue-600/30 text-blue-200 ring-1 ring-blue-500/40"
          : "bg-slate-800/60 text-slate-400 hover:bg-slate-700/60"
      }`}
    >
      {children}
    </button>
  );
}

function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
}) {
  return (
    <label className="flex cursor-pointer items-center gap-2 text-xs text-slate-300">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      {label}
    </label>
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
