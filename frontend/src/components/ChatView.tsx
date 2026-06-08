import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import type { ApiClient } from "../lib/api";
import { parseCitations, stripStreamingCitations } from "../lib/citations";
import { wrapBareLatex } from "../lib/latex";
import { useSettings } from "../lib/settings";
import type {
  ChatRequest,
  CollectionInfo,
  Hit,
  ResearchMeta,
  RetrievalMode,
  StreamEvent,
} from "../lib/types";
import {
  deriveTitle,
  loadConversation,
  newConversationId,
  saveConversation,
} from "../lib/conversations";
import { SourcesPanel, type PanelTab, type SourceItem } from "./SourcesPanel";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  hits?: Hit[];
  context?: string;
  latency?: number;
  status?: string;
  error?: string;
  streaming?: boolean;
  /** 本条回答使用的模式, 用于气泡上的标识 */
  expert?: boolean;
  research?: ResearchMeta | null;
  /** 专家模式流式"思考过程" (规划/每轮评估/综述思考), 与正文 content 分开展示 */
  thinking?: string;
  signals?: {
    needs_clarify?: boolean;
    no_answer?: boolean;
    retry_count?: number;
  };
}

const SAMPLES = [
  "MoS2 的晶格常数是多少？",
  "耐候钢的耐腐蚀机理是什么？",
  "锌铝镁镀层的主要优势有哪些？",
];

function CollectionSelector({
  value,
  onChange,
  disabled,
  api,
}: {
  value: string;
  onChange: (v: string) => void;
  disabled?: boolean;
  api: ApiClient;
}) {
  const [collections, setCollections] = useState<CollectionInfo[]>([]);

  useEffect(() => {
    let alive = true;
    api.listCollections()
      .then((r) => alive && setCollections(r.collections))
      .catch(() => {});
    return () => { alive = false; };
  }, [api]);

  return (
    <select
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value)}
      className="rounded-lg border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-300 outline-none focus:border-blue-500"
      title="选择检索的知识库"
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

export function ChatView({
  api,
  conversationId = null,
  onConversationIdChange,
  onConversationsChanged,
}: {
  api: ApiClient;
  conversationId?: string | null;
  onConversationIdChange?: (id: string | null) => void;
  onConversationsChanged?: () => void;
}) {
  const { settings, update } = useSettings();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sourcesFor, setSourcesFor] = useState<string | null>(null);
  const [panelTab, setPanelTab] = useState<PanelTab>("summary");
  const [highlight, setHighlight] = useState<{
    msgId: string;
    num?: number;
    docId?: string;
  } | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // ── 对话留存: 当前会话 id (写入目标) + 最新消息/会话快照 (供完成时持久化) ──
  // 初值用 undefined 哨兵 (而非 conversationId): 否则组件挂载时若已带 conversationId
  // (例如从其他标签页点开历史对话, ChatView 重新挂载), 装载副作用的 guard
  // `conversationId === ref` 会因相等而提前 return, 导致历史消息加载不出来 → 点开空白。
  const activeConvIdRef = useRef<string | null | undefined>(undefined);
  const messagesRef = useRef<Message[]>(messages);
  const sessionIdRef = useRef<string | null>(sessionId);
  const prevBusyRef = useRef(false);
  messagesRef.current = messages;
  sessionIdRef.current = sessionId;

  // 外部 (侧栏) 切换会话 → 装载该会话的消息与 session_id; null = 新对话
  useEffect(() => {
    if (conversationId === activeConvIdRef.current) return;
    activeConvIdRef.current = conversationId;
    if (!conversationId) {
      setMessages([]);
      setSessionId(null);
    } else {
      const conv = loadConversation(conversationId);
      setMessages(conv?.messages ?? []);
      setSessionId(conv?.sessionId ?? null);
    }
    setSourcesFor(null);
    setHighlight(null);
  }, [conversationId]);

  // 一轮对话完成 (busy true→false) 时持久化当前会话
  useEffect(() => {
    if (prevBusyRef.current && !busy) {
      const convId = activeConvIdRef.current;
      const msgs = messagesRef.current;
      if (convId && msgs.some((m) => m.role === "assistant")) {
        saveConversation({
          id: convId,
          title: deriveTitle(msgs),
          sessionId: sessionIdRef.current,
          expert: settings.professional,
          messages: msgs,
          updatedAt: Date.now(),
        });
        onConversationsChanged?.();
      }
    }
    prevBusyRef.current = busy;
  }, [busy, settings.professional, onConversationsChanged]);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  const patchMsg = (id: string, patch: Partial<Message>) =>
    setMessages((ms) =>
      ms.map((m) => (m.id === id ? { ...m, ...patch } : m))
    );

  const send = async () => {
    const q = input.trim();
    if (!q || busy) return;
    setInput("");
    setBusy(true);

    // 首条消息时为本对话分配 id, 以便留存到侧栏并支持后续继续执行
    if (!activeConvIdRef.current) {
      const newId = newConversationId();
      activeConvIdRef.current = newId;
      onConversationIdChange?.(newId);
    }

    const expert = settings.professional;
    const userMsg: Message = { id: rid(), role: "user", content: q };
    const aMsg: Message = {
      id: rid(),
      role: "assistant",
      content: "",
      streaming: true,
      expert,
      status: expert ? "研究中：思考检索策略…" : "检索中…",
    };
    setMessages((ms) => [...ms, userMsg, aMsg]);

    const req: ChatRequest = {
      query: q,
      session_id: sessionId,
      use_agentic: settings.useAgentic,
      mode: settings.mode === "auto" ? null : (settings.mode as RetrievalMode),
      top_k: settings.topK,
      professional: expert,
      collection: settings.collection || null,
    };

    try {
      if (settings.stream) {
        const ctrl = new AbortController();
        abortRef.current = ctrl;
        await api.chatStream(
          req,
          (ev: StreamEvent) => handleEvent(aMsg.id, ev),
          ctrl.signal
        );
      } else {
        const res = await api.chat(req);
        if (res.session_id) setSessionId(res.session_id);
        patchMsg(aMsg.id, {
          content: res.answer || "（无回答）",
          hits: res.hits,
          context: res.context,
          latency: res.latency_s,
          streaming: false,
          status: undefined,
          error: res.error ?? undefined,
          research: res.research ?? null,
          signals: {
            needs_clarify: res.needs_clarify,
            no_answer: res.no_answer,
            retry_count: res.retry_count,
          },
        });
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      patchMsg(aMsg.id, {
        streaming: false,
        status: undefined,
        error: msg,
        content: "",
      });
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  };

  const handleEvent = (msgId: string, ev: StreamEvent) => {
    switch (ev.type) {
      case "status":
        setMessages((ms) =>
          ms.map((m) =>
            m.id === msgId
              ? {
                  ...m,
                  status:
                    ev.stage === "generating"
                      ? m.expert
                        ? "证据充分，撰写综述…"
                        : "生成中…"
                      : m.expert
                      ? "研究中：多轮检索…"
                      : "检索中…",
                }
              : m
          )
        );
        break;
      case "thinking":
        // 专家模式"思考过程": 规划/每轮评估整段下发, 综述阶段为 token 流; 累积到 thinking
        setMessages((ms) =>
          ms.map((m) => {
            if (m.id !== msgId) return m;
            const prev = m.thinking ?? "";
            // 规划/评估等整段思考之间补换行; 综述思考是连续 token 直接拼接
            const sep =
              prev && ev.phase !== "synthesis" && !prev.endsWith("\n")
                ? "\n\n"
                : "";
            return {
              ...m,
              thinking: prev + sep + ev.content,
              status: m.content ? undefined : "思考中…",
            };
          })
        );
        break;
      case "text":
        setMessages((ms) =>
          ms.map((m) =>
            m.id === msgId
              ? { ...m, content: m.content + ev.content, status: undefined }
              : m
          )
        );
        break;
      case "done":
        if (typeof ev.session_id === "string" && ev.session_id) {
          setSessionId(ev.session_id);
        }
        patchMsg(msgId, {
          streaming: false,
          status: undefined,
          hits: ev.hits ?? [],
          context: typeof ev.context === "string" ? ev.context : undefined,
          latency: ev.latency_s,
          research: ev.research ?? null,
          signals: {
            needs_clarify: ev.needs_clarify,
            no_answer: ev.no_answer,
          },
          content:
            (ev.answer && ev.answer.length > 0
              ? ev.answer
              : undefined) as string | undefined,
        });
        break;
      case "error":
        patchMsg(msgId, {
          streaming: false,
          status: undefined,
          error: ev.message,
        });
        break;
    }
  };

  const stop = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setBusy(false);
  };

  const newSession = async () => {
    setMessages([]);
    setSourcesFor(null);
    setHighlight(null);
    setSessionId(null);
    activeConvIdRef.current = null;
    onConversationIdChange?.(null);
  };

  const showSources = (msgId: string) => {
    setSourcesFor(msgId);
    setPanelTab("chunks");
    setHighlight(null);
  };

  const onCite = (msgId: string, num: number, docId?: string) => {
    setSourcesFor(msgId);
    setPanelTab("summary");
    setHighlight({ msgId, num, docId });
  };

  // 当前展示来源的消息: 优先只列回答里实际引用的块, 没有引用时回退到全部命中
  const activeMsg = messages.find((m) => m.id === sourcesFor);
  const activeItems: SourceItem[] = useMemo(() => {
    if (!activeMsg) return [];
    const hits = activeMsg.hits ?? [];
    const parsed = parseCitations(activeMsg.content, hits, activeMsg.context, {
      research: activeMsg.expert,
    });
    if (parsed.hasCitations) {
      return parsed.citedHits.map((c) => ({ num: c.num, hit: c.hit }));
    }
    return hits.map((h, i) => ({ num: i + 1, hit: h }));
  }, [activeMsg]);

  return (
    <div className="flex h-full min-w-0 flex-1">
      <div className="flex min-w-0 flex-1 flex-col">
        {/* Toolbar */}
        <header className="flex items-center justify-between border-b border-slate-800 px-6 py-3">
          <div className="flex items-center gap-2 text-sm">
            <span className="font-semibold">智能问答</span>
            {sessionId && (
              <span className="rounded bg-slate-800 px-2 py-0.5 text-xs text-slate-400">
                会话 {sessionId.slice(0, 8)}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <ModeToggle
              expert={settings.professional}
              disabled={busy}
              onChange={(expert) => update({ professional: expert })}
            />
            <CollectionSelector
              value={settings.collection}
              onChange={(v) => update({ collection: v })}
              disabled={busy}
              api={api}
            />
            <Pill
              active={settings.useAgentic}
              onClick={() => update({ useAgentic: !settings.useAgentic })}
            >
              Agentic
            </Pill>
            <Pill
              active={settings.stream}
              onClick={() => update({ stream: !settings.stream })}
            >
              流式
            </Pill>
            <span className="rounded-full bg-slate-800 px-2.5 py-1 text-xs text-slate-400">
              {settings.mode} · top{settings.topK}
            </span>
            <button
              onClick={newSession}
              className="rounded-lg border border-slate-700 px-3 py-1.5 text-xs text-slate-300 hover:bg-slate-800"
            >
              新对话
            </button>
          </div>
        </header>

        {/* Messages */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-6">
          {messages.length === 0 ? (
            <Empty
              onPick={(s) => setInput(s)}
            />
          ) : (
            <div className="mx-auto flex max-w-3xl flex-col gap-5">
              {messages.map((m) => (
                <Bubble
                  key={m.id}
                  msg={m}
                  onShowSources={() => showSources(m.id)}
                  onCite={(num, docId) => onCite(m.id, num, docId)}
                  active={sourcesFor === m.id}
                />
              ))}
            </div>
          )}
        </div>

        {/* Composer */}
        <div className="border-t border-slate-800 px-6 py-4">
          <div className="mx-auto flex max-w-3xl items-end gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              rows={1}
              placeholder="输入你的问题，Enter 发送，Shift+Enter 换行"
              className="max-h-40 min-h-[2.75rem] flex-1 resize-none rounded-xl border border-slate-700 bg-slate-900 px-4 py-3 text-sm outline-none focus:border-blue-500"
            />
            {busy ? (
              <button
                onClick={stop}
                className="h-11 rounded-xl bg-rose-600 px-5 text-sm font-medium text-white hover:bg-rose-700"
              >
                停止
              </button>
            ) : (
              <button
                onClick={send}
                disabled={!input.trim()}
                className="btn-primary h-11"
              >
                发送
              </button>
            )}
          </div>
        </div>
      </div>

      {sourcesFor && (
        <SourcesPanel
          items={activeItems}
          api={api}
          tab={panelTab}
          onTabChange={setPanelTab}
          highlightNum={highlight?.msgId === sourcesFor ? highlight.num : null}
          highlightDocId={highlight?.msgId === sourcesFor ? highlight.docId : null}
          onClose={() => {
            setSourcesFor(null);
            setHighlight(null);
          }}
        />
      )}
    </div>
  );
}

function Bubble({
  msg,
  onShowSources,
  onCite,
  active,
}: {
  msg: Message;
  onShowSources: () => void;
  onCite: (num: number, docId?: string) => void;
  active: boolean;
}) {
  const parsed = useMemo(
    () => parseCitations(msg.content, msg.hits ?? [], msg.context, { research: msg.expert }),
    [msg.content, msg.hits, msg.context, msg.expert]
  );
  const sourceCount = parsed.hasCitations
    ? parsed.citedHits.length
    : msg.hits?.length ?? 0;

  // num -> docId, 供角标点击时定位到对应文献简介
  const docIdByNum = useMemo(() => {
    const map = new Map<number, string>();
    for (const c of parsed.citedHits) {
      map.set(c.num, c.hit.doc_id || c.hit.doc_name || "");
    }
    return map;
  }, [parsed.citedHits]);

  // 流式生成时隐藏原始引用方括号; 生成完成后再渲染带角标的 markdown;
  // wrapBareLatex 兜底: 为缺少 $ 定界符的裸 LaTeX 自动补包裹
  const rendered = wrapBareLatex(
    msg.streaming
      ? stripStreamingCitations(msg.content)
      : parsed.markdown
  );

  const mdComponents = useMemo(
    () => ({
      a({ href, children, ...rest }: { href?: string; children?: React.ReactNode }) {
        if (typeof href === "string" && href.startsWith("#cite-")) {
          const num = Number(href.slice("#cite-".length));
          return (
            <button
              type="button"
              className="cite-marker"
              onClick={(e) => {
                e.preventDefault();
                onCite(num, docIdByNum.get(num));
              }}
              title="查看引用文献简介"
            >
              {num}
            </button>
          );
        }
        return (
          <a href={href} target="_blank" rel="noreferrer" {...rest}>
            {children}
          </a>
        );
      },
    }),
    [onCite, docIdByNum]
  );

  if (msg.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-blue-600 px-4 py-2.5 text-sm text-white">
          {msg.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[88%]">
        <div className="rounded-2xl rounded-bl-md border border-slate-800 bg-slate-900/70 px-4 py-3 text-sm">
          {msg.status && (
            <div className="flex items-center gap-2 text-slate-400">
              <Spinner />
              {msg.status}
            </div>
          )}
          {msg.thinking && (
            <ThinkingPanel
              text={msg.thinking}
              streaming={!!msg.streaming && !msg.content}
            />
          )}
          {msg.error && (
            <div className="text-rose-400">
              ⚠ 出错了：{msg.error}
            </div>
          )}
          {msg.content && (
            <div className="markdown text-slate-100">
              <ReactMarkdown
                remarkPlugins={[remarkMath, remarkGfm]}
                rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: false }]]}
                components={mdComponents}
              >
                {rendered}
              </ReactMarkdown>
              {msg.streaming && <span className="cursor-blink">▋</span>}
            </div>
          )}
          {msg.signals?.needs_clarify && (
            <div className="mt-2 rounded-lg bg-amber-500/10 px-2 py-1 text-xs text-amber-300">
              需要澄清：请补充更多信息
            </div>
          )}
          {!msg.streaming && msg.research && msg.research.rounds > 0 && (
            <ResearchBadges research={msg.research} />
          )}
        </div>

        {(msg.hits?.length || msg.latency != null) && !msg.streaming && (
          <div className="mt-1.5 flex items-center gap-3 px-1 text-xs text-slate-500">
            {msg.hits && msg.hits.length > 0 && (
              <button
                onClick={onShowSources}
                className={`flex items-center gap-1 hover:text-blue-400 ${
                  active ? "text-blue-400" : ""
                }`}
              >
                📎 {sourceCount} 条
                {parsed.hasCitations ? "引用" : "来源"}
              </button>
            )}
            {msg.latency != null && <span>· {msg.latency.toFixed(2)}s</span>}
            {msg.signals?.retry_count ? (
              <span>· 重试 {msg.signals.retry_count}</span>
            ) : null}
          </div>
        )}
      </div>
    </div>
  );
}

function Empty({ onPick }: { onPick: (s: string) => void }) {
  return (
    <div className="flex h-full flex-col items-center justify-center text-center">
      <div className="mb-3 flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 text-2xl">
        💬
      </div>
      <h2 className="text-lg font-semibold">向科研文献知识库提问</h2>
      <p className="mt-1 max-w-md text-sm text-slate-400">
        基于 Agentic RAG 检索与生成，回答会附带可溯源的文献片段。
      </p>
      <p className="mt-2 max-w-md text-xs text-slate-500">
        <span className="text-slate-400">⚡ 快速检索</span> 适合查事实；
        <span className="text-violet-300">🎓 专家模式</span>{" "}
        做多轮递进式文献研究并综合成综述（耗时更长）。可在右上角切换。
      </p>
      <div className="mt-6 flex flex-wrap justify-center gap-2">
        {SAMPLES.map((s) => (
          <button
            key={s}
            onClick={() => onPick(s)}
            className="rounded-full border border-slate-700 bg-slate-900/60 px-3.5 py-1.5 text-sm text-slate-300 hover:border-blue-500/60 hover:text-blue-200"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}

function Pill({
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
      onClick={onClick}
      className={`rounded-full px-2.5 py-1 text-xs transition ${
        active
          ? "bg-blue-600/25 text-blue-200 ring-1 ring-blue-500/40"
          : "bg-slate-800 text-slate-400 hover:text-slate-200"
      }`}
    >
      {children}
    </button>
  );
}

function ModeToggle({
  expert,
  disabled,
  onChange,
}: {
  expert: boolean;
  disabled?: boolean;
  onChange: (expert: boolean) => void;
}) {
  return (
    <div
      className="flex items-center rounded-full bg-slate-800 p-0.5 text-xs"
      role="tablist"
      aria-label="检索模式"
    >
      <button
        role="tab"
        aria-selected={!expert}
        disabled={disabled}
        onClick={() => onChange(false)}
        title="快速检索：单轮 Agentic RAG，秒级返回，适合查事实"
        className={`rounded-full px-3 py-1 transition disabled:opacity-50 ${
          !expert
            ? "bg-blue-600 text-white shadow"
            : "text-slate-400 hover:text-slate-200"
        }`}
      >
        ⚡ 快速检索
      </button>
      <button
        role="tab"
        aria-selected={expert}
        disabled={disabled}
        onClick={() => onChange(true)}
        title="专家模式：多轮递进式文献检索 + 综述综合，耗时更长，适合做文献研究"
        className={`rounded-full px-3 py-1 transition disabled:opacity-50 ${
          expert
            ? "bg-violet-600 text-white shadow"
            : "text-slate-400 hover:text-slate-200"
        }`}
      >
        🎓 专家模式
      </button>
    </div>
  );
}

function ThinkingPanel({
  text,
  streaming,
}: {
  text: string;
  streaming: boolean;
}) {
  // 思考时默认展开实时跟读; 思考结束 (开始输出正文) 后默认折叠, 与"真正输出内容"区分开。
  const [open, setOpen] = useState(true);
  useEffect(() => {
    if (!streaming) setOpen(false);
  }, [streaming]);

  return (
    <div className="mb-2 rounded-lg border border-violet-500/20 bg-violet-500/5">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-1.5 px-2.5 py-1.5 text-xs text-violet-300/90 hover:text-violet-200"
      >
        {streaming ? <Spinner /> : <span className="text-violet-400">🧠</span>}
        <span className="font-medium">思考过程</span>
        <span className="ml-auto text-violet-400/60">{open ? "收起 ▲" : "展开 ▼"}</span>
      </button>
      {open && (
        <div className="max-h-72 overflow-y-auto whitespace-pre-wrap border-t border-violet-500/15 px-3 py-2 text-xs leading-relaxed text-slate-400">
          {text}
          {streaming && <span className="cursor-blink">▋</span>}
        </div>
      )}
    </div>
  );
}

function ResearchBadges({ research }: { research: ResearchMeta }) {
  const insufficient = research.status === "insufficient";
  return (
    <div className="mt-2 flex flex-wrap items-center gap-1.5 text-xs">
      {insufficient ? (
        <span className="rounded bg-amber-500/15 px-2 py-0.5 text-amber-300 ring-1 ring-amber-500/30">
          🎓 专家模式 · 证据不足
        </span>
      ) : (
        <span className="rounded bg-violet-500/15 px-2 py-0.5 text-violet-300 ring-1 ring-violet-500/30">
          🎓 专家模式
        </span>
      )}
      <span className="rounded bg-slate-800 px-2 py-0.5 text-slate-300">
        {research.rounds} 轮检索
      </span>
      <span className="rounded bg-slate-800 px-2 py-0.5 text-slate-300">
        {research.evidence_docs} 篇文献 · {research.evidence_chunks} 条证据
      </span>
      {research.gaps && research.gaps.length > 0 && (
        <span
          className="rounded bg-amber-500/10 px-2 py-0.5 text-amber-300"
          title={research.gaps.join("\n")}
        >
          ⚠ {research.gaps.length} 项证据缺口
        </span>
      )}
    </div>
  );
}

function Spinner() {
  return (
    <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-slate-600 border-t-blue-400" />
  );
}

function rid() {
  return Math.random().toString(36).slice(2, 10);
}
