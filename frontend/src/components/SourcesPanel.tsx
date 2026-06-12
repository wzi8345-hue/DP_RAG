import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import type { ApiClient } from "../lib/api";
import type { BBox, DocSummaryResponse, Hit } from "../lib/types";
import { wrapBareLatex } from "../lib/latex";
import { HitCard } from "./HitCard";
import { PdfViewer } from "./PdfViewer";

export interface SourceItem {
  num: number;
  hit: Hit;
}

export type PanelTab = "summary" | "chunks" | "pdf";

/** 「原文」tab 当前定位的目标 chunk 及其定位框 */
export interface PdfTarget {
  docId: string;
  docName?: string;
  chunkId: string;
  bboxes: BBox[];
  /** 每次点击新角标/原文定位时变化, 触发 PdfViewer 重新滚动 */
  key: string;
}

interface DocRef {
  docId: string;
  docName: string;
  nums: number[];
}

function distinctDocs(items: SourceItem[]): DocRef[] {
  const map = new Map<string, DocRef>();
  for (const { num, hit } of items) {
    const docId = hit.doc_id || hit.doc_name || `#${num}`;
    const existing = map.get(docId);
    if (existing) existing.nums.push(num);
    else
      map.set(docId, {
        docId,
        docName: hit.doc_name || hit.doc_id || "未知文献",
        nums: [num],
      });
  }
  return [...map.values()];
}

export function SourcesPanel({
  items,
  api,
  collection,
  tab,
  onTabChange,
  highlightNum,
  highlightDocId,
  pdfTarget,
  onOpenPdf,
  onClose,
}: {
  items: SourceItem[];
  api: ApiClient;
  /** 当前对话使用的知识库集合名 (取 PDF / 定位框时需要) */
  collection: string;
  tab: PanelTab;
  onTabChange: (t: PanelTab) => void;
  highlightNum?: number | null;
  highlightDocId?: string | null;
  pdfTarget?: PdfTarget | null;
  /** 引用块卡片「原文定位」回调 */
  onOpenPdf?: (docId: string, chunkId: string) => void;
  onClose: () => void;
}) {
  const chunkRefs = useRef<Record<number, HTMLDivElement | null>>({});
  const docRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const docs = distinctDocs(items);

  useEffect(() => {
    if (tab === "chunks" && highlightNum != null) {
      chunkRefs.current[highlightNum]?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    }
  }, [tab, highlightNum]);

  useEffect(() => {
    if (tab === "summary" && highlightDocId) {
      docRefs.current[highlightDocId]?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    }
  }, [tab, highlightDocId, docs.length]);

  return (
    <div
      className={`flex h-full shrink-0 flex-col border-l border-slate-800 bg-slate-950/60 transition-[width] ${
        tab === "pdf" ? "w-[42rem] max-w-[50vw]" : "w-96"
      }`}
    >
      <div className="flex items-center justify-between border-b border-slate-800 px-4 py-2.5">
        <div className="flex gap-1">
          <TabBtn active={tab === "summary"} onClick={() => onTabChange("summary")}>
            文献简介 ({docs.length})
          </TabBtn>
          <TabBtn active={tab === "chunks"} onClick={() => onTabChange("chunks")}>
            引用块 ({items.length})
          </TabBtn>
          <TabBtn active={tab === "pdf"} onClick={() => onTabChange("pdf")}>
            原文
          </TabBtn>
        </div>
        <button
          onClick={onClose}
          className="rounded p-1 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
          aria-label="关闭"
        >
          ✕
        </button>
      </div>

      {tab === "pdf" ? (
        <PdfTab api={api} collection={collection} target={pdfTarget} />
      ) : (
        <div className="flex-1 space-y-3 overflow-y-auto p-4">
          {items.length === 0 ? (
            <p className="mt-8 text-center text-sm text-slate-500">
              本次回答没有引用检索结果
            </p>
          ) : tab === "summary" ? (
            docs.map((d) => (
              <div
                key={d.docId}
                ref={(el) => {
                  docRefs.current[d.docId] = el;
                }}
              >
                <DocSummaryCard
                  api={api}
                  docRef={d}
                  highlight={highlightDocId === d.docId}
                />
              </div>
            ))
          ) : (
            items.map(({ num, hit }) => (
              <div
                key={num}
                ref={(el) => {
                  chunkRefs.current[num] = el;
                }}
              >
                <HitCard
                  hit={hit}
                  num={num}
                  highlight={highlightNum === num}
                  onOpenPdf={onOpenPdf}
                />
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}

function PdfTab({
  api,
  collection,
  target,
}: {
  api: ApiClient;
  collection: string;
  target?: PdfTarget | null;
}) {
  if (!target?.docId) {
    return (
      <div className="flex flex-1 items-center justify-center p-6 text-center text-xs text-slate-500">
        点击回答中的引用角标，或引用块里的「📄 原文定位」，
        即可在此查看原文 PDF 并高亮对应位置。
      </div>
    );
  }
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="truncate border-b border-slate-800 px-4 py-1.5 text-xs text-slate-400">
        {target.docName || target.docId}
        {target.bboxes.length === 0 && (
          <span className="ml-2 text-slate-500">（该块无定位框，仅展示原文）</span>
        )}
      </div>
      <div className="min-h-0 flex-1">
        <PdfViewer
          api={api}
          collection={collection}
          docId={target.docId}
          highlightBboxes={target.bboxes}
          highlightKey={target.key}
        />
      </div>
    </div>
  );
}

function DocSummaryCard({
  api,
  docRef,
  highlight,
}: {
  api: ApiClient;
  docRef: DocRef;
  highlight?: boolean;
}) {
  const [data, setData] = useState<DocSummaryResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    api
      .docSummary(docRef.docId)
      .then((d) => alive && (setData(d), setErr(null)))
      .catch((e) => alive && setErr(e instanceof Error ? e.message : String(e)))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [api, docRef.docId]);

  return (
    <div
      className={`rounded-xl border bg-slate-900/60 p-3 text-sm transition ${
        highlight ? "border-blue-500 ring-1 ring-blue-500/50" : "border-slate-800"
      }`}
    >
      <div className="mb-1.5 flex items-start gap-2">
        <div className="flex flex-wrap gap-1">
          {docRef.nums.map((n) => (
            <span
              key={n}
              className="flex h-5 min-w-5 items-center justify-center rounded bg-blue-600/30 px-1 text-xs font-medium text-blue-200"
            >
              {n}
            </span>
          ))}
        </div>
        <span className="min-w-0 font-medium text-slate-200" title={docRef.docName}>
          {data?.title && data.title !== docRef.docId ? data.title : docRef.docName}
        </span>
      </div>

      {data?.year ? (
        <div className="mb-1.5 text-xs text-slate-400">{data.year} 年</div>
      ) : null}

      {loading ? (
        <p className="text-xs text-slate-500">加载简介中…</p>
      ) : err ? (
        <p className="text-xs text-rose-400">简介加载失败：{err}</p>
      ) : data?.summary ? (
        <div className="markdown leading-relaxed text-slate-300">
          <ReactMarkdown
            remarkPlugins={[remarkMath, remarkGfm]}
            rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: false }]]}
          >
            {wrapBareLatex(data.summary)}
          </ReactMarkdown>
        </div>
      ) : (
        <p className="text-xs italic text-slate-500">（该文献暂无简介摘要）</p>
      )}
    </div>
  );
}

function TabBtn({
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
      className={`rounded-lg px-2.5 py-1 text-xs font-medium transition ${
        active
          ? "bg-blue-600/25 text-blue-200 ring-1 ring-blue-500/40"
          : "text-slate-400 hover:bg-slate-800 hover:text-slate-200"
      }`}
    >
      {children}
    </button>
  );
}
