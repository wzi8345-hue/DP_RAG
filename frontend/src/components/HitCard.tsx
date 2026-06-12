import { useState } from "react";
import type { Hit } from "../lib/types";

export function HitCard({
  hit,
  num,
  highlight,
  onOpenPdf,
}: {
  hit: Hit;
  num: number;
  highlight?: boolean;
  /** 点击「原文定位」: 在 PDF 中高亮该 chunk (需 doc_id + chunk_id) */
  onOpenPdf?: (docId: string, chunkId: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const score =
    hit.rerank_score ?? hit.rrf_score ?? hit.score ?? undefined;
  const text = hit.content || hit.context || "";
  const long = text.length > 280;
  const canLocate = !!(onOpenPdf && hit.doc_id && hit.chunk_id);

  return (
    <div
      className={`rounded-xl border bg-slate-900/60 p-3 text-sm transition ${
        highlight
          ? "border-blue-500 ring-1 ring-blue-500/50"
          : "border-slate-800"
      }`}
    >
      <div className="mb-1.5 flex items-start justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded bg-blue-600/30 text-xs font-medium text-blue-200">
            {num}
          </span>
          <span className="truncate font-medium text-slate-200" title={hit.doc_name}>
            {hit.doc_name || hit.doc_id || "未知文档"}
          </span>
        </div>
        {typeof score === "number" && (
          <span className="shrink-0 rounded bg-slate-800 px-1.5 py-0.5 text-xs text-slate-400">
            {score.toFixed(3)}
          </span>
        )}
      </div>

      <div className="mb-2 flex flex-wrap gap-1.5 text-xs text-slate-400">
        {hit.section && <Tag>§ {hit.section}</Tag>}
        {typeof hit.page_start === "number" && hit.page_start >= 0 && (
          <Tag>p.{hit.page_start}</Tag>
        )}
        {hit.type && <Tag>{hit.type}</Tag>}
        {hit.publication_year ? <Tag>{hit.publication_year}</Tag> : null}
        {hit.sources?.map((s) => (
          <Tag key={s} accent>
            {s}
          </Tag>
        ))}
      </div>

      <p
        className={`whitespace-pre-wrap leading-relaxed text-slate-300 ${
          !expanded && long ? "line-clamp-4" : ""
        }`}
      >
        {text || <span className="italic text-slate-500">（无文本内容）</span>}
      </p>

      <div className="mt-1 flex items-center gap-3">
        {long && (
          <button
            onClick={() => setExpanded((v) => !v)}
            className="text-xs text-blue-400 hover:text-blue-300"
          >
            {expanded ? "收起" : "展开全文"}
          </button>
        )}
        {canLocate && (
          <button
            onClick={() => onOpenPdf!(hit.doc_id!, hit.chunk_id!)}
            className="text-xs text-amber-400 hover:text-amber-300"
            title="在原文 PDF 中定位高亮"
          >
            📄 原文定位
          </button>
        )}
      </div>

      {hit.matched_keywords && hit.matched_keywords.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {hit.matched_keywords.map((k) => (
            <span
              key={k}
              className="rounded bg-amber-500/15 px-1.5 py-0.5 text-xs text-amber-300"
            >
              {k}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function Tag({
  children,
  accent,
}: {
  children: React.ReactNode;
  accent?: boolean;
}) {
  return (
    <span
      className={`rounded px-1.5 py-0.5 ${
        accent
          ? "bg-indigo-500/15 text-indigo-300"
          : "bg-slate-800 text-slate-400"
      }`}
    >
      {children}
    </span>
  );
}
