import { forwardRef, useEffect, useLayoutEffect, useRef, useState } from "react";
import * as pdfjsLib from "pdfjs-dist";
import workerSrc from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import type { PDFPageProxy } from "pdfjs-dist";
import type { ApiClient } from "../lib/api";
import type { BBox } from "../lib/types";

pdfjsLib.GlobalWorkerOptions.workerSrc = workerSrc;

/**
 * 内嵌 PDF 阅读器: 整篇渲染 + 对指定 chunk 的源块定位框做高亮叠加并自动滚动定位。
 *
 * bbox 约定: page 为 0-based 页码, bbox=[x1,y1,x2,y2] 为页内归一化坐标
 * (0~1, 原点左上), 与渲染后的页面像素尺寸相乘即得叠加框位置。
 */
export function PdfViewer({
  api,
  collection,
  docId,
  highlightBboxes,
  highlightKey,
}: {
  api: ApiClient;
  collection: string;
  docId: string;
  /** 当前要高亮的 chunk 的定位框 (可跨页) */
  highlightBboxes: BBox[];
  /** 每次点击新角标时变化, 触发重新滚动定位 (即便同一篇文档) */
  highlightKey: string;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const pageRefs = useRef<(HTMLDivElement | null)[]>([]);
  const [proxies, setProxies] = useState<PDFPageProxy[]>([]);
  const [scale, setScale] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let doc: pdfjsLib.PDFDocumentProxy | null = null;
    setLoading(true);
    setError(null);
    setProxies([]);
    pageRefs.current = [];

    const container = scrollRef.current;
    const containerWidth = container ? container.clientWidth - 24 : 760;

    (async () => {
      try {
        const task = pdfjsLib.getDocument({
          url: api.pdfUrl(collection, docId),
          httpHeaders: api.authHeaders(),
          withCredentials: false,
        });
        doc = await task.promise;
        if (cancelled) return;
        const all: PDFPageProxy[] = [];
        for (let i = 1; i <= doc.numPages; i++) {
          all.push(await doc.getPage(i));
          if (cancelled) return;
        }
        const base = all[0].getViewport({ scale: 1 });
        if (cancelled) return;
        setScale(containerWidth / base.width);
        setProxies(all);
        setLoading(false);
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
          setLoading(false);
        }
      }
    })();

    return () => {
      cancelled = true;
      doc?.destroy();
    };
  }, [api, collection, docId]);

  // 滚动定位到高亮块所在页 (高亮块或点击的角标变化时)
  useLayoutEffect(() => {
    if (loading || proxies.length === 0 || highlightBboxes.length === 0) return;
    const firstPage = Math.min(...highlightBboxes.map((b) => b.page));
    const wrapper = pageRefs.current[firstPage];
    const proxy = proxies[firstPage];
    if (!wrapper || !proxy) return;
    const h = proxy.getViewport({ scale }).height;
    const box = highlightBboxes.find((b) => b.page === firstPage);
    const offsetTop = box ? box.bbox[1] * h : 0;
    scrollRef.current?.scrollTo({
      top: Math.max(0, wrapper.offsetTop + offsetTop - 120),
      behavior: "smooth",
    });
  }, [highlightKey, loading, proxies, scale]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div ref={scrollRef} className="h-full overflow-y-auto bg-slate-800/40 p-3">
      {error ? (
        <div className="mt-8 rounded-lg border border-slate-700 bg-slate-900/60 p-4 text-center text-xs text-slate-400">
          原文 PDF 暂不可用
          <div className="mt-1 break-all text-slate-500">{error}</div>
          <div className="mt-2 text-slate-500">
            （仅对重新灌入、且保留了原始 PDF 的文献可用）
          </div>
        </div>
      ) : null}

      {loading && !error ? (
        <div className="mt-8 text-center text-xs text-slate-500">加载原文 PDF…</div>
      ) : null}

      {proxies.map((proxy, i) => (
        <PdfPage
          key={i}
          ref={(el) => {
            pageRefs.current[i] = el;
          }}
          page={proxy}
          scale={scale}
          boxes={highlightBboxes.filter((b) => b.page === i)}
        />
      ))}
    </div>
  );
}

const PdfPage = forwardRef<
  HTMLDivElement,
  { page: PDFPageProxy; scale: number; boxes: BBox[] }
>(function PdfPage({ page, scale, boxes }, ref) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const viewport = page.getViewport({ scale });
  const width = Math.floor(viewport.width);
  const height = Math.floor(viewport.height);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = width;
    canvas.height = height;
    const task = page.render({ canvasContext: ctx, viewport });
    task.promise.catch(() => {
      /* 渲染被取消 (重渲染/卸载) 时忽略 */
    });
    return () => task.cancel();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page, scale]);

  return (
    <div
      ref={ref}
      className="relative mx-auto mb-3 shadow-lg shadow-black/40"
      style={{ width, height }}
    >
      <canvas ref={canvasRef} className="block" style={{ width, height }} />
      {boxes.map((b, j) => (
        <div
          key={j}
          className="pointer-events-none absolute rounded-sm bg-amber-400/25 ring-2 ring-amber-400/80"
          style={{
            left: b.bbox[0] * width,
            top: b.bbox[1] * height,
            width: (b.bbox[2] - b.bbox[0]) * width,
            height: (b.bbox[3] - b.bbox[1]) * height,
          }}
        />
      ))}
    </div>
  );
});
