import type { Hit } from "./types";

// 模型输出的行内引用形如:
//   [text_8bb1f28e, 2．1 挂片试验溶液的选择, page 2, para 9]
//   [summary_1a23fe52, 1 锌铝镁镀层概述, page 0]
// 第一个 token (text_/table_/image_/summary_ ...) 即 chunk_id。
// 把整段方括号引用替换成可点击的脚注角标 [n](#cite-n)。
//
// 关键: agentic 模式下模型引用的 chunk 来自多路径完整 context, 其中很多并不在
// 返回的 hits 数组里 (例如 summary 摘要块、被精排丢弃的块)。因此除了用 hits 解析,
// 还要解析后端回传的 context 字符串 —— 它才是「模型实际看到、可被引用」的全集。

const CITATION_RE = /\[[^[\]\n]*?[a-z]+_[0-9a-f]{4,}[^[\]\n]*?\]/gi;
const CHUNK_ID_RE = /[a-z]+_[0-9a-f]{4,}/gi;

// 专家模式综述里模型按"研究证据材料"的文献序号引用, 形如 [11] 或 [18, 20]。
// 这些数字对应 research context 中的 "## [n] 文献名" 标题, 而非 chunk_id。
const NUMERIC_CITATION_RE = /\[(\d{1,3}(?:\s*[,，、]\s*\d{1,3})*)\]/g;

// 解析 research context 里的文献序号标题: "## [1] 文献名" → Map<序号, 文献名>。
function parseResearchDocHeaders(context: string): Map<number, string> {
  const map = new Map<number, string>();
  if (!context) return map;
  const re = /^\s*#{1,3}\s*\[(\d+)\]\s+(.+?)\s*$/gm;
  let m: RegExpExecArray | null;
  while ((m = re.exec(context))) {
    const idx = Number(m[1]);
    if (!map.has(idx)) map.set(idx, m[2].trim());
  }
  return map;
}

// 文献名/doc_id → 该文献的首个 hit, 供专家模式角标定位到文献。
function buildDocIndex(hits: Hit[]): Map<string, Hit> {
  const idx = new Map<string, Hit>();
  for (const h of hits) {
    if (h.doc_name && !idx.has(h.doc_name)) idx.set(h.doc_name, h);
    if (h.doc_id && !idx.has(h.doc_id)) idx.set(h.doc_id, h);
  }
  return idx;
}

export interface CitedHit {
  num: number;
  chunkId: string;
  hit: Hit;
}

export interface ParsedAnswer {
  /** 角标化后的 markdown (引用替换为 [n](#cite-n)) */
  markdown: string;
  /** 回答里实际引用、且能解析到内容的块, 按出现顺序去重编号 */
  citedHits: CitedHit[];
  hasCitations: boolean;
}

// 流式生成过程中, hits/context 尚未到达, 无法解析角标。此时直接把引用方括号
// 隐藏掉, 避免闪现 "[text_77336824, 3, page 7, para 18]" 这种原始文本;
// 生成结束 (done) 后再用 parseCitations 渲染成角标。
export function stripStreamingCitations(content: string): string {
  if (!content) return content;
  // 1) 去掉已闭合的引用方括号
  let out = content.replace(CITATION_RE, "");
  // 2) 去掉结尾尚未闭合、且像引用开头的片段 (例: "...优良 [text_77" )
  out = out.replace(/\[[^\]\n]*$/, (tail) =>
    /[a-z]+_?[0-9a-f]*/i.test(tail) && /[_]/.test(tail) ? "" : tail
  );
  return out;
}

function buildHitIndex(hits: Hit[]): Map<string, Hit> {
  const idx = new Map<string, Hit>();
  for (const h of hits) {
    if (h.chunk_id) idx.set(h.chunk_id, h);
    if (h.pk) {
      idx.set(h.pk, h);
      const tail = h.pk.split("::").pop();
      if (tail) idx.set(tail, h);
    }
  }
  return idx;
}

interface ContextChunk {
  chunkId: string;
  docId?: string;
  section?: string;
  page?: number;
  para?: number;
  type?: string;
  content: string;
}

// 解析后端 context 字符串里的 chunk 头 + 正文。
// 头部格式 (agentic / simple 两种, 共同锚点是含 "chunk_id=" 的方括号编号行):
//   [1] TEXT | chunk_id=xxx | doc=xxx | section=xxx | page=N | para=M | year=Y
//   ## [1] TEXT | source=... | doc=xxx | page=N | chunk_id=xxx
export function parseContextChunks(context: string): Map<string, ContextChunk> {
  const map = new Map<string, ContextChunk>();
  if (!context) return map;
  let cur: ContextChunk | null = null;

  const flush = () => {
    if (cur && cur.chunkId && !map.has(cur.chunkId)) {
      cur.content = cur.content.trim();
      map.set(cur.chunkId, cur);
    }
    cur = null;
  };

  for (const line of context.split(/\r?\n/)) {
    const idM = line.match(/chunk_id=([^\s|]+)/);
    const isHeader = !!idM && /^\s*#{0,2}\s*\[\d+\]/.test(line);

    if (isHeader) {
      flush();
      // "已在 [route] 路径以 #N 展示" 这类去重引用行没有正文, 但仍带 chunk_id;
      // 它出现在 chunk 首次展示之后, map 里已有该 id, flush 时会被忽略, 安全。
      const pageM = line.match(/page=(\d+)/);
      const paraM = line.match(/para=(\d+)/);
      cur = {
        chunkId: idM![1],
        docId: line.match(/doc=([^|]+?)(?:\s*\||$)/)?.[1]?.trim(),
        section: line.match(/section=([^|]+?)(?:\s*\||$)/)?.[1]?.trim(),
        page: pageM ? Number(pageM[1]) : undefined,
        para: paraM ? Number(paraM[1]) : undefined,
        type: line.match(/\[\d+\]\s+([A-Za-z]+)/)?.[1]?.toLowerCase(),
        content: "",
      };
      continue;
    }
    if (!cur) continue;

    // 边界: 下一节标题 / 分隔线 / 截断标记
    if (/^\s*(#|---|\*\()/.test(line)) {
      flush();
      continue;
    }
    // 元数据行
    const secLine = line.match(/^\s*\*\*Section:\*\*\s*(.+)$/);
    if (secLine) {
      if (!cur.section) cur.section = secLine[1].trim();
      continue;
    }
    if (/^\s*\*\*(Matched|Related|Context|Related Section Context)/.test(line)) {
      continue;
    }
    if (/^\s*\[Related Section Context\]/.test(line)) continue;

    cur.content += (cur.content ? "\n" : "") + line;
  }
  flush();
  return map;
}

function chunkToHit(c: ContextChunk): Hit {
  return {
    chunk_id: c.chunkId,
    doc_id: c.docId,
    doc_name: c.docId,
    section: c.section,
    page_start: c.page,
    paragraph_index: c.para,
    type: c.type,
    content: c.content,
  };
}

export function parseCitations(
  content: string,
  hits: Hit[],
  context?: string,
  opts?: { research?: boolean }
): ParsedAnswer {
  if (!content) {
    return { markdown: content, citedHits: [], hasCitations: false };
  }
  const hitIndex = buildHitIndex(hits);
  const ctxIndex = context ? parseContextChunks(context) : new Map();
  const numByChunk = new Map<string, number>();
  const citedHits: CitedHit[] = [];

  const resolve = (id: string): Hit | null => {
    const hit = hitIndex.get(id);
    if (hit) return hit;
    const ctx = ctxIndex.get(id);
    if (ctx) return chunkToHit(ctx);
    return null;
  };

  let markdown = content.replace(CITATION_RE, (match) => {
    const ids = match.match(CHUNK_ID_RE) || [];
    const markers: string[] = [];
    for (const id of ids) {
      const hit = resolve(id);
      if (!hit) continue;
      let num = numByChunk.get(id);
      if (num == null) {
        num = citedHits.length + 1;
        numByChunk.set(id, num);
        citedHits.push({ num, chunkId: id, hit });
      }
      markers.push(`[${num}](#cite-${num})`);
    }
    return markers.length > 0 ? markers.join("") : match;
  });

  // 专家模式: 把数字文献序号引用 [11] / [18, 20] 同样转成角标 [n](#cite-n),
  // 与快速检索一致 (按出现顺序重新连续编号, 仅转换能对应到文献标题的序号)。
  if (opts?.research) {
    const docHeaders = parseResearchDocHeaders(context || "");
    if (docHeaders.size > 0) {
      const docIndex = buildDocIndex(hits);
      markdown = markdown.replace(NUMERIC_CITATION_RE, (match, body: string) => {
        const indices = body.split(/[,，、]/).map((s) => Number(s.trim()));
        const markers: string[] = [];
        for (const docIdx of indices) {
          const docName = docHeaders.get(docIdx);
          if (docName == null) continue; // 非文献序号 (如年份), 原样保留
          const key = `doc#${docIdx}`;
          let num = numByChunk.get(key);
          if (num == null) {
            num = citedHits.length + 1;
            numByChunk.set(key, num);
            const hit: Hit =
              docIndex.get(docName) || { doc_id: docName, doc_name: docName, content: "" };
            citedHits.push({ num, chunkId: key, hit });
          }
          markers.push(`[${num}](#cite-${num})`);
        }
        return markers.length > 0 ? markers.join("") : match;
      });
    }
  }

  return { markdown, citedHits, hasCitations: citedHits.length > 0 };
}
