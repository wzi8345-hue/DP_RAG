import type { Hit } from '@/api/types'

// 行内引用形如 [text_8bb1f28e, 章节, page 2, para 9]；第一个 token 即 chunk_id。
// 把方括号引用替换成可点击角标 [n](#cite-n)。专家模式综述用文献序号 [11] / [18,20]。

const CITATION_RE = /\[[^[\]\n]*?[a-z]+_[0-9a-f]{4,}[^[\]\n]*?\]/gi
const CHUNK_ID_RE = /[a-z]+_[0-9a-f]{4,}/gi
const NUMERIC_CITATION_RE = /\[(\d{1,3}(?:\s*[,，、]\s*\d{1,3})*)\]/g

function parseResearchDocHeaders(context: string): Map<number, string> {
  const map = new Map<number, string>()
  if (!context) return map
  const re = /^\s*#{1,3}\s*\[(\d+)\]\s+(.+?)\s*$/gm
  let m: RegExpExecArray | null
  while ((m = re.exec(context))) {
    const idx = Number(m[1])
    if (!map.has(idx)) map.set(idx, m[2].trim())
  }
  return map
}

function buildDocIndex(hits: Hit[]): Map<string, Hit> {
  const idx = new Map<string, Hit>()
  for (const h of hits) {
    if (h.doc_name && !idx.has(h.doc_name)) idx.set(h.doc_name, h)
    if (h.doc_id && !idx.has(h.doc_id)) idx.set(h.doc_id, h)
  }
  return idx
}

export interface CitedHit {
  num: number
  chunkId: string
  hit: Hit
}

export interface ParsedAnswer {
  markdown: string
  citedHits: CitedHit[]
  hasCitations: boolean
}

export function stripStreamingCitations(content: string): string {
  if (!content) return content
  let out = content.replace(CITATION_RE, '')
  out = out.replace(/\[[^\]\n]*$/, (tail) =>
    /[a-z]+_?[0-9a-f]*/i.test(tail) && /[_]/.test(tail) ? '' : tail,
  )
  return out
}

function buildHitIndex(hits: Hit[]): Map<string, Hit> {
  const idx = new Map<string, Hit>()
  for (const h of hits) {
    if (h.chunk_id) idx.set(h.chunk_id, h)
    if (h.pk) {
      idx.set(h.pk, h)
      const tail = h.pk.split('::').pop()
      if (tail) idx.set(tail, h)
    }
  }
  return idx
}

interface ContextChunk {
  chunkId: string
  docId?: string
  section?: string
  page?: number
  para?: number
  type?: string
  content: string
}

export function parseContextChunks(context: string): Map<string, ContextChunk> {
  const map = new Map<string, ContextChunk>()
  if (!context) return map
  let cur: ContextChunk | null = null

  const flush = () => {
    if (cur && cur.chunkId && !map.has(cur.chunkId)) {
      cur.content = cur.content.trim()
      map.set(cur.chunkId, cur)
    }
    cur = null
  }

  for (const line of context.split(/\r?\n/)) {
    const idM = line.match(/chunk_id=([^\s|]+)/)
    const isHeader = !!idM && /^\s*#{0,2}\s*\[\d+\]/.test(line)

    if (isHeader) {
      flush()
      const pageM = line.match(/page=(\d+)/)
      const paraM = line.match(/para=(\d+)/)
      cur = {
        chunkId: idM![1],
        docId: line.match(/doc=([^|]+?)(?:\s*\||$)/)?.[1]?.trim(),
        section: line.match(/section=([^|]+?)(?:\s*\||$)/)?.[1]?.trim(),
        page: pageM ? Number(pageM[1]) : undefined,
        para: paraM ? Number(paraM[1]) : undefined,
        type: line.match(/\[\d+\]\s+([A-Za-z]+)/)?.[1]?.toLowerCase(),
        content: '',
      }
      continue
    }
    if (!cur) continue
    if (/^\s*(#|---|\*\()/.test(line)) {
      flush()
      continue
    }
    const secLine = line.match(/^\s*\*\*Section:\*\*\s*(.+)$/)
    if (secLine) {
      if (!cur.section) cur.section = secLine[1].trim()
      continue
    }
    if (/^\s*\*\*(Matched|Related|Context|Related Section Context)/.test(line)) continue
    if (/^\s*\[Related Section Context\]/.test(line)) continue
    cur.content += (cur.content ? '\n' : '') + line
  }
  flush()
  return map
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
  }
}

export function parseCitations(
  content: string,
  hits: Hit[],
  context?: string,
  opts?: { research?: boolean },
): ParsedAnswer {
  if (!content) return { markdown: content, citedHits: [], hasCitations: false }
  const hitIndex = buildHitIndex(hits)
  const ctxIndex = context ? parseContextChunks(context) : new Map<string, ContextChunk>()
  const numByChunk = new Map<string, number>()
  const citedHits: CitedHit[] = []

  const resolve = (id: string): Hit | null => {
    const hit = hitIndex.get(id)
    if (hit) return hit
    const ctx = ctxIndex.get(id)
    if (ctx) return chunkToHit(ctx)
    return null
  }

  let markdown = content.replace(CITATION_RE, (match) => {
    const ids = match.match(CHUNK_ID_RE) || []
    const markers: string[] = []
    for (const id of ids) {
      const hit = resolve(id)
      if (!hit) continue
      let num = numByChunk.get(id)
      if (num == null) {
        num = citedHits.length + 1
        numByChunk.set(id, num)
        citedHits.push({ num, chunkId: id, hit })
      }
      markers.push(`[${num}](#cite-${num})`)
    }
    return markers.length > 0 ? markers.join('') : match
  })

  if (opts?.research) {
    const docHeaders = parseResearchDocHeaders(context || '')
    if (docHeaders.size > 0) {
      const docIndex = buildDocIndex(hits)
      markdown = markdown.replace(NUMERIC_CITATION_RE, (match, body: string) => {
        const indices = body.split(/[,，、]/).map((s) => Number(s.trim()))
        const markers: string[] = []
        for (const docIdx of indices) {
          const docName = docHeaders.get(docIdx)
          if (docName == null) continue
          const key = `doc#${docIdx}`
          let num = numByChunk.get(key)
          if (num == null) {
            num = citedHits.length + 1
            numByChunk.set(key, num)
            const hit: Hit = docIndex.get(docName) || { doc_id: docName, doc_name: docName, content: '' }
            citedHits.push({ num, chunkId: key, hit })
          }
          markers.push(`[${num}](#cite-${num})`)
        }
        return markers.length > 0 ? markers.join('') : match
      })
    }
  }

  return { markdown, citedHits, hasCitations: citedHits.length > 0 }
}
