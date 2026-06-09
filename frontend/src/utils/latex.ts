/**
 * 前端 LaTeX 安全网：为缺少 $...$ / $$...$$ 定界符的裸 LaTeX 自动补包裹，
 * 使 markdown 数学渲染（KaTeX）能正确识别。保守检测，宁可漏掉也不误伤正文。
 */

const LEFT_RIGHT_PAIR = /\\left\s*[(\\{].*\\right\s*[)\\}]/s
const LATEX_CMD = /\\[a-zA-Z]/g
const CJK_CHAR = /[\u4e00-\u9fff\u3400-\u4dbf]/g

function normalizeLatexSpaces(latex: string): string {
  return latex.replace(/(\\[a-zA-Z]+)\s+\{/g, '$1{').replace(/([_^])\s+\{/g, '$1{')
}

export function wrapBareLatex(md: string): string {
  if (!md) return md
  const lines = md.split('\n')
  const out: string[] = []
  let inCodeBlock = false
  let inMathBlock = false

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const trimmed = line.trim()

    if (trimmed.startsWith('```')) {
      inCodeBlock = !inCodeBlock
      out.push(line)
      continue
    }
    if (inCodeBlock) {
      out.push(line)
      continue
    }
    if (trimmed === '$$') {
      inMathBlock = !inMathBlock
      out.push(line)
      continue
    }
    if (inMathBlock) {
      out.push(line)
      continue
    }
    if (line.includes('$')) {
      out.push(line)
      continue
    }
    if (
      !trimmed ||
      /^#{1,6}\s/.test(trimmed) ||
      /^[>\-*+]\s/.test(trimmed) ||
      /^\d+\.\s/.test(trimmed) ||
      trimmed === '---' ||
      trimmed === '***'
    ) {
      out.push(line)
      continue
    }

    const normalized = normalizeLatexSpaces(trimmed)

    if (/\\tag\s*\{/.test(normalized)) {
      out.push('$$', normalized, '$$')
      continue
    }
    if (/\\begin\s*\{(?:equation|align|gather|multline|cases|split|aligned)\b/.test(normalized)) {
      out.push('$$', normalized)
      let j = i + 1
      for (; j < lines.length; j++) {
        const innerTrimmed = normalizeLatexSpaces(lines[j].trim())
        out.push(innerTrimmed)
        if (/\\end\s*\{(?:equation|align|gather|multline|cases|split|aligned)\b/.test(innerTrimmed)) break
      }
      out.push('$$')
      i = j
      continue
    }
    if (LEFT_RIGHT_PAIR.test(normalized)) {
      const cjkCount = (normalized.match(CJK_CHAR) || []).length
      const cmdCount = (normalized.match(LATEX_CMD) || []).length
      if (cmdCount >= 2 && cjkCount <= cmdCount) {
        out.push('$$', normalized, '$$')
        continue
      }
    }
    {
      const cmdCount = (normalized.match(LATEX_CMD) || []).length
      const cjkCount = (normalized.match(CJK_CHAR) || []).length
      if (cmdCount >= 3 && cjkCount <= 1) {
        out.push('$$', normalized, '$$')
        continue
      }
    }
    out.push(line)
  }
  return out.join('\n')
}
