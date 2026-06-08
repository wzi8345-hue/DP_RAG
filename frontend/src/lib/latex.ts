/**
 * 前端 LaTeX 安全网: 检测模型输出中未包裹 $...$ / $$...$$ 定界符的裸 LaTeX,
 * 自动补充定界符, 使 remarkMath + rehypeKatex 能正确渲染。
 *
 * 兜底策略 (保守, 避免误伤):
 *   - 含 \tag{...} 的行 → 独立公式块 $$...$$
 *   - \begin{equation/align/...} ... \end{...} 块 → $$...$$
 *   - 含 \left(...\right) 且无中文段落特征的行 → $$...$$
 *   - 含多个 LaTeX 命令且明显是公式的行 → $$...$$
 *   - 已经有 $ 定界符的行不做处理
 *
 * 同时对 LaTeX 内容做空格归一化, 消除模型输出中的不规范空格:
 *   \frac {x}{y} → \frac{x}{y}
 *   \sigma^ {A}  → \sigma^{A}
 *   \tag {3-3}   → \tag{3-3}
 * 这些空格在标准 LaTeX 中无语义差异, 但部分 KaTeX 版本可能解析异常。
 */

/** \left(...\right) 或 \left[...\right] 配对 */
const LEFT_RIGHT_PAIR = /\\left\s*[\(\\{].*\\right\s*[\)\\}]/s;

/** 任意 LaTeX 命令 (反斜杠 + 字母) */
const LATEX_CMD = /\\[a-zA-Z]/g;

/** 中文字符 (用于区分公式行与中文段落) */
const CJK_CHAR = /[一-鿿㐀-䶿]/g;

/**
 * 归一化 LaTeX 中的不规范空格:
 * - \command { → \command{  (命令名与花括号之间的空格)
 * - _ { → _{   (下标运算符与花括号之间的空格)
 * - ^ { → ^{   (上标运算符与花括号之间的空格)
 */
function normalizeLatexSpaces(latex: string): string {
  return latex
    .replace(/(\\[a-zA-Z]+)\s+\{/g, "$1{")
    .replace(/([_^])\s+\{/g, "$1{");
}

/**
 * 将裸 LaTeX 包裹在 $...$ / $$...$$ 定界符中,
 * 并对 LaTeX 内容做空格归一化。
 * 只做保守检测, 宁可漏掉也不误伤正文段落。
 */
export function wrapBareLatex(md: string): string {
  if (!md) return md;

  const lines = md.split("\n");
  const out: string[] = [];
  let inCodeBlock = false;
  let inMathBlock = false; // 已经在 $$...$$ 内

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    // ── 跟踪代码块 ──
    if (trimmed.startsWith("```")) {
      inCodeBlock = !inCodeBlock;
      out.push(line);
      continue;
    }
    if (inCodeBlock) {
      out.push(line);
      continue;
    }

    // ── 跟踪已有 $$ 块 ──
    if (trimmed === "$$") {
      inMathBlock = !inMathBlock;
      out.push(line);
      continue;
    }
    if (inMathBlock) {
      out.push(line);
      continue;
    }

    // ── 跳过已有 $ 定界符的行 ──
    if (line.includes("$")) {
      out.push(line);
      continue;
    }

    // ── 跳过 markdown 结构行 ──
    if (
      !trimmed ||
      /^#{1,6}\s/.test(trimmed) ||
      /^[>\-*+]\s/.test(trimmed) ||
      /^\d+\.\s/.test(trimmed) ||
      trimmed === "---" ||
      trimmed === "***"
    ) {
      out.push(line);
      continue;
    }

    // ── 检测裸 LaTeX ──
    // 归一化后的内容 (用于包裹后输出)
    const normalized = normalizeLatexSpaces(trimmed);

    // 1) \tag{...} → 一定是独立公式
    if (/\\tag\s*\{/.test(normalized)) {
      out.push("$$");
      out.push(normalized);
      out.push("$$");
      continue;
    }

    // 2) \begin{equation/align/...} 块
    if (/\\begin\s*\{(?:equation|align|gather|multline|cases|split|aligned)\b/.test(normalized)) {
      // 收集到 \end{...}
      out.push("$$");
      out.push(normalized);
      let j = i + 1;
      for (; j < lines.length; j++) {
        const innerTrimmed = normalizeLatexSpaces(lines[j].trim());
        out.push(innerTrimmed);
        if (/\\end\s*\{(?:equation|align|gather|multline|cases|split|aligned)\b/.test(innerTrimmed)) {
          break;
        }
      }
      out.push("$$");
      i = j; // 跳过已处理的行
      continue;
    }

    // 3) 含 \left(...\right) 且不像中文段落 → 独立公式
    if (LEFT_RIGHT_PAIR.test(normalized)) {
      const cjkCount = (normalized.match(CJK_CHAR) || []).length;
      const cmdCount = (normalized.match(LATEX_CMD) || []).length;
      // 如果中文字符远多于 LaTeX 命令, 说明是中文段落里偶然含 \left...\right
      if (cmdCount >= 2 && cjkCount <= cmdCount) {
        out.push("$$");
        out.push(normalized);
        out.push("$$");
        continue;
      }
    }

    // 4) 含多个 LaTeX 命令且几乎没有中文 → 大概率是独立公式
    {
      const cmdCount = (normalized.match(LATEX_CMD) || []).length;
      const cjkCount = (normalized.match(CJK_CHAR) || []).length;
      if (cmdCount >= 3 && cjkCount <= 1) {
        out.push("$$");
        out.push(normalized);
        out.push("$$");
        continue;
      }
    }

    out.push(line);
  }

  return out.join("\n");
}
