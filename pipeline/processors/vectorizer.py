"""文本向量化处理: 将知识块通过 Embedding 模型转为向量。

从原始 chunk2vector.py 搬入, 逻辑完全保留。

包含:
- HTML 表格 -> 纯文本转换
- 不同 chunk 类型 -> 统一 embedding 文本拼接
- 批量向量化
"""

from __future__ import annotations

import html
import json
import logging
import re
from typing import Any, Dict, List, Optional

from ..clients.embedding import EmbeddingClient
from .chem_symbols import (
    clean_latex_numbers,
    composition_descriptor,
    gloss_cells,
    is_composition_row,
    split_cells,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------

DEFAULT_MAX_CHARS = 8000


# ---------------------------------------------------------------------------
# HTML / image-path 等 noise 提取
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_CELL_RE = re.compile(r"<t[hd][^>]*>(.*?)</t[hd]>", re.DOTALL | re.IGNORECASE)

# content 中 [Caption] / [Image Path] / [Table Image Path] / [Table HTML] 段提取
_CAPTION_LINE_RE = re.compile(r"^\[Caption\]\s*(.*)$", re.MULTILINE)
_TABLE_HTML_BLOCK_RE = re.compile(
    r"\[Table HTML\]\s*\n(.*?)(?=\n\[[A-Z][^\]]*\]|\Z)",
    re.DOTALL,
)


def html_table_to_text(html_str: str) -> str:
    """把 MinerU 给的 <table> HTML 转成对 embedding 友好的行文本。

    每行用 ' | ' 分隔, 行间用换行, 去除嵌套标签和空白。
    """
    if not html_str:
        return ""
    rows: List[str] = []
    for tr in _TR_RE.finditer(html_str):
        cells: List[str] = []
        for cell in _CELL_RE.finditer(tr.group(1)):
            inner = _HTML_TAG_RE.sub("", cell.group(1))
            inner = html.unescape(inner)
            inner = _WS_RE.sub(" ", inner).strip()
            cells.append(inner)
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _extract_caption_text(content: str) -> str:
    """从 image/table 的 content 中提取 [Caption] 一行的纯文本。"""
    m = _CAPTION_LINE_RE.search(content or "")
    if not m:
        return ""
    cap = m.group(1).strip()
    # 兼容 "[Image without caption]" / "[Table without caption]"
    if cap.startswith("[") and cap.endswith("]"):
        return ""
    return cap


def _extract_table_html(content: str) -> str:
    """从 table 的 content 中抽出 [Table HTML] 段的原始 HTML。"""
    m = _TABLE_HTML_BLOCK_RE.search(content or "")
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# 不同 chunk 类型 -> 统一 embedding 文本
# ---------------------------------------------------------------------------

def compose_embedding_text(
    chunk: Dict[str, Any], max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """根据 chunk 类型拼出最终用于 embedding 的文本。

    设计原则 (v3, 比 v2 更聚焦语义、去噪):
    1. 只保留对语义匹配真正有信号的字段, 避免污染向量主语义
       - 去掉 [Type] (与 schema 中 type 字段重复, 检索可用 filter)
       - 去掉 [Related] (relation label 不属于本 chunk 语义)
       - image/table 不再注入文件路径 (无语义信号且占字符)
    2. text/summary/title: [Section] + content
    3. table: [Section] + Caption + 纯文本行 (HTML 标签和图片路径不进 embedding)
    4. image: [Section] + Caption + footnote
       (不直接 embed 二进制图像, 由 Caption + 周边 context 承担语义)
    5. context (公式 / 表格 footnote 等) 仅对 image/table 保留, 对 text/summary
       不再单独段落化以避免向量被公式 LaTeX 噪声主导
    6. 末尾按 max_chars 截断 (仅作硬上限, 长 chunk 应在 chunker 阶段已被语义切分)
    """
    section = (chunk.get("section") or "").strip()
    chunk_type = (chunk.get("type") or "text").strip()
    content = chunk.get("content") or ""
    context = (chunk.get("context") or "").strip()

    parts: List[str] = []
    if section:
        parts.append(f"[Section] {section}")

    if chunk_type == "table":
        caption = _extract_caption_text(content)
        if caption:
            parts.append(f"[Caption] {caption}")
        table_text = clean_latex_numbers(html_table_to_text(_extract_table_html(content)))
        if table_text:
            # 化学成分表: 表头是元素符号 (C/Si/Mn...), 而问句用中文 (碳/硅) + "化学成分/含量"。
            # 注入中文注释 + 合成描述, 让 dense 与 BM25 都能挂上查询词 (索引侧修复检索短板)。
            rows = table_text.split("\n")
            header = split_cells(rows[0]) if rows else []
            if is_composition_row(header):
                parts.append(composition_descriptor(header))
                rows[0] = gloss_cells(header)
                table_text = "\n".join(rows)
            parts.append(table_text)
        if context:
            parts.append(context)
    elif chunk_type == "image":
        caption = _extract_caption_text(content)
        if caption:
            parts.append(f"[Caption] {caption}")
        if context:
            parts.append(context)
    elif chunk_type == "equation":
        # 公式: LaTeX + context anchor 句 (chunker 注入), 提升语义/BM25 双路命中.
        body = content.strip()
        if body:
            parts.append(body)
        if context:
            parts.append(context)
    elif chunk_type == "references":
        # 参考文献: 不加 [Section] (永远是 "References"), 也不附 context,
        # 直接 embedding 条目原文; 这样查作者/标题/年份的精确召回更准.
        # 注意覆盖前面已经 append 的 [Section] 行 (如果有的话).
        body = content.strip()
        parts = [body] if body else []
    else:
        # text / summary / title: 清掉 OCR 残留的 LaTeX 包裹 + 合并被拆开的数字
        # ("$\mathrm{ C } 0 . 1 6$" → "C 0.16"), 让数值类问句能召回到正文里的数据。
        body = clean_latex_numbers(content.strip())
        if body:
            parts.append(body)

    text = "\n\n".join(p for p in parts if p).strip()
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    return text


# ---------------------------------------------------------------------------
# 批量向量化
# ---------------------------------------------------------------------------

def vectorize_chunks(
    chunks: List[Dict[str, Any]],
    embedder: EmbeddingClient,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> List[Dict[str, Any]]:
    """对每个 chunk 生成 embedding_text 并向量化, 返回新的 chunk 列表。"""
    embed_texts = [compose_embedding_text(c, max_chars=max_chars) for c in chunks]
    vectors = embedder.embed_all(embed_texts)
    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"返回向量数 ({len(vectors)}) 与 chunk 数 ({len(chunks)}) 不一致"
        )

    out: List[Dict[str, Any]] = []
    for chunk, text, vec in zip(chunks, embed_texts, vectors):
        new_chunk = dict(chunk)
        new_chunk["embedding_text"] = text
        new_chunk["embedding"] = vec
        new_chunk["embedding_model"] = embedder.model
        new_chunk["embedding_dim"] = len(vec)
        out.append(new_chunk)
    return out
