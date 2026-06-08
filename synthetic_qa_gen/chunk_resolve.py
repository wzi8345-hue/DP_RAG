"""knowledge block id 与原文互转（后处理用）。"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

BLOCK_ID_RE = re.compile(
    r"^(title|text|table|image|summary)_[a-z0-9]+(?:_p\d+)?$",
    re.IGNORECASE,
)


def is_block_id(value: str) -> bool:
    return bool(BLOCK_ID_RE.match((value or "").strip()))


def normalize_block_id_ref(ref: str) -> str:
    """从 LLM 返回值中提取 block id（兼容 `text_xxx`、id=text_xxx 等写法）。"""
    ref = (ref or "").strip().strip("`\"'")
    if is_block_id(ref):
        return ref.lower()
    match = re.search(
        r"\b((?:title|text|table|image|summary)_[a-z0-9]+(?:_p\d+)?)\b",
        ref,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).lower()
    return ref


def extract_chunk_text(chunk: Dict[str, Any]) -> str:
    """提取单个 knowledge block 的可读正文。"""
    ctype = chunk.get("type", "")
    content = (chunk.get("content") or "").strip()
    ctx = (chunk.get("context") or "").strip()

    if ctype == "image" and "[Image Path]" in content:
        caption_match = re.search(
            r"\[Caption\]\s*(.*?)(?:\n|\[Image Path\])",
            content,
            re.DOTALL,
        )
        if caption_match:
            content = f"[图片说明] {caption_match.group(1).strip()}"
        else:
            return ""

    if ctype == "table" and content:
        content = _simplify_table_content(content)

    parts: List[str] = []
    if ctx and ctx not in content:
        parts.append(ctx)
    if content:
        parts.append(content)
    return "\n".join(parts).strip()


def _simplify_table_content(content: str) -> str:
    """表格块保留 caption 与 HTML 表格，去掉图片路径行。"""
    lines = []
    for line in content.splitlines():
        if line.strip().startswith("[Table Image Path]"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def build_block_index(chunks: List[Dict[str, Any]]) -> Dict[str, str]:
    """id -> 块正文；长段切分子块会按 parent_chunk_id 合并为父 id。"""
    index: Dict[str, str] = {}
    children_by_parent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for chunk in chunks:
        block_id = chunk.get("id")
        if not block_id:
            continue
        text = extract_chunk_text(chunk)
        if text:
            index[block_id] = text

        parent_id = chunk.get("parent_chunk_id")
        if parent_id:
            children_by_parent[parent_id].append(chunk)

    for parent_id, children in children_by_parent.items():
        if parent_id in index:
            continue
        children.sort(key=lambda c: c.get("chunk_index", 0))
        merged = [extract_chunk_text(c) for c in children]
        merged = [t for t in merged if t]
        if merged:
            index[parent_id] = "\n".join(merged)

    return index


def format_chunk_for_prompt(chunk: Dict[str, Any]) -> Optional[str]:
    """带 block id 的块文本，供 LLM 引用 ground_contexts。"""
    block_id = chunk.get("id", "")
    text = extract_chunk_text(chunk)
    if not text:
        return None

    ctype = chunk.get("type", "")
    section = (chunk.get("section") or "").strip()
    ctx = (chunk.get("context") or "").strip()

    header_parts = [ctype, f"id={block_id}"]
    if section:
        header_parts.append(section)
    if ctx and ctx not in text:
        header_parts.append(ctx[:80] + ("..." if len(ctx) > 80 else ""))

    return f"[{' | '.join(header_parts)}]\n{text}"


def assemble_full_text(chunks: List[Dict[str, Any]]) -> str:
    """拼接文献全文，每块标注 id 便于 LLM 返回 block id。"""
    parts: List[str] = []
    for chunk in chunks:
        block = format_chunk_for_prompt(chunk)
        if block:
            parts.append(block)
    return "\n\n".join(parts)


def resolve_ground_contexts(
    contexts: List[Any],
    block_index: Dict[str, str],
) -> List[str]:
    """将 ground_contexts 中的 block id 后处理为对应原文。"""
    resolved: List[str] = []
    for item in contexts:
        if isinstance(item, dict):
            raw = str(item.get("chunk_id") or item.get("id") or "").strip()
        else:
            raw = str(item).strip()
        if not raw:
            continue

        block_id = normalize_block_id_ref(raw)
        if block_id in block_index:
            resolved.append(block_index[block_id])
            continue

        if is_block_id(block_id):
            logger.warning("未解析到 block id 对应原文: %s", block_id)
            continue

        resolved.append(raw)

    return resolved
