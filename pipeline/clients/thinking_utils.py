"""从 LLM 原始输出中剥离思考块, 供 JSON / tool_call 解析前使用。

Qwen / DeepSeek 等推理模型可能输出:
  - `` (vLLM reasoning_parser)
  - `` (部分后端/截断场景)

支持三种边界情况 (与 agentic 历史行为对齐):
  1. 完整闭合块 — 整段删除
  2. 仅有闭合标签 (开头被 max_tokens 截断) — 保留闭合标签之后的内容
  3. 仅有开始标签 (未闭合) — 保留开始标签之前的内容
"""

from __future__ import annotations

import re

# 完整块: 支持 open/close 标签混用 (部分模型输出 <think>...</think>)
_COMPLETE_BLOCK_RES = (
    re.compile(
        r"<redacted_thinking\b[^>]*>[\s\S]*?</think>",
        re.IGNORECASE,
    ),
    re.compile(
        r"<think\b[^>]*>[\s\S]*?</think\s*>",
        re.IGNORECASE,
    ),
    re.compile(
        r"<think\b[^>]*>[\s\S]*?</think>",
        re.IGNORECASE,
    ),
    re.compile(
        r"<redacted_thinking\b[^>]*>[\s\S]*?</think\s*>",
        re.IGNORECASE,
    ),
)

_CLOSE_RES = (
    re.compile(r"</think>", re.IGNORECASE),
    re.compile(r"</think\s*>", re.IGNORECASE),
)

_OPEN_RES = (
    re.compile(r"<redacted_thinking\b[^>]*>", re.IGNORECASE),
    re.compile(r"<think\b[^>]*>", re.IGNORECASE),
)


def strip_think_blocks(text: str) -> str:
    """剥离思考块; 截断场景下尽量 salvage 后面的 JSON / tool_call。"""
    if not text:
        return text

    cleaned = text
    # 多轮替换, 处理嵌套或连续多块
    while True:
        prev = cleaned
        for pat in _COMPLETE_BLOCK_RES:
            cleaned = pat.sub("", cleaned)
        if cleaned == prev:
            break

    # 开头被截断: 第一个闭合标签之后才是有效 payload
    for close_pat in _CLOSE_RES:
        close_match = close_pat.search(cleaned)
        if close_match:
            cleaned = cleaned[close_match.end():]
            break

    # 末尾未闭合: 开始标签之前才是有效 payload
    for open_pat in _OPEN_RES:
        open_match = open_pat.search(cleaned)
        if open_match:
            cleaned = cleaned[:open_match.start()]
            break

    return cleaned.strip()
