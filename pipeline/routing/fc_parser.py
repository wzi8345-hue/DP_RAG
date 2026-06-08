"""Tool call 响应解析器: 兼容 OpenAI / vLLM / Qwen 系几种常见格式。

支持的 LLM 响应形态:

1. OpenAI 标准: response.choices[0].message.tool_calls 数组
     [{"id":"...","type":"function","function":{"name":"plan","arguments":"{...}"}}]
   兼容: GPUGeek / DeepSeek / Aliyun DashScope / OpenAI 等。

2. vLLM Qwen 文本格式: message.content 含 <tool_call>...</tool_call> 块
     <tool_call>
     {"name":"plan","arguments":{...}}
     </tool_call>
   兼容: vLLM 部署的 Qwen2.5/Qwen3 系列, 部分 Llama tool 模型。

3. vLLM Hermes 风格: 在 message.content 里直接是 ```json ...``` 包裹的工具调用 JSON。
   作为兜底分支处理。

解析失败时返回空列表, 由上游 (RoutingCore) 决定走 json_schema 二次兜底。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 单条工具调用的归一化结构
# ---------------------------------------------------------------------------

class ParsedToolCall:
    """统一的工具调用表示: name + arguments (dict)。"""

    __slots__ = ("name", "arguments", "raw")

    def __init__(self, name: str, arguments: Dict[str, Any], raw: Any = None) -> None:
        self.name = name
        self.arguments = arguments
        self.raw = raw

    def __repr__(self) -> str:
        return f"ParsedToolCall(name={self.name!r}, arguments={self.arguments!r})"


# ---------------------------------------------------------------------------
# 正则: vLLM Qwen <tool_call> 块
# ---------------------------------------------------------------------------

_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", re.IGNORECASE,
)
# 兼容仅有开标签未闭合的情形 (vLLM 偶尔截断)
_TOOL_CALL_OPEN_RE = re.compile(
    r"<tool_call>\s*(\{[\s\S]*\})\s*$", re.IGNORECASE,
)
# 兜底: ```json {...} ``` 围栏
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
# 兜底: 裸 {...} 顶层 JSON
_BARE_JSON_RE = re.compile(r"(\{[\s\S]*\})", re.IGNORECASE)
# 兜底: 仅有 </tool_call> 闭合 (开标签被截掉)
_TOOL_CALL_CLOSE_RE = re.compile(
    r"(\{[\s\S]*?\})\s*</tool_call>", re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# OpenAI 标准格式
# ---------------------------------------------------------------------------

def _parse_openai_tool_calls(message: Dict[str, Any]) -> List[ParsedToolCall]:
    """从 OpenAI message.tool_calls 解析。返回空列表表示 message 没有 tool_calls。"""
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return []

    out: List[ParsedToolCall] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        name = fn.get("name") if isinstance(fn, dict) else None
        if not name:
            continue
        args_raw = fn.get("arguments") if isinstance(fn, dict) else None
        args: Dict[str, Any] = {}
        if isinstance(args_raw, dict):
            args = args_raw
        elif isinstance(args_raw, str):
            try:
                args = json.loads(args_raw) if args_raw.strip() else {}
            except json.JSONDecodeError as e:
                logger.warning(
                    f"[fc_parser] tool_call arguments JSON 解析失败 (name={name}): {e}; "
                    f"raw 前 200 字={args_raw[:200]!r}"
                )
                args = {}
        out.append(ParsedToolCall(name=name, arguments=args, raw=tc))
    return out


# ---------------------------------------------------------------------------
# vLLM Qwen <tool_call> 文本格式
# ---------------------------------------------------------------------------

def _parse_vllm_tool_call_blocks(content: str) -> List[ParsedToolCall]:
    """从 <tool_call>...</tool_call> 文本块解析。返回空列表表示没找到任何 tool_call 块。"""
    if not content:
        return []

    out: List[ParsedToolCall] = []
    # 优先匹配完整闭合的 block
    for m in _TOOL_CALL_BLOCK_RE.finditer(content):
        blob = m.group(1).strip()
        parsed = _safe_load_tool_dict(blob)
        if parsed:
            out.append(parsed)

    if out:
        return out

    # 兜底 1: 仅有 </tool_call> 闭合 (开标签可能在思考链里被截)
    m = _TOOL_CALL_CLOSE_RE.search(content)
    if m:
        blob = m.group(1).strip()
        parsed = _safe_load_tool_dict(blob)
        if parsed:
            return [parsed]

    # 兜底 2: 仅有 <tool_call> 未闭合
    m = _TOOL_CALL_OPEN_RE.search(content)
    if m:
        blob = m.group(1).strip()
        parsed = _safe_load_tool_dict(blob)
        if parsed:
            return [parsed]

    return []


def _parse_json_fence_blocks(content: str) -> List[ParsedToolCall]:
    """从 ```json {...} ``` 围栏解析 (Hermes / 通用兜底)。"""
    if not content:
        return []
    out: List[ParsedToolCall] = []
    for m in _JSON_FENCE_RE.finditer(content):
        blob = m.group(1).strip()
        parsed = _safe_load_tool_dict(blob)
        if parsed:
            out.append(parsed)
    return out


def _parse_bare_json(content: str) -> List[ParsedToolCall]:
    """从顶层裸 {...} JSON 解析 (最后兜底)。仅返回 1 个工具调用。"""
    if not content:
        return []
    m = _BARE_JSON_RE.search(content)
    if not m:
        return []
    parsed = _safe_load_tool_dict(m.group(1).strip())
    return [parsed] if parsed else []


def _safe_load_tool_dict(blob: str) -> Optional[ParsedToolCall]:
    """把 '{"name":"plan","arguments":{...}}' 字符串解析为 ParsedToolCall。"""
    try:
        d = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict):
        return None
    name = d.get("name")
    if not isinstance(name, str) or not name:
        return None
    args = d.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return ParsedToolCall(name=name, arguments=args, raw=d)


# ---------------------------------------------------------------------------
# 顶层 API: 一次性尝试所有解析路径
# ---------------------------------------------------------------------------

def parse_tool_calls(
    response: Dict[str, Any],
) -> Tuple[List[ParsedToolCall], str]:
    """解析 LLM chat completion 响应里的 tool_calls。

    入参:
        response: 调用 LLMClient.chat_with_tools 返回的字典, 期望包含 'raw' 字段
            (OpenAI 完整响应) 或 'message' 字段 (单条 message dict)。
            也接受裸的 OpenAI 响应 (含 choices)。

    返回:
        (tool_calls, source): tool_calls 是 ParsedToolCall 列表; source 是解析来源标识,
        用于日志/指标 ("openai_tool_calls" / "vllm_tool_call_block" / "json_fence" /
        "bare_json" / "empty")。
    """
    # 归一到 message dict
    message = _extract_message(response)
    if not message:
        return [], "empty"

    # 1. OpenAI 标准: message.tool_calls
    parsed = _parse_openai_tool_calls(message)
    if parsed:
        return parsed, "openai_tool_calls"

    content = message.get("content") or ""
    if isinstance(content, list):
        # 部分后端 content 是分段数组
        content = "".join(
            (c.get("text") or c.get("content") or "")
            for c in content
            if isinstance(c, dict)
        )
    if not isinstance(content, str):
        content = str(content) if content else ""

    # 2. vLLM Qwen <tool_call> 文本块
    parsed = _parse_vllm_tool_call_blocks(content)
    if parsed:
        return parsed, "vllm_tool_call_block"

    # 3. ```json``` 围栏
    parsed = _parse_json_fence_blocks(content)
    if parsed:
        return parsed, "json_fence"

    # 4. 裸 JSON 兜底
    parsed = _parse_bare_json(content)
    if parsed:
        return parsed, "bare_json"

    return [], "empty"


def _extract_message(response: Any) -> Optional[Dict[str, Any]]:
    """从多种响应形态里抽出 OpenAI message dict。"""
    if not isinstance(response, dict):
        return None

    # 已经是 message 形态
    if "tool_calls" in response or "role" in response:
        return response

    # LLMClient 包装的 {"answer": ..., "raw": <openai response>, "usage": ...}
    raw = response.get("raw")
    if isinstance(raw, dict):
        choices = raw.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message")
                if isinstance(msg, dict):
                    return msg
                delta = first.get("delta")
                if isinstance(delta, dict):
                    return delta

    # 裸 OpenAI 响应
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message")
            if isinstance(msg, dict):
                return msg

    # 退化形态: 只有 answer 字符串, 当作 content 看待
    answer = response.get("answer")
    if isinstance(answer, str):
        return {"content": answer}

    return None
