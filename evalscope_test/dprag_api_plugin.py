"""evalscope perf 自定义 API 插件: 适配 DP-RAG 查询/对话接口。

支持两种压测模式 (由 Arguments.stream 控制):

1. **流式 (默认, 对齐前端 ChatView)**
   - POST /api/v1/chat/stream (SSE)
   - **TTFT (首 token 时延)**: 收到首个 ``{"type":"text","content":...}`` 且 content 非空
     的时刻 (与界面上答案文字开始展示一致; ``thinking``/``status`` 不计入)。
   - **Latency (端到端)**: ``done`` 事件的 ``latency_s`` (与服务端统计一致), 缺失时用墙钟。

2. **非流式 (兼容旧行为)**
   - POST /api/v1/query (普通 JSON)
   - TTFT ≈ 端到端时延 (无流式首包可区分)

通过 @register_api("dprag") 注册, 压测时传 api="dprag" 即可。
"""

from __future__ import annotations

import json
import re
import sys
import time
import traceback
from typing import Any, Dict, List, Tuple, Union

import aiohttp

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.default_api import DefaultApiPlugin, StreamedResponseHandler
from evalscope.perf.plugin.registry import register_api
from evalscope.perf.utils.benchmark_util import BenchmarkData
from evalscope.utils.logger import get_logger

logger = get_logger()

# 这些键从 extra_args 透传到查询请求体, 控制 RAG 的检索/生成行为。
_PASSTHROUGH_KEYS = ("use_agentic", "professional", "top_k", "mode", "collection")

_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_WORD = re.compile(r"[A-Za-z0-9]+")


def _estimate_tokens(text: str) -> int:
    """粗略 token 估算 (无需下载分词器): 中文按字数 + 英文/数字按词数。"""
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    words = len(_WORD.findall(text))
    return cjk + words


def _extract_prompt(messages: Union[List[Dict], str]) -> str:
    """从 dataset 产出的 prompt (字符串) 或 chat messages 列表中取出用户问题文本。"""
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list):
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content") or "")
        return " ".join(
            str(m.get("content") or "") for m in messages if isinstance(m, dict)
        )
    return str(messages)


def _parse_sse_payload(message: str) -> Any | None:
    """从 SSE message 中解析 JSON payload; 跳过注释/ping。"""
    message = message.strip()
    if not message or message.startswith(":"):
        return None
    chunk = message.removeprefix("data:").strip()
    if not chunk or chunk == "[DONE]":
        return None
    try:
        return json.loads(chunk)
    except json.JSONDecodeError:
        return None


@register_api("dprag")
class DPRagApiPlugin(DefaultApiPlugin):
    """DP-RAG /api/v1/query 或 /api/v1/chat/stream 压测插件。"""

    def build_request(
        self, messages: Union[List[Dict], str], param: Arguments = None
    ) -> Dict:
        param = param or self.param
        prompt = _extract_prompt(messages)

        body: Dict[str, Any] = {"query": prompt}
        extra = param.extra_args or {}
        for k in _PASSTHROUGH_KEYS:
            if k in extra and extra[k] is not None:
                body[k] = extra[k]
        return body

    async def process_request(
        self, client_session: aiohttp.ClientSession, url: str, headers: Dict, body: Dict
    ) -> BenchmarkData:
        """流式走 DP-RAG SSE 解析 (真实 TTFT); 非流式委托 DefaultApiPlugin。"""
        if not self.param.stream:
            return await super().process_request(client_session, url, headers, body)
        return await self._process_stream_request(client_session, url, headers, body)

    async def _process_stream_request(
        self, client_session: aiohttp.ClientSession, url: str, headers: Dict, body: Dict
    ) -> BenchmarkData:
        """处理 /chat/stream SSE: TTFT = 首个 type=text 非空 content (对齐前端展示)。"""
        headers = {"Content-Type": "application/json", **headers}
        data = json.dumps(body, ensure_ascii=False)

        output = BenchmarkData()
        st = time.perf_counter()
        output.start_time = st
        output.request = data

        ttft = 0.0
        generated_text = ""
        most_recent_timestamp = st
        done_event: Dict[str, Any] | None = None

        try:
            async with client_session.post(url=url, data=data, headers=headers) as response:
                content_type = response.headers.get("Content-Type", "")
                if response.status != 200:
                    output.status_code = response.status
                    try:
                        err_payload = await response.json()
                        output.error = json.dumps(err_payload, ensure_ascii=False)
                    except Exception:
                        try:
                            output.error = await response.text()
                        except Exception:
                            output.error = response.reason or ""
                    output.success = False
                    return output

                if "text/event-stream" not in content_type:
                    # 服务端未返回 SSE: 回退非流式解析
                    return await super().process_request(
                        client_session, url, headers, body
                    )

                handler = StreamedResponseHandler()
                async for chunk_bytes in response.content.iter_any():
                    if not chunk_bytes:
                        continue
                    for message in handler.add_chunk(chunk_bytes):
                        payload = _parse_sse_payload(message)
                        if not isinstance(payload, dict):
                            continue

                        timestamp = time.perf_counter()
                        output.response_messages.append(payload)
                        ev_type = payload.get("type")

                        if ev_type == "text":
                            content = str(payload.get("content") or "")
                            if content:
                                if ttft == 0.0:
                                    ttft = timestamp - st
                                    output.first_chunk_latency = ttft
                                else:
                                    output.inter_chunk_latency.append(
                                        timestamp - most_recent_timestamp
                                    )
                                generated_text += content
                                most_recent_timestamp = timestamp

                        elif ev_type == "done":
                            done_event = payload
                            most_recent_timestamp = timestamp

                        elif ev_type == "error":
                            output.error = str(payload.get("message") or "unknown error")
                            output.success = False
                            most_recent_timestamp = timestamp

                output.generated_text = generated_text
                if done_event and not generated_text:
                    generated_text = str(done_event.get("answer") or "")
                    output.generated_text = generated_text

                if done_event and done_event.get("latency_s") is not None:
                    try:
                        output.query_latency = float(done_event["latency_s"])
                    except (TypeError, ValueError):
                        output.query_latency = most_recent_timestamp - st
                else:
                    output.query_latency = most_recent_timestamp - st

                # 无 text 事件 (如仅 done 带 answer): TTFT 退化为端到端
                if ttft == 0.0 and output.query_latency > 0:
                    output.first_chunk_latency = output.query_latency

                output.completed_time = most_recent_timestamp
                output.success = output.error is None and bool(
                    generated_text or done_event
                )

        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
            logger.error(output.error)

        return output

    def parse_responses(
        self, responses: List[Any], request: str = None, **kwargs: Any
    ) -> Tuple[int, int]:
        """返回 (prompt_tokens, completion_tokens)。

        流式: 从 done 事件取 usage/answer; 非流式: 从最终 JSON 取。
        """
        try:
            done: Dict[str, Any] | None = None
            answer_parts: List[str] = []

            for item in responses or []:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "done":
                    done = item
                elif item.get("type") == "text":
                    answer_parts.append(str(item.get("content") or ""))

            if done is not None:
                usage = done.get("usage")
                if isinstance(usage, dict) and usage:
                    p = usage.get("prompt_tokens")
                    c = usage.get("completion_tokens")
                    if p is not None and c is not None:
                        return int(p), int(c)
                answer_text = str(done.get("answer") or "".join(answer_parts))
            else:
                # 非流式: responses[-1] 是完整 JSON 响应
                last = responses[-1] if responses else {}
                if isinstance(last, str):
                    try:
                        last = json.loads(last)
                    except Exception:
                        last = {}
                if not isinstance(last, dict):
                    last = {}
                usage = last.get("usage")
                if isinstance(usage, dict) and usage:
                    p = usage.get("prompt_tokens")
                    c = usage.get("completion_tokens")
                    if p is not None and c is not None:
                        return int(p), int(c)
                answer_text = str(last.get("answer") or "")

            prompt_text = ""
            if request:
                try:
                    req = json.loads(request) if isinstance(request, str) else request
                    if isinstance(req, dict):
                        prompt_text = str(req.get("query") or "")
                except Exception:
                    pass
            return _estimate_tokens(prompt_text), _estimate_tokens(answer_text)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[dprag] parse_responses 出错: {e}")
            return 0, 0
