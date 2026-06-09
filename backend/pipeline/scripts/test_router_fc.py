"""Router FC 调试: 直接打印模型原始输出, 观察 think 和 tool_calls 在哪个字段。

用法:
    cd <project_root>
    python -m pipeline.scripts.test_router_fc "有没有腐蚀钢相关的文献"
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pipeline.clients.llm import LLMClient
from pipeline.routing.fc_schema import router_tools
from pipeline.routing.prompts import render_router_system_fc

query = sys.argv[1] if len(sys.argv) > 1 else "有没有腐蚀钢相关的文献资料"

llm = LLMClient(
    api_base=os.environ.get("ROUTER_API_BASE", "http://localhost:8000/v1"),
    model=os.environ.get("ROUTER_MODEL", "/models/Qwen3.5-9B"),
    api_key=os.environ.get("ROUTER_API_KEY", "EMPTY"),
    timeout=60,
    max_retries=1,
)

system_prompt = render_router_system_fc(2026)
user_msg = f"用户问题: {query}\n\n请通过 function calling 调用 plan / multi / ask 之一完成路由。"

response = llm.chat_with_tools(
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ],
    tools=router_tools(enable_multi=True, enable_ask=True),
    tool_choice="required",
    temperature=0.0,
    max_tokens=1000,
    disable_thinking=False,
    parallel_tool_calls=False,
)

raw = response.get("raw", {})
message = raw.get("choices", [{}])[0].get("message", {})

print("=== reasoning (思考) ===")
print(message.get("reasoning") or message.get("reasoning_content") or "(空)")

print("\n=== content (正文) ===")
content = message.get("content", "")
print(content or "(空)")

print("\n=== tool_calls ===")
tool_calls = message.get("tool_calls", [])
if tool_calls:
    print(json.dumps(tool_calls, ensure_ascii=False, indent=2))
else:
    print("(空 — 如果 content 里有 Ǥ...gios 或 ```json``` 块, 说明 vLLM 没解析出来)")
    # 尝试从 content 手动提取
    from pipeline.routing.fc_parser import parse_tool_calls
    parsed, source = parse_tool_calls(response)
    if parsed:
        print(f"\nfc_parser 从 content 解析到 (source={source}):")
        for call in parsed:
            print(f"  name: {call.name}")
            print(f"  arguments: {json.dumps(call.arguments, ensure_ascii=False, indent=4)}")
    else:
        print("fc_parser 也未解析到任何 tool call")

print("\n=== finish_reason ===")
print(raw.get("choices", [{}])[0].get("finish_reason", ""))

print("\n=== usage ===")
print(json.dumps(raw.get("usage", {}), indent=2))
