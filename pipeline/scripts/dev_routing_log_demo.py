"""开发者验证脚本: 直接 mock 一个 LLM, 跑通 FC 路由 & 反思的日志可视化。

用法:
    cd /Users/dp/Desktop/工作文件/DP_rag_skill
    python -m pipeline.scripts.dev_routing_log_demo

期望输出 (节选, INFO 级):
    [routing.route] begin: query='帕金森的诊断要点' history_msgs=0 doc_registry=2 ...
    [routing.route] FC call: model=mock-llm tools=['plan', 'multi'] parallel=False ...
    [routing.route] FC response: source=openai_tool_calls tool_calls=1 llm_ms=0 usage={}
    [routing.route] FC tool selected: name=plan args_keys=['paths', ...]
    [routing.route] DONE: type=RouteDecision chain=fc tool=plan conf=0.85 ...
    [routing.reflect] begin: total_hits=3 retry=0/2 ...
    [routing.reflect] DONE: needs_retry=False (ok) reason='覆盖足够' ...

这个脚本不会真的调任何远端 LLM, 完全离线, 适合开发期验证日志结构。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("dev_routing_log_demo")

from pipeline.routing.core import RoutingCore  # noqa: E402


class MockLLM:
    """最小化 mock, 模拟 OpenAI compatible chat_with_tools 的返回结构。

    模拟 vLLM 启用 reasoning_parser 时的 message.reasoning_content (隐式 CoT)。
    """

    model = "mock-llm"

    def __init__(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        reasoning_content: str = "",
    ):
        self.tool_name = tool_name
        self.arguments = arguments
        self.reasoning_content = reasoning_content

    def chat_with_tools(self, **kwargs) -> Dict[str, Any]:  # noqa: D401
        return {
            "answer": "",
            "raw": {},
            "usage": {"prompt_tokens": 350, "completion_tokens": 22, "total_tokens": 372},
            "tool_calls": [
                {
                    "id": "call_001",
                    "type": "function",
                    "function": {
                        "name": self.tool_name,
                        "arguments": json.dumps(self.arguments, ensure_ascii=False),
                    },
                }
            ],
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [],
                "reasoning_content": self.reasoning_content,
            },
            "reasoning_content": self.reasoning_content,
        }

    def chat(self, *a, **k):  # 不需要 chat 方法, 但接口完整
        raise NotImplementedError


def fake_validate(
    raw: Dict[str, Any], _raw_text: str, query: str, *, doc_registry=None,
):
    """跳过完整 QueryRouter._validate_decision, 直接信任 raw 字段。

    真实签名: validate_fn(raw_dict, raw_text, query, doc_registry=doc_registry) -> RouteDecision
    """
    from pipeline.models import RouteDecision

    # 把 rewrites 里非 str 的值统一转字符串 (RouteDecision.rewrites: Dict[str, str])
    rewrites_in = raw.get("rewrites") or {}
    rewrites: Dict[str, str] = {}
    for k, v in rewrites_in.items():
        if isinstance(v, list):
            v = " ".join(str(x) for x in v)
        rewrites[str(k)] = str(v)

    kwargs = dict(
        routes=raw.get("routes") or ["progressive"],
        rewrites=rewrites,
        chunk_type=raw.get("chunk_type"),
        target_docs=raw.get("target_docs", []) or [],
        fig_refs=raw.get("fig_refs", []) or [],
        table_refs=raw.get("table_refs", []) or [],
        page_refs=raw.get("page_refs", []) or [],
        paragraph_refs=raw.get("paragraph_refs", []) or [],
        entities=raw.get("entities", []) or [],
        reasoning=raw.get("reasoning", "") or "(fake-validate)",
    )
    t = raw.get("time")
    if t:
        kwargs["time"] = t
    return RouteDecision(**kwargs)


def fake_heuristic(query: str, year: int):
    from pipeline.models import RouteDecision
    return RouteDecision(
        routes=["progressive"],
        rewrites={"progressive": query},
        reasoning=f"(mock-heuristic year={year})",
    )


def demo_simple_plan() -> None:
    print("\n" + "=" * 70)
    print("场景 1: 简单 plan (route='local', 单文献)")
    print("=" * 70)

    plan_args = {
        "paths": [
            {
                "t": "local",
                "kw": ["帕金森", "诊断", "临床要点"],
                "docs": ["神经病学指南2023"],
                "ctype": "narrative",
            }
        ],
    }

    thinking = (
        "用户问的是'文献 1 里...具体临床要点', 锁定了具体文献 (神经病学指南2023). "
        "按决策表#2命中 local. 不是 metadata (没说图/表/页). 不是 summary/progressive. "
        "kw 提取出诊断/临床要点两个名词性概念, conf=0.9."
    )
    core = RoutingCore(
        router_llm=MockLLM("plan", plan_args, reasoning_content=thinking),
        reflect_llm=None,
        validate_fn=fake_validate,
        heuristic_fn=fake_heuristic,
        enable_multi=True,
        enable_ask=False,
        router_disable_thinking=False,   # 演示: 显式开启 router 思考
    )

    outcome = core.route(
        query="文献 1 里帕金森的诊断要点是什么",
        history=[
            {"role": "user", "content": "帕金森是什么病"},
            {"role": "assistant", "content": "帕金森是一种神经退行性病..."},
        ],
        doc_registry=[
            {"id": "1", "title": "神经病学指南2023"},
            {"id": "2", "title": "运动障碍综述2022"},
        ],
        correlation_id="demo-001",
    )
    print(f"\n→ 决策类型: {type(outcome).__name__}, routes={outcome.routes}")


def demo_multi() -> None:
    print("\n" + "=" * 70)
    print("场景 2: 复合查询 multi (2 个独立子查询)")
    print("=" * 70)

    multi_args = {
        "subs": [
            {
                "id": "s1",
                "q": "AlphaFold 训练目标函数",
                "paths": [
                    {
                        "t": "local",
                        "kw": ["AlphaFold", "损失函数", "训练目标"],
                        "docs": ["AlphaFold paper 2021"],
                    }
                ],
            },
            {
                "id": "s2",
                "q": "OpenFold 与 AlphaFold 区别",
                "paths": [
                    {"t": "summary", "kw": ["OpenFold", "AlphaFold", "对比"]},
                    {"t": "progressive", "kw": ["OpenFold", "改进", "AlphaFold"]},
                ],
            },
        ],
        "synth": "先按子查询分别检索, 最后对比 AlphaFold 与 OpenFold 的训练目标差异",
    }

    core = RoutingCore(
        router_llm=MockLLM("multi", multi_args),
        reflect_llm=None,
        validate_fn=fake_validate,
        heuristic_fn=fake_heuristic,
        enable_multi=True,
    )
    outcome = core.route(
        query="AlphaFold 的训练目标是什么, 跟 OpenFold 有什么区别",
        correlation_id="demo-002",
    )
    print(f"\n→ 决策类型: {type(outcome).__name__}")


def demo_reflect_ok() -> None:
    print("\n" + "=" * 70)
    print("场景 3: 反思 ok (结果充分, 不重试)")
    print("=" * 70)

    core = RoutingCore(
        router_llm=None,
        reflect_llm=MockLLM(
            "ok", {},
            reasoning_content="覆盖度: 3 条 hit 都对准帕金森诊断要点; 相关度高; 数量充分. 选 ok.",
        ),
        validate_fn=fake_validate,
        heuristic_fn=fake_heuristic,
    )
    from pipeline.models import RouteDecision
    last = RouteDecision(routes=["local"], rewrites={"local": "test"}, reasoning="prev")
    verdict = core.reflect(
        query="问题",
        last_decision=last,
        results_summary="[local]\n- hit 1: 帕金森诊断...\n- hit 2: 运动迟缓...\n- hit 3: ...",
        total_hits=3,
        retry_count=0,
        max_retries=2,
        correlation_id="demo-003",
    )
    print(f"\n→ verdict.needs_retry={verdict.needs_retry} partial={verdict.partial}")


def demo_reflect_retry() -> None:
    print("\n" + "=" * 70)
    print("场景 4: 反思 retry (改路径 + 加 keyword 重试)")
    print("=" * 70)

    retry_args = {
        "cause": "off",
        "plan": {
            "paths": [
                {"t": "progressive", "kw": ["帕金森", "运动症状", "详细", "震颤"]},
                {"t": "summary", "kw": ["帕金森", "运动症状", "概览"]},
            ],
        },
    }
    core = RoutingCore(
        router_llm=None,
        reflect_llm=MockLLM("retry", retry_args),
        validate_fn=fake_validate,
        heuristic_fn=fake_heuristic,
    )
    from pipeline.models import RouteDecision
    last = RouteDecision(routes=["local"], rewrites={"local": "原查询"}, reasoning="prev")
    verdict = core.reflect(
        query="帕金森的主要运动症状是什么",
        last_decision=last,
        results_summary="[local]\n- hit 1: 阿尔茨海默与帕金森的区别...\n- hit 2: 步态...",
        total_hits=2,
        retry_count=0,
        max_retries=2,
        correlation_id="demo-004",
    )
    print(f"\n→ verdict.needs_retry={verdict.needs_retry} cause={verdict.cause}")
    if verdict.decision:
        print(f"   new routes={verdict.decision.routes} rewrites={dict(verdict.decision.rewrites)}")


def demo_zero_hits_fast_path() -> None:
    print("\n" + "=" * 70)
    print("场景 5: 反思 0 命中快速路径 (不调 LLM)")
    print("=" * 70)

    core = RoutingCore(
        router_llm=None,
        reflect_llm=MockLLM("ok", {}),  # 不会被调用 (zero_hits 快速路径短路)
        validate_fn=fake_validate,
        heuristic_fn=fake_heuristic,
    )
    verdict = core.reflect(
        query="超冷门问题",
        last_decision=None,
        results_summary="",
        total_hits=0,
        retry_count=0,
        max_retries=2,
        correlation_id="demo-005",
    )
    print(f"\n→ verdict.needs_retry={verdict.needs_retry} cause={verdict.cause}")


if __name__ == "__main__":
    demo_simple_plan()
    demo_multi()
    demo_reflect_ok()
    demo_reflect_retry()
    demo_zero_hits_fast_path()
    print("\n" + "=" * 70)
    print("全部 demo 通过. 上面的 INFO 行就是接入 LangGraph 后会看到的运行日志。")
    print("=" * 70)
