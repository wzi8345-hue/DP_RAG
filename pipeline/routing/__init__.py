"""pipeline.routing: Function Calling 驱动的 RAG 路由 + 反思模块 (v4)。

设计目标:
- ≤2 次 LLM 调用完成单 query 的"路由 → 检索 → 反思 → 重试"全流程
- 复杂场景 (泛 query / 上下文回指 / 复合查询) 全部在单次 FC 内完成
- 与现有 QueryRouter / RouteDecision / _heuristic_fallback_decision 完全兼容
- 不引入 HyDE (文献领域风险高于收益)

核心类: RoutingCore
便利工厂: build_routing_core_from_query_router (复用现有 QueryRouter 依赖)

公共类型:
- RouteOutcome = Union[RouteDecision, MultiRouteDecision, ClarifyRequest]
- ReflectVerdict (含 needs_retry, decision, partial, cause, meta)

详见同目录下的 INTEGRATION.md 关于如何接入现有 LangGraph 节点。
"""

from __future__ import annotations

from .core import (
    ReflectVerdict,
    RoutingCore,
    build_routing_core_from_query_router,
)
from .decision_builder import (
    ClarifyRequest,
    MultiRouteDecision,
    REUSE_MODES,
    REUSE_STANDALONE_MODES,
    ReuseRequest,
    RouteOutcome,
    SubqueryDecision,
)
from .limits import (
    DEFAULT_ROUTING_LIMITS,
    RoutingLimits,
    compound_intent_hint,
    estimate_compound_intents,
    normalize_routes,
    paths_should_split_to_multi,
    split_plan_args_to_multi_args,
)
from .fc_schema import (
    ASK_TOOL,
    MULTI_TOOL,
    OK_TOOL,
    PARTIAL_TOOL,
    PLAN_TOOL,
    RETRY_TOOL,
    REFLECT_TOOL_NAMES,
    REUSE_TOOL,
    ROUTER_TOOL_NAMES,
    TOOL_ASK,
    TOOL_MULTI,
    TOOL_OK,
    TOOL_PARTIAL,
    TOOL_PLAN,
    TOOL_RETRY,
    TOOL_REUSE,
    build_multi_tool,
    build_plan_tool,
    build_retry_tool,
    reflect_tools,
    router_tools,
)

__all__ = [
    # 主类
    "RoutingCore",
    "ReflectVerdict",
    "build_routing_core_from_query_router",
    # 决策类型
    "RouteOutcome",
    "MultiRouteDecision",
    "SubqueryDecision",
    "ClarifyRequest",
    "ReuseRequest",
    "REUSE_MODES",
    "REUSE_STANDALONE_MODES",
    # 路由容量
    "RoutingLimits",
    "DEFAULT_ROUTING_LIMITS",
    "normalize_routes",
    "estimate_compound_intents",
    "compound_intent_hint",
    "paths_should_split_to_multi",
    "split_plan_args_to_multi_args",
    # 工具 schema (供测试或独立调用)
    "PLAN_TOOL", "MULTI_TOOL", "ASK_TOOL", "REUSE_TOOL",
    "OK_TOOL", "RETRY_TOOL", "PARTIAL_TOOL",
    "build_plan_tool", "build_multi_tool", "build_retry_tool",
    "router_tools", "reflect_tools",
    # 工具名常量
    "TOOL_PLAN", "TOOL_MULTI", "TOOL_ASK", "TOOL_REUSE",
    "TOOL_OK", "TOOL_RETRY", "TOOL_PARTIAL",
    "ROUTER_TOOL_NAMES", "REFLECT_TOOL_NAMES",
    # 专业研究模式 (新增, 与现有路由隔离)
    "ResearchPlan", "ResearchFacet", "PolicyDecision",
    "plan_research", "decide_policy", "batches_to_multi_decision",
]

# 专业研究模式 (professional): 独立子模块, 仅在被显式 import 时生效, 对现有路由零影响
from .research import (  # noqa: E402
    PolicyDecision,
    ResearchFacet,
    ResearchPlan,
    batches_to_multi_decision,
    decide_policy,
    plan_research,
)
