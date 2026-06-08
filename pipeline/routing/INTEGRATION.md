# pipeline.routing 接入说明

本模块是 **可插拔的** FC 路由 + 反思组件, 默认**不影响**现有 `langgraph_agent.py` 行为。
本文档说明接入步骤 (后续 PR 再做)。

## 模块结构

```
pipeline/routing/
├── __init__.py          # 公共 API 出口
├── fc_schema.py         # 6 个工具的 OpenAI tools schema
├── fc_parser.py         # 兼容 OpenAI / vLLM <tool_call> / JSON 围栏 的解析器
├── decision_builder.py  # FC 输出 → RouteDecision / MultiRouteDecision / ClarifyRequest
├── core.py              # RoutingCore (route + reflect)
├── prompts/
│   ├── __init__.py             # 渲染器 (复用 prompts/router_rules.md)
│   ├── router_system_fc.md     # FC 版 router system prompt
│   └── reflect_system_fc.md    # FC 版 reflect system prompt
└── INTEGRATION.md       # 本文档
```

## 与现有架构的关系

| 既有 | 关系 | 改动 |
|---|---|---|
| `pipeline/retrieval/agentic.py:QueryRouter` | **完全复用** (`_validate_decision` 通过依赖注入) | 不动 |
| `pipeline/retrieval/agentic.py:_heuristic_fallback_decision` | **完全复用** (FC 全失败时兜底) | 不动 |
| `pipeline/models.py:RouteDecision` | **完全兼容** | 不动 |
| `pipeline/prompts/router_rules.md` | **完全复用** (FC 版 system prompt 内嵌该文件) | 不动 |
| `pipeline/prompts/reflect_system.md` | 不再使用 (FC 版替换为 `routing/prompts/reflect_system_fc.md`) | 不动 |
| `pipeline/clients/llm.py:LLMClient` | 新增 `chat_with_tools()` 方法 (向后兼容) | ✅ 已加 |
| `pipeline/retrieval/langgraph_agent.py` | 接入点; 见下文 | ✅ 已接入 (含 clarify / multi 子查询并行 / partial_note) |

## 调用契约

```python
from pipeline.routing import (
    RoutingCore, build_routing_core_from_query_router,
    RouteOutcome, MultiRouteDecision, ClarifyRequest,
    ReflectVerdict,
)
from pipeline.models import RouteDecision

# ── 构造 (推荐复用 QueryRouter) ──
core = build_routing_core_from_query_router(
    query_router=existing_query_router,
    reflect_llm=reflection_llm_client,
    enable_multi=True,         # 业务中有复合 query 就开
    enable_ask=False,          # 默认关; 体验受影响, 仅极泛 query 开
    disable_thinking=True,     # vLLM 后端
    parallel_tool_calls=False, # 单决策不需并行
    history_turns=1,
)

# ── 路由 ──
outcome: RouteOutcome = core.route(
    query="...",
    history=session.recent_messages(),
    doc_registry=last_round_docs,
)

if isinstance(outcome, RouteDecision):
    # 90% 场景: 与现有 router 完全等价的单一决策
    ...
elif isinstance(outcome, MultiRouteDecision):
    # 复合查询: 含 2-3 个 SubqueryDecision, 每个有独立 RouteDecision
    for sub in outcome.subqueries:
        sub.id, sub.decision   # RouteDecision
    outcome.synth_hint         # 上下文拼接提示
elif isinstance(outcome, ClarifyRequest):
    # 反问出口: 跳过本轮检索, 把 outcome.question / outcome.options 返回 user
    ...

# ── 反思 ──
verdict: ReflectVerdict = core.reflect(
    query="...",
    last_decision=outcome,
    results_summary=summary_text,  # 来自现有 _summarize_results
    total_hits=N,
    this_round_docs=this_round,
    retry_count=k,
    max_retries=m,
)

if verdict.needs_retry:
    new_outcome = verdict.decision    # RouteDecision 或 MultiRouteDecision
elif verdict.partial:
    # 把 verdict.partial_note 拼到 context 末尾, 让 LLM 知道信息有限
    ...
else:
    # ok: 直接进入 context_build
    ...
```

## LangGraph 接入现状 (已落地)

实际接入已完成; 与下方早期伪代码的差异:

- **复合查询不再退化**: `MultiRouteDecision` 写入 `state["subquery_decisions"]`,
  `retrieve_node` 对每个 sub 独立检索, 结果通过 `_merge_route_results()` 按
  `pk` 增量合并; 展示 `decision` 由 `_merge_route_decisions()` 合成, `synth_hint`
  追加到 `context` 末尾.
- **ClarifyRequest 有独立出口**: graph 增加 `clarify` 节点和 `_after_router`
  条件边 (`router → clarify | retrieve`), clarify 节点直接产出反问文本写入
  `state["clarify_answer"]` / `context`, 跳过 retrieve/reflect/生成 LLM.
- **Retry 增量合并**: `rewrite_node` 不再清空 `route_results`,
  `retrieve_node` 调用 `_merge_route_results()` 把新一轮命中并入前轮结果.
- **Reranker 过滤可恢复**: rerank 前快照写入 `state["route_results_pre_rerank"]`,
  低分场景保留完整集合给 reflect; reflect 判 OK 时 `context_build` 优先恢复快照.
- **`LangGraphAgent.answer()` 实现**: 单/多轮 × 流/非流共 4 个分支均接通生成 LLM,
  `clarify` 出口跳过生成直接返回反问.
- **`session_meta` 兜底**: `LangGraphAgent.run()` 新增可选 `session_id` 参数,
  缺失 `session_meta["doc_registry"]` 时从进程内 LRU 恢复; clarify 轮在 `QueryFlow`
  层把上一轮 `session_meta` 透传, 防止用户答复澄清后丢失"第X篇"锚点.

下方原始伪代码保留作为接入历史参考。

### 改动点 1: `default_config.yaml`

```yaml
retrieval:
  langgraph:
    routing:
      mode: "legacy"        # legacy | fc; legacy = 现状, fc = 启用本模块
      enable_multi: true
      enable_ask: false
      parallel_tool_calls: false
```

### 改动点 2: `_make_router_node` 工厂 (`langgraph_agent.py`)

伪代码 (实际改动 ~10 行):

```python
def _make_router_node(router: QueryRouter, routing_core=None) -> callable:
    def router_node(state: AgentState) -> AgentState:
        query = state["query"]
        history = state.get("history")
        last_round_docs = state.get("last_round_docs") or []

        if routing_core is not None:
            # ── FC 路径 ──
            outcome = routing_core.route(query, history=history, doc_registry=last_round_docs)
            if isinstance(outcome, ClarifyRequest):
                state["clarify_request"] = {"q": outcome.question, "opts": outcome.options}
                state["decision"] = None
            elif isinstance(outcome, MultiRouteDecision):
                state["multi_decision"] = outcome
                # 简单兼容: 取第一个 subquery 的 decision 作为主 decision, 让现有 retrieve 跑通
                state["decision"] = outcome.subqueries[0].decision
                # P5 阶段才接通 subquery 分组检索
            else:
                state["decision"] = outcome
        else:
            # ── legacy JSON 路径 ──
            try:
                state["decision"] = router.route(query, history=history, doc_registry=last_round_docs)
            except Exception:
                state["decision"] = _heuristic_fallback_decision(query, datetime.datetime.now().year)
        return state
    return router_node
```

### 改动点 3: `_make_reflect_node` 工厂

伪代码:

```python
def _make_reflect_node(reflect_llm, routing_core=None, ...) -> callable:
    def reflect_node(state: AgentState) -> AgentState:
        if routing_core is not None and routing_core.reflect_llm is not None:
            # ── FC 反思路径 ──
            results_summary, total = _summarize_results(state.get("route_results", {}))
            verdict = routing_core.reflect(
                query=state["query"],
                last_decision=state.get("decision"),
                results_summary=results_summary,
                total_hits=total,
                this_round_docs=state.get("this_round_docs") or [],
                retry_count=state.get("retry_count", 0),
                max_retries=state.get("max_retries", 2),
            )
            state["needs_retry"] = verdict.needs_retry
            state["rewrite_hint"] = verdict.decision if verdict.needs_retry else None
            if verdict.partial:
                state["partial_note"] = verdict.partial_note
        else:
            # ── legacy JSON 反思 ──
            <现有 _make_reflect_node 逻辑保持不变>
        return state
    return reflect_node
```

### 改动点 4: `build_langgraph_agent_from_pipeline`

```python
def build_langgraph_agent_from_pipeline(pipeline, *, routing_mode="legacy", ...):
    routing_core = None
    if routing_mode == "fc":
        from ..routing import build_routing_core_from_query_router
        routing_core = build_routing_core_from_query_router(
            query_router=pipeline.router,
            reflect_llm=reflection_llm,
            enable_multi=...,
            disable_thinking=...,
        )
    compiled = build_langgraph_agent(
        router=pipeline.router,
        routing_core=routing_core,   # 新参数, None 时走 legacy
        ...
    )
    return LangGraphAgent(compiled_graph=compiled, ...)
```

## 测试方式 (落地前可以先单独跑)

```python
from pipeline.config import load_config
from pipeline.clients.llm import LLMClient
from pipeline.retrieval.agentic import QueryRouter
from pipeline.routing import build_routing_core_from_query_router
import datetime

cfg = load_config()
gen_cfg = cfg.generation
llm = LLMClient(
    api_base=gen_cfg["api_base"], model=gen_cfg["model"],
    api_key=gen_cfg["api_key"],
    disable_thinking_extra_body=gen_cfg.get("disable_thinking_extra_body", False),
)
router = QueryRouter(llm=llm, current_year=datetime.datetime.now().year)
core = build_routing_core_from_query_router(
    query_router=router, reflect_llm=llm,
    enable_multi=True, enable_ask=False,
    disable_thinking=gen_cfg.get("disable_thinking", False),
)

# 用例 1: 单一意图
out = core.route("X 文献里 LiNiCoMnO2 的循环寿命")
print(out)

# 用例 2: 上下文回指
history = [
    {"role": "user", "content": "X 文献讲的什么"},
    {"role": "assistant", "content": "X 文献研究 LiNiCoMnO2 正极材料"},
]
last_docs = [{"doc_id": "x1", "doc_name": "X 文献"}]
out = core.route("它的优点呢", history=history, doc_registry=last_docs)
print(out)

# 用例 3: 复合查询
out = core.route("X 文献里图 3, 再讲讲 Y 文献的方法")
print(out)   # 期望 MultiRouteDecision
```

## 路由容量与复合查询 (#8)

| 配置项 | 默认 | 含义 |
|---|---|---|
| `routing.max_paths_per_sub` | 2 | 单个 plan / multi.sub 内最多 paths (互补双路径) |
| `routing.max_subqueries` | 3 | multi 最多子查询数 |

**设计要点:**
- 多个**互斥 filter** (不同图/页/参考文献) 不应合并进同一 `plan.paths` → 应走 `multi`, 每意图一个 sub。
- 若 LLM 仍用 plan 提交互斥 paths, `RoutingCore` 会在服务端 **auto-split** 为 `MultiRouteDecision` (`tool_name=multi_auto_split`)。
- 检测到多意图时, FC user message 会注入 `[系统提示]` 引导使用 multi。
- legacy reflect JSON 路径的 `_normalize_routes` 与 FC schema 共用 `routing.limits`, 上限来自 `routing_core.routing_limits`。

## 风险与回滚

| 风险 | 对策 |
|---|---|
| vLLM 后端 FC 解析失败 | `fc_parser` 走多分支兜底; 全部失败时 `RoutingCore._fc_unsupported=True`, 后续走 heuristic. **不会抛出未捕获异常给 graph** |
| LLM 调了未知工具名 | `RoutingCore._dispatch_router_call` 退化为 heuristic |
| 决策合并 bug | 现有 `QueryRouter._validate_decision` 通过 `validate_fn` 注入, 复用 200 行防御逻辑; 即使 FC 输出有漏洞, 兜底仍能产出合法 RouteDecision |
| 接入后回归现状 | `routing_mode="legacy"` 即关闭, **完全等价于现状** |

## LLM 调用次数 (硬上限)

| 路径 | 调用次数 |
|---|---|
| 单 query, conf 高, reranker 通过 | **1** (router only) |
| 单 query, 反思 ok / partial | **2** (router + reflect) |
| 单 query, 反思 retry, 第二轮 reranker 通过 | **2** (router + reflect; rewrite_node 不再调 LLM) |
| **永不超过** | **2** |

预算余 1 次给未来加 "独立 retry-router" 用更强模型重做策略。
