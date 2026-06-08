"""LangGraph Agentic RAG: 带自我反思循环的检索-生成智能体。

与现有 AgenticRAGPipeline 并行共存，通过 config 切换。
核心改进: 检索后自动评估结果质量，不足时改写查询重试 (最多 max_retries 次)。

依赖:
  - langgraph (可选, pip install langgraph)
  - 复用现有: QueryRouter, SummaryRetriever, ProgressiveLocalRetriever,
    EnhancedMetadataRetriever, AgenticContextBuilder, LLMClient
"""

from __future__ import annotations

import datetime
import json
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, TypedDict

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    StateGraph = None  # type: ignore[assignment,misc]
    END = None  # type: ignore[assignment]

from .agentic import (
    AgenticContextBuilder,
    AgenticRAGPipeline,
    EnhancedMetadataRetriever,
    ProgressiveLocalRetriever,
    QueryRouter,
    RouteDecision,
    SummaryRetriever,
    _heuristic_fallback_decision,
    _format_doc_registry_block,
    ROUTE_SUMMARY,
    ROUTE_PROGRESSIVE,
    ROUTE_LOCAL,
    ROUTE_METADATA,
    VALID_ROUTES,
    ROUTE_ALIAS,
    NON_SUMMARY_TYPE_FILTER,
    _and_filter,
    _escape_like,
    LocalRetrieveResult,
    DEFAULT_AGENTIC_SYSTEM_PROMPT,
    AGENTIC_USER_TEMPLATE,
)
from .route_filters import chunk_type_for_route, describe_route_chunk_types
from .retrievers import Hit
from .reflect_summary import (
    ReflectSummaryConfig,
    should_accept_structural_only_results,
    summarize_for_reflect,
)
from ..clients.query_format import collect_prewarm_embed_texts, compose_rerank_document, synthesize_rerank_query
from .structural_retrieval import (
    ExemptDecision,
    hit_exempt_decision,
    hit_exempt_from_rerank_filter,  # 兼容旧引用
)
from .quality_thresholds import RouteThresholds
from .hybrid_weights import infer_retrieve_bias_heuristic, normalize_retrieve_bias
from ..clients.llm import LLMClient
from ..clients.reranker import RerankerClient

# routing 模块为可选: 未启用 FC 时不强依赖
try:
    from ..routing import (
        RoutingCore,
        ReflectVerdict,
        MultiRouteDecision,
        SubqueryDecision,
        ClarifyRequest,
        ReuseRequest,
    )
except ImportError:  # pragma: no cover
    RoutingCore = None  # type: ignore[assignment,misc]
    ReflectVerdict = None  # type: ignore[assignment,misc]
    MultiRouteDecision = None  # type: ignore[assignment,misc]
    SubqueryDecision = None  # type: ignore[assignment,misc]
    ClarifyRequest = None  # type: ignore[assignment,misc]
    ReuseRequest = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State 定义
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """LangGraph agent 共享状态。"""

    # -- 输入 --
    query: str
    history: Optional[List[Dict[str, str]]]

    # -- 跨轮文献编号 (issue #1, 修订) --
    # last_round_docs: 会话 browseable 文献 catalog (持久化于 session_meta.doc_registry);
    #   router 解析用户 "第X篇" 的唯一锚点, 指定文献下钻时不应被缩成单篇。
    # this_round_docs: 本轮所有 retrieve 累积命中的文献 (含 retry 后追加的);
    #   仅用于反思器自我引用; run() 结束时经 _persist_doc_registry 写回 catalog。
    last_round_docs: List[Dict[str, str]]
    this_round_docs: List[Dict[str, str]]

    # -- Router --
    decision: RouteDecision
    subquery_decisions: List[RouteDecision]   # MultiRouteDecision 各子查询独立决策
    multi_decision: Any                     # 原始 MultiRouteDecision (含 synth_hint)
    synth_hint: str                         # 复合查询上下文拼接提示

    # -- Policy / 控制层 --
    agent_phase: str                       # 当前所处阶段 (router/retrieve/reflect/...) 
    next_action: str                        # policy 决定的下一步动作
    action_reason: str                      # policy 决策原因
    action_history: List[Dict[str, Any]]    # 决策轨迹
    evidence_gaps: List[str]                # 当前证据缺口
    sufficient: bool                        # 当前是否足够回答
    uncertainty_note: str                   # 不确定性说明

    # -- Clarify (FC ask 工具) --
    needs_clarify: bool
    clarify_request: Dict[str, Any]         # {"q": str, "opts": List[str]}
    clarify_answer: str                     # 格式化后的反问文本, 直接返回 user

    # -- Reuse (FC reuse 工具: 不检索直接生成) --
    needs_reuse: bool                       # True=本轮跳过 retrieve, 走 generate_reuse
    reuse_request: Dict[str, Any]           # {"mode": str, "op": str}

    # -- 跨轮记忆 (router 复用判定 + reuse 生成时引用) --
    last_answer: str                        # 上一轮最终 answer (≤ persist 上限)
    last_context: str                       # 上一轮 retrieve 后构建的 context (≤ persist 上限)
    clarify_pending: Dict[str, Any]         # 上一轮 ask 的反问 {"q","opts"}

    # -- 检索 --
    route_results: Dict[str, Any]       # route → List[Hit] | LocalRetrieveResult
    route_results_pre_rerank: Dict[str, Any]  # reranker 过滤前的完整快照 (可恢复)
    route_errors: Dict[str, str]        # route → error msg

    # -- 反思 --
    needs_retry: bool                   # True=检索不足需重试
    rewrite_hint: Optional[Any]         # RouteDecision (reflect 输出的完整检索策略)
    partial_note: str                   # FC reflect partial 工具输出, 拼到 context 末尾

    # -- Reranker --
    reranker_score: float               # reranker 计算的 top-k 平均相关性得分
    rerank_diagnosis_summary: str       # 低分诊断摘要, 注入 reflect prompt
    rerank_skip_reflect: bool           # Phase 2: 高置信诊断时跳过 reflect 直走 rewrite
    rerank_diagnosis_cause: str         # 诊断 cause 码 (日志/指标)
    rerank_diagnosis_confidence: float  # 诊断规则置信度 (与 skip_reflect_confidence 对比)

    # -- 重试 --
    retry_count: int
    max_retries: int

    # -- 上下文 & 生成 --
    context: str
    answer: str
    usage: Optional[Dict[str, Any]]

    # -- 可观测性 --
    correlation_id: str
    node_timings: Dict[str, float]


# ---------------------------------------------------------------------------
# 反思 Prompt (issue #3)
#
# 设计原则: 反思器与路由器必须遵守同一份"路径/改写/filters"契约, 否则反思
# 重写出的 RouteDecision 拿到 retrieve_node 时会因为不一致而行为漂移。
# 内容从 prompts/ 目录加载, 修改 prompt 只需编辑 MD 文件。
# ---------------------------------------------------------------------------

from ..prompts import render_reflect_system as _render_reflect_system_from_file


def _reflect_system_prompt(current_year: int) -> str:
    """组装反思 prompt: 从 reflect_system.md 加载 (含嵌入的 router_rules)。"""
    return _render_reflect_system_from_file(current_year)



REFLECT_USER_TEMPLATE = """问题: {query}{doc_registry_block}

上一轮策略:
  路径: {routes}
  改写词: {rewrites}
  过滤: {filters_json}

检索结果 (共 {total_hits} 条):
{results_summary}{rerank_diagnosis_block}

请评估并输出 JSON。"""


# Reuse 出口专用 user 模板 (替代 AGENTIC_USER_TEMPLATE 里的"检索到的多路径上下文"措辞).
# context 来自 _build_reuse_context, 内部已经写好系统模式说明 + 上一轮素材, 不需要再加引用提示.
REUSE_USER_TEMPLATE = (
    "{context}\n\n"
    "请严格按上面的【系统模式】与【路由器指令 (op)】要求, "
    "直接给出最终回答。不要假装做了新检索, 也不要捏造未在上一轮 context 中出现的事实。"
)


# ---------------------------------------------------------------------------
# JSON 解析工具
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _extract_json(text: str) -> Optional[str]:
    from ..clients.thinking_utils import strip_think_blocks

    text = strip_think_blocks(text)
    if not text:
        return None
    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    return None


# ---------------------------------------------------------------------------
# 结果摘要 (供反思 LLM 消费)
# ---------------------------------------------------------------------------

def _summarize_results(
    route_results: Dict[str, Any], max_chars: int = 2000,
) -> Tuple[str, int]:
    """Deprecated: 反思请用 summarize_for_reflect (排除结构化 chunk)。"""
    return summarize_for_reflect(
        route_results,
        config=ReflectSummaryConfig(max_total_chars=max_chars),
    )


# ---------------------------------------------------------------------------
# 节点函数
# ---------------------------------------------------------------------------

def _make_router_node(
    router: QueryRouter,
    routing_core: Optional[Any] = None,
) -> callable:
    """构建 router_node。

    Args:
        router: 既有 QueryRouter (legacy JSON 路径)
        routing_core: 可选 RoutingCore (FC 路径); 注入则优先走 FC, 失败自动降级到 router
    """
    backend = "fc" if routing_core is not None else "legacy"
    logger.info(f"[router] router_node backend={backend}")

    def router_node(state: AgentState) -> AgentState:
        t0 = time.time()
        cid = state.get("correlation_id", "?")
        query = state["query"]
        history = state.get("history")
        # router 看的"第X篇"必须锚定到上一轮最终结果, 不是本轮中间态 (issue #1 修订)
        last_round_docs = state.get("last_round_docs") or []
        # 跨轮 reuse 判定信号 (P0-1 / P1-2)
        last_answer = state.get("last_answer") or ""
        last_context = state.get("last_context") or ""
        clarify_pending = state.get("clarify_pending") or None

        logger.info(
            f"[{cid}] [router] begin: backend={backend} query={query[:80]!r} "
            f"history_msgs={len(history) if history else 0} "
            f"last_round_docs={len(last_round_docs)} "
            f"last_answer={len(last_answer)}c last_context={len(last_context)}c "
            f"clarify_pending={bool(clarify_pending)}"
        )

        decision: Optional[RouteDecision] = None

        # ── FC 路径 ──
        if routing_core is not None:
            try:
                outcome = routing_core.route(
                    query, history=history, doc_registry=last_round_docs,
                    correlation_id=cid,
                    last_answer=last_answer or None,
                    last_context_preview=last_context or None,
                    clarify_pending=clarify_pending,
                )
                if isinstance(outcome, RouteDecision):
                    decision = outcome
                elif MultiRouteDecision is not None and isinstance(outcome, MultiRouteDecision):
                    sub_decs = [s.decision for s in outcome.subqueries]
                    logger.info(
                        f"[{cid}] [router] FC 返回 MultiRouteDecision "
                        f"(subs={len(sub_decs)}), 将并行执行全部子查询"
                    )
                    state["multi_decision"] = outcome
                    state["subquery_decisions"] = sub_decs
                    state["synth_hint"] = outcome.synth_hint or ""
                    decision = _merge_route_decisions(sub_decs) if sub_decs else None
                elif ClarifyRequest is not None and isinstance(outcome, ClarifyRequest):
                    logger.info(
                        f"[{cid}] [router] FC 返回 ClarifyRequest "
                        f"question={outcome.question[:80]!r}, 跳过检索走 clarify 出口"
                    )
                    state["needs_clarify"] = True
                    state["clarify_request"] = {
                        "q": outcome.question,
                        "opts": outcome.options,
                    }
                    state["decision"] = None
                    state["agent_phase"] = "router"
                    state.setdefault("node_timings", {})["router"] = time.time() - t0
                    return state
                elif ReuseRequest is not None and isinstance(outcome, ReuseRequest):
                    lock_brief = (
                        f" refs={outcome.doc_refs} docs={outcome.target_docs}"
                        if outcome.doc_refs or outcome.target_docs
                        else ""
                    )
                    # drilldown / continue 需要更多证据，不应跳过检索；
                    # 转为 local + refs + expand 走正常检索流程，避免生成 LLM 编造。
                    if outcome.mode in ("drilldown", "continue"):
                        rewrite_text = (outcome.op or query).strip() or query
                        decision = RouteDecision(
                            routes=[ROUTE_LOCAL],
                            rewrites={ROUTE_LOCAL: rewrite_text},
                            target_docs=list(outcome.target_docs or []),
                            target_doc_ids=list(outcome.target_doc_ids or []),
                            expand_neighbors=["assets", "adjacent"],
                            reasoning=f"(reuse-{outcome.mode}-converted-to-local)",
                        )
                        logger.info(
                            f"[{cid}] [router] FC 返回 ReuseRequest "
                            f"mode={outcome.mode} op={outcome.op[:80]!r}{lock_brief}, "
                            f"转为 local+expand 走实际检索 (避免 drilldown 跳过检索导致编造)"
                        )
                        state["decision"] = decision
                        state["subquery_decisions"] = [decision]
                        state["agent_phase"] = "router"
                        state.setdefault("node_timings", {})["router"] = time.time() - t0
                        return state
                    # reformat / metasession / confirm / chitchat / out_of_scope：
                    # 这些模式确实不需要新检索，保持原 reuse 行为
                    logger.info(
                        f"[{cid}] [router] FC 返回 ReuseRequest "
                        f"mode={outcome.mode} op={outcome.op[:80]!r}{lock_brief}, "
                        f"跳过检索走 reuse 出口"
                    )
                    state["needs_reuse"] = True
                    state["reuse_request"] = {
                        "mode": outcome.mode,
                        "op": outcome.op,
                        "doc_refs": list(outcome.doc_refs or []),
                        "target_doc_ids": list(outcome.target_doc_ids or []),
                        "target_docs": list(outcome.target_docs or []),
                    }
                    state["decision"] = None
                    state["agent_phase"] = "router"
                    state.setdefault("node_timings", {})["router"] = time.time() - t0
                    return state
                else:
                    logger.warning(
                        f"[{cid}] [router] FC 返回未知类型 {type(outcome).__name__}, "
                        f"走 heuristic"
                    )
                    decision = _heuristic_fallback_decision(query, datetime.datetime.now().year)
            except Exception as e:
                logger.warning(
                    f"[{cid}] [router] routing_core.route 异常 ({type(e).__name__}: {e}), "
                    f"降级到 legacy router"
                )
                decision = None

        # ── Legacy 路径 (FC 未启用 或 FC 失败) ──
        if decision is None:
            try:
                decision = router.route(
                    query, history=history, doc_registry=last_round_docs,
                )
            except Exception as e:
                logger.warning(f"[{cid}] [router] legacy LLM 失败, 走 heuristic: {e}")
                decision = _heuristic_fallback_decision(query, datetime.datetime.now().year)

        rerank_src = (
            "rewrite_kw"
            if decision.rerank_mode is True
            else "user_query"
        )
        logger.info(
            f"[{cid}] [router] decision: routes={decision.routes} "
            f"rewrites={dict(decision.rewrites or {})} "
            f"rerank_mode={decision.rerank_mode!r} rerank_query_src={rerank_src} "
            f"target_docs={decision.target_docs} "
            f"target_doc_ids={decision.target_doc_ids} "
            f"fig_refs={decision.fig_refs} table_refs={decision.table_refs} "
            f"page_refs={decision.page_refs} paragraph_refs={decision.paragraph_refs} "
            f"entities={decision.entities} time={decision.time!r} chunk_type={decision.chunk_type}"
        )

        state["decision"] = decision
        state["agent_phase"] = "router"
        state.setdefault("node_timings", {})["router"] = time.time() - t0
        return state
    return router_node


# ---------------------------------------------------------------------------
# 文献编号列表维护 (issue #1)
# ---------------------------------------------------------------------------

# P1-3: doc_registry 滑动窗口上限. 超过此值时会淘汰最早的"未 pin"项;
# 用户用 refs 显式回指过的文献会被 pin 住, 不参与淘汰.
_DOC_REGISTRY_MAX_ENTRIES = 40


def _normalize_registry_entry(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """把任意来源的 entry 归一为 {doc_id, doc_name, pinned} 三元组。"""
    if not isinstance(entry, dict):
        return None
    did = entry.get("doc_id")
    if not did:
        return None
    return {
        "doc_id": str(did),
        "doc_name": str(entry.get("doc_name") or did),
        "pinned": bool(entry.get("pinned", False)),
    }


def _merge_doc_registry(
    base: List[Dict[str, Any]],
    extra: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """按 doc_id 去重合并文献列表, 保持 base 顺序; pinned 标志 OR 合并。"""
    merged: List[Dict[str, Any]] = []
    idx_by_id: Dict[str, int] = {}
    for b in base:
        n = _normalize_registry_entry(b)
        if n is None:
            continue
        merged.append(n)
        idx_by_id[n["doc_id"]] = len(merged) - 1
    for e in extra:
        n = _normalize_registry_entry(e)
        if n is None:
            continue
        existing = idx_by_id.get(n["doc_id"])
        if existing is not None:
            if n.get("pinned"):
                merged[existing]["pinned"] = True
            continue
        idx_by_id[n["doc_id"]] = len(merged)
        merged.append(n)
    return merged


def _apply_registry_sliding_window(
    registry: List[Dict[str, Any]],
    *,
    max_entries: int = _DOC_REGISTRY_MAX_ENTRIES,
) -> List[Dict[str, Any]]:
    """滑动窗淘汰: 超过 max_entries 时丢弃最早出现的未 pin 项, pinned 项永远保留。

    保留顺序: pinned 项相对位置不变, 末尾未 pin 项优先保留 (LRU 语义),
    最终列表中 pinned 与 非-pinned 项按"原插入顺序"穿插。
    """
    if max_entries <= 0 or len(registry) <= max_entries:
        return list(registry)

    pinned_indices = [i for i, e in enumerate(registry) if e.get("pinned")]
    n_pin = len(pinned_indices)
    if n_pin >= max_entries:
        # pin 已经超过上限 — 保留全部 pin, 但仍要警告
        logger.warning(
            f"[doc_registry] pinned 文献数 ({n_pin}) >= max_entries ({max_entries}), "
            f"全部保留 (无法再淘汰)"
        )
        return [registry[i] for i in pinned_indices]

    keep_unpinned = max_entries - n_pin
    unpinned_indices = [i for i, e in enumerate(registry) if not e.get("pinned")]
    # LRU: 保留靠近末尾的未 pin 项
    keep_set = set(pinned_indices) | set(unpinned_indices[-keep_unpinned:])
    dropped = len(registry) - len(keep_set)
    logger.info(
        f"[doc_registry] 滑动窗淘汰: total={len(registry)} → keep={len(keep_set)} "
        f"(pinned={n_pin}, unpinned_kept={keep_unpinned}, dropped={dropped})"
    )
    return [registry[i] for i in sorted(keep_set)]


def _pin_explicit_refs(
    registry: List[Dict[str, Any]],
    decision: Optional[RouteDecision],
    subquery_decisions: Optional[List[RouteDecision]] = None,
) -> List[Dict[str, Any]]:
    """把本轮 decision.target_docs 中确实出现在 registry 里的项标记为 pinned。

    判定标准: user 用 refs / docs 明确指向某篇 → 该篇标 pinned, 长期保留。
    """
    if not registry:
        return registry
    pin_names: set = set()
    decisions_to_scan: List[RouteDecision] = []
    if decision is not None:
        decisions_to_scan.append(decision)
    if subquery_decisions:
        decisions_to_scan.extend(d for d in subquery_decisions if isinstance(d, RouteDecision))
    for d in decisions_to_scan:
        for name in (d.target_docs or []):
            if name:
                pin_names.add(name)
    if not pin_names:
        return registry
    pinned_count = 0
    for entry in registry:
        if entry.get("doc_name") in pin_names and not entry.get("pinned"):
            entry["pinned"] = True
            pinned_count += 1
    if pinned_count:
        logger.info(
            f"[doc_registry] pin {pinned_count} 篇 (用户显式 target_docs 回指): "
            f"{sorted(pin_names)}"
        )
    return registry


def _is_catalog_shrinking_drilldown(
    decision: Optional[RouteDecision],
    prev_registry: List[Dict[str, Any]],
    this_round_docs: List[Dict[str, Any]],
) -> bool:
    """本轮是否仅为在已有 catalog 内指定单篇/少数文献的 local 下钻。"""
    if not prev_registry or not this_round_docs:
        return False
    if len(this_round_docs) >= len(prev_registry):
        return False
    prev_ids = {e.get("doc_id") for e in prev_registry if e.get("doc_id")}
    this_ids = {e.get("doc_id") for e in this_round_docs if e.get("doc_id")}
    if not this_ids or not this_ids.issubset(prev_ids):
        return False
    if decision is None:
        return True
    if decision.has(ROUTE_SUMMARY):
        return False
    if decision.has(ROUTE_LOCAL) and decision.target_docs:
        return True
    return False


def _persist_doc_registry(
    prev_registry: List[Dict[str, Any]],
    this_round_docs: List[Dict[str, Any]],
    decision: Optional[RouteDecision],
    subquery_decisions: Optional[List[RouteDecision]] = None,
) -> List[Dict[str, Any]]:
    """计算写入 session_meta 的 browseable doc_registry。

    - 文献发现 (summary / 多文献 progressive): 用本轮列表刷新 catalog (但保留 prev 中的 pinned 项)
    - local 指定文献下钻: 保留上一轮完整 catalog, 避免 "第X篇" 锚点被缩成 1 篇
    - 其余: 并集合并

    P1-3 改造:
      - registry 条目支持 pinned 标记; user 用 refs 回指过的文献会被 pin 长期保留
      - 超过 _DOC_REGISTRY_MAX_ENTRIES 时滑动窗淘汰未 pin 的最早项
    """
    prev_norm = [n for n in (_normalize_registry_entry(e) for e in prev_registry) if n]
    this_norm = [n for n in (_normalize_registry_entry(e) for e in this_round_docs) if n]

    # 先按业务规则决定 "base" 用上轮还是本轮
    if not this_norm:
        merged = list(prev_norm)
    elif _is_catalog_shrinking_drilldown(decision, prev_norm, this_norm):
        logger.info(
            "[doc_registry] local 指定文献检索 (本轮 %d 篇 ⊆ 上轮 %d 篇), "
            "保留完整 browseable 列表供后续 '第X篇' 回指",
            len(this_norm), len(prev_norm),
        )
        merged = _merge_doc_registry(prev_norm, this_norm)  # this 不影响 base, 但补 pin
    elif not prev_norm:
        merged = list(this_norm)
    elif decision and decision.has(ROUTE_SUMMARY):
        # 文献发现: 用本轮覆盖, 但保留之前 pin 过的项
        prev_pinned = [e for e in prev_norm if e.get("pinned")]
        merged = _merge_doc_registry(this_norm, prev_pinned)
    elif decision and decision.has(ROUTE_PROGRESSIVE) and len(this_norm) >= 2:
        prev_pinned = [e for e in prev_norm if e.get("pinned")]
        merged = _merge_doc_registry(this_norm, prev_pinned)
    elif len(this_norm) > len(prev_norm):
        prev_pinned = [e for e in prev_norm if e.get("pinned")]
        merged = _merge_doc_registry(this_norm, prev_pinned)
    else:
        merged = _merge_doc_registry(prev_norm, this_norm)

    # 标记本轮用户显式 refs/docs 回指过的为 pinned
    merged = _pin_explicit_refs(merged, decision, subquery_decisions)

    # 滑动窗淘汰
    merged = _apply_registry_sliding_window(merged)
    return merged


def _update_this_round_docs(
    state: AgentState, route_results: Dict[str, Any],
) -> List[Dict[str, str]]:
    """把本轮 route_results 命中的 (doc_id, doc_name) 增量并入 this_round_docs。

    保持插入顺序; 同一 doc_id 只入册一次。retry 时本函数被多次调用, 累积
    覆盖。本轮结束后 state["this_round_docs"] 会被 run() 持久化, 在下一轮
    入参时作为 last_round_docs 给 router 使用。
    """
    docs: List[Dict[str, str]] = list(state.get("this_round_docs") or [])
    seen_ids = {entry.get("doc_id") for entry in docs if entry.get("doc_id")}

    def _try_add(doc_id: Optional[str], doc_name: Optional[str]) -> None:
        if not doc_id or doc_id in seen_ids:
            return
        seen_ids.add(doc_id)
        docs.append({"doc_id": doc_id, "doc_name": doc_name or doc_id})

    for res in route_results.values():
        if isinstance(res, LocalRetrieveResult):
            for cd in res.candidate_docs:
                _try_add(cd.doc_id, cd.doc_name)
            for h in res.chunk_hits:
                _try_add(h.doc_id, h.doc_name)
        elif isinstance(res, list):
            for h in res:
                if isinstance(h, Hit):
                    _try_add(h.doc_id, h.doc_name)

    state["this_round_docs"] = docs
    return docs


# ---------------------------------------------------------------------------
# route_results 合并 / 快照 (issue: retry 增量合并 + reranker 可恢复)
# ---------------------------------------------------------------------------

def _count_hits(route_results: Dict[str, Any]) -> int:
    """累计 route_results 中所有命中 chunk 数 (跨 list / LocalRetrieveResult)。"""
    total = 0
    for v in (route_results or {}).values():
        if isinstance(v, LocalRetrieveResult):
            total += len(v.chunk_hits)
        elif isinstance(v, list):
            total += len(v)
    return total


def _copy_route_results(route_results: Dict[str, Any]) -> Dict[str, Any]:
    """浅拷贝 route_results, 列表/LocalRetrieveResult 内部复制一份。"""
    out: Dict[str, Any] = {}
    for route, res in (route_results or {}).items():
        if isinstance(res, LocalRetrieveResult):
            out[route] = LocalRetrieveResult(
                candidate_docs=list(res.candidate_docs),
                chunk_hits=list(res.chunk_hits),
            )
        elif isinstance(res, list):
            out[route] = list(res)
        else:
            out[route] = res
    return out


def _merge_hit_lists(existing: List[Hit], new_hits: List[Hit]) -> List[Hit]:
    """按 pk 去重合并 hit 列表, 保留 existing 顺序, 追加 new 中未见 pk。"""
    seen = {h.pk for h in existing if h.pk}
    merged = list(existing)
    for h in new_hits:
        if not h.pk or h.pk in seen:
            continue
        seen.add(h.pk)
        merged.append(h)
    return merged


def _merge_local_results(
    existing: LocalRetrieveResult, new_res: LocalRetrieveResult,
) -> LocalRetrieveResult:
    """合并两个 LocalRetrieveResult (candidate_docs + chunk_hits 均按 id/pk 去重)。"""
    seen_doc_ids = {cd.doc_id for cd in existing.candidate_docs if cd.doc_id}
    merged_docs = list(existing.candidate_docs)
    for cd in new_res.candidate_docs:
        if cd.doc_id and cd.doc_id not in seen_doc_ids:
            seen_doc_ids.add(cd.doc_id)
            merged_docs.append(cd)
    merged_hits = _merge_hit_lists(existing.chunk_hits, new_res.chunk_hits)
    return LocalRetrieveResult(candidate_docs=merged_docs, chunk_hits=merged_hits)


def _merge_route_results(
    existing: Dict[str, Any], new_results: Dict[str, Any],
) -> Dict[str, Any]:
    """增量合并 route_results: 同 route 按 pk 去重, 新 route 直接加入。"""
    merged = _copy_route_results(existing or {})
    for route, res in (new_results or {}).items():
        prev = merged.get(route)
        if isinstance(res, LocalRetrieveResult):
            if isinstance(prev, LocalRetrieveResult):
                merged[route] = _merge_local_results(prev, res)
            else:
                merged[route] = LocalRetrieveResult(
                    candidate_docs=list(res.candidate_docs),
                    chunk_hits=list(res.chunk_hits),
                )
        elif isinstance(res, list):
            prev_hits = prev if isinstance(prev, list) else []
            merged[route] = _merge_hit_lists(prev_hits, res)
        else:
            merged[route] = res
    return merged


def _merge_route_errors(
    existing: Dict[str, str], new_errors: Dict[str, str],
) -> Dict[str, str]:
    merged = dict(existing or {})
    merged.update(new_errors or {})
    return merged


def _subquery_id(index: int) -> str:
    return f"sub{index + 1}"


def _subquery_rerank_query(decision: RouteDecision, fallback: str) -> str:
    """子查询用于 rerank 的文本: 优先用户自然语言问句。"""
    return synthesize_rerank_query(decision, fallback)


def _decision_for_subquery_id(
    subquery_id: str,
    subquery_decisions: Optional[List[RouteDecision]],
) -> Optional[RouteDecision]:
    if not subquery_id or not subquery_decisions:
        return None
    m = re.match(r"^sub(\d+)$", (subquery_id or "").strip())
    if not m:
        return None
    idx = int(m.group(1)) - 1
    if 0 <= idx < len(subquery_decisions):
        sub = subquery_decisions[idx]
        return sub if isinstance(sub, RouteDecision) else None
    return None


def _resolve_rerank_query_for_hit(
    hit: Hit,
    user_query: str,
    decision: Optional[RouteDecision],
    subquery_decisions: Optional[List[RouteDecision]],
) -> str:
    """为单条 hit 解析 rerank query (单路/复合统一走 synthesize_rerank_query)。"""
    sub_dec = _decision_for_subquery_id(hit.subquery_id, subquery_decisions)
    if sub_dec is not None:
        rq = synthesize_rerank_query(sub_dec, user_query)
    elif isinstance(decision, RouteDecision):
        rq = synthesize_rerank_query(decision, user_query)
    else:
        rq = (hit.subquery_rewrite or user_query or "").strip()
    return rq or (user_query or "").strip() or user_query


def _stamp_subquery_on_route_results(
    results: Dict[str, Any],
    subquery_id: str,
    subquery_rewrite: str,
) -> Dict[str, Any]:
    for res in results.values():
        if isinstance(res, LocalRetrieveResult):
            for h in res.chunk_hits:
                h.subquery_id = subquery_id
                h.subquery_rewrite = subquery_rewrite
        elif isinstance(res, list):
            for h in res:
                if isinstance(h, Hit):
                    h.subquery_id = subquery_id
                    h.subquery_rewrite = subquery_rewrite
    return results


def _prefix_route_result_keys(
    results: Dict[str, Any],
    subquery_id: str,
) -> Dict[str, Any]:
    return {f"{subquery_id}:{route}": res for route, res in results.items()}


def _prefix_route_error_keys(
    errors: Dict[str, str],
    subquery_id: str,
) -> Dict[str, str]:
    return {f"{subquery_id}:{route}": msg for route, msg in errors.items()}


def _merge_route_decisions(decisions: List[RouteDecision]) -> RouteDecision:
    """把多个子查询 RouteDecision 合成一个, 供 context_builder 展示。"""
    if not decisions:
        return RouteDecision()
    if len(decisions) == 1:
        return decisions[0]

    routes: List[str] = []
    rewrites: Dict[str, str] = {}
    target_docs: List[str] = []
    target_doc_ids: List[str] = []
    fig_refs: List[str] = []
    table_refs: List[str] = []
    page_refs: List[int] = []
    paragraph_refs: List[int] = []
    entities: List[str] = []

    def _extend_unique(dst: List, src: List) -> None:
        seen = set(dst)
        for x in src:
            if x not in seen:
                seen.add(x)
                dst.append(x)

    for d in decisions:
        for r in d.routes:
            if r not in routes:
                routes.append(r)
        for route, rw in (d.rewrites or {}).items():
            if route not in rewrites:
                rewrites[route] = rw
            elif rw and rw != rewrites[route]:
                rewrites[route] = f"{rewrites[route]} | {rw}"
        _extend_unique(target_docs, d.target_docs or [])
        _extend_unique(target_doc_ids, d.target_doc_ids or [])
        _extend_unique(fig_refs, d.fig_refs or [])
        _extend_unique(table_refs, d.table_refs or [])
        _extend_unique(page_refs, d.page_refs or [])
        _extend_unique(paragraph_refs, d.paragraph_refs or [])
        _extend_unique(entities, d.entities or [])

    chunk_type = next((d.chunk_type for d in decisions if d.chunk_type), None)
    time_val = next((d.time for d in decisions if d.time), "")
    retrieve_bias = next((d.retrieve_bias for d in decisions if d.retrieve_bias), None)
    rerank_mode = True if any(d.rerank_mode is True for d in decisions) else None
    reasoning_parts = [d.reasoning for d in decisions if d.reasoning]
    return RouteDecision(
        routes=routes,
        rewrites=rewrites,
        time=time_val,
        chunk_type=chunk_type,
        target_docs=target_docs,
        target_doc_ids=target_doc_ids,
        fig_refs=fig_refs,
        table_refs=table_refs,
        page_refs=page_refs,
        paragraph_refs=paragraph_refs,
        entities=entities,
        retrieve_bias=retrieve_bias,
        rerank_mode=rerank_mode,
        reasoning=" | ".join(reasoning_parts) if reasoning_parts else "(compound)",
    )


def _decision_filters_dict(decision: RouteDecision) -> Dict[str, Any]:
    return {
        k: v for k, v in {
            "chunk_type": decision.chunk_type,
            "target_docs": decision.target_docs,
            "target_doc_ids": decision.target_doc_ids,
            "fig_refs": decision.fig_refs,
            "table_refs": decision.table_refs,
            "page_refs": decision.page_refs,
            "paragraph_refs": decision.paragraph_refs,
            "entities": decision.entities,
            "time": decision.time,
            "retrieve_bias": decision.retrieve_bias,
        }.items() if v
    }


def _format_reflect_strategy(
    decision: Optional[RouteDecision],
    subquery_decisions: Optional[List[RouteDecision]],
) -> Tuple[str, str, str]:
    """构建 reflect prompt 用的策略字段; 复合查询按子查询分段, 避免合并 rewrite。"""
    subs = subquery_decisions or []
    if len(subs) > 1:
        routes_str = ", ".join(
            f"sub{i + 1}:{'/'.join(d.routes or [])}" for i, d in enumerate(subs)
        )
        rewrites_str = json.dumps(
            {f"sub{i + 1}": dict(d.rewrites or {}) for i, d in enumerate(subs)},
            ensure_ascii=False,
        )
        filters_json = json.dumps(
            [{"sub_id": f"sub{i + 1}", **_decision_filters_dict(d)} for i, d in enumerate(subs)],
            ensure_ascii=False,
        )
        return routes_str, rewrites_str, filters_json
    if decision:
        return (
            ", ".join(decision.routes or []),
            json.dumps(decision.rewrites, ensure_ascii=False),
            json.dumps(_decision_filters_dict(decision), ensure_ascii=False),
        )
    return "", "{}", "{}"


def _has_rerank_retry_fallback(state: AgentState) -> bool:
    """reranker 门控失败且已写入诊断 rewrite 时, reflect 失败应保留兜底。"""
    if not state.get("needs_retry"):
        return False
    if state.get("rewrite_hint") is not None:
        return True
    return bool(state.get("subquery_decisions"))


def _apply_reflect_failure_state(
    state: AgentState,
    cid: str,
    *,
    source: str,
    error: BaseException | str,
) -> None:
    """reflect 调用/解析失败: 有 rerank 兜底则保留 needs_retry+rewrite_hint。"""
    err_s = str(error)
    if _has_rerank_retry_fallback(state):
        logger.info(
            f"[{cid}] [reflect] {source} 失败 ({err_s}), "
            f"保留 reranker 诊断兜底 "
            f"(cause={state.get('rerank_diagnosis_cause') or '?'}, "
            f"skip_reflect={state.get('rerank_skip_reflect', False)})"
        )
        return
    logger.warning(
        f"[{cid}] [reflect] {source} 失败 ({err_s}), 无 rerank 兜底, 默认 no_retry"
    )
    state["needs_retry"] = False
    state["rewrite_hint"] = None


def _fallback_context_on_build_error(
    query: str,
    route_results: Dict[str, Any],
    error: BaseException,
) -> str:
    n = _count_hits(route_results)
    return (
        f"# 用户问题\n{query}\n\n"
        f"# [系统] 上下文构建失败\n"
        f"已召回 {n} 条结果，但无法组装可读上下文 "
        f"({type(error).__name__}: {error})。"
    )


def _reflect_last_decision_for_fc(
    state: AgentState,
    decision: Optional[RouteDecision],
) -> Any:
    """FC reflect 优先传 MultiRouteDecision, 避免合并 rewrite 误导反思 LLM。"""
    sub_decs = state.get("subquery_decisions") or []
    if len(sub_decs) > 1:
        multi = state.get("multi_decision")
        if multi is not None:
            return multi
        if MultiRouteDecision is not None and SubqueryDecision is not None:
            return MultiRouteDecision(
                subqueries=[
                    SubqueryDecision(id=f"sub{i + 1}", decision=d)
                    for i, d in enumerate(sub_decs)
                ],
                synth_hint=state.get("synth_hint") or "",
            )
    return decision


def _truncate_for_persist(text: str, limit: int) -> str:
    """将字符串截断到指定长度, 超出部分以 '...(truncated)' 结尾, 用于 session_meta 持久化。"""
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 16)] + "...(truncated)"


def _format_clarify_answer(clarify_request: Dict[str, Any]) -> str:
    """把 ClarifyRequest 格式化为可直接返回用户的反问文本。"""
    question = str(clarify_request.get("q") or "").strip() or "请补充说明您的问题。"
    opts = clarify_request.get("opts") or []
    if isinstance(opts, list) and opts:
        opts_text = "\n".join(f"- {str(o).strip()}" for o in opts if str(o).strip())
        if opts_text:
            return f"{question}\n\n可选方向:\n{opts_text}"
    return question


def _format_no_answer(query: str, state: AgentState) -> str:
    note = (state.get("uncertainty_note") or "").strip()
    gaps = state.get("evidence_gaps") or []
    detail = note or "检索结果相关性不足，且已用完可用的改写重试预算。"
    if gaps and not note:
        detail += f" 证据缺口: {', '.join(str(g) for g in gaps)}。"
    return (
        "我在当前文献库中没有找到足够可靠的依据来回答这个问题，因此不建议直接给出结论。\n\n"
        f"问题：{query}\n"
        f"原因：{detail}\n\n"
        "你可以换更具体的材料/工艺/文献名再问，或让我先帮你查找相关文献清单。"
    )


def _collect_decisions_to_run(state: AgentState) -> List[RouteDecision]:
    """从 state 取出本轮需要执行的 RouteDecision 列表 (复合 / 单路 / retry)。"""
    subs = state.get("subquery_decisions") or []
    if subs:
        return subs
    decision = state.get("decision")
    if decision:
        return [decision]
    return []


def _dispatch_retrieval_tasks(
    decision: RouteDecision,
    query: str,
    cid: str,
    summary_r: SummaryRetriever,
    local_r: ProgressiveLocalRetriever,
    metadata_r: EnhancedMetadataRetriever,
    time_filter: Optional[str],
    max_workers: int = 4,
    *,
    summary_top_docs: int = 5,
    summary_per_query_k: int = 5,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """对单个 RouteDecision 提交并收集多路径检索任务。"""
    route_results: Dict[str, Any] = {}
    route_errors: Dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        tasks: Dict[str, Any] = {}

        if decision.has(ROUTE_SUMMARY):
            tasks[ROUTE_SUMMARY] = ex.submit(
                summary_r.retrieve,
                decision.get_rewrite(ROUTE_SUMMARY, query),
                summary_top_docs, summary_per_query_k, time_filter,
            )

        if decision.has(ROUTE_PROGRESSIVE):
            tasks[ROUTE_PROGRESSIVE] = ex.submit(
                local_r.retrieve,
                decision.get_rewrite(ROUTE_PROGRESSIVE, query),
                5, 8, 5, 8, time_filter,
                chunk_type_for_route(ROUTE_PROGRESSIVE, decision),
                decision.retrieve_bias,
            )

        if decision.has(ROUTE_LOCAL):
            tasks[ROUTE_LOCAL] = ex.submit(
                local_r.retrieve_direct,
                decision.get_rewrite(ROUTE_LOCAL, query),
                decision.target_docs, 8, 5, 8, time_filter,
                chunk_type_for_route(ROUTE_LOCAL, decision),
                decision.retrieve_bias,
                target_doc_ids=list(decision.target_doc_ids or []),
            )

        if decision.has(ROUTE_METADATA):
            meta_ct = chunk_type_for_route(ROUTE_METADATA, decision)
            has_refs = bool(decision.fig_refs or decision.table_refs)
            has_filter = bool(
                decision.page_refs or decision.paragraph_refs or decision.entities
            )
            # metadata 按 doc_id 过滤; 优先用 registry 解析的 target_doc_ids[0],
            # 否则退到 target_docs[0] (历史行为).
            metadata_doc_id = (
                decision.target_doc_ids[0]
                if decision.target_doc_ids
                else (decision.target_docs[0] if decision.target_docs else None)
            )

            if has_filter:
                tasks[ROUTE_METADATA] = ex.submit(
                    metadata_r.retrieve,
                    [], decision.fig_refs, decision.table_refs,
                    8, 200, time_filter, meta_ct, metadata_doc_id,
                    decision.page_refs, decision.paragraph_refs, decision.entities,
                )
            elif has_refs:
                tasks[ROUTE_METADATA] = ex.submit(
                    metadata_r.retrieve_by_refs,
                    decision.fig_refs, decision.table_refs, 8, 50,
                    time_filter, meta_ct, doc_id=metadata_doc_id,
                )
            else:
                logger.warning(
                    f"[{cid}] [metadata] 无 filters 且无 refs, 跳过 "
                    f"(硬约束: metadata 必须有 filters)"
                )

        for route, fut in tasks.items():
            try:
                route_results[route] = fut.result()
            except Exception as e:
                logger.warning(f"[{cid}] [{route}] 路径失败: {e}")
                route_errors[route] = str(e)
                if route in (ROUTE_PROGRESSIVE, ROUTE_LOCAL):
                    route_results[route] = LocalRetrieveResult()
                else:
                    route_results[route] = []

    return route_results, route_errors


def _execute_single_retrieval(
    decision: RouteDecision,
    query: str,
    cid: str,
    summary_r: SummaryRetriever,
    local_r: ProgressiveLocalRetriever,
    metadata_r: EnhancedMetadataRetriever,
    max_workers: int = 4,
    *,
    subquery_id: Optional[str] = None,
    subquery_rewrite: str = "",
    summary_top_docs: int = 5,
    summary_per_query_k: int = 5,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """对单个 RouteDecision 执行多路径检索, 返回 (route_results, route_errors)。"""
    time_filter = decision.to_time_filter()
    route_ct = describe_route_chunk_types(decision)
    if route_ct:
        logger.info(
            f"[{cid}] [retrieve] chunk_type decision={decision.chunk_type!r} "
            f"per_route={route_ct}"
        )

    route_results, route_errors = _dispatch_retrieval_tasks(
        decision, query, cid, summary_r, local_r, metadata_r,
        time_filter, max_workers,
        summary_top_docs=summary_top_docs,
        summary_per_query_k=summary_per_query_k,
    )

    # time 误提取时: 带 time 过滤零命中 → 降级为全量文献检索
    if time_filter and _count_hits(route_results) == 0:
        logger.info(
            f"[{cid}] [retrieve] time={decision.time!r} 过滤后零命中, "
            f"降级为全量文献检索 (忽略 time 条件)"
        )
        route_results, fallback_errors = _dispatch_retrieval_tasks(
            decision, query, cid, summary_r, local_r, metadata_r,
            None, max_workers,
            summary_top_docs=summary_top_docs,
            summary_per_query_k=summary_per_query_k,
        )
        for route, err in fallback_errors.items():
            route_errors.setdefault(route, err)

    if subquery_id:
        rewrite = subquery_rewrite or _subquery_rerank_query(decision, query)
        route_results = _stamp_subquery_on_route_results(
            route_results, subquery_id, rewrite,
        )
        route_results = _prefix_route_result_keys(route_results, subquery_id)
        route_errors = _prefix_route_error_keys(route_errors, subquery_id)

    return route_results, route_errors


def _prewarm_decisions_embeddings(
    summary_r: SummaryRetriever,
    decisions: List[RouteDecision],
    query: str,
) -> None:
    """LangGraph 版的 batch-embed 预热: 把本轮所有 decision 的 rewrite 收齐一次性 embed.

    与 ``AgenticRAGPipeline._prewarm_query_embeddings`` 同等效果, 但支持复合查询
    (多个 RouteDecision). 失败时 logger.warning, 不阻塞后续单条 embed fallback.
    """
    embedder = getattr(getattr(summary_r, "vec", None), "embedder", None)
    if embedder is None:
        return
    if hasattr(embedder, "begin_request"):
        embedder.begin_request()

    enabled = bool(getattr(embedder, "query_instruct_enabled", True))
    instructs = getattr(embedder, "query_instructs", None)
    texts = collect_prewarm_embed_texts(
        decisions, query, enabled=enabled, instructs=instructs,
    )
    if not texts:
        return
    try:
        vecs = embedder.embed_batch(texts)
    except Exception as e:
        logger.warning(f"[prewarm] batch embed 失败 (将逐路 fallback): {e}")
        return
    for txt, vec in zip(texts, vecs):
        if hasattr(embedder, "_cache_put"):
            embedder._cache_put(txt, vec)


def _make_retrieve_node(
    summary_r: SummaryRetriever,
    local_r: ProgressiveLocalRetriever,
    metadata_r: EnhancedMetadataRetriever,
    max_workers: int = 4,
    neighbor_expander: Optional[Any] = None,
    *,
    summary_top_docs: int = 5,
    summary_per_query_k: int = 5,
) -> callable:
    def retrieve_node(state: AgentState) -> AgentState:
        t0 = time.time()
        cid = state.get("correlation_id", "?")
        decisions = _collect_decisions_to_run(state)
        if not decisions:
            logger.warning(f"[{cid}] [retrieve] 无路由决策, 跳过")
            state.setdefault("node_timings", {})["retrieve"] = time.time() - t0
            return state

        query = state["query"]
        # 预 batch-embed: 把本轮所有 rewrite 一次性灌入 EmbeddingClient LRU,
        # 让下面 _execute_single_retrieval 里多次 vec.retrieve(query) 直接命中.
        _prewarm_decisions_embeddings(summary_r, decisions, query)

        existing_results = state.get("route_results") or {}
        existing_errors = state.get("route_errors") or {}
        route_results = dict(existing_results)
        route_errors = dict(existing_errors)

        for i, decision in enumerate(decisions):
            sub_label = f"sub{i + 1}/{len(decisions)}" if len(decisions) > 1 else "single"
            logger.info(
                f"[{cid}] [retrieve] {sub_label}: routes={decision.routes} "
                f"rewrites={dict(decision.rewrites or {})} "
                f"rerank_mode={decision.rerank_mode!r}"
            )
            sub_id = _subquery_id(i) if len(decisions) > 1 else None
            sub_rewrite = _subquery_rerank_query(decision, query) if sub_id else ""
            sub_results, sub_errors = _execute_single_retrieval(
                decision, query, cid, summary_r, local_r, metadata_r, max_workers,
                subquery_id=sub_id,
                subquery_rewrite=sub_rewrite,
                summary_top_docs=summary_top_docs,
                summary_per_query_k=summary_per_query_k,
            )
            route_results = _merge_route_results(route_results, sub_results)
            route_errors = _merge_route_errors(route_errors, sub_errors)

        # 邻域扩展 (依赖图谱场景): 任一 decision 显式要求时才跑
        if neighbor_expander is not None:
            from .neighbor_expansion import (
                apply_neighbor_expansion,
                collect_expand_modes,
            )
            expand_modes = collect_expand_modes(decisions)
            if expand_modes:
                route_results = apply_neighbor_expansion(
                    route_results,
                    modes=expand_modes,
                    expander=neighbor_expander,
                    cid=cid,
                )

        state["route_results"] = route_results
        state["route_errors"] = route_errors
        # 新一轮 retrieve 后清除 reranker 快照, 避免 restore 用到过期数据
        state.pop("route_results_pre_rerank", None)

        # 累积本轮命中的文献 (issue #1 修订): this_round_docs 仅在本轮内使用,
        # 反思器可引用; 本轮结束后会作为下一轮的 last_round_docs 写回 session_meta
        docs = _update_this_round_docs(state, route_results)
        if docs:
            logger.info(
                f"[{cid}] [retrieve] this_round_docs 更新: 累计 {len(docs)} 篇 -> "
                + "; ".join(f"#{i + 1} {d['doc_name']}" for i, d in enumerate(docs[:5]))
                + (" ..." if len(docs) > 5 else "")
            )

        state.setdefault("node_timings", {})["retrieve"] = time.time() - t0

        # 打印每条路径的详细检索得分
        summary_parts: List[str] = []
        for r, v in route_results.items():
            if isinstance(v, LocalRetrieveResult):
                n = len(v.chunk_hits)
                summary_parts.append(f"{r}={n}")
                for i, h in enumerate(v.chunk_hits):
                    logger.info(
                        f"[{cid}] [retrieve] [{r}] hit#{i} "
                        f"emb_score={h.score:.4f} rrf={h.rrf_score:.4f} "
                        f"sources={h.sources} type={h.type} "
                        f"doc={h.doc_name or h.doc_id} page={h.page_start + 1} "
                        f"section={h.section} "
                        f"content={h.content[:80].replace(chr(10), ' ')}..."
                    )
            elif isinstance(v, list):
                n = len(v)
                summary_parts.append(f"{r}={n}")
                for i, h in enumerate(v):
                    if isinstance(h, Hit):
                        logger.info(
                            f"[{cid}] [retrieve] [{r}] hit#{i} "
                            f"emb_score={h.score:.4f} rrf={h.rrf_score:.4f} "
                            f"sources={h.sources} type={h.type} "
                            f"doc={h.doc_name or h.doc_id} page={h.page_start + 1} "
                            f"section={h.section} "
                            f"content={h.content[:80].replace(chr(10), ' ')}..."
                        )
        logger.info(f"[{cid}] [retrieve] " + " | ".join(summary_parts))
        state["agent_phase"] = "retrieve"
        return state
    return retrieve_node


def _hit_dedupe_key(hit: Hit) -> str:
    return hit.pk or hit.chunk_id or ""


def _top_hits_by_retrieval_source(
    route_indices: List[int],
    all_hits: List[Tuple[str, Hit]],
    source: str,
    k: int = 2,
) -> List[Hit]:
    """按 retrieve 原始分 (emb/BM25) 取某路 top-k, 仅含 sources 带该路的 hit。"""
    pool: List[Tuple[float, Hit]] = []
    for idx in route_indices:
        _, hit = all_hits[idx]
        if source in hit.sources:
            pool.append((float(hit.score), hit))
    pool.sort(key=lambda x: -x[0])
    out: List[Hit] = []
    seen: set[str] = set()
    for _, hit in pool:
        key = _hit_dedupe_key(hit)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(hit)
        if len(out) >= k:
            break
    return out


def _merge_rerank_top_with_retrieval_paths(
    route_top: List[Tuple[int, Hit, float]],
    route_indices: List[int],
    all_hits: List[Tuple[str, Hit]],
    *,
    retrieval_top_k: int = 2,
) -> List[Hit]:
    """rerank top-k ∪ bm25 top2 ∪ vector top2, 按 pk 去重; rerank 序优先。"""
    merged: List[Hit] = []
    seen: set[str] = set()

    def _add(hit: Hit) -> None:
        key = _hit_dedupe_key(hit)
        if not key or key in seen:
            return
        seen.add(key)
        merged.append(hit)

    for _, hit, _ in route_top:
        _add(hit)
    for hit in _top_hits_by_retrieval_source(
        route_indices, all_hits, "bm25", retrieval_top_k,
    ):
        _add(hit)
    for hit in _top_hits_by_retrieval_source(
        route_indices, all_hits, "vector", retrieval_top_k,
    ):
        _add(hit)
    return merged


def _make_reranker_node(
    reranker_client: RerankerClient,
    top_k: int = 5,
    quality_k: int = 3,
    quality_threshold: float = 0.5,
    diagnosis_config: Optional[Any] = None,
    quality_threshold_by_type: Optional[Dict[str, float]] = None,
    route_thresholds: Optional[RouteThresholds] = None,
    fail_open_min_emb_quality: Optional[float] = None,
) -> callable:
    """构建 reranker 节点: 按路径分别重排序 + 质量门控, 再汇总供 reflect。

    每条 route 的 rankable hit 独立 rerank 取 top_k; metadata/结构化 chunk 全量保留。

    质量门控改动 (本次优化):
      P0 #1: quality_score 改为 max+mean 加权混合, 不被弱路径稀释
      P0 #3: 零召回也触发 needs_retry, 走 progressive 兜底 + reflect
      P0 #10: 区分 reranker 未评分 (None) vs 评 0 分 (0.0), 不污染统计
      P1 #5: 按路径主导 chunk_type 选阈值 (image/table/references 用更低阈值)
      P1 #14: rerank_score 写入 Hit.rerank_score, 供 reflect/context 使用
      P1 #17: reranker API 失败时 fail-open: 跳过门控, 用 emb_score 排序
      P0.2: 把 exempt 拆为 (skip_topk_truncation, skip_quality_scoring):
            - metadata + entity-only: topk_only (保留全量但参与评分, 修复 #4)
            - 混合路径里 image/table: 不豁免, 参与评分 (修复 #5)
            - per-type 阈值 image/table/references 真正生效 (修复 #1)
      保送: 每路径 rerank top-k 后再并入 bm25/vector 各 retrieve top-2 (去重)
    """
    type_thresholds: Dict[str, float] = {
        (k or "").lower(): float(v)
        for k, v in (quality_threshold_by_type or {}).items()
    }
    default_threshold = float(quality_threshold)
    # P1.1: 统一阈值矩阵 (route × stage × type), 兜底吸收旧 quality_threshold_by_type
    thresholds = route_thresholds or RouteThresholds(
        default=default_threshold,
        by_type=type_thresholds,
        by_route={},
    )

    def _route_threshold_for(
        route: str,
        quality_pool: List[Tuple[int, Hit, float]],
    ) -> Tuple[float, str]:
        """按 (route, dominant_stage, dominant_chunk_type) 选阈值, 返回 (thresh, dom_type)。"""
        if not quality_pool:
            return thresholds.for_(route, None, None)[0], ""
        type_counts: Dict[str, int] = {}
        stage_counts: Dict[str, int] = {}
        for _, h, _ in quality_pool:
            t = (h.type or "").lower()
            if t:
                type_counts[t] = type_counts.get(t, 0) + 1
            s = (getattr(h, "stage", "") or "").lower()
            if s:
                stage_counts[s] = stage_counts.get(s, 0) + 1
        dom_type = max(type_counts.items(), key=lambda kv: kv[1])[0] if type_counts else ""
        dom_stage = max(stage_counts.items(), key=lambda kv: kv[1])[0] if stage_counts else ""
        thresh, _ = thresholds.for_(route, dom_stage, dom_type)
        return thresh, dom_type

    def reranker_node(state: AgentState) -> AgentState:
        t0 = time.time()
        cid = state.get("correlation_id", "?")
        query = state["query"]
        route_results = state.get("route_results", {})

        # 收集所有 hit 及其所属 route
        all_hits: List[Tuple[str, Hit]] = []
        for route, res in route_results.items():
            if isinstance(res, LocalRetrieveResult):
                for h in res.chunk_hits:
                    all_hits.append((route, h))
            elif isinstance(res, list):
                for h in res:
                    if isinstance(h, Hit):
                        all_hits.append((route, h))

        # P0 #3: 零召回不再是 "不重试": 一定走 reflect 兜底, 让上层改写后重试
        # P0-2 修订: 复合查询零召回时 per-sub 兜底, 不覆盖原 multi 结构
        if not all_hits:
            from .rerank_diagnosis import _extract_retry_keywords

            existing_subs = state.get("subquery_decisions") or []
            multi_decision = state.get("multi_decision")
            is_compound = (
                MultiRouteDecision is not None
                and multi_decision is not None
                and isinstance(multi_decision, MultiRouteDecision)
                and len(existing_subs) > 1
            )

            if is_compound:
                fallback_subs: List[RouteDecision] = []
                for sub in existing_subs:
                    sub_kw = ""
                    if isinstance(sub, RouteDecision):
                        for route in sub.routes:
                            rw = (sub.rewrites or {}).get(route)
                            if rw:
                                sub_kw = rw
                                break
                    sub_query = sub_kw or query
                    fallback_subs.append(RouteDecision(
                        routes=[ROUTE_PROGRESSIVE],
                        rewrites={
                            ROUTE_PROGRESSIVE: _extract_retry_keywords(sub_query),
                        },
                        chunk_type=None,
                        time="",
                        reasoning="(rerank-zero-recall-per-sub)",
                    ))
                state["subquery_decisions"] = fallback_subs
                state["rewrite_hint"] = _merge_route_decisions(fallback_subs)
                fallback_summary = (
                    f"multi {len(fallback_subs)} subs → progressive "
                    f"(per-sub 放宽 chunk_type/time)"
                )
            else:
                fallback = RouteDecision(
                    routes=[ROUTE_PROGRESSIVE],
                    rewrites={ROUTE_PROGRESSIVE: _extract_retry_keywords(query)},
                    chunk_type=None,
                    time="",
                    reasoning="(rerank-zero-recall)",
                )
                state["subquery_decisions"] = [fallback]
                state["rewrite_hint"] = fallback
                fallback_summary = f"single progressive routes={fallback.routes}"

            state["needs_retry"] = True
            state["rerank_diagnosis_summary"] = (
                f"[Reranker 诊断] cause=zero_recall confidence=0.40\n"
                f"  检索零命中, 兜底改写为 {fallback_summary}\n"
                f"  说明: reflect 可覆盖此建议。"
            )
            state["rerank_diagnosis_cause"] = "zero_recall"
            state["rerank_diagnosis_confidence"] = 0.40
            state["rerank_skip_reflect"] = False  # 低置信兜底: 必须 reflect
            state["reranker_score"] = 0.0
            state["agent_phase"] = "reranker"
            state.setdefault("node_timings", {})["reranker"] = time.time() - t0
            logger.warning(
                f"[{cid}] [reranker] 零召回 → needs_retry=true, fallback={fallback_summary}, "
                f"走 reflect (skip_reflect=False)"
            )
            return state

        # 过滤前快照: reflect 判定 OK 时可恢复被 threshold 裁掉的 chunk
        state["route_results_pre_rerank"] = _copy_route_results(route_results)

        decision = state.get("decision")
        subquery_decisions = state.get("subquery_decisions") or []

        # P0.2 (2026-05): 拆 exempt 为 (skip_topk_truncation, skip_quality_scoring)
        #   - exempt_full:  保留全量 + 不评分 (高置信结构化命中)
        #   - exempt_topk_only_indices: 保留全量 + 仍参与评分 (修复 #4 metadata 无质量兜底)
        #   - rankable:     完全参与 (可截断 + 计入评分)
        exempt_full_hits: List[Tuple[str, Hit]] = []
        exempt_full_indices: set[int] = set()
        exempt_topk_only_indices: set[int] = set()
        rankable_indices: List[int] = []
        for i, (route, hit) in enumerate(all_hits):
            edec = hit_exempt_decision(route, hit, decision)
            if edec.skip_topk_truncation and edec.skip_quality_scoring:
                exempt_full_hits.append((route, hit))
                exempt_full_indices.add(i)
            elif edec.skip_topk_truncation:
                exempt_topk_only_indices.add(i)
            else:
                rankable_indices.append(i)

        # exempt_indices 保留供日志兼容: 任何不参与截断的 hit
        exempt_indices: set[int] = exempt_full_indices | exempt_topk_only_indices

        if exempt_full_hits or exempt_topk_only_indices:
            logger.info(
                f"[{cid}] [reranker] exempt 分类: "
                f"full={len(exempt_full_hits)} (不评分+不截断), "
                f"topk_only={len(exempt_topk_only_indices)} (评分+不截断), "
                f"rankable={len(rankable_indices)} (评分+可截断)"
            )

        # P1 #17: reranker API 调用包 try/except — 失败时退化为不 rerank
        # (此时 reranker_client.rerank() 在 fail_open=True 时已返回 [])
        rerank_failed = False
        score_map: Dict[int, Optional[float]] = {i: None for i in range(len(all_hits))}

        # R6: 按 synthesize 后的 rerank query 分组 (单路/复合统一, 泛化发话 fallback rewrite)
        # P0.2: full-exempt 不送 rerank, 省调用; topk_only + rankable 都送
        rerank_groups: Dict[str, List[int]] = {}
        for i, (_, hit) in enumerate(all_hits):
            if i in exempt_full_indices:
                continue
            rq = _resolve_rerank_query_for_hit(
                hit, query, decision, subquery_decisions,
            )
            rerank_groups.setdefault(rq, []).append(i)

        for rq, indices in rerank_groups.items():
            sample_hit = all_hits[indices[0]][1] if indices else None
            sub_dec = (
                _decision_for_subquery_id(
                    getattr(sample_hit, "subquery_id", "") or "",
                    subquery_decisions,
                )
                if sample_hit is not None
                else None
            )
            dec_for_mode = sub_dec if sub_dec is not None else decision
            use_rewrite = (
                isinstance(dec_for_mode, RouteDecision)
                and dec_for_mode.rerank_mode is True
            )
            rq_src = "rewrite_kw" if use_rewrite else "user_query"
            logger.info(
                f"[{cid}] [reranker] rerank_query group n_hits={len(indices)} "
                f"rerank_mode={getattr(dec_for_mode, 'rerank_mode', None)!r} "
                f"rerank_query_src={rq_src} q={rq[:160]!r}"
            )

        try:
            for rq, indices in rerank_groups.items():
                documents = [
                    compose_rerank_document(all_hits[i][1])
                    for i in indices
                ]
                group_results = reranker_client.rerank(rq, documents, top_k=len(documents))
                for r in group_results:
                    if 0 <= r.index < len(indices):
                        score_map[indices[r.index]] = float(r.score)
        except Exception as e:  # pragma: no cover — fail_open=True 时不会到这
            rerank_failed = True
            logger.error(
                f"[{cid}] [reranker] API 调用异常 (fail_open=False?), 降级为不 rerank: {e}"
            )
        if not rerank_failed and not any(v is not None for v in score_map.values()):
            # score_map 全 None: 两种成因, 分开记录避免误导 —
            #   1. rerank_groups 为空: 本轮所有 hit 都被豁免 (如 chunk_type=image/table
            #      的结构化召回), 根本没调用 rerank API — 属正常, 按 emb_score 保留即可;
            #   2. rerank_groups 非空但 API 全返空: 才是真正的 rerank API 异常。
            rerank_failed = True
            if not rerank_groups:
                logger.info(
                    f"[{cid}] [reranker] 全部 {len(all_hits)} 条 hit 豁免 rerank "
                    f"(结构化/指定类型命中), 跳过打分, 按 emb_score 保留"
                )
            else:
                logger.warning(
                    f"[{cid}] [reranker] API 返回空, 降级为不 rerank (按 emb_score 排序)"
                )

        # P1 #14: rerank_score 写回 Hit, 供下游 reflect/context 使用
        for i, (_, hit) in enumerate(all_hits):
            hit.rerank_score = score_map.get(i)

        # 打印每个 hit 的 emb_score + rerank_score 对照
        for i, (route, hit) in enumerate(all_hits):
            rs = score_map.get(i)
            rs_str = f"{rs:.4f}" if rs is not None else "None"
            exempt_tag = " exempt" if i in exempt_indices else ""
            logger.info(
                f"[{cid}] [reranker] [{route}] hit#{i}{exempt_tag} "
                f"emb_score={hit.score:.4f} rerank_score={rs_str} "
                f"type={hit.type} doc={hit.doc_name or hit.doc_id} "
                f"page={hit.page_start + 1} "
                f"content={hit.content[:80].replace(chr(10), ' ')}..."
            )

        # ── reranker 失败 fail-open 路径: 不做门控, 直接用 retrieve 原结果 ──
        if rerank_failed:
            # P2.3 (#10 修复): 若配置了 fail_open_min_emb_quality, 检查 emb_score 安全网
            #   - top-N (取 quality_k) emb_score 平均 < 阈值 → 强制 needs_retry
            #   - 否则保持旧 fail-open 行为 (全放行)
            emb_safety_triggered = False
            emb_quality: Optional[float] = None
            if fail_open_min_emb_quality is not None and all_hits:
                emb_scores = sorted(
                    (h.score for _, h in all_hits if h.score and h.score > 0),
                    reverse=True,
                )
                if emb_scores:
                    emb_quality = sum(emb_scores[:quality_k]) / min(len(emb_scores), quality_k)
                    if emb_quality < fail_open_min_emb_quality:
                        emb_safety_triggered = True

            if emb_safety_triggered:
                # 兜底改写: progressive + 关键词 + 放宽过滤
                from .rerank_diagnosis import _extract_retry_keywords
                fallback = RouteDecision(
                    routes=[ROUTE_PROGRESSIVE],
                    rewrites={ROUTE_PROGRESSIVE: _extract_retry_keywords(query)},
                    chunk_type=None,
                    time="",
                    reasoning="(rerank-failed-emb-quality-low)",
                )
                state["needs_retry"] = True
                state["rewrite_hint"] = fallback
                state["subquery_decisions"] = [fallback]
                state["rerank_skip_reflect"] = False  # 失败兜底必须走 reflect
                state["rerank_diagnosis_cause"] = "rerank_failed_low_emb"
                state["rerank_diagnosis_confidence"] = 0.35
                state["rerank_diagnosis_summary"] = (
                    f"[Reranker 诊断] cause=rerank_failed_low_emb confidence=0.35\n"
                    f"  rerank API 失败 + top-{quality_k} emb_score 平均 "
                    f"{emb_quality:.4f} < 安全网 {fail_open_min_emb_quality:.4f}\n"
                    f"  兜底改写: progressive + 放宽过滤"
                )
                state["reranker_score"] = -1.0  # sentinel
                state.pop("route_results_pre_rerank", None)
                # 关键: 必须标记 phase=reranker, 否则 policy 仍读到 phase=retrieve,
                # 会无限路由回 reranker (fail-open 死循环 → 撞递归上限)。
                state["agent_phase"] = "reranker"
                state.setdefault("node_timings", {})["reranker"] = time.time() - t0
                logger.warning(
                    f"[{cid}] [reranker] fail-open 安全网触发: "
                    f"emb_quality={emb_quality:.4f} < {fail_open_min_emb_quality:.4f} "
                    f"→ needs_retry=true, fallback=progressive, 走 reflect"
                )
                return state

            state["needs_retry"] = False
            state["rewrite_hint"] = None
            state["rerank_skip_reflect"] = False
            state["rerank_diagnosis_cause"] = "rerank_failed"
            state["rerank_diagnosis_confidence"] = 0.0
            state["rerank_diagnosis_summary"] = ""
            state["reranker_score"] = -1.0  # sentinel: 未测量
            # route_results 保留原样, 丢掉快照 (没有可恢复内容)
            state.pop("route_results_pre_rerank", None)
            # 关键: 必须标记 phase=reranker, 否则 policy 仍读到 phase=retrieve,
            # 会无限路由回 reranker (fail-open 死循环 → 撞递归上限)。
            state["agent_phase"] = "reranker"
            state.setdefault("node_timings", {})["reranker"] = time.time() - t0
            safety_note = (
                f" emb_quality={emb_quality:.4f} ≥ {fail_open_min_emb_quality:.4f}, "
                if emb_quality is not None else ""
            )
            logger.warning(
                f"[{cid}] [reranker] fail-open: 跳过门控,{safety_note}"
                f"保留 {len(all_hits)} 条 retrieve 结果 直接进入 context_build"
            )
            return state

        # 可排序 hit: 每条 route 独立取 top_k; 结构化/metadata 全量并入
        # P0 #10: score is None (未评分) → 丢弃; 0.0 (评不相关) → 保留但排在底部
        # P0.2: 同时收集 topk_only (metadata entity-only) hits — 也参与评分但不截断
        rankable_by_route: Dict[str, List[Tuple[int, Hit, float]]] = {}
        topk_only_by_route: Dict[str, List[Tuple[int, Hit, float]]] = {}
        for idx in rankable_indices:
            route, hit = all_hits[idx]
            score = score_map.get(idx)
            if score is None:
                continue
            rankable_by_route.setdefault(route, []).append((idx, hit, float(score)))
        for idx in exempt_topk_only_indices:
            route, hit = all_hits[idx]
            score = score_map.get(idx)
            if score is None:
                continue
            topk_only_by_route.setdefault(route, []).append((idx, hit, float(score)))

        filtered_by_route: Dict[str, List[Hit]] = {}
        # full-exempt hits: 保留, 不评分
        for route, hit in exempt_full_hits:
            filtered_by_route.setdefault(route, []).append(hit)

        per_route_top: List[Tuple[str, Hit, float]] = []
        route_quality_avgs: List[float] = []
        # P1 #5: per-route gate
        route_gate_decisions: List[Tuple[str, float, float, str, bool]] = []
        all_routes = sorted(set(rankable_by_route) | set(topk_only_by_route))
        for route in all_routes:
            items = list(rankable_by_route.get(route, []))
            items.sort(key=lambda x: x[2], reverse=True)
            route_top = items[:top_k]
            route_rankable_indices = [
                idx for idx in rankable_indices if all_hits[idx][0] == route
            ]
            merged_hits = _merge_rerank_top_with_retrieval_paths(
                route_top, route_rankable_indices, all_hits, retrieval_top_k=2,
            )
            rescue_n = max(0, len(merged_hits) - len(route_top))
            per_route_top.extend(
                (route, hit, float(score_map.get(idx) or 0.0))
                for idx, hit, _ in route_top
            )
            for hit in merged_hits[len(route_top):]:
                rs = hit.rerank_score
                per_route_top.append(
                    (route, hit, float(rs) if rs is not None else 0.0),
                )
            for hit in merged_hits:
                filtered_by_route.setdefault(route, []).append(hit)

            # P0.2: topk_only hits 按 rerank score 排序后全量并入, 不参与截断
            topk_only_items = sorted(
                topk_only_by_route.get(route, []),
                key=lambda x: x[2], reverse=True,
            )
            for _, hit, _ in topk_only_items:
                filtered_by_route.setdefault(route, []).append(hit)

            if rescue_n:
                logger.info(
                    f"[{cid}] [reranker] [{route}] retrieve 保送 +{rescue_n} "
                    f"(bm25 top2 + vector top2 去重并入)"
                )

            # P0.2: quality_pool 用全量 rankable + topk_only, 不用被截断的 route_top
            # 否则 top_k=2 时只看 2 条最高分, 错过 #5 修复的混合污染信号
            # (e.g., 1 text 高分 + 5 image 低分, route_top=[text, img] 看不出 image 拖累)
            quality_pool = sorted(
                items + topk_only_items,
                key=lambda x: x[2], reverse=True,
            )[:quality_k]
            if quality_pool:
                route_scores = [s for _, _, s in quality_pool]
                route_avg = sum(route_scores) / len(route_scores)
                route_quality_avgs.append(route_avg)
                thresh, dom_type = _route_threshold_for(route, quality_pool)
                passes = route_avg >= thresh
                route_gate_decisions.append((route, route_avg, thresh, dom_type, passes))

        logger.info(
            f"[{cid}] [reranker] 分路径 top-{top_k}: "
            + " | ".join(
                f"[{route}] {hit.chunk_id[:8]}.. s={score:.4f}"
                for route, hit, score in per_route_top
            )
            + (f" | exempt_full={len(exempt_full_hits)}" if exempt_full_hits else "")
            + (f" | topk_only={len(exempt_topk_only_indices)}" if exempt_topk_only_indices else "")
        )

        filtered_route_results: Dict[str, Any] = {}
        for route, res in route_results.items():
            hits = filtered_by_route.get(route, [])
            if isinstance(res, LocalRetrieveResult):
                filtered_route_results[route] = LocalRetrieveResult(
                    candidate_docs=res.candidate_docs,
                    chunk_hits=hits,
                )
            elif isinstance(res, list):
                filtered_route_results[route] = hits

        # P0 #1: quality_score 用 0.7*max + 0.3*mean 混合, 一条好路径不再被弱路径稀释
        # P1 #5: 同时保留 per-route gate, 任一路径过自己的阈值即视为通过
        # P0.2: exempt_rescue 只在 ALL 路径全为 full-exempt 时触发 (问题 #4 不再依赖 rescue)
        exempt_rescue = False
        if route_quality_avgs:
            top_scores = list(route_quality_avgs)
            if len(route_quality_avgs) == 1:
                quality_score = route_quality_avgs[0]
            else:
                qmax = max(route_quality_avgs)
                qmean = sum(route_quality_avgs) / len(route_quality_avgs)
                quality_score = 0.7 * qmax + 0.3 * qmean
        elif exempt_full_hits:
            # exempt-full-only: 没有任何可评分 hit (e.g., metadata+fig_refs 命中); 兜底放行
            top_scores = []
            quality_score = default_threshold
            exempt_rescue = True
        else:
            top_scores = []
            quality_score = 0.0
        state["reranker_score"] = quality_score

        # 决定门控是否通过: 任一路径 per-route gate 通过, 或 exempt rescue 启用
        any_route_passes = any(passes for *_, passes in route_gate_decisions)
        passes_gate = any_route_passes or exempt_rescue

        if route_gate_decisions:
            gate_lines = ", ".join(
                f"{r}={avg:.3f}/{th:.3f}({dom or '?'}){'✓' if ok else '✗'}"
                for r, avg, th, dom, ok in route_gate_decisions
            )
        else:
            gate_lines = "(no rankable route)"
        logger.info(
            f"[{cid}] [reranker] per-route gate: {gate_lines} "
            f"| blended={quality_score:.4f} (default_threshold={default_threshold:.3f}) "
            f"| exempt_rescue={exempt_rescue} | passes_gate={passes_gate}"
        )

        # 门控判定: 改用 passes_gate (而不是 quality_score < quality_threshold)
        if not passes_gate:
            state["needs_retry"] = True
            from .rerank_diagnosis import (
                RerankDiagnosisConfig,
                diagnose_rerank_failure,
            )

            diag_cfg = (
                diagnosis_config
                if diagnosis_config is not None
                else RerankDiagnosisConfig(enabled=False)
            )
            if diag_cfg.enabled:
                # P2.2: 把 rerank_groups (List[int] per group_key) 翻转为 {idx: group_key},
                # 供 diagnose 做 z-score 跨子查询归一化
                idx_to_group: Dict[int, str] = {}
                for gid, idx_list in rerank_groups.items():
                    for ix in idx_list:
                        idx_to_group[ix] = gid
                diagnosis = diagnose_rerank_failure(
                    query=query,
                    decision=state.get("decision"),
                    all_hits=all_hits,
                    score_map=score_map,
                    quality_score=quality_score,
                    quality_threshold=default_threshold,
                    this_round_docs=state.get("this_round_docs") or [],
                    config=diag_cfg,
                    route_thresholds=thresholds,  # P1.2: 同源阈值
                    rerank_groups=idx_to_group,   # P2.2: 跨 group 归一化
                )
                state["rewrite_hint"] = diagnosis.suggested
                state["subquery_decisions"] = [diagnosis.suggested]
                state["rerank_diagnosis_summary"] = diagnosis.summary
                state["rerank_skip_reflect"] = diagnosis.skip_reflect
                state["rerank_diagnosis_cause"] = diagnosis.cause
                state["rerank_diagnosis_confidence"] = diagnosis.confidence
                state["agent_phase"] = "reranker"
                action = (
                    "rewrite (skip reflect)"
                    if diagnosis.skip_reflect
                    else "reflect"
                )
                skip_cmp = (
                    "pass"
                    if diagnosis.confidence >= diag_cfg.skip_reflect_confidence
                    else "fail"
                )
                logger.info(
                    f"[{cid}] [reranker] diagnosis cause={diagnosis.cause} "
                    f"diag_confidence={diagnosis.confidence:.2f} "
                    f"skip_reflect_threshold={diag_cfg.skip_reflect_confidence:.2f} "
                    f"({skip_cmp}) "
                    f"skip_causes={list(diag_cfg.skip_reflect_causes)} "
                    f"routes={diagnosis.suggested.routes} "
                    f"skip_reflect={diagnosis.skip_reflect} "
                    f"→ needs_retry=true, {action}, "
                    f"保留完整 {len(all_hits)} 条结果"
                )
            else:
                state["rewrite_hint"] = None
                state["rerank_diagnosis_summary"] = ""
                state["rerank_skip_reflect"] = False
                state["rerank_diagnosis_cause"] = ""
                state["rerank_diagnosis_confidence"] = 0.0
                state["agent_phase"] = "reranker"
                logger.info(
                    f"[{cid}] [reranker] gate fail, blended={quality_score:.4f}, "
                    f"diagnosis disabled, needs_retry=true → reflect"
                )
            # route_results 保持 pre_rerank 完整集, 不在此处覆盖
        else:
            state["needs_retry"] = False
            state["rewrite_hint"] = None
            state["rerank_skip_reflect"] = False
            state["rerank_diagnosis_cause"] = ""
            state["rerank_diagnosis_confidence"] = 0.0
            state["route_results"] = filtered_route_results
            state.pop("route_results_pre_rerank", None)
            state["agent_phase"] = "reranker"
            logger.info(
                f"[{cid}] [reranker] gate pass, blended={quality_score:.4f}, "
                f"采用 top-{top_k} 过滤结果, 跳过 reflect"
            )

        state.setdefault("node_timings", {})["reranker"] = time.time() - t0
        return state
    return reranker_node


def _make_reflect_node(
    reflection_llm: Optional[LLMClient],
    temperature: float = 0.0,
    max_tokens: int = 400,
    disable_thinking: Optional[bool] = None,
    routing_core: Optional[Any] = None,
    reflect_summary_config: Optional[ReflectSummaryConfig] = None,
) -> callable:
    """构建反思节点。

    Args:
        disable_thinking: 是否关闭"思考模式"。
            - None  (默认): 不向 chat() 传入该参数 -> 不下发 vLLM 专属的
              ``extra_body.chat_template_kwargs``, 也不追加 ``/no_think`` 文本。
              适用于调用阿里云 / DeepSeek / GPUGeek 等 OpenAI 兼容云平台。
            - True/False: 显式开关; 仅当反思 LLM 后端是 vLLM (即 LLMClient 构造
              时设了 ``disable_thinking_extra_body=True``) 时生效, 通过
              ``chat_template_kwargs.enable_thinking`` 控制思考开关。
        routing_core: 可选 RoutingCore. 注入时反思走 FC (ok/retry/partial 三选一);
            None 时走原 JSON 反思路径 (与历史行为一致).
    """
    # 反思 prompt 与 router 共享规则块, 仅在 reflect 端追加任务/输出格式 (issue #3)
    reflect_system_prompt = _reflect_system_prompt(datetime.datetime.now().year)
    backend = "fc" if routing_core is not None else "legacy"
    summary_cfg = reflect_summary_config or ReflectSummaryConfig()
    logger.info(f"[reflect] reflect_node backend={backend}")

    def _reflect_summary(route_results: Dict[str, Any]) -> Tuple[str, int]:
        return summarize_for_reflect(
            route_results,
            decision=None,
            config=summary_cfg,
        )

    def reflect_node(state: AgentState) -> AgentState:
        t0 = time.time()
        cid = state.get("correlation_id", "?")
        query = state["query"]
        decision = state.get("decision")
        route_results = state.get("route_results", {})
        incoming_needs_retry = bool(state.get("needs_retry"))

        max_retries = state.get("max_retries", 2)
        retry_count = state.get("retry_count", 0)

        logger.info(
            f"[{cid}] [reflect] begin: backend={backend} "
            f"retry={retry_count}/{max_retries}"
        )

        # ── FC 路径: 整段交给 routing_core.reflect, 自带关闸/快速路径/调用 ──
        if routing_core is not None:
            results_summary, total_hits = _reflect_summary(route_results)
            if total_hits == 0 and should_accept_structural_only_results(
                route_results, decision,
            ):
                state["needs_retry"] = False
                state["rewrite_hint"] = None
                state["sufficient"] = True
                state["evidence_gaps"] = []
                state["uncertainty_note"] = ""
                state["agent_phase"] = "reflect"
                state.setdefault("node_timings", {})["reflect"] = time.time() - t0
                logger.info(
                    f"[{cid}] [reflect] 纯结构化检索已有命中, 跳过 FC reflect/retry"
                )
                return state
            diag_block = (state.get("rerank_diagnosis_summary") or "").strip()
            if diag_block:
                results_summary = f"{results_summary}\n\n{diag_block}"
            try:
                verdict = routing_core.reflect(
                    query=query,
                    last_decision=_reflect_last_decision_for_fc(state, decision),
                    results_summary=results_summary,
                    total_hits=total_hits,
                    this_round_docs=state.get("this_round_docs") or [],
                    retry_count=retry_count,
                    max_retries=max_retries,
                    correlation_id=cid,
                )
            except Exception as e:
                _apply_reflect_failure_state(
                    state, cid, source="routing_core.reflect", error=e,
                )
                state["agent_phase"] = "reflect"
                state.setdefault("node_timings", {})["reflect"] = time.time() - t0
                return state

            state["needs_retry"] = bool(verdict.needs_retry)
            if verdict.needs_retry and verdict.decision is not None:
                if isinstance(verdict.decision, RouteDecision):
                    state["subquery_decisions"] = [verdict.decision]
                    state["rewrite_hint"] = verdict.decision
                    state["decision"] = verdict.decision
                elif MultiRouteDecision is not None and isinstance(
                    verdict.decision, MultiRouteDecision,
                ):
                    sub_decs = [s.decision for s in verdict.decision.subqueries]
                    logger.info(
                        f"[{cid}] [reflect] FC retry 返回 MultiRouteDecision "
                        f"(subs={len(sub_decs)}), 将全部子查询写入 rewrite"
                    )
                    state["multi_decision"] = verdict.decision
                    state["subquery_decisions"] = sub_decs
                    state["synth_hint"] = verdict.decision.synth_hint or ""
                    merged = _merge_route_decisions(sub_decs) if sub_decs else None
                    state["rewrite_hint"] = merged
                    state["decision"] = merged
                else:
                    logger.warning(
                        f"[{cid}] [reflect] FC retry 返回未知类型 "
                        f"{type(verdict.decision).__name__}, no_retry 兜底"
                    )
                    state["needs_retry"] = False
                    state["rewrite_hint"] = None
                    state["sufficient"] = True
                    state["evidence_gaps"] = []
                    state["uncertainty_note"] = ""
            else:
                state["rewrite_hint"] = None
                skip_reason = (verdict.meta or {}).get("skip_reason") if hasattr(verdict, "meta") else ""
                if incoming_needs_retry and skip_reason == "max_retries_exhausted":
                    state["sufficient"] = False
                    state["evidence_gaps"] = ["insufficient_evidence"]
                    state["uncertainty_note"] = "证据不足，且已用尽重试预算"
                    state["no_answer"] = True
                else:
                    state["sufficient"] = True
                    state["evidence_gaps"] = []
                    state["uncertainty_note"] = ""

            if verdict.needs_retry:
                state["sufficient"] = False
                state["evidence_gaps"] = ["insufficient_evidence"]
                state["uncertainty_note"] = "当前证据不足，准备改写重试"

            if verdict.partial:
                state["partial_note"] = verdict.partial_note
                if not verdict.needs_retry:
                    state["uncertainty_note"] = verdict.partial_note

            state["agent_phase"] = "reflect"

            state.setdefault("node_timings", {})["reflect"] = time.time() - t0
            return state

        # ── Legacy JSON 反思路径 (下方保留原逻辑不变) ──

        # 硬过滤: 反思被禁用或已用尽重试预算时直接跳过 (issue #4)
        # - max_retries == 0     -> 完全不反思
        # - retry_count >= max_retries -> 即使反思了也无法触发 rewrite, 直接节省一次 LLM 调用
        if max_retries <= 0 or reflection_llm is None or retry_count >= max_retries:
            state["needs_retry"] = False
            state["rewrite_hint"] = None
            if incoming_needs_retry and retry_count >= max_retries:
                state["sufficient"] = False
                state["evidence_gaps"] = ["insufficient_evidence"]
                state["uncertainty_note"] = "证据不足，且已用尽重试预算"
                state["no_answer"] = True
            else:
                state["sufficient"] = True
                state["evidence_gaps"] = []
                state["uncertainty_note"] = ""
            state["agent_phase"] = "reflect"
            state.setdefault("node_timings", {})["reflect"] = time.time() - t0
            if max_retries > 0 and retry_count >= max_retries and reflection_llm is not None:
                logger.info(
                    f"[{cid}] [reflect] 已用尽重试预算 (retry_count={retry_count}/"
                    f"max_retries={max_retries}), 跳过 LLM 反思"
                )
            return state

        # 快速判断: 无可评估语义结果时用默认策略重试
        results_summary, reflect_eligible = _reflect_summary(route_results)
        if reflect_eligible == 0 and should_accept_structural_only_results(
            route_results, decision,
        ):
            state["needs_retry"] = False
            state["rewrite_hint"] = None
            state["sufficient"] = True
            state["evidence_gaps"] = []
            state["uncertainty_note"] = ""
            state["agent_phase"] = "reflect"
            state.setdefault("node_timings", {})["reflect"] = time.time() - t0
            logger.info(
                f"[{cid}] [reflect] 纯结构化检索已有命中, 跳过默认 progressive 重试"
            )
            return state
        if reflect_eligible == 0:
            rb = (decision.retrieve_bias if decision else None) or infer_retrieve_bias_heuristic(
                query, chunk_type=decision.chunk_type if decision else None,
            )
            empty_decision = RouteDecision(
                routes=[ROUTE_PROGRESSIVE],
                rewrites={ROUTE_PROGRESSIVE: query},
                retrieve_bias=rb,
                reasoning="(reflect-empty)",
            )
            state["needs_retry"] = True
            state["rewrite_hint"] = empty_decision
            state["subquery_decisions"] = [empty_decision]
            state["sufficient"] = False
            state["evidence_gaps"] = ["no_semantic_results"]
            state["uncertainty_note"] = "无可评估语义结果，默认 progressive 重试"
            state["agent_phase"] = "reflect"
            state.setdefault("node_timings", {})["reflect"] = time.time() - t0
            logger.info(f"[{cid}] [reflect] 无可评估语义结果, 默认 progressive 重试")
            return state

        # 构建反思 prompt (legacy JSON)
        total = reflect_eligible
        sub_decs = state.get("subquery_decisions") or []
        routes_str, rewrites_str, filters_json = _format_reflect_strategy(decision, sub_decs)
        # 反思器看的是"本轮已检索到的文献" (区别于 router 看的"上一轮"),
        # 这样反思器若想 doc_refs 回指本轮已发现的文献, 编号严格 1-based 对齐
        registry_block = _format_doc_registry_block(
            state.get("this_round_docs") or [],
            label="本轮已检索到的文献列表",
        )
        diag_block = (state.get("rerank_diagnosis_summary") or "").strip()
        rerank_diagnosis_block = f"\n\n{diag_block}" if diag_block else ""

        user_msg = REFLECT_USER_TEMPLATE.format(
            query=query,
            doc_registry_block=registry_block,
            routes=routes_str,
            rewrites=rewrites_str,
            filters_json=filters_json,
            total_hits=total,
            results_summary=results_summary,
            rerank_diagnosis_block=rerank_diagnosis_block,
        )

        # 调用反思 LLM
        # disable_thinking 仅在显式配置时下发, 避免污染云平台请求体
        # (None -> 走 LLMClient 默认, 既不发 chat_template_kwargs 也不加 /no_think)
        chat_kwargs: Dict[str, Any] = {
            "system": reflect_system_prompt,
            "user": user_msg,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if disable_thinking is not None:
            chat_kwargs["disable_thinking"] = disable_thinking
        try:
            result = reflection_llm.chat(**chat_kwargs)
            raw = result.get("answer", "")
        except Exception as e:
            _apply_reflect_failure_state(
                state, cid, source="reflect LLM", error=e,
            )
            state["agent_phase"] = "reflect"
            state.setdefault("node_timings", {})["reflect"] = time.time() - t0
            return state

        # 解析 JSON
        blob = _extract_json(raw)
        if not blob:
            _apply_reflect_failure_state(
                state,
                cid,
                source="reflect JSON missing",
                error=f"raw={raw[:200]!r}",
            )
            state["agent_phase"] = "reflect"
            state.setdefault("node_timings", {})["reflect"] = time.time() - t0
            return state

        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError as e:
            _apply_reflect_failure_state(
                state, cid, source="reflect JSON decode", error=e,
            )
            state["agent_phase"] = "reflect"
            state.setdefault("node_timings", {})["reflect"] = time.time() - t0
            return state

        needs_retry = bool(parsed.get("needs_retry", False))

        if not needs_retry:
            state["needs_retry"] = False
            state["rewrite_hint"] = None
            state["sufficient"] = True
            state["evidence_gaps"] = []
            state["uncertainty_note"] = state.get("partial_note", "") or ""
            state["agent_phase"] = "reflect"
            state.setdefault("node_timings", {})["reflect"] = time.time() - t0
            logger.info(f"[{cid}] [reflect] needs_retry=false")
            return state

        # 解析为 RouteDecision (格式与 router 输出对齐)
        max_paths = 2
        if routing_core is not None and getattr(routing_core, "routing_limits", None):
            max_paths = routing_core.routing_limits.max_paths_per_sub
        new_routes = _normalize_routes(parsed.get("routes", []), max_paths=max_paths)
        if not new_routes:
            new_routes = [ROUTE_PROGRESSIVE]

        new_rewrites = _parse_rewrites(parsed.get("rewrites", {}), new_routes, query)
        # metadata 路径硬约束: 即使反思器写了 rewrite 也忽略 (issue #2)
        if new_rewrites.pop(ROUTE_METADATA, None):
            logger.info(f"[{cid}] [reflect] 忽略反思器为 metadata 输出的 rewrites")

        filters_in = parsed.get("filters") or {}
        target_docs = _parse_str_list(filters_in, "target_docs")
        target_doc_ids: List[str] = []
        # 反思器的 doc_refs 锚定"本轮已检索到的文献" (issue #1 修订);
        # 它看到的列表就是 this_round_docs, 所以编号查的也是同一份
        this_round = state.get("this_round_docs") or []
        for ref in _parse_int_list(filters_in, "doc_refs"):
            idx = ref - 1
            if 0 <= idx < len(this_round):
                entry = this_round[idx] if isinstance(this_round[idx], dict) else {}
                name = str(entry.get("doc_name") or entry.get("doc_id") or "")
                did = str(entry.get("doc_id") or "")
                if name and name not in target_docs:
                    target_docs.append(name)
                if did and did not in target_doc_ids:
                    target_doc_ids.append(did)
        # 若 base decision 已经有 target_doc_ids (router 已经锚定过, 比如代词自动锚定),
        # reflect 不应丢失 — 继承过来
        if decision and getattr(decision, "target_doc_ids", None):
            for did in decision.target_doc_ids:
                if did and did not in target_doc_ids:
                    target_doc_ids.append(did)

        fig_refs = _parse_str_list(filters_in, "fig_refs", upper=True)
        table_refs = _parse_str_list(filters_in, "table_refs", upper=True)
        page_refs = _parse_int_list(filters_in, "page_refs")
        paragraph_refs = _parse_int_list(filters_in, "paragraph_refs")
        entities = _parse_str_list(filters_in, "entities")
        rb_raw = parsed.get("retrieve_bias") or filters_in.get("retrieve_bias")
        retrieve_bias = normalize_retrieve_bias(rb_raw) or (
            decision.retrieve_bias if decision else None
        )

        # metadata 没有任何 filter 时强制踢出 (issue #2)
        if ROUTE_METADATA in new_routes and not (
            fig_refs or table_refs or page_refs or paragraph_refs or entities
        ):
            logger.info(f"[{cid}] [reflect] metadata 无 filter, 已剔除")
            new_routes = [r for r in new_routes if r != ROUTE_METADATA] or [ROUTE_PROGRESSIVE]
            if ROUTE_PROGRESSIVE in new_routes and ROUTE_PROGRESSIVE not in new_rewrites:
                new_rewrites[ROUTE_PROGRESSIVE] = query

        new_decision = RouteDecision(
            routes=new_routes,
            rewrites=new_rewrites,
            time=_parse_time(filters_in),
            chunk_type=_parse_chunk_type(filters_in),
            target_docs=target_docs,
            target_doc_ids=target_doc_ids,
            fig_refs=fig_refs,
            table_refs=table_refs,
            page_refs=page_refs,
            paragraph_refs=paragraph_refs,
            entities=entities,
            retrieve_bias=retrieve_bias,
            reasoning="(reflect-retry)",
        )

        state["needs_retry"] = True
        state["rewrite_hint"] = new_decision
        state["subquery_decisions"] = [new_decision]
        state["sufficient"] = False
        state["evidence_gaps"] = ["insufficient_evidence"]
        state["uncertainty_note"] = "当前证据不足，准备改写重试"
        state["agent_phase"] = "reflect"
        state.setdefault("node_timings", {})["reflect"] = time.time() - t0
        logger.info(
            f"[{cid}] [reflect] needs_retry=true, "
            f"routes={new_routes} rewrites={new_rewrites} "
            f"target_docs={target_docs} target_doc_ids={target_doc_ids}"
        )
        return state
    return reflect_node


def _make_policy_node() -> callable:
    """轻量 policy: 纯状态机, 不额外调用 LLM。"""

    def policy_node(state: AgentState) -> AgentState:
        t0 = time.time()
        cid = state.get("correlation_id", "?")

        phase = str(state.get("agent_phase") or "").strip()
        needs_clarify = bool(state.get("needs_clarify"))
        needs_reuse = bool(state.get("needs_reuse"))
        needs_retry = bool(state.get("needs_retry"))
        sufficient = bool(state.get("sufficient"))
        retry_count = int(state.get("retry_count") or 0)
        max_retries = int(state.get("max_retries") or 0)
        route_results = state.get("route_results") or {}
        route_errors = state.get("route_errors") or {}
        rewrite_hint = state.get("rewrite_hint")
        decision = state.get("decision")
        rerank_skip_reflect = bool(state.get("rerank_skip_reflect"))
        has_router_decision = isinstance(decision, RouteDecision)

        next_action = "router"
        reason = "默认从 router 进入检索流程"

        if phase in {"clarify", "reuse", "context_build"}:
            next_action = "stop"
            reason = f"阶段 {phase} 已形成终局输出"
        elif needs_clarify:
            next_action = "clarify"
            reason = "router 判定需要先澄清用户问题"
        elif needs_reuse:
            next_action = "reuse"
            reason = "router 判定可直接复用上一轮上下文生成"
        elif phase in {"start", "policy"}:
            next_action = "router"
            reason = "初始化后先进入 router"
        elif phase == "router":
            next_action = "retrieve"
            reason = "router 已输出 decision，进入检索"
        elif phase == "retrieve":
            next_action = "reranker" if route_results else "reflect"
            reason = "检索完成后进入 reranker/reflect"
        elif phase == "reranker":
            if not needs_retry:
                next_action = "context_build"
                reason = "reranker gate 通过，直接进入上下文构建"
            elif rerank_skip_reflect and retry_count < max_retries:
                next_action = "rewrite"
                reason = "reranker 指示跳过 reflect，直接改写"
            elif retry_count >= max_retries:
                state["no_answer"] = True
                state["answer"] = _format_no_answer(state.get("query", ""), state)
                state["context"] = state["answer"]
                state.setdefault("evidence_gaps", ["insufficient_evidence"])
                if not state.get("uncertainty_note"):
                    state["uncertainty_note"] = "reranker 判定证据不足，且已无重试预算"
                next_action = "answer"
                reason = "reranker 判定证据不足且无重试预算，直接返回无可靠依据"
            else:
                next_action = "reflect"
                reason = "reranker 需要进一步反思判定"
        elif phase == "reflect":
            if sufficient:
                next_action = "context_build"
                reason = "证据足够，进入上下文构建"
            elif needs_retry and retry_count < max_retries:
                next_action = "rewrite"
                reason = "证据不足且仍有重试预算，进入改写"
            else:
                state["no_answer"] = True
                state["answer"] = _format_no_answer(state.get("query", ""), state)
                state["context"] = state["answer"]
                state.setdefault("evidence_gaps", ["insufficient_evidence"])
                if not state.get("uncertainty_note"):
                    state["uncertainty_note"] = "反思判定证据不足，且已无重试预算"
                next_action = "answer"
                reason = "证据不足且无继续重试空间，直接返回无可靠依据"
        elif phase == "rewrite":
            next_action = "retrieve"
            reason = "改写完成后重新检索"
        elif has_router_decision and not route_results and not route_errors:
            next_action = "retrieve"
            reason = "已有路由决策但尚未检索，补走 retrieve"

        if not route_results and route_errors and not needs_clarify and not needs_reuse:
            if phase in {"retrieve", "reranker", "reflect"} and retry_count < max_retries:
                next_action = "rewrite"
                reason = "当前无有效检索结果且仍有预算，优先改写重试"

        if isinstance(rewrite_hint, RouteDecision) and next_action == "rewrite":
            state["decision"] = rewrite_hint

        state["agent_phase"] = phase or "policy"
        state["next_action"] = next_action
        state["action_reason"] = reason
        history = list(state.get("action_history") or [])
        history.append({
            "phase": phase or "policy",
            "next_action": next_action,
            "reason": reason,
            "retry_count": retry_count,
        })
        state["action_history"] = history
        state.setdefault("node_timings", {})["policy"] = time.time() - t0
        logger.info(
            f"[{cid}] [policy] phase={phase!r} next_action={next_action!r} "
            f"reason={reason!r} retry_count={retry_count}/{max_retries} "
            f"has_results={bool(route_results)} has_errors={bool(route_errors)}"
        )
        return state

    return policy_node


# ---------------------------------------------------------------------------
# Reflect 输出解析工具 (与 router _validate_decision 对齐)
# ---------------------------------------------------------------------------

def _normalize_routes(routes_raw: Any, *, max_paths: int = 2) -> List[str]:
    from ..routing.limits import normalize_routes as _normalize

    return _normalize(
        routes_raw,
        valid_routes=VALID_ROUTES,
        route_alias=ROUTE_ALIAS,
        max_paths=max_paths,
    )


def _parse_rewrites(
    rewrites_in: Any, routes: List[str], fallback_query: str,
) -> Dict[str, str]:
    if not isinstance(rewrites_in, dict):
        rewrites_in = {}
    clean: Dict[str, str] = {}
    for route in routes:
        val = rewrites_in.get(route)
        if isinstance(val, str) and val.strip():
            clean[route] = val.strip()
        elif isinstance(val, list):
            kws = [str(v).strip() for v in val if v and str(v).strip()]
            if kws:
                clean[route] = " ".join(kws)
        if route not in clean:
            clean[route] = fallback_query
    return clean


def _parse_str_list(d: Dict, key: str, *, upper: bool = False) -> List[str]:
    raw = d.get(key)
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    seen: set = set()
    for x in raw:
        s = str(x).strip()
        if not s:
            continue
        if upper:
            s = s.upper()
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _parse_int_list(d: Dict, key: str) -> List[int]:
    raw = d.get(key)
    if not isinstance(raw, list):
        return []
    out: List[int] = []
    for x in raw:
        try:
            v = int(x)
        except (TypeError, ValueError):
            continue
        if v >= 1 and v not in out:
            out.append(v)
    return out


def _parse_time(d: Dict) -> str:
    t = d.get("time")
    return str(t).strip() if isinstance(t, str) and t.strip() else ""


def _parse_chunk_type(d: Dict) -> Optional[str]:
    ct = d.get("chunk_type")
    if isinstance(ct, str) and ct.lower() in ("image", "table", "equation", "references"):
        return ct.lower()
    return None


def _make_rewrite_node() -> callable:
    def rewrite_node(state: AgentState) -> AgentState:
        t0 = time.time()
        cid = state.get("correlation_id", "?")
        new_decision = state.get("rewrite_hint")
        sub_decs = state.get("subquery_decisions") or []

        if sub_decs:
            state["decision"] = _merge_route_decisions(sub_decs)
        elif isinstance(new_decision, RouteDecision):
            state["decision"] = new_decision
            state["subquery_decisions"] = [new_decision]
        else:
            logger.warning(f"[{cid}] [rewrite] rewrite_hint 不是 RouteDecision, 保持原策略")

        state["retry_count"] = state.get("retry_count", 0) + 1
        # 增量重试: 保留前轮 route_results, retrieve 会按 pk 合并新结果
        state["agent_phase"] = "rewrite"
        state.setdefault("node_timings", {})["rewrite"] = time.time() - t0
        decision = state.get("decision")
        kept_hits = _count_hits(state.get("route_results") or {})
        skip_note = ""
        if state.get("rerank_skip_reflect"):
            diag_conf = float(state.get("rerank_diagnosis_confidence") or 0.0)
            cause = state.get("rerank_diagnosis_cause") or "?"
            skip_note = (
                f" from_skip_reflect cause={cause} diag_confidence={diag_conf:.2f}"
            )
        logger.info(
            f"[{cid}] [rewrite] retry={state['retry_count']} "
            f"subs={len(sub_decs) or 1} "
            f"routes={decision.routes if decision else []} "
            f"rewrites={decision.rewrites if decision else {}} "
            f"kept_hits={kept_hits}{skip_note}"
        )
        return state
    return rewrite_node


# ---------------------------------------------------------------------------
# 锁定文献补充检索: reuse / local 下探共用 — 在已有 chunk 之外再召回正文
# ---------------------------------------------------------------------------

_CHUNK_ID_IN_CONTEXT_RE = re.compile(r"chunk_id=([^\s|\]]+)")
_SUPPLEMENT_REUSE_MODES = frozenset({"drilldown", "continue"})
_DEFAULT_SUPPLEMENT_TOP_K = 12
# 自动 asset hydration: rerank 后把命中块关联的图/表/公式补进上下文的回填上限
_AUTO_ASSET_MAX_TOTAL = 8


def _extract_chunk_ids_from_context(context: str) -> set[str]:
    """从已格式化的 context 字符串里解析 chunk_id= 标记."""
    if not context:
        return set()
    return {m.group(1).strip() for m in _CHUNK_ID_IN_CONTEXT_RE.finditer(context) if m.group(1).strip()}


def _collect_known_chunk_ids(
    last_context: str,
    route_results: Optional[Dict[str, Any]] = None,
) -> set[str]:
    """合并 last_context 与本轮 route_results 里已出现的 chunk_id/pk."""
    known = _extract_chunk_ids_from_context(last_context or "")
    for res in (route_results or {}).values():
        if isinstance(res, LocalRetrieveResult):
            hits = res.chunk_hits
        elif isinstance(res, list):
            hits = [h for h in res if isinstance(h, Hit)]
        else:
            continue
        for h in hits:
            cid = (h.chunk_id or h.pk or "").strip()
            if cid:
                known.add(cid)
    return known


def _should_supplement_for_reuse(mode: str, target_doc_ids: List[str]) -> bool:
    if not target_doc_ids:
        return False
    return (mode or "").lower() in _SUPPLEMENT_REUSE_MODES


def _should_supplement_for_local(decision: Optional[RouteDecision]) -> bool:
    if decision is None or not decision.has(ROUTE_LOCAL):
        return False
    return bool(decision.target_doc_ids or decision.target_docs)


def _fetch_supplemental_locked_doc_chunks(
    local_r: ProgressiveLocalRetriever,
    *,
    query: str,
    target_doc_ids: List[str],
    target_docs: List[str],
    exclude_chunk_ids: set[str],
    top_k_chunks: int = _DEFAULT_SUPPLEMENT_TOP_K,
    chunk_type: Optional[str] = None,
) -> List[Hit]:
    """在锁定 doc_id 内做 local 补充召回, 排除 context/本轮已出现的 chunk."""
    if not target_doc_ids and not target_docs:
        return []
    search_query = (query or "").strip() or " "
    result = local_r.retrieve_direct(
        search_query,
        target_docs=target_docs,
        top_k_chunks=top_k_chunks,
        per_query_k=top_k_chunks,
        per_retriever_k=max(top_k_chunks, 10),
        chunk_type=chunk_type,
        target_doc_ids=list(target_doc_ids or []) or None,
    )
    supplemental: List[Hit] = []
    for hit in result.chunk_hits:
        cid = (hit.chunk_id or hit.pk or "").strip()
        if cid and cid in exclude_chunk_ids:
            continue
        sources = list(hit.sources or [])
        if "supplement" not in sources:
            sources.append("supplement")
        hit.sources = sources
        supplemental.append(hit)
    return supplemental


def _format_supplement_context_section(
    hits: List[Hit],
    context_builder: AgenticContextBuilder,
    *,
    title: str = "锁定文献补充检索 (本轮新召回, 与已有 chunk 去重)",
) -> str:
    if not hits:
        return ""
    lines = [f"# {title}"]
    for i, hit in enumerate(hits, 1):
        lines.append(context_builder._format_hit(i, hit))
    return context_builder.SEP.join(lines)


def _append_supplement_to_local_route(
    route_results: Dict[str, Any],
    new_hits: List[Hit],
) -> Dict[str, Any]:
    """把补充 chunk 并入 ROUTE_LOCAL 的 LocalRetrieveResult (按 pk 去重)."""
    if not new_hits:
        return route_results
    merged = dict(route_results or {})
    prev = merged.get(ROUTE_LOCAL)
    if isinstance(prev, LocalRetrieveResult):
        merged[ROUTE_LOCAL] = LocalRetrieveResult(
            candidate_docs=list(prev.candidate_docs),
            chunk_hits=_merge_hit_lists(prev.chunk_hits, new_hits),
        )
    else:
        merged[ROUTE_LOCAL] = LocalRetrieveResult(chunk_hits=list(new_hits))
    return merged


def _make_context_build_node(
    context_builder: AgenticContextBuilder,
    local_r: Optional[ProgressiveLocalRetriever] = None,
    *,
    supplement_top_k: int = _DEFAULT_SUPPLEMENT_TOP_K,
    neighbor_expander: Optional[Any] = None,
    auto_asset_expansion: bool = True,
) -> callable:
    def context_build_node(state: AgentState) -> AgentState:
        t0 = time.time()
        cid = state.get("correlation_id", "?")
        query = state["query"]
        decision = state.get("decision")
        route_results = state.get("route_results", {})

        # reranker 低分 → reflect OK: route_results 已是完整集; 若仍有快照则优先恢复
        pre_rerank = state.get("route_results_pre_rerank")
        if pre_rerank and not state.get("needs_retry", False):
            route_results = pre_rerank
            state["route_results"] = route_results
            restored = _count_hits(route_results)
            logger.info(
                f"[{cid}] [context_build] reflect OK, 恢复 reranker 过滤前 "
                f"{restored} 条结果"
            )

        if local_r is not None and _should_supplement_for_local(decision):
            assert decision is not None
            known = _collect_known_chunk_ids(
                state.get("last_context") or "",
                route_results,
            )
            t_sup = time.time()
            new_hits = _fetch_supplemental_locked_doc_chunks(
                local_r,
                query=query,
                target_doc_ids=list(decision.target_doc_ids or []),
                target_docs=list(decision.target_docs or []),
                exclude_chunk_ids=known,
                top_k_chunks=supplement_top_k,
                chunk_type=chunk_type_for_route(ROUTE_LOCAL, decision),
            )
            if new_hits:
                route_results = _append_supplement_to_local_route(route_results, new_hits)
                state["route_results"] = route_results
                logger.info(
                    f"[{cid}] [context_build] local 补充检索 +{len(new_hits)} chunk "
                    f"(已知 {len(known)} 条, 去重后)"
                )
            state.setdefault("node_timings", {})["local_supplement"] = time.time() - t_sup

        # 自动 asset hydration (issue: 图表块难检索): rerank 后把本轮命中块关联的
        # 图/表/公式 (related_assets) 各扩 1 跳补进上下文, 让文本块与图表块互为补充。
        # 不依赖 router 显式 expand; 仅在命中块确有 related_assets 时才会回填 (否则零行为变化)。
        if neighbor_expander is not None and auto_asset_expansion:
            from .neighbor_expansion import EXPAND_ASSETS, apply_neighbor_expansion
            t_asset = time.time()
            before = _count_hits(route_results)
            route_results = apply_neighbor_expansion(
                route_results,
                modes=[EXPAND_ASSETS],
                expander=neighbor_expander,
                max_total=_AUTO_ASSET_MAX_TOTAL,
                cid=cid,
            )
            state["route_results"] = route_results
            added = _count_hits(route_results) - before
            if added > 0:
                logger.info(
                    f"[{cid}] [context_build] 自动 asset hydration +{added} chunk "
                    f"(命中块关联图/表/公式)"
                )
            state.setdefault("node_timings", {})["asset_hydration"] = time.time() - t_asset

        if not decision:
            state["context"] = "[No routing decision]"
        else:
            sub_decs = state.get("subquery_decisions") or []
            try:
                context = context_builder.build(
                    query, decision, route_results,
                    subquery_decisions=sub_decs if len(sub_decs) > 1 else None,
                )
                synth_hint = (state.get("synth_hint") or "").strip()
                if synth_hint:
                    context = f"{context}\n\n# 复合查询拼接提示\n{synth_hint}"
                partial_note = (state.get("partial_note") or "").strip()
                if partial_note:
                    context = f"{context}\n\n# [系统] 信息有限\n{partial_note}"
                state["context"] = context
            except Exception as e:
                logger.exception(
                    f"[{cid}] [context_build] context_builder.build 失败"
                )
                state["context"] = _fallback_context_on_build_error(
                    query, route_results, e,
                )

        state["agent_phase"] = "context_build"
        state.setdefault("node_timings", {})["context_build"] = time.time() - t0
        logger.info(f"[{cid}] [context_build] context_len={len(state.get('context', ''))}")
        return state
    return context_build_node


def _make_clarify_node() -> callable:
    """FC ask 工具触发的反问出口: 跳过检索/反思, 直接返回澄清问题。"""

    def clarify_node(state: AgentState) -> AgentState:
        t0 = time.time()
        cid = state.get("correlation_id", "?")
        req = state.get("clarify_request") or {}
        answer = _format_clarify_answer(req)
        state["clarify_answer"] = answer
        state["context"] = answer
        state["agent_phase"] = "clarify"
        state.setdefault("node_timings", {})["clarify"] = time.time() - t0
        logger.info(
            f"[{cid}] [clarify] 返回反问: {answer[:120]!r}"
            + ("..." if len(answer) > 120 else "")
        )
        return state

    return clarify_node


# ---------------------------------------------------------------------------
# Reuse 出口: 把上轮 context/answer + 本轮 op 拼成可直接喂给生成 LLM 的提示
# ---------------------------------------------------------------------------

_REUSE_MODE_INSTRUCTIONS: Dict[str, str] = {
    "reformat": (
        "用户希望你换一种表达方式重写上一轮的回答 (例如更通俗 / 翻译 / 压缩)。"
        "严格基于上一轮 context 与 answer, 不要引入未在那里出现的事实。"
    ),
    "drilldown": (
        "用户希望你针对锁定文献或上一轮 answer 中的某一点做更详细的展开。"
        "优先使用「补充检索」段与上一轮 context; 若仍不含足够信息, 请显式说明 "
        "'当前资料未覆盖到该点'。"
    ),
    "metasession": (
        "用户在询问会话/检索状态本身 (例如刚才检索了哪几篇)。"
        "请基于上一轮文献列表与 context 摘要直接回答, 不要编造新的检索结果。"
    ),
    "confirm": (
        "用户在确认上一轮的结论。请仔细复核上一轮 answer 与 context, "
        "明确告知用户该结论是否成立, 必要时指出局限。"
    ),
    "continue": (
        "用户希望你顺着上一轮的回答继续说下去。请在保持事实一致性的前提下, "
        "基于上一轮 context 给出后续内容; 若信息已穷尽, 请如实说明。"
    ),
    "chitchat": (
        "用户在进行闲聊/致意, 不涉及文献检索。请用简短、自然、礼貌的语言回应即可, "
        "不要假装做了检索。"
    ),
    "out_of_scope": (
        "用户的问题不在文献知识库覆盖范围内 (例如非检索/与文献无关)。请礼貌告知本系统"
        "专注于文献检索, 不要强行编造文献依据; 可以给出一句话的方向性建议, 但不要捏造数据。"
    ),
}


def _filter_reuse_context_by_doc_ids(
    context: str,
    target_doc_ids: List[str],
) -> str:
    """按 doc_id 裁剪上轮 context, 只保留锁定文献的 chunk 段。"""
    if not context or not target_doc_ids:
        return context
    ids = {d.strip() for d in target_doc_ids if d and d.strip()}
    if not ids:
        return context

    sep = "\n\n---\n\n"
    parts = context.split(sep)
    kept: List[str] = []
    matched_chunks = 0
    for part in parts:
        if any(f"doc={did}" in part for did in ids):
            kept.append(part)
            matched_chunks += 1
        elif part.startswith("# ") and "doc=" not in part:
            kept.append(part)

    if matched_chunks == 0:
        logger.warning(
            f"[reuse] target_doc_ids={sorted(ids)} 未在上轮 context 中命中 chunk, "
            f"保留完整 context ({len(context)}c)"
        )
        return context

    filtered = sep.join(kept)
    if len(filtered) < len(context):
        logger.info(
            f"[reuse] context 按文献锁定: docs={len(ids)} "
            f"{len(context)}c → {len(filtered)}c ({matched_chunks} chunk 段)"
        )
    return filtered


def _build_reuse_context(
    query: str,
    mode: str,
    op: str,
    *,
    last_answer: str,
    last_context: str,
    doc_registry: Optional[List[Dict[str, str]]] = None,
    target_doc_ids: Optional[List[str]] = None,
    target_docs: Optional[List[str]] = None,
    doc_refs: Optional[List[int]] = None,
    supplement_context: str = "",
) -> str:
    """把 reuse 所需的全部材料组装成生成 LLM 直接可读的 context 字符串。"""
    instruction = _REUSE_MODE_INSTRUCTIONS.get(
        mode, _REUSE_MODE_INSTRUCTIONS["reformat"],
    )
    mode_header = (
        "# [系统模式] reuse — 基于已有材料 + 锁定文献补充检索生成最终答复"
        if supplement_context.strip()
        else "# [系统模式] reuse — 不再进行新检索, 基于已有材料生成最终答复"
    )
    parts: List[str] = [
        mode_header,
        f"模式: {mode}",
        f"路由器指令 (op): {op or '(空)'}",
        f"行为约束: {instruction}",
    ]

    if target_docs or target_doc_ids:
        lock_names = ", ".join(target_docs or target_doc_ids or [])
        ref_note = f" (refs={doc_refs})" if doc_refs else ""
        parts.append(f"\n# 锁定文献{ref_note}\n{lock_names}")
    elif doc_registry and mode in ("drilldown", "metasession", "continue", "reformat"):
        names = ", ".join(
            f"#{i + 1} {e.get('doc_name') or e.get('doc_id')}"
            for i, e in enumerate(doc_registry[:10])
            if (e.get("doc_name") or e.get("doc_id"))
        )
        if names:
            parts.append(f"\n# 已知文献列表 (上一轮 doc_registry)\n{names}")

    scoped_context = last_context
    if target_doc_ids and mode not in ("chitchat", "out_of_scope"):
        scoped_context = _filter_reuse_context_by_doc_ids(last_context, target_doc_ids)

    if mode not in ("chitchat", "out_of_scope"):
        if scoped_context:
            ctx_label = (
                "上一轮检索 context (已按锁定文献裁剪)"
                if target_doc_ids and scoped_context != last_context
                else "上一轮检索 context (复用依据)"
            )
            parts.append(f"\n# {ctx_label}\n" + scoped_context.strip())
        if last_answer:
            parts.append("\n# 上一轮最终 answer\n" + last_answer.strip())
        if not scoped_context and not last_answer and not supplement_context.strip():
            parts.append(
                "\n# [警告] 当前会话尚无上一轮 context/answer 可复用, "
                "请直接基于用户问题给出最合理回答, 并在必要时说明信息有限。"
            )

    if supplement_context.strip():
        parts.append("\n" + supplement_context.strip())

    parts.append(f"\n# 本轮用户发话\n{query}")
    return "\n".join(parts).strip()


def _make_reuse_node(
    local_r: ProgressiveLocalRetriever,
    context_builder: AgenticContextBuilder,
    *,
    supplement_top_k: int = _DEFAULT_SUPPLEMENT_TOP_K,
) -> callable:
    """FC reuse 出口: 复用上轮材料; 锁定文献且需下探时对 doc 做补充 chunk 召回。

    本节点不调用生成 LLM, 仅组装 state["context"]; 实际生成在上层完成。
    """

    def reuse_node(state: AgentState) -> AgentState:
        t0 = time.time()
        cid = state.get("correlation_id", "?")
        query = state["query"]
        req = state.get("reuse_request") or {}
        mode = str(req.get("mode") or "reformat").lower()
        op = str(req.get("op") or "")
        target_doc_ids = list(req.get("target_doc_ids") or [])
        target_docs = list(req.get("target_docs") or [])
        doc_refs = list(req.get("doc_refs") or [])
        last_answer = state.get("last_answer") or ""
        last_context = state.get("last_context") or ""
        doc_registry = state.get("last_round_docs") or []

        supplement_context = ""
        supplement_hits = 0
        if _should_supplement_for_reuse(mode, target_doc_ids):
            known = _collect_known_chunk_ids(last_context)
            search_q = (query or "").strip() or op
            t_sup = time.time()
            new_hits = _fetch_supplemental_locked_doc_chunks(
                local_r,
                query=search_q,
                target_doc_ids=target_doc_ids,
                target_docs=target_docs,
                exclude_chunk_ids=known,
                top_k_chunks=supplement_top_k,
            )
            supplement_hits = len(new_hits)
            supplement_context = _format_supplement_context_section(new_hits, context_builder)
            state.setdefault("node_timings", {})["reuse_supplement"] = time.time() - t_sup

        ctx = _build_reuse_context(
            query, mode, op,
            last_answer=last_answer,
            last_context=last_context,
            doc_registry=doc_registry,
            target_doc_ids=target_doc_ids,
            target_docs=target_docs,
            doc_refs=doc_refs,
            supplement_context=supplement_context,
        )
        state["context"] = ctx
        state["agent_phase"] = "reuse"
        state.setdefault("node_timings", {})["reuse"] = time.time() - t0
        lock_brief = (
            f" refs={doc_refs} docs={target_docs}"
            if doc_refs or target_docs
            else ""
        )
        logger.info(
            f"[{cid}] [reuse] mode={mode} op={op[:60]!r}{lock_brief} "
            f"last_ans={len(last_answer)}c last_ctx={len(last_context)}c "
            f"supplement_hits={supplement_hits} ctx_built={len(ctx)}c"
        )
        return state

    return reuse_node


# ---------------------------------------------------------------------------
# 条件边
# ---------------------------------------------------------------------------

def _after_router(state: AgentState) -> str:
    """Router 后: clarify / reuse 出口 或 正常检索。"""
    if state.get("needs_clarify"):
        return "clarify"
    if state.get("needs_reuse"):
        return "reuse"
    return "retrieve"


def _after_policy(state: AgentState) -> str:
    """Policy 后: 统一按 next_action 分流。"""
    action = str(state.get("next_action") or "router").strip().lower()
    if action in {
        "router", "retrieve", "reranker", "reflect", "rewrite",
        "clarify", "reuse", "context_build", "answer", "stop",
    }:
        return action
    return "router"


def _should_retry(state: AgentState) -> str:
    """反思后决定走向: "rewrite" | "context_build"."""
    needs_retry = state.get("needs_retry", False)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    if needs_retry and retry_count < max_retries:
        return "rewrite"
    return "context_build"


def _after_reranker(state: AgentState) -> str:
    """Reranker 后: context_build | reflect | rewrite (Phase 2 高置信跳过 reflect)。"""
    if not state.get("needs_retry", False):
        return "context_build"

    from .rerank_diagnosis import should_skip_reflect_after_reranker

    cid = state.get("correlation_id", "?")
    diag_conf = float(state.get("rerank_diagnosis_confidence") or 0.0)
    cause = state.get("rerank_diagnosis_cause") or ""

    if should_skip_reflect_after_reranker(
        skip_reflect=bool(state.get("rerank_skip_reflect")),
        rewrite_hint=state.get("rewrite_hint"),
        subquery_decisions=state.get("subquery_decisions"),
        retry_count=state.get("retry_count", 0),
        max_retries=state.get("max_retries", 2),
    ):
        logger.info(
            f"[{cid}] [reranker→rewrite] skip reflect: "
            f"cause={cause} diag_confidence={diag_conf:.2f} "
            f"retry={state.get('retry_count', 0)}/{state.get('max_retries', 2)}"
        )
        return "rewrite"
    logger.info(
        f"[{cid}] [reranker→reflect] diag_confidence={diag_conf:.2f} "
        f"cause={cause} skip_reflect={state.get('rerank_skip_reflect', False)}"
    )
    return "reflect"


# ---------------------------------------------------------------------------
# Graph 构建工厂
# ---------------------------------------------------------------------------

def build_langgraph_agent(
    router: QueryRouter,
    summary_retriever: SummaryRetriever,
    local_retriever: ProgressiveLocalRetriever,
    metadata_retriever: EnhancedMetadataRetriever,
    context_builder: AgenticContextBuilder,
    reflection_llm: Optional[LLMClient] = None,
    max_retries: int = 2,
    reflection_temperature: float = 0.0,
    reflection_max_tokens: int = 200,
    max_workers: int = 3,
    reranker_client: Optional[RerankerClient] = None,
    reranker_top_k: int = 5,
    reranker_quality_k: int = 3,
    reranker_quality_threshold: float = 0.5,
    reranker_quality_threshold_by_type: Optional[Dict[str, float]] = None,
    reranker_route_thresholds: Optional[RouteThresholds] = None,
    reranker_diagnosis_config: Optional[Any] = None,
    fail_open_min_emb_quality: Optional[float] = None,
    disable_thinking: Optional[bool] = None,
    routing_core: Optional[Any] = None,
    reflect_summary_config: Optional[ReflectSummaryConfig] = None,
    summary_top_docs: int = 5,
    summary_per_query_k: int = 5,
) -> Any:
    """构建 LangGraph agent (CompiledStateGraph)。

    Args:
        router: 复用现有 QueryRouter
        summary_retriever: 复用现有 SummaryRetriever
        local_retriever: 复用现有 ProgressiveLocalRetriever
        metadata_retriever: 复用现有 EnhancedMetadataRetriever
        context_builder: 复用现有 AgenticContextBuilder
        reflection_llm: 反思用 LLMClient (None=禁用反思)
        max_retries: 最大重试次数 (0=禁用反思)
        reranker_client: RerankerClient (None=禁用 reranker)
        reranker_top_k: reranker 每条路径保留的 top-k 条数
        reranker_quality_k: reranker 质量得分取 top-k 条
        reranker_quality_threshold: reranker 质量得分阈值, 低于此值走 reflect
        disable_thinking: 反思 LLM 的"思考模式"开关 (默认 None).
            - None: 不向 LLM 请求里下发 ``chat_template_kwargs`` 等 vLLM 专属参数,
              也不追加 ``/no_think`` 文本; 适合阿里云 / DeepSeek / GPUGeek 等云平台.
            - True/False: 仅当反思 LLMClient 启用了 ``disable_thinking_extra_body``
              (即后端是 vLLM) 时生效, 通过 ``chat_template_kwargs.enable_thinking``
              开/关思考模式.
        routing_core: 可选 pipeline.routing.RoutingCore 实例.
            - None  (默认): router/reflect 走原 JSON-mode 路径 (零行为变化)
            - 提供: router/reflect 改走 Function Calling (FC) 路径; 失败自动降级到 legacy.
        其余: 节点配置参数

    Returns:
        编译后的 LangGraph StateGraph

    Raises:
        ImportError: langgraph 未安装
    """
    if StateGraph is None:
        raise ImportError(
            "langgraph 未安装, 请运行: pip install langgraph"
        )

    use_reranker = reranker_client is not None

    if routing_core is not None:
        lim = getattr(routing_core, "routing_limits", None)
        logger.info(
            f"[langgraph] build with routing_core (FC enabled): "
            f"enable_multi={routing_core.enable_multi} "
            f"enable_ask={routing_core.enable_ask} "
            f"parallel_tool_calls={routing_core.parallel_tool_calls} "
            f"max_paths_per_sub={lim.max_paths_per_sub if lim else 2} "
            f"max_subqueries={lim.max_subqueries if lim else 3}"
        )
    else:
        logger.info("[langgraph] build with legacy router (JSON mode)")

    if routing_core is not None and getattr(routing_core, "routing_limits", None):
        max_workers = max(max_workers, routing_core.routing_limits.max_subqueries)

    # 构建节点函数 (闭包注入依赖)
    router_node = _make_router_node(router, routing_core=routing_core)
    policy_node = _make_policy_node()
    from .neighbor_expansion import NeighborExpander
    _neighbor_expander = NeighborExpander(
        client=metadata_retriever.client,
        collection=metadata_retriever.collection,
        vector_retriever=getattr(local_retriever, "vec", None)
        or getattr(summary_retriever, "vec", None),
    )
    retrieve_node = _make_retrieve_node(
        summary_retriever, local_retriever, metadata_retriever, max_workers,
        neighbor_expander=_neighbor_expander,
        summary_top_docs=summary_top_docs,
        summary_per_query_k=summary_per_query_k,
    )
    reflect_node = _make_reflect_node(
        reflection_llm,
        temperature=reflection_temperature,
        max_tokens=reflection_max_tokens,
        disable_thinking=disable_thinking,
        routing_core=routing_core,
        reflect_summary_config=reflect_summary_config,
    )
    rewrite_node = _make_rewrite_node()
    context_build_node = _make_context_build_node(
        context_builder, local_retriever,
        neighbor_expander=_neighbor_expander,
    )
    clarify_node = _make_clarify_node()
    reuse_node = _make_reuse_node(local_retriever, context_builder)

    # 构建 graph
    graph = StateGraph(AgentState)

    graph.add_node("policy", policy_node)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("context_build", context_build_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("reuse", reuse_node)

    if use_reranker:
        reranker_node = _make_reranker_node(
            reranker_client,
            top_k=reranker_top_k,
            quality_k=reranker_quality_k,
            quality_threshold=reranker_quality_threshold,
            quality_threshold_by_type=reranker_quality_threshold_by_type,
            route_thresholds=reranker_route_thresholds,
            diagnosis_config=reranker_diagnosis_config,
            fail_open_min_emb_quality=fail_open_min_emb_quality,
        )
        graph.add_node("reranker", reranker_node)

    # 入口 → policy → (router | clarify | reuse | retrieve | reflect | rewrite | context_build | stop)
    graph.set_entry_point("policy")
    graph.add_conditional_edges("policy", _after_policy, {
        "router": "router",
        "retrieve": "retrieve",
        "reranker": "reranker" if use_reranker else "retrieve",
        "reflect": "reflect",
        "rewrite": "rewrite",
        "clarify": "clarify",
        "reuse": "reuse",
        "context_build": "context_build",
        "answer": END,
        "stop": END,
    })

    graph.add_edge("clarify", END)
    graph.add_edge("reuse", END)

    if use_reranker:
        # retrieve → reranker
        graph.add_edge("retrieve", "reranker")
    else:
        # 无 reranker 时也统一回到 policy, 再由 policy 决定进入 reflect。
        # 避免 retrieve 同时拥有普通边和 policy 边导致重复/歧义执行。
        graph.add_edge("retrieve", "policy")

    # 关键执行节点回到 policy, 由 policy 统一决定下一步
    graph.add_edge("router", "policy")
    graph.add_edge("reflect", "policy")
    graph.add_edge("rewrite", "policy")
    if use_reranker:
        graph.add_edge("reranker", "policy")
    else:
        # 无 reranker 时, retrieve 之后由 policy 决定进入 reflect
        pass

    # context_build 后结束
    graph.add_edge("context_build", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# 高层调用接口 (供 QueryFlow 使用)
# ---------------------------------------------------------------------------

class LangGraphAgent:
    """LangGraph Agentic RAG 智能体, 接口与 AgenticRAGPipeline.answer() 对齐。"""

    # 内部 session_id → last_round_docs 缓存上限. 防止长跑进程无限增长.
    _SESSION_CACHE_MAX = 256

    # last_context / last_answer 持久化时的字节上限 (避免 session_meta 膨胀)
    _LAST_CONTEXT_PERSIST_LIMIT = 8000
    _LAST_ANSWER_PERSIST_LIMIT = 1500

    def __init__(
        self,
        compiled_graph: Any,
        max_retries: int = 2,
        generation_system_prompt: str = DEFAULT_AGENTIC_SYSTEM_PROMPT,
        generation_temperature: float = 0,
        generation_max_tokens: int = 2048,
        llm: Optional[LLMClient] = None,
    ) -> None:
        self.graph = compiled_graph
        self.max_retries = max_retries
        self.system_prompt = generation_system_prompt
        self.temperature = generation_temperature
        self.max_tokens = generation_max_tokens
        # 生成 LLM: answer() 用. None 时调用 answer() 会抛错, 但 run() 仍可用
        # (上层 QueryFlow._run_langgraph 自己有 LLM, 不依赖这里).
        self.llm = llm
        # session_id → last_round_docs 的进程内缓存. 仅当上层调用方既不传
        # session_meta 又传了 session_id 时作为兜底锚点 (issue #2).
        # 用 dict 的插入顺序模拟 FIFO; 超过上限丢最早的.
        self._session_docs: "Dict[str, List[Dict[str, str]]]" = {}
        # session_id → 上一轮 (last_context, last_answer, clarify_pending) 兜底缓存,
        # 同样用于调用方未持久化 session_meta 时让 reuse 路径仍可用。
        self._session_reuse: "Dict[str, Dict[str, Any]]" = {}

    def _remember_round_docs(
        self, session_id: Optional[str], docs: List[Dict[str, str]],
    ) -> None:
        if not session_id:
            return
        # 先 pop 再写, 保持最近使用的在末尾 (LRU 语义)
        self._session_docs.pop(session_id, None)
        self._session_docs[session_id] = list(docs or [])
        while len(self._session_docs) > self._SESSION_CACHE_MAX:
            # popitem(last=False) 等价行为: 删除最早插入的 key
            oldest_key = next(iter(self._session_docs))
            self._session_docs.pop(oldest_key, None)

    def _recall_round_docs(
        self, session_id: Optional[str],
    ) -> List[Dict[str, str]]:
        if not session_id:
            return []
        return list(self._session_docs.get(session_id, []) or [])

    def _remember_reuse_state(
        self,
        session_id: Optional[str],
        *,
        last_context: str,
        last_answer: str,
        clarify_pending: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not session_id:
            return
        self._session_reuse.pop(session_id, None)
        self._session_reuse[session_id] = {
            "last_context": last_context,
            "last_answer": last_answer,
            "clarify_pending": clarify_pending or None,
        }
        while len(self._session_reuse) > self._SESSION_CACHE_MAX:
            oldest_key = next(iter(self._session_reuse))
            self._session_reuse.pop(oldest_key, None)

    def _recall_reuse_state(
        self, session_id: Optional[str],
    ) -> Dict[str, Any]:
        if not session_id:
            return {}
        return dict(self._session_reuse.get(session_id, {}) or {})

    def run(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        session_meta: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """执行完整检索流程 (不含生成), 返回与 AgenticRAGPipeline.run() 一致的结果。

        Args:
            session_meta: 上轮 QueryResult.session_meta. 我们从中读 doc_registry
                作为 last_round_docs (上一轮的最终结果), 用于 router 解析用户
                "第X篇" (issue #1 修订: 锚定上一轮, 不是全会话累计).
                缺省即新会话, last_round_docs 为空.
            session_id: 可选会话标识. issue #2 兜底: 若上层调用方 (例如自定义 API)
                没有持久化 session_meta, 但传了稳定的 session_id, agent 会在进程内
                缓存上一轮的 doc_registry, 下一轮自动读回作为 last_round_docs.
                注意: 仅当 session_meta 缺失时才使用; session_meta 显式传入优先.
        """
        t0 = time.time()
        # session_meta["doc_registry"] 是上轮 run() 持久化的 this_round_docs;
        # 这一轮把它读回作为 last_round_docs (router 看到的"上轮列表")。
        last_round_docs: List[Dict[str, str]] = []
        meta_has_registry = False
        if session_meta:
            raw = session_meta.get("doc_registry") or []
            if isinstance(raw, list):
                for entry in raw:
                    if isinstance(entry, dict) and entry.get("doc_id"):
                        last_round_docs.append({
                            "doc_id": str(entry["doc_id"]),
                            "doc_name": str(entry.get("doc_name") or entry["doc_id"]),
                            "pinned": bool(entry.get("pinned", False)),
                        })
                # 即便上层显式传了 doc_registry=[] (新会话), 也视为"已经管理过"
                meta_has_registry = "doc_registry" in session_meta

        # P0-1 / P1-2: 取出上一轮 last_context / last_answer / clarify_pending 供 router 判定
        last_answer = ""
        last_context = ""
        clarify_pending_in: Optional[Dict[str, Any]] = None
        if session_meta:
            la = session_meta.get("last_answer")
            if isinstance(la, str):
                last_answer = la
            lc = session_meta.get("last_context")
            if isinstance(lc, str):
                last_context = lc
            cp = session_meta.get("clarify_pending")
            if isinstance(cp, dict) and cp:
                clarify_pending_in = cp

        # 兜底: session_meta 不可用时, 尝试用 session_id 从进程内缓存读上一轮.
        if not last_round_docs and not meta_has_registry and session_id:
            last_round_docs = self._recall_round_docs(session_id)
            if last_round_docs:
                logger.debug(
                    f"[doc_registry] session_meta 缺失, 已从 session_id="
                    f"{session_id} 的进程内缓存恢复 {len(last_round_docs)} 篇上一轮文献."
                )
        if not last_answer and not last_context and session_id:
            cached = self._recall_reuse_state(session_id)
            if cached:
                last_answer = cached.get("last_answer") or ""
                last_context = cached.get("last_context") or ""
                if clarify_pending_in is None:
                    clarify_pending_in = cached.get("clarify_pending") or None

        # issue #2: 多轮场景但 doc_registry 锚点完全缺失 → 显式告警, 否则用户用
        # "第X篇" 回指时会静默失效, 上层难以排查.
        if history and not last_round_docs and not meta_has_registry and not session_id:
            logger.warning(
                "[doc_registry] 检测到多轮对话 (history 非空) 但既未传 session_meta"
                "['doc_registry'] 也未传 session_id, 用户若使用 '第X篇' 等回指将无法"
                "解析. 修复方式 (任选其一): "
                "(1) 调用方持久化 QueryResult.session_meta 并回传; "
                "(2) 传入稳定的 session_id 让 agent 进程内缓存上一轮的 doc_registry."
            )

        initial_state: AgentState = {
            "query": query,
            "history": history,
            "last_round_docs": last_round_docs,
            "this_round_docs": [],   # 本轮从空开始累计
            "last_answer": last_answer,
            "last_context": last_context,
            "clarify_pending": clarify_pending_in or {},
            "retry_count": 0,
            "max_retries": self.max_retries,
            "agent_phase": "start",
            "next_action": "router",
            "action_reason": "初始化进入 policy",
            "action_history": [],
            "evidence_gaps": [],
            "sufficient": False,
            "uncertainty_note": "",
            "correlation_id": uuid.uuid4().hex[:8],
            "route_results": {},
            "route_errors": {},
            "node_timings": {},
        }

        final_state = self.graph.invoke(initial_state)

        retrieval_total = time.time() - t0
        node_timings = final_state.get("node_timings", {})

        this_round_docs = final_state.get("this_round_docs", []) or []
        persisted_registry = _persist_doc_registry(
            last_round_docs,
            this_round_docs,
            final_state.get("decision"),
            subquery_decisions=final_state.get("subquery_decisions") or [],
        )
        # 兜底缓存: 若调用方传了 session_id, 本轮 doc_registry 写进程内 LRU,
        # 下一轮即便不传 session_meta 也能恢复. 不影响 session_meta 显式契约.
        self._remember_round_docs(session_id, persisted_registry)

        needs_clarify = bool(final_state.get("needs_clarify"))
        clarify_answer = final_state.get("clarify_answer", "") or ""
        needs_reuse = bool(final_state.get("needs_reuse"))
        no_answer = bool(final_state.get("no_answer"))
        reuse_request_out = final_state.get("reuse_request") or {}

        # P0-1: reuse 路径下 session_meta 应当透传上一轮的 last_context/last_answer
        # (本轮没产新检索, 不应该把它们清掉); 正常检索路径下用本轮新生成的。
        if needs_clarify or needs_reuse:
            persist_last_context = last_context
            persist_last_answer = last_answer
        else:
            # 本轮新检索的 context 持久化以供下一轮 reuse 使用
            persist_last_context = _truncate_for_persist(
                final_state.get("context", "") or "",
                self._LAST_CONTEXT_PERSIST_LIMIT,
            )
            # last_answer 在 run() 阶段还没生成, 由调用方在 answer 后写回 session_meta
            # (见 flows/query.py._run_langgraph); run() 这里只透传 (新检索时清空旧值)
            persist_last_answer = ""

        # 兜底缓存: 同样把 reuse 相关字段写进 _session_reuse, 即便调用方不持久化 session_meta
        self._remember_reuse_state(
            session_id,
            last_context=persist_last_context,
            last_answer=persist_last_answer,
            clarify_pending=(
                {
                    "q": (final_state.get("clarify_request") or {}).get("q", ""),
                    "opts": (final_state.get("clarify_request") or {}).get("opts", []),
                } if needs_clarify else None
            ),
        )

        # 持久化: 本轮的 this_round_docs 写出去, 下一轮就成为 last_round_docs。
        # 字段名对外保持 doc_registry 不变, 避免破坏 QueryResult.session_meta API。
        result: Dict[str, Any] = {
            "query": query,
            "decision": final_state.get("decision"),
            "results": final_state.get("route_results", {}),
            "context": final_state.get("context", ""),
            "retry_count": final_state.get("retry_count", 0),
            "correlation_id": final_state.get("correlation_id", ""),
            "agent_phase": final_state.get("agent_phase", ""),
            "next_action": final_state.get("next_action", ""),
            "action_reason": final_state.get("action_reason", ""),
            "action_history": final_state.get("action_history", []) or [],
            "evidence_gaps": final_state.get("evidence_gaps", []) or [],
            "sufficient": bool(final_state.get("sufficient", False)),
            "uncertainty_note": final_state.get("uncertainty_note", "") or "",
            "reranker_score": final_state.get("reranker_score", 0.0),
            "doc_registry": persisted_registry,
            "router_metrics": {},
            "needs_clarify": needs_clarify,
            "clarify_request": final_state.get("clarify_request"),
            "needs_reuse": needs_reuse,
            "reuse_request": reuse_request_out,
            "no_answer": no_answer,
            "answer": final_state.get("answer", "") or "",
            "persist_last_context": persist_last_context,
            "persist_last_answer": persist_last_answer,
            "latency": {
                "route_s": round(node_timings.get("router", 0), 3),
                "retrieve_s": round(node_timings.get("retrieve", 0), 3),
                "reranker_s": round(node_timings.get("reranker", 0), 3),
                "reflect_s": round(node_timings.get("reflect", 0), 3),
                "rewrite_s": round(node_timings.get("rewrite", 0), 3),
                "policy_s": round(node_timings.get("policy", 0), 3),
                "render_s": round(
                    node_timings.get("context_build", 0)
                    or node_timings.get("clarify", 0)
                    or node_timings.get("reuse", 0),
                    3,
                ),
                "total_s": round(retrieval_total, 3),
            },
        }
        logger.info(
            f"[{result.get('correlation_id','?')}] [run] final_phase={result.get('agent_phase')!r} "
            f"next_action={result.get('next_action')!r} sufficient={result.get('sufficient')} "
            f"evidence_gaps={result.get('evidence_gaps')} actions={len(result.get('action_history') or [])}"
        )
        if needs_clarify:
            result["answer"] = clarify_answer
        elif no_answer and not result.get("answer"):
            result["answer"] = _format_no_answer(query, final_state)
        return result

    def answer(
        self,
        query: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        history: Optional[List[Dict[str, str]]] = None,
        chat_messages: Optional[List[Dict[str, str]]] = None,
        disable_thinking: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """执行检索+生成, 返回与 AgenticRAGPipeline.answer() 一致的结果。

        Args:
            chat_messages: 生成 LLM 用的历史 messages (OpenAI 格式, 不含 system 和当前 user).
                给定即走多轮生成路径, 否则单轮.
            history: 仅给检索路由用, 不参与生成 messages. 与 chat_messages 解耦.
            disable_thinking: 关闭推理模式. None 表示沿用 LLMClient 的默认行为
                (云平台 = 不下发, vLLM 见 client.disable_thinking_extra_body).
        """
        if self.llm is None:
            raise RuntimeError(
                "LangGraphAgent 未配置生成 LLM, 无法 answer(). 请在构造时通过 "
                "build_langgraph_agent_from_pipeline(pipeline, ...) 或直接传入 "
                "llm=LLMClient(...) 注入生成模型. (注: QueryFlow._run_langgraph "
                "走自己的 _get_simple_llm, 不依赖本字段, 仅独立使用 LangGraphAgent "
                "时需要.)"
            )

        system = system or self.system_prompt
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens

        run_result = self.run(query, history=history, **kwargs)

        if run_result.get("needs_clarify"):
            clarify_answer = str(run_result.get("answer") or run_result.get("context") or "")
            run_result["answer"] = clarify_answer
            run_result["usage"] = None
            run_result["user_message"] = ""
            run_result["latency"]["generate_s"] = 0.0
            logger.info(f"[generate] clarify 出口, 跳过 LLM 生成: {clarify_answer[:80]!r}")
            return run_result

        if run_result.get("no_answer"):
            answer = str(run_result.get("answer") or run_result.get("context") or "").strip()
            run_result["answer"] = answer
            run_result["usage"] = None
            run_result["user_message"] = ""
            run_result["latency"]["generate_s"] = 0.0
            logger.info(f"[generate] no_answer 出口, 跳过 LLM 生成: {answer[:80]!r}")
            return run_result

        context = run_result["context"]
        # Reuse 出口: 用 REUSE_USER_TEMPLATE, 不走"检索到的上下文"模板, 避免 LLM 误以为这是新检索
        if run_result.get("needs_reuse"):
            user_msg = REUSE_USER_TEMPLATE.format(context=context, query=query)
        else:
            user_msg = AGENTIC_USER_TEMPLATE.format(context=context)

        logger.info(
            f"[generate] 开始生成: prompt_chars={len(user_msg)} "
            f"retry_count={run_result.get('retry_count', 0)} "
            f"multi_turn={bool(chat_messages)} stream={stream}"
        )

        # disable_thinking 透传契约与 AgenticRAGPipeline.answer 对齐:
        # None → 不传 (LLMClient 用自己的默认值); 显式给值 → 透传
        dt_kwargs: Dict[str, Any] = {}
        if disable_thinking is not None:
            dt_kwargs["disable_thinking"] = disable_thinking

        t0 = time.time()
        ttft: Optional[float] = None
        answer: str = ""
        usage: Optional[Dict[str, Any]] = None

        if chat_messages:
            # 多轮: system + history + 当前 user
            messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
            messages.extend(chat_messages)
            messages.append({"role": "user", "content": user_msg})

            if stream:
                chunks_list: List[str] = []
                for piece in self.llm.chat_messages_stream(
                    messages, temperature=temperature, max_tokens=max_tokens,
                    **dt_kwargs,
                ):
                    if ttft is None:
                        ttft = time.time() - t0
                        print(f"\n[首包] ttft={ttft:.2f}s", flush=True)
                    chunks_list.append(piece)
                    print(piece, end="", flush=True)
                print()
                answer = "".join(chunks_list)
            else:
                chat_res = self.llm.chat_messages(
                    messages, temperature=temperature, max_tokens=max_tokens,
                    **dt_kwargs,
                )
                answer = chat_res["answer"]
                usage = chat_res.get("usage")
        else:
            # 单轮
            if stream:
                chunks_list = []
                for piece in self.llm.chat_stream(
                    system=system, user=user_msg,
                    temperature=temperature, max_tokens=max_tokens,
                    **dt_kwargs,
                ):
                    if ttft is None:
                        ttft = time.time() - t0
                        print(f"\n[首包] ttft={ttft:.2f}s", flush=True)
                    chunks_list.append(piece)
                    print(piece, end="", flush=True)
                print()
                answer = "".join(chunks_list)
            else:
                chat_res = self.llm.chat(
                    system=system, user=user_msg,
                    temperature=temperature, max_tokens=max_tokens,
                    **dt_kwargs,
                )
                answer = chat_res["answer"]
                usage = chat_res.get("usage")

        t_gen = time.time() - t0

        run_result["answer"] = answer
        run_result["usage"] = usage
        run_result["user_message"] = user_msg
        run_result["latency"]["generate_s"] = round(t_gen, 3)
        if ttft is not None:
            run_result["latency"]["ttft_s"] = round(ttft, 3)
        run_result["latency"]["total_s"] = round(
            run_result["latency"].get("total_s", 0) + t_gen, 3,
        )

        lat = run_result["latency"]
        gen_part = (
            f"generate={lat['generate_s']:.2f}s"
            + (f" (ttft={lat['ttft_s']:.2f}s)" if "ttft_s" in lat else "")
        )
        logger.info(
            f"[耗时-langgraph-端到端] route={lat.get('route_s', 0):.2f}s | "
            f"retrieve={lat.get('retrieve_s', 0):.2f}s | "
            f"reranker={lat.get('reranker_s', 0):.2f}s | "
            f"reflect={lat.get('reflect_s', 0):.2f}s | "
            f"rewrite={lat.get('rewrite_s', 0):.2f}s | "
            f"render={lat.get('render_s', 0):.2f}s | "
            f"{gen_part} | total={lat['total_s']:.2f}s"
        )

        return run_result


def build_langgraph_agent_from_pipeline(
    pipeline: AgenticRAGPipeline,
    reflection_llm: Optional[LLMClient] = None,
    max_retries: int = 2,
    reflection_temperature: float = 0.0,
    reflection_max_tokens: int = 200,
    reranker_client: Optional[RerankerClient] = None,
    reranker_top_k: int = 5,
    reranker_quality_k: int = 3,
    reranker_quality_threshold: float = 0.5,
    reranker_quality_threshold_by_type: Optional[Dict[str, float]] = None,
    reranker_route_thresholds: Optional[RouteThresholds] = None,
    reranker_diagnosis_config: Optional[Any] = None,
    fail_open_min_emb_quality: Optional[float] = None,
    disable_thinking: Optional[bool] = None,
    routing_core: Optional[Any] = None,
    generation_llm: Optional[LLMClient] = None,
    reflect_summary_config: Optional[ReflectSummaryConfig] = None,
    summary_top_docs: int = 5,
    summary_per_query_k: int = 5,
) -> LangGraphAgent:
    """从现有 AgenticRAGPipeline 实例构建 LangGraph agent。

    复用 pipeline 的所有组件 (router, retrievers, context_builder, llm)，
    零重复构建。

    Args:
        routing_core: 可选 pipeline.routing.RoutingCore 实例; 注入则 router/reflect
            走 Function Calling 路径, None 时保持 legacy 行为.
        generation_llm: 可选生成 LLM 覆盖. 缺省复用 pipeline.llm. 仅 answer() 用到;
            run() 不依赖生成 LLM. 上层 QueryFlow._run_langgraph 自带 _get_simple_llm,
            不依赖这里, 因此 pipeline.llm 为 None 时仍可正常工作 (只是直接调用
            LangGraphAgent.answer() 会抛错, 需显式传 generation_llm).
    """
    compiled = build_langgraph_agent(
        router=pipeline.router,
        summary_retriever=pipeline.summary_r,
        local_retriever=pipeline.local_r,
        metadata_retriever=pipeline.metadata_r,
        context_builder=pipeline.context_builder,
        reflection_llm=reflection_llm,
        max_retries=max_retries,
        reflection_temperature=reflection_temperature,
        reflection_max_tokens=reflection_max_tokens,
        reranker_client=reranker_client,
        reranker_top_k=reranker_top_k,
        reranker_quality_k=reranker_quality_k,
        reranker_quality_threshold=reranker_quality_threshold,
        reranker_quality_threshold_by_type=reranker_quality_threshold_by_type,
        reranker_route_thresholds=reranker_route_thresholds,
        fail_open_min_emb_quality=fail_open_min_emb_quality,
        disable_thinking=disable_thinking,
        routing_core=routing_core,
        reranker_diagnosis_config=reranker_diagnosis_config,
        reflect_summary_config=reflect_summary_config,
        summary_top_docs=summary_top_docs,
        summary_per_query_k=summary_per_query_k,
    )
    return LangGraphAgent(
        compiled_graph=compiled,
        max_retries=max_retries,
        llm=generation_llm if generation_llm is not None else pipeline.llm,
    )
