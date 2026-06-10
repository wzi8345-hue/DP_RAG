"""RoutingCore: FC 化 router + reflect 的公共入口。

调用契约 (供 LangGraph 节点消费):

    core = RoutingCore(
        router_llm=...,            # LLMClient, 必填
        reflect_llm=...,           # LLMClient | None; None=禁用反思
        current_year=2026,
        enable_multi=True,
        enable_ask=False,
        fc_fallback_to_json=True,  # FC 失败时自动降级 json_schema 路径
    )

    # ── 路由 ──
    outcome = core.route(query, history=history, doc_registry=last_round_docs)
    # outcome ∈ {RouteDecision, MultiRouteDecision, ClarifyRequest}

    # ── 反思 ──
    verdict = core.reflect(
        query=query,
        last_decision=outcome (RouteDecision 或 MultiRouteDecision),
        results_summary=summary_text,
        total_hits=N,
        this_round_docs=this_round,
        retry_count=k,
        max_retries=m,
    )
    # verdict ∈ ReflectVerdict (含 needs_retry, decision, partial, cause, _meta)

设计要点:
1. **永不抛**: 任何 LLM/解析/转换错误都退化到 heuristic 兜底, _meta.fallback_chain 记录所有降级路径;
2. **复用现有 QueryRouter._validate_decision**: 通过依赖注入 validate_fn (在 __init__ 接受);
3. **LLM 调用次数严格 ≤ 1 次/route + ≤ 1 次/reflect**: 不做 ReAct 多轮;
4. **观测**: 每次返回都附 _meta = {used_fc, fallback_chain, latency_ms, model, tool_name, conf}.
"""

from __future__ import annotations

import datetime
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..clients.llm import LLMClient
from ..models import RouteDecision
from ..retrieval.hybrid_weights import infer_retrieve_bias_heuristic
from .registry_scope import apply_registry_scope_guard, apply_registry_scope_to_reuse
from .limits import (
    DEFAULT_ROUTING_LIMITS,
    RoutingLimits,
    compound_intent_hint,
    paths_should_split_to_multi,
    split_plan_args_to_multi_args,
)
from .decision_builder import (
    ClarifyRequest,
    MultiRouteDecision,
    ReuseRequest,
    RouteOutcome,
    SubqueryDecision,
    build_from_ask_args,
    build_from_multi_args,
    build_from_plan_args,
    build_from_reuse_args,
    build_reflect_retry,
)
from .fc_parser import ParsedToolCall, parse_tool_calls
from .fc_schema import (
    REFLECT_TOOL_NAMES,
    ROUTER_TOOL_NAMES,
    TOOL_ASK,
    TOOL_MULTI,
    TOOL_OK,
    TOOL_PARTIAL,
    TOOL_PLAN,
    TOOL_RETRY,
    TOOL_REUSE,
    reflect_tools,
    router_tools,
)
from .prompts import render_reflect_system_fc, render_router_system_fc

logger = logging.getLogger(__name__)


_STRUCTURAL_METADATA_LABELS = {
    "fig": "图",
    "table": "表",
    "page": "页码",
    "paragraph": "段落",
}

_INVENTORY_QUERY_RE = re.compile(
    r"(共有|总共|一共|多少篇|几篇|全部|所有|完整列出|列出所有|按年份|统计|数量|count|all)",
    re.IGNORECASE,
)


def _single_registry_entry(
    doc_registry: Optional[List[Dict[str, str]]],
) -> Optional[Dict[str, str]]:
    """仅在无歧义时返回唯一可锁定文献: 单 pinned 或 registry 只有一篇。"""
    if not doc_registry:
        return None
    pinned = [
        e for e in doc_registry
        if isinstance(e, dict) and e.get("pinned") and e.get("doc_id")
    ]
    if len(pinned) == 1:
        return pinned[0]
    valid = [e for e in doc_registry if isinstance(e, dict) and e.get("doc_id")]
    if len(valid) == 1:
        return valid[0]
    return None


def _metadata_structural_labels(decision: RouteDecision) -> List[str]:
    labels: List[str] = []
    if decision.fig_refs:
        labels.append(_STRUCTURAL_METADATA_LABELS["fig"])
    if decision.table_refs:
        labels.append(_STRUCTURAL_METADATA_LABELS["table"])
    if decision.page_refs:
        labels.append(_STRUCTURAL_METADATA_LABELS["page"])
    if decision.paragraph_refs:
        labels.append(_STRUCTURAL_METADATA_LABELS["paragraph"])
    return labels


def _decision_has_unanchored_references(decision: RouteDecision) -> bool:
    if (decision.chunk_type or "").lower() != "references":
        return False
    if decision.target_doc_ids or decision.target_docs:
        return False
    return "progressive" in (decision.routes or []) or "local" in (decision.routes or [])


# 仅当 query 显式索取"某文献引用/参考了哪些文献"(引文列表本身) 时, 才认可 references
# 过滤。仅提到标准/方法/试验/规范名 (如 "ASTM 标准/XX 测试方法") 不算 — 那是正文事实问句。
_EXPLICIT_REFERENCE_PATTERNS = [
    re.compile(r"参考文献"),
    re.compile(r"引用文献"),
    re.compile(r"引文"),
    re.compile(r"引用了?哪些(文献|论文|工作)"),
    re.compile(r"\breferences?\b", re.IGNORECASE),
    re.compile(r"\bbibliography\b", re.IGNORECASE),
    re.compile(r"\b(works\s+cited|cited\s+(works|references|literature))\b", re.IGNORECASE),
]


def _query_has_explicit_reference_intent(query: str) -> bool:
    """query 是否显式表达了"索取参考文献/引文列表"的意图。"""
    q = query or ""
    return any(p.search(q) for p in _EXPLICIT_REFERENCE_PATTERNS)


def _guard_inventory_query(
    outcome: RouteOutcome,
    *,
    query: str,
    enable_ask: bool,
    cid: str = "-",
) -> RouteOutcome:
    """全量盘点/统计类问题不让普通 summary top-k 冒充全集。"""
    if not enable_ask or not _INVENTORY_QUERY_RE.search(query or ""):
        return outcome
    if isinstance(outcome, RouteDecision):
        routes = outcome.routes or []
    elif isinstance(outcome, MultiRouteDecision):
        routes = [r for s in outcome.subqueries for r in (s.decision.routes or [])]
    else:
        return outcome
    if "summary" not in routes:
        return outcome
    logger.info(
        f"[{cid}] [routing.route] 检测到全量盘点/统计 query, 转 ask 避免 top-k 冒充全集"
    )
    return ClarifyRequest(
        question=(
            "这个问题需要全库盘点/统计，普通摘要检索只能返回相关 top-k，不能保证覆盖全部。"
            "请确认：你是要先看最相关的若干篇，还是要执行全库统计任务？"
        ),
        options=["先看最相关的文献", "执行全库统计/导出清单"],
        raw={"source": "inventory_query_guard", "query": query},
    )


def _guard_ambiguous_metadata(
    outcome: RouteOutcome,
    *,
    query: str,
    doc_registry: Optional[List[Dict[str, str]]],
    enable_ask: bool,
    cid: str = "-",
) -> RouteOutcome:
    """结构化 metadata 查询必须有文献锚点; 多篇/无锚点时转 ask。

    图/表/页/段落编号在全库范围内天然重复。若 LLM 产出 metadata 过滤但没有
    docs/refs/doc_id, 直接全库硬过滤会高概率走偏。这里在执行前做策略级防护:
      - registry 中唯一文献或唯一 pinned 文献: 自动补 doc_id/doc_name;
      - 多篇或无 registry: 若 ask 开启, 返回 ClarifyRequest;
      - ask 关闭: 保持原策略, 但写日志显式暴露风险。
    entities 精确查找允许全库, 不纳入本 guard。
    """

    decisions: List[RouteDecision] = []
    if isinstance(outcome, RouteDecision):
        decisions = [outcome]
    elif isinstance(outcome, MultiRouteDecision):
        decisions = [s.decision for s in outcome.subqueries]
    else:
        return outcome

    ambiguous: List[RouteDecision] = []
    unanchored_references: List[RouteDecision] = []
    for dec in decisions:
        if (
            "metadata" in (dec.routes or [])
            and not (dec.target_doc_ids or dec.target_docs)
            and _metadata_structural_labels(dec)
        ):
            ambiguous.append(dec)
        if _decision_has_unanchored_references(dec):
            # 只有 query 显式提到"参考文献/引文列表"才认可 references 检索意图;
            # 否则视为 LLM 误判 (如把 "ASTM 标准方法" 当成引文), 撤销 ctype=references
            # 回退正常正文检索, 不再弹"是哪篇文献"的澄清。
            if _query_has_explicit_reference_intent(query):
                unanchored_references.append(dec)
            else:
                dec.chunk_type = None
                logger.info(
                    f"[{cid}] [routing.route] query 未显式索取参考文献, "
                    f"撤销误判的 ctype=references, 回退正文检索"
                )
    if not ambiguous and not unanchored_references:
        return outcome

    picked = _single_registry_entry(doc_registry)
    if picked is not None:
        did = str(picked.get("doc_id") or "").strip()
        name = str(picked.get("doc_name") or did or "").strip()
        refs_to_local = set(id(dec) for dec in unanchored_references)
        for dec in ambiguous + unanchored_references:
            if did and did not in dec.target_doc_ids:
                dec.target_doc_ids.append(did)
            if name and name not in dec.target_docs:
                dec.target_docs.append(name)
            if id(dec) in refs_to_local:
                dec.routes = ["local"]
                dec.rewrites = {"local": dec.rewrites.get("progressive") or dec.rewrites.get("local") or query}
        logger.info(
            f"[{cid}] [routing.route] 结构化查询无 docs/refs 但 registry 无歧义, "
            f"自动锁定 doc_id={did!r} doc_name={name!r}"
        )
        return outcome

    labels = sorted({label for dec in ambiguous for label in _metadata_structural_labels(dec)})
    if unanchored_references:
        labels.append("参考文献")
    if enable_ask:
        opts: List[str] = []
        if doc_registry:
            for i, entry in enumerate(doc_registry[:5], 1):
                name = str(entry.get("doc_name") or entry.get("doc_id") or "").strip()
                if name:
                    opts.append(f"第 {i} 篇：{name}")
        question = (
            f"你问到{'/'.join(labels)}，但我还不确定具体是哪篇文献。"
            "请指定上一轮列表中的文献编号，或直接给出文献标题。"
        )
        logger.info(
            f"[{cid}] [routing.route] metadata 缺少文献锚点, 转 ask 澄清 "
            f"labels={labels} registry={len(doc_registry) if doc_registry else 0}"
        )
        return ClarifyRequest(
            question=question,
            options=opts,
            raw={
                "source": "metadata_anchor_guard",
                "query": query,
                "labels": labels,
            },
        )

    logger.warning(
        f"[{cid}] [routing.route] metadata 缺少文献锚点但 ask 未开启, "
        f"保留原策略; 可能全库误召回 labels={labels}"
    )
    return outcome


def _apply_route_guards(
    outcome: RouteOutcome,
    *,
    query: str,
    doc_registry: Optional[List[Dict[str, str]]],
    enable_ask: bool,
    cid: str = "-",
) -> RouteOutcome:
    guarded = apply_registry_scope_guard(
        outcome,
        query=query,
        doc_registry=doc_registry,
        cid=cid,
    )
    guarded = _guard_ambiguous_metadata(
        guarded,
        query=query,
        doc_registry=doc_registry,
        enable_ask=enable_ask,
        cid=cid,
    )
    guarded = _guard_inventory_query(
        guarded,
        query=query,
        enable_ask=enable_ask,
        cid=cid,
    )
    return guarded


def _format_fc_tool_json(call: ParsedToolCall) -> str:
    """把 ParsedToolCall 格式化为可观测的 func JSON 字符串。"""
    payload = {"name": call.name, "arguments": call.arguments or {}}
    return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 反思结果
# ---------------------------------------------------------------------------

@dataclass
class ReflectVerdict:
    """反思的统一返回。"""
    needs_retry: bool = False
    decision: Optional[RouteOutcome] = None      # retry 时填: 新策略 (RouteDecision 或 MultiRouteDecision)
    partial: bool = False                        # True = 标记部分回答
    partial_note: str = ""
    cause: str = ""                              # retry 原因码 (zero/off/narrow/broad/compound)
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 文献注册块渲染 (与 agentic._format_doc_registry_block 对齐, 不直接 import 避免循环依赖)
# ---------------------------------------------------------------------------

_REGISTRY_PROMPT_LIMIT = 20


def _format_doc_registry_block(
    doc_registry: Optional[List[Dict[str, str]]],
    label: str = "上一轮检索结果中的文献列表",
) -> str:
    if not doc_registry:
        return ""
    lines = ["", "", f"【{label} (按编号; 1-based)】"]
    pinned_marks: List[str] = []
    for i, entry in enumerate(doc_registry[:_REGISTRY_PROMPT_LIMIT], 1):
        name = entry.get("doc_name") or entry.get("doc_id") or "(unknown)"
        is_pinned = bool(entry.get("pinned"))
        marker = " [pinned]" if is_pinned else ""
        if is_pinned:
            pinned_marks.append(str(i))
        lines.append(f"{i}. {name}{marker}")
    if len(doc_registry) > _REGISTRY_PROMPT_LIMIT:
        lines.append(f"... (共 {len(doc_registry)} 篇, 已截断到前 {_REGISTRY_PROMPT_LIMIT})")
    if pinned_marks:
        lines.append(
            f"(注: [pinned] 标记的是用户之前明确回指过的文献, 当本轮用户用 '上面那篇/它' "
            f"等模糊代词时, 优先指向这些 pinned 项, 编号 {', '.join(pinned_marks)})"
        )
    lines.append("(若用户回指上述某篇, 触发 local/summary 并填 refs: [编号])")
    return "\n".join(lines)


_LAST_ANSWER_PREVIEW_LIMIT = 600
_LAST_CONTEXT_PREVIEW_LIMIT = 400


def _format_session_context_block(
    *,
    last_answer: Optional[str] = None,
    last_context_preview: Optional[str] = None,
    clarify_pending: Optional[Dict[str, Any]] = None,
) -> str:
    """把上一轮 answer / context 摘要 / 待回复反问拼成一段 router 可读的提示块。

    任何为空字段会被自动跳过。该块仅用于 router LLM 判路, 不会再次出现在生成阶段。
    """
    sections: List[str] = []

    if clarify_pending and isinstance(clarify_pending, dict):
        q = str(clarify_pending.get("q") or "").strip()
        opts = clarify_pending.get("opts") or []
        if q:
            sections.append(
                f"【上一轮我向用户的反问 (用户此次发话是在回答它)】\n问题: {q}"
                + (
                    f"\n选项: {', '.join(str(o) for o in opts)}"
                    if isinstance(opts, list) and opts else ""
                )
                + "\n(指导: 把用户回答融合进一个明确的检索意图, 不要再调 ask)"
            )

    if last_answer:
        preview = last_answer.strip()
        if preview:
            if len(preview) > _LAST_ANSWER_PREVIEW_LIMIT:
                preview = preview[:_LAST_ANSWER_PREVIEW_LIMIT] + "...(截断)"
            sections.append(f"【上一轮我给出的回答 (供判断指代/复用)】\n{preview}")

    if last_context_preview:
        preview = last_context_preview.strip()
        if preview:
            if len(preview) > _LAST_CONTEXT_PREVIEW_LIMIT:
                preview = preview[:_LAST_CONTEXT_PREVIEW_LIMIT] + "...(截断)"
            sections.append(f"【上一轮检索 context 摘要】\n{preview}")

    if not sections:
        return ""
    return "\n\n" + "\n\n".join(sections)


# ---------------------------------------------------------------------------
# RoutingCore
# ---------------------------------------------------------------------------

class RoutingCore:
    """FC 化路由 + 反思的公共类。

    Args:
        router_llm: 路由用 LLM (必填); None 时直接走 heuristic_route.
        reflect_llm: 反思用 LLM; None 表示禁用反思 (reflect() 永远返回 needs_retry=False).
        current_year: 用于 prompt 中 __CURRENT_YEAR__ 替换; None=系统当前年.
        validate_fn: 注入 QueryRouter._validate_decision (避免循环 import); None 走极简路径.
        enable_multi: 暴露 multi 工具 (复合查询).
        enable_ask: 暴露 ask 工具 (反问).
        router_temperature: router LLM 温度 (默认 0).
        router_max_tokens: router LLM 输出上限 (默认 600, FC 输出短, 600 足够 multi).
        reflect_temperature: 反思温度.
        reflect_max_tokens: 反思输出上限.
        disable_thinking: **统一**思考开关 (向后兼容). 同时影响 router 与 reflect.
            - None  (默认): router 回退为 True (关思考, 直出 FC); reflect 同 disable_thinking.
            - True  : 关闭思考模式 (Qwen3 vLLM: chat_template_kwargs.enable_thinking=False).
              **router 推荐**, 与 router_system_fc.md 一致.
            - False : 开启思考模式; 仅建议在 reflect 等需要 CoT 的场景使用.
            被 router_disable_thinking / reflect_disable_thinking 单独覆盖时优先用细粒度.
        router_disable_thinking: 仅控制 router LLM 思考开关; None=回退到 disable_thinking.
        reflect_disable_thinking: 仅控制 reflect LLM 思考开关; None=回退到 disable_thinking.
        fc_fallback_to_json: FC 全失败时是否降级到 json_schema 路径 (复用现有 QueryRouter).
        heuristic_fn: 注入 _heuristic_fallback_decision 函数 (避免循环 import).
        parallel_tool_calls: 是否允许并行 tool calls. v4 设计单决策即可, 默认 False 节省 token.
    """

    def __init__(
        self,
        *,
        router_llm: Optional[LLMClient],
        reflect_llm: Optional[LLMClient] = None,
        current_year: Optional[int] = None,
        validate_fn: Optional[Callable] = None,
        json_route_fn: Optional[Callable] = None,
        heuristic_fn: Optional[Callable] = None,
        enable_multi: bool = True,
        enable_ask: bool = False,
        enable_reuse: bool = True,
        router_temperature: float = 0.0,
        router_max_tokens: int = 600,
        reflect_temperature: float = 0.0,
        reflect_max_tokens: int = 500,
        disable_thinking: Optional[bool] = None,
        router_disable_thinking: Optional[bool] = None,
        reflect_disable_thinking: Optional[bool] = None,
        fc_fallback_to_json: bool = True,
        parallel_tool_calls: Optional[bool] = False,
        history_turns: int = 1,
        routing_limits: Optional[RoutingLimits] = None,
    ) -> None:
        self.router_llm = router_llm
        self.reflect_llm = reflect_llm
        self.current_year = current_year or datetime.datetime.now().year
        self._validate_fn = validate_fn
        self._json_route_fn = json_route_fn
        self._heuristic_fn = heuristic_fn
        self.enable_multi = enable_multi
        self.enable_ask = enable_ask
        self.enable_reuse = enable_reuse
        self.router_temperature = router_temperature
        self.router_max_tokens = router_max_tokens
        self.reflect_temperature = reflect_temperature
        self.reflect_max_tokens = reflect_max_tokens
        # 统一开关保留 (向后兼容). router/reflect 细粒度开关优先, 未设时回退到统一开关。
        self.disable_thinking = disable_thinking
        self.router_disable_thinking = (
            router_disable_thinking if router_disable_thinking is not None
            else disable_thinking if disable_thinking is not None
            else True  # router 默认关思考, 只输出 FC
        )
        self.reflect_disable_thinking = (
            reflect_disable_thinking if reflect_disable_thinking is not None
            else disable_thinking
        )
        self.fc_fallback_to_json = fc_fallback_to_json
        self.parallel_tool_calls = parallel_tool_calls
        self.history_turns = max(0, int(history_turns))
        self.routing_limits = routing_limits or DEFAULT_ROUTING_LIMITS

        # 探测缓存: FC 不被后端支持时一次性关闭, 后续走 json_schema
        self._fc_unsupported: bool = False

    # ── 公共 API ──────────────────────────────────────────────────────────

    def route(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        doc_registry: Optional[List[Dict[str, str]]] = None,
        correlation_id: Optional[str] = None,
        *,
        last_answer: Optional[str] = None,
        last_context_preview: Optional[str] = None,
        clarify_pending: Optional[Dict[str, Any]] = None,
    ) -> RouteOutcome:
        """对用户 query 做检索路由决策。返回 RouteDecision / MultiRouteDecision /
        ClarifyRequest / ReuseRequest。

        Args:
            correlation_id: 可选关联 id, 透传到日志 (与 LangGraph state.correlation_id 对齐)。
                None 时不打 [cid] 前缀。
            last_answer: 上一轮的最终 answer (≤ 600 字会被截断). 喂给 router prompt 让它
                能判定"用户是否在指代上轮内容", 是 reuse 路径的关键信号.
            last_context_preview: 上一轮检索后构建的 context 摘要 (≤ 400 字). 同样用于
                帮助 reuse drilldown / continue 等模式的判断.
            clarify_pending: 上一轮 ask 工具触发的反问 {"q": ..., "opts": [...]}. 注入后
                router 知道当前用户发话是在回答反问, 应按反问意图重新路由.
        """
        t0 = time.time()
        fallback_chain: List[str] = []
        cid = correlation_id or "-"

        logger.info(
            f"[{cid}] [routing.route] begin: query={query[:80]!r}"
            + (f" (...total {len(query)} chars)" if len(query) > 80 else "")
            + f" history_msgs={len(history) if history else 0}"
            + f" doc_registry={len(doc_registry) if doc_registry else 0}"
            + f" last_answer={len(last_answer) if last_answer else 0}c"
            + f" last_ctx_preview={len(last_context_preview) if last_context_preview else 0}c"
            + f" clarify_pending={bool(clarify_pending)}"
            + f" fc_unsupported={self._fc_unsupported}"
        )

        # 0) 无 LLM: 直接 heuristic
        if self.router_llm is None:
            fallback_chain.append("no_llm")
            logger.info(f"[{cid}] [routing.route] router_llm=None, 走 heuristic 兜底")
            decision = self._heuristic(query)
            decision = _apply_route_guards(
                decision,
                query=query,
                doc_registry=doc_registry,
                enable_ask=self.enable_ask,
                cid=cid,
            )
            out = self._wrap_meta(decision, fallback_chain, t0, model="heuristic", cid=cid)
            self._log_outcome(out, t0, fallback_chain, cid=cid)
            return out

        # 1) 首选: function calling
        if not self._fc_unsupported:
            try:
                outcome, tool_name = self._route_via_fc(
                    query, history, doc_registry, cid=cid,
                    last_answer=last_answer,
                    last_context_preview=last_context_preview,
                    clarify_pending=clarify_pending,
                )
                fallback_chain.append("fc")
                out = self._wrap_meta(
                    outcome, fallback_chain, t0,
                    model=self.router_llm.model, tool_name=tool_name, cid=cid,
                )
                self._log_outcome(out, t0, fallback_chain, cid=cid, tool_name=tool_name)
                return out
            except _FCNotSupported as e:
                self._fc_unsupported = True
                logger.warning(
                    f"[{cid}] [routing.route] 后端不支持 FC, 一次性关闭并降级: {e}"
                )
                fallback_chain.append("fc_unsupported")
            except _FCParseFailure as e:
                logger.warning(
                    f"[{cid}] [routing.route] FC 解析失败, 走 legacy/heuristic 兜底: {e}"
                )
                fallback_chain.append("fc_parse_failed")
            except Exception as e:
                logger.warning(
                    f"[{cid}] [routing.route] FC 调用异常 ({type(e).__name__}: {e}), "
                    "走 legacy/heuristic 兜底"
                )
                fallback_chain.append("fc_error")

        # 2) 兜底: legacy JSON router (复用 QueryRouter.route), 再不行才 heuristic.
        if self.fc_fallback_to_json and self._json_route_fn is not None:
            try:
                fallback_chain.append("legacy_json")
                logger.info(f"[{cid}] [routing.route] FC 未产出可用结果, 走 legacy JSON router")
                decision = self._json_route_fn(
                    query, history=history, doc_registry=doc_registry,
                )
                decision = _apply_route_guards(
                    decision,
                    query=query,
                    doc_registry=doc_registry,
                    enable_ask=self.enable_ask,
                    cid=cid,
                )
                out = self._wrap_meta(
                    decision, fallback_chain, t0,
                    model=self.router_llm.model if self.router_llm else "legacy_json",
                    tool_name="legacy_json", cid=cid,
                )
                self._log_outcome(out, t0, fallback_chain, cid=cid, tool_name="legacy_json")
                return out
            except Exception as e:
                logger.warning(
                    f"[{cid}] [routing.route] legacy JSON router 失败 "
                    f"({type(e).__name__}: {e}), 走 heuristic"
                )
                fallback_chain.append("legacy_json_error")

        # 3) 兜底: heuristic
        fallback_chain.append("heuristic")
        decision = self._heuristic(query)
        decision = _apply_route_guards(
            decision,
            query=query,
            doc_registry=doc_registry,
            enable_ask=self.enable_ask,
            cid=cid,
        )
        model = self.router_llm.model if self.router_llm else "heuristic"
        out = self._wrap_meta(decision, fallback_chain, t0, model=model, cid=cid)
        self._log_outcome(out, t0, fallback_chain, cid=cid)
        return out

    def reflect(
        self,
        *,
        query: str,
        last_decision: Optional[RouteOutcome],
        results_summary: str,
        total_hits: int,
        this_round_docs: Optional[List[Dict[str, str]]] = None,
        retry_count: int = 0,
        max_retries: int = 1,
        correlation_id: Optional[str] = None,
    ) -> ReflectVerdict:
        """评估检索结果是否需要重试。"""
        t0 = time.time()
        cid = correlation_id or "-"

        logger.info(
            f"[{cid}] [routing.reflect] begin: total_hits={total_hits} "
            f"retry={retry_count}/{max_retries} this_round_docs={len(this_round_docs) if this_round_docs else 0} "
            f"summary_len={len(results_summary) if results_summary else 0}"
        )

        # 硬关闸: max_retries=0 / 无 reflect_llm / 已用尽预算 → 直接 no_retry
        if (
            self.reflect_llm is None
            or max_retries <= 0
            or retry_count >= max_retries
        ):
            reason = (
                "reflect_llm_disabled" if self.reflect_llm is None
                else "max_retries_exhausted" if retry_count >= max_retries
                else "max_retries_zero"
            )
            logger.info(
                f"[{cid}] [routing.reflect] short-circuit: skip_reason={reason}, "
                f"needs_retry=False"
            )
            return ReflectVerdict(
                needs_retry=False,
                meta={
                    "skipped": True, "skip_reason": reason,
                    "latency_ms": int((time.time() - t0) * 1000),
                    "fallback_chain": ["skip"],
                },
            )

        # 0 命中快速路径: 直接重试默认 progressive (避免无意义 LLM 调用)
        if total_hits == 0:
            prev_bias = None
            if isinstance(last_decision, RouteDecision):
                prev_bias = last_decision.retrieve_bias
            new_dec = RouteDecision(
                routes=["progressive"],
                rewrites={"progressive": query},
                retrieve_bias=prev_bias or infer_retrieve_bias_heuristic(query),
                reasoning="(reflect-empty-fast-path)",
            )
            logger.info(
                f"[{cid}] [routing.reflect] zero_hits fast-path: needs_retry=True "
                f"cause=zero new_routes=['progressive']"
            )
            return ReflectVerdict(
                needs_retry=True, decision=new_dec, cause="zero",
                meta={
                    "skipped": True, "skip_reason": "zero_hits_fast_path",
                    "latency_ms": int((time.time() - t0) * 1000),
                    "fallback_chain": ["fast_path"],
                },
            )

        # 走 FC 反思
        if not self._fc_unsupported:
            try:
                verdict = self._reflect_via_fc(
                    query=query, last_decision=last_decision,
                    results_summary=results_summary, total_hits=total_hits,
                    this_round_docs=this_round_docs, cid=cid,
                )
                verdict.meta["latency_ms"] = int((time.time() - t0) * 1000)
                verdict.meta.setdefault("fallback_chain", ["fc"])
                verdict.meta.setdefault("model", self.reflect_llm.model)
                self._log_verdict(verdict, t0, cid=cid)
                return verdict
            except _FCNotSupported as e:
                self._fc_unsupported = True
                logger.warning(
                    f"[{cid}] [routing.reflect] 后端不支持 FC, 一次性关闭: {e}"
                )
            except _FCParseFailure as e:
                logger.warning(
                    f"[{cid}] [routing.reflect] FC 解析失败, 默认 no_retry: {e}"
                )
            except Exception as e:
                logger.warning(
                    f"[{cid}] [routing.reflect] FC 异常 ({type(e).__name__}: {e}), 默认 no_retry"
                )

        # 反思失败兜底: 默认 no_retry (倾向 ok)
        logger.info(
            f"[{cid}] [routing.reflect] fallback verdict: needs_retry=False (default)"
        )
        return ReflectVerdict(
            needs_retry=False,
            meta={
                "latency_ms": int((time.time() - t0) * 1000),
                "fallback_chain": ["fc_failed_default_ok"],
            },
        )

    # ── 内部: FC 调用 ─────────────────────────────────────────────────────

    def _route_via_fc(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]],
        doc_registry: Optional[List[Dict[str, str]]],
        *,
        cid: str = "-",
        last_answer: Optional[str] = None,
        last_context_preview: Optional[str] = None,
        clarify_pending: Optional[Dict[str, Any]] = None,
    ) -> tuple:  # (outcome, tool_name)
        system_prompt = render_router_system_fc(self.current_year)
        registry_block = _format_doc_registry_block(doc_registry)
        session_block = _format_session_context_block(
            last_answer=last_answer,
            last_context_preview=last_context_preview,
            clarify_pending=clarify_pending,
        )
        # 工具列表名称提示要随 enable_* 动态调整
        tool_hint_parts = ["plan"]
        if self.enable_multi:
            tool_hint_parts.append("multi")
        if self.enable_ask:
            tool_hint_parts.append("ask")
        if self.enable_reuse:
            tool_hint_parts.append("reuse")
        tool_hint = " / ".join(tool_hint_parts)
        user_msg = (
            f"用户问题: {query}"
            f"{registry_block}"
            f"{session_block}"
            f"{compound_intent_hint(query, limits=self.routing_limits, enable_multi=self.enable_multi)}"
            f"\n\n请直接通过 function calling 调用 {tool_hint} 之一完成路由。"
            f"禁止输出思考过程、解释或任何 tool call 之外的文本。"
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        truncated = self._truncate_history(history)
        if truncated:
            messages.extend(truncated)
        messages.append({"role": "user", "content": user_msg})

        tools = router_tools(
            enable_multi=self.enable_multi,
            enable_ask=self.enable_ask,
            enable_reuse=self.enable_reuse,
            limits=self.routing_limits,
        )
        tool_names = [t["function"]["name"] for t in tools]

        chat_kwargs: Dict[str, Any] = {
            "messages": messages,
            "tools": tools,
            "tool_choice": "required",
            "temperature": self.router_temperature,
            "max_tokens": self.router_max_tokens,
            "parallel_tool_calls": self.parallel_tool_calls,
        }
        if self.router_disable_thinking is not None:
            chat_kwargs["disable_thinking"] = self.router_disable_thinking

        # think_mode 字段: True=关思考, False=开思考, None=未指定 (后端默认)
        think_mode = (
            "off" if self.router_disable_thinking is True
            else "on" if self.router_disable_thinking is False
            else "default"
        )
        logger.info(
            f"[{cid}] [routing.route] FC call: model={self.router_llm.model} "
            f"tools={tool_names} parallel={self.parallel_tool_calls} "
            f"thinking={think_mode} "
            f"history_truncated={len(truncated) if truncated else 0}"
        )

        llm_t0 = time.time()
        try:
            response = self.router_llm.chat_with_tools(**chat_kwargs)
        except Exception as e:
            if _is_fc_unsupported_error(e):
                raise _FCNotSupported(str(e)) from e
            raise
        llm_ms = int((time.time() - llm_t0) * 1000)

        tool_calls, source = parse_tool_calls(response)
        usage = response.get("usage") or {}
        finish_reason = response.get("finish_reason") or ""
        logger.info(
            f"[{cid}] [routing.route] FC response: source={source} "
            f"tool_calls={len(tool_calls)} llm_ms={llm_ms} "
            f"finish_reason={finish_reason!r} usage={usage}"
        )

        if not tool_calls:
            self._diagnose_no_tool_calls(response, cid=cid, phase="route")
            answer_preview = (response.get("answer") or "")[:300]
            raise _FCParseFailure(f"未解析到 tool_calls; answer 前 300 字: {answer_preview!r}")

        # 取第一个工具调用 (tool_choice=required + parallel=False 保证 ≤1 个)
        call = tool_calls[0]
        func_json = _format_fc_tool_json(call)
        logger.info(
            f"[{cid}] [routing.route] FC tool selected: name={call.name} "
            f"func_json={func_json}"
        )
        return self._dispatch_router_call(call, query=query, doc_registry=doc_registry, cid=cid)

    def _dispatch_router_call(
        self,
        call: ParsedToolCall,
        *,
        query: str,
        doc_registry: Optional[List[Dict[str, str]]],
        cid: str = "-",
    ) -> tuple:
        """返回 (outcome, tool_name). v4.1: 不再读 LLM 的 conf 字段 (模型填不准, 已从 schema 移除)。"""
        name = call.name
        args = call.arguments or {}

        if name == TOOL_PLAN:
            paths = args.get("paths") if isinstance(args, dict) else None
            if (
                self.enable_multi
                and isinstance(paths, list)
                and paths_should_split_to_multi(paths)
            ):
                logger.info(
                    f"[{cid}] [routing.route] plan.paths 含互斥 filter, 自动拆分为 multi "
                    f"(paths={len(paths)})"
                )
                multi_args = split_plan_args_to_multi_args(args)
                multi = build_from_multi_args(
                    multi_args,
                    query=query,
                    doc_registry=doc_registry,
                    validate_fn=self._validate_fn,
                    limits=self.routing_limits,
                )
                return _apply_route_guards(
                    multi,
                    query=query,
                    doc_registry=doc_registry,
                    enable_ask=self.enable_ask,
                    cid=cid,
                ), "multi_auto_split"

            decision = build_from_plan_args(
                args, query=query, doc_registry=doc_registry,
                validate_fn=self._validate_fn, reasoning_tag="(fc-plan)",
                limits=self.routing_limits,
            )
            return _apply_route_guards(
                decision,
                query=query,
                doc_registry=doc_registry,
                enable_ask=self.enable_ask,
                cid=cid,
            ), name

        if name == TOOL_MULTI:
            multi = build_from_multi_args(
                args, query=query, doc_registry=doc_registry,
                validate_fn=self._validate_fn,
                limits=self.routing_limits,
            )
            return _apply_route_guards(
                multi,
                query=query,
                doc_registry=doc_registry,
                enable_ask=self.enable_ask,
                cid=cid,
            ), name

        if name == TOOL_ASK:
            req = build_from_ask_args(args)
            return req, name

        if name == TOOL_REUSE:
            reuse = build_from_reuse_args(
                args, doc_registry=doc_registry, query=query,
            )
            reuse = apply_registry_scope_to_reuse(
                reuse, query=query, doc_registry=doc_registry, cid=cid,
            )
            return reuse, name

        # LLM 调用了非预期工具名 (基本不应发生, tool_choice=required + tools list 限定)
        logger.warning(
            f"[{cid}] [routing.route] 收到未识别的工具调用 name={name!r}, 退化为 heuristic"
        )
        decision = self._heuristic(query)
        return decision, "unknown_tool"

    def _reflect_via_fc(
        self,
        *,
        query: str,
        last_decision: Optional[RouteOutcome],
        results_summary: str,
        total_hits: int,
        this_round_docs: Optional[List[Dict[str, str]]],
        cid: str = "-",
    ) -> ReflectVerdict:
        system_prompt = render_reflect_system_fc(self.current_year)
        registry_block = _format_doc_registry_block(
            this_round_docs, label="本轮已检索到的文献列表",
        )

        # 把上一轮策略压成短摘要 (避免 prompt 膨胀)
        last_summary = self._summarize_last_decision(last_decision)

        user_msg = (
            f"问题: {query}{registry_block}\n\n"
            f"上一轮策略:\n{last_summary}\n\n"
            f"检索结果 (共 {total_hits} 条):\n{results_summary}\n\n"
            f"请通过 function calling 调用 ok / retry / partial 之一完成评估。"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        tools = reflect_tools(limits=self.routing_limits)
        tool_names = [t["function"]["name"] for t in tools]

        chat_kwargs: Dict[str, Any] = {
            "messages": messages,
            "tools": tools,
            "tool_choice": "required",
            "temperature": self.reflect_temperature,
            "max_tokens": self.reflect_max_tokens,
            "parallel_tool_calls": self.parallel_tool_calls,
        }
        if self.reflect_disable_thinking is not None:
            chat_kwargs["disable_thinking"] = self.reflect_disable_thinking

        think_mode = (
            "off" if self.reflect_disable_thinking is True
            else "on" if self.reflect_disable_thinking is False
            else "default"
        )
        logger.info(
            f"[{cid}] [routing.reflect] FC call: model={self.reflect_llm.model} "
            f"tools={tool_names} thinking={think_mode} "
            f"last_strategy={last_summary[:120]!r}"
        )

        llm_t0 = time.time()
        try:
            response = self.reflect_llm.chat_with_tools(**chat_kwargs)
        except Exception as e:
            if _is_fc_unsupported_error(e):
                raise _FCNotSupported(str(e)) from e
            raise
        llm_ms = int((time.time() - llm_t0) * 1000)

        tool_calls, source = parse_tool_calls(response)
        usage = response.get("usage") or {}
        finish_reason = response.get("finish_reason") or ""
        logger.info(
            f"[{cid}] [routing.reflect] FC response: source={source} "
            f"tool_calls={len(tool_calls)} llm_ms={llm_ms} "
            f"finish_reason={finish_reason!r} usage={usage}"
        )
        # 隐式 CoT: 同 route, 仅打日志不进决策
        rc = response.get("reasoning_content") or ""
        if rc:
            rc_preview = rc[:400] + ("..." if len(rc) > 400 else "")
            logger.info(
                f"[{cid}] [routing.reflect] LLM thinking ({len(rc)} chars): {rc_preview}"
            )

        if not tool_calls:
            self._diagnose_no_tool_calls(response, cid=cid, phase="reflect")
            answer_preview = (response.get("answer") or "")[:200]
            raise _FCParseFailure(f"reflect 未解析到 tool_calls; answer: {answer_preview!r}")

        call = tool_calls[0]
        func_json = _format_fc_tool_json(call)
        logger.info(
            f"[{cid}] [routing.reflect] FC tool selected: name={call.name} "
            f"func_json={func_json}"
        )
        return self._dispatch_reflect_call(
            call, query=query, doc_registry=this_round_docs, cid=cid,
        )

    def _dispatch_reflect_call(
        self,
        call: ParsedToolCall,
        *,
        query: str,
        doc_registry: Optional[List[Dict[str, str]]],
        cid: str = "-",
    ) -> ReflectVerdict:
        name = call.name
        args = call.arguments or {}

        if name == TOOL_OK:
            return ReflectVerdict(
                needs_retry=False,
                meta={"tool": name},
            )

        if name == TOOL_PARTIAL:
            return ReflectVerdict(
                needs_retry=False, partial=True,
                partial_note=str(args.get("note", ""))[:200],
                meta={"tool": name},
            )

        if name == TOOL_RETRY:
            cause = str(args.get("cause", "")).strip() or "off"
            new_decision = build_reflect_retry(
                args, query=query, doc_registry=doc_registry,
                validate_fn=self._validate_fn,
                limits=self.routing_limits,
            )
            return ReflectVerdict(
                needs_retry=True, decision=new_decision, cause=cause,
                meta={"tool": name},
            )

        logger.warning(
            f"[{cid}] [routing.reflect] 收到未识别的工具调用 name={name!r}, 默认 no_retry"
        )
        return ReflectVerdict(
            needs_retry=False,
            meta={"tool": name, "unknown": True, "fallback_chain": ["unknown_tool"]},
        )

    # ── 辅助方法 ──────────────────────────────────────────────────────────

    def _heuristic(self, query: str) -> RouteDecision:
        """无 LLM 兜底: 优先用注入的 heuristic_fn (= _heuristic_fallback_decision)。"""
        if self._heuristic_fn is not None:
            try:
                return self._heuristic_fn(query, self.current_year)
            except Exception as e:
                logger.warning(f"[routing] heuristic_fn 失败, 走零字段 fallback: {e}")
        # 最低保证
        return RouteDecision(
            routes=["progressive"], rewrites={"progressive": query},
            reasoning="(zero-fallback)",
        )

    def _truncate_history(
        self, history: Optional[List[Dict[str, str]]],
    ) -> Optional[List[Dict[str, str]]]:
        if not history or self.history_turns <= 0:
            return None
        max_msgs = 2 * self.history_turns
        tail = list(history[-max_msgs:])
        while tail and tail[0].get("role") != "user":
            tail = tail[1:]
        return tail or None

    def _summarize_last_decision(
        self, last: Optional[RouteOutcome],
    ) -> str:
        """把上一轮策略压成 ≤200 字提示, 供反思 prompt 引用。"""
        if last is None:
            return "(无)"
        if isinstance(last, RouteDecision):
            parts = [
                f"路径: {', '.join(last.routes or [])}",
                f"改写: {dict(last.rewrites or {})}",
            ]
            filters = {
                k: v for k, v in {
                    "chunk_type": last.chunk_type,
                    "target_docs": last.target_docs,
                    "fig_refs": last.fig_refs,
                    "table_refs": last.table_refs,
                    "page_refs": last.page_refs,
                    "paragraph_refs": last.paragraph_refs,
                    "entities": last.entities,
                    "time": last.time,
                    "retrieve_bias": last.retrieve_bias,
                }.items() if v
            }
            if filters:
                parts.append(f"过滤: {filters}")
            return " | ".join(parts)
        if isinstance(last, MultiRouteDecision):
            sub_summaries: List[str] = []
            for sub in last.subqueries:
                sub_summaries.append(f"[{sub.id}] " + self._summarize_last_decision(sub.decision))
            return f"(multi, synth={last.synth_hint!r}) " + " ;; ".join(sub_summaries)
        if isinstance(last, ClarifyRequest):
            return f"(ask) {last.question!r}"
        if isinstance(last, ReuseRequest):
            return f"(reuse mode={last.mode}) {last.op!r}"
        return str(last)

    def _wrap_meta(
        self, outcome: RouteOutcome, fallback_chain: List[str], t0: float,
        *, model: str, tool_name: str = "",
        cid: str = "-",
    ) -> RouteOutcome:
        meta = {
            "fallback_chain": fallback_chain,
            "latency_ms": int((time.time() - t0) * 1000),
            "model": model,
            "tool_name": tool_name,
            "cid": cid,
        }
        if isinstance(outcome, RouteDecision):
            # RouteDecision 是 pydantic 模型, 通过 object.__setattr__ 旁挂私有属性供观测层读取
            try:
                object.__setattr__(outcome, "_routing_meta", meta)
            except Exception:
                pass
        elif isinstance(outcome, MultiRouteDecision):
            outcome.raw.setdefault("_meta", meta)
        elif isinstance(outcome, ClarifyRequest):
            outcome.raw.setdefault("_meta", meta)
        elif isinstance(outcome, ReuseRequest):
            outcome.raw.setdefault("_meta", meta)
        return outcome

    def _diagnose_no_tool_calls(
        self,
        response: Dict[str, Any],
        *,
        cid: str = "-",
        phase: str = "route",
    ) -> None:
        """tool_calls 为空时打详细诊断, 帮助定位 vLLM 配置 / max_tokens / 输出格式问题。

        最常见的 4 类问题:
          1. vLLM 没配 --tool-call-parser / --reasoning-parser
             → 表现: finish_reason='stop', completion_tokens 较大, content 或 raw 里能看到
               <tool_call>...</tool_call> 或 <think>...</think> 文本块未被剥离
          2. max_tokens 太小, 思考阶段就被截断
             → 表现: finish_reason='length', reasoning_content 已有内容但 tool_call 缺失
          3. 模型走了不规范的输出格式 (没用 <tool_call> 而是裸 JSON 或 markdown 代码块)
             → 表现: content 里有 ```json {...} ``` 之类的, fc_parser 应该能 fallback 解析,
               若仍失败说明格式更怪
          4. tool_choice='required' 被后端静默忽略, 模型走了普通对话
             → 表现: content 里是自然语言而非任何结构化输出
        """
        msg = response.get("message") or {}
        finish_reason = response.get("finish_reason") or ""
        rc = response.get("reasoning_content") or ""
        ans = response.get("answer") or ""
        usage = response.get("usage") or {}

        # 把 message 的所有 key 列出来 (帮助识别后端自带的非标准字段, 如 vLLM 旧版本的 reasoning 字段)
        msg_keys = list(msg.keys()) if isinstance(msg, dict) else []

        # 把 raw 响应里 choices[0] 的完整 keys 列出来
        raw = response.get("raw") or {}
        choices = raw.get("choices") if isinstance(raw, dict) else None
        first_choice_keys: List[str] = []
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            first_choice_keys = list(choices[0].keys())

        # content / reasoning_content 的预览
        content_preview = (ans or "")[:400]
        rc_preview = (rc or "")[:400]

        logger.warning(
            f"[{cid}] [routing.{phase}] ❌ tool_calls 为空, 诊断信息:\n"
            f"  finish_reason = {finish_reason!r} "
            f"(常见: stop=正常结束, length=被 max_tokens 截断, tool_calls=正常)\n"
            f"  usage         = {usage}\n"
            f"  message.keys  = {msg_keys}\n"
            f"  choice.keys   = {first_choice_keys}\n"
            f"  content       ({len(ans)} chars) = {content_preview!r}"
            f"{'...' if len(ans) > 400 else ''}\n"
            f"  reasoning_content ({len(rc)} chars) = {rc_preview!r}"
            f"{'...' if len(rc) > 400 else ''}"
        )

        # 给出具体的诊断结论 + 修复建议
        completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        if finish_reason == "length":
            logger.warning(
                f"[{cid}] [routing.{phase}] 💡 finish_reason=length: 输出被 max_tokens 截断. "
                f"completion_tokens={completion_tokens}. 建议调大 routing.router_max_tokens; "
                f"router 应关思考 (router_disable_thinking=true) 且 prompt 禁止思考输出."
            )
        elif rc and not ans and not first_choice_keys:
            logger.warning(
                f"[{cid}] [routing.{phase}] 💡 仅有 reasoning_content 而无 tool_call: "
                f"模型整段都在 <think> 里, 没输出 tool_call. 检查: "
                f"(1) vLLM --reasoning-parser 是否启用 (没启用时 thinking 不进 reasoning_content); "
                f"(2) vLLM --tool-call-parser hermes 是否启用; "
                f"(3) max_tokens 是否够大让模型走完 think→tool_call 两阶段."
            )
        elif completion_tokens and completion_tokens > 100 and not ans and not rc:
            logger.warning(
                f"[{cid}] [routing.{phase}] 💡 输出了 {completion_tokens} tokens 但 content 与 "
                f"reasoning_content 都为空: 模型输出可能没被任何 parser 接住. "
                f"最可能: vLLM 启动缺 --tool-call-parser hermes (且模型用了 <think> 但没配 reasoning-parser, "
                f"hermes parser 看到 <think> 前缀就识别失败). "
                f"修复: vLLM 启动加 --tool-call-parser hermes --reasoning-parser deepseek_r1 一起."
            )
        elif ans and not rc:
            logger.warning(
                f"[{cid}] [routing.{phase}] 💡 content 有内容但解析失败: 模型可能用了非 hermes 格式 "
                f"(裸 JSON / markdown 代码块 / 普通自然语言). 检查上面 content 预览, 若是自然语言说明 "
                f"tool_choice='required' 被后端忽略, 检查 vLLM 是否升到 ≥0.6.x 支持 required 模式."
            )

        # DEBUG 级: 打 raw 响应前 2K 字, 完整看模型实际吐了什么 (开 DEBUG log 才出现)
        if logger.isEnabledFor(logging.DEBUG):
            try:
                import json as _json
                raw_dump = _json.dumps(raw, ensure_ascii=False)[:2000]
            except Exception:
                raw_dump = repr(raw)[:2000]
            logger.debug(
                f"[{cid}] [routing.{phase}] raw 响应 (前 2000 字): {raw_dump}"
            )

    def _log_outcome(
        self,
        outcome: RouteOutcome,
        t0: float,
        fallback_chain: List[str],
        *,
        cid: str = "-",
        tool_name: str = "",
    ) -> None:
        """统一打 route() 末尾日志, 把决策的核心字段一行打出, 便于人工读。"""
        ms = int((time.time() - t0) * 1000)
        chain = "→".join(fallback_chain) if fallback_chain else "none"

        if isinstance(outcome, RouteDecision):
            filters_summary = {
                k: v for k, v in {
                    "ctype": outcome.chunk_type,
                    "docs": outcome.target_docs,
                    "figs": outcome.fig_refs,
                    "tabs": outcome.table_refs,
                    "pages": outcome.page_refs,
                    "paras": outcome.paragraph_refs,
                    "ents": outcome.entities,
                    "time": outcome.time,
                }.items() if v
            }
            rerank_src = (
                "rewrite_kw"
                if outcome.rerank_mode is True
                else "user_query"
            )
            logger.info(
                f"[{cid}] [routing.route] DONE: type=RouteDecision "
                f"chain={chain} tool={tool_name or 'n/a'} "
                f"routes={outcome.routes} rewrites={outcome.rewrites} "
                f"retrieve_bias={outcome.retrieve_bias or '-'} "
                f"rerank_mode={outcome.rerank_mode!r} rerank_query_src={rerank_src} "
                f"filters={filters_summary} total_ms={ms}"
            )
        elif isinstance(outcome, MultiRouteDecision):
            sub_brief = [
                f"{s.id}={s.decision.routes}"
                + (f"+docs={s.decision.target_docs}" if s.decision.target_docs else "")
                + (
                    f"+rerank_mode={s.decision.rerank_mode!r}"
                    if s.decision.rerank_mode is True
                    else ""
                )
                for s in outcome.subqueries
            ]
            logger.info(
                f"[{cid}] [routing.route] DONE: type=MultiRouteDecision "
                f"chain={chain} tool={tool_name} "
                f"subs={sub_brief} synth={outcome.synth_hint!r} total_ms={ms}"
            )
        elif isinstance(outcome, ClarifyRequest):
            logger.info(
                f"[{cid}] [routing.route] DONE: type=ClarifyRequest "
                f"chain={chain} tool={tool_name} "
                f"question={outcome.question[:100]!r} opts={outcome.options} total_ms={ms}"
            )
        elif isinstance(outcome, ReuseRequest):
            refs_brief = (
                f" refs={outcome.doc_refs} docs={outcome.target_docs}"
                if outcome.doc_refs or outcome.target_docs
                else ""
            )
            logger.info(
                f"[{cid}] [routing.route] DONE: type=ReuseRequest "
                f"chain={chain} tool={tool_name} "
                f"mode={outcome.mode} op={outcome.op[:80]!r}{refs_brief} total_ms={ms}"
            )
        else:
            logger.info(
                f"[{cid}] [routing.route] DONE: type={type(outcome).__name__} "
                f"chain={chain} total_ms={ms}"
            )

    def _log_verdict(
        self,
        verdict: "ReflectVerdict",
        t0: float,
        *,
        cid: str = "-",
    ) -> None:
        ms = int((time.time() - t0) * 1000)
        if verdict.needs_retry and verdict.decision is not None:
            dec = verdict.decision
            if isinstance(dec, RouteDecision):
                brief = f"routes={dec.routes} rewrites={dec.rewrites} docs={dec.target_docs}"
            elif isinstance(dec, MultiRouteDecision):
                brief = f"multi sub_count={len(dec.subqueries)}"
            else:
                brief = f"type={type(dec).__name__}"
            logger.info(
                f"[{cid}] [routing.reflect] DONE: needs_retry=True cause={verdict.cause} "
                f"new_strategy=({brief}) total_ms={ms}"
            )
        elif verdict.partial:
            logger.info(
                f"[{cid}] [routing.reflect] DONE: partial=True "
                f"note={verdict.partial_note[:80]!r} total_ms={ms}"
            )
        else:
            logger.info(
                f"[{cid}] [routing.reflect] DONE: needs_retry=False (ok) total_ms={ms}"
            )


# ---------------------------------------------------------------------------
# 异常 / 错误检测
# ---------------------------------------------------------------------------

class _FCNotSupported(RuntimeError):
    """LLM 后端不支持 tools / tool_choice (一次性关闭 FC 后续走兜底)."""


class _FCParseFailure(RuntimeError):
    """LLM 返回了内容但解析不到 tool_calls."""


_FC_UNSUPPORTED_HINTS = (
    "tools is not supported",
    "tool_choice",
    "function calling",
    "not support tool",
    "unknown field",
    "unsupported parameter",
    "unrecognized request argument",
)


def _is_fc_unsupported_error(err: Exception) -> bool:
    msg = str(err).lower()
    return any(h in msg for h in _FC_UNSUPPORTED_HINTS)


# ---------------------------------------------------------------------------
# 便利工厂: 从既有 QueryRouter / heuristic 注入依赖, 减少调用方样板
# ---------------------------------------------------------------------------

def build_routing_core_from_query_router(
    query_router: Any,                           # pipeline.retrieval.agentic.QueryRouter
    reflect_llm: Optional[LLMClient] = None,
    *,
    enable_multi: bool = True,
    enable_ask: bool = False,
    enable_reuse: bool = True,
    fc_fallback_to_json: bool = True,
    disable_thinking: Optional[bool] = None,
    router_disable_thinking: Optional[bool] = None,
    reflect_disable_thinking: Optional[bool] = None,
    router_max_tokens: Optional[int] = None,
    reflect_max_tokens: Optional[int] = None,
    parallel_tool_calls: Optional[bool] = False,
    history_turns: int = 1,
    routing_limits: Optional[RoutingLimits] = None,
) -> "RoutingCore":
    """从现有 QueryRouter 实例构造 RoutingCore (复用 LLMClient + _validate_decision + heuristic)。

    建议在 LangGraph 工厂里这样初始化, 避免循环 import 与重复构造 LLMClient.

    思考开关优先级 (从高到低):
      router_disable_thinking / reflect_disable_thinking (细粒度) > disable_thinking (统一)

    max_tokens 优先级:
      router_max_tokens / reflect_max_tokens 显式参数 > query_router.max_tokens (router 端兜底)
                                                     > RoutingCore 内置默认 (500)
    """
    # 延迟 import 防止循环
    from ..retrieval.agentic import QueryRouter as _QR, _heuristic_fallback_decision

    if not isinstance(query_router, _QR):
        raise TypeError(f"query_router 必须是 QueryRouter 实例, 实得 {type(query_router).__name__}")

    # router_max_tokens: 显式参数 > query_router.max_tokens > 内置默认 600
    effective_router_max = (
        int(router_max_tokens) if router_max_tokens is not None
        else max(query_router.max_tokens, 600)
    )
    # reflect_max_tokens: 显式参数 > 内置默认 500 (无 query_router 可继承, reflect 独立 LLM)
    rc_kwargs: Dict[str, Any] = {
        "router_llm": query_router.llm,
        "reflect_llm": reflect_llm,
        "current_year": query_router.current_year,
        "validate_fn": query_router._validate_decision,
        "json_route_fn": query_router.route,
        "heuristic_fn": _heuristic_fallback_decision,
        "enable_multi": enable_multi,
        "enable_ask": enable_ask,
        "enable_reuse": enable_reuse,
        "router_temperature": query_router.temperature,
        "router_max_tokens": effective_router_max,
        "disable_thinking": disable_thinking,
        "router_disable_thinking": router_disable_thinking,
        "reflect_disable_thinking": reflect_disable_thinking,
        "fc_fallback_to_json": fc_fallback_to_json,
        "parallel_tool_calls": parallel_tool_calls,
        "history_turns": history_turns,
    }
    if reflect_max_tokens is not None:
        rc_kwargs["reflect_max_tokens"] = int(reflect_max_tokens)
    if routing_limits is not None:
        rc_kwargs["routing_limits"] = routing_limits
    return RoutingCore(**rc_kwargs)
