"""把 FC 输出 (plan / multi / ask) 转成 RouteDecision (或 MultiRouteDecision / ClarifyRequest)。

关键设计:
1. **不绕过** 现有 QueryRouter._validate_decision: FC schema 已经做了 90% 硬约束 (oneOf/enum/anyOf),
   但 doc_refs→target_docs 映射、time 字符串解析、启发式补足 page/paragraph/entities 这些
   归一化逻辑仍然复用现有代码, 避免行为漂移;
2. **保持向后兼容**: 单意图统一返回 RouteDecision (pipeline.models 现有模型), 复合查询返回
   新增的 MultiRouteDecision dataclass (轻量, 不污染 pydantic 模型);
3. **schema 短字段 → RouteDecision 长字段** 的映射在 _path_to_raw_decision 函数里集中处理,
   字段名变更只改这一处。

字段映射约定 (与 fc_schema.py 顶部注释保持同步):
  paths[].t        →  routes 元素
  paths[].kw       →  rewrites[route]  (列表 join 为字符串)
  paths[].docs     →  target_docs
  paths[].refs     →  doc_refs (经 doc_registry 查表后写入 target_docs)
  paths[].figs     →  fig_refs
  paths[].tabs     →  table_refs
  paths[].pages    →  page_refs
  paths[].paras    →  paragraph_refs
  paths[].ents     →  entities
  paths[].ctype    →  chunk_type
  顶层 time        →  time
  顶层 retrieve_bias → retrieve_bias (hybrid 权重偏好)
  顶层 rerank_mode   → rerank_mode (true=精排用 kw rewrite; 省略=用户原话)

  reuse.refs       → doc_refs → target_doc_ids / target_docs (经 doc_registry 查表)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from ..models import RouteDecision
from ..retrieval.hybrid_weights import normalize_retrieve_bias
from .limits import (
    DEFAULT_ROUTING_LIMITS,
    RoutingLimits,
    paths_should_split_to_multi,
    split_plan_args_to_multi_args,
    trim_paths,
    trim_subs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 复合查询的轻量数据结构 (不进 pydantic, 避免污染主模型)
# ---------------------------------------------------------------------------

@dataclass
class SubqueryDecision:
    """一个子查询的检索决策。"""
    id: str
    decision: RouteDecision
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiRouteDecision:
    """复合查询的整体决策, 含 2-3 个 SubqueryDecision。

    Note: conf 字段 v4.1 已废弃 (LLM 填不准, 已从 schema 移除); 字段保留默认 0.0 仅为
    向后兼容, 不再有调用方写入或消费。
    """
    subqueries: List[SubqueryDecision] = field(default_factory=list)
    synth_hint: str = ""
    conf: float = 0.0   # deprecated, kept for backward-compat
    raw: Dict[str, Any] = field(default_factory=dict)

    def is_compound(self) -> bool:
        return len(self.subqueries) >= 2


@dataclass
class ClarifyRequest:
    """ask 工具触发的反问请求, 让 graph 跳过本轮检索。"""
    question: str
    options: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


# reuse 工具支持的 7 种模式 (与 fc_schema.REUSE_TOOL.enum 严格一致)
REUSE_MODES = frozenset({
    "reformat", "drilldown", "metasession",
    "confirm", "continue", "chitchat", "out_of_scope",
})

# 不依赖上轮 context 即可生成的 mode (graph 层用来判断是否必须从 session_meta 取 last_context)
REUSE_STANDALONE_MODES = frozenset({"chitchat", "out_of_scope"})


@dataclass
class ReuseRequest:
    """reuse 工具触发的"不检索直接答"请求。

    与 ClarifyRequest 不同, ReuseRequest 仍然会调用生成 LLM —— 只是跳过 retrieve /
    reranker / reflect, 让 generate 节点直接基于 last_context / last_answer 重写或闲聊。

    mode 决定生成节点拼 prompt 的方式 (见 langgraph_agent._make_reuse_node).
    """
    mode: str = "reformat"
    op: str = ""
    doc_refs: List[int] = field(default_factory=list)
    target_doc_ids: List[str] = field(default_factory=list)
    target_docs: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def is_standalone(self) -> bool:
        """True = 不需要上轮 context (chitchat / out_of_scope), 直接由 LLM 凭 query 答。"""
        return self.mode in REUSE_STANDALONE_MODES


# reuse 可锁定具体文献的模式 (chitchat/oos/confirm 不需要 refs)
REUSE_DOC_SCOPED_MODES = frozenset({"reformat", "drilldown", "continue", "metasession"})


# 上层调用约定: route() 的返回类型
RouteOutcome = Union[RouteDecision, MultiRouteDecision, ClarifyRequest, ReuseRequest]


# ---------------------------------------------------------------------------
# FC 短字段 → router JSON 老字段 的反序列化
# ---------------------------------------------------------------------------

# 路径类型枚举 (与 agentic.ROUTE_* 一致, 不直接 import 避免循环依赖)
_ROUTE_SUMMARY = "summary"
_ROUTE_PROGRESSIVE = "progressive"
_ROUTE_LOCAL = "local"
_ROUTE_METADATA = "metadata"
_VALID_ROUTES = (_ROUTE_SUMMARY, _ROUTE_PROGRESSIVE, _ROUTE_LOCAL, _ROUTE_METADATA)


def _safe_str_list(raw: Any, *, upper: bool = False) -> List[str]:
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


# 元话语/检索 framing: 写进 kw 会在大库里 BM25 命中海量摘要/关键词块, 与路径语义重复.
_REWRITE_META_KW = frozenset({
    "文献", "论文", "研究", "资料", "期刊", "文章", "学术", "报道", "著作",
    "有没有", "哪些", "是否存在", "是否", "请问", "查询", "检索", "查找", "寻找",
    "关于", "相关", "方面", "领域", "方向", "工作", "内容", "情况",
    "总结", "汇总", "对比", "概述", "盘点", "发现",
    "是什么", "为什么", "如何", "怎么", "多少", "怎样", "请告诉",
    "介绍", "说明", "讲述", "描述",
})


def sanitize_rewrite_keywords(keywords: List[str]) -> List[str]:
    """从 router kw 数组里去掉元话语/framing 词, 保留主体+问点实体."""
    out: List[str] = []
    seen: set = set()
    for kw in keywords:
        s = (kw or "").strip()
        if not s or s in _REWRITE_META_KW:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _safe_int_list(raw: Any) -> List[int]:
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


def _parse_rerank_mode(raw: Any) -> Optional[bool]:
    """FC rerank_mode: 仅 true 生效; false/省略/非法 → None (用用户原话)。"""
    return True if raw is True else None


def _apply_top_level_extras(
    raw: Dict[str, Any],
    args: Dict[str, Any],
) -> None:
    """把 plan/sub 顶层的 retrieve_bias / rerank_mode 写入 raw decision。"""
    retrieve_bias = normalize_retrieve_bias(args.get("retrieve_bias"))
    if retrieve_bias:
        raw["retrieve_bias"] = retrieve_bias
    rerank_mode = _parse_rerank_mode(args.get("rerank_mode"))
    if rerank_mode is True:
        raw["rerank_mode"] = True


def _path_to_raw_decision(
    paths: List[Dict[str, Any]],
    *,
    time_str: str = "",
) -> Dict[str, Any]:
    """把 FC 的 paths 数组合并成 router 老 JSON 格式 (供 _validate_decision 二次校验)。

    输出形如:
        {
          "routes": ["progressive", "metadata"],
          "rewrites": {"progressive": ["kw1","kw2"]},   # 数组形态, 与 router JSON 一致
          "filters": {
              "chunk_type": "image",
              "target_docs": [...],
              "doc_refs": [...],
              "fig_refs": [...], "table_refs": [...],
              "page_refs": [...], "paragraph_refs": [...],
              "entities": [...],
              "time": "2020-2026"
          }
        }
    """
    routes: List[str] = []
    rewrites: Dict[str, List[str]] = {}

    target_docs: List[str] = []
    doc_refs: List[int] = []
    fig_refs: List[str] = []
    table_refs: List[str] = []
    page_refs: List[int] = []
    paragraph_refs: List[int] = []
    entities: List[str] = []
    chunk_type: Optional[str] = None
    expand_neighbors: List[str] = []

    for path in paths or []:
        if not isinstance(path, dict):
            continue
        t = path.get("t")
        if t not in _VALID_ROUTES:
            logger.warning(f"[decision_builder] 未知 path.t={t!r}, 跳过")
            continue
        if t not in routes:
            routes.append(t)

        # ── kw → rewrites (metadata 不应有 kw, schema 已经禁止; 防御性再过滤一次) ──
        if t != _ROUTE_METADATA:
            kw_list = sanitize_rewrite_keywords(_safe_str_list(path.get("kw")))
            if kw_list:
                # rewrites 用列表形态, _validate_decision 会自动 join
                rewrites[t] = kw_list

        # ── target_docs / refs (local / metadata 都可有) ──
        for d in _safe_str_list(path.get("docs")):
            if d not in target_docs:
                target_docs.append(d)
        for r in _safe_int_list(path.get("refs")):
            if r not in doc_refs:
                doc_refs.append(r)

        # ── metadata 专属硬过滤字段 ──
        if t == _ROUTE_METADATA:
            for x in _safe_str_list(path.get("figs"), upper=True):
                if x not in fig_refs:
                    fig_refs.append(x)
            for x in _safe_str_list(path.get("tabs"), upper=True):
                if x not in table_refs:
                    table_refs.append(x)
            for x in _safe_int_list(path.get("pages")):
                if x not in page_refs:
                    page_refs.append(x)
            for x in _safe_int_list(path.get("paras")):
                if x not in paragraph_refs:
                    paragraph_refs.append(x)
            for x in _safe_str_list(path.get("ents")):
                if x not in entities:
                    entities.append(x)

        # ── ctype: progressive / local / metadata 均可带 (4 种 chunk 类型) ──
        ct_raw = path.get("ctype")
        if isinstance(ct_raw, str) and ct_raw.lower() in (
            "image", "table", "equation", "references"
        ):
            chunk_type = ct_raw.lower()

        # ── expand: 邻域扩展模式 (各 path 合并去重) ──
        for m in _safe_str_list(path.get("expand")):
            ml = m.lower()
            if ml in ("assets", "adjacent", "page", "similar") and ml not in expand_neighbors:
                expand_neighbors.append(ml)

    filters: Dict[str, Any] = {}
    if chunk_type:
        filters["chunk_type"] = chunk_type
    if target_docs:
        filters["target_docs"] = target_docs
    if doc_refs:
        filters["doc_refs"] = doc_refs
    if fig_refs:
        filters["fig_refs"] = fig_refs
    if table_refs:
        filters["table_refs"] = table_refs
    if page_refs:
        filters["page_refs"] = page_refs
    if paragraph_refs:
        filters["paragraph_refs"] = paragraph_refs
    if entities:
        filters["entities"] = entities
    if time_str and time_str.strip():
        filters["time"] = time_str.strip()

    raw: Dict[str, Any] = {"routes": routes, "rewrites": rewrites}
    if filters:
        raw["filters"] = filters
    if expand_neighbors:
        raw["expand_neighbors"] = expand_neighbors
    return raw


# ---------------------------------------------------------------------------
# 顶层 API
# ---------------------------------------------------------------------------

def build_from_plan_args(
    args: Dict[str, Any],
    *,
    query: str,
    doc_registry: Optional[List[Dict[str, str]]] = None,
    validate_fn: Optional[callable] = None,
    reasoning_tag: str = "(fc-plan)",
    limits: Optional[RoutingLimits] = None,
) -> RouteDecision:
    """把 plan 工具的 arguments 转成 RouteDecision。

    Args:
        args: plan 工具的 arguments 字典 (含 paths/time)
        query: 原始用户 query, 用于 fallback rewrite 与启发式补齐
        doc_registry: 上一轮文献列表; doc_refs 映射要用它查表
        validate_fn: 注入 QueryRouter._validate_decision (4-arg)。注入时复用所有归一化与防御逻辑;
            None 表示走最小路径 (仅依赖 schema 强约束), 仅用于单测/极简场景。
        reasoning_tag: 写入 RouteDecision.reasoning 的标签, 便于事后审计 ("(fc-plan)"/"(fc-retry)"/...)
    """
    paths = args.get("paths") if isinstance(args, dict) else None
    if not isinstance(paths, list) or not paths:
        logger.warning(f"[decision_builder] plan.paths 为空, 退化为单 progressive")
        paths = [{"t": _ROUTE_PROGRESSIVE, "kw": [query]}]

    lim = limits or DEFAULT_ROUTING_LIMITS
    paths = trim_paths(paths, max_paths=lim.max_paths_per_sub)

    time_str = ""
    if isinstance(args, dict):
        t_raw = args.get("time")
        if isinstance(t_raw, str):
            time_str = t_raw.strip()

    raw = _path_to_raw_decision(paths, time_str=time_str)
    if isinstance(args, dict):
        _apply_top_level_extras(raw, args)

    if validate_fn is not None:
        decision = validate_fn(raw, "", query, doc_registry=doc_registry)
    else:
        # 极简路径: 直接用 RouteDecision 构造, 不做 doc_refs→docs / 启发式补齐
        decision = _minimal_route_decision(raw, query)

    # 标记来源 (router 直出 FC; LLM 不应输出 reasoning/thinking 文本)
    if not decision.reasoning:
        decision.reasoning = reasoning_tag
    elif reasoning_tag not in decision.reasoning:
        decision.reasoning = f"{reasoning_tag} {decision.reasoning}".strip()

    return decision


def build_from_multi_args(
    args: Dict[str, Any],
    *,
    query: str,
    doc_registry: Optional[List[Dict[str, str]]] = None,
    validate_fn: Optional[callable] = None,
    limits: Optional[RoutingLimits] = None,
) -> MultiRouteDecision:
    """把 multi 工具的 arguments 转成 MultiRouteDecision (含 2+ 个 SubqueryDecision)。"""
    lim = limits or DEFAULT_ROUTING_LIMITS
    subs_raw = args.get("subs") if isinstance(args, dict) else None
    truncated_note = ""
    if isinstance(subs_raw, list):
        original_sub_count = len(subs_raw)
        subs_raw = trim_subs(subs_raw, max_subqueries=lim.max_subqueries)
        if original_sub_count > len(subs_raw):
            truncated_note = (
                f"本轮复合查询最多处理 {len(subs_raw)} 个子意图; "
                f"原始识别到 {original_sub_count} 个, 其余意图请用户下一轮继续追问。"
            )
    if not isinstance(subs_raw, list) or len(subs_raw) < 2:
        logger.warning(
            f"[decision_builder] multi.subs 不足 2 个 (len={len(subs_raw) if isinstance(subs_raw, list) else 'NA'}), "
            f"退化为单 plan"
        )
        # 退化: 把单一 sub (或空) 当成 plan 走单决策, 包装成 1 个 sub 的 MultiRouteDecision
        single_args = subs_raw[0] if isinstance(subs_raw, list) and subs_raw else {"paths": [{"t": "progressive", "kw": [query]}]}
        sub_dec = build_from_plan_args(
            single_args, query=query, doc_registry=doc_registry,
            validate_fn=validate_fn, reasoning_tag="(fc-multi-degraded)",
            limits=lim,
        )
        synth_hint = str(args.get("synth", "")) if isinstance(args, dict) else ""
        if truncated_note:
            synth_hint = f"{synth_hint}\n{truncated_note}".strip()
        return MultiRouteDecision(
            subqueries=[SubqueryDecision(id="sub1", decision=sub_dec, raw=single_args)],
            synth_hint=synth_hint,
            raw=args if isinstance(args, dict) else {},
        )

    subqueries: List[SubqueryDecision] = []
    for i, sub_args in enumerate(subs_raw):
        if not isinstance(sub_args, dict):
            continue
        sub_id = str(sub_args.get("id") or f"sub{i + 1}")
        sub_dec = build_from_plan_args(
            sub_args, query=query, doc_registry=doc_registry,
            validate_fn=validate_fn, reasoning_tag=f"(fc-multi:{sub_id})",
            limits=lim,
        )
        subqueries.append(SubqueryDecision(id=sub_id, decision=sub_dec, raw=sub_args))

    synth_hint = str(args.get("synth", ""))
    if truncated_note:
        synth_hint = f"{synth_hint}\n{truncated_note}".strip()
    return MultiRouteDecision(
        subqueries=subqueries,
        synth_hint=synth_hint,
        raw=args,
    )


def build_from_ask_args(args: Dict[str, Any]) -> ClarifyRequest:
    """把 ask 工具的 arguments 转成 ClarifyRequest。"""
    q = ""
    opts: List[str] = []
    if isinstance(args, dict):
        q = str(args.get("q") or "").strip()
        opts = _safe_str_list(args.get("opts"))
    if not q:
        q = "请补充更多上下文 (例如指定要查的文献或主题)。"
    return ClarifyRequest(question=q, options=opts, raw=args if isinstance(args, dict) else {})


def _resolve_registry_refs(
    doc_refs: List[int],
    doc_registry: Optional[List[Dict[str, str]]],
) -> Tuple[List[str], List[str]]:
    """doc_registry 1-based refs → (target_doc_ids, target_docs)。"""
    target_doc_ids: List[str] = []
    target_docs: List[str] = []
    if not doc_refs or not doc_registry:
        return target_doc_ids, target_docs
    seen_ids: set = set()
    seen_names: set = set()
    for ref in doc_refs:
        idx = ref - 1
        if 0 <= idx < len(doc_registry):
            entry = doc_registry[idx] if isinstance(doc_registry[idx], dict) else {}
            did = str(entry.get("doc_id") or "").strip()
            name = str(entry.get("doc_name") or did or "").strip()
            if did and did not in seen_ids:
                seen_ids.add(did)
                target_doc_ids.append(did)
            if name and name not in seen_names:
                seen_names.add(name)
                target_docs.append(name)
        else:
            logger.warning(
                f"[decision_builder] reuse.refs={ref} 越界 "
                f"(registry 共 {len(doc_registry)} 篇), 已忽略"
            )
    return target_doc_ids, target_docs


def _pick_single_registry_doc(
    doc_registry: Optional[List[Dict[str, str]]],
) -> Optional[Dict[str, str]]:
    """代词回指但 router 未填 refs 时, 仅在无歧义时自动锁定 1 篇。"""
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


def build_from_reuse_args(
    args: Dict[str, Any],
    *,
    doc_registry: Optional[List[Dict[str, str]]] = None,
    query: str = "",
) -> ReuseRequest:
    """把 reuse 工具的 arguments 转成 ReuseRequest, 非法 mode 兜底为 reformat。"""
    mode = ""
    op = ""
    doc_refs: List[int] = []
    if isinstance(args, dict):
        mode = str(args.get("mode") or "").strip().lower()
        op = str(args.get("op") or "").strip()
        doc_refs = _safe_int_list(args.get("refs"))
    if mode not in REUSE_MODES:
        logger.warning(
            f"[decision_builder] reuse.mode={mode!r} 非法 (允许: {sorted(REUSE_MODES)}), "
            f"兜底为 reformat"
        )
        mode = "reformat"
    if not op:
        # op 留空时给个默认动作描述, 避免生成节点拿不到任何指示
        op = {
            "reformat": "rephrase the previous answer more clearly",
            "drilldown": "expand on the relevant point from the previous answer",
            "metasession": "describe what was retrieved or discussed previously",
            "confirm": "confirm whether the previous answer is correct",
            "continue": "continue from where the previous answer left off",
            "chitchat": "respond briefly and politely",
            "out_of_scope": "politely decline as it is outside the literature knowledge base",
        }.get(mode, "rephrase the previous answer")
    # op 长度上限 (避免 LLM 把整段回答抄进来)
    if len(op) > 200:
        op = op[:200]

    target_doc_ids: List[str] = []
    target_docs: List[str] = []
    if doc_refs:
        target_doc_ids, target_docs = _resolve_registry_refs(doc_refs, doc_registry)
    elif doc_refs and not doc_registry:
        logger.warning(
            f"[decision_builder] reuse.refs={doc_refs} 但当前会话无 doc_registry, 已忽略"
        )

    return ReuseRequest(
        mode=mode,
        op=op,
        doc_refs=doc_refs,
        target_doc_ids=target_doc_ids,
        target_docs=target_docs,
        raw=args if isinstance(args, dict) else {},
    )


# ---------------------------------------------------------------------------
# Reflect 端: retry.plan 的形态判别 (plan-shape vs multi-shape)
# ---------------------------------------------------------------------------

def build_reflect_retry(
    args: Dict[str, Any],
    *,
    query: str,
    doc_registry: Optional[List[Dict[str, str]]] = None,
    validate_fn: Optional[callable] = None,
    limits: Optional[RoutingLimits] = None,
) -> RouteOutcome:
    """把 reflect 端 retry.plan 解析成 RouteDecision 或 MultiRouteDecision。"""
    lim = limits or DEFAULT_ROUTING_LIMITS
    plan_raw = args.get("plan") if isinstance(args, dict) else None
    if not isinstance(plan_raw, dict):
        plan_raw = {}

    if "subs" in plan_raw:
        return build_from_multi_args(
            plan_raw, query=query, doc_registry=doc_registry,
            validate_fn=validate_fn, limits=lim,
        )

    paths = plan_raw.get("paths")
    if isinstance(paths, list) and paths_should_split_to_multi(paths):
        logger.info(
            "[decision_builder] retry.plan paths 含互斥 filter, 自动拆分为 multi"
        )
        multi_args = split_plan_args_to_multi_args(plan_raw)
        return build_from_multi_args(
            multi_args, query=query, doc_registry=doc_registry,
            validate_fn=validate_fn, limits=lim,
        )

    return build_from_plan_args(
        plan_raw, query=query, doc_registry=doc_registry,
        validate_fn=validate_fn, reasoning_tag="(fc-retry)", limits=lim,
    )


# ---------------------------------------------------------------------------
# 极简 RouteDecision 构造 (validate_fn=None 时的退化路径)
# ---------------------------------------------------------------------------

def _minimal_route_decision(raw: Dict[str, Any], query: str) -> RouteDecision:
    """不依赖 QueryRouter._validate_decision 的最简构造。

    仅用于单测和无 LLM 环境; 真正的 RAG 调用都会注入 validate_fn 以走完整防御链。
    """
    routes = list(raw.get("routes") or [])
    if not routes:
        routes = [_ROUTE_PROGRESSIVE]
    rewrites_in = raw.get("rewrites") or {}
    rewrites: Dict[str, str] = {}
    for r in routes:
        if r == _ROUTE_METADATA:
            continue
        val = rewrites_in.get(r)
        if isinstance(val, list):
            kws = [str(x).strip() for x in val if x and str(x).strip()]
            if kws:
                rewrites[r] = " ".join(kws)
        elif isinstance(val, str) and val.strip():
            rewrites[r] = val.strip()
        if r not in rewrites:
            rewrites[r] = query

    filters_in = raw.get("filters") or {}
    rb_top = normalize_retrieve_bias(raw.get("retrieve_bias"))
    rb_filter = normalize_retrieve_bias(filters_in.get("retrieve_bias"))
    retrieve_bias = rb_top or rb_filter
    rerank_mode = True if raw.get("rerank_mode") is True else None

    return RouteDecision(
        routes=routes,
        rewrites=rewrites,
        time=str(filters_in.get("time") or ""),
        chunk_type=filters_in.get("chunk_type"),
        target_docs=_safe_str_list(filters_in.get("target_docs")),
        target_doc_ids=_safe_str_list(filters_in.get("target_doc_ids")),
        fig_refs=_safe_str_list(filters_in.get("fig_refs"), upper=True),
        table_refs=_safe_str_list(filters_in.get("table_refs"), upper=True),
        page_refs=_safe_int_list(filters_in.get("page_refs")),
        paragraph_refs=_safe_int_list(filters_in.get("paragraph_refs")),
        entities=_safe_str_list(filters_in.get("entities")),
        retrieve_bias=retrieve_bias,
        rerank_mode=rerank_mode,
        expand_neighbors=_safe_str_list(raw.get("expand_neighbors")),
        reasoning="(fc-minimal)",
    )


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f:  # NaN
        return default
    return max(0.0, min(1.0, f))
