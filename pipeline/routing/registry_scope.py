"""多轮检索文献范围判定: 何时在上轮 catalog 内下探, 何时全库检索.

三条规则 (快速检索 / FC router 后置 guard):
1. 用户明确继续问「上面/这些/它们…」等, 但未指定单篇 → 锁定上一轮**全量**文献;
2. 用户指定第 N 篇 / refs → 只锁定对应文献;
3. 用户未明确表达从上轮结果中搜 (仅话题相关) → 禁止下探, 清除 target_doc_ids.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from ..models import RouteDecision
from ..retrieval.route_filters import ROUTE_LOCAL, ROUTE_PROGRESSIVE, ROUTE_SUMMARY
from .decision_builder import (
    MultiRouteDecision,
    ReuseRequest,
    RouteOutcome,
    _resolve_registry_refs,
)

logger = logging.getLogger(__name__)

# 明确回指「上一轮检索到的文献集合」(单篇或批量)
_EXPLICIT_REGISTRY_SCOPE_RE = re.compile(
    r"(?:"
    r"上面(?:的|那|这)?(?:几篇|那些|这些|文献|论文|资料|结果|列表|内容)"
    r"|上述(?:的)?(?:几篇|文献|论文|资料|结果)?"
    r"|刚才(?:的|那|检索|找|列|说|提到)?(?:的)?(?:几篇|那些|这些|文献|论文|资料|结果)?"
    r"|之前(?:的|那)?(?:几篇|那些|这些|文献|论文|资料)?"
    r"|前面(?:的|那)?(?:几篇|那些|这些|文献|论文)?"
    r"|上一轮(?:的|检索)?(?:几篇|那些|这些|文献|论文|资料|结果)?"
    r"|这些(?:文献|论文|资料|篇|研究)"
    r"|这几篇"
    r"|那些(?:文献|论文|资料|篇)"
    r"|它们(?:的)?"
    r"|各篇"
    r"|分别(?:讲|说|用|采用|使用)"
    r"|(?:那篇|这篇|该|此)(?:的|那|这)?(?:文献|论文|文章|paper|article|document)"
    r"|(?:那篇|这篇|该|此|它|这个|那个|上面|之前|上一|刚才|前面|前一)(?:的|那|这)?"
    r"|第\s*[0-9一二三四五六七八九十百]+\s*篇"
    r"|上面那篇|下面那篇|出处"
    r")",
    re.IGNORECASE,
)

# 批量回指: 应对上轮 catalog **全量** 下探 (非单篇)
_BATCH_REGISTRY_SCOPE_RE = re.compile(
    r"(?:"
    r"上面(?:的|那|这)?(?:几篇|那些|这些|文献|论文|资料|结果|列表)"
    r"|上述(?:的)?(?:几篇|文献|论文|资料|结果)?"
    r"|刚才(?:的|那|检索|找|列)?(?:的)?(?:几篇|那些|这些|文献|论文|资料|结果)?"
    r"|之前(?:的|那)?(?:几篇|那些|这些|文献|论文|资料)?"
    r"|这些(?:文献|论文|资料|篇|研究)"
    r"|这几篇"
    r"|那些(?:文献|论文|资料|篇)"
    r"|它们(?:的)?"
    r"|各篇"
    r"|分别(?:讲|说|用|采用|使用)"
    r"|上面(?:列|检索|找)(?:出|到)?的"
    r"|刚才(?:列|检索|找)(?:出|到)?的"
    r")",
    re.IGNORECASE,
)

_DOC_INDEX_RE = re.compile(
    r"第\s*([0-9一二三四五六七八九十百]+)\s*篇",
    re.IGNORECASE,
)

_CN_NUM_MAP = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7,
               "八": 8, "九": 9, "十": 10}


def _parse_cn_num(token: str) -> Optional[int]:
    token = (token or "").strip()
    if not token:
        return None
    if token.isdigit():
        return int(token)
    if len(token) == 1 and token in _CN_NUM_MAP:
        return _CN_NUM_MAP[token]
    if token == "十":
        return 10
    if token.startswith("十") and len(token) == 2 and token[1] in _CN_NUM_MAP:
        return 10 + _CN_NUM_MAP[token[1]]
    if token.endswith("十") and len(token) == 2 and token[0] in _CN_NUM_MAP:
        return _CN_NUM_MAP[token[0]] * 10
    return None


def _pick_focus_doc(
    doc_registry: Optional[List[Dict[str, Any]]],
) -> Optional[Tuple[Dict[str, Any], str]]:
    """单篇指代且无 refs 时, 仅在无歧义时自动锁定 1 篇 (与 agentic 逻辑一致)."""
    if not doc_registry:
        return None
    pinned = [e for e in doc_registry if isinstance(e, dict) and e.get("pinned")]
    if len(pinned) == 1:
        return pinned[0], "single-pinned"
    valid = [e for e in doc_registry if isinstance(e, dict) and e.get("doc_id")]
    if len(valid) == 1:
        return valid[0], "single-entry"
    return None


def query_has_explicit_registry_scope(query: str) -> bool:
    """用户是否明确表达「从上轮检索结果里」继续搜/问."""
    return bool(_EXPLICIT_REGISTRY_SCOPE_RE.search(query or ""))


def query_has_batch_registry_scope(query: str) -> bool:
    """用户是否回指上一轮 catalog 的**批量/全量**, 而非单篇."""
    return bool(_BATCH_REGISTRY_SCOPE_RE.search(query or ""))


def extract_doc_ref_indices_from_query(query: str) -> List[int]:
    """从问句解析「第 N 篇」编号 (1-based)."""
    out: List[int] = []
    seen: set = set()
    for m in _DOC_INDEX_RE.finditer(query or ""):
        raw = m.group(1)
        num = int(raw) if raw.isdigit() else _parse_cn_num(raw)
        if num is not None and num >= 1 and num not in seen:
            seen.add(num)
            out.append(num)
    return out


def all_registry_ref_indices(
    doc_registry: Optional[List[Dict[str, str]]],
) -> List[int]:
    if not doc_registry:
        return []
    return list(range(1, len(doc_registry) + 1))


def resolve_scope_refs(
    query: str,
    doc_registry: Optional[List[Dict[str, str]]],
    *,
    existing_refs: Optional[List[int]] = None,
) -> Tuple[List[int], List[str], List[str]]:
    """根据问句 + 已有 refs 解析应锁定的 (refs, target_doc_ids, target_docs)."""
    if not doc_registry:
        return [], [], []

    query_refs = extract_doc_ref_indices_from_query(query)
    if query_refs:
        refs = query_refs
    elif existing_refs:
        refs = list(existing_refs)
    elif query_has_batch_registry_scope(query):
        refs = all_registry_ref_indices(doc_registry)
    elif query_has_explicit_registry_scope(query):
        picked = _pick_focus_doc(doc_registry)
        if picked is not None:
            entry, _ = picked
            for i, e in enumerate(doc_registry, 1):
                if isinstance(e, dict) and e.get("doc_id") == entry.get("doc_id"):
                    refs = [i]
                    break
            else:
                refs = []
        else:
            refs = []
    else:
        refs = []

    target_doc_ids, target_docs = _resolve_registry_refs(refs, doc_registry)
    return refs, target_doc_ids, target_docs


def _lock_decision_to_registry(
    decision: RouteDecision,
    *,
    target_doc_ids: List[str],
    target_docs: List[str],
    query: str,
    cid: str,
    reason: str,
) -> None:
    """把 RouteDecision 切到 local 下探并写入文献锁定."""
    if not target_doc_ids:
        return
    decision.target_doc_ids = list(target_doc_ids)
    decision.target_docs = list(target_docs)

    routes = list(decision.routes or [])
    if ROUTE_SUMMARY in routes and len(target_doc_ids) == 1:
        # summary + 单篇 refs 保留 summary
        return

    if ROUTE_PROGRESSIVE in routes and ROUTE_LOCAL not in routes:
        kw = decision.rewrites.get(ROUTE_PROGRESSIVE) or decision.rewrites.get(ROUTE_LOCAL) or query
        routes = [ROUTE_LOCAL] + [r for r in routes if r != ROUTE_PROGRESSIVE]
        decision.rewrites[ROUTE_LOCAL] = kw if isinstance(kw, str) else query
        decision.rewrites.pop(ROUTE_PROGRESSIVE, None)
    elif ROUTE_LOCAL not in routes:
        routes = [ROUTE_LOCAL] + routes

    decision.routes = routes
    logger.info(
        f"[{cid}] [registry_scope] {reason}: "
        f"锁定 {len(target_doc_ids)} 篇 → local 下探, docs={target_docs[:3]}"
        + ("..." if len(target_docs) > 3 else "")
    )


def _clear_decision_registry_lock(decision: RouteDecision, *, cid: str, query: str) -> None:
    if not (decision.target_doc_ids or decision.target_docs):
        return
    logger.info(
        f"[{cid}] [registry_scope] 无明确上轮范围指代, 清除文献锁定 "
        f"(query={query[:60]!r})"
    )
    decision.target_doc_ids = []
    decision.target_docs = []


def apply_registry_scope_to_decision(
    decision: RouteDecision,
    *,
    query: str,
    doc_registry: Optional[List[Dict[str, str]]],
    cid: str = "-",
) -> RouteDecision:
    """对单条 RouteDecision 应用文献范围 guard."""
    if not doc_registry:
        return decision

    explicit = query_has_explicit_registry_scope(query)
    if not explicit:
        _clear_decision_registry_lock(decision, cid=cid, query=query)
        return decision

    query_refs = extract_doc_ref_indices_from_query(query)
    existing_refs: List[int] = []
    if decision.target_doc_ids and doc_registry:
        id_set = set(decision.target_doc_ids)
        for i, entry in enumerate(doc_registry, 1):
            if isinstance(entry, dict) and entry.get("doc_id") in id_set:
                existing_refs.append(i)

    refs, target_doc_ids, target_docs = resolve_scope_refs(
        query, doc_registry, existing_refs=existing_refs or None,
    )

    if query_refs:
        _lock_decision_to_registry(
            decision,
            target_doc_ids=target_doc_ids,
            target_docs=target_docs,
            query=query,
            cid=cid,
            reason=f"指定第 {query_refs} 篇",
        )
    elif query_has_batch_registry_scope(query):
        _lock_decision_to_registry(
            decision,
            target_doc_ids=target_doc_ids,
            target_docs=target_docs,
            query=query,
            cid=cid,
            reason=f"批量回指上轮全量 {len(target_doc_ids)} 篇",
        )
    elif target_doc_ids:
        # 单篇指代 (这篇/它) 或 LLM 已填 refs — 保留/补全
        _lock_decision_to_registry(
            decision,
            target_doc_ids=target_doc_ids,
            target_docs=target_docs,
            query=query,
            cid=cid,
            reason="单篇回指或已有 refs",
        )
    return decision


def apply_registry_scope_guard(
    outcome: RouteOutcome,
    *,
    query: str,
    doc_registry: Optional[List[Dict[str, str]]],
    cid: str = "-",
) -> RouteOutcome:
    """对 plan/multi 路由结果应用文献范围 guard; reuse/clarify 原样返回."""
    if not doc_registry:
        return outcome

    if isinstance(outcome, RouteDecision):
        return apply_registry_scope_to_decision(
            outcome, query=query, doc_registry=doc_registry, cid=cid,
        )

    if isinstance(outcome, MultiRouteDecision):
        for sub in outcome.subqueries:
            apply_registry_scope_to_decision(
                sub.decision, query=query, doc_registry=doc_registry, cid=cid,
            )
        return outcome

    return outcome


def apply_registry_scope_to_reuse(
    reuse: ReuseRequest,
    *,
    query: str,
    doc_registry: Optional[List[Dict[str, str]]],
    cid: str = "-",
) -> ReuseRequest:
    """reuse 出口: 仅在有明确范围指代时锁定文献; 批量则全量 refs."""
    if not doc_registry:
        return reuse

    from .decision_builder import REUSE_DOC_SCOPED_MODES

    if reuse.mode not in REUSE_DOC_SCOPED_MODES:
        return reuse

    if not query_has_explicit_registry_scope(query):
        if reuse.target_doc_ids or reuse.doc_refs:
            logger.info(
                f"[{cid}] [registry_scope] reuse 无明确上轮指代, 清除 refs 锁定"
            )
            reuse.doc_refs = []
            reuse.target_doc_ids = []
            reuse.target_docs = []
        return reuse

    refs, target_doc_ids, target_docs = resolve_scope_refs(
        query, doc_registry, existing_refs=reuse.doc_refs or None,
    )
    if target_doc_ids:
        reuse.doc_refs = refs
        reuse.target_doc_ids = target_doc_ids
        reuse.target_docs = target_docs
        logger.info(
            f"[{cid}] [registry_scope] reuse mode={reuse.mode} "
            f"锁定 {len(target_doc_ids)} 篇 refs={refs}"
        )
    return reuse
