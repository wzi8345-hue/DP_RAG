"""路由容量约束与复合意图检测 (问题 #8)。

- 单个子策略 (plan / multi.sub) 内 paths 上限: max_paths_per_sub (默认 2)
- 复合查询子查询数上限: max_subqueries (默认 3)
- 互斥 filter 的 plan.paths 自动拆分为 multi.subs
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_ROUTE_METADATA = "metadata"

# 与 agentic 中启发式一致 (避免循环 import)
_FIG_REF_LITE_RE = re.compile(
    r"(?:图|figure|fig\.?)\s*([0-9]+|[A-Za-z])", re.IGNORECASE,
)
_TAB_REF_LITE_RE = re.compile(
    r"(?:表|table|tab\.?)\s*([0-9]+|[A-Za-z])", re.IGNORECASE,
)
_REFERENCES_HINT_RE = re.compile(
    r"参考文献|引用文献|references|bibliography|引文|\brefs\b",
    re.IGNORECASE,
)
_COMPOUND_SEP_RE = re.compile(
    r"(?:以及|还有|并且|同时|分别|另外|再者|再讲讲|再查|再帮我|"
    r"并且|与|和|、|；|;)+\s*",
)
_SUMMARY_Q_RE = re.compile(
    r"总结|汇总|概述|综述|对比|主要内容|主要贡献|summarize|overview|main",
    re.IGNORECASE,
)
_PAGE_REF_RE = re.compile(
    r"(?:第\s*)?(\d+)\s*(?:页|page|p\.?)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RoutingLimits:
    """FC schema / reflect / retrieve 共用的路由容量约束。"""

    max_paths_per_sub: int = 2
    max_subqueries: int = 3

    def __post_init__(self) -> None:
        if self.max_paths_per_sub < 1:
            object.__setattr__(self, "max_paths_per_sub", 1)
        if self.max_subqueries < 2:
            object.__setattr__(self, "max_subqueries", 2)


DEFAULT_ROUTING_LIMITS = RoutingLimits()


def normalize_routes(
    routes_raw: Any,
    *,
    valid_routes: Optional[set] = None,
    route_alias: Optional[Dict[str, str]] = None,
    max_paths: Optional[int] = None,
) -> List[str]:
    """去重、别名映射、可选上限截断。"""
    if not isinstance(routes_raw, list):
        return []
    alias = route_alias or {}
    valid = valid_routes or {"summary", "progressive", "local", "metadata"}
    out: List[str] = []
    for r in routes_raw:
        r = alias.get(str(r), str(r))
        if r in valid and r not in out:
            out.append(r)
    if max_paths is not None and max_paths > 0:
        if len(out) > max_paths:
            logger.warning(
                f"[routing.limits] routes 超过上限 {max_paths}, 截断: {out} -> {out[:max_paths]}"
            )
        return out[:max_paths]
    return out


def _has_metadata_filters(path: Dict[str, Any]) -> bool:
    return bool(
        path.get("figs") or path.get("tabs") or path.get("pages")
        or path.get("paras") or path.get("ents")
    )


def _metadata_filter_signature(path: Dict[str, Any]) -> Tuple[Any, ...]:
    figs = tuple(sorted(_safe_str_list(path.get("figs"), upper=True)))
    tabs = tuple(sorted(_safe_str_list(path.get("tabs"), upper=True)))
    pages = tuple(sorted(_safe_int_list(path.get("pages"))))
    paras = tuple(sorted(_safe_int_list(path.get("paras"))))
    ents = tuple(sorted(_safe_str_list(path.get("ents"))))
    return (figs, tabs, pages, paras, ents)


def _safe_str_list(raw: Any, *, upper: bool = False) -> List[str]:
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for x in raw:
        s = str(x).strip()
        if not s:
            continue
        if upper:
            s = s.upper()
        if s not in out:
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


def paths_should_split_to_multi(paths: List[Dict[str, Any]]) -> bool:
    """plan.paths 若合并会污染 filter, 应拆成 multi.subs。"""
    if not isinstance(paths, list) or len(paths) <= 1:
        return False

    # 经典双路径: progressive/local/summary + metadata (同一意图互补)
    route_types = [p.get("t") for p in paths if isinstance(p, dict)]
    if len(paths) == 2 and route_types.count(_ROUTE_METADATA) == 1:
        meta = next(p for p in paths if p.get("t") == _ROUTE_METADATA)
        non_meta = next(p for p in paths if p.get("t") != _ROUTE_METADATA)
        if _has_metadata_filters(meta):
            non_meta_ct = str(non_meta.get("ctype") or "").lower()
            if non_meta_ct != "references":
                return False

    meta_paths = [p for p in paths if isinstance(p, dict) and p.get("t") == _ROUTE_METADATA]
    if len(meta_paths) >= 2:
        sigs = {_metadata_filter_signature(p) for p in meta_paths if _has_metadata_filters(p)}
        sigs.discard((( ), ( ), ( ), ( ), ( )))
        if len(sigs) >= 2:
            return True

    has_refs = any(
        isinstance(p, dict) and str(p.get("ctype") or "").lower() == "references"
        for p in paths
    )
    has_meta_structural = any(
        isinstance(p, dict) and p.get("t") == _ROUTE_METADATA and _has_metadata_filters(p)
        for p in paths
    )
    if has_refs and has_meta_structural:
        return True

    non_meta_ctypes = {
        str(p.get("ctype") or "").lower()
        for p in paths
        if isinstance(p, dict) and p.get("t") != _ROUTE_METADATA and p.get("ctype")
    }
    if non_meta_ctypes and has_meta_structural:
        return True

    return False


def split_plan_args_to_multi_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """把 plan arguments 拆成 multi arguments (每个 path 一个 sub)。"""
    paths = args.get("paths") if isinstance(args, dict) else None
    if not isinstance(paths, list):
        paths = []
    time_str = ""
    if isinstance(args, dict) and isinstance(args.get("time"), str):
        time_str = args["time"].strip()
    subs: List[Dict[str, Any]] = []
    for i, path in enumerate(paths):
        if not isinstance(path, dict):
            continue
        sub: Dict[str, Any] = {"paths": [path], "id": f"sub{i + 1}"}
        if time_str:
            sub["time"] = time_str
        subs.append(sub)
    return {
        "subs": subs,
        "synth": str(args.get("synth") or "") if isinstance(args, dict) else "",
    }


def estimate_compound_intents(query: str) -> int:
    """从 query 文本估计独立检索意图数 (用于 FC prompt 提示)。"""
    if not query or not query.strip():
        return 0

    categories = 0
    if _FIG_REF_LITE_RE.search(query):
        categories += 1
    if _TAB_REF_LITE_RE.search(query):
        categories += 1
    if _PAGE_REF_RE.search(query):
        categories += 1
    if _REFERENCES_HINT_RE.search(query):
        categories += 1
    if _SUMMARY_Q_RE.search(query):
        categories += 1

    clause_bonus = 0
    if _COMPOUND_SEP_RE.search(query):
        parts = _COMPOUND_SEP_RE.split(query)
        clause_bonus = max(0, len([p for p in parts if p.strip()]) - 1)

    if categories >= 2:
        return categories
    if categories >= 1 and clause_bonus >= 1:
        return categories + clause_bonus
    if clause_bonus >= 2:
        return clause_bonus
    return 0


def compound_intent_hint(
    query: str,
    *,
    limits: RoutingLimits = DEFAULT_ROUTING_LIMITS,
    enable_multi: bool = True,
) -> str:
    """若检测到多意图, 返回注入 FC user message 的提示块。"""
    if not enable_multi:
        return ""
    n = estimate_compound_intents(query)
    if n < 2:
        return ""
    cap = limits.max_subqueries
    return (
        f"\n\n[系统提示] 该问题约含 {n} 个独立检索意图。"
        f"请使用 multi 工具拆分为 {min(n, cap)} 个子查询;"
        f"每个子查询使用独立 filters (禁止把图/表/页/参考文献等不同 filter 合并进同一 plan)。"
    )


def trim_paths(paths: List[Any], *, max_paths: int) -> List[Any]:
    if max_paths <= 0 or len(paths) <= max_paths:
        return paths
    logger.warning(
        f"[routing.limits] paths 超过上限 {max_paths}, 截断 {len(paths)} -> {max_paths}"
    )
    return paths[:max_paths]


def trim_subs(subs: List[Any], *, max_subqueries: int) -> List[Any]:
    if max_subqueries <= 0 or len(subs) <= max_subqueries:
        return subs
    logger.warning(
        f"[routing.limits] subs 超过上限 {max_subqueries}, 截断 {len(subs)} -> {max_subqueries}"
    )
    return subs[:max_subqueries]
