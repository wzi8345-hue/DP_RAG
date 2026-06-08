"""按检索路径解析 chunk_type 过滤 (#10)。

decision.chunk_type 是决策级字段, 各路径语义不同:
- summary     : 永远忽略 (只查 summary/title)
- progressive : L2 应用; L1 文献定位忽略 (见 level1_global_probe_chunk_type)
- local       : 仅 L2 chunk 检索应用
- metadata    : 由 fig/tab refs 或 image/table/equation 推导, 不用 references
"""

from __future__ import annotations

from typing import Optional

from ..models import RouteDecision

ROUTE_SUMMARY = "summary"
ROUTE_PROGRESSIVE = "progressive"
ROUTE_LOCAL = "local"
ROUTE_METADATA = "metadata"


def _metadata_handles_structured(decision: RouteDecision) -> bool:
    if ROUTE_METADATA not in (decision.routes or []):
        return False
    return bool(
        decision.fig_refs
        or decision.table_refs
        or decision.page_refs
        or decision.paragraph_refs
        or decision.entities
    )


def chunk_type_for_route(route: str, decision: RouteDecision) -> Optional[str]:
    """返回某条路径应使用的 Milvus type 过滤; None 表示走默认 NON_SUMMARY 池。"""
    ct = (decision.chunk_type or "").strip().lower() or None
    routes = decision.routes or []

    if route == ROUTE_SUMMARY:
        return None

    if route == ROUTE_METADATA:
        if decision.fig_refs and not decision.table_refs:
            return "image"
        if decision.table_refs and not decision.fig_refs:
            return "table"
        if ct in ("image", "table", "equation"):
            return ct
        return None

    if route in (ROUTE_PROGRESSIVE, ROUTE_LOCAL):
        if ct == "references":
            return "references"
        if ct == "equation":
            return "equation"
        if ct in ("image", "table"):
            if _metadata_handles_structured(decision):
                return None
            return ct
        return None

    return None


def level1_global_probe_chunk_type(l2_chunk_type: Optional[str]) -> Optional[str]:
    """Progressive L1 全库 doc 探测: 不用 references/image/table 收窄, 避免漏斗漏 doc。"""
    if not l2_chunk_type:
        return None
    if l2_chunk_type in ("references", "image", "table", "equation"):
        return None
    return l2_chunk_type


def describe_route_chunk_types(decision: RouteDecision) -> dict:
    """供日志/调试: 各路径实际使用的 chunk_type。"""
    out = {}
    for route in decision.routes or []:
        out[route] = chunk_type_for_route(route, decision)
    return out
