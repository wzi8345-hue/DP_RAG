"""反思 LLM 用的检索结果摘要: 排除结构化 chunk, 按路径展示可排序正文。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .structural_retrieval import (
    ROUTE_METADATA,
    decision_requests_structural_full_recall,
    is_structural_hit,
)
from .agentic import LocalRetrieveResult
from .retrievers import Hit

if TYPE_CHECKING:
    from .agentic import RouteDecision

DEFAULT_SNIPPET_CHARS = 400
DEFAULT_MAX_HITS_PER_ROUTE = 6
DEFAULT_MAX_TOTAL_CHARS = 5000
DEFAULT_MAX_CHARS_PER_ROUTE = 2000


@dataclass(frozen=True)
class ReflectSummaryConfig:
    snippet_chars: int = DEFAULT_SNIPPET_CHARS
    max_hits_per_route: int = DEFAULT_MAX_HITS_PER_ROUTE
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS
    max_chars_per_route: int = DEFAULT_MAX_CHARS_PER_ROUTE


def _base_route(route: str) -> str:
    """复合查询 route key 如 sub1:metadata → metadata。"""
    if ":" in route:
        return route.split(":", 1)[1]
    return route


def is_reflect_eligible(route: str, hit: Hit) -> bool:
    """反思评估只看语义正文; metadata 路由与 references/image/table 不进入摘要。"""
    if _base_route(route) == ROUTE_METADATA:
        return False
    if is_structural_hit(hit):
        return False
    return True


def should_accept_structural_only_results(
    route_results: Dict[str, Any],
    decision: Optional["RouteDecision"],
) -> bool:
    """纯结构化检索有命中时不应视为 reflect 空结果 (避免误触发 progressive retry)。"""
    if not decision_requests_structural_full_recall(decision):
        return False
    total = 0
    for v in (route_results or {}).values():
        if isinstance(v, LocalRetrieveResult):
            total += len(v.chunk_hits)
        elif isinstance(v, list):
            total += len(v)
    if total == 0:
        return False
    eligible = collect_reflect_hits_by_route(route_results)
    return sum(len(h) for h in eligible.values()) == 0


def collect_reflect_hits_by_route(
    route_results: Dict[str, Any],
) -> Dict[str, List[Hit]]:
    """按 route 收集可参与反思评估的 hit (已过滤结构化/metadata)。"""
    out: Dict[str, List[Hit]] = {}
    for route, res in (route_results or {}).items():
        hits: List[Hit] = []
        if isinstance(res, LocalRetrieveResult):
            hits = list(res.chunk_hits)
        elif isinstance(res, list):
            hits = [h for h in res if hasattr(h, "type")]
        eligible = [h for h in hits if is_reflect_eligible(route, h)]
        if eligible:
            out[route] = eligible
    return out


def _snippet(text: str, max_chars: int) -> str:
    cleaned = (text or "").replace("\n", " ").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    head = max(1, int(max_chars * 0.75))
    tail = max(1, max_chars - head - 5)
    return f"{cleaned[:head]} ... {cleaned[-tail:]}"


def summarize_for_reflect(
    route_results: Dict[str, Any],
    *,
    decision: Optional["RouteDecision"] = None,
    config: Optional[ReflectSummaryConfig] = None,
) -> Tuple[str, int]:
    """生成反思 prompt 用的检索摘要; 返回 (summary_text, reflect_eligible_hit_count)。"""
    cfg = config or ReflectSummaryConfig()
    by_route = collect_reflect_hits_by_route(route_results)
    total = sum(len(hits) for hits in by_route.values())

    if total == 0:
        return "(无可评估的语义检索结果; 结构化/metadata 命中已省略)", 0

    parts: List[str] = []
    used = 0

    for route in sorted(by_route.keys()):
        hits = by_route[route]
        route_parts: List[str] = []

        res = (route_results or {}).get(route)
        if isinstance(res, LocalRetrieveResult) and res.candidate_docs:
            doc_names = [cd.doc_name for cd in res.candidate_docs[:5] if cd.doc_name]
            if doc_names:
                route_parts.append(f"  候选文献: {'; '.join(doc_names)}")

        type_counts: Dict[str, int] = {}
        for h in hits:
            type_counts[h.type] = type_counts.get(h.type, 0) + 1
        type_str = ", ".join(f"{t}={c}" for t, c in sorted(type_counts.items()))
        route_parts.append(f"  可评估 {len(hits)} 条 ({type_str})")

        # P1 #14: 优先按 rerank_score 排序 (更可靠), 缺失时回退到 emb_score
        def _rank_key(h: Hit) -> float:
            return -(h.rerank_score if h.rerank_score is not None else (h.score or 0.0))

        sorted_hits = sorted(hits, key=_rank_key)
        for h in sorted_hits[: cfg.max_hits_per_route]:
            doc = h.doc_name or h.doc_id or "?"
            snippet = _snippet(h.content, cfg.snippet_chars)
            # 同时展示 emb_score 和 rerank_score, 给 reflect LLM 看到两个信号
            score_parts: List[str] = []
            if h.score:
                score_parts.append(f"emb={h.score:.3f}")
            if h.rerank_score is not None:
                score_parts.append(f"rerank={h.rerank_score:.3f}")
            score_note = f" {' '.join(score_parts)}" if score_parts else ""
            route_parts.append(
                f"  [{h.type}] doc={doc} page={h.page_start + 1}{score_note}: {snippet}"
            )

        route_block = f"[{route}]\n" + "\n".join(route_parts)
        if len(route_block) > cfg.max_chars_per_route:
            route_block = route_block[: cfg.max_chars_per_route] + "\n  [...路径摘要截断]"
        if used + len(route_block) > cfg.max_total_chars:
            parts.append("[...剩余路径省略]")
            break
        parts.append(route_block)
        used += len(route_block)

    note = (
        "\n(说明: references/image/table 与 metadata 路径命中不参与反思评估, "
        "已在 context 阶段单独使用。)"
    )
    text = "\n\n".join(parts) + note
    if len(text) > cfg.max_total_chars:
        text = text[: cfg.max_total_chars] + "\n[...截断]"
    return text, total
