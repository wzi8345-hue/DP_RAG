"""检索 query 格式化: Qwen3 Embedding instruct / Rerank 文档对齐 / 预热阶段映射。"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import RouteDecision

# Embedding 检索阶段 (仅 query 侧加 instruct, document 侧不加)
EMBED_STAGE_SUMMARY = "summary"
EMBED_STAGE_PASSAGE = "passage"
EMBED_STAGE_ENTITY = "entity"

DEFAULT_EMBED_INSTRUCTS: Dict[str, str] = {
    EMBED_STAGE_SUMMARY: (
        "Given a research topic query, retrieve relevant paper summaries and titles "
        "that discuss this topic"
    ),
    EMBED_STAGE_PASSAGE: (
        "Given a scientific question, retrieve relevant passages from research papers "
        "that answer the question"
    ),
    EMBED_STAGE_ENTITY: (
        "Given an entity or technical term, retrieve passages that mention this entity"
    ),
}

ROUTE_SUMMARY = "summary"
ROUTE_PROGRESSIVE = "progressive"
ROUTE_LOCAL = "local"
ROUTE_METADATA = "metadata"


def format_qwen3_embed_query(
    query: str,
    stage: Optional[str] = None,
    *,
    enabled: bool = True,
    instructs: Optional[Dict[str, str]] = None,
) -> str:
    """Qwen3-Embedding query 侧 instruct 包装 (document 侧勿用)。"""
    text = (query or "").strip()
    if not text or not enabled or not stage:
        return text
    task = (instructs or DEFAULT_EMBED_INSTRUCTS).get(
        stage, DEFAULT_EMBED_INSTRUCTS[EMBED_STAGE_PASSAGE],
    )
    return f"Instruct: {task}\nQuery:{text}"


def embed_stages_for_route(route: str) -> Tuple[str, ...]:
    """某检索路径在 dense 召回中可能用到的 embed stage (用于 LRU 预热)。"""
    if route == ROUTE_SUMMARY:
        return (EMBED_STAGE_SUMMARY,)
    if route == ROUTE_PROGRESSIVE:
        return (EMBED_STAGE_SUMMARY, EMBED_STAGE_PASSAGE)
    if route == ROUTE_LOCAL:
        return (EMBED_STAGE_PASSAGE,)
    return (EMBED_STAGE_PASSAGE,)


def collect_prewarm_embed_texts(
    decisions: Iterable["RouteDecision"],
    fallback_query: str,
    *,
    enabled: bool = True,
    instructs: Optional[Dict[str, str]] = None,
) -> List[str]:
    """收集本轮检索需预热的 formatted embed 文本 (去重)。"""
    if not enabled:
        return []
    seen: Set[str] = set()
    out: List[str] = []
    for decision in decisions:
        for route in (ROUTE_SUMMARY, ROUTE_PROGRESSIVE, ROUTE_LOCAL):
            if not decision.has(route):
                continue
            raw = (decision.get_rewrite(route, fallback_query) or fallback_query).strip()
            if not raw:
                continue
            for stage in embed_stages_for_route(route):
                fmt = format_qwen3_embed_query(
                    raw, stage, enabled=True, instructs=instructs,
                )
                if fmt not in seen:
                    seen.add(fmt)
                    out.append(fmt)
    return out


def compose_rerank_document(hit: Any, *, max_chars: int = 8000) -> str:
    """Rerank document 与入库 embedding_text 对齐: Section + content + context。"""
    section = (getattr(hit, "section", None) or "").strip()
    content = (getattr(hit, "content", None) or "").strip()
    context = (getattr(hit, "context", None) or "").strip()
    chunk_type = (getattr(hit, "type", None) or "text").strip().lower()

    parts: List[str] = []
    if section and chunk_type != "references":
        parts.append(f"[Section] {section}")

    if chunk_type == "table":
        parts.append(content)
        if context:
            parts.append(context)
    elif chunk_type == "image":
        if content:
            parts.append(content)
        if context:
            parts.append(context)
    elif chunk_type == "references":
        if content:
            parts.append(content)
    else:
        if content:
            parts.append(content)
        if context and chunk_type in ("equation", "image", "table"):
            parts.append(context)

    text = "\n\n".join(p for p in parts if p).strip()
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    return text or (content or "")


def _keyword_rewrite_fallback(decision: "RouteDecision", fallback: str) -> str:
    for route in decision.routes:
        rw = (decision.rewrites or {}).get(route, "")
        if rw:
            return rw
    parts: List[str] = []
    if decision.target_docs:
        parts.extend(decision.target_docs)
    if decision.fig_refs:
        parts.extend(f"Fig.{r}" for r in decision.fig_refs)
    if decision.table_refs:
        parts.extend(f"Table {r}" for r in decision.table_refs)
    if decision.entities:
        parts.extend(decision.entities)
    return " ".join(parts) if parts else fallback


def synthesize_rerank_query(decision: Optional["RouteDecision"], user_query: str) -> str:
    """Rerank query: 默认用户原话; router 设 rerank_mode=true 时用 kw rewrite。"""
    base = (user_query or "").strip()
    if decision is None:
        return base

    hints: List[str] = []
    if decision.fig_refs:
        hints.append("figure " + " ".join(decision.fig_refs))
    if decision.table_refs:
        hints.append("table " + " ".join(decision.table_refs))
    if decision.page_refs:
        hints.append("page " + " ".join(str(p) for p in decision.page_refs))
    if decision.paragraph_refs:
        hints.append("paragraph " + " ".join(str(p) for p in decision.paragraph_refs))
    if decision.entities:
        hints.append("entity " + " ".join(decision.entities))
    if decision.target_docs:
        hints.append("document " + " ".join(decision.target_docs[:2]))

    rewrite_kw = _keyword_rewrite_fallback(decision, "").strip()

    if not base:
        return rewrite_kw or _keyword_rewrite_fallback(decision, user_query)

    if hints and len(decision.routes) == 1 and decision.has(ROUTE_METADATA):
        return f"{base} ({'; '.join(hints)})" if base else "; ".join(hints)

    if decision.rerank_mode is True and rewrite_kw:
        return rewrite_kw

    return base


def merge_instruct_config(cfg: Optional[Dict[str, Any]]) -> Tuple[bool, Dict[str, str]]:
    """从 embedding.query_instruct 配置节解析开关与自定义 instruct。"""
    block = (cfg or {}).get("query_instruct") or {}
    if isinstance(block, bool):
        return bool(block), dict(DEFAULT_EMBED_INSTRUCTS)
    enabled = bool(block.get("enabled", True))
    custom = dict(DEFAULT_EMBED_INSTRUCTS)
    overrides = block.get("instructs") or {}
    if isinstance(overrides, dict):
        for k, v in overrides.items():
            if v and isinstance(v, str):
                custom[str(k)] = v.strip()
    return enabled, custom


def instruct_kwargs_from_embedding_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """供 build_retrievers 使用的 instruct 参数字典。"""
    enabled, instructs = merge_instruct_config(cfg)
    return {
        "embed_query_instruct_enabled": enabled,
        "embed_query_instructs": instructs,
    }
