"""观测 Progressive 双路 top-K + 平均分配信路由 (mock)。

跑法: cd .. && PYTHONPATH=. python -m pipeline.tests.observe_progressive_bounded
"""

from __future__ import annotations

from unittest.mock import MagicMock

from pipeline.retrieval.agentic import ProgressiveLocalRetriever
from pipeline.retrieval.progressive_config import ProgressiveRetrieveConfig
from pipeline.retrieval.retrievers import Hit


def _hit(
    doc_id: str,
    *,
    score: float = 0.0,
    hit_type: str = "summary",
    sources: list | None = None,
) -> Hit:
    h = Hit(
        pk=f"pk-{doc_id}-{hit_type}-{score}",
        doc_id=doc_id,
        doc_name=doc_id,
        score=score,
        type=hit_type,
        content=f"content {doc_id}",
    )
    if sources:
        h.sources = list(sources)
    return h


def _summary_filter(fe: str) -> bool:
    return 'type == "summary"' in fe or 'type == "title"' in fe


def _build(
    *,
    summary_vec: list,
    summary_bm25: list,
    chunk_vec: list,
    chunk_bm25: list,
    config: ProgressiveRetrieveConfig | None = None,
):
    vec = MagicMock()
    hybrid = MagicMock()
    bm25 = MagicMock()
    vec_log: list[dict] = []

    def vec_side(query, top_k, filter_expr=None, **kwargs):
        fe = filter_expr or ""
        vec_log.append({"top_k": top_k, "filter": fe})
        if _summary_filter(fe):
            return list(summary_vec)
        if "doc_id" in fe:
            return list(chunk_vec)
        if 'type != "summary"' in fe:
            return list(chunk_vec)
        return []

    def bm25_side(query, top_k, filter_expr=None, **kwargs):
        fe = filter_expr or ""
        if _summary_filter(fe):
            return list(summary_bm25)
        if "doc_id" in fe or 'type != "summary"' in fe:
            return list(chunk_bm25)
        return []

    vec.retrieve.side_effect = vec_side
    bm25.retrieve.side_effect = bm25_side
    hybrid.bm25 = bm25
    r = ProgressiveLocalRetriever(vec, hybrid, bm25_retriever=bm25, config=config or ProgressiveRetrieveConfig())
    return r, vec_log


def scenario_high_conf_doc_scoped():
    print("=" * 72)
    print("场景 A: 合并 doc 内探测高置信 → L2 doc-scoped")
    print("=" * 72)
    r, log = _build(
        summary_vec=[_hit("d1", score=0.7), _hit("d2", score=0.4)],
        summary_bm25=[_hit("d1", score=0.65), _hit("d3", score=0.3)],
        chunk_vec=[_hit("d1", hit_type="text", score=0.8)],
        chunk_bm25=[_hit("d1", hit_type="text", score=0.75)],
        config=ProgressiveRetrieveConfig(l2_drill_min_score=0.55),
    )
    result = r.retrieve("离子镀 炮管", top_k_docs=5)
    l2_doc = [c for c in log if "doc_id" in c["filter"]]
    l2_global = [c for c in log if 'type != "summary"' in c["filter"] and "doc_id" not in c["filter"]]
    print(f"  候选 doc: {[c.doc_name for c in result.candidate_docs]}")
    print(f"  L2 hits: {len(result.chunk_hits)}")
    print(f"  L2 doc-scoped 调用: {len(l2_doc)}")
    print(f"  L2 global 调用: {len(l2_global)} (终检应=0)")
    print()


def scenario_low_conf_global():
    print("=" * 72)
    print("场景 B: 合并 doc 内探测低置信 → L2 global")
    print("=" * 72)
    r, log = _build(
        summary_vec=[_hit("d1", score=0.4)],
        summary_bm25=[_hit("d2", score=0.35)],
        chunk_vec=[_hit("d1", hit_type="text", score=0.3)],
        chunk_bm25=[_hit("d2", hit_type="text", score=0.25)],
        config=ProgressiveRetrieveConfig(l2_drill_min_score=0.55),
    )
    result = r.retrieve("q", top_k_docs=5)
    l2_global = [c for c in log if 'type != "summary"' in c["filter"] and "doc_id" not in c["filter"]]
    l2_doc = [c for c in log if "doc_id" in c["filter"]]
    print(f"  候选 doc: {[c.doc_name for c in result.candidate_docs]}")
    print(f"  L2 global 终检调用: {len(l2_global)}")
    print(f"  L2 doc 探测调用: {sum(1 for c in l2_doc if c['top_k'] == 10)}")
    print()


def main():
    print()
    scenario_high_conf_doc_scoped()
    scenario_low_conf_global()
    print("=" * 72)
    print("观测完毕")
    print("=" * 72)


if __name__ == "__main__":
    main()
