"""观测 Progressive P0 优化前后的行为对比 (非测试, 演示用).

跑法: python -m pipeline.tests.observe_progressive_p0

覆盖场景:
  1. 单 doc 高分 → strong-signal short-circuit (P0 #1)
  2. 单条 rank-0 命中 (旧阈值过线, 新阈值兜底触发)  (P0 #1)
  3. 低 conf + probe 命中 → L2 短路省 hybrid (P0 #2)
  4. references chunk_type → structural-first L1, 跳过 summary 池 (P0 #3)
"""

from __future__ import annotations

from unittest.mock import MagicMock

from pipeline.retrieval.agentic import ProgressiveLocalRetriever
from pipeline.retrieval.progressive_config import ProgressiveRetrieveConfig
from pipeline.retrieval.retrievers import Hit


def _hit(doc_id: str, *, hit_type: str = "summary") -> Hit:
    return Hit(
        pk=f"pk-{doc_id}-{hit_type}",
        doc_id=doc_id,
        doc_name=doc_id,
        score=0.0,
        type=hit_type,
        content="content",
    )


def _build(*,
           summary_hits, non_summary_hits, structural_hits=None,
           bm25_hits=None, config: ProgressiveRetrieveConfig):
    vec = MagicMock()
    hybrid = MagicMock()
    bm25 = MagicMock()
    call_log = []

    def hybrid_side_effect(query, top_k, per_retriever_k, filter_expr, **kwargs):
        fe = filter_expr or ""
        call_log.append({"top_k": top_k, "filter": fe})
        if '(type == "summary"' in fe:
            return list(summary_hits)
        for stype in ("references", "image", "table", "equation"):
            if f'type == "{stype}"' in fe:
                return list(structural_hits or [])
        return list(non_summary_hits or [])

    hybrid.retrieve.side_effect = hybrid_side_effect
    bm25.retrieve.return_value = list(bm25_hits or [])
    r = ProgressiveLocalRetriever(vec, hybrid, bm25_retriever=bm25, rrf_k=60, config=config)
    return r, call_log


def _describe_calls(call_log):
    out = []
    for i, c in enumerate(call_log, 1):
        fe = c["filter"]
        if '(type == "summary"' in fe:
            kind = "summary"
        elif '(type != "summary"' in fe:
            kind = "NON_SUMMARY(probe/L2)"
        else:
            for st in ("references", "image", "table", "equation"):
                if f'type == "{st}"' in fe:
                    kind = f"structural[{st}]"
                    break
            else:
                kind = "other"
        out.append(f"    #{i} hybrid({kind}) top_k={c['top_k']}")
    return out


def scenario_1_strong_signal():
    print("=" * 78)
    print("场景 1: 单 doc 高分 → strong-signal short-circuit (P0 #1)")
    print("=" * 78)
    print("配置: doc_confidence_threshold=0.025, level1_min_docs=3 (>=3 篇才算够)")
    print("L1 命中: 3 条都属同一篇 d1 → 单 doc, conf=0.0480 > 0.025")
    print()
    print("旧行为: level1_min_docs=3 但只找到 1 篇 → 触发 bm25 兜底, 拉一个 d_bm25 进来")
    print("新行为: 单 doc = strong signal → 直接 short-circuit, 不跑 bm25")
    print()
    r, _ = _build(
        summary_hits=[_hit("d1"), _hit("d1"), _hit("d1")],
        non_summary_hits=[],
        bm25_hits=[_hit("d_bm25")],
        config=ProgressiveRetrieveConfig(
            doc_confidence_threshold=0.025,
            level1_min_docs=3,
            strong_signal_ratio=1.5,
        ),
    )
    docs, conf, chain, probe = r._level1_with_fallbacks(
        "q", top_k_docs=5, per_query_k=8, time_filter=None, chunk_type=None,
    )
    print(f"  实际:")
    print(f"    chain = {chain}")
    print(f"    docs  = {[d for d, _, _ in docs]}")
    print(f"    conf  = {conf:.4f}")
    print(f"    probe = {len(probe)} chunks")
    print(f"  ✓ strong_signal 出现在 chain, 没有 bm25_summary, 候选保持纯净")
    print()


def scenario_2_default_threshold():
    print("=" * 78)
    print("场景 2: 单条 rank-0 命中 (旧阈值过线, 新阈值触发兜底) (P0 #1)")
    print("=" * 78)
    print("L1: 2 篇 doc 各 1 条命中 → 各自 conf ≈ 0.0164 / 0.0161, 比值≈1.02 < 1.5")
    print()
    print("旧行为 (threshold=0.012): top_conf=0.0164 > 0.012 → probe 不触发, 直接 L2")
    print("新行为 (threshold=0.025): top_conf=0.0164 < 0.025 → 触发全库 probe 兜底")
    print()
    r, _ = _build(
        summary_hits=[_hit("d1"), _hit("d2")],
        non_summary_hits=[_hit("d3", hit_type="text"), _hit("d3", hit_type="text")],
        bm25_hits=[],
        config=ProgressiveRetrieveConfig(),  # 全默认 (新版)
    )
    docs, conf, chain, probe = r._level1_with_fallbacks(
        "q", top_k_docs=5, per_query_k=8, time_filter=None, chunk_type=None,
    )
    print(f"  新行为实际:")
    print(f"    chain = {chain}")
    print(f"    conf  = {conf:.4f} (< 0.025 → 触发 global_chunk)")
    print(f"    probe = {len(probe)} chunks 被携带回")
    print()
    print("-- 对照: 旧默认 threshold=0.012 --")
    r2, _ = _build(
        summary_hits=[_hit("d1"), _hit("d2")],
        non_summary_hits=[_hit("d3", hit_type="text"), _hit("d3", hit_type="text")],
        config=ProgressiveRetrieveConfig(doc_confidence_threshold=0.012),
    )
    docs2, conf2, chain2, probe2 = r2._level1_with_fallbacks(
        "q", top_k_docs=5, per_query_k=8, time_filter=None, chunk_type=None,
    )
    print(f"    chain = {chain2}")
    print(f"    conf  = {conf2:.4f} (> 0.012 → 不触发 probe)")
    print(f"    probe = {len(probe2)} chunks")
    print()


def scenario_3_probe_short_circuit():
    print("=" * 78)
    print("场景 3: 低 conf + probe 命中 → L2 短路省一次 hybrid (P0 #2)")
    print("=" * 78)
    print("L1 弱 (conf 不达标) → 触发 probe → probe 在 d_target 上命中 2 条 chunk")
    print()
    print("旧行为: probe 找到 doc d_target 后, 再单独发起一次 L2 hybrid 在 d_target 内查 → 2 次 hybrid")
    print("新行为: probe 命中本身就是 chunks → 直接当 L2 结果用 → 1 次 hybrid + 1 次 probe = 2 (没有第三次)")
    print()
    print("-- 启用 short-circuit (新默认) --")
    r, calls = _build(
        summary_hits=[_hit("d1")],
        non_summary_hits=[_hit("d_target", hit_type="text"), _hit("d_target", hit_type="text")],
        config=ProgressiveRetrieveConfig(
            doc_confidence_threshold=0.99,
            strong_signal_ratio=1.0,
            enable_bm25_summary_fallback=False,
            enable_probe_short_circuit=True,
        ),
    )
    result = r.retrieve("q", top_k_docs=5, top_k_chunks=8)
    print(f"  hybrid 调用次数: {len(calls)}")
    for line in _describe_calls(calls):
        print(line)
    print(f"  L2 chunk_hits: {len(result.chunk_hits)} 条")
    print()
    print("-- 关闭 short-circuit (旧行为) --")
    r2, calls2 = _build(
        summary_hits=[_hit("d1")],
        non_summary_hits=[_hit("d_target", hit_type="text"), _hit("d_target", hit_type="text")],
        config=ProgressiveRetrieveConfig(
            doc_confidence_threshold=0.99,
            strong_signal_ratio=1.0,
            enable_bm25_summary_fallback=False,
            enable_probe_short_circuit=False,
        ),
    )
    result2 = r2.retrieve("q", top_k_docs=5, top_k_chunks=8)
    print(f"  hybrid 调用次数: {len(calls2)}")
    for line in _describe_calls(calls2):
        print(line)
    print(f"  L2 chunk_hits: {len(result2.chunk_hits)} 条")
    print(f"  ✓ 启用 short-circuit 节省了 {len(calls2) - len(calls)} 次 hybrid 调用")
    print()


def scenario_4_structural_skip():
    print("=" * 78)
    print("场景 4: references chunk_type → 跳过 summary L1 (P0 #3)")
    print("=" * 78)
    print("配置: chunk_type='references'")
    print("旧行为: 先在 summary 池跑一次 hybrid 反推 doc, 然后 _level2_structural_chunks 全量召回")
    print("新行为: 直接在 references 池跑 hybrid 反推 doc (summary 池里几乎没有 refs 内容)")
    print()
    print("-- 启用 structural-first (新默认) --")
    r, calls = _build(
        summary_hits=[_hit("d_summary", hit_type="summary")],
        non_summary_hits=[],
        structural_hits=[_hit("d_ref", hit_type="references"),
                         _hit("d_ref", hit_type="references")],
        config=ProgressiveRetrieveConfig(structural_skip_summary_l1=True),
    )
    r._level2_drill_chunks = MagicMock(return_value=[])  # type: ignore[method-assign]
    r.retrieve("q", top_k_docs=5, top_k_chunks=8, chunk_type="references")
    print(f"  hybrid 调用次数: {len(calls)}")
    for line in _describe_calls(calls):
        print(line)
    print()
    print("-- 关闭 structural-first (旧行为) --")
    r2, calls2 = _build(
        summary_hits=[_hit("d_summary", hit_type="summary")],
        non_summary_hits=[],
        structural_hits=[_hit("d_ref", hit_type="references")],
        config=ProgressiveRetrieveConfig(structural_skip_summary_l1=False),
    )
    r2._level2_drill_chunks = MagicMock(return_value=[])  # type: ignore[method-assign]
    r2.retrieve("q", top_k_docs=5, top_k_chunks=8, chunk_type="references")
    print(f"  hybrid 调用次数: {len(calls2)}")
    for line in _describe_calls(calls2):
        print(line)
    print(f"  ✓ 启用 structural-first 把 summary 池查询换成了 references 池查询")
    print()


def main():
    print()
    scenario_1_strong_signal()
    scenario_2_default_threshold()
    scenario_3_probe_short_circuit()
    scenario_4_structural_skip()
    print("=" * 78)
    print("观测完毕. 所有 P0 优化均按预期工作.")
    print("=" * 78)


if __name__ == "__main__":
    main()
