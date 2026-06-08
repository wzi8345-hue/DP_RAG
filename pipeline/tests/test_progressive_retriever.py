"""ProgressiveLocalRetriever L1 fallback 逻辑 (#9 Phase B)。"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from pipeline.retrieval.agentic import ProgressiveLocalRetriever
from pipeline.retrieval.progressive_config import (
    ProgressiveRetrieveConfig,
    chunk_type_skips_summary_l1,
)
from pipeline.retrieval.retrievers import Hit


def _hit(doc_id: str, *, doc_name: str = "", score: float = 0.0,
         hit_type: str = "summary", sources: list | None = None) -> Hit:
    h = Hit(
        pk=f"pk-{doc_id}-{hit_type}",
        doc_id=doc_id,
        doc_name=doc_name or doc_id,
        score=score,
        type=hit_type,
        content="content",
    )
    if sources:
        h.sources = list(sources)
    return h


def _summary_filter(filter_expr: str | None) -> bool:
    """仅匹配 L1 summary 池 filter, 避免 `type != "summary"` 误命中。"""
    fe = filter_expr or ""
    return '(type == "summary" or type == "title")' in fe


class TestProgressiveLevel1Fallback(unittest.TestCase):
    def _make_retriever(
        self,
        *,
        l1_hits: list | None = None,
        bm25_hits: list | None = None,
        global_hits: list | None = None,
        config: ProgressiveRetrieveConfig | None = None,
    ) -> ProgressiveLocalRetriever:
        vec = MagicMock()
        hybrid = MagicMock()
        bm25 = MagicMock()

        cfg = config or ProgressiveRetrieveConfig(
            doc_confidence_threshold=0.05,
            level1_min_docs=2,
        )

        def hybrid_side_effect(query, top_k, per_retriever_k, filter_expr, **kwargs):
            return list(l1_hits or [])

        def vec_side_effect(query, top_k, filter_expr=None, **kwargs):
            if _summary_filter(filter_expr):
                return list(l1_hits or [])
            return list(global_hits or [])

        def bm25_side_effect(query, top_k, filter_expr=None, **kwargs):
            if _summary_filter(filter_expr):
                return list(bm25_hits or [])
            return list(global_hits or [])

        hybrid.retrieve.side_effect = hybrid_side_effect
        bm25.retrieve.side_effect = bm25_side_effect
        vec.retrieve.side_effect = vec_side_effect
        hybrid.bm25 = bm25

        return ProgressiveLocalRetriever(
            vec, hybrid, bm25_retriever=bm25, rrf_k=60, config=cfg,
        )

    def test_l1_dual_top5_merges_paths(self):
        r = self._make_retriever(
            l1_hits=[_hit("d1", score=0.9), _hit("d2", score=0.4)],
            bm25_hits=[_hit("d1", score=0.7), _hit("d3", score=0.3)],
        )
        docs, conf, chain, _ = r._level1_with_fallbacks(
            "q", top_k_docs=5, per_query_k=8, time_filter=None, chunk_type=None,
        )
        ids = [d for d, _, _ in docs]
        self.assertIn("l1_dual_top5", chain)
        self.assertIn("d1", ids)
        self.assertIn("d2", ids)
        self.assertIn("d3", ids)
        self.assertGreaterEqual(conf, 0.9)

    def test_dual_path_confidence_avg(self):
        high, avg, tv, tb = ProgressiveLocalRetriever._dual_path_confidence_from_hits(
            [_hit("d1", score=0.6)],
            [_hit("d1", score=0.5)],
            0.55,
        )
        self.assertTrue(high)
        self.assertAlmostEqual(avg, 0.55)
        self.assertFalse(
            ProgressiveLocalRetriever._dual_path_confidence_from_hits(
                [_hit("d1", score=0.3)], [_hit("d2", score=0.2)], 0.55,
            )[0]
        )


class TestProgressiveP0Optimizations(unittest.TestCase):
    """P0 优化覆盖: strong-signal short-circuit, probe→L2 短路, structural-first L1。"""

    def _make_retriever_filter_aware(
        self,
        *,
        summary_hits: list,           # type=summary 命中
        non_summary_hits: list,        # NON_SUMMARY 过滤器命中 (probe text/正文)
        structural_hits: list | None = None,  # 当 force chunk_type 时返回的命中
        bm25_hits: list | None = None,
        config: ProgressiveRetrieveConfig | None = None,
    ) -> tuple[ProgressiveLocalRetriever, list[dict], list[dict]]:
        """工厂; 返回 hybrid 与 vector 调用日志, 供断言用。"""
        vec = MagicMock()
        hybrid = MagicMock()
        bm25 = MagicMock()
        call_log: list[dict] = []
        vec_call_log: list[dict] = []

        cfg = config or ProgressiveRetrieveConfig()

        def _by_content_type_clause(fe: str, hits: list) -> list:
            """v7 内容池分离: filter 含 text/equation/image/table 明确类型时,
            只返回对应类型的 hit (忠实反映 Milvus type 过滤); 否则 (NON_SUMMARY 单池) 全返回。"""
            wanted = {
                t for t in ("text", "equation", "image", "table")
                if f'type == "{t}"' in fe
            }
            if not wanted:
                return list(hits)
            return [h for h in hits if h.type in wanted]

        def hybrid_side_effect(query, top_k, per_retriever_k, filter_expr, **kwargs):
            fe = filter_expr or ""
            call_log.append({
                "top_k": top_k,
                "filter": fe,
                "chunk_type": kwargs.get("chunk_type"),
            })
            # summary 池
            if '(type == "summary"' in fe:
                return list(summary_hits)
            # structural 池 (force_chunk_type 时直接走 type == "<x>")
            for stype in ("references", "image", "table", "equation"):
                if f'type == "{stype}"' in fe:
                    return list(structural_hits or [])
            # L2 NON_SUMMARY drill
            return _by_content_type_clause(fe, non_summary_hits or [])

        def vec_side_effect(query, top_k, filter_expr=None, **kwargs):
            fe = filter_expr or ""
            vec_call_log.append({"top_k": top_k, "filter": fe})
            if _summary_filter(fe):
                return list(summary_hits)
            if 'type != "summary"' in fe or "doc_id" in fe:
                return _by_content_type_clause(fe, non_summary_hits or [])
            for stype in ("references", "image", "table", "equation"):
                if f'type == "{stype}"' in fe:
                    return list(structural_hits or [])
            return []

        def bm25_side_effect(query, top_k, filter_expr=None, **kwargs):
            fe = filter_expr or ""
            if _summary_filter(fe):
                return list(bm25_hits if bm25_hits is not None else [])
            if 'type != "summary"' in fe or "doc_id" in fe:
                extra = [h for h in (bm25_hits or []) if h.type != "summary"]
                return _by_content_type_clause(fe, list(non_summary_hits or []) + extra)
            return []

        hybrid.retrieve.side_effect = hybrid_side_effect
        vec.retrieve.side_effect = vec_side_effect
        bm25.retrieve.side_effect = bm25_side_effect
        hybrid.bm25 = bm25

        retr = ProgressiveLocalRetriever(
            vec, hybrid, bm25_retriever=bm25, rrf_k=60, config=cfg,
        )
        return retr, call_log, vec_call_log

    # ── L1 multi-path + L2 global ───────────────────────────────────

    def test_l1_dual_path_separate_ranking(self):
        r, _, _ = self._make_retriever_filter_aware(
            summary_hits=[_hit("d1", score=0.4), _hit("d2", score=0.9)],
            bm25_hits=[_hit("d3", score=0.95)],
            non_summary_hits=[],
        )
        docs, conf, chain, _ = r._level1_with_fallbacks(
            "q", top_k_docs=5, per_query_k=8, time_filter=None, chunk_type=None,
        )
        self.assertIn("l1_dual_top5", chain)
        self.assertEqual(docs[0][0], "d3")
        self.assertAlmostEqual(conf, 0.95)

    def test_l2_high_conf_routes_doc_scoped(self):
        r, _, vec_log = self._make_retriever_filter_aware(
            summary_hits=[_hit("d1", score=0.7)],
            bm25_hits=[_hit("d1", score=0.65)],
            non_summary_hits=[
                _hit("d1", hit_type="text", score=0.8),
                _hit("d1", hit_type="text", score=0.75),
            ],
            config=ProgressiveRetrieveConfig(l2_drill_min_score=0.55),
        )
        r.retrieve("q", top_k_docs=5, top_k_chunks=8)
        self.assertTrue(any("doc_id" in c["filter"] for c in vec_log))

    def test_l2_low_conf_routes_global(self):
        r, _, vec_log = self._make_retriever_filter_aware(
            summary_hits=[_hit("d1", score=0.4)],
            bm25_hits=[_hit("d2", score=0.35)],
            non_summary_hits=[
                _hit("d1", hit_type="text", score=0.2),
                _hit("d2", hit_type="text", score=0.15),
            ],
            config=ProgressiveRetrieveConfig(l2_drill_min_score=0.55),
        )
        r.retrieve("q", top_k_docs=5, top_k_chunks=8)
        # v7 分池后, 全库 L2 走内容池过滤 (text/equation 或 image/table), 仍无 doc_id;
        # 兼容单池 (type != "summary") 与分池两种 filter 形态。
        global_calls = [
            c for c in vec_log
            if "doc_id" not in c["filter"] and (
                'type != "summary"' in c["filter"]
                or 'type == "text"' in c["filter"]
                or 'type == "image"' in c["filter"]
            )
        ]
        self.assertGreaterEqual(len(global_calls), 1)

    def test_l2_high_conf_multi_path_merges_chunks(self):
        """高置信 → doc-scoped L2; vector+bm25 各 top-K 合并 (无 RRF)。"""
        vec_hits = [
            _hit("d_target", hit_type="text", score=0.9),
            _hit("d_other", hit_type="text", score=0.8),
        ]
        bm25_only = [_hit("d_bm25", hit_type="text", score=0.7)]
        r, _, vec_call_log = self._make_retriever_filter_aware(
            summary_hits=[_hit("d1", score=0.8), _hit("d2", score=0.75)],
            non_summary_hits=vec_hits,
            bm25_hits=bm25_only,
            config=ProgressiveRetrieveConfig(l2_drill_min_score=0.55),
        )
        result = r.retrieve(
            query="q", top_k_docs=5, top_k_chunks=8,
            per_query_k=8, per_retriever_k=10,
        )
        self.assertEqual(len(result.chunk_hits), 3)
        from pipeline.retrieval.agentic import ROUTE_PROGRESSIVE
        self.assertTrue(all(ROUTE_PROGRESSIVE in h.sources for h in result.chunk_hits))
        l2_doc = [c for c in vec_call_log if "doc_id" in c["filter"]]
        self.assertGreater(len(l2_doc), 0)

    def test_l2_multi_path_merges_without_rrf(self):
        r, _, _ = self._make_retriever_filter_aware(
            summary_hits=[_hit("d1"), _hit("d2")],
            non_summary_hits=[
                _hit("shared", hit_type="text", score=0.5),
                _hit("vec_only", hit_type="text", score=0.4),
            ],
            bm25_hits=[
                _hit("shared", hit_type="text", score=0.6),
                _hit("bm25_only", hit_type="text", score=0.3),
            ],
            config=ProgressiveRetrieveConfig(
                enable_global_chunk_fallback=False,
            ),
        )
        hits = r._level2_drill_chunks(
            "q",
            candidate_doc_ids=["d1"],
            top_k_chunks=2,
            per_query_k=5,
            per_retriever_k=10,
            time_filter=None,
        )
        self.assertEqual(len(hits), 3)
        shared = next(h for h in hits if h.doc_id == "shared")
        self.assertEqual(shared.score, 0.6)
        self.assertIn("vector", shared.sources)
        self.assertIn("bm25", shared.sources)

    def test_global_fallback_disabled_runs_doc_scoped_l2(self):
        r, call_log, vec_call_log = self._make_retriever_filter_aware(
            summary_hits=[_hit("d1", score=0.3)],
            non_summary_hits=[_hit("d_target", hit_type="text", score=0.8)],
            config=ProgressiveRetrieveConfig(
                enable_global_chunk_fallback=False,
            ),
        )
        r.retrieve(query="q", top_k_docs=5, top_k_chunks=8)
        # 无全库 enrich/L2; doc-scoped L2 走 doc_id 过滤
        self.assertFalse(
            any('type != "summary"' in c["filter"] and "doc_id" not in c["filter"]
                for c in vec_call_log)
        )
        doc_scoped = [c for c in vec_call_log if "doc_id" in c["filter"]]
        self.assertGreaterEqual(len(doc_scoped), 1)

    # ── P0 #3: structural 跳过 summary L1 ─────────────────────────────

    def test_p0_3_structural_skips_summary_l1(self):
        # references chunk_type → 不应该查 summary 池
        r, call_log, vec_call_log = self._make_retriever_filter_aware(
            summary_hits=[_hit("d_summary", hit_type="summary")],
            non_summary_hits=[],
            structural_hits=[
                _hit("d_ref", hit_type="references"),
                _hit("d_ref", hit_type="references"),
            ],
            config=ProgressiveRetrieveConfig(
                structural_skip_summary_l1=True,
            ),
        )
        # 短路 mock 一下 _level2_drill_chunks/_level2_structural_chunks 让流程跑通
        r._level2_drill_chunks = MagicMock(return_value=[])  # type: ignore[method-assign]
        result = r.retrieve(
            query="q", top_k_docs=5, top_k_chunks=8, chunk_type="references",
        )
        # 没有任何 summary filter 出现在 hybrid 调用里
        for c in call_log:
            self.assertNotIn('(type == "summary"', c["filter"])
        # structural-first L1 走 vector references 池
        self.assertTrue(
            any('type == "references"' in c["filter"] for c in vec_call_log)
        )
        self.assertIsNotNone(result)

    def test_p0_3_disabled_falls_back_to_summary_l1(self):
        r, call_log, vec_call_log = self._make_retriever_filter_aware(
            summary_hits=[_hit("d_summary", hit_type="summary")],
            non_summary_hits=[],
            structural_hits=[_hit("d_ref", hit_type="references")],
            config=ProgressiveRetrieveConfig(
                structural_skip_summary_l1=False,
            ),
        )
        r._level2_drill_chunks = MagicMock(return_value=[])  # type: ignore[method-assign]
        r.retrieve(query="q", top_k_docs=5, top_k_chunks=8, chunk_type="references")
        self.assertTrue(
            any(_summary_filter(c["filter"]) for c in vec_call_log)
        )

    # ── 辅助函数 ──────────────────────────────────────────────────────

    def test_chunk_type_skips_summary_l1_helper(self):
        self.assertTrue(chunk_type_skips_summary_l1("references"))
        self.assertTrue(chunk_type_skips_summary_l1("image"))
        self.assertTrue(chunk_type_skips_summary_l1("table"))
        self.assertTrue(chunk_type_skips_summary_l1("equation"))
        self.assertTrue(chunk_type_skips_summary_l1("REFERENCES"))  # 大小写
        self.assertFalse(chunk_type_skips_summary_l1("text"))
        self.assertFalse(chunk_type_skips_summary_l1(None))
        self.assertFalse(chunk_type_skips_summary_l1(""))

    def test_is_strong_signal_single_doc(self):
        # 单 doc → strong
        self.assertTrue(
            ProgressiveRetrieveConfig.is_strong_signal([("d1", 0.05, "d1")], 1.5)
        )

    def test_is_strong_signal_dominant_top1(self):
        # top1=0.05, top2=0.02, 比值=2.5 ≥ 1.5 → strong
        self.assertTrue(
            ProgressiveRetrieveConfig.is_strong_signal(
                [("d1", 0.05, "d1"), ("d2", 0.02, "d2")], 1.5,
            )
        )

    def test_is_strong_signal_close_competitors(self):
        # top1=0.05, top2=0.04, 比值=1.25 < 1.5 → 非 strong
        self.assertFalse(
            ProgressiveRetrieveConfig.is_strong_signal(
                [("d1", 0.05, "d1"), ("d2", 0.04, "d2")], 1.5,
            )
        )

    def test_is_strong_signal_empty(self):
        self.assertFalse(ProgressiveRetrieveConfig.is_strong_signal([], 1.5))


class TestLevel2SplitContentPool(unittest.TestCase):
    """v7: 正文/图表分池召回 — 正文全量 + 图表截断到 structural_content_top_k。"""

    def _retriever(self, *, prose_hits, struct_hits, cfg=None):
        vec = MagicMock()
        hybrid = MagicMock()
        bm25 = MagicMock()
        hybrid.bm25 = bm25

        def _pool_for(fe):
            if 'type == "text"' in fe or 'type == "equation"' in fe:
                return list(prose_hits)
            if 'type == "image"' in fe or 'type == "table"' in fe:
                return list(struct_hits)
            return list(prose_hits) + list(struct_hits)

        vec.retrieve.side_effect = lambda query, top_k, filter_expr=None, **kw: _pool_for(filter_expr or "")
        bm25.retrieve.side_effect = lambda query, top_k, filter_expr=None, **kw: []
        return ProgressiveLocalRetriever(
            vec, hybrid, bm25_retriever=bm25,
            config=cfg or ProgressiveRetrieveConfig(),
        )

    def test_structural_capped_prose_kept(self):
        prose = [_hit("d1", hit_type="text", score=0.9) for _ in range(5)]
        # 给每个 prose 不同 pk 否则会被按 pk 去重
        for i, h in enumerate(prose):
            h.pk = f"prose-{i}"
        struct = []
        for i in range(6):
            h = _hit("d1", hit_type="table", score=0.95)
            h.pk = f"tab-{i}"
            struct.append(h)
        r = self._retriever(
            prose_hits=prose, struct_hits=struct,
            cfg=ProgressiveRetrieveConfig(structural_content_top_k=2),
        )
        hits = r._level2_drill_chunks(
            "q", candidate_doc_ids=["d1"], top_k_chunks=8,
            per_query_k=8, per_retriever_k=10, time_filter=None,
        )
        n_struct = sum(1 for h in hits if h.type in ("image", "table"))
        n_prose = sum(1 for h in hits if h.type in ("text", "equation"))
        self.assertEqual(n_prose, 5)       # 正文全量保留
        self.assertEqual(n_struct, 2)      # 图表截断到 top-2

    def test_structural_top_k_zero_drops_figures(self):
        prose = [_hit("d1", hit_type="text", score=0.9)]
        prose[0].pk = "prose-0"
        struct = [_hit("d1", hit_type="image", score=0.95)]
        struct[0].pk = "img-0"
        r = self._retriever(
            prose_hits=prose, struct_hits=struct,
            cfg=ProgressiveRetrieveConfig(structural_content_top_k=0),
        )
        hits = r._level2_drill_chunks(
            "q", candidate_doc_ids=["d1"], top_k_chunks=8,
            per_query_k=8, per_retriever_k=10, time_filter=None,
        )
        self.assertTrue(all(h.type not in ("image", "table") for h in hits))
        self.assertEqual(len(hits), 1)

    def test_split_disabled_uses_single_pool(self):
        prose = [_hit("d1", hit_type="text", score=0.9)]
        prose[0].pk = "prose-0"
        struct = [_hit("d1", hit_type="table", score=0.95)]
        struct[0].pk = "tab-0"
        r = self._retriever(
            prose_hits=prose, struct_hits=struct,
            cfg=ProgressiveRetrieveConfig(split_content_pool=False),
        )
        hits = r._level2_drill_chunks(
            "q", candidate_doc_ids=["d1"], top_k_chunks=8,
            per_query_k=8, per_retriever_k=10, time_filter=None,
        )
        # 单池: NON_SUMMARY 一次召回, prose+struct 混在一起
        self.assertEqual(len(hits), 2)


class TestLevel2MultiPathChunks(unittest.TestCase):
    def test_bm25_retrieve_does_not_receive_embed_stage(self):
        vec = MagicMock()
        hybrid = MagicMock()
        bm25 = MagicMock()
        hybrid.bm25 = bm25

        vec.retrieve.return_value = [_hit("d1", score=0.9, hit_type="text")]
        bm25.retrieve.return_value = [_hit("d1", score=0.7, hit_type="text")]

        r = ProgressiveLocalRetriever(vec, hybrid, bm25_retriever=bm25)
        hits = r._level2_multi_path_chunks(
            "2012 船板 产量 企业 排名 top 5",
            chunk_filter='doc_id == "doc1"',
            per_path_k=10,
        )

        self.assertEqual(len(hits), 1)
        vec.retrieve.assert_called_once()
        vec_kwargs = vec.retrieve.call_args.kwargs
        self.assertEqual(vec_kwargs.get("embed_stage"), "passage")

        bm25.retrieve.assert_called_once()
        self.assertNotIn("embed_stage", bm25.retrieve.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
