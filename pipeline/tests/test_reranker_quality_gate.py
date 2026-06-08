"""Reranker 质量门控优化覆盖测试 (P0/P1 改动)。

覆盖:
  P0 #1  quality_score 用 0.7*max + 0.3*mean 加权混合
  P0 #3  零召回 → needs_retry=True + progressive 兜底
  P0 #10 reranker 未评分 (None) 不污染 stats / rankable
  P1 #5  chunk_type-aware 阈值表
  P1 #9  wrong_type confidence 分层 (strong=0.92 / weak=0.80)
  P1 #14 rerank_score 写入 Hit
  P1 #17 reranker API 失败 fail-open
"""

from __future__ import annotations

import unittest
from typing import List, Optional
from unittest.mock import MagicMock

from pipeline.clients.reranker import RerankResult, RerankerClient
from pipeline.models import RouteDecision
from pipeline.retrieval.agentic import (
    ROUTE_LOCAL,
    ROUTE_METADATA,
    ROUTE_PROGRESSIVE,
    ROUTE_SUMMARY,
)
from pipeline.retrieval.langgraph_agent import _make_reranker_node, _resolve_rerank_query_for_hit
from pipeline.retrieval.rerank_diagnosis import (
    RerankDiagnosisConfig,
    diagnose_rerank_failure,
)
from pipeline.retrieval.retrievers import Hit


def _hit(chunk_type: str = "text", content: str = "chunk", *, pk: str = "p") -> Hit:
    return Hit(pk=pk, chunk_id=pk, type=chunk_type, content=content)


def _mock_reranker(scores_by_idx: dict) -> MagicMock:
    """构造一个 mock reranker, scores_by_idx 仅在指定 idx 返回分数。

    未在字典中的 idx 不会出现在返回结果里 → 上游应视为 "未评分 (None)"。
    """
    rr = MagicMock()

    def side(query, documents, top_k):
        out: List[RerankResult] = []
        for i in range(len(documents)):
            if i in scores_by_idx:
                out.append(RerankResult(index=i, score=scores_by_idx[i], content=documents[i]))
        return out

    rr.rerank.side_effect = side
    return rr


# ---------------------------------------------------------------------------
# P0 #3: 零召回 → needs_retry=True
# ---------------------------------------------------------------------------

class TestZeroRecallTriggersRetry(unittest.TestCase):
    def test_empty_hits_forces_retry_with_progressive_fallback(self):
        rr = MagicMock()
        node = _make_reranker_node(rr, top_k=5, quality_k=3, quality_threshold=0.3)
        state = {
            "correlation_id": "t1",
            "query": "MoS2 晶格常数是多少",
            "route_results": {},  # 零召回
        }
        out = node(state)
        # 关键断言: 必须重试, 而不是悄悄放弃
        self.assertTrue(out["needs_retry"])
        self.assertEqual(out["rerank_diagnosis_cause"], "zero_recall")
        # 兜底建议必须是 RouteDecision, 路径为 progressive
        hint = out["rewrite_hint"]
        self.assertIsInstance(hint, RouteDecision)
        self.assertIn(ROUTE_PROGRESSIVE, hint.routes)
        # 零召回置信度低 → 不跳过 reflect, 让 LLM 兜底
        self.assertFalse(out["rerank_skip_reflect"])
        # reranker_client.rerank 不应被调用 (没东西可以 rerank)
        rr.rerank.assert_not_called()


# ---------------------------------------------------------------------------
# P0 #1 + P1 #14: quality_score 公式 + rerank_score 注入
# ---------------------------------------------------------------------------

class TestQualityScoreFormula(unittest.TestCase):
    def test_blended_max_mean_not_diluted_by_weak_route(self):
        """一条路径 0.9, 另一条 0.1 → 0.7*0.9 + 0.3*0.5 = 0.78 (vs 旧的 0.50)。"""
        # 4 hits, 2 routes; reranker 给出固定分数
        rr = _mock_reranker({0: 0.92, 1: 0.88, 2: 0.10, 3: 0.10})
        node = _make_reranker_node(rr, top_k=2, quality_k=2, quality_threshold=0.3)
        text_a = [_hit("text", f"a{i}", pk=f"pa{i}") for i in range(2)]
        text_b = [_hit("text", f"b{i}", pk=f"pb{i}") for i in range(2)]
        state = {
            "correlation_id": "t2",
            "query": "X",
            "route_results": {ROUTE_PROGRESSIVE: text_a, ROUTE_SUMMARY: text_b},
        }
        out = node(state)
        # max = 0.90 (progressive top-2 avg), mean = (0.90 + 0.10) / 2 = 0.50
        # blended = 0.7 * 0.90 + 0.3 * 0.50 = 0.78
        self.assertAlmostEqual(out["reranker_score"], 0.78, places=2)
        # 因 progressive 路径 avg=0.90 ≥ 0.3 → gate pass, needs_retry=False
        self.assertFalse(out["needs_retry"])

    def test_merge_bm25_vector_top2_into_rerank_results(self):
        """BM25/vector retrieve top2 并入 rerank top-k, 避免高分低 rerank 被截断。"""
        rr = _mock_reranker({0: 0.08, 1: 0.10, 2: 0.75, 3: 0.24, 4: 0.70})
        node = _make_reranker_node(rr, top_k=2, quality_k=2, quality_threshold=0.3)
        answer = Hit(
            pk="p0", chunk_id="p0", type="text", content="蒙乃尔管板堆焊变形",
            score=60.0, sources=["vector", "bm25"],
        )
        cite = Hit(
            pk="p1", chunk_id="p1", type="text", content="蒙乃尔换热器焊接题录",
            score=40.0, sources=["vector", "bm25"],
        )
        bm25_a = Hit(
            pk="p2", chunk_id="p2", type="text", content="铜镍管板气孔",
            score=15.0, sources=["bm25"],
        )
        bm25_b = Hit(
            pk="p3", chunk_id="p3", type="text", content="镀锌机组改造",
            score=12.0, sources=["bm25"],
        )
        vec_only = Hit(
            pk="p4", chunk_id="p4", type="text", content="不锈钢热处理",
            score=0.5, sources=["vector"],
        )
        hits = [answer, cite, bm25_a, bm25_b, vec_only]
        state = {
            "correlation_id": "t_bm25_vec",
            "query": "蒙乃尔合金换热器管板堆焊变形",
            "route_results": {ROUTE_PROGRESSIVE: hits},
        }
        out = node(state)
        filtered = out["route_results"][ROUTE_PROGRESSIVE]
        filtered_pks = [h.pk for h in filtered]
        # rerank top2 = p2,p4; 保送 bm25 top2 p0,p1 + vector top2 p0,p1
        self.assertGreaterEqual(len(filtered), 4)
        self.assertIn("p0", filtered_pks)
        self.assertIn("p1", filtered_pks)
        self.assertEqual(filtered_pks[0], "p2")
        self.assertEqual(filtered_pks[1], "p4")

    def test_rerank_score_written_back_to_hits(self):
        """P1 #14: 每个 hit 都应拿到 rerank_score (供 reflect/context 使用)。"""
        rr = _mock_reranker({0: 0.7, 1: 0.2})
        node = _make_reranker_node(rr, top_k=2, quality_k=2, quality_threshold=0.3)
        hits = [_hit("text", "h0", pk="p0"), _hit("text", "h1", pk="p1")]
        state = {
            "correlation_id": "t3",
            "query": "X",
            "route_results": {ROUTE_PROGRESSIVE: hits},
        }
        node(state)
        self.assertAlmostEqual(hits[0].rerank_score, 0.7, places=3)
        self.assertAlmostEqual(hits[1].rerank_score, 0.2, places=3)


# ---------------------------------------------------------------------------
# P0 #10: 未评分 (None) 不污染统计
# ---------------------------------------------------------------------------

class TestMissingScoreNotPolluting(unittest.TestCase):
    def test_diagnose_skips_none_scores_in_stats(self):
        """仅前 2 条评分, 后 2 条 None; 不应把 None 当 0 拉低均值。"""
        from pipeline.retrieval.rerank_diagnosis import _aggregate_scores

        hits_named = [("progressive", _hit("text", f"h{i}", pk=f"p{i}")) for i in range(4)]
        score_map = {0: 0.8, 1: 0.7, 2: None, 3: None}
        stats = _aggregate_scores(hits_named, score_map)
        # 旧逻辑: avg = (0.8 + 0.7 + 0 + 0) / 4 = 0.375
        # 新逻辑: avg = (0.8 + 0.7) / 2 = 0.75
        self.assertAlmostEqual(stats.global_avg, 0.75, places=3)
        # type_counts 仍按全量 hit 算 (反映检索池实际分布)
        self.assertEqual(stats.type_counts.get("text", 0), 4)

    def test_reranker_node_drops_missing_from_rankable(self):
        """reranker 只返回部分 hits → 未评分 hit 不进入 top-k。"""
        # 3 hits, 只对 idx=0,1 给分; idx=2 missing
        rr = _mock_reranker({0: 0.9, 1: 0.8})
        node = _make_reranker_node(rr, top_k=5, quality_k=3, quality_threshold=0.3)
        hits = [_hit("text", f"h{i}", pk=f"p{i}") for i in range(3)]
        state = {
            "correlation_id": "t4",
            "query": "X",
            "route_results": {ROUTE_PROGRESSIVE: hits},
        }
        out = node(state)
        self.assertFalse(out["needs_retry"])
        # 高分通过 gate → filtered_route_results 写回; 应只有 2 条 (idx=2 因 None 丢弃)
        filtered = out["route_results"][ROUTE_PROGRESSIVE]
        self.assertEqual(len(filtered), 2)
        # missing hit 的 rerank_score 仍是 None (P1 #14 + P0 #10)
        self.assertIsNone(hits[2].rerank_score)


# ---------------------------------------------------------------------------
# P1 #5: chunk_type-aware 阈值
# ---------------------------------------------------------------------------

class TestChunkTypeAwareThreshold(unittest.TestCase):
    def test_image_route_uses_lower_threshold(self):
        """image 路径平均分 0.20: 全局阈值 0.30 会 fail, 但 image 专属 0.18 应 pass。

        P0.2 (#1 修复): image hits 不再被自动豁免, 真正走 image 阈值。
        """
        # 2 hits 全是 image type, 分数 0.22, 0.18 (avg=0.20)
        rr = _mock_reranker({0: 0.22, 1: 0.18})
        node = _make_reranker_node(
            rr,
            top_k=2, quality_k=2,
            quality_threshold=0.30,
            quality_threshold_by_type={"image": 0.18},
        )
        hits = [_hit("image", f"fig{i}", pk=f"pi{i}") for i in range(2)]
        state = {
            "correlation_id": "t5",
            "query": "X",
            "route_results": {ROUTE_PROGRESSIVE: hits},
        }
        out = node(state)
        # 用 image 专属阈值 0.18, route_avg=0.20 ≥ 0.18 → pass
        self.assertFalse(out["needs_retry"])

    def test_text_route_still_uses_default(self):
        """text 路径未在 by_type 中指定 → 回退到全局 quality_threshold。"""
        rr = _mock_reranker({0: 0.22, 1: 0.18})
        node = _make_reranker_node(
            rr,
            top_k=2, quality_k=2,
            quality_threshold=0.30,
            quality_threshold_by_type={"image": 0.18},
        )
        hits = [_hit("text", f"t{i}", pk=f"pt{i}") for i in range(2)]
        state = {
            "correlation_id": "t6",
            "query": "X",
            "route_results": {ROUTE_PROGRESSIVE: hits},
        }
        out = node(state)
        # text → 0.30, route_avg=0.20 < 0.30 → gate fail → needs_retry
        self.assertTrue(out["needs_retry"])


class TestRouteThresholdsIntegration(unittest.TestCase):
    """P1.1: 验证 RouteThresholds 矩阵在 reranker_node 端到端生效。"""

    def test_progressive_l1_uses_loose_threshold(self):
        """L1 hits 分数 0.18, progressive.l1.text=0.15 → pass."""
        from pipeline.retrieval.quality_thresholds import RouteThresholds
        from pipeline.retrieval.retrievers import Hit

        rr = _mock_reranker({0: 0.20, 1: 0.16})
        thresholds = RouteThresholds.from_dict({
            "default": 0.25,
            "by_route": {
                "progressive": {
                    "l1": {"text": 0.15, "default": 0.10},
                    "l2": {"text": 0.32, "default": 0.20},
                },
            },
        })
        node = _make_reranker_node(
            rr,
            top_k=2, quality_k=2,
            quality_threshold=0.30,
            route_thresholds=thresholds,
        )
        l1_hits = [
            Hit(pk=f"p{i}", chunk_id=f"p{i}", type="text", content=f"l1-{i}", stage="l1")
            for i in range(2)
        ]
        state = {
            "correlation_id": "t-l1",
            "query": "q",
            "route_results": {ROUTE_PROGRESSIVE: l1_hits},
        }
        out = node(state)
        # avg=0.18 ≥ 0.15 (l1.text) → pass
        self.assertFalse(out["needs_retry"])

    def test_progressive_l2_uses_strict_threshold(self):
        """同分数 0.18, 但 stage=l2 → l2.text=0.32 → fail."""
        from pipeline.retrieval.quality_thresholds import RouteThresholds
        from pipeline.retrieval.retrievers import Hit

        rr = _mock_reranker({0: 0.20, 1: 0.16})
        thresholds = RouteThresholds.from_dict({
            "default": 0.25,
            "by_route": {
                "progressive": {
                    "l1": {"text": 0.15},
                    "l2": {"text": 0.32},
                },
            },
        })
        node = _make_reranker_node(
            rr,
            top_k=2, quality_k=2,
            quality_threshold=0.30,
            route_thresholds=thresholds,
        )
        l2_hits = [
            Hit(pk=f"p{i}", chunk_id=f"p{i}", type="text", content=f"l2-{i}", stage="l2")
            for i in range(2)
        ]
        state = {
            "correlation_id": "t-l2",
            "query": "q",
            "route_results": {ROUTE_PROGRESSIVE: l2_hits},
        }
        out = node(state)
        # avg=0.18 < 0.32 → fail
        self.assertTrue(out["needs_retry"])


class TestExemptSplit(unittest.TestCase):
    """P0.2: exempt 拆分 — 修复 #4 (metadata 无质量兜底) + #5 (混合路径污染)。"""

    def test_metadata_entity_only_participates_in_quality(self):
        """metadata route 无显式结构化引用 → topk_only → hits 仍走 rerank 评分。"""
        from pipeline.retrieval.retrievers import Hit

        # metadata route, 但无 fig_refs/page_refs → entity-only 模式
        rr = _mock_reranker({0: 0.08, 1: 0.05})  # 都很低
        node = _make_reranker_node(
            rr, top_k=5, quality_k=2,
            quality_threshold=0.20,
        )
        meta_hits = [
            Hit(pk=f"m{i}", chunk_id=f"m{i}", type="text", content=f"meta-{i}")
            for i in range(2)
        ]
        state = {
            "correlation_id": "t-meta-entity",
            "query": "MoS2 性质",
            "decision": RouteDecision(routes=["metadata"], entities=["MoS2"]),
            "route_results": {ROUTE_METADATA: meta_hits},
        }
        out = node(state)
        # 旧行为: metadata 全 exempt → exempt_rescue=True → 假通过
        # 新行为 P0.2: avg=0.065 < 0.20 → gate fail → needs_retry
        self.assertTrue(out["needs_retry"])

    def test_metadata_with_fig_refs_still_passes(self):
        """metadata + fig_refs → full exempt → exempt_rescue 通过 (硬过滤高置信)。"""
        from pipeline.retrieval.retrievers import Hit

        # 全部都是 image 类型 + fig_refs 引用 → full exempt → 不评分
        rr = _mock_reranker({})  # 不会被调用
        node = _make_reranker_node(
            rr, top_k=5, quality_k=2,
            quality_threshold=0.20,
        )
        meta_hits = [
            Hit(pk=f"m{i}", chunk_id=f"m{i}", type="image", content=f"Fig.{i+1}")
            for i in range(2)
        ]
        state = {
            "correlation_id": "t-meta-fig",
            "query": "图3说明什么",
            "decision": RouteDecision(routes=["metadata"], fig_refs=["3"]),
            "route_results": {ROUTE_METADATA: meta_hits},
        }
        out = node(state)
        # full exempt + exempt_rescue → pass
        self.assertFalse(out["needs_retry"])

    def test_mixed_route_image_pollution_now_visible(self):
        """progressive route 同时召回 text(高分) + image(低分, 与 query 无关):
        - 旧行为: 5 image 全 exempt, 只看 1 text → 路径过 gate
        - 新行为 P0.2: image 也参与评分, 拉低 route_avg → fail
        """
        from pipeline.retrieval.retrievers import Hit

        # 1 text 高分 + 5 image 低分 (caption 与 query 完全无关)
        scores = {0: 0.85, 1: 0.04, 2: 0.05, 3: 0.06, 4: 0.07, 5: 0.08}
        rr = _mock_reranker(scores)
        node = _make_reranker_node(
            rr, top_k=2, quality_k=4,    # quality_k=4 让 image 进入评分池
            quality_threshold=0.30,
        )
        hits = [
            Hit(pk="t1", chunk_id="t1", type="text", content="高度相关 text"),
        ] + [
            Hit(pk=f"img{i}", chunk_id=f"img{i}", type="image", content=f"无关 Fig.{i}")
            for i in range(5)
        ]
        state = {
            "correlation_id": "t-mix",
            "query": "X",
            "decision": None,  # 不指定 chunk_type → image 不豁免
            "route_results": {ROUTE_PROGRESSIVE: hits},
        }
        out = node(state)
        # 旧行为: image 全 exempt, 只看 t1=0.85 → pass
        # 新行为: top-4 评分 = [0.85, 0.08, 0.07, 0.06] avg=0.265 < 0.30 → fail
        self.assertTrue(out["needs_retry"])


# ---------------------------------------------------------------------------
# Rerank query: 单路泛化发话 → router rewrite
# ---------------------------------------------------------------------------

class TestRerankQuerySynthesis(unittest.TestCase):
    def test_single_path_vague_query_uses_router_rewrite(self):
        queries_seen: list[str] = []

        def side(query, documents, top_k):
            queries_seen.append(query)
            return [RerankResult(index=0, score=0.85, content=documents[0])]

        rr = MagicMock()
        rr.rerank.side_effect = side
        node = _make_reranker_node(rr, top_k=2, quality_k=1, quality_threshold=0.1)
        decision = RouteDecision(
            routes=[ROUTE_SUMMARY],
            rewrites={"summary": "固态电池 solid-state battery ASSB"},
            rerank_mode=True,
        )
        state = {
            "correlation_id": "t-vague",
            "query": "这方面有什么研究",
            "decision": decision,
            "route_results": {ROUTE_SUMMARY: [_hit("summary", "ASSB review", pk="s1")]},
        }
        node(state)
        self.assertEqual(queries_seen, ["固态电池 solid-state battery ASSB"])

    def test_resolve_rerank_query_for_hit_single_path(self):
        decision = RouteDecision(
            routes=[ROUTE_PROGRESSIVE],
            rewrites={"progressive": "MoS2 晶格常数"},
        )
        hit = Hit(pk="p1", chunk_id="p1", content="c")
        rq = _resolve_rerank_query_for_hit(
            hit, "MoS2 的晶格常数是多少", decision, [],
        )
        self.assertEqual(rq, "MoS2 的晶格常数是多少")

    def test_resolve_rerank_query_vague_single_path(self):
        decision = RouteDecision(
            routes=[ROUTE_SUMMARY],
            rewrites={"summary": "MoS2 lattice constant"},
            rerank_mode=True,
        )
        hit = Hit(pk="p1", chunk_id="p1", content="c")
        rq = _resolve_rerank_query_for_hit(
            hit, "这方面有什么", decision, [],
        )
        self.assertEqual(rq, "MoS2 lattice constant")


# ---------------------------------------------------------------------------
# P1 #17: reranker API 失败 fail-open
# ---------------------------------------------------------------------------

class TestRerankerFailOpen(unittest.TestCase):
    def test_empty_rerank_results_treated_as_failure(self):
        """rerank() 返回 [] (fail_open=True 时的退化输出) → 不门控, 不重试。"""
        rr = MagicMock()
        rr.rerank.return_value = []
        node = _make_reranker_node(rr, top_k=5, quality_k=3, quality_threshold=0.3)
        hits = [_hit("text", "x", pk="p0")]
        state = {
            "correlation_id": "t7",
            "query": "X",
            "route_results": {ROUTE_PROGRESSIVE: hits},
        }
        out = node(state)
        self.assertFalse(out["needs_retry"])
        self.assertEqual(out["rerank_diagnosis_cause"], "rerank_failed")
        self.assertEqual(out["reranker_score"], -1.0)
        # 原 route_results 保留, 没有写 filtered (上游降级用 emb_score)
        self.assertIs(out["route_results"][ROUTE_PROGRESSIVE], hits)

    def test_rerank_exception_caught_via_fail_open(self):
        """rerank() 抛异常时, 节点不应崩溃; 走 fail-open。"""
        rr = MagicMock()
        rr.rerank.side_effect = RuntimeError("connection refused")
        node = _make_reranker_node(rr, top_k=5, quality_k=3, quality_threshold=0.3)
        hits = [_hit("text", "x", pk="p0")]
        state = {
            "correlation_id": "t8",
            "query": "X",
            "route_results": {ROUTE_PROGRESSIVE: hits},
        }
        # 不应抛: node 内部捕获 + 降级
        out = node(state)
        self.assertFalse(out["needs_retry"])
        self.assertEqual(out["rerank_diagnosis_cause"], "rerank_failed")


class TestRerankerClientFailOpen(unittest.TestCase):
    def test_fail_open_returns_empty_on_total_failure(self):
        """RerankerClient.fail_open=True 时, 全部重试失败应返回 [] 而非抛错。"""
        client = RerankerClient(
            api_base="http://invalid-host-12345.local/v1",
            model="x",
            top_k=3,
            timeout=1,
            max_retries=1,
            fail_open=True,
        )
        # httpx 会立即抛 (DNS / connect), 但 fail_open=True 应吞掉
        out = client.rerank("q", ["a", "b"], top_k=2)
        self.assertEqual(out, [])

    def test_max_retries_zero_clamped_to_one(self):
        """max_retries=0 会触发 raise None bug; 应被规约为最少 1。"""
        client = RerankerClient(
            api_base="http://x.local/v1",
            max_retries=0,
            fail_open=True,
        )
        self.assertEqual(client.max_retries, 1)


# ---------------------------------------------------------------------------
# P0 #8: R1 wrong_type 收紧 - 单纯实体不触发 metadata
# ---------------------------------------------------------------------------

class TestR1WrongTypeTightening(unittest.TestCase):
    def _run(self, query: str, hits: list, scores: list, *, docs=None):
        all_hits = [("progressive", h) for h in hits]
        # 用 Optional[float] 形式与新签名匹配
        score_map = {i: s for i, s in enumerate(scores)}
        decision = RouteDecision(
            routes=[ROUTE_PROGRESSIVE],
            rewrites={ROUTE_PROGRESSIVE: query},
        )
        return diagnose_rerank_failure(
            query=query,
            decision=decision,
            all_hits=all_hits,
            score_map=score_map,
            quality_score=0.1,
            quality_threshold=0.3,
            this_round_docs=docs or [],
            config=RerankDiagnosisConfig(),
        )

    def test_entity_only_query_does_not_trigger_wrong_type(self):
        """'MoS2 晶格常数' 只含实体, 不应被误判为 wrong_type/metadata。"""
        hits = [_hit("text", f"c{i}", pk=f"p{i}") for i in range(4)]
        scores = [0.10, 0.08, 0.05, 0.07]
        d = self._run("MoS2 晶格常数是多少", hits, scores)
        # 旧逻辑: 实体存在即 wrong_type, 添加 metadata 路径
        # 新逻辑 (P0 #8): 应当落入 off_topic / too_narrow / too_broad / zero 等其他 cause
        self.assertNotEqual(d.cause, "wrong_type")
        self.assertNotIn(ROUTE_METADATA, d.suggested.routes)

    def test_fig_ref_still_triggers_wrong_type_strong(self):
        """'图3' 引用 + image 缺失 → wrong_type strong (P2.1: 强信号 confidence ≥ 0.92, 跳过 reflect)。"""
        hits = [_hit("text", f"c{i}", pk=f"p{i}") for i in range(4)]
        scores = [0.12, 0.10, 0.08, 0.11]
        d = self._run("图3 显示了什么内容", hits, scores)
        self.assertEqual(d.cause, "wrong_type")
        self.assertIn(ROUTE_METADATA, d.suggested.routes)
        # P2.1: confidence 由信号强度连续计算, 强信号 ≥ base (0.92) 且 ≤ 0.99
        self.assertGreaterEqual(d.confidence, 0.92)
        self.assertLessEqual(d.confidence, 0.99)
        # 默认 skip_reflect_confidence=0.90, 强信号 ≥ 0.90 → 跳过
        self.assertTrue(d.skip_reflect)

    def test_page_ref_only_uses_weak_confidence(self):
        """'第5页' 单独引用 → wrong_type weak (默认 0.80), 在默认 skip 阈 0.90 下不跳过 reflect。"""
        hits = [_hit("text", f"c{i}", pk=f"p{i}") for i in range(4)]
        scores = [0.12, 0.10, 0.08, 0.11]
        d = self._run("第5页讲了什么", hits, scores)
        # 仍是 wrong_type, 但置信度降为 weak
        self.assertEqual(d.cause, "wrong_type")
        # P2.1: 弱信号 confidence 在 base (0.80) 附近 ±0.06 浮动
        self.assertGreaterEqual(d.confidence, 0.74)
        self.assertLessEqual(d.confidence, 0.86)
        # 默认 skip 阈 0.90, weak < 0.90 → 不跳过 reflect (走 LLM 兜底)
        self.assertFalse(d.skip_reflect)


# ---------------------------------------------------------------------------
# P1 #9: skip_reflect 阈值与 confidence 分层的协同
# ---------------------------------------------------------------------------

class TestSkipReflectConfidenceLayering(unittest.TestCase):
    def test_default_skip_threshold_is_090(self):
        """RerankDiagnosisConfig.skip_reflect_confidence 默认应该是 0.90 (不再是 0.85)。"""
        cfg = RerankDiagnosisConfig()
        self.assertAlmostEqual(cfg.skip_reflect_confidence, 0.90, places=2)

    def test_wrong_type_strong_default_confidence_is_092(self):
        cfg = RerankDiagnosisConfig()
        self.assertAlmostEqual(cfg.wrong_type_strong_confidence, 0.92, places=2)

    def test_wrong_type_weak_default_confidence_is_080(self):
        cfg = RerankDiagnosisConfig()
        self.assertAlmostEqual(cfg.wrong_type_weak_confidence, 0.80, places=2)


class TestFailOpenEmbSafetyNet(unittest.TestCase):
    """P2.3: rerank 失败时若 emb_score 也低, 应触发安全网 retry."""

    def test_safety_net_triggers_retry_when_emb_low(self):
        from pipeline.retrieval.retrievers import Hit
        rr = MagicMock()
        rr.rerank.return_value = []  # fail_open 退化
        node = _make_reranker_node(
            rr, top_k=2, quality_k=2, quality_threshold=0.3,
            fail_open_min_emb_quality=0.4,
        )
        # emb_score 都很低 (0.05, 0.08), top-2 avg = 0.065 < 0.4
        hits = [
            Hit(pk="p0", chunk_id="p0", type="text", content="x", score=0.05),
            Hit(pk="p1", chunk_id="p1", type="text", content="y", score=0.08),
        ]
        state = {
            "correlation_id": "t-emb-low",
            "query": "X",
            "route_results": {ROUTE_PROGRESSIVE: hits},
        }
        out = node(state)
        self.assertTrue(out["needs_retry"])
        self.assertEqual(out["rerank_diagnosis_cause"], "rerank_failed_low_emb")
        self.assertIsInstance(out["rewrite_hint"], RouteDecision)

    def test_safety_net_not_triggered_when_emb_high(self):
        from pipeline.retrieval.retrievers import Hit
        rr = MagicMock()
        rr.rerank.return_value = []
        node = _make_reranker_node(
            rr, top_k=2, quality_k=2, quality_threshold=0.3,
            fail_open_min_emb_quality=0.4,
        )
        # emb_score 高 (0.85, 0.9), top-2 avg = 0.875 ≥ 0.4 → 旧 fail-open 行为
        hits = [
            Hit(pk="p0", chunk_id="p0", type="text", content="x", score=0.85),
            Hit(pk="p1", chunk_id="p1", type="text", content="y", score=0.9),
        ]
        state = {
            "correlation_id": "t-emb-high",
            "query": "X",
            "route_results": {ROUTE_PROGRESSIVE: hits},
        }
        out = node(state)
        self.assertFalse(out["needs_retry"])
        self.assertEqual(out["rerank_diagnosis_cause"], "rerank_failed")


class TestMultiCausePatchMerge(unittest.TestCase):
    """P1.3: 多 cause 并发时, 次因的 routes/filter 应合并入主因 patch."""

    def test_secondary_routes_merged_into_primary(self):
        from pipeline.retrieval.rerank_diagnosis import (
            RerankDiagnosisConfig,
            diagnose_rerank_failure,
        )
        from pipeline.retrieval.retrievers import Hit

        # 场景: 命中多 (too_broad) + 有 fig_refs (wrong_type strong)
        # 主因应该是 wrong_type (信号强), 次因 too_broad 的 LOCAL 路径应合并入
        hits = [Hit(pk=f"p{i}", chunk_id=f"p{i}", type="text", content=f"c{i}") for i in range(12)]
        scores = [0.10] * 12  # 所有分数低
        all_hits = [("progressive", h) for h in hits]
        score_map = {i: 0.10 for i in range(12)}
        decision = RouteDecision(
            routes=[ROUTE_PROGRESSIVE],
            rewrites={ROUTE_PROGRESSIVE: "图3 解释机制"},
        )
        # 提供 docs 让 too_broad 走 local 分支
        docs = [{"doc_id": "d1", "doc_name": "Paper A"}]
        d = diagnose_rerank_failure(
            query="图3 解释机制",
            decision=decision,
            all_hits=all_hits,
            score_map=score_map,
            quality_score=0.10,
            quality_threshold=0.30,
            this_round_docs=docs,
            config=RerankDiagnosisConfig(broad_hit_floor=10),
        )
        # 主因应该是 wrong_type (强信号)
        self.assertEqual(d.cause, "wrong_type")
        # 主因 patch routes 含 metadata
        self.assertIn(ROUTE_METADATA, d.suggested.routes)
        # 次因 too_broad 提供的 LOCAL 路径应合并进来
        self.assertIn(ROUTE_LOCAL, d.suggested.routes)


class TestDiagnosisThresholdsConsistency(unittest.TestCase):
    """P1.2: 诊断与 gate 用同一份 RouteThresholds 矩阵."""

    def test_diagnosis_uses_per_type_threshold_not_global(self):
        from pipeline.retrieval.quality_thresholds import RouteThresholds
        from pipeline.retrieval.rerank_diagnosis import (
            RerankDiagnosisConfig,
            diagnose_rerank_failure,
        )
        from pipeline.retrieval.retrievers import Hit

        # 11 image hits, 全是低分 (0.08), quality_score=0.08
        # 旧逻辑: too_broad 用 quality_threshold=0.30, 0.08 < 0.30 → 触发
        # 新逻辑 (P1.2): 用 by_type[image]=0.10 → 0.08 < 0.10 仍触发 (但阈值变了)
        hits = [Hit(pk=f"p{i}", chunk_id=f"p{i}", type="image", content=f"c{i}") for i in range(11)]
        all_hits = [("progressive", h) for h in hits]
        score_map = {i: 0.08 for i in range(11)}
        decision = RouteDecision(routes=[ROUTE_PROGRESSIVE], rewrites={ROUTE_PROGRESSIVE: "q"})
        thresholds = RouteThresholds.from_dict({
            "default": 0.30,
            "by_type": {"image": 0.10},
        })
        d = diagnose_rerank_failure(
            query="MoS2 性质",
            decision=decision,
            all_hits=all_hits,
            score_map=score_map,
            quality_score=0.08,
            quality_threshold=0.30,
            this_round_docs=[],
            config=RerankDiagnosisConfig(broad_hit_floor=10),
            route_thresholds=thresholds,
        )
        # quality_score=0.08 < image 阈值 0.10 → too_broad 触发
        self.assertEqual(d.cause, "too_broad")


class TestZScoreNormalization(unittest.TestCase):
    """P2.2: 多 group 时, 跨 group z-score 归一化避免不同 query 分数尺度污染统计."""

    def test_aggregate_with_groups_normalizes(self):
        from pipeline.retrieval.rerank_diagnosis import _aggregate_scores
        from pipeline.retrieval.retrievers import Hit

        # 2 groups: A 是高难度子查询 (0.05-0.15), B 是常识 (0.7-0.9)
        all_hits = [("progressive", Hit(pk=f"p{i}", type="text", content="x")) for i in range(6)]
        score_map = {0: 0.05, 1: 0.10, 2: 0.15, 3: 0.70, 4: 0.80, 5: 0.90}
        rerank_groups = {0: "A", 1: "A", 2: "A", 3: "B", 4: "B", 5: "B"}

        # 没传 groups: global_avg ≈ 0.45 (混合)
        stats_raw = _aggregate_scores(all_hits, score_map)
        self.assertAlmostEqual(stats_raw.global_avg, 0.45, places=2)

        # 传 groups: 每组 z-score 后 global_avg 接近 0 (均值 0)
        stats_norm = _aggregate_scores(all_hits, score_map, rerank_groups=rerank_groups)
        self.assertAlmostEqual(stats_norm.global_avg, 0.0, places=2)


if __name__ == "__main__":
    unittest.main()
