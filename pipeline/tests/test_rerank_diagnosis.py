"""rerank_diagnosis 单元测试 (synthetic hits, 不依赖 reranker API)。"""

from __future__ import annotations

import unittest

from pipeline.models import RouteDecision
from pipeline.retrieval.agentic import (
    ROUTE_METADATA,
    ROUTE_PROGRESSIVE,
    ROUTE_LOCAL,
    ROUTE_SUMMARY,
)
from pipeline.retrieval.retrievers import Hit
from pipeline.retrieval.rerank_diagnosis import (
    RerankDiagnosisConfig,
    diagnose_rerank_failure,
    should_skip_reflect_after_reranker,
)


def _hit(chunk_type: str = "text", content: str = "chunk") -> Hit:
    return Hit(pk=f"pk-{chunk_type}-{content[:4]}", type=chunk_type, content=content)


class TestRerankDiagnosis(unittest.TestCase):
    def _run(
        self,
        query: str,
        decision: RouteDecision,
        hits: list,
        scores: list,
        *,
        quality_score: float = 0.1,
        threshold: float = 0.3,
        docs: list | None = None,
        config: RerankDiagnosisConfig | None = None,
    ):
        all_hits = [("progressive", h) for h in hits]
        score_map = {i: s for i, s in enumerate(scores)}
        return diagnose_rerank_failure(
            query=query,
            decision=decision,
            all_hits=all_hits,
            score_map=score_map,
            quality_score=quality_score,
            quality_threshold=threshold,
            this_round_docs=docs or [],
            config=config or RerankDiagnosisConfig(),
        )

    def test_fig3_wrong_type_adds_metadata(self):
        decision = RouteDecision(
            routes=[ROUTE_PROGRESSIVE],
            rewrites={ROUTE_PROGRESSIVE: "图3说明了什么"},
        )
        hits = [_hit("text", "paragraph about structure")] * 4
        scores = [0.12, 0.10, 0.08, 0.11]
        d = self._run("图3说明了什么", decision, hits, scores)
        self.assertEqual(d.cause, "wrong_type")
        self.assertIn(ROUTE_METADATA, d.suggested.routes)
        self.assertTrue(d.suggested.fig_refs)
        self.assertTrue(d.skip_reflect)

    def test_references_intent(self):
        decision = RouteDecision(routes=[ROUTE_PROGRESSIVE], rewrites={ROUTE_PROGRESSIVE: "q"})
        hits = [_hit("text", "body")] * 3
        scores = [0.1, 0.12, 0.09]
        d = self._run("这篇的参考文献有哪些", decision, hits, scores)
        self.assertEqual(d.cause, "wrong_type")
        self.assertEqual(d.suggested.chunk_type, "references")

    def test_too_broad_switches_local_with_docs(self):
        decision = RouteDecision(routes=[ROUTE_PROGRESSIVE], rewrites={ROUTE_PROGRESSIVE: "q"})
        hits = [_hit("text", f"c{i}") for i in range(16)]
        scores = [0.08] * 16
        docs = [{"doc_id": "d1", "doc_name": "Paper A"}]
        d = self._run("MoS2 性质", decision, hits, scores, docs=docs)
        self.assertEqual(d.cause, "too_broad")
        self.assertIn(ROUTE_LOCAL, d.suggested.routes)

    def test_off_topic_fallback_keywords(self):
        decision = RouteDecision(routes=[ROUTE_PROGRESSIVE], rewrites={ROUTE_PROGRESSIVE: "q"})
        hits = [_hit("text", "x"), _hit("text", "y")]
        scores = [0.2, 0.18]
        d = self._run("LiNiCoMnO2 合成温度", decision, hits, scores, quality_score=0.19)
        self.assertEqual(d.cause, "off_topic")
        self.assertIn(ROUTE_PROGRESSIVE, d.suggested.routes)
        rw = d.suggested.rewrites.get(ROUTE_PROGRESSIVE, "")
        self.assertNotEqual(rw.strip(), "")

    def test_suggested_is_route_decision(self):
        decision = RouteDecision(routes=[ROUTE_SUMMARY], rewrites={ROUTE_SUMMARY: "总结"})
        hits = [_hit("text", "a")]
        scores = [0.05]
        d = self._run("总结", decision, hits, scores)
        self.assertIsInstance(d.suggested, RouteDecision)
        self.assertFalse(d.skip_reflect)  # off_topic 不在 skip_reflect_causes 白名单

    def test_skip_reflect_requires_cause_whitelist(self):
        decision = RouteDecision(
            routes=[ROUTE_PROGRESSIVE],
            rewrites={ROUTE_PROGRESSIVE: "图3说明了什么"},
        )
        hits = [_hit("text", "paragraph about structure")] * 4
        scores = [0.12, 0.10, 0.08, 0.11]
        cfg = RerankDiagnosisConfig(skip_reflect_causes=())
        d = self._run("图3说明了什么", decision, hits, scores, config=cfg)
        self.assertEqual(d.cause, "wrong_type")
        self.assertFalse(d.skip_reflect)

    def test_should_skip_reflect_after_reranker(self):
        hint = RouteDecision(routes=[ROUTE_METADATA], reasoning="x")
        self.assertTrue(
            should_skip_reflect_after_reranker(
                skip_reflect=True,
                rewrite_hint=hint,
                subquery_decisions=[hint],
                retry_count=0,
                max_retries=2,
            )
        )
        self.assertFalse(
            should_skip_reflect_after_reranker(
                skip_reflect=True,
                rewrite_hint=hint,
                subquery_decisions=[hint],
                retry_count=2,
                max_retries=2,
            )
        )
        self.assertFalse(
            should_skip_reflect_after_reranker(
                skip_reflect=False,
                rewrite_hint=hint,
                subquery_decisions=[hint],
                retry_count=0,
                max_retries=2,
            )
        )

    def test_after_reranker_routing(self):
        from pipeline.retrieval.langgraph_agent import _after_reranker

        hint = RouteDecision(routes=[ROUTE_METADATA], reasoning="x")
        self.assertEqual(
            _after_reranker({
                "needs_retry": True,
                "rerank_skip_reflect": True,
                "rewrite_hint": hint,
                "subquery_decisions": [hint],
                "retry_count": 0,
                "max_retries": 2,
            }),
            "rewrite",
        )
        self.assertEqual(
            _after_reranker({
                "needs_retry": True,
                "rerank_skip_reflect": False,
                "rewrite_hint": hint,
                "retry_count": 0,
                "max_retries": 2,
            }),
            "reflect",
        )
        self.assertEqual(
            _after_reranker({"needs_retry": False}),
            "context_build",
        )

    def test_summary_in_block(self):
        decision = RouteDecision(routes=[ROUTE_PROGRESSIVE], rewrites={ROUTE_PROGRESSIVE: "q"})
        hits = [_hit("text", "a")]
        scores = [0.05]
        d = self._run("图3内容", decision, hits, scores)
        self.assertIn("Reranker 诊断", d.summary)
        self.assertIn("cause=", d.summary)


if __name__ == "__main__":
    unittest.main()
