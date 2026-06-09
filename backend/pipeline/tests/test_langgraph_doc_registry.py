"""langgraph doc_registry 持久化与 time 降级逻辑单元测试。"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from pipeline.models import RouteDecision
from pipeline.retrieval.agentic import LocalRetrieveResult
from pipeline.retrieval.langgraph_agent import (
    _apply_reflect_failure_state,
    _count_hits,
    _execute_single_retrieval,
    _fallback_context_on_build_error,
    _format_reflect_strategy,
    _has_rerank_retry_fallback,
    _is_catalog_shrinking_drilldown,
    _make_context_build_node,
    _parse_chunk_type,
    _persist_doc_registry,
    _make_reranker_node,
)
from pipeline.retrieval.agentic import ROUTE_PROGRESSIVE
from pipeline.retrieval.retrievers import Hit
from pipeline.clients.reranker import RerankResult


def _docs(
    n: int, prefix: str = "paper", *, pinned: bool = False,
) -> list[dict[str, object]]:
    return [
        {
            "doc_id": f"{prefix}-{i}",
            "doc_name": f"{prefix.title()} {i}",
            "pinned": pinned,
        }
        for i in range(1, n + 1)
    ]


class TestDocRegistryPersistence(unittest.TestCase):
    def test_local_drilldown_preserves_catalog(self):
        prev = _docs(5)
        this = [_docs(1)[0]]
        decision = RouteDecision(routes=["local"], target_docs=["Paper 1"])
        self.assertTrue(_is_catalog_shrinking_drilldown(decision, prev, this))
        out = _persist_doc_registry(prev, this, decision)
        self.assertEqual(len(out), 5)
        self.assertEqual(out[2]["doc_name"], "Paper 3")

    def test_summary_discovery_refreshes_catalog(self):
        prev = _docs(3, "old")
        this = _docs(4, "new")
        decision = RouteDecision(routes=["summary"], rewrites={"summary": "kw"})
        out = _persist_doc_registry(prev, this, decision)
        self.assertEqual(len(out), 4)
        self.assertEqual(out[0]["doc_id"], "new-1")

    def test_progressive_single_subset_merges_catalog(self):
        prev = _docs(5)
        this = [_docs(5)[0]]
        decision = RouteDecision(
            routes=["progressive"],
            rewrites={"progressive": "methods in paper 1"},
        )
        self.assertFalse(_is_catalog_shrinking_drilldown(decision, prev, this))
        out = _persist_doc_registry(prev, this, decision)
        self.assertEqual(len(out), 5)

    def test_progressive_multi_subset_refreshes_catalog(self):
        prev = _docs(5)
        this = _docs(3)
        decision = RouteDecision(
            routes=["progressive"],
            rewrites={"progressive": "methods A B C"},
        )
        self.assertFalse(_is_catalog_shrinking_drilldown(decision, prev, this))
        out = _persist_doc_registry(prev, this, decision)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[2]["doc_id"], "paper-3")

    def test_empty_this_round_keeps_prev(self):
        prev = _docs(2)
        out = _persist_doc_registry(prev, [], None)
        self.assertEqual(out, prev)


class TestTimeFallback(unittest.TestCase):
    def test_count_hits_empty(self):
        self.assertEqual(_count_hits({"summary": []}), 0)
        self.assertEqual(_count_hits({"local": LocalRetrieveResult()}), 0)

    def test_execute_single_retrieval_retries_without_time_on_zero_hits(self):
        decision = RouteDecision(
            routes=["summary"],
            rewrites={"summary": "test"},
            time="2099-2099",
        )
        summary_r = MagicMock()
        summary_r.retrieve.side_effect = [[], [MagicMock(pk="h1")]]
        local_r = MagicMock()
        metadata_r = MagicMock()

        results, _ = _execute_single_retrieval(
            decision, "q", "cid", summary_r, local_r, metadata_r,
        )
        self.assertEqual(summary_r.retrieve.call_count, 2)
        self.assertIsNone(summary_r.retrieve.call_args_list[1].args[3])
        self.assertEqual(_count_hits(results), 1)

    def test_compound_subquery_prefixes_route_keys(self):
        decision = RouteDecision(
            routes=["progressive"],
            rewrites={"progressive": "paper A methods"},
        )
        hit = Hit(pk="h1", chunk_id="h1", content="method chunk")
        local_r = MagicMock()
        local_r.retrieve.return_value = LocalRetrieveResult(chunk_hits=[hit])
        summary_r = MagicMock()
        metadata_r = MagicMock()

        results, errors = _execute_single_retrieval(
            decision, "compound q", "cid", summary_r, local_r, metadata_r,
            subquery_id="sub1",
            subquery_rewrite="paper A methods",
        )
        self.assertIn("sub1:progressive", results)
        self.assertNotIn("progressive", results)
        stamped = results["sub1:progressive"].chunk_hits[0]
        self.assertEqual(stamped.subquery_id, "sub1")
        self.assertEqual(stamped.subquery_rewrite, "paper A methods")


class TestCompoundRerank(unittest.TestCase):
    def test_reranker_uses_subquery_rewrite_per_group(self):
        hit_a = Hit(
            pk="a", chunk_id="a", content="methods text",
            subquery_id="sub1",
        )
        hit_b = Hit(
            pk="b", chunk_id="b", content="data text",
            subquery_id="sub2",
        )
        rr = MagicMock()
        queries_seen: list[str] = []

        def side(query, documents, top_k):
            queries_seen.append(query)
            return [
                RerankResult(index=0, score=0.9, content=documents[0]),
            ]

        rr.rerank.side_effect = side
        node = _make_reranker_node(rr, top_k=2, quality_threshold=0.1)
        state = {
            "correlation_id": "t",
            "query": "A methods and B data",
            "route_results": {
                "sub1:progressive": LocalRetrieveResult(chunk_hits=[hit_a]),
                "sub2:progressive": LocalRetrieveResult(chunk_hits=[hit_b]),
            },
            "decision": RouteDecision(routes=["progressive"]),
            "subquery_decisions": [
                RouteDecision(
                    routes=["progressive"],
                    rewrites={"progressive": "paper A methods"},
                    rerank_mode=True,
                ),
                RouteDecision(
                    routes=["progressive"],
                    rewrites={"progressive": "paper B data"},
                    rerank_mode=True,
                ),
            ],
        }
        out = node(state)
        self.assertIn("paper A methods", queries_seen)
        self.assertIn("paper B data", queries_seen)
        self.assertNotIn("A methods and B data", queries_seen)
        self.assertFalse(out.get("needs_retry", False))


class TestParseChunkType(unittest.TestCase):
    def test_accepts_equation_and_references(self):
        self.assertEqual(_parse_chunk_type({"chunk_type": "references"}), "references")
        self.assertEqual(_parse_chunk_type({"chunk_type": "equation"}), "equation")
        self.assertEqual(_parse_chunk_type({"chunk_type": "image"}), "image")
        self.assertIsNone(_parse_chunk_type({"chunk_type": "text"}))


class TestReflectStrategyFormat(unittest.TestCase):
    def test_compound_uses_per_subquery_rewrites(self):
        subs = [
            RouteDecision(
                routes=["progressive"],
                rewrites={"progressive": "paper A methods"},
            ),
            RouteDecision(
                routes=["progressive"],
                rewrites={"progressive": "paper B data"},
            ),
        ]
        routes, rewrites, filters = _format_reflect_strategy(
            RouteDecision(routes=["progressive"], rewrites={"progressive": "paper A methods | paper B data"}),
            subs,
        )
        self.assertIn("sub1:progressive", routes)
        self.assertIn("sub2:progressive", routes)
        self.assertIn("paper A methods", rewrites)
        self.assertIn("paper B data", rewrites)
        self.assertNotIn(" | ", rewrites)
        self.assertIn("sub1", filters)
        self.assertIn("sub2", filters)


class TestReflectRerankFallback(unittest.TestCase):
    def test_has_rerank_fallback_requires_rewrite(self):
        hint = RouteDecision(
            routes=[ROUTE_PROGRESSIVE],
            rewrites={ROUTE_PROGRESSIVE: "q2"},
            reasoning="diag",
        )
        self.assertTrue(_has_rerank_retry_fallback({
            "needs_retry": True,
            "rewrite_hint": hint,
        }))
        self.assertFalse(_has_rerank_retry_fallback({"needs_retry": True}))
        self.assertFalse(_has_rerank_retry_fallback({
            "needs_retry": False,
            "rewrite_hint": hint,
        }))

    def test_apply_reflect_failure_preserves_rerank_hint(self):
        hint = RouteDecision(
            routes=[ROUTE_PROGRESSIVE],
            rewrites={ROUTE_PROGRESSIVE: "broader q"},
            reasoning="too_narrow",
        )
        state = {
            "needs_retry": True,
            "rewrite_hint": hint,
            "subquery_decisions": [hint],
            "rerank_diagnosis_cause": "too_narrow",
            "rerank_skip_reflect": False,
        }
        _apply_reflect_failure_state(
            state, "t1", source="routing_core.reflect",
            error=RuntimeError("fc timeout"),
        )
        self.assertTrue(state["needs_retry"])
        self.assertIs(state["rewrite_hint"], hint)
        self.assertEqual(state["subquery_decisions"], [hint])

    def test_apply_reflect_failure_clears_without_fallback(self):
        state = {"needs_retry": True, "rewrite_hint": None}
        _apply_reflect_failure_state(
            state, "t2", source="routing_core.reflect", error="err",
        )
        self.assertFalse(state["needs_retry"])
        self.assertIsNone(state["rewrite_hint"])


class TestContextBuildResilience(unittest.TestCase):
    def test_fallback_context_message(self):
        ctx = _fallback_context_on_build_error(
            "离子镀",
            {"summary": [_hit("h1")]},
            ValueError("render broke"),
        )
        self.assertIn("离子镀", ctx)
        self.assertIn("上下文构建失败", ctx)
        self.assertIn("ValueError", ctx)

    def test_context_build_node_survives_build_error(self):
        builder = MagicMock()
        builder.build.side_effect = RuntimeError("boom")
        node = _make_context_build_node(builder)
        decision = RouteDecision(
            routes=["summary"],
            rewrites={"summary": "q"},
        )
        out = node({
            "query": "q",
            "decision": decision,
            "route_results": {"summary": []},
            "correlation_id": "cb-test",
            "needs_retry": False,
        })
        self.assertIn("上下文构建失败", out["context"])
        builder.build.assert_called_once()


def _hit(doc_id: str) -> Hit:
    return Hit(
        pk=f"pk-{doc_id}",
        doc_id=doc_id,
        doc_name=doc_id,
        score=0.5,
        type="text",
        content="c",
    )


class TestNoAnswerExit(unittest.TestCase):
    def test_policy_reflect_exhausted_sets_no_answer(self):
        from pipeline.retrieval.langgraph_agent import _make_policy_node

        node = _make_policy_node()
        state = {
            "correlation_id": "na1",
            "query": "不存在的问题",
            "agent_phase": "reflect",
            "needs_retry": True,
            "sufficient": False,
            "retry_count": 1,
            "max_retries": 1,
            "evidence_gaps": ["insufficient_evidence"],
            "node_timings": {},
        }
        out = node(state)
        self.assertEqual(out["next_action"], "answer")
        self.assertTrue(out["no_answer"])
        self.assertIn("没有找到足够可靠的依据", out["answer"])


if __name__ == "__main__":
    unittest.main()
