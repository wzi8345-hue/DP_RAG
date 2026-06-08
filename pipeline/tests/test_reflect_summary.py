"""reflect_summary: 反思摘要排除结构化 chunk, 按路径展示。"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from pipeline.retrieval.agentic import LocalRetrieveResult
from pipeline.retrieval.langgraph_agent import _make_reranker_node
from pipeline.retrieval.reflect_summary import (
    ReflectSummaryConfig,
    is_reflect_eligible,
    should_accept_structural_only_results,
    summarize_for_reflect,
)
from pipeline.models import RouteDecision
from pipeline.retrieval.retrievers import Hit


def _hit(pk: str, *, ctype: str, content: str) -> Hit:
    return Hit(pk=pk, chunk_id=pk, type=ctype, content=content, score=0.5)


class TestReflectEligible(unittest.TestCase):
    def test_metadata_route_excluded(self):
        self.assertFalse(is_reflect_eligible("metadata", _hit("1", ctype="text", content="x")))
        self.assertFalse(is_reflect_eligible("sub1:metadata", _hit("1", ctype="text", content="x")))

    def test_structural_types_excluded(self):
        self.assertFalse(is_reflect_eligible("progressive", _hit("1", ctype="image", content="Fig")))
        self.assertFalse(is_reflect_eligible("progressive", _hit("2", ctype="references", content="[1]")))

    def test_text_included(self):
        self.assertTrue(is_reflect_eligible("progressive", _hit("1", ctype="text", content="body")))


class TestSummarizeForReflect(unittest.TestCase):
    def test_excludes_metadata_and_structural(self):
        route_results = {
            "metadata": [_hit("m1", ctype="image", content="Fig.1 caption")],
            "progressive": [
                _hit("t1", ctype="text", content="A" * 500),
                _hit("t2", ctype="table", content="Table data"),
            ],
        }
        summary, total = summarize_for_reflect(route_results)
        self.assertEqual(total, 1)
        self.assertIn("[progressive]", summary)
        self.assertNotIn("[metadata]", summary)
        self.assertNotIn("Fig.1", summary)
        self.assertNotIn("Table data", summary)
        self.assertGreater(len(summary), 200)

    def test_per_route_sections(self):
        route_results = {
            "progressive": [_hit("p1", ctype="text", content="progressive hit")],
            "local": [_hit("l1", ctype="text", content="local hit")],
        }
        summary, total = summarize_for_reflect(route_results)
        self.assertEqual(total, 2)
        self.assertIn("[progressive]", summary)
        self.assertIn("[local]", summary)

    def test_long_snippet_uses_head_tail(self):
        long_text = ("START-" + "x" * 600 + "-END")
        route_results = {"local": [_hit("1", ctype="text", content=long_text)]}
        summary, _ = summarize_for_reflect(
            route_results,
            config=ReflectSummaryConfig(snippet_chars=400),
        )
        self.assertIn("START-", summary)
        self.assertIn("-END", summary)


class TestStructuralOnlyAccept(unittest.TestCase):
    def test_references_only_not_empty_for_reflect_gate(self):
        decision = RouteDecision(routes=["progressive"], chunk_type="references")
        route_results = {
            "progressive": [_hit("r1", ctype="references", content="[1] ref")],
        }
        self.assertTrue(should_accept_structural_only_results(route_results, decision))

    def test_mixed_text_and_structural_not_accept(self):
        decision = RouteDecision(routes=["progressive"], chunk_type="references")
        route_results = {
            "progressive": [
                _hit("t1", ctype="text", content="body"),
                _hit("r1", ctype="references", content="[1]"),
            ],
        }
        self.assertFalse(should_accept_structural_only_results(route_results, decision))


class TestPerRouteRerank(unittest.TestCase):
    def test_each_route_gets_own_top_k(self):
        reranker = MagicMock()

        def rerank_side_effect(query, documents, top_k):
            from pipeline.clients.reranker import RerankResult

            out = []
            for i, doc in enumerate(documents):
                score = 0.95 if "high" in doc else 0.2
                out.append(RerankResult(index=i, score=score, content=doc))
            return out

        reranker.rerank.side_effect = rerank_side_effect
        node = _make_reranker_node(reranker, top_k=1, quality_k=1, quality_threshold=0.5)

        state = {
            "correlation_id": "test",
            "query": "q",
            "decision": None,
            "route_results": {
                "progressive": [
                    Hit(pk="p1", chunk_id="p1", type="text", content="p low"),
                    Hit(pk="p2", chunk_id="p2", type="text", content="p high"),
                ],
                "local": [
                    Hit(pk="l1", chunk_id="l1", type="text", content="l low"),
                    Hit(pk="l2", chunk_id="l2", type="text", content="l high"),
                ],
            },
        }
        out = node(state)
        self.assertEqual(len(out["route_results"]["progressive"]), 1)
        self.assertEqual(len(out["route_results"]["local"]), 1)
        self.assertEqual(out["route_results"]["progressive"][0].pk, "p2")
        self.assertEqual(out["route_results"]["local"][0].pk, "l2")


if __name__ == "__main__":
    unittest.main()
