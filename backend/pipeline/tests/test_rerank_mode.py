"""Tests for rerank_mode routing field."""

from __future__ import annotations

import unittest

from pipeline.models import RouteDecision
from pipeline.clients.query_format import synthesize_rerank_query
from pipeline.routing.decision_builder import build_from_plan_args


class TestRerankMode(unittest.TestCase):
    def test_synthesize_default_uses_user_query(self):
        d = RouteDecision(
            routes=["progressive"],
            rewrites={"progressive": "钒电池 VRFB cycle life"},
        )
        q = synthesize_rerank_query(d, "钒电池的最高循环寿命是多少？")
        self.assertEqual(q, "钒电池的最高循环寿命是多少？")

    def test_synthesize_rerank_mode_true_uses_rewrite(self):
        d = RouteDecision(
            routes=["summary"],
            rewrites={"summary": "固态电池 solid-state battery ASSB"},
            rerank_mode=True,
        )
        q = synthesize_rerank_query(d, "这方面有什么研究")
        self.assertEqual(q, "固态电池 solid-state battery ASSB")

    def test_fc_plan_parses_rerank_mode(self):
        decision = build_from_plan_args(
            {
                "paths": [{"t": "summary", "kw": ["钒电池", "VRFB"]}],
                "rerank_mode": True,
            },
            query="有没有相关文献",
        )
        self.assertTrue(decision.rerank_mode)
        self.assertEqual(decision.rewrites.get("summary"), "钒电池 VRFB")

    def test_fc_plan_omits_rerank_mode_by_default(self):
        decision = build_from_plan_args(
            {"paths": [{"t": "progressive", "kw": ["MoS2", "lattice"]}]},
            query="MoS2 晶格常数是多少",
        )
        self.assertIsNone(decision.rerank_mode)

    def test_fc_plan_false_treated_as_omit(self):
        decision = build_from_plan_args(
            {
                "paths": [{"t": "summary", "kw": ["钒电池"]}],
                "rerank_mode": False,
            },
            query="有没有钒电池文献",
        )
        self.assertIsNone(decision.rerank_mode)


if __name__ == "__main__":
    unittest.main()
