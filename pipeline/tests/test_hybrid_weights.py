"""Tests for router retrieve_bias → hybrid weights (#14)."""

from __future__ import annotations

import unittest

from pipeline.models import RouteDecision
from pipeline.retrieval.hybrid_config import hybrid_config_from_dict
from pipeline.retrieval.hybrid_weights import (
    STAGE_LOCAL_L2,
    STAGE_PROGRESSIVE_L1,
    STAGE_PROGRESSIVE_L2,
    STAGE_SIMPLE,
    infer_hybrid_weights,
    infer_retrieve_bias_heuristic,
    normalize_retrieve_bias,
)


class TestRetrieveBiasNormalization(unittest.TestCase):
    def test_valid_values(self):
        self.assertEqual(normalize_retrieve_bias("semantic"), "semantic")
        self.assertEqual(normalize_retrieve_bias("ENTITY_HEAVY"), "entity_heavy")

    def test_invalid_returns_none(self):
        self.assertIsNone(normalize_retrieve_bias("unknown"))


class TestRouterBiasWeights(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = hybrid_config_from_dict({})

    def test_router_semantic_l1(self):
        r = infer_hybrid_weights(
            STAGE_PROGRESSIVE_L1,
            "任意",
            retrieve_bias="semantic",
            config=self.cfg,
        )
        self.assertEqual(r.source, "router")
        self.assertEqual(r.retrieve_bias, "semantic")
        self.assertGreater(r.dense, r.bm25)

    def test_router_entity_heavy_local_l2(self):
        r = infer_hybrid_weights(
            STAGE_LOCAL_L2,
            "任意",
            retrieve_bias="entity_heavy",
            config=self.cfg,
        )
        self.assertGreater(r.bm25, r.dense)
        self.assertAlmostEqual(r.dense, 0.30, places=2)

    def test_missing_bias_uses_heuristic(self):
        r = infer_hybrid_weights(
            STAGE_PROGRESSIVE_L2,
            "LiNiCoMnO2 性能",
            retrieve_bias=None,
            config=self.cfg,
        )
        self.assertEqual(r.source, "heuristic")
        self.assertEqual(r.retrieve_bias, "entity_heavy")

    def test_balanced_profile(self):
        r = infer_hybrid_weights(
            STAGE_PROGRESSIVE_L2,
            "耐候钢 文献",
            retrieve_bias="balanced",
            config=self.cfg,
        )
        self.assertAlmostEqual(r.dense, 0.55, places=2)
        self.assertAlmostEqual(r.bm25, 0.45, places=2)

    def test_static_mode(self):
        cfg = hybrid_config_from_dict({"mode": "static", "static_dense_weight": 0.8})
        r = infer_hybrid_weights(
            STAGE_SIMPLE, "q", retrieve_bias="entity_heavy", config=cfg,
        )
        self.assertEqual(r.source, "static")
        self.assertAlmostEqual(r.dense + r.bm25, 1.0, places=4)
        self.assertGreater(r.dense, r.bm25)

    def test_route_decision_wiring(self):
        d = RouteDecision(routes=["progressive"], retrieve_bias="keyword")
        r = infer_hybrid_weights(
            STAGE_PROGRESSIVE_L2, "test", retrieve_bias=d.retrieve_bias, config=self.cfg,
        )
        self.assertEqual(r.retrieve_bias, "keyword")
        self.assertEqual(r.source, "router")


class TestHeuristicFallbackBias(unittest.TestCase):
    def test_references_chunk_type(self):
        self.assertEqual(
            infer_retrieve_bias_heuristic("query", chunk_type="references"),
            "entity_heavy",
        )

    def test_semantic_query(self):
        self.assertEqual(
            infer_retrieve_bias_heuristic("腐蚀机理是什么"),
            "semantic",
        )


class TestBiasWeightMatrix(unittest.TestCase):
    def test_print_matrix(self) -> None:
        cfg = hybrid_config_from_dict({})
        biases = ["balanced", "semantic", "entity_heavy", "keyword"]
        stages = [STAGE_PROGRESSIVE_L1, STAGE_PROGRESSIVE_L2, STAGE_LOCAL_L2, STAGE_SIMPLE]
        lines = ["\n=== Router retrieve_bias → weights ==="]
        for bias in biases:
            lines.append(f"\nbias={bias} (router)")
            for stage in stages:
                r = infer_hybrid_weights(stage, "q", retrieve_bias=bias, config=cfg)
                lines.append(f"  {stage:24s} dense={r.dense:.2f} bm25={r.bm25:.2f}")
        print("\n".join(lines))


if __name__ == "__main__":
    unittest.main()
