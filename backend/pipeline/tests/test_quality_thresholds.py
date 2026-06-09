"""P1.1: RouteThresholds 阶梯查找单测."""

from __future__ import annotations

import unittest

from pipeline.retrieval.quality_thresholds import RouteThresholds


class TestRouteThresholdsLookup(unittest.TestCase):
    def setUp(self):
        self.matrix = RouteThresholds.from_dict({
            "default": 0.25,
            "by_type": {
                "text": 0.30,
                "image": 0.10,
            },
            "by_route": {
                "progressive": {
                    "l1": {"text": 0.18, "default": 0.12},
                    "l2": {"text": 0.32, "default": 0.18},
                    "default": 0.20,
                },
                "summary": {"default": 0.35},
                "local": {"text": 0.30, "default": 0.18},
            },
        })

    def test_route_stage_type_full_match(self):
        v, src = self.matrix.for_("progressive", "l1", "text")
        self.assertAlmostEqual(v, 0.18)
        self.assertIn("by_route[progressive][l1][text]", src)

    def test_route_stage_default(self):
        """progressive l1 没配 image → 回退到 l1.default."""
        v, src = self.matrix.for_("progressive", "l1", "image")
        self.assertAlmostEqual(v, 0.12)
        self.assertIn("by_route[progressive][l1].default", src)

    def test_route_default_when_no_stage(self):
        """没 stage 信息时回退到 route.default."""
        v, src = self.matrix.for_("progressive", None, "text")
        # progressive.default = 0.20 (flat, not dict)
        self.assertAlmostEqual(v, 0.20)
        self.assertIn("by_route[progressive].default", src)

    def test_route_default_flat_value(self):
        """summary.default 是 flat float, 任何 type 都用同值."""
        v, src = self.matrix.for_("summary", None, "table")
        self.assertAlmostEqual(v, 0.35)
        self.assertIn("by_route[summary].default", src)

    def test_route_type_no_stage(self):
        """local.text 直接配 → 取该值."""
        v, src = self.matrix.for_("local", None, "text")
        self.assertAlmostEqual(v, 0.30)
        self.assertIn("by_route[local][text]", src)

    def test_fallback_to_by_type_when_route_missing(self):
        """metadata 路径 + image 没配 → 回退 by_type[image]."""
        v, src = self.matrix.for_("metadata", None, "image")
        self.assertAlmostEqual(v, 0.10)
        self.assertIn("by_type[image]", src)

    def test_fallback_to_default(self):
        v, src = self.matrix.for_("unknown_route", "unknown_stage", "unknown_type")
        self.assertAlmostEqual(v, 0.25)
        self.assertEqual(src, "default")

    def test_case_insensitive(self):
        v, _ = self.matrix.for_("PROGRESSIVE", "L1", "TEXT")
        self.assertAlmostEqual(v, 0.18)

    def test_for_chunk_type_shortcut(self):
        """诊断层接口: 没有 route/stage, 只看 chunk_type."""
        self.assertAlmostEqual(self.matrix.for_chunk_type("text"), 0.30)
        self.assertAlmostEqual(self.matrix.for_chunk_type("image"), 0.10)
        self.assertAlmostEqual(self.matrix.for_chunk_type("unknown"), 0.25)


class TestRouteThresholdsLegacyBridge(unittest.TestCase):
    def test_legacy_default_only(self):
        m = RouteThresholds.from_dict(None, legacy_default=0.5)
        v, src = m.for_(None, None, "text")
        self.assertAlmostEqual(v, 0.5)
        self.assertEqual(src, "default")

    def test_legacy_by_type_fallback(self):
        m = RouteThresholds.from_dict(
            None,
            legacy_default=0.4,
            legacy_by_type={"image": 0.08, "table": 0.09},
        )
        self.assertAlmostEqual(m.for_chunk_type("image"), 0.08)
        self.assertAlmostEqual(m.for_chunk_type("text"), 0.4)

    def test_new_overrides_legacy(self):
        """新 schema 提供时, legacy 参数不再被读取."""
        m = RouteThresholds.from_dict(
            {"default": 0.2, "by_type": {"text": 0.25}},
            legacy_default=0.99,
            legacy_by_type={"text": 0.99},
        )
        self.assertAlmostEqual(m.for_chunk_type("text"), 0.25)
        self.assertAlmostEqual(m.for_chunk_type("unknown"), 0.2)


if __name__ == "__main__":
    unittest.main()
