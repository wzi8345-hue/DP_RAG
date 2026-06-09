"""routing.limits 与 plan→multi 自动拆分 (#8)。"""

from __future__ import annotations

import unittest

from pipeline.routing.decision_builder import build_from_plan_args
from pipeline.routing.fc_schema import build_multi_tool, build_plan_tool
from pipeline.routing.limits import (
    RoutingLimits,
    compound_intent_hint,
    estimate_compound_intents,
    normalize_routes,
    paths_should_split_to_multi,
    split_plan_args_to_multi_args,
)


class TestRoutingLimits(unittest.TestCase):
    def test_estimate_compound_three_intents(self):
        q = "我想看图3的说明、第5页的数据表、以及参考文献中关于X的引用"
        self.assertGreaterEqual(estimate_compound_intents(q), 3)

    def test_compound_hint_injected(self):
        q = "图3说明以及第5页表格"
        hint = compound_intent_hint(q, limits=RoutingLimits(), enable_multi=True)
        self.assertIn("multi", hint)

    def test_complementary_dual_path_not_split(self):
        paths = [
            {"t": "progressive", "kw": ["图3", "说明"]},
            {"t": "metadata", "figs": ["3"]},
        ]
        self.assertFalse(paths_should_split_to_multi(paths))

    def test_conflicting_metadata_paths_split(self):
        paths = [
            {"t": "metadata", "figs": ["3"]},
            {"t": "metadata", "pages": [5], "ctype": "table"},
        ]
        self.assertTrue(paths_should_split_to_multi(paths))

    def test_refs_plus_metadata_split(self):
        paths = [
            {"t": "progressive", "kw": ["X"], "ctype": "references"},
            {"t": "metadata", "figs": ["3"]},
        ]
        self.assertTrue(paths_should_split_to_multi(paths))

    def test_split_plan_to_multi_args(self):
        args = {
            "paths": [
                {"t": "metadata", "figs": ["3"]},
                {"t": "metadata", "pages": [5]},
            ],
            "time": "2020-2026",
        }
        multi = split_plan_args_to_multi_args(args)
        self.assertEqual(len(multi["subs"]), 2)
        self.assertEqual(multi["subs"][0]["paths"][0]["figs"], ["3"])
        self.assertEqual(multi["subs"][1]["paths"][0]["pages"], [5])

    def test_normalize_routes_respects_max(self):
        raw = ["progressive", "local", "metadata"]
        out = normalize_routes(raw, max_paths=2)
        self.assertEqual(out, ["progressive", "local"])

    def test_schema_max_items_from_limits(self):
        lim = RoutingLimits(max_paths_per_sub=2, max_subqueries=4)
        plan = build_plan_tool(limits=lim)
        multi = build_multi_tool(limits=lim)
        self.assertEqual(
            plan["function"]["parameters"]["properties"]["paths"]["maxItems"],
            2,
        )
        self.assertEqual(
            multi["function"]["parameters"]["properties"]["subs"]["maxItems"],
            4,
        )

    def test_build_from_plan_trims_paths(self):
        args = {
            "paths": [
                {"t": "progressive", "kw": ["a"]},
                {"t": "local", "kw": ["b"]},
                {"t": "summary", "kw": ["c"]},
            ],
        }
        dec = build_from_plan_args(
            args,
            query="q",
            limits=RoutingLimits(max_paths_per_sub=2),
        )
        self.assertEqual(len(dec.routes), 2)


class TestMultiTruncationHint(unittest.TestCase):
    def test_truncated_subqueries_add_synth_hint(self):
        from pipeline.routing.decision_builder import build_from_multi_args
        from pipeline.routing.limits import RoutingLimits

        multi = build_from_multi_args(
            {
                "subs": [
                    {"paths": [{"t": "progressive", "kw": ["a"]}]},
                    {"paths": [{"t": "progressive", "kw": ["b"]}]},
                    {"paths": [{"t": "progressive", "kw": ["c"]}]},
                ],
                "synth": "compare",
            },
            query="q",
            limits=RoutingLimits(max_subqueries=2),
        )
        self.assertEqual(len(multi.subqueries), 2)
        self.assertIn("最多处理 2 个子意图", multi.synth_hint)


if __name__ == "__main__":
    unittest.main()
