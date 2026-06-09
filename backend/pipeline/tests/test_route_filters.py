"""route_filters 单元测试 (#10)。"""

from __future__ import annotations

import unittest

from pipeline.models import RouteDecision
from pipeline.retrieval.route_filters import (
    chunk_type_for_route,
    level1_global_probe_chunk_type,
)


class TestRouteFilters(unittest.TestCase):
    def test_summary_ignores_chunk_type(self):
        d = RouteDecision(routes=["summary"], chunk_type="references")
        self.assertIsNone(chunk_type_for_route("summary", d))

    def test_local_references(self):
        d = RouteDecision(
            routes=["local"],
            chunk_type="references",
            target_docs=["Paper A"],
        )
        self.assertEqual(chunk_type_for_route("local", d), "references")

    def test_progressive_metadata_dual_path_no_image_on_progressive(self):
        d = RouteDecision(
            routes=["progressive", "metadata"],
            chunk_type="image",
            fig_refs=["3"],
        )
        self.assertIsNone(chunk_type_for_route("progressive", d))
        self.assertEqual(chunk_type_for_route("metadata", d), "image")

    def test_progressive_references_without_metadata(self):
        d = RouteDecision(routes=["progressive"], chunk_type="references")
        self.assertEqual(chunk_type_for_route("progressive", d), "references")

    def test_metadata_infers_table(self):
        d = RouteDecision(routes=["metadata"], table_refs=["2"])
        self.assertEqual(chunk_type_for_route("metadata", d), "table")

    def test_metadata_skips_references_ct(self):
        d = RouteDecision(routes=["metadata"], chunk_type="references")
        self.assertIsNone(chunk_type_for_route("metadata", d))

    def test_level1_probe_skips_references(self):
        self.assertIsNone(level1_global_probe_chunk_type("references"))
        self.assertIsNone(level1_global_probe_chunk_type("image"))


if __name__ == "__main__":
    unittest.main()
