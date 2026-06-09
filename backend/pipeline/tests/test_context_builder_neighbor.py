"""AgenticContextBuilder 渲染 neighbor 路由 (asset hydration 结果) 测试。"""

from __future__ import annotations

import unittest

from pipeline.models import RouteDecision
from pipeline.retrieval.agentic import (
    AgenticContextBuilder,
    LocalRetrieveResult,
    ROUTE_NEIGHBOR,
)
from pipeline.retrieval.retrievers import Hit


def _hit(chunk_id: str, ctype: str, content: str, doc_id: str = "doc1") -> Hit:
    return Hit(
        pk=chunk_id, chunk_id=chunk_id, doc_id=doc_id, doc_name="Paper",
        type=ctype, section="2 结果", page_start=2, content=content,
    )


class TestNeighborRendering(unittest.TestCase):
    def setUp(self):
        self.cb = AgenticContextBuilder()

    def test_neighbor_assets_rendered_into_context(self):
        decision = RouteDecision(routes=["progressive"], rewrites={"progressive": "石击坑深度"})
        results = {
            "progressive": LocalRetrieveResult(
                chunk_hits=[_hit("text_1", "text", "由表1可见石击坑深度数据。")],
            ),
            ROUTE_NEIGHBOR: [
                _hit("table_1", "table", "[Caption] 表1 石击坑深度\nGI275 55.1 ZM275 23.0"),
            ],
        }
        ctx = self.cb.build("石击坑深度是多少", decision, results)
        self.assertIn("table_1", ctx)
        self.assertIn("石击坑深度", ctx)
        self.assertIn("关联补充", ctx)

    def test_no_neighbor_key_unchanged(self):
        decision = RouteDecision(routes=["progressive"], rewrites={"progressive": "x"})
        results = {
            "progressive": LocalRetrieveResult(chunk_hits=[_hit("text_1", "text", "正文内容")]),
        }
        ctx = self.cb.build("q", decision, results)
        self.assertNotIn("关联补充", ctx)

    def test_neighbor_dedup_with_main_route(self):
        # 若 neighbor 块已在主路径渲染过, 不重复整块渲染
        decision = RouteDecision(routes=["progressive"], rewrites={"progressive": "x"})
        shared = _hit("text_1", "text", "共享块内容唯一标记ABC")
        results = {
            "progressive": LocalRetrieveResult(chunk_hits=[shared]),
            ROUTE_NEIGHBOR: [shared],
        }
        ctx = self.cb.build("q", decision, results)
        self.assertEqual(ctx.count("共享块内容唯一标记ABC"), 1)


if __name__ == "__main__":
    unittest.main()
