"""SummaryRetriever 分池双路召回测试 (summary/title × vector/bm25)。"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from pipeline.retrieval.agentic import (
    SUMMARY_CHUNK_FILTER,
    TITLE_CHUNK_FILTER,
    SummaryRetriever,
)
from pipeline.retrieval.retrievers import Hit


def _hit(
    pk: str,
    doc_id: str,
    content: str,
    *,
    hit_type: str = "summary",
    score: float = 1.0,
) -> Hit:
    return Hit(
        pk=pk,
        chunk_id=pk,
        doc_id=doc_id,
        type=hit_type,
        content=content,
        score=score,
    )


class TestSummaryRetrieverDualPath(unittest.TestCase):
    def _make_retriever(self, vec_hits_by_filter, bm25_hits_by_filter):
        vec = MagicMock()
        bm25 = MagicMock()

        def vec_side(query, top_k, filter_expr=None, **kwargs):
            fe = filter_expr or ""
            if SUMMARY_CHUNK_FILTER in fe:
                return list(vec_hits_by_filter.get("summary", []))
            if TITLE_CHUNK_FILTER in fe:
                return list(vec_hits_by_filter.get("title", []))
            return []

        def bm25_side(query, top_k, filter_expr=None, **kwargs):
            fe = filter_expr or ""
            if SUMMARY_CHUNK_FILTER in fe:
                return list(bm25_hits_by_filter.get("summary", []))
            if TITLE_CHUNK_FILTER in fe:
                return list(bm25_hits_by_filter.get("title", []))
            return []

        vec.retrieve.side_effect = vec_side
        bm25.retrieve.side_effect = bm25_side
        return SummaryRetriever(vec, bm25_retriever=bm25)

    def test_summary_and_title_pools_both_recalled(self):
        r = self._make_retriever(
            {
                "summary": [_hit("s1", "doc-a", "摘要正文", hit_type="summary", score=0.9)],
                "title": [_hit("t1", "doc-b", "文献标题", hit_type="title", score=0.8)],
            },
            {
                "summary": [_hit("s2", "doc-c", "另一篇摘要", hit_type="summary", score=0.7)],
                "title": [],
            },
        )
        hits = r.retrieve("蒙乃尔 堆焊", top_k_docs=5, per_query_k=3)
        pks = {h.pk for h in hits}
        types = {h.type for h in hits}
        self.assertIn("s1", pks)
        self.assertIn("t1", pks)
        self.assertIn("s2", pks)
        self.assertIn("summary", types)
        self.assertIn("title", types)

    def test_bm25_supplements_vec_in_same_pool(self):
        r = self._make_retriever(
            {"summary": [_hit("v1", "doc-a", "vec summary", score=0.5)]},
            {"summary": [_hit("b1", "doc-b", "bm25 exact", score=0.95)]},
        )
        hits = r.retrieve("LiNiCoMnO2", top_k_docs=5, per_query_k=5)
        doc_ids = {h.doc_id for h in hits}
        self.assertIn("doc-a", doc_ids)
        self.assertIn("doc-b", doc_ids)
        bm25_hit = next(h for h in hits if h.doc_id == "doc-b")
        self.assertIn("bm25", bm25_hit.sources)
        self.assertIn("vector", next(h for h in hits if h.doc_id == "doc-a").sources)

    def test_vec_only_when_no_bm25(self):
        vec = MagicMock()
        vec.retrieve.return_value = [
            _hit("v1", "doc-a", "topic", hit_type="summary", score=0.6),
        ]
        r = SummaryRetriever(vec)
        hits = r.retrieve("topic", top_k_docs=3)
        self.assertEqual(len(hits), 1)
        self.assertIn("vector", hits[0].sources)
        self.assertEqual(hits[0].type, "summary")
        self.assertEqual(vec.retrieve.call_count, 2)


if __name__ == "__main__":
    unittest.main()
