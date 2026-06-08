"""结构化 chunk 全量召回 (references/image/table) 与 rerank 豁免。"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from pipeline.retrieval.agentic import (
    EnhancedMetadataRetriever,
    ProgressiveLocalRetriever,
    RouteDecision,
)
from pipeline.retrieval.langgraph_agent import _make_reranker_node
from pipeline.retrieval.progressive_config import ProgressiveRetrieveConfig
from pipeline.retrieval.retrievers import Hit
from pipeline.retrieval.structural_retrieval import (
    ExemptDecision,
    decision_requests_structural_full_recall,
    hit_exempt_decision,
    hit_exempt_from_rerank_filter,
    is_structural_chunk_type,
)


def _row(pk: str, *, ctype: str, content: str, doc_id: str = "d1") -> dict:
    return {
        "pk": pk,
        "chunk_id": f"cid-{pk}",
        "doc_id": doc_id,
        "doc_name": doc_id,
        "type": ctype,
        "section": "",
        "page_start": 0,
        "paragraph_index": 0,
        "publication_year": 2024,
        "content": content,
        "context": "",
        "related_assets": "",
    }


class TestStructuralHelpers(unittest.TestCase):
    def test_chunk_types(self):
        self.assertTrue(is_structural_chunk_type("references"))
        self.assertTrue(is_structural_chunk_type("image"))
        self.assertFalse(is_structural_chunk_type("text"))

    def test_decision_metadata_refs(self):
        d = RouteDecision(routes=["metadata"], fig_refs=["3"])
        self.assertTrue(decision_requests_structural_full_recall(d))

    def test_hit_exempt_metadata_route(self):
        hit = Hit(pk="1", type="text", content="Fig.3")
        # P0.2: metadata 路径无显式结构化引用 → topk_only (仍参与评分)
        self.assertTrue(hit_exempt_from_rerank_filter("metadata", hit))
        edec = hit_exempt_decision("metadata", hit)
        self.assertEqual(edec, ExemptDecision.topk_only())

    def test_hit_exempt_metadata_with_fig_refs_is_full(self):
        """P0.2: metadata + fig_refs → 完全豁免 (硬过滤高置信)。"""
        hit = Hit(pk="1", type="image", content="Fig.3 caption")
        decision = RouteDecision(routes=["metadata"], fig_refs=["3"])
        edec = hit_exempt_decision("metadata", hit, decision)
        self.assertEqual(edec, ExemptDecision.full())

    def test_hit_exempt_structural_type_with_matching_chunk_type(self):
        """P0.2: 路径明确请求 references → 该 chunk 完全豁免。"""
        hit = Hit(pk="1", type="references", content="[1] Author")
        decision = RouteDecision(chunk_type="references")
        self.assertTrue(hit_exempt_from_rerank_filter("progressive", hit, decision))
        edec = hit_exempt_decision("progressive", hit, decision)
        self.assertEqual(edec, ExemptDecision.full())

    def test_hit_no_longer_exempt_when_path_does_not_request(self):
        """P0.2 #5 修复: 混合路径里意外召回的 references 不再自动豁免, 必须参与评分。"""
        hit = Hit(pk="1", type="references", content="[1] Author")
        # 路径未指定 chunk_type → references chunk 必须参与 rerank
        self.assertFalse(hit_exempt_from_rerank_filter("progressive", hit, None))
        edec = hit_exempt_decision("progressive", hit)
        self.assertEqual(edec, ExemptDecision.none())

    def test_hit_not_exempt_normal_text(self):
        hit = Hit(pk="1", type="text", content="body")
        self.assertFalse(hit_exempt_from_rerank_filter("progressive", hit))


class TestMetadataFullRecall(unittest.TestCase):
    def test_label_match_returns_all_without_top_k_cut(self):
        rows = [
            _row("p1", ctype="image", content="Fig.1 overview"),
            _row("p2", ctype="image", content="Fig.2 detail"),
            _row("p3", ctype="image", content="Fig.3 results"),
        ]
        client = MagicMock()
        client.query.return_value = rows
        retriever = EnhancedMetadataRetriever(client, collection="lit")

        hits = retriever._retrieve_by_label_match(
            fig_refs=["1", "2", "3"],
            table_refs=[],
            top_k=2,
            max_candidates=50,
        )
        self.assertEqual(len(hits), 3)


class TestProgressiveStructuralL2(unittest.TestCase):
    def test_references_skips_hybrid(self):
        vec = MagicMock()
        vec.client.query.return_value = [
            _row("r1", ctype="references", content="[1] A"),
            _row("r2", ctype="references", content="[2] B"),
        ]
        vec.collection = "lit"
        hybrid = MagicMock()

        r = ProgressiveLocalRetriever(
            vec, hybrid, bm25_retriever=MagicMock(), config=ProgressiveRetrieveConfig(),
        )
        hybrid.bm25 = r.bm25
        hits = r._level2_drill_chunks(
            "refs",
            candidate_doc_ids=["d1"],
            top_k_chunks=1,
            per_query_k=5,
            per_retriever_k=5,
            time_filter=None,
            chunk_type="references",
        )
        hybrid.retrieve.assert_not_called()
        self.assertEqual(len(hits), 2)


class TestRerankerStructuralExempt(unittest.TestCase):
    def test_metadata_hits_kept_when_rankable_filtered(self):
        reranker = MagicMock()

        def rerank_side_effect(query, documents, top_k):
            from pipeline.clients.reranker import RerankResult

            return [
                RerankResult(index=i, score=0.9 - i * 0.1, content=documents[i])
                for i in range(len(documents))
            ]

        reranker.rerank.side_effect = rerank_side_effect
        node = _make_reranker_node(reranker, top_k=1, quality_k=1, quality_threshold=0.5)

        meta_hits = [
            Hit(pk="m1", chunk_id="m1", type="image", content="Fig.1"),
            Hit(pk="m2", chunk_id="m2", type="table", content="Table 1"),
        ]
        text_hits = [
            Hit(pk="t1", chunk_id="t1", type="text", content="low relevance"),
            Hit(pk="t2", chunk_id="t2", type="text", content="high relevance"),
        ]
        state = {
            "correlation_id": "test",
            "query": "Fig.1 and background",
            "decision": RouteDecision(routes=["metadata", "progressive"], fig_refs=["1"]),
            "route_results": {
                "metadata": meta_hits,
                "progressive": text_hits,
            },
        }
        out = node(state)
        filtered = out["route_results"]
        self.assertEqual(len(filtered["metadata"]), 2)
        self.assertEqual(len(filtered["progressive"]), 1)
        self.assertEqual(out["needs_retry"], False)


if __name__ == "__main__":
    unittest.main()
