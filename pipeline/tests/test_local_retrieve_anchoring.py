"""P0-B/C: retrieve_direct 通过 doc_id 短路 + _locate_docs_by_name BM25 单篇安全阀."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from pipeline.retrieval.agentic import ProgressiveLocalRetriever
from pipeline.retrieval.progressive_config import ProgressiveRetrieveConfig
from pipeline.retrieval.retrievers import Hit


def _row(pk: str, *, ctype: str, doc_id: str = "d1", content: str = "x") -> dict:
    return {
        "pk": pk,
        "chunk_id": f"cid-{pk}",
        "doc_id": doc_id,
        "doc_name": f"{doc_id}-name",
        "type": ctype,
        "section": "",
        "page_start": 0,
        "paragraph_index": 0,
        "publication_year": 2024,
        "content": content,
        "context": "",
        "related_assets": "",
    }


def _summary_row(doc_id: str, doc_name: str) -> dict:
    return {"doc_id": doc_id, "doc_name": doc_name}


def _make_retriever(
    *, summary_query_rows: list | None = None,
    structural_query_rows: list | None = None,
    bm25_hits: list | None = None,
):
    vec = MagicMock()
    vec.collection = "lit"

    def _query_side_effect(collection_name, filter, output_fields, limit, **kwargs):
        # _locate_docs_by_id / _locate_exact / _locate_like 都查 summary;
        # _level2_structural_chunks 查 references / image / table.
        if 'type == "summary" or type == "title"' in (filter or ""):
            return list(summary_query_rows or [])
        return list(structural_query_rows or [])

    vec.client.query.side_effect = _query_side_effect

    bm25 = MagicMock()
    bm25.retrieve.return_value = list(bm25_hits or [])

    hybrid = MagicMock()
    hybrid.bm25 = bm25

    r = ProgressiveLocalRetriever(
        vec, hybrid, bm25_retriever=bm25,
        config=ProgressiveRetrieveConfig(),
    )
    return r, vec, bm25


class TestRetrieveDirectDocIdShortCircuit(unittest.TestCase):
    def test_doc_id_short_circuits_name_resolution(self):
        """target_doc_ids 提供时, retrieve_direct 不应再调 _locate_docs_by_name 的
        exact/like/bm25 三级降级 — 直接用 doc_id 锁定."""
        r, vec, bm25 = _make_retriever(
            summary_query_rows=[
                _summary_row("paper-X", "Canonical Title of X"),
            ],
            structural_query_rows=[
                _row("ref1", ctype="references", doc_id="paper-X"),
                _row("ref2", ctype="references", doc_id="paper-X"),
            ],
        )
        # spy: _locate_docs_by_name 不应被调用
        r._locate_docs_by_name = MagicMock(side_effect=AssertionError("不应调用"))

        result = r.retrieve_direct(
            query="参考文献",
            target_docs=["something-llm-hallucinated"],
            chunk_type="references",
            target_doc_ids=["paper-X"],
        )
        self.assertEqual(len(result.candidate_docs), 1)
        self.assertEqual(result.candidate_docs[0].doc_id, "paper-X")
        self.assertEqual(result.candidate_docs[0].doc_name, "Canonical Title of X")
        self.assertEqual(len(result.chunk_hits), 2)
        # BM25 兜底也不该被调用
        bm25.retrieve.assert_not_called()

    def test_doc_id_missing_summary_falls_back_to_id_as_name(self):
        """summary 不存在 (例如还没分块) → 仍用 doc_id 当 name 兜底, 不放弃锚点."""
        r, vec, bm25 = _make_retriever(
            summary_query_rows=[],  # 空
            structural_query_rows=[
                _row("ref1", ctype="references", doc_id="paper-Y"),
            ],
        )
        result = r.retrieve_direct(
            query="refs",
            target_docs=[],
            chunk_type="references",
            target_doc_ids=["paper-Y"],
        )
        self.assertEqual(len(result.candidate_docs), 1)
        self.assertEqual(result.candidate_docs[0].doc_id, "paper-Y")
        # name 字段也兜底成 doc_id
        self.assertEqual(result.candidate_docs[0].doc_name, "paper-Y")

    def test_no_doc_id_falls_back_to_name_resolution(self):
        """没传 target_doc_ids 时, 仍走原有 name 解析路径."""
        r, vec, bm25 = _make_retriever(
            summary_query_rows=[
                _summary_row("paper-Z", "Paper Z"),
            ],
            structural_query_rows=[
                _row("ref-z", ctype="references", doc_id="paper-Z"),
            ],
        )
        result = r.retrieve_direct(
            query="参考文献",
            target_docs=["Paper Z"],
            chunk_type="references",
            target_doc_ids=None,
        )
        self.assertEqual(len(result.candidate_docs), 1)
        self.assertEqual(result.candidate_docs[0].doc_id, "paper-Z")


class TestBm25SingleDocSafetyValve(unittest.TestCase):
    def test_single_target_bm25_capped_to_one(self):
        """len(target_docs)==1 且 exact/like 失败 → BM25 兜底最多返回 1 篇."""
        # 准备: exact 查询返回空, like 查询返回空, BM25 返回 5 个
        vec = MagicMock()
        vec.collection = "lit"
        vec.client.query.return_value = []  # exact + like 都查 summary, 都空

        bm25 = MagicMock()
        bm25.retrieve.return_value = [
            Hit(pk=f"p{i}", doc_id=f"doc-{i}", doc_name=f"Doc {i}", score=1.0 - i * 0.1)
            for i in range(5)
        ]
        hybrid = MagicMock()
        hybrid.bm25 = bm25

        r = ProgressiveLocalRetriever(
            vec, hybrid, bm25_retriever=bm25,
            config=ProgressiveRetrieveConfig(),
        )

        results = r._locate_docs_by_name(
            ["Hallucinated Title"], time_filter=None, bm25=bm25,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "doc-0")

    def test_multi_target_bm25_not_capped(self):
        """len(target_docs)>1 时不自动设上限, 保留原行为."""
        vec = MagicMock()
        vec.collection = "lit"
        vec.client.query.return_value = []

        bm25 = MagicMock()
        bm25.retrieve.return_value = [
            Hit(pk=f"p{i}", doc_id=f"doc-{i}", doc_name=f"Doc {i}", score=1.0)
            for i in range(3)
        ]
        hybrid = MagicMock()
        hybrid.bm25 = bm25

        r = ProgressiveLocalRetriever(
            vec, hybrid, bm25_retriever=bm25,
            config=ProgressiveRetrieveConfig(),
        )

        results = r._locate_docs_by_name(
            ["A", "B"], time_filter=None, bm25=bm25,
        )
        self.assertEqual(len(results), 3)  # 不截断

    def test_explicit_max_results_overrides(self):
        """显式传 bm25_max_results 优先于自动 1 篇."""
        vec = MagicMock()
        vec.collection = "lit"
        vec.client.query.return_value = []

        bm25 = MagicMock()
        bm25.retrieve.return_value = [
            Hit(pk=f"p{i}", doc_id=f"doc-{i}", doc_name=f"Doc {i}", score=1.0)
            for i in range(5)
        ]
        hybrid = MagicMock()
        hybrid.bm25 = bm25

        r = ProgressiveLocalRetriever(
            vec, hybrid, bm25_retriever=bm25,
            config=ProgressiveRetrieveConfig(),
        )

        results = r._locate_docs_by_name(
            ["X"], time_filter=None, bm25=bm25, bm25_max_results=3,
        )
        self.assertEqual(len(results), 3)

    def test_low_score_bm25_rejected(self):
        """BM25 标题兜底低分时拒绝自动锁文献, 避免 local 走偏."""
        vec = MagicMock()
        vec.collection = "lit"
        vec.client.query.return_value = []

        bm25 = MagicMock()
        bm25.retrieve.return_value = [
            Hit(pk="p0", doc_id="doc-low", doc_name="Weak Match", score=0.01),
        ]
        hybrid = MagicMock()
        hybrid.bm25 = bm25

        r = ProgressiveLocalRetriever(
            vec, hybrid, bm25_retriever=bm25,
            config=ProgressiveRetrieveConfig(),
        )

        results = r._locate_docs_by_name(
            ["Hallucinated Title"], time_filter=None, bm25=bm25,
        )
        self.assertEqual(results, [])


class TestRetrieveDirectIntegration(unittest.TestCase):
    """端到端: 模拟用户问"这篇文献的参考文献"场景."""

    def test_single_doc_references_no_bm25_spread(self):
        """target_doc_ids 锁定单篇 + ctype=references → 不应触发 BM25 兜底."""
        r, vec, bm25 = _make_retriever(
            summary_query_rows=[_summary_row("the-paper", "The Paper")],
            structural_query_rows=[
                _row("r1", ctype="references", doc_id="the-paper", content="[1] A"),
                _row("r2", ctype="references", doc_id="the-paper", content="[2] B"),
                _row("r3", ctype="references", doc_id="the-paper", content="[3] C"),
            ],
        )

        result = r.retrieve_direct(
            query="参考文献中有哪些关于海上平台防护涂料的研究",
            target_docs=["Progress in Offshore Platform Coatings: Review & Outlook"],
            chunk_type="references",
            target_doc_ids=["the-paper"],
        )
        # 只锁定 1 篇
        self.assertEqual(len(result.candidate_docs), 1)
        self.assertEqual(result.candidate_docs[0].doc_id, "the-paper")
        # 3 条 references 全部返回
        self.assertEqual(len(result.chunk_hits), 3)
        # BM25 兜底不应被触发
        bm25.retrieve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
