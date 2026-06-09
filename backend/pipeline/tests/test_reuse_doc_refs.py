"""Tests for reuse refs → doc_id locking and context filtering."""

from __future__ import annotations

import unittest

from pipeline.retrieval.langgraph_agent import (
    _build_reuse_context,
    _filter_reuse_context_by_doc_ids,
)
from pipeline.routing.decision_builder import build_from_reuse_args
from pipeline.routing.registry_scope import apply_registry_scope_to_reuse


REGISTRY = [
    {"doc_id": "doc_a", "doc_name": "Paper A"},
    {"doc_id": "doc_b", "doc_name": "Paper B", "pinned": True},
]


class TestReuseDocRefsParsing(unittest.TestCase):
    def test_reuse_refs_resolve_to_doc_ids(self):
        req = build_from_reuse_args(
            {"mode": "drilldown", "op": "summarize paper", "refs": [2]},
            doc_registry=REGISTRY,
        )
        self.assertEqual(req.doc_refs, [2])
        self.assertEqual(req.target_doc_ids, ["doc_b"])
        self.assertEqual(req.target_docs, ["Paper B"])

    def test_reuse_multi_refs(self):
        req = build_from_reuse_args(
            {"mode": "reformat", "op": "compare", "refs": [1, 2]},
            doc_registry=REGISTRY,
        )
        self.assertEqual(req.target_doc_ids, ["doc_a", "doc_b"])

    def test_reuse_auto_pick_single_pinned_when_refs_missing(self):
        req = build_from_reuse_args(
            {"mode": "drilldown", "op": "summarize"},
            doc_registry=REGISTRY,
            query="这篇论文主要讲了什么",
        )
        req = apply_registry_scope_to_reuse(
            req, query="这篇论文主要讲了什么", doc_registry=REGISTRY,
        )
        self.assertEqual(req.target_doc_ids, ["doc_b"])

    def test_chitchat_ignores_refs_resolution(self):
        req = build_from_reuse_args(
            {"mode": "chitchat", "op": "hi", "refs": [1]},
            doc_registry=REGISTRY,
        )
        self.assertEqual(req.doc_refs, [1])
        self.assertEqual(req.target_doc_ids, ["doc_a"])


class TestReuseContextFilter(unittest.TestCase):
    SAMPLE_CTX = (
        "# 用户问题\nq\n\n---\n\n"
        "# 来自 [local] 的内容\n\n"
        "[1] TEXT | chunk_id=c1 | doc=doc_a | page=1\n"
        "content A\n\n---\n\n"
        "[1] TEXT | chunk_id=c2 | doc=doc_b | page=2\n"
        "content B"
    )

    def test_filter_keeps_only_locked_doc_chunks(self):
        out = _filter_reuse_context_by_doc_ids(self.SAMPLE_CTX, ["doc_b"])
        self.assertIn("doc=doc_b", out)
        self.assertNotIn("doc=doc_a", out)
        self.assertIn("# 用户问题", out)

    def test_filter_fallback_when_no_match(self):
        out = _filter_reuse_context_by_doc_ids(self.SAMPLE_CTX, ["doc_missing"])
        self.assertEqual(out, self.SAMPLE_CTX)

    def test_build_reuse_context_uses_filtered_context(self):
        ctx = _build_reuse_context(
            "这篇论文讲了什么",
            "drilldown",
            "summarize main content",
            last_answer="prev answer",
            last_context=self.SAMPLE_CTX,
            target_doc_ids=["doc_b"],
            target_docs=["Paper B"],
            doc_refs=[2],
        )
        self.assertIn("# 锁定文献 (refs=[2])", ctx)
        self.assertIn("Paper B", ctx)
        self.assertIn("已按锁定文献裁剪", ctx)
        self.assertIn("doc=doc_b", ctx)
        self.assertNotIn("doc=doc_a", ctx)


if __name__ == "__main__":
    unittest.main()
