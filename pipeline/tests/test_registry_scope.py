"""多轮文献范围 guard: 全量/指定/禁止下探."""

from __future__ import annotations

import unittest

from pipeline.models import RouteDecision
from pipeline.routing.core import _apply_route_guards
from pipeline.routing.decision_builder import build_from_reuse_args
from pipeline.routing.registry_scope import (
    apply_registry_scope_to_reuse,
    extract_doc_ref_indices_from_query,
    query_has_batch_registry_scope,
    query_has_explicit_registry_scope,
)
from pipeline.retrieval.agentic import QueryRouter


def _registry(*entries: dict) -> list[dict]:
    return [
        {
            "doc_id": e["doc_id"],
            "doc_name": e.get("doc_name", e["doc_id"]),
            "pinned": e.get("pinned", False),
        }
        for e in entries
    ]


REGISTRY_3 = _registry(
    {"doc_id": "a", "doc_name": "Paper A"},
    {"doc_id": "b", "doc_name": "Paper B"},
    {"doc_id": "c", "doc_name": "Paper C"},
)


class TestRegistryScopeDetection(unittest.TestCase):
    def test_batch_scope_markers(self):
        self.assertTrue(query_has_batch_registry_scope("它们用了什么关键技术"))
        self.assertTrue(query_has_batch_registry_scope("上面这几篇的主要结论"))
        self.assertFalse(query_has_batch_registry_scope("这篇用了什么方法"))

    def test_explicit_vs_none(self):
        self.assertTrue(query_has_explicit_registry_scope("第6篇的关键技术"))
        self.assertFalse(query_has_explicit_registry_scope("锌铝镁镀层的主要优势有哪些"))

    def test_extract_doc_indices(self):
        self.assertEqual(extract_doc_ref_indices_from_query("看一下第6篇"), [6])
        self.assertEqual(extract_doc_ref_indices_from_query("对比第1篇和第3篇"), [1, 3])


class TestRegistryScopeGuard(unittest.TestCase):
    def test_batch_scope_locks_all_docs(self):
        decision = RouteDecision(
            routes=["progressive"],
            rewrites={"progressive": "关键技术"},
        )
        out = _apply_route_guards(
            decision,
            query="它们分别用了什么关键技术",
            doc_registry=REGISTRY_3,
            enable_ask=False,
        )
        self.assertEqual(out.routes, ["local"])
        self.assertEqual(out.target_doc_ids, ["a", "b", "c"])

    def test_specific_ref_locks_one_doc(self):
        decision = RouteDecision(
            routes=["progressive"],
            rewrites={"progressive": "关键技术"},
        )
        out = _apply_route_guards(
            decision,
            query="看一下第2篇用到了什么关键技术",
            doc_registry=REGISTRY_3,
            enable_ask=False,
        )
        self.assertEqual(out.routes, ["local"])
        self.assertEqual(out.target_doc_ids, ["b"])
        self.assertEqual(out.target_docs, ["Paper B"])

    def test_no_explicit_scope_clears_erroneous_lock(self):
        decision = RouteDecision(
            routes=["progressive"],
            rewrites={"progressive": "成本效益"},
            target_doc_ids=["b"],
            target_docs=["Paper B"],
        )
        out = _apply_route_guards(
            decision,
            query="钒电池的成本效益怎么样",
            doc_registry=REGISTRY_3,
            enable_ask=False,
        )
        self.assertEqual(out.routes, ["progressive"])
        self.assertEqual(out.target_doc_ids, [])
        self.assertEqual(out.target_docs, [])

    def test_singular_pronoun_picks_focus_doc(self):
        reg = _registry(
            {"doc_id": "a"},
            {"doc_id": "b", "pinned": True, "doc_name": "Focused"},
        )
        decision = RouteDecision(
            routes=["local"],
            rewrites={"local": "参考文献"},
            chunk_type="references",
        )
        out = _apply_route_guards(
            decision,
            query="这篇文献的参考文献有哪些",
            doc_registry=reg,
            enable_ask=False,
        )
        self.assertEqual(out.target_doc_ids, ["b"])


class TestReuseRegistryScope(unittest.TestCase):
    def test_reuse_without_explicit_scope_no_auto_lock(self):
        req = build_from_reuse_args(
            {"mode": "drilldown", "op": "summarize"},
            doc_registry=REGISTRY_3,
            query="锌铝镁镀层的主要优势",
        )
        self.assertEqual(req.target_doc_ids, [])

    def test_reuse_batch_scope_locks_all(self):
        req = build_from_reuse_args(
            {"mode": "drilldown", "op": "expand"},
            doc_registry=REGISTRY_3,
            query="上面这几篇分别讲了什么",
        )
        req = apply_registry_scope_to_reuse(
            req, query="上面这几篇分别讲了什么", doc_registry=REGISTRY_3,
        )
        self.assertEqual(req.doc_refs, [1, 2, 3])
        self.assertEqual(req.target_doc_ids, ["a", "b", "c"])

    def test_reuse_specific_ref_with_explicit_query(self):
        req = build_from_reuse_args(
            {"mode": "drilldown", "op": "summarize", "refs": [6]},
            doc_registry=REGISTRY_3 + [{"doc_id": f"d{i}"} for i in range(4, 7)],
            query="第6篇讲了什么",
        )
        req = apply_registry_scope_to_reuse(
            req,
            query="第6篇讲了什么",
            doc_registry=REGISTRY_3 + [{"doc_id": f"d{i}"} for i in range(4, 7)],
        )
        self.assertEqual(req.target_doc_ids, ["d6"])


class TestValidateDecisionWithGuard(unittest.TestCase):
    """端到端: validate 自动锚定 + guard 清除/保留."""

    def setUp(self):
        self.router = QueryRouter(llm=None)

    def test_progressive_topic_related_not_scoped_after_guard(self):
        raw = {
            "routes": ["progressive"],
            "rewrites": {"progressive": ["钒电池", "成本效益"]},
        }
        decision = self.router._validate_decision(
            raw, "", "钒电池的成本效益怎么样", doc_registry=REGISTRY_3,
        )
        guarded = _apply_route_guards(
            decision,
            query="钒电池的成本效益怎么样",
            doc_registry=REGISTRY_3,
            enable_ask=False,
        )
        self.assertEqual(guarded.target_doc_ids, [])


if __name__ == "__main__":
    unittest.main()
