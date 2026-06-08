"""RoutingCore FC guard / fallback behavior tests."""

from __future__ import annotations

import json
import unittest

from pipeline.models import RouteDecision
from pipeline.retrieval.agentic import QueryRouter
from pipeline.routing import ClarifyRequest, RoutingCore


class _FakeToolLLM:
    model = "fake-tool"

    def __init__(self, *, tool_name: str = "plan", arguments: dict | None = None, no_tool: bool = False):
        self.tool_name = tool_name
        self.arguments = arguments or {}
        self.no_tool = no_tool

    def chat_with_tools(self, **kwargs):
        if self.no_tool:
            return {"answer": "plain text", "raw": {"choices": [{"message": {"content": "plain text"}}]}}
        return {
            "raw": {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": self.tool_name,
                                        "arguments": json.dumps(self.arguments, ensure_ascii=False),
                                    }
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {},
            }
        }


def _registry(*entries: dict) -> list[dict]:
    return [
        {
            "doc_id": e["doc_id"],
            "doc_name": e.get("doc_name", e["doc_id"]),
            "pinned": e.get("pinned", False),
        }
        for e in entries
    ]


class TestRoutingCoreMetadataGuard(unittest.TestCase):
    def _core(self, llm, *, enable_ask: bool = True):
        router = QueryRouter(None)
        return RoutingCore(
            router_llm=llm,
            validate_fn=router._validate_decision,
            enable_ask=enable_ask,
            enable_reuse=False,
        )

    def test_ambiguous_metadata_fig_turns_into_clarify(self):
        core = self._core(_FakeToolLLM(arguments={"paths": [{"t": "metadata", "figs": ["3"]}]}))
        outcome = core.route(
            "图3说明了什么",
            doc_registry=_registry({"doc_id": "a"}, {"doc_id": "b"}),
        )
        self.assertIsInstance(outcome, ClarifyRequest)
        self.assertIn("哪篇文献", outcome.question)
        self.assertEqual(outcome.raw.get("source"), "metadata_anchor_guard")

    def test_unique_registry_auto_anchors_metadata(self):
        core = self._core(_FakeToolLLM(arguments={"paths": [{"t": "metadata", "figs": ["3"]}]}))
        outcome = core.route(
            "图3说明了什么",
            doc_registry=_registry({"doc_id": "only", "doc_name": "Only Paper"}),
        )
        self.assertIsInstance(outcome, RouteDecision)
        self.assertEqual(outcome.target_doc_ids, ["only"])
        self.assertEqual(outcome.target_docs, ["Only Paper"])


class TestRoutingCoreFallback(unittest.TestCase):
    def test_fc_parse_failure_uses_legacy_json_before_heuristic(self):
        legacy_decision = RouteDecision(routes=["summary"], rewrites={"summary": "耐候钢"})
        calls = []

        def legacy_route(query, history=None, doc_registry=None):
            calls.append((query, history, doc_registry))
            return legacy_decision

        core = RoutingCore(
            router_llm=_FakeToolLLM(no_tool=True),
            json_route_fn=legacy_route,
            heuristic_fn=lambda query, current_year: RouteDecision(routes=["progressive"]),
            enable_reuse=False,
        )
        outcome = core.route("有没有耐候钢文献", history=[{"role": "user", "content": "hi"}])
        self.assertIs(outcome, legacy_decision)
        self.assertEqual(len(calls), 1)
        meta = getattr(outcome, "_routing_meta")
        self.assertIn("fc_parse_failed", meta["fallback_chain"])
        self.assertIn("legacy_json", meta["fallback_chain"])
        self.assertNotIn("heuristic", meta["fallback_chain"])


class TestRoutingCoreP0P1Guards(unittest.TestCase):
    def _core(self, arguments: dict, *, enable_ask: bool = True):
        router = QueryRouter(None)
        return RoutingCore(
            router_llm=_FakeToolLLM(arguments=arguments),
            validate_fn=router._validate_decision,
            enable_ask=enable_ask,
            enable_reuse=False,
        )

    def test_inventory_summary_turns_into_clarify(self):
        core = self._core({"paths": [{"t": "summary", "kw": ["耐候钢"]}]})
        outcome = core.route("共有多少篇耐候钢相关文献")
        self.assertIsInstance(outcome, ClarifyRequest)
        self.assertEqual(outcome.raw.get("source"), "inventory_query_guard")

    def test_unanchored_references_turns_into_clarify(self):
        core = self._core({"paths": [{"t": "progressive", "kw": ["references"], "ctype": "references"}]})
        outcome = core.route(
            "参考文献有哪些",
            doc_registry=_registry({"doc_id": "a"}, {"doc_id": "b"}),
        )
        self.assertIsInstance(outcome, ClarifyRequest)
        self.assertIn("参考文献", outcome.question)

    def test_unique_references_auto_anchors_and_switches_to_local(self):
        core = self._core({"paths": [{"t": "progressive", "kw": ["references"], "ctype": "references"}]})
        outcome = core.route(
            "参考文献有哪些",
            doc_registry=_registry({"doc_id": "only", "doc_name": "Only Paper"}),
        )
        self.assertIsInstance(outcome, RouteDecision)
        self.assertEqual(outcome.routes, ["local"])
        self.assertEqual(outcome.target_doc_ids, ["only"])
        self.assertEqual(outcome.chunk_type, "references")


if __name__ == "__main__":
    unittest.main()
