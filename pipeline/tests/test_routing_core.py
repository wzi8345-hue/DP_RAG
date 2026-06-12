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

    def test_misfired_references_without_explicit_intent_falls_back(self):
        # query 没有显式索取"参考文献", 即便 LLM 误填了 ctype=references,
        # 也不应弹"是哪篇文献"的澄清, 而是撤销过滤回退正文检索。
        core = self._core(
            {"paths": [{"t": "progressive", "kw": ["ASTM", "pitting"], "ctype": "references"}]}
        )
        outcome = core.route(
            "Which ASTM standard test method was used to characterize pitting corrosion susceptibility for the Nitinol ocular device",
            doc_registry=_registry({"doc_id": "a"}, {"doc_id": "b"}),
        )
        self.assertIsInstance(outcome, RouteDecision)
        self.assertNotEqual((outcome.chunk_type or "").lower(), "references")


class TestReflectZeroHitsFastPath(unittest.TestCase):
    """0 命中快速路径应保留原决策的锚点/过滤与 LLM 改写关键词 (HIGH-1 回归)。"""

    def _core(self):
        # reflect_llm 非 None 才会进入 reflect 主体; total_hits=0 时不会真的调用它
        return RoutingCore(router_llm=None, reflect_llm=_FakeToolLLM())

    def test_zero_hits_preserves_anchors_and_rewrites(self):
        last = RouteDecision(
            routes=["local"],
            rewrites={"local": "参考文献 references 列表"},
            chunk_type="references",
            target_docs=["Some Paper Title"],
            target_doc_ids=["doc_x"],
            time="2018-2024",
            fig_refs=["3"],
            page_refs=[5],
            entities=["Nitinol"],
            retrieve_bias="keyword",
        )
        verdict = self._core().reflect(
            query="这篇文献的参考文献有哪些",
            last_decision=last,
            results_summary="",
            total_hits=0,
            max_retries=1,
        )
        self.assertTrue(verdict.needs_retry)
        self.assertEqual(verdict.cause, "zero")
        dec = verdict.decision
        self.assertIsInstance(dec, RouteDecision)
        # 路径切到 progressive 做广搜
        self.assertEqual(dec.routes, ["progressive"])
        # 关键: 锚点/过滤被保留, 而非清空
        self.assertEqual(dec.chunk_type, "references")
        self.assertEqual(dec.target_docs, ["Some Paper Title"])
        self.assertEqual(dec.target_doc_ids, ["doc_x"])
        self.assertEqual(dec.time, "2018-2024")
        self.assertEqual(dec.fig_refs, ["3"])
        self.assertEqual(dec.page_refs, [5])
        self.assertEqual(dec.entities, ["Nitinol"])
        self.assertEqual(dec.retrieve_bias, "keyword")
        # 关键: 沿用 LLM 改写关键词, 而非回退用户原话
        self.assertEqual(dec.rewrites.get("progressive"), "参考文献 references 列表")

    def test_zero_hits_without_prior_decision_falls_back_to_query(self):
        verdict = self._core().reflect(
            query="某个问题",
            last_decision=None,
            results_summary="",
            total_hits=0,
            max_retries=1,
        )
        self.assertTrue(verdict.needs_retry)
        dec = verdict.decision
        self.assertEqual(dec.routes, ["progressive"])
        self.assertEqual(dec.rewrites.get("progressive"), "某个问题")


if __name__ == "__main__":
    unittest.main()
