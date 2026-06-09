"""锁定文献补充检索: reuse drilldown / local 下探共用."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from pipeline.models import RouteDecision
from pipeline.retrieval.agentic import AgenticContextBuilder, LocalRetrieveResult, ROUTE_LOCAL
from pipeline.retrieval.langgraph_agent import (
    _append_supplement_to_local_route,
    _build_reuse_context,
    _collect_known_chunk_ids,
    _extract_chunk_ids_from_context,
    _fetch_supplemental_locked_doc_chunks,
    _make_reuse_node,
    _should_supplement_for_local,
    _should_supplement_for_reuse,
)
from pipeline.retrieval.retrievers import Hit


def _hit(pk: str, *, chunk_id: str | None = None, doc_id: str = "doc_b") -> Hit:
    return Hit(
        pk=pk,
        chunk_id=chunk_id or pk,
        doc_id=doc_id,
        doc_name="Paper B",
        type="text",
        content=f"body {pk}",
        score=0.9,
    )


SAMPLE_CTX = (
    "# 用户问题\nq\n\n---\n\n"
    "[1] SUMMARY | chunk_id=summary_abc | doc=doc_b | page=1\n"
    "summary only"
)


class TestSupplementHelpers(unittest.TestCase):
    def test_extract_chunk_ids_from_context(self):
        ids = _extract_chunk_ids_from_context(SAMPLE_CTX)
        self.assertEqual(ids, {"summary_abc"})

    def test_collect_known_merges_route_results(self):
        route_results = {
            ROUTE_LOCAL: LocalRetrieveResult(chunk_hits=[_hit("c1"), _hit("c2")]),
        }
        known = _collect_known_chunk_ids(SAMPLE_CTX, route_results)
        self.assertEqual(known, {"summary_abc", "c1", "c2"})

    def test_should_supplement_for_reuse_drilldown(self):
        self.assertTrue(_should_supplement_for_reuse("drilldown", ["doc_b"]))
        self.assertFalse(_should_supplement_for_reuse("reformat", ["doc_b"]))
        self.assertFalse(_should_supplement_for_reuse("drilldown", []))

    def test_should_supplement_for_local(self):
        decision = RouteDecision(
            routes=[ROUTE_LOCAL],
            target_docs=["Paper B"],
            target_doc_ids=["doc_b"],
        )
        self.assertTrue(_should_supplement_for_local(decision))
        self.assertFalse(_should_supplement_for_local(RouteDecision(routes=[ROUTE_LOCAL])))


class TestFetchSupplementalChunks(unittest.TestCase):
    def test_excludes_known_chunk_ids(self):
        local_r = MagicMock()
        local_r.retrieve_direct.return_value = LocalRetrieveResult(
            candidate_docs=[],
            chunk_hits=[_hit("summary_abc"), _hit("text_new")],
        )
        hits = _fetch_supplemental_locked_doc_chunks(
            local_r,
            query="总结全文",
            target_doc_ids=["doc_b"],
            target_docs=["Paper B"],
            exclude_chunk_ids={"summary_abc"},
        )
        self.assertEqual([h.pk for h in hits], ["text_new"])
        self.assertIn("supplement", hits[0].sources)

    def test_append_supplement_to_local_route(self):
        route_results = {
            ROUTE_LOCAL: LocalRetrieveResult(chunk_hits=[_hit("c1")]),
        }
        merged = _append_supplement_to_local_route(route_results, [_hit("c2"), _hit("c1")])
        local = merged[ROUTE_LOCAL]
        self.assertEqual([h.pk for h in local.chunk_hits], ["c1", "c2"])


class TestReuseNodeSupplement(unittest.TestCase):
    def test_reuse_drilldown_includes_supplement_section(self):
        local_r = MagicMock()
        local_r.retrieve_direct.return_value = LocalRetrieveResult(
            chunk_hits=[_hit("body_1"), _hit("body_2")],
        )
        reuse_node = _make_reuse_node(local_r, AgenticContextBuilder())
        state = {
            "correlation_id": "t1",
            "query": "帮我总结一下这篇文献的全文吧",
            "reuse_request": {
                "mode": "drilldown",
                "op": "summarize the full text",
                "doc_refs": [2],
                "target_doc_ids": ["doc_b"],
                "target_docs": ["Paper B"],
            },
            "last_answer": "prev",
            "last_context": SAMPLE_CTX,
            "last_round_docs": [{"doc_id": "doc_b", "doc_name": "Paper B"}],
        }
        out = reuse_node(state)
        ctx = out["context"]
        self.assertIn("锁定文献补充检索", ctx)
        self.assertIn("body body_1", ctx)
        self.assertIn("补充检索", ctx.splitlines()[0])
        local_r.retrieve_direct.assert_called_once()

    def test_reuse_reformat_skips_supplement(self):
        local_r = MagicMock()
        reuse_node = _make_reuse_node(local_r, AgenticContextBuilder())
        state = {
            "correlation_id": "t2",
            "query": "翻译一下",
            "reuse_request": {
                "mode": "reformat",
                "op": "translate",
                "target_doc_ids": ["doc_b"],
                "target_docs": ["Paper B"],
            },
            "last_answer": "prev",
            "last_context": SAMPLE_CTX,
            "last_round_docs": [],
        }
        out = reuse_node(state)
        self.assertNotIn("锁定文献补充检索", out["context"])
        local_r.retrieve_direct.assert_not_called()

    def test_build_reuse_context_header_when_supplement_present(self):
        ctx = _build_reuse_context(
            "q",
            "drilldown",
            "summarize",
            last_answer="a",
            last_context=SAMPLE_CTX,
            supplement_context="# 锁定文献补充检索\n[1] TEXT | chunk_id=x\nbody",
        )
        self.assertIn("补充检索生成最终答复", ctx.splitlines()[0])


if __name__ == "__main__":
    unittest.main()
