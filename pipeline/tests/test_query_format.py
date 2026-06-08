"""Tests for retrieval query formatting (Qwen3 instruct / rerank alignment)."""

from __future__ import annotations

import unittest

from pipeline.models import RouteDecision
from pipeline.clients.query_format import (
    EMBED_STAGE_PASSAGE,
    EMBED_STAGE_SUMMARY,
    collect_prewarm_embed_texts,
    compose_rerank_document,
    format_qwen3_embed_query,
    instruct_kwargs_from_embedding_cfg,
    synthesize_rerank_query,
)
from pipeline.retrieval.retrievers import Hit


class TestQueryFormat(unittest.TestCase):
    def test_format_qwen3_embed_query_with_instruct(self):
        out = format_qwen3_embed_query(
            "钒电池 循环寿命", EMBED_STAGE_PASSAGE, enabled=True,
        )
        self.assertTrue(out.startswith("Instruct:"))
        self.assertIn("Query:钒电池 循环寿命", out)

    def test_format_disabled_returns_raw(self):
        self.assertEqual(
            format_qwen3_embed_query("hello", EMBED_STAGE_SUMMARY, enabled=False),
            "hello",
        )

    def test_prewarm_progressive_includes_two_stages(self):
        d = RouteDecision(
            routes=["progressive"],
            rewrites={"progressive": "MoS2 lattice constant"},
        )
        texts = collect_prewarm_embed_texts([d], "fallback")
        self.assertEqual(len(texts), 2)
        self.assertTrue(all(t.startswith("Instruct:") for t in texts))

    def test_compose_rerank_document_includes_section_and_context(self):
        hit = Hit(
            section="Results",
            content="Main paragraph.",
            context="Anchor sentence.",
            type="text",
        )
        doc = compose_rerank_document(hit)
        self.assertIn("[Section] Results", doc)
        self.assertIn("Main paragraph.", doc)

    def test_compose_rerank_document_equation_includes_context(self):
        hit = Hit(content="$$ E = mc^2 $$", context="Energy relation.", type="equation")
        doc = compose_rerank_document(hit)
        self.assertIn("Energy relation.", doc)

    def test_synthesize_rerank_query_prefers_user_query(self):
        d = RouteDecision(
            routes=["progressive"],
            rewrites={"progressive": "钒电池 循环寿命"},
        )
        q = synthesize_rerank_query(d, "钒电池的最高循环寿命是多少？")
        self.assertEqual(q, "钒电池的最高循环寿命是多少？")

    def test_synthesize_rerank_vague_uses_rewrite(self):
        d = RouteDecision(
            routes=["summary"],
            rewrites={"summary": "固态电池 solid-state battery ASSB"},
            rerank_mode=True,
        )
        q = synthesize_rerank_query(d, "这方面有什么研究")
        self.assertEqual(q, "固态电池 solid-state battery ASSB")

    def test_synthesize_rerank_without_mode_keeps_user_query(self):
        d = RouteDecision(
            routes=["summary"],
            rewrites={"summary": "固态电池 solid-state battery ASSB"},
        )
        q = synthesize_rerank_query(d, "这方面有什么研究")
        self.assertEqual(q, "这方面有什么研究")

    def test_synthesize_rerank_specific_keeps_user_query(self):
        d = RouteDecision(routes=["metadata"], fig_refs=["3"])
        q = synthesize_rerank_query(d, "图3说明了什么")
        self.assertIn("figure 3", q)

    def test_instruct_kwargs_from_config(self):
        cfg = {
            "query_instruct": {
                "enabled": True,
                "instructs": {"passage": "Custom passage task"},
            },
        }
        kw = instruct_kwargs_from_embedding_cfg(cfg)
        self.assertTrue(kw["embed_query_instruct_enabled"])
        self.assertEqual(kw["embed_query_instructs"]["passage"], "Custom passage task")


if __name__ == "__main__":
    unittest.main()
