"""P0-A/B 单元测试: QueryRouter._validate_decision 通过 doc_registry 解析 target_doc_ids,
并对代词回指做 focus doc 自动锚定."""

from __future__ import annotations

import unittest

from pipeline.retrieval.agentic import (
    QueryRouter,
    _match_registry_entry,
    _pick_focus_doc,
)


def _registry(*entries: dict) -> list:
    out = []
    for e in entries:
        out.append({
            "doc_id": e["doc_id"],
            "doc_name": e.get("doc_name", e["doc_id"]),
            "pinned": e.get("pinned", False),
        })
    return out


class TestMatchRegistryEntry(unittest.TestCase):
    def test_exact_doc_id(self):
        reg = _registry({"doc_id": "abc", "doc_name": "Foo Bar"})
        m = _match_registry_entry("abc", reg)
        self.assertIsNotNone(m)
        self.assertEqual(m["doc_id"], "abc")

    def test_exact_doc_name_case_insensitive(self):
        reg = _registry({"doc_id": "x1", "doc_name": "Offshore Platform Coatings: Review"})
        m = _match_registry_entry("offshore platform coatings: review", reg)
        self.assertIsNotNone(m)
        self.assertEqual(m["doc_id"], "x1")

    def test_substring_match_llm_overshoots(self):
        # LLM 给的标题 (英文翻译) 不等于 registry 中的中文名, 但 normalized 后是子串
        reg = _registry({
            "doc_id": "swfs01",
            "doc_name": "Progress in Offshore Platform Coatings: Review & Outlook",
        })
        # LLM 抄了完全相同的英文标题
        m = _match_registry_entry(
            "Progress in Offshore Platform Coatings", reg,
        )
        self.assertIsNotNone(m)
        self.assertEqual(m["doc_id"], "swfs01")

    def test_no_match_returns_none(self):
        reg = _registry({"doc_id": "x", "doc_name": "完全不相关的中文标题"})
        self.assertIsNone(_match_registry_entry("Offshore Coatings Review", reg))

    def test_empty_registry(self):
        self.assertIsNone(_match_registry_entry("anything", []))
        self.assertIsNone(_match_registry_entry("anything", None))

    def test_short_string_not_substring_matched(self):
        # 防止 "钢" 这种 1 字 query 命中所有名字含 "钢" 的 entry
        reg = _registry({"doc_id": "x", "doc_name": "海上平台防腐涂层综述"})
        self.assertIsNone(_match_registry_entry("钢", reg))


class TestPickFocusDoc(unittest.TestCase):
    def test_single_pinned_wins(self):
        reg = _registry(
            {"doc_id": "a"},
            {"doc_id": "b", "pinned": True},
            {"doc_id": "c"},
        )
        picked = _pick_focus_doc(reg)
        self.assertIsNotNone(picked)
        entry, reason = picked
        self.assertEqual(entry["doc_id"], "b")
        self.assertEqual(reason, "single-pinned")

    def test_single_entry_wins(self):
        reg = _registry({"doc_id": "only"})
        picked = _pick_focus_doc(reg)
        self.assertIsNotNone(picked)
        entry, reason = picked
        self.assertEqual(entry["doc_id"], "only")
        self.assertEqual(reason, "single-entry")

    def test_multiple_unpinned_no_focus(self):
        reg = _registry({"doc_id": "a"}, {"doc_id": "b"}, {"doc_id": "c"})
        self.assertIsNone(_pick_focus_doc(reg))

    def test_multiple_pinned_no_focus(self):
        reg = _registry(
            {"doc_id": "a", "pinned": True},
            {"doc_id": "b", "pinned": True},
        )
        self.assertIsNone(_pick_focus_doc(reg))

    def test_empty_registry(self):
        self.assertIsNone(_pick_focus_doc([]))
        self.assertIsNone(_pick_focus_doc(None))


class TestValidateDecisionRegistry(unittest.TestCase):
    """_validate_decision 的 P0-A/B 端到端单元测试."""

    def setUp(self):
        self.router = QueryRouter(llm=None)

    def test_doc_refs_resolves_to_doc_id(self):
        reg = _registry(
            {"doc_id": "paper-A", "doc_name": "A"},
            {"doc_id": "paper-B", "doc_name": "B"},
        )
        raw = {
            "routes": ["local"],
            "rewrites": {"local": ["catalyst"]},
            "filters": {"doc_refs": [2], "chunk_type": "references"},
        }
        decision = self.router._validate_decision(
            raw, "", "B 的参考文献有哪些", doc_registry=reg,
        )
        self.assertEqual(decision.target_doc_ids, ["paper-B"])
        self.assertIn("B", decision.target_docs)
        self.assertEqual(decision.chunk_type, "references")

    def test_target_docs_string_resolved_via_registry(self):
        """LLM 输出英文标题, 与 registry 是子串关系 → 应映射出 doc_id."""
        reg = _registry({
            "doc_id": "swfs-coating-2010",
            "doc_name": "Progress in Offshore Platform Coatings: Review & Outlook",
        })
        raw = {
            "routes": ["local"],
            "rewrites": {"local": ["offshore platform coatings references"]},
            "filters": {
                "target_docs": ["Progress in Offshore Platform Coatings"],
                "chunk_type": "references",
            },
        }
        decision = self.router._validate_decision(
            raw, "", "这篇文献的参考文献有哪些涂层研究", doc_registry=reg,
        )
        self.assertEqual(decision.target_doc_ids, ["swfs-coating-2010"])
        # canonical 名字被替换进 target_docs
        self.assertEqual(
            decision.target_docs,
            ["Progress in Offshore Platform Coatings: Review & Outlook"],
        )

    def test_pronoun_with_single_pinned_auto_anchors(self):
        """代词回指 + local + target_docs 空 + registry 有 1 pinned → 自动锚定."""
        reg = _registry(
            {"doc_id": "a"},
            {"doc_id": "focused", "doc_name": "Focused Paper", "pinned": True},
        )
        raw = {
            "routes": ["local"],
            "rewrites": {"local": ["参考文献"]},
            "filters": {"chunk_type": "references"},
        }
        decision = self.router._validate_decision(
            raw, "", "这篇文献的参考文献有哪些", doc_registry=reg,
        )
        self.assertEqual(decision.target_doc_ids, ["focused"])
        self.assertEqual(decision.target_docs, ["Focused Paper"])

    def test_pronoun_without_focus_doc_stays_empty(self):
        """多篇未 pin 的 registry → 不应乱锚定, 保持空."""
        reg = _registry(
            {"doc_id": "a"},
            {"doc_id": "b"},
            {"doc_id": "c"},
        )
        raw = {
            "routes": ["local"],
            "rewrites": {"local": ["参考文献"]},
            "filters": {"chunk_type": "references"},
        }
        decision = self.router._validate_decision(
            raw, "", "这篇文献的参考文献有哪些", doc_registry=reg,
        )
        self.assertEqual(decision.target_doc_ids, [])
        self.assertEqual(decision.target_docs, [])

    def test_unresolved_target_docs_preserved(self):
        """target_docs 字符串在 registry 中找不到 → 保留原值, 让 name 解析继续兜底."""
        reg = _registry({"doc_id": "X", "doc_name": "完全不同的标题"})
        raw = {
            "routes": ["local"],
            "filters": {"target_docs": ["Some Hallucinated Title"]},
        }
        decision = self.router._validate_decision(
            raw, "", "some query", doc_registry=reg,
        )
        self.assertEqual(decision.target_docs, ["Some Hallucinated Title"])
        self.assertEqual(decision.target_doc_ids, [])

    def test_progressive_pronoun_no_auto_anchor(self):
        """progressive 非结构化 ctype 时不自动锚定 (避免误锁文献)."""
        reg = _registry(
            {"doc_id": "only", "doc_name": "Only Paper", "pinned": True},
        )
        raw = {
            "routes": ["progressive"],
            "rewrites": {"progressive": ["topic"]},
        }
        decision = self.router._validate_decision(
            raw, "", "这篇文献里讲了什么", doc_registry=reg,
        )
        self.assertEqual(decision.target_doc_ids, [])

    def test_metadata_pronoun_single_pinned_auto_anchors(self):
        reg = _registry(
            {"doc_id": "a"},
            {"doc_id": "focused", "doc_name": "Focused", "pinned": True},
        )
        raw = {
            "routes": ["metadata"],
            "filters": {"fig_refs": ["3"]},
        }
        decision = self.router._validate_decision(
            raw, "", "这篇的图3", doc_registry=reg,
        )
        self.assertEqual(decision.target_doc_ids, ["focused"])

    def test_progressive_references_pronoun_single_pinned_auto_anchors(self):
        reg = _registry(
            {"doc_id": "a"},
            {"doc_id": "focused", "doc_name": "Focused", "pinned": True},
        )
        raw = {
            "routes": ["progressive"],
            "rewrites": {"progressive": ["references"]},
            "filters": {"chunk_type": "references"},
        }
        decision = self.router._validate_decision(
            raw, "", "它的参考文献有哪些", doc_registry=reg,
        )
        self.assertEqual(decision.target_doc_ids, ["focused"])
        self.assertEqual(decision.chunk_type, "references")

    def test_metadata_pronoun_multi_doc_no_auto_anchor(self):
        reg = _registry({"doc_id": "a"}, {"doc_id": "b"}, {"doc_id": "c"})
        raw = {"routes": ["metadata"], "filters": {"fig_refs": ["3"]}}
        decision = self.router._validate_decision(
            raw, "", "这篇的图3", doc_registry=reg,
        )
        self.assertEqual(decision.target_doc_ids, [])

    def test_no_registry_no_resolution(self):
        raw = {
            "routes": ["local"],
            "filters": {"target_docs": ["Paper A"]},
        }
        decision = self.router._validate_decision(
            raw, "", "Paper A 的方法", doc_registry=None,
        )
        self.assertEqual(decision.target_docs, ["Paper A"])
        self.assertEqual(decision.target_doc_ids, [])


if __name__ == "__main__":
    unittest.main()
