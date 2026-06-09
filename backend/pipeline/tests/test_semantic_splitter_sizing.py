"""semantic_splitter: token-aware sizing (引入2) + 句子边界 overlap (引入3)。

不依赖真实 embedding: 用 FakeEmbedder 返回可控向量, 验证尺寸度量与 overlap 行为。
"""

from __future__ import annotations

import unittest

from pipeline.processors.semantic_splitter import (
    SemanticChunk,
    _apply_overlap,
    estimate_tokens,
    semantic_split,
)


class _FakeEmbedder:
    """对每个 buffered 句子返回一个固定维度向量; 让相邻距离=0 (无断点),
    这样切分完全由 max_chars/target_chars (尺寸度量) 驱动, 便于断言。"""

    def embed_all(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class TestEstimateTokens(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(estimate_tokens(""), 0)

    def test_cjk_one_token_per_char(self):
        # 10 个汉字 -> ~10 token
        self.assertEqual(estimate_tokens("中" * 10), 10)

    def test_latin_quarter(self):
        # 纯 ASCII 约 1 token / 4 字符
        self.assertEqual(estimate_tokens("a" * 40), 10)

    def test_mixed_cjk_dominates_char_count(self):
        text = "测试abcd"  # 2 CJK + 4 latin -> 2 + 1 = 3 token, 但 6 字符
        self.assertEqual(estimate_tokens(text), 3)
        self.assertLess(estimate_tokens(text), len(text))


class TestTokenAwareSizing(unittest.TestCase):
    def test_token_unit_yields_fewer_chunks_than_char_for_latin(self):
        # 拉丁文: 4 字符 ~ 1 token, 字符数 >> token 数。相同阈值数字下,
        # char 模式更快触顶 max -> 切更多块; token 模式块更少 -> 更一致。
        text = ". ".join(["this is a plain english sentence for testing"
                          for _ in range(40)])

        char_chunks = semantic_split(
            text, _FakeEmbedder(),
            target_chars=200, max_chars=300, min_chars=50,
            breakpoint_percentile=85,
        )
        token_chunks = semantic_split(
            text, _FakeEmbedder(),
            target_chars=200, max_chars=300, min_chars=50,
            breakpoint_percentile=85,
            length_fn=estimate_tokens,
        )
        self.assertGreater(len(char_chunks), len(token_chunks))

    def test_token_mode_respects_max_in_tokens(self):
        text = "。".join(["中文句子内容样例文本" for _ in range(30)])
        chunks = semantic_split(
            text, _FakeEmbedder(),
            target_chars=40, max_chars=60, min_chars=10,
            breakpoint_percentile=85,
            length_fn=estimate_tokens,
        )
        self.assertGreater(len(chunks), 1)
        for c in chunks[:-1]:
            # 每块 token 数不应远超 max (允许最后一句跨过阈值的粒度)
            self.assertLessEqual(estimate_tokens(c.text), 60 + 20)


class TestSentenceBoundaryOverlap(unittest.TestCase):
    def test_overlap_prepends_whole_sentences(self):
        sentences = ["AAAA.", "BBBB.", "CCCC.", "DDDD."]
        chunks = [
            SemanticChunk(text="AAAA. BBBB.", sentence_indices=[0, 1]),
            SemanticChunk(text="CCCC. DDDD.", sentence_indices=[2, 3]),
        ]
        out = _apply_overlap(chunks, sentences, overlap_chars=5, max_chars=1000)
        self.assertEqual(len(out), 2)
        # 第二块应被前置上一块尾句 "BBBB."
        self.assertIn("BBBB.", out[1].text)
        self.assertTrue(out[1].text.startswith("BBBB."))
        # overlap 文本是完整句子, 不在句中截断
        self.assertNotIn("BBB ", out[1].text.replace("BBBB.", ""))

    def test_overlap_skipped_when_exceeds_max(self):
        sentences = ["AAAA.", "BBBBBBBBBB."]
        chunks = [
            SemanticChunk(text="AAAA.", sentence_indices=[0]),
            SemanticChunk(text="BBBBBBBBBB.", sentence_indices=[1]),
        ]
        # max 太小, 加 overlap 会超 -> 放弃 overlap, 保留原块
        out = _apply_overlap(chunks, sentences, overlap_chars=5, max_chars=11)
        self.assertEqual(out[1].text, "BBBBBBBBBB.")

    def test_first_chunk_unchanged(self):
        sentences = ["AAAA.", "BBBB."]
        chunks = [
            SemanticChunk(text="AAAA.", sentence_indices=[0]),
            SemanticChunk(text="BBBB.", sentence_indices=[1]),
        ]
        out = _apply_overlap(chunks, sentences, overlap_chars=5, max_chars=1000)
        self.assertEqual(out[0].text, "AAAA.")

    def test_no_overlap_default_off(self):
        # overlap_chars=0 -> semantic_split 不调用 overlap, 行为不变
        text = "。".join(["这是测试用的中文句子样例" for _ in range(20)])
        chunks = semantic_split(
            text, _FakeEmbedder(),
            target_chars=100, max_chars=150, min_chars=30,
            breakpoint_percentile=85,
        )
        # 相邻块之间没有重复前缀注入 (无 overlap)
        self.assertGreater(len(chunks), 1)


if __name__ == "__main__":
    unittest.main()
