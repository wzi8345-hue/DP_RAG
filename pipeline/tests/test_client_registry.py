"""Tests for ClientRegistry singleton + EmbeddingClient LRU + HybridRetriever parallel.

覆盖以下优化点:
1. ClientRegistry 按 (api_base, model, api_key, normalize) 复用 EmbeddingClient
2. ClientRegistry.get_llm 按 (api_base, model, api_key, extra_body) 复用 LLMClient
3. EmbeddingClient.embed(text) 命中 LRU 时不发 HTTP
4. EmbeddingClient.begin_request() 清空 LRU
5. HybridRetriever.retrieve vec/bm25 真正并行 (耗时近似 max 而非 sum)
6. VectorRetriever.retrieve_with_vector 跳过 embed
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock

from pipeline.clients.client_registry import ClientRegistry
from pipeline.clients.embedding import EmbeddingClient
from pipeline.retrieval.retrievers import (
    HybridRetriever,
    VectorRetriever,
    Hit,
    run_in_parallel,
)


class _RecordingEmbedding(EmbeddingClient):
    """覆盖 _post_embeddings, 不发真实 HTTP, 记录调用次数."""

    def __init__(self):
        super().__init__(api_base="http://x", model="m", api_key="k")
        self.http_calls = 0

    def _post_embeddings(self, texts):
        self.http_calls += 1
        return [[0.1, 0.2, 0.3] for _ in texts]


class TestClientRegistry(unittest.TestCase):
    def test_embedder_singleton_by_identity_key(self):
        reg = ClientRegistry()
        e1 = reg.get_embedder(
            api_base="http://a", model="m", api_key="k", normalize=False,
        )
        e2 = reg.get_embedder(
            api_base="http://a", model="m", api_key="k", normalize=False,
        )
        self.assertIs(e1, e2)

    def test_embedder_different_normalize_creates_new(self):
        reg = ClientRegistry()
        e1 = reg.get_embedder(
            api_base="http://a", model="m", api_key="k", normalize=False,
        )
        e2 = reg.get_embedder(
            api_base="http://a", model="m", api_key="k", normalize=True,
        )
        self.assertIsNot(e1, e2)

    def test_embedder_different_api_key_creates_new(self):
        reg = ClientRegistry()
        e1 = reg.get_embedder(api_base="http://a", model="m", api_key="k1")
        e2 = reg.get_embedder(api_base="http://a", model="m", api_key="k2")
        self.assertIsNot(e1, e2)

    def test_get_llm_requires_api_key(self):
        reg = ClientRegistry()
        with self.assertRaises(ValueError):
            reg.get_llm(api_base="http://a", model="m", api_key="")

    def test_stats(self):
        reg = ClientRegistry()
        reg.get_embedder(api_base="http://a", model="m", api_key="k")
        reg.get_embedder(api_base="http://a", model="m", api_key="k", normalize=True)
        self.assertEqual(reg.stats()["embedders"], 2)
        self.assertEqual(reg.stats()["llms"], 0)


class TestEmbeddingClientLRU(unittest.TestCase):
    def test_embed_hits_cache_on_repeat(self):
        m = _RecordingEmbedding()
        m.embed("q1")
        m.embed("q1")  # cache hit
        m.embed("q2")
        m.embed("q1")  # cache hit
        # 2 unique texts -> 2 HTTP calls only
        self.assertEqual(m.http_calls, 2)
        hits, miss, size = m.query_cache_stats()
        self.assertEqual(hits, 2)
        self.assertEqual(miss, 2)
        self.assertEqual(size, 2)

    def test_begin_request_resets_cache(self):
        m = _RecordingEmbedding()
        m.embed("q1")
        m.embed("q1")  # hit
        m.begin_request()
        m.embed("q1")  # miss again
        self.assertEqual(m.http_calls, 2)

    def test_empty_text_short_circuits(self):
        m = _RecordingEmbedding()
        self.assertEqual(m.embed(""), [])
        self.assertEqual(m.http_calls, 0)

    def test_disabled_cache_size_zero(self):
        m = _RecordingEmbedding()
        m._query_cache_size = 0  # 关闭 LRU
        m.embed("q1")
        m.embed("q1")
        # 关闭后每次都打 HTTP
        self.assertEqual(m.http_calls, 2)


class TestHybridRetrieverParallel(unittest.TestCase):
    def test_vec_and_bm25_run_concurrently(self):
        """vec/bm25 各 sleep 100ms; 串行应 ~200ms, 并行应 ~100ms."""
        SLEEP = 0.1

        mock_vec = MagicMock()
        mock_bm25 = MagicMock()

        def slow_vec(*args, **kwargs):
            time.sleep(SLEEP)
            return [Hit(pk="v1")]

        def slow_bm25(*args, **kwargs):
            time.sleep(SLEEP)
            return [Hit(pk="b1")]

        mock_vec.retrieve = slow_vec
        mock_bm25.retrieve = slow_bm25

        hybrid = HybridRetriever(mock_vec, mock_bm25)
        t0 = time.time()
        hits = hybrid.retrieve("test", top_k=5)
        elapsed = time.time() - t0

        # 留 80% 余量, 避免 CI 抖动误报; 串行 ~0.2s, 并行 ~0.1s, 临界 0.18s
        self.assertLess(elapsed, SLEEP * 1.8)
        # 双路结果都应该被 merge
        self.assertEqual({h.pk for h in hits}, {"v1", "b1"})

    def test_vec_failure_does_not_break_bm25(self):
        """vec 抛异常时, HybridRetriever 应回退到只用 bm25, 不抛."""
        mock_vec = MagicMock()
        mock_bm25 = MagicMock()
        mock_vec.retrieve.side_effect = RuntimeError("vec down")
        mock_bm25.retrieve.return_value = [Hit(pk="b1")]
        hybrid = HybridRetriever(mock_vec, mock_bm25)
        hits = hybrid.retrieve("test", top_k=5)
        self.assertEqual({h.pk for h in hits}, {"b1"})


class TestVectorRetrieverWithVector(unittest.TestCase):
    def test_retrieve_with_vector_skips_embed(self):
        """传入预算 qvec 时, embedder.embed 不应被调用."""
        mock_client = MagicMock()
        mock_client.search.return_value = [[{"entity": {"pk": "x1"}, "distance": 0.9}]]
        mock_embedder = MagicMock()

        vec_r = VectorRetriever(mock_client, mock_embedder, collection="c")
        qvec = [0.1, 0.2, 0.3]
        hits = vec_r.retrieve_with_vector(qvec, top_k=3)

        self.assertEqual([h.pk for h in hits], ["x1"])
        mock_embedder.embed.assert_not_called()
        # 同样的向量进了 Milvus.search
        call_args = mock_client.search.call_args
        self.assertEqual(call_args.kwargs["data"], [qvec])

    def test_retrieve_falls_back_to_embed(self):
        """不带 qvec 时仍走 embedder.embed → retrieve_with_vector 链路."""
        mock_client = MagicMock()
        mock_client.search.return_value = [[]]
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.5, 0.5]

        vec_r = VectorRetriever(mock_client, mock_embedder, collection="c")
        vec_r.retrieve("q", top_k=3)
        mock_embedder.embed_for_retrieval.assert_called_once()
        call_args = mock_embedder.embed_for_retrieval.call_args
        self.assertEqual(call_args[0][0], "q")

    def test_empty_qvec_returns_empty(self):
        mock_client = MagicMock()
        mock_embedder = MagicMock()
        vec_r = VectorRetriever(mock_client, mock_embedder, collection="c")
        hits = vec_r.retrieve_with_vector([], top_k=3)
        self.assertEqual(hits, [])
        mock_client.search.assert_not_called()


class TestRunInParallel(unittest.TestCase):
    def test_collects_named_results(self):
        out = run_in_parallel([
            ("a", lambda: 1),
            ("b", lambda: 2),
            ("c", lambda: 3),
        ])
        self.assertEqual(out, {"a": 1, "b": 2, "c": 3})

    def test_on_error_fallback(self):
        def boom():
            raise ValueError("nope")

        out = run_in_parallel(
            [("ok", lambda: "good"), ("bad", boom)],
            on_error=lambda name, exc: f"err-{name}",
        )
        self.assertEqual(out, {"ok": "good", "bad": "err-bad"})

    def test_single_task_runs_inline(self):
        """1 个任务时不进线程池, 直接同步执行 (减小开销)."""
        out = run_in_parallel([("a", lambda: 42)])
        self.assertEqual(out, {"a": 42})

    def test_empty_tasks_returns_empty(self):
        self.assertEqual(run_in_parallel([]), {})


if __name__ == "__main__":
    unittest.main()
