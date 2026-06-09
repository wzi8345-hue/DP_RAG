"""统一 Embedding 客户端 (OpenAI 兼容)。

接口:
- embed_batch(texts)  -> List[List[float]]  批量向量化
- embed_all(texts)    -> List[List[float]]  分批批量向量化
- embed(text)         -> List[float]        单条向量化 (检索时用); 带 LRU 缓存
- begin_request()     -> None               清空 query 缓存 (单次 query 边界)

可选: normalize=True 对返回向量做 L2 归一化, 配合 Milvus IP metric 等价于 cosine。

查询缓存:
- ``embed(text)`` 内部维护一个有界 LRU (默认 128 条), 在 Agentic 多路径场景下同一
  query/rewrite 被多次 embed (summary 池 + title 池 + L1 + L2 probe 等) 时只发一次
  HTTP 请求, 后续直接复用向量.
- ``begin_request()`` 在 ``AgenticRAGPipeline.answer / run`` 入口被调用, 显式清空缓存
  → 避免不同 query 之间互相干扰 (虽然 LRU 已经能自动淘汰, 但显式清空更安全).
- 批量接口 (``embed_batch / embed_all``) 不走该缓存 — 入库 chunk 通常每条都不一样,
  缓存命中率低反而增加内存压力.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import requests

from .query_format import DEFAULT_EMBED_INSTRUCTS, format_qwen3_embed_query

logger = logging.getLogger(__name__)

# 单条 embed (检索 query) 的 LRU 上限; 单次 agentic answer 通常涉及 4~8 条独立
# rewrite, 128 足够覆盖一次 chat session 内多轮对话的所有 query, 仍可控.
_QUERY_CACHE_MAX = 128


def _safe_resp_text(resp: "requests.Response", limit: int = 300) -> str:
    """按 UTF-8 解码响应体, 失败时回退到 latin-1, 避免中文报错乱码。"""
    try:
        text = resp.content.decode("utf-8", errors="replace")
    except Exception:
        text = resp.text
    return text[:limit]


def _l2_normalize(vec: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in vec))
    if n == 0:
        return vec
    return [x / n for x in vec]


class EmbeddingClient:
    """OpenAI 兼容的 Embedding 客户端 (vLLM / TIONE / 任何 OpenAI 接口)。

    设计要点:
    - embed_batch / embed 共享同一份带重试 + 4xx/5xx 区分的内部请求方法
    - 4xx 类客户端错误 (鉴权、参数) 立即抛出, 不浪费重试; 5xx / 网络异常做指数退避
    - 可选 normalize=True: 服务端没归一化向量时, 由客户端做 L2 归一化以保证
      Milvus IP metric == cosine 相似度
    """

    def __init__(
        self,
        api_base: str = "",
        model: str = "model",
        api_key: str = "",
        batch_size: int = 16,
        timeout: int = 120,
        max_retries: int = 3,
        normalize: bool = False,
        query_cache_size: int = _QUERY_CACHE_MAX,
        query_instruct_enabled: bool = True,
        query_instructs: Optional[Dict[str, str]] = None,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.normalize = normalize
        self.query_instruct_enabled = bool(query_instruct_enabled)
        self.query_instructs = dict(query_instructs or DEFAULT_EMBED_INSTRUCTS)
        # 复用 HTTP 连接池: 减少 TLS 握手开销
        self._session = requests.Session()
        # 单条 query 的 LRU 缓存 (text -> vector). Agentic 多路径会让同一 rewrite
        # 被 embed 多次, 这里命中即跳过 HTTP. 用 OrderedDict + 锁保证线程安全
        # (ThreadPoolExecutor 双路并行时多线程读写).
        self._query_cache: "OrderedDict[str, List[float]]" = OrderedDict()
        self._query_cache_size = max(0, int(query_cache_size))
        self._query_cache_lock = threading.Lock()
        self._query_cache_hits = 0
        self._query_cache_miss = 0

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _post_embeddings(self, texts: List[str]) -> List[List[float]]:
        """统一的请求方法: 4xx 立即抛, 5xx / 网络异常重试。"""
        url = f"{self.api_base}/embeddings"
        payload = {"model": self.model, "input": texts}

        last_err: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.post(
                    url, headers=self._headers(),
                    json=payload, timeout=self.timeout,
                )
                if resp.status_code == 200:
                    items = resp.json().get("data", [])
                    items.sort(key=lambda d: d.get("index", 0))
                    vecs = [d["embedding"] for d in items]
                    if self.normalize:
                        vecs = [_l2_normalize(v) for v in vecs]
                    return vecs

                # 4xx: 不重试, 立即抛 (鉴权、模型名错、payload 超限等)
                if 400 <= resp.status_code < 500:
                    raise RuntimeError(
                        f"Embedding 请求 {resp.status_code} (不重试): "
                        f"{_safe_resp_text(resp)}"
                    )

                # 5xx: 进入重试
                last_err = f"HTTP {resp.status_code}: {_safe_resp_text(resp, 200)}"
            except RuntimeError:
                raise
            except Exception as e:
                last_err = str(e)

            if attempt < self.max_retries:
                wait = 2 ** attempt
                logger.debug(
                    f"  [embed retry {attempt}/{self.max_retries}] {last_err} -> wait {wait}s"
                )
                time.sleep(wait)
        raise RuntimeError(f"Embedding 请求失败: {last_err}")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量向量化 (单次请求, 受 batch_size 限制)。"""
        if not texts:
            return []
        return self._post_embeddings(texts)

    def embed_all(self, texts: List[str]) -> List[List[float]]:
        """分批向量化所有文本。"""
        results: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            logger.debug(f"  embedding [{i + 1}-{i + len(batch)}]/{len(texts)}")
            results.extend(self.embed_batch(batch))
        return results

    # ── 单条 query 缓存 ────────────────────────────────────────────────

    def _cache_get(self, text: str) -> Optional[List[float]]:
        if self._query_cache_size <= 0:
            return None
        with self._query_cache_lock:
            vec = self._query_cache.get(text)
            if vec is None:
                return None
            # LRU 触摸: 移到末尾
            self._query_cache.move_to_end(text)
            self._query_cache_hits += 1
            return vec

    def _cache_put(self, text: str, vec: List[float]) -> None:
        if self._query_cache_size <= 0 or not vec:
            return
        with self._query_cache_lock:
            self._query_cache[text] = vec
            self._query_cache.move_to_end(text)
            while len(self._query_cache) > self._query_cache_size:
                self._query_cache.popitem(last=False)
            self._query_cache_miss += 1

    def begin_request(self) -> None:
        """单次 user query 入口处调用: 清空 query 缓存, 防止跨 query 复用旧向量.

        在 ``AgenticRAGPipeline.run`` / ``langgraph_agent.run`` 等入口调用; 不调用
        也能工作 (LRU 自动淘汰), 但显式清空可以释放内存 + 让日志统计每轮独立.
        """
        if self._query_cache_size <= 0:
            return
        with self._query_cache_lock:
            self._query_cache.clear()
            self._query_cache_hits = 0
            self._query_cache_miss = 0

    def query_cache_stats(self) -> Tuple[int, int, int]:
        """返回 (hits, miss, size). 用于性能日志."""
        with self._query_cache_lock:
            return (
                self._query_cache_hits, self._query_cache_miss,
                len(self._query_cache),
            )

    def format_retrieval_query(
        self, text: str, stage: Optional[str] = None,
    ) -> str:
        """检索 query 格式化 (Qwen3 instruct); 入库 document 勿调用。"""
        return format_qwen3_embed_query(
            text,
            stage,
            enabled=self.query_instruct_enabled,
            instructs=self.query_instructs,
        )

    def embed_for_retrieval(
        self, text: str, stage: Optional[str] = None,
    ) -> List[float]:
        """检索侧 embed: 按 stage 加 instruct 后走 LRU 缓存。"""
        formatted = self.format_retrieval_query(text, stage)
        return self.embed(formatted)

    def embed(self, text: str) -> List[float]:
        """单条文本向量化 (检索时常用); 与 embed_batch 共享重试逻辑.

        命中 query 级 LRU 时直接复用; miss 时下发 HTTP 并写回缓存. 空文本不缓存,
        失败抛出的异常按 ``_post_embeddings`` 的语义透传 (不会写入缓存).
        """
        if not text:
            return []
        cached = self._cache_get(text)
        if cached is not None:
            return cached
        vecs = self._post_embeddings([text])
        vec = vecs[0] if vecs else []
        self._cache_put(text, vec)
        return vec
