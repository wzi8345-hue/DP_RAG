"""ClientRegistry: 单进程内共享 HTTP / gRPC 客户端连接池.

设计目标:
- 避免 ``ChunkStep / EmbedStep / StoreStep / GenerateStep / RetrieveStep``
  每次调用都 ``new EmbeddingClient()`` / ``new MilvusIngester()`` / ``new LLMClient()``,
  其内部的 ``requests.Session`` / gRPC channel / TLS 握手开销随调用次数线性放大.
- 同一 ``Pipeline`` 实例生命周期内, 按 ``(api_base, model, api_key, ...)`` 维度做单例缓存.
- 完全向后兼容: 注册表未挂到 ``config.clients`` 时, 各模块仍可走原 ``new XxxClient()`` 路径.

线程安全:
- 所有 ``get_*`` 通过模块级锁 + 双检 (``if key in cache``), 即便多线程同时拿同一 client
  也只会创建一次. ThreadPoolExecutor 在 ``AgenticRAGPipeline._dispatch`` 里跑多路并行,
  此处必须保证安全.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional, Tuple

from .embedding import EmbeddingClient
from .llm import LLMClient

logger = logging.getLogger(__name__)


# 缓存 key: 仅按"连接身份"字段, 不要把 batch_size / timeout 等可变运行时参数纳入
# (否则同一个后端会因为 batch_size 差异被实例化成两份, 失去复用意义).
EmbeddingCacheKey = Tuple[str, str, str, bool, bool]   # (api_base, model, api_key, normalize, query_instruct)
LLMCacheKey = Tuple[str, str, str, bool]         # (api_base, model, api_key, disable_thinking_extra_body)
IngesterCacheKey = Tuple[str, str, str, str, int]  # (uri, token, db_name, collection, dim)


class ClientRegistry:
    """轻量级单例客户端注册表 (挂在 Pipeline 实例上).

    使用模式::

        registry = ClientRegistry()
        emb = registry.get_embedder(
            api_base="https://...", model="...", api_key="sk-...",
            batch_size=16, timeout=120, max_retries=3, normalize=True,
        )
        llm = registry.get_llm(api_base="...", model="...", api_key="...")
        ingester = registry.get_milvus_ingester(uri="...", collection="...", dim=1024)

    后续相同参数的调用直接复用已创建实例; 新参数会创建新实例并加入缓存.
    """

    def __init__(self) -> None:
        self._embed_cache: Dict[EmbeddingCacheKey, EmbeddingClient] = {}
        self._llm_cache: Dict[LLMCacheKey, LLMClient] = {}
        # MilvusIngester 维持惰性导入: 这个类依赖 pymilvus, 而 clients/__init__.py
        # 已经 import MilvusIngester, 不会有循环引用; 不过为避免和已存在的
        # _milvus_client_cache 重复 (retrievers.py 里的), 这里仅缓存 Ingester 本体.
        self._ingester_cache: Dict[IngesterCacheKey, Any] = {}
        self._lock = threading.Lock()

    # ── Embedding ──────────────────────────────────────────────────────

    def get_embedder(
        self,
        api_base: str = "",
        model: str = "model",
        api_key: str = "",
        batch_size: int = 16,
        timeout: int = 120,
        max_retries: int = 3,
        normalize: bool = False,
        query_instruct_enabled: bool = True,
        query_instructs: Optional[Dict[str, str]] = None,
    ) -> EmbeddingClient:
        """按 (api_base, model, api_key, normalize, query_instruct) 复用 ``EmbeddingClient``。

        ``batch_size`` / ``timeout`` / ``max_retries`` 只在首次创建生效;
        生产实践里这些参数对同一后端通常是固定值, 没必要为它们拆出多份 client.
        """
        key: EmbeddingCacheKey = (
            (api_base or "").rstrip("/"), model or "", api_key or "", bool(normalize),
            bool(query_instruct_enabled),
        )
        cli = self._embed_cache.get(key)
        if cli is not None:
            return cli
        with self._lock:
            cli = self._embed_cache.get(key)
            if cli is not None:
                return cli
            cli = EmbeddingClient(
                api_base=api_base, model=model, api_key=api_key,
                batch_size=batch_size, timeout=timeout, max_retries=max_retries,
                normalize=normalize,
                query_instruct_enabled=query_instruct_enabled,
                query_instructs=query_instructs,
            )
            self._embed_cache[key] = cli
            logger.debug(
                f"[client-registry] new EmbeddingClient cached "
                f"api_base={api_base!r} model={model!r} normalize={normalize}"
            )
            return cli

    # ── LLM ────────────────────────────────────────────────────────────

    def get_llm(
        self,
        api_base: str = "",
        model: str = "",
        api_key: str = "",
        timeout: int = 120,
        max_retries: int = 3,
        disable_thinking_extra_body: bool = False,
    ) -> LLMClient:
        """按 (api_base, model, api_key, extra_body) 复用 ``LLMClient``."""
        if not api_key:
            raise ValueError(
                "ClientRegistry.get_llm 缺少 api_key, 请通过 config 提供"
            )
        key: LLMCacheKey = (
            (api_base or "").rstrip("/"), model or "", api_key,
            bool(disable_thinking_extra_body),
        )
        cli = self._llm_cache.get(key)
        if cli is not None:
            return cli
        with self._lock:
            cli = self._llm_cache.get(key)
            if cli is not None:
                return cli
            cli = LLMClient(
                api_base=api_base, model=model, api_key=api_key,
                timeout=timeout, max_retries=max_retries,
                disable_thinking_extra_body=disable_thinking_extra_body,
            )
            self._llm_cache[key] = cli
            logger.debug(
                f"[client-registry] new LLMClient cached "
                f"api_base={api_base!r} model={model!r}"
            )
            return cli

    # ── Milvus Ingester ────────────────────────────────────────────────

    def get_milvus_ingester(
        self,
        uri: str,
        token: str = "",
        db_name: str = "",
        collection: str = "literature_chunks",
        dim: int = 1024,
        recreate: bool = False,
        analyzer_params: Optional[Dict[str, Any]] = None,
        dense_index_type: str = "AUTOINDEX",
        dense_metric: str = "IP",
        dense_index_params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """按 (uri, token, db_name, collection, dim) 复用 ``MilvusIngester``.

        ``recreate=True`` 会强制重建集合, 此时**不复用缓存**(否则同 key 下第二次
        recreate 会被静默吞掉). 重建后的 ingester 替换缓存里的旧实例.
        """
        from .milvus import MilvusIngester

        key: IngesterCacheKey = (
            uri or "", token or "", db_name or "",
            collection or "", int(dim),
        )
        if not recreate:
            cli = self._ingester_cache.get(key)
            if cli is not None:
                return cli
        with self._lock:
            if not recreate:
                cli = self._ingester_cache.get(key)
                if cli is not None:
                    return cli
            cli = MilvusIngester(
                uri=uri, token=token, db_name=db_name,
                collection=collection, dim=dim, recreate=recreate,
                analyzer_params=analyzer_params,
                dense_index_type=dense_index_type,
                dense_metric=dense_metric,
                dense_index_params=dense_index_params,
            )
            self._ingester_cache[key] = cli
            logger.debug(
                f"[client-registry] new MilvusIngester cached "
                f"uri={uri!r} collection={collection!r} dim={dim} recreate={recreate}"
            )
            return cli

    def evict_milvus_ingester(
        self,
        uri: str,
        token: str = "",
        db_name: str = "",
        collection: str = "literature_chunks",
        dim: int = 1024,
    ) -> None:
        """从缓存移除 MilvusIngester (gRPC 断线后强制下次 get 时重建连接)。"""
        key: IngesterCacheKey = (
            uri or "", token or "", db_name or "",
            collection or "", int(dim),
        )
        with self._lock:
            old = self._ingester_cache.pop(key, None)
        if old is not None:
            try:
                close_fn = getattr(getattr(old, "client", None), "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception:
                pass
            logger.debug(f"[client-registry] evicted MilvusIngester key={key!r}")

    # ── 调试 / 测试辅助 ────────────────────────────────────────────────

    def clear(self) -> None:
        """清空所有缓存 (测试用)."""
        with self._lock:
            self._embed_cache.clear()
            self._llm_cache.clear()
            self._ingester_cache.clear()

    def stats(self) -> Dict[str, int]:
        """返回各缓存大小, 便于调试."""
        return {
            "embedders": len(self._embed_cache),
            "llms": len(self._llm_cache),
            "ingesters": len(self._ingester_cache),
        }


# 全局兜底单例 (当 Pipeline 没有显式注入 registry 时使用; 同进程内仍然能复用).
_GLOBAL_REGISTRY: Optional[ClientRegistry] = None
_GLOBAL_REGISTRY_LOCK = threading.Lock()


def get_global_registry() -> ClientRegistry:
    """获取全局 ClientRegistry 单例 (lazy init).

    设计目的: 即便老代码没显式持有 ``Pipeline`` 实例 (如某些 CLI 入口直接调
    Step.run), 也能享受连接复用. ``Pipeline.__init__`` 会调用 ``set_global_registry``
    把自己的 registry 注册成全局, 让所有 step 都自动命中同一份缓存.
    """
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is not None:
        return _GLOBAL_REGISTRY
    with _GLOBAL_REGISTRY_LOCK:
        if _GLOBAL_REGISTRY is None:
            _GLOBAL_REGISTRY = ClientRegistry()
        return _GLOBAL_REGISTRY


def set_global_registry(registry: ClientRegistry) -> None:
    """注入一个 ClientRegistry 作为全局单例 (通常由 Pipeline 调用)."""
    global _GLOBAL_REGISTRY
    with _GLOBAL_REGISTRY_LOCK:
        _GLOBAL_REGISTRY = registry


def reset_global_registry() -> None:
    """重置全局 registry (测试用)."""
    global _GLOBAL_REGISTRY
    with _GLOBAL_REGISTRY_LOCK:
        _GLOBAL_REGISTRY = None
