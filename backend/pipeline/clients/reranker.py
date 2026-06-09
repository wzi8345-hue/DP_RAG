"""Reranker 客户端: 调用 OpenAI 兼容的 /v1/rerank 接口对检索结果重排序。

使用 httpx 同步请求调用远程 rerank 服务, 通过 top_n 参数控制返回数量。
支持重试与超时, 无需本地 GPU。

用法:
    client = RerankerClient(
        api_base="http://localhost:8001/v1",
        model="Qwen/Qwen3-Reranker-0.6B",
    )
    results = client.rerank("MoS2 晶格常数", ["文档1", "文档2"], top_k=3)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class RerankResult:
    """单条 rerank 结果。"""
    index: int           # 原始列表中的下标
    score: float         # 相关性得分 [0, 1]
    content: str         # 原始内容 (透传)


# ---------------------------------------------------------------------------
# Reranker 客户端 (API 模式)
# ---------------------------------------------------------------------------

class RerankerClient:
    """基于远程 /v1/rerank API 的 Reranker 客户端。

    调用 OpenAI 兼容的 rerank 接口, 传入 query + documents, 获取相关性
    得分后按得分降序返回。

    用法:
        client = RerankerClient(
            api_base="http://localhost:8001/v1",
            model="Qwen/Qwen3-Reranker-0.6B",
        )
        results = client.rerank("MoS2 晶格常数", ["文档1", "文档2"], top_k=3)
    """

    def __init__(
        self,
        api_base: str = "http://localhost:8001/v1",
        model: str = "Qwen/Qwen3-Reranker-0.6B",
        api_key: str = "",
        top_k: int = 5,
        score_threshold: float = 0.5,
        timeout: int = 60,
        max_retries: int = 2,
        fail_open: bool = True,
        **kwargs: Any,
    ) -> None:
        """初始化 RerankerClient。

        Args:
            api_base: rerank 服务地址 (如 http://localhost:8001/v1)
            model: rerank 模型名
            api_key: vLLM --api-key 对应的 Bearer token (与 LLM/Embedding 共用)
            top_k: 默认保留的 top-k 条数
            score_threshold: 质量门控阈值
            timeout: 请求超时 (秒)
            max_retries: 最大重试次数 (最少 1)
            fail_open: True=API 全部重试失败时返回 [] 而非抛异常 (允许上游降级为不 rerank)
        """
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = (api_key or "").strip()
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.timeout = timeout
        # max_retries=0 时 for…else 会触发 raise last_err=None → 强制下限 1
        self.max_retries = max(1, int(max_retries))
        self.fail_open = bool(fail_open)

        # 兼容旧配置字段, 忽略
        self._device = kwargs.get("device")
        self._max_length = kwargs.get("max_length")
        self._batch_size = kwargs.get("batch_size")

    def _call_rerank_api(
        self, query: str, documents: List[str], top_n: int,
    ) -> List[RerankResult]:
        """调用 /v1/rerank 接口, 返回排序后的结果列表。"""
        url = f"{self.api_base}/rerank"
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }

        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_err: Optional[Exception] = None
        data: Optional[Dict[str, Any]] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                break
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    wait = min(2 ** attempt, 8)
                    logger.warning(
                        f"[reranker] API 调用失败 (attempt {attempt}/{self.max_retries}), "
                        f"{wait}s 后重试: {e}"
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        f"[reranker] API 调用失败, 已用尽重试: {e}"
                    )
        # 用尽重试后仍无 data: fail_open 时返回空列表 (上游降级为不 rerank);
        # 否则抛出最后一次异常, 让调用方自行决定
        if data is None:
            if self.fail_open:
                logger.warning(
                    f"[reranker] fail_open=True, 返回空列表; "
                    f"调用方应回退到 emb_score 排序: {last_err}"
                )
                return []
            assert last_err is not None
            raise last_err

        # 解析响应: 兼容两种常见格式
        # 格式1: {"results": [{"index": 0, "relevance_score": 0.95}, ...]}
        # 格式2: {"results": [{"index": 0, "score": 0.95}, ...]}
        raw_results = data.get("results", [])
        results: List[RerankResult] = []
        for item in raw_results:
            idx = item.get("index", 0)
            score = item.get("relevance_score") or item.get("score") or 0.0
            # 部分 API 会在结果中返回 document.text
            doc_text = ""
            doc_obj = item.get("document")
            if isinstance(doc_obj, dict):
                doc_text = doc_obj.get("text", "")
            elif isinstance(doc_obj, str):
                doc_text = doc_obj

            # 如果 API 没返回文本, 从原始 documents 中取
            content = doc_text if doc_text else (
                documents[idx] if 0 <= idx < len(documents) else ""
            )

            results.append(RerankResult(
                index=idx,
                score=max(0.0, min(1.0, float(score))),
                content=content,
            ))

        return results

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        """对 documents 列表打分并按得分降序返回。

        Args:
            query: 用户查询
            documents: 待排序的文档/片段列表
            top_k: 仅保留前 K 条 (None 则使用 self.top_k)

        Returns:
            按 score 降序排列的 RerankResult 列表
        """
        if not documents:
            return []

        top_k = top_k or self.top_k
        results = self._call_rerank_api(query, documents, top_n=top_k)

        # 按 score 降序排序 (API 可能已排序, 但不保证)
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def compute_top_k_score(
        self,
        query: str,
        documents: List[str],
        k: int = 3,
    ) -> float:
        """计算 top-k 文档的平均相关性得分。

        用于判断检索质量: 得分低于阈值时触发 reflect 重试。
        """
        if not documents:
            return 0.0
        ranked = self.rerank(query, documents, top_k=k)
        if not ranked:
            return 0.0
        return sum(r.score for r in ranked) / len(ranked)
