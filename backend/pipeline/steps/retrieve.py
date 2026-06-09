"""步骤 5: 检索 — 从 Milvus 中召回相关 chunks。

使用 pipeline.retrieval.retrievers, 逻辑与原 RetrieveStep 一致。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .base import BaseStep, StepResult, register_step
from ..clients.milvus import resolve_milvus_connection
from ..clients.query_format import instruct_kwargs_from_embedding_cfg
from ..retrieval.retrievers import build_retrievers, _build_filter_expr
from ..retrieval.context_builder import ContextBuilder
from ..retrieval.hybrid_config import hybrid_config_from_dict
from ..retrieval.hybrid_weights import (
    STAGE_SIMPLE,
    infer_hybrid_weights,
    infer_retrieve_bias_heuristic,
    format_weight_log,
)

logger = logging.getLogger(__name__)


@register_step
class RetrieveStep(BaseStep):
    """从 Milvus 检索与 query 相关的 chunks。"""

    name = "retrieve"

    def run(self, **kwargs) -> StepResult:
        query = kwargs.get("query")
        if not query:
            return StepResult(self.name, success=False, error="未提供 query")

        ret_cfg = self.config.retrieval
        milvus_cfg = self.config.milvus
        emb_cfg = self.config.embedding
        # 之前漏掉了 dense_metric / embed_normalize / search_params, 会导致
        # 索引按 COSINE 建, 搜索却按默认 IP, 返回结果完全错乱.
        index_cfg = milvus_cfg.get("index", {}) or {}
        search_cfg = ret_cfg.get("search", {}) or {}

        mode = kwargs.get("mode") or ret_cfg.get("mode", "hybrid")
        top_k = kwargs.get("top_k") or ret_cfg.get("top_k", 5)
        per_retriever_k = kwargs.get("per_retriever_k") or ret_cfg.get("per_retriever_k", 10)
        doc_id = kwargs.get("doc_id") or ret_cfg.get("doc_id")
        chunk_type = kwargs.get("chunk_type") or ret_cfg.get("chunk_type")

        milvus_uri, milvus_token, milvus_db = resolve_milvus_connection(milvus_cfg)
        meta, vec, bm25, hybrid = build_retrievers(
            milvus_uri=milvus_uri,
            milvus_token=milvus_token,
            db_name=milvus_db,
            collection=milvus_cfg.get("collection", "literature_chunks"),
            embed_api_base=emb_cfg.get("api_base", ""),
            embed_model=emb_cfg.get("model", "model"),
            embed_api_key=emb_cfg.get("api_key", ""),
            embed_normalize=bool(emb_cfg.get("normalize", False)),
            dense_weight=float(ret_cfg.get("dense_weight", 0.6)),
            bm25_weight=float(ret_cfg.get("bm25_weight", 0.4)),
            dense_metric=str(index_cfg.get("dense_metric", "IP")),
            dense_search_params=search_cfg.get("dense") or None,
            bm25_search_params=search_cfg.get("bm25") or None,
            **instruct_kwargs_from_embedding_cfg(emb_cfg),
        )

        retriever_map = {
            "metadata": meta, "vector": vec, "bm25": bm25, "hybrid": hybrid,
        }
        retriever = retriever_map.get(mode, hybrid)

        filter_expr = _build_filter_expr(doc_id, chunk_type)

        retrieve_kwargs: Dict[str, Any] = {
            "top_k": top_k,
            "filter_expr": filter_expr,
        }
        if mode == "hybrid":
            retrieve_kwargs["per_retriever_k"] = per_retriever_k
            hybrid_cfg = hybrid_config_from_dict(ret_cfg.get("hybrid"))
            bias = infer_retrieve_bias_heuristic(query, chunk_type=chunk_type)
            weights = infer_hybrid_weights(
                STAGE_SIMPLE, query, retrieve_bias=bias, chunk_type=chunk_type,
                config=hybrid_cfg,
            )
            retrieve_kwargs["dense_weight"] = weights.dense
            retrieve_kwargs["bm25_weight"] = weights.bm25
            logger.info(f"[hybrid] {format_weight_log(weights)}")

        hits = retriever.retrieve(query, **retrieve_kwargs)
        context = ContextBuilder().build(hits, query=query)

        logger.info(f"[{mode}] retrieved {len(hits)} hits")

        return StepResult(self.name, success=True, data={
            "hits": hits,
            "context": context,
            "mode": mode,
            "top_k": top_k,
        })
