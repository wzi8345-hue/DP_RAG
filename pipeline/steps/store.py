"""步骤 4: 存储 — 将向量化的知识块灌入 Milvus。

使用 pipeline.clients.milvus.MilvusIngester, 逻辑与原 StoreStep 一致。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from .base import BaseStep, StepResult, register_step
from ..clients.client_registry import get_global_registry
from ..clients.milvus import resolve_milvus_connection

logger = logging.getLogger(__name__)


@register_step
class StoreStep(BaseStep):
    """将 vectorized chunks 灌入 Milvus 向量数据库。"""

    name = "store"

    def run(self, **kwargs) -> StepResult:
        cfg = self.config.milvus
        input_path = kwargs.get("input_path") or cfg.get("input_path", "knowledge_blocks_vec.json")
        doc_id = kwargs.get("doc_id") or cfg.get("doc_id")
        doc_name = kwargs.get("doc_name") or cfg.get("doc_name")
        stats_only = kwargs.get("stats_only", False)
        # stats_only 模式只查询不入库, 必须强制 recreate=False, 否则会把现有
        # 集合 drop 掉, 然后查到一个空集合.
        recreate = (
            False if stats_only
            else kwargs.get("recreate", cfg.get("recreate", False))
        )

        index_cfg = cfg.get("index", {}) or {}
        bm25_cfg = cfg.get("bm25", {}) or {}
        uri, token, db_name = resolve_milvus_connection(cfg)
        # 注意: get_milvus_ingester 在 recreate=True 时仍会新建实例
        # (避免静默吞掉 drop_collection 语义).
        ingester = get_global_registry().get_milvus_ingester(
            uri=uri,
            token=token,
            db_name=db_name,
            collection=cfg.get("collection", "literature_chunks"),
            dim=cfg.get("dim", 1024),
            recreate=recreate,
            analyzer_params=bm25_cfg.get("analyzer") or None,
            dense_index_type=str(index_cfg.get("dense_type", "AUTOINDEX")),
            dense_metric=str(index_cfg.get("dense_metric", "IP")),
            dense_index_params=index_cfg.get("dense_params") or None,
        )

        if stats_only:
            stat = ingester.stats()
            logger.info(f"集合统计: {stat}")
            return StepResult(self.name, success=True, data={"stats": stat})

        if not os.path.exists(input_path):
            return StepResult(self.name, success=False, error=f"输入文件不存在: {input_path}")

        result = ingester.ingest_file(
            input_path,
            doc_id=doc_id,
            doc_name=doc_name,
            purge_existing=True,
            batch_size=cfg.get("batch_size", 100),
        )

        stat = ingester.stats()
        logger.info(f"灌入完成: {result}")
        logger.info(f"集合统计: {stat}")

        return StepResult(self.name, success=True, data={
            "ingest_result": result,
            "stats": stat,
        })
