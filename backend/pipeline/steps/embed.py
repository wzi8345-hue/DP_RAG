"""步骤 3: 向量化 — 将知识块通过 Embedding 模型转为向量。

使用 pipeline.processors.vectorizer 和 pipeline.clients.embedding, 逻辑与原 EmbedStep 一致。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from .base import BaseStep, StepResult, register_step
from ..clients.client_registry import get_global_registry
from ..processors.vectorizer import vectorize_chunks, compose_embedding_text

logger = logging.getLogger(__name__)


@register_step
class EmbedStep(BaseStep):
    """对 knowledge blocks 进行向量化。"""

    name = "embed"

    def run(self, **kwargs) -> StepResult:
        cfg = self.config.embedding
        input_path = kwargs.get("input_path") or cfg.get("input_path", "knowledge_blocks.json")
        output_path = kwargs.get("output_path") or cfg.get("output_path", "knowledge_blocks_vec.json")
        dry_run = kwargs.get("dry_run", False)

        if not os.path.exists(input_path):
            return StepResult(self.name, success=False, error=f"输入文件不存在: {input_path}")

        with open(input_path, "r", encoding="utf-8") as f:
            chunks: List[Dict[str, Any]] = json.load(f)
        logger.info(f"读取 {len(chunks)} 个 chunks: {input_path}")

        if dry_run:
            preview = []
            for c in chunks:
                txt = compose_embedding_text(c, max_chars=cfg.get("max_chars", 8000))
                new_c = dict(c)
                new_c["embedding_text"] = txt
                preview.append(new_c)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(preview, f, ensure_ascii=False, indent=2)
            logger.info(f"[dry-run] 已写入: {output_path}")
            return StepResult(self.name, success=True, data={
                "output_path": output_path, "total": len(preview), "dry_run": True,
            })

        embedder = get_global_registry().get_embedder(
            api_base=cfg.get("api_base", ""),
            model=cfg.get("model", "model"),
            api_key=cfg.get("api_key", ""),
            batch_size=cfg.get("batch_size", 16),
            timeout=cfg.get("timeout", 120),
            max_retries=cfg.get("max_retries", 3),
            normalize=bool(cfg.get("normalize", False)),
        )
        logger.info(f"调用 Embedding @ {cfg.get('api_base')}, model={cfg.get('model')}")

        vectorized = vectorize_chunks(chunks, embedder, max_chars=cfg.get("max_chars", 8000))

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(vectorized, f, ensure_ascii=False, indent=2)

        dim = vectorized[0]["embedding_dim"] if vectorized else 0
        logger.info(f"已写入 {len(vectorized)} 个带向量 chunks: {output_path}, dim={dim}")

        return StepResult(self.name, success=True, data={
            "output_path": output_path,
            "total": len(vectorized),
            "embedding_dim": dim,
        })
