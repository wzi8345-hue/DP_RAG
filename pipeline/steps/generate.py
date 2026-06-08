"""步骤 6: 生成 — 检索 + LLM 生成回答。

使用 pipeline.retrieval.agentic 和 pipeline.clients.llm, 逻辑与原 GenerateStep 一致。
默认使用 Agentic RAG 模式; 可通过 use_agentic=False 回退到简单检索+生成。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Iterator, List, Optional

from .base import BaseStep, StepResult, register_step
from ..clients.client_registry import get_global_registry
from ..clients.llm import LLMClient
from ..clients.milvus import resolve_milvus_connection
from ..retrieval.agentic import (
    AgenticRAGPipeline,
    build_agentic_pipeline,
    DEFAULT_AGENTIC_SYSTEM_PROMPT,
    AGENTIC_USER_TEMPLATE,
)
from ..retrieval.retrievers import build_retrievers
from ..retrieval.context_builder import ContextBuilder
from dataclasses import asdict

from ..retrieval.retrievers import Hit

logger = logging.getLogger(__name__)


@register_step
class GenerateStep(BaseStep):
    """基于检索结果调用 LLM 生成回答。"""

    name = "generate"

    def run(self, **kwargs) -> StepResult:
        query = kwargs.get("query")
        if not query:
            return StepResult(self.name, success=False, error="未提供 query")

        use_agentic = kwargs.get("use_agentic", True)

        if use_agentic:
            return self._run_agentic(query, **kwargs)
        else:
            return self._run_simple(query, **kwargs)

    def _run_agentic(self, query: str, **kwargs) -> StepResult:
        """Agentic RAG 模式生成。"""
        gen_cfg = self.config.generation
        milvus_cfg = self.config.milvus
        emb_cfg = self.config.embedding

        milvus_uri, milvus_token, milvus_db = resolve_milvus_connection(milvus_cfg)
        pipeline = build_agentic_pipeline(
            milvus_uri=milvus_uri,
            milvus_token=milvus_token,
            db_name=milvus_db,
            collection=milvus_cfg.get("collection", "literature_chunks"),
            embed_api_base=emb_cfg.get("api_base", ""),
            embed_model=emb_cfg.get("model", "model"),
            embed_api_key=emb_cfg.get("api_key", ""),
            llm_api_base=gen_cfg.get("api_base", "https://api.gpugeek.com/v1"),
            llm_model=gen_cfg.get("model", "DeepSeek/DeepSeek-V3-0324"),
            llm_api_key=gen_cfg.get("api_key", ""),
            use_router_llm=True,
            enable_generation_llm=True,
            disable_thinking=bool(gen_cfg.get("disable_thinking", True)),
            disable_thinking_extra_body=bool(gen_cfg.get("disable_thinking_extra_body", False)),
        )

        system_prompt = kwargs.get("system_prompt") or gen_cfg.get("system_prompt") or DEFAULT_AGENTIC_SYSTEM_PROMPT
        temperature = kwargs.get("temperature", gen_cfg.get("temperature", 0))
        max_tokens = kwargs.get("max_tokens", gen_cfg.get("max_tokens", 2048))
        stream = kwargs.get("stream", gen_cfg.get("stream", False))

        t0 = time.time()
        try:
            result = pipeline.answer(
                query, system=system_prompt,
                temperature=temperature, max_tokens=max_tokens,
                stream=stream,
            )
        except Exception as e:
            return StepResult(self.name, success=False, error=str(e))

        elapsed = time.time() - t0
        answer = result.get("answer", "")
        usage = result.get("usage")
        context = result.get("context", "")

        # 序列化 hits
        hits_data: List[Dict[str, Any]] = []
        for route, res in result.get("results", {}).items():
            if hasattr(res, "chunk_hits"):
                hits_data.extend([asdict(h) if isinstance(h, Hit) else h for h in res.chunk_hits])
            elif isinstance(res, list):
                hits_data.extend([asdict(h) if isinstance(h, Hit) else h for h in res])

        output_data = {
            "query": query,
            "answer": answer,
            "hits": hits_data,
            "context": context,
            "usage": usage,
            "latency_s": round(elapsed, 3),
        }

        output_file = kwargs.get("output_file")
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            logger.info(f"结果已写入: {output_file}")

        return StepResult(self.name, success=True, data=output_data)

    def _run_simple(self, query: str, **kwargs) -> StepResult:
        """简单模式: 已有 hits 则直接用, 否则先检索。"""
        gen_cfg = self.config.generation
        milvus_cfg = self.config.milvus
        emb_cfg = self.config.embedding
        ret_cfg = self.config.retrieval

        hits = kwargs.get("hits")
        context = kwargs.get("context")

        if hits is None:
            from .retrieve import RetrieveStep
            retrieve_step = RetrieveStep(self.config)
            ret_result = retrieve_step._execute(query=query, **kwargs)
            if not ret_result.success:
                return StepResult(self.name, success=False, error=f"检索失败: {ret_result.error}")
            hits = ret_result.data["hits"]
            context = ret_result.data["context"]

        llm = get_global_registry().get_llm(
            api_base=gen_cfg.get("api_base", "https://api.gpugeek.com/v1"),
            model=gen_cfg.get("model", "DeepSeek/DeepSeek-V3-0324"),
            api_key=gen_cfg.get("api_key", ""),
            timeout=gen_cfg.get("timeout", 120),
            max_retries=gen_cfg.get("max_retries", 3),
            disable_thinking_extra_body=bool(gen_cfg.get("disable_thinking_extra_body", False)),
        )

        system_prompt = kwargs.get("system_prompt") or gen_cfg.get("system_prompt", "")
        temperature = kwargs.get("temperature", gen_cfg.get("temperature", 0))
        max_tokens = kwargs.get("max_tokens", gen_cfg.get("max_tokens", 2048))
        stream = kwargs.get("stream", gen_cfg.get("stream", False))
        disable_thinking = bool(gen_cfg.get("disable_thinking", True))

        user_msg = _build_user_message(context, query)

        t0 = time.time()
        ttft = None
        if stream:
            answer_parts = []
            for piece in llm.chat_stream(
                system=system_prompt, user=user_msg,
                temperature=temperature, max_tokens=max_tokens,
                disable_thinking=disable_thinking,
            ):
                if ttft is None:
                    ttft = time.time() - t0
                    print(f"\n[首包] ttft={ttft:.2f}s", flush=True)
                answer_parts.append(piece)
                print(piece, end="", flush=True)
            print()
            answer = "".join(answer_parts)
            usage = None
        else:
            result = llm.chat(
                system=system_prompt, user=user_msg,
                temperature=temperature, max_tokens=max_tokens,
                disable_thinking=disable_thinking,
            )
            answer = result["answer"]
            usage = result.get("usage")
        elapsed = time.time() - t0

        if ttft is not None:
            logger.info(f"[耗时] generate={elapsed:.2f}s (ttft={ttft:.2f}s)")
        else:
            logger.info(f"[耗时] generate={elapsed:.2f}s")

        hits_serialized = [asdict(h) if isinstance(h, Hit) else h for h in (hits or [])]

        output_data = {
            "query": query,
            "answer": answer,
            "hits": hits_serialized,
            "context": context,
            "usage": usage,
            "latency_s": round(elapsed, 3),
        }

        output_file = kwargs.get("output_file")
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            logger.info(f"结果已写入: {output_file}")

        return StepResult(self.name, success=True, data=output_data)


def _build_user_message(context: str, query: str) -> str:
    return (
        "# 检索到的上下文\n"
        f"{context}\n\n"
        "# 用户问题\n"
        f"{query}\n\n"
        "请基于上下文给出严谨、有引用的回答。"
    )
