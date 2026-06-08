"""步骤 2: 分块 — 把解析产物 (MinerU content_list_v2.json 或 UniParser
uniparser_result.json) 转为 v5 知识块.

按 ``parsing.backend`` 或 kwargs.backend 分发到对应 chunker. 默认 mineru.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, Optional

from .base import BaseStep, StepResult, register_step
from ..clients.client_registry import get_global_registry
from ..processors.chunker import build_knowledge_blocks, autodiscover_content_list_v2
from ..processors.semantic_splitter import estimate_tokens
from ..processors.uniparser_chunker import (
    build_knowledge_blocks_uniparser,
    autodiscover_uniparser_result,
    load_uniparser_result,
    write_meta_sidecar as write_uniparser_meta_sidecar,
)
from ..flows.ingest import _summary_kwargs_from_config, _build_summary_llm

logger = logging.getLogger(__name__)

# ── doc_title 兜底清理 (P0-2) ────────────────────────────────────────────
# MinerU 目录名常带期刊编号后缀 (CNKI/SNAD/CYJB/万方/...), 例如:
#   "典型耐候钢..._JSCX202308015"
#   "锌铝镁合金镀层板研究开发_SNAD000001828487"
#   "NH35q耐大气腐蚀钢冷裂敏感性试验研究_HSJJ601.000"
# 这些编号进 title chunk -> BM25 高 IDF token, 容易跨文献误召回 (rare token 命中等价 99%).
# 这里把它们剥掉, 完整 stem 由调用方保留在外部 (doc_id / Milvus pk).
_DOC_TITLE_ID_SUFFIX_RE = re.compile(r"_[A-Z]{2,8}[\d\.]+$")


def _split_runtime_from_cfg(split_cfg: Dict) -> Dict:
    """从 semantic_split 配置解析运行时切分参数 (size_unit / overlap)。

    返回可直接作为 kwargs 传给 build_knowledge_blocks(_uniparser) 的字典:
    - split_length_fn: size_unit=token 时为 estimate_tokens, 否则 None (按字符)
    - split_overlap:   句子边界 overlap 预算 (size_unit 单位), 默认 0
    """
    out: Dict = {}
    size_unit = str(split_cfg.get("size_unit", "char")).strip().lower()
    if size_unit == "token":
        out["split_length_fn"] = estimate_tokens
    overlap = split_cfg.get("overlap")
    if overlap is not None:
        try:
            out["split_overlap"] = max(0, int(overlap))
        except (TypeError, ValueError):
            pass
    return out


def _sanitize_doc_title(stem: str) -> str:
    """剥掉 CNKI / SNAD / CYJB 等期刊编号后缀, 保留可读标题.

    Examples:
        >>> _sanitize_doc_title("典型耐候钢..._JSCX202308015")
        '典型耐候钢...'
        >>> _sanitize_doc_title("锌铝镁合金镀层板研究开发_SNAD000001828487")
        '锌铝镁合金镀层板研究开发'
        >>> _sanitize_doc_title("NH35q耐大气腐蚀钢冷裂敏感性试验研究_HSJJ601.000")
        'NH35q耐大气腐蚀钢冷裂敏感性试验研究'
        >>> _sanitize_doc_title("没有编号后缀的标题")
        '没有编号后缀的标题'
        >>> _sanitize_doc_title("")
        ''
    """
    if not stem:
        return stem
    cleaned = _DOC_TITLE_ID_SUFFIX_RE.sub("", stem).rstrip("_").strip()
    return cleaned or stem


def _resolve_backend(config, kwargs: Dict) -> str:
    """从 kwargs 或 config.parsing.backend 决定 chunk 走哪条支路。"""
    b = (kwargs.get("backend") or "").strip().lower()
    if not b:
        parsing_cfg = (config.parsing or {})
        b = (parsing_cfg.get("backend") or "mineru").strip().lower()
    if b not in ("mineru", "uniparser"):
        raise ValueError(f"未知 backend={b!r}; 仅支持 mineru / uniparser")
    return b


@register_step
class ChunkStep(BaseStep):
    """将解析结果 (MinerU 或 UniParser) 转为 v5 知识块。"""

    name = "chunk"

    def run(self, **kwargs) -> StepResult:
        backend = _resolve_backend(self.config, kwargs)
        if backend == "uniparser":
            return self._run_uniparser(**kwargs)
        return self._run_mineru(**kwargs)

    # ── MinerU 支路 (原 ChunkStep 实现) ───────────────────────────────────

    def _run_mineru(self, **kwargs) -> StepResult:
        cfg = self.config.chunking
        content_list_path = kwargs.get("content_list_path") or cfg.get("content_list_path")
        output_path = kwargs.get("output_path") or cfg.get("output_path", "knowledge_blocks.json")
        summary_title_count = kwargs.get("summary_title_count")
        if summary_title_count is None:
            summary_title_count = cfg.get("summary_title_count", 2)
        summary_sim_threshold = kwargs.get("summary_sim_threshold")
        if summary_sim_threshold is None:
            summary_sim_threshold = cfg.get("summary_sim_threshold", 0.72)
        doc_title = kwargs.get("doc_title")

        if not content_list_path:
            content_list_path = autodiscover_content_list_v2(
                kwargs.get("mineru_output_dir", "mineru_result")
            )

        # 没显式传 doc_title 时, 用解析目录名作为兜底 (mineru 标准布局: <pdf_stem>/<pdf_stem>/)
        # 兜底名常含 CNKI/SNAD/CYJB 等期刊编号后缀, 这里 sanitize 剥掉 (见 _sanitize_doc_title).
        if not doc_title and content_list_path:
            parent = os.path.dirname(content_list_path)
            grandparent = os.path.dirname(parent)
            stem_candidate = os.path.basename(grandparent) or os.path.basename(parent)
            if stem_candidate:
                doc_title = _sanitize_doc_title(stem_candidate)
                if doc_title != stem_candidate:
                    logger.info(
                        f"  [doc_title] 已剥编号后缀: {stem_candidate!r} -> {doc_title!r}"
                    )

        if not content_list_path or not os.path.exists(content_list_path):
            return StepResult(
                self.name, success=False,
                error="未找到 content_list_v2.json, 请先运行 parse 步骤",
            )

        logger.info(
            f"读取: {content_list_path} "
            f"(summary_title_count={summary_title_count}, summary_sim_threshold={summary_sim_threshold})"
        )
        with open(content_list_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list) or (data and not isinstance(data[0], list)):
            return StepResult(
                self.name, success=False,
                error="content_list_v2.json 应是二维 list (按页分组), 实际格式不符",
            )

        images_root = os.path.dirname(content_list_path)
        embed_cfg = self.config.embedding
        embedder = get_global_registry().get_embedder(
            api_base=embed_cfg.get("api_base", ""),
            model=embed_cfg.get("model", "model"),
            api_key=embed_cfg.get("api_key", ""),
            batch_size=embed_cfg.get("batch_size", 16),
            timeout=embed_cfg.get("timeout", 120),
            max_retries=embed_cfg.get("max_retries", 3),
            normalize=bool(embed_cfg.get("normalize", False)),
        )

        # 摘要兜底 LLM (chunking.summary.llm + generation 回退)
        llm = _build_summary_llm(self.config)

        # semantic_split 参数 (从 config 注入, 缺省回退到 chunker 内置默认)
        split_cfg = (cfg.get("semantic_split") or {}) if isinstance(cfg, dict) else {}
        split_kwargs: Dict[str, object] = {}
        for k_yaml, k_arg in (
            ("target_chars", "split_target_chars"),
            ("max_chars", "split_max_chars"),
            ("min_chars", "split_min_chars"),
            ("breakpoint_percentile", "split_breakpoint_percentile"),
        ):
            if k_yaml in split_cfg and split_cfg[k_yaml] is not None:
                try:
                    split_kwargs[k_arg] = int(split_cfg[k_yaml])
                except (TypeError, ValueError):
                    pass

        split_kwargs.update(_split_runtime_from_cfg(split_cfg))
        summary_kwargs = _summary_kwargs_from_config(self.config)
        blocks = build_knowledge_blocks(
            data,
            images_root=images_root,
            summary_title_count=int(summary_title_count),
            embedder=embedder,
            llm=llm,
            doc_title=doc_title,
            references_batch_size=int(cfg.get("references_batch_size", 5)),
            **split_kwargs,
            **summary_kwargs,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(blocks, f, ensure_ascii=False, indent=2)

        type_count: Dict[str, int] = {}
        for b in blocks:
            type_count[b["type"]] = type_count.get(b["type"], 0) + 1

        logger.info(f"共生成 {len(blocks)} 个知识块: {type_count}")
        logger.info(f"已写入: {output_path}")

        return StepResult(self.name, success=True, data={
            "output_path": output_path,
            "total_chunks": len(blocks),
            "type_count": type_count,
        })

    # ── UniParser 支路 ────────────────────────────────────────────────────

    def _run_uniparser(self, **kwargs) -> StepResult:
        cfg = self.config.chunking
        uni_cfg = (cfg.get("uniparser") or {}) if isinstance(cfg, dict) else {}
        result_path = kwargs.get("uniparser_result_path")
        output_path = kwargs.get("output_path") or cfg.get("output_path", "knowledge_blocks.json")
        summary_sim_threshold = kwargs.get("summary_sim_threshold")
        if summary_sim_threshold is None:
            summary_sim_threshold = cfg.get("summary_sim_threshold", 0.72)
        doc_title = kwargs.get("doc_title")
        if not result_path:
            uni_output_dir = (
                kwargs.get("uniparser_output_dir")
                or self.config.uniparser.get("output_dir", "uniparser_result")
            )
            result_path = autodiscover_uniparser_result(uni_output_dir)

        if not doc_title and result_path:
            stem_candidate = os.path.basename(os.path.dirname(result_path))
            if stem_candidate:
                doc_title = _sanitize_doc_title(stem_candidate)
                if doc_title != stem_candidate:
                    logger.info(
                        f"  [doc_title] 已剥编号后缀: {stem_candidate!r} -> {doc_title!r}"
                    )

        if not result_path or not os.path.exists(result_path):
            return StepResult(
                self.name, success=False,
                error="未找到 uniparser_result.json, 请先运行 parse 步骤 (--parser uniparser)",
            )

        logger.info(
            f"[chunk-uniparser] 读取: {result_path} "
            f"(summary_sim_threshold={summary_sim_threshold})"
        )
        result_json = load_uniparser_result(result_path)

        embed_cfg = self.config.embedding
        embedder = get_global_registry().get_embedder(
            api_base=embed_cfg.get("api_base", ""),
            model=embed_cfg.get("model", "model"),
            api_key=embed_cfg.get("api_key", ""),
            batch_size=embed_cfg.get("batch_size", 16),
            timeout=embed_cfg.get("timeout", 120),
            max_retries=embed_cfg.get("max_retries", 3),
            normalize=bool(embed_cfg.get("normalize", False)),
        )

        llm = _build_summary_llm(self.config)

        split_cfg = (cfg.get("semantic_split") or {}) if isinstance(cfg, dict) else {}
        split_kwargs: Dict[str, object] = {}
        for k_yaml, k_arg in (
            ("target_chars", "split_target_chars"),
            ("max_chars", "split_max_chars"),
            ("min_chars", "split_min_chars"),
            ("breakpoint_percentile", "split_breakpoint_percentile"),
        ):
            if k_yaml in split_cfg and split_cfg[k_yaml] is not None:
                try:
                    split_kwargs[k_arg] = int(split_cfg[k_yaml])
                except (TypeError, ValueError):
                    pass

        split_kwargs.update(_split_runtime_from_cfg(split_cfg))
        summary_kwargs = _summary_kwargs_from_config(self.config)
        blocks = build_knowledge_blocks_uniparser(
            result_json,
            doc_title=doc_title,
            embedder=embedder,
            llm=llm,
            references_batch_size=int(uni_cfg.get("references_batch_size", 5)),
            min_conf=float(uni_cfg.get("min_conf", 0.5)),
            source=str(uni_cfg.get("source", "pages_dict")),
            **split_kwargs,
            **summary_kwargs,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(blocks, f, ensure_ascii=False, indent=2)

        try:
            write_uniparser_meta_sidecar(output_path, result_json)
        except Exception as e:
            logger.warning(f"[chunk-uniparser] 写 meta sidecar 失败: {e}")

        type_count: Dict[str, int] = {}
        for b in blocks:
            type_count[b["type"]] = type_count.get(b["type"], 0) + 1

        logger.info(f"[chunk-uniparser] 共生成 {len(blocks)} 个知识块: {type_count}")
        logger.info(f"[chunk-uniparser] 已写入: {output_path}")
        return StepResult(self.name, success=True, data={
            "output_path": output_path,
            "total_chunks": len(blocks),
            "type_count": type_count,
            "backend": "uniparser",
        })
