"""Ingest 流程: PDF 识别 → 分块 → 向量化 → 存入 Milvus 向量数据库。

支持:
1. 指定 PDF 文件列表灌入
2. 从目录批量扫描 PDF 文件灌入
"""

from __future__ import annotations

import glob
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import Config
from ..clients.mineru import MinerUClient
from ..clients.uniparser import UniParserClient
from ..clients.client_registry import get_global_registry
from ..clients.llm import LLMClient
from ..clients.milvus import MilvusIngester, resolve_milvus_connection
from ..processors.chunker import build_knowledge_blocks, autodiscover_content_list_v2
from ..processors.uniparser_chunker import (
    build_knowledge_blocks_uniparser,
    autodiscover_uniparser_result,
    load_uniparser_result,
    write_meta_sidecar as write_uniparser_meta_sidecar,
)
from ..processors.vectorizer import vectorize_chunks

logger = logging.getLogger(__name__)


def _sort_pdfs_by_size(pdf_files: List[str]) -> List[str]:
    """按文件大小升序排列 PDF, 小文件优先解析。"""
    return sorted(
        pdf_files,
        key=lambda p: (os.path.getsize(p), os.path.basename(p).lower()),
    )


def _parse_result_exists(output_dir: str, backend: str) -> bool:
    """检查 output_dir 下是否已有对应 backend 的解析产物。"""
    if not os.path.isdir(output_dir):
        return False
    backend = (backend or "mineru").strip().lower()
    if backend == "uniparser":
        result_path = autodiscover_uniparser_result(output_dir)
    else:
        result_path = autodiscover_content_list_v2(output_dir)
    return bool(result_path and os.path.isfile(result_path))


def _summary_kwargs_from_config(config: "Config") -> Dict[str, Any]:
    """从 yaml ``chunking.summary`` 子表抽出 4 级 fallback 的参数包.

    返回值直接作为 ``build_knowledge_blocks(...)`` /
    ``build_knowledge_blocks_uniparser(...)`` 的 ``**kwargs`` 透传; 字段顺序
    与签名一一对应. 没配 summary 子表时回退到 build_knowledge_blocks 的默认值
    (chunker.py 模块级常量).
    """
    import re as _re

    chunk_cfg = config.chunking or {}
    sm = chunk_cfg.get("summary") or {}
    bm25 = sm.get("bm25") or {}
    emb = sm.get("embedding") or {}
    llm = sm.get("llm") or {}

    def _compile_list(raw):
        if raw is None:
            return None
        out: List[_re.Pattern[str]] = []
        for s in raw:
            try:
                out.append(_re.compile(s))
            except _re.error as e:
                logger.warning(f"[summary-config] 无效 regex {s!r}, 跳过: {e}")
        return out or None

    return {
        "summary_enabled": bool(sm.get("enabled", True)),
        "summary_max_sections": int(sm.get("max_summary_sections", 2)),
        "summary_title_patterns": _compile_list(sm.get("title_patterns")),
        "summary_text_patterns": _compile_list(sm.get("text_patterns")),
        "summary_stop_patterns": _compile_list(sm.get("stop_patterns")),
        # tier 2 (BM25)
        "summary_bm25_enabled": bool(bm25.get("enabled", True)),
        "summary_bm25_queries": bm25.get("query_texts"),
        "summary_bm25_threshold": float(bm25.get("threshold", 0.5)),
        # tier 3 (embedding) — 默认关 (用户反馈 "embedding 没什么用")
        "summary_embedding_enabled": bool(emb.get("enabled", False)),
        "summary_query_texts": emb.get("query_texts"),
        # 兼容老 yaml: 优先用 summary.embedding.threshold, 没配回退到老的
        # chunking.summary_sim_threshold; 都没配走 chunker 模块默认 (1.4).
        "summary_sim_threshold": float(
            emb.get("threshold", chunk_cfg.get("summary_sim_threshold", 1.4))
        ),
        # tier 4 (LLM) — 默认关 (用户要求 "都没办法采用 LLM 兜底")
        "summary_llm_enabled": bool(llm.get("enabled", False)),
        "summary_llm_max_input_chars": int(llm.get("max_input_chars", 6000)),
        **_summary_llm_call_kwargs(config),
    }


def _optional_cfg_bool(section: dict, key: str, fallback: bool) -> bool:
    """section[key] 显式设 true/false 时取用; null/缺省时用 fallback."""
    if key not in section or section[key] is None:
        return fallback
    return bool(section[key])


def _summary_llm_call_kwargs(config: "Config") -> Dict[str, Any]:
    """从 chunking.summary.llm + generation 解析 tier4 LLM 的 chat 调用参数。"""
    gen_cfg = config.generation or {}
    llm_cfg = ((config.chunking or {}).get("summary") or {}).get("llm") or {}
    return {
        "summary_llm_temperature": float(
            llm_cfg.get("temperature", gen_cfg.get("temperature", 0.0))
        ),
        "summary_llm_max_tokens": int(
            llm_cfg.get("max_tokens", gen_cfg.get("max_tokens", 1024))
        ),
        "summary_llm_disable_thinking": _optional_cfg_bool(
            llm_cfg, "disable_thinking",
            bool(gen_cfg.get("disable_thinking", True)),
        ),
        "summary_llm_system_prompt": llm_cfg.get("system_prompt"),
        "summary_llm_user_template": llm_cfg.get("user_prompt_template"),
    }


def _resolve_summary_llm_connection(config: "Config") -> Dict[str, Any]:
    """解析 tier4 摘要 LLM 的连接参数 (summary.llm 优先, 缺省回退 generation)。"""
    gen_cfg = config.generation or {}
    llm_cfg = ((config.chunking or {}).get("summary") or {}).get("llm") or {}

    def _pick(key: str, default: Any = None) -> Any:
        v = llm_cfg.get(key)
        if v is not None:
            return v
        v = gen_cfg.get(key)
        if v is not None:
            return v
        return default

    api_key = (_pick("api_key", "") or "").strip()
    return {
        "api_base": _pick("api_base", ""),
        "model": _pick("model", ""),
        "api_key": api_key,
        "timeout": int(_pick("timeout", 120)),
        "max_retries": int(_pick("max_retries", 3)),
        "disable_thinking_extra_body": _optional_cfg_bool(
            llm_cfg, "disable_thinking_extra_body",
            bool(gen_cfg.get("disable_thinking_extra_body", False)),
        ),
    }


def _build_summary_llm(config: Config) -> Optional[LLMClient]:
    """从 chunking.summary.llm (+ generation 回退) 创建摘要兜底 LLMClient。"""
    llm_cfg = ((config.chunking or {}).get("summary") or {}).get("llm") or {}
    if not bool(llm_cfg.get("enabled", False)):
        return None

    conn = _resolve_summary_llm_connection(config)
    if not conn["api_key"]:
        logger.info(
            "[chunk] chunking.summary.llm / generation.api_key 均未配置, "
            "跳过摘要 LLM 兜底 (若文档没有强摘要信号, 该文档将没有 summary chunk)"
        )
        return None
    try:
        llm = get_global_registry().get_llm(
            api_base=conn["api_base"],
            model=conn["model"],
            api_key=conn["api_key"],
            timeout=conn["timeout"],
            max_retries=conn["max_retries"],
            disable_thinking_extra_body=conn["disable_thinking_extra_body"],
        )
        logger.info(
            f"[chunk] 摘要 LLM 已就绪: model={conn['model']!r} "
            f"api_base={conn['api_base']!r} "
            f"disable_thinking_extra_body={conn['disable_thinking_extra_body']}"
        )
        return llm
    except Exception as e:
        logger.warning(f"[chunk] 摘要 LLMClient 初始化失败, 跳过 LLM 兜底: {e}")
        return None


@dataclass
class IngestStepSummary:
    """单步执行结果摘要。"""
    step: str
    success: bool
    elapsed: float = 0.0
    error: Optional[str] = None


@dataclass
class IngestResult:
    """ingest 流程的输出。"""
    file_paths: List[str] = field(default_factory=list)
    steps: List[IngestStepSummary] = field(default_factory=list)
    total_chunks: int = 0
    doc_id: Optional[str] = None


class IngestFlow:
    """PDF → 向量数据库的完整灌入流程。

    用法:
        flow = IngestFlow(config)
        result = flow.run(["论文1.pdf", "论文2.pdf"])

        # 或者从目录批量扫描
        result = flow.run_from_directory("./pdfs/")
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._steps: List[IngestStepSummary] = []
        # 批量灌入时, 集合只在第一个文档上 recreate, 后续强制 recreate=False;
        # 否则每篇 PDF 都会 drop 整个集合, 后入会覆盖前入的所有数据.
        self._collection_recreated: bool = False
        # 整个 flow 生命周期内复用同一个 MilvusIngester (= 一条 gRPC 连接),
        # 避免每篇 PDF 都新建连接产生 grpc keepalive too_many_pings 噪音 +
        # 重复跑 _ensure_collection schema 检查.
        self._ingester: Optional[MilvusIngester] = None
        # CLI rebuild/append 显式传入的 recreate 覆盖值; None 表示沿用 config.
        # 用实例字段而不是 mutate config, 避免污染下游 (如 StoreStep) 看到的配置.
        self._force_recreate: Optional[bool] = None

    def _get_ingester(self) -> MilvusIngester:
        """惰性创建单例 MilvusIngester; 第一次按 config.recreate 决定是否重建集合,
        后续复用同一个 client, 不再 recreate."""
        if self._ingester is not None:
            return self._ingester
        cfg = self.config.milvus
        # _force_recreate (CLI override) 优先于 config.milvus.recreate
        if self._force_recreate is not None:
            cfg_recreate = bool(self._force_recreate)
        else:
            cfg_recreate = bool(cfg.get("recreate", False))
        # _collection_recreated 已经为 True 时 (理论上不会, 因为 ingester 还没创建)
        # 也强制 recreate=False, 防御一下.
        recreate = cfg_recreate and not self._collection_recreated
        index_cfg = cfg.get("index", {}) or {}
        bm25_cfg = cfg.get("bm25", {}) or {}
        uri, token, db_name = resolve_milvus_connection(cfg)
        self._ingester = get_global_registry().get_milvus_ingester(
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
        if recreate:
            self._collection_recreated = True
        return self._ingester

    def _record_step(self, name: str, success: bool, elapsed: float = 0.0, error: Optional[str] = None) -> None:
        self._steps.append(IngestStepSummary(step=name, success=success, elapsed=elapsed, error=error))

    def run(self, file_paths: List[str], output_dir: Optional[str] = None, parse_timeout: Optional[int] = None) -> IngestResult:
        """完整灌入流程: parse → chunk → embed → store。

        Args:
            file_paths: PDF 文件路径列表
            output_dir: 中间产物输出目录。为 None 时使用配置文件中的路径。
                        批量处理时建议为每个 PDF 设置独立目录。
            parse_timeout: 单个 PDF 解析超时秒数, 超过则抛出 TimeoutError

        v5: backend = "uniparser" 也走完整 parse→chunk→embed→store 链;
        chunker 自动按 backend 分发到 mineru / uniparser 两条实现.
        """
        self._steps = []
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        backend = self._parsing_backend()

        # 1) 解析 PDF (已有解析产物则跳过)
        t0 = time.time()
        if output_dir and _parse_result_exists(output_dir, backend):
            logger.info(
                f"[ingest] 跳过解析 (已有结果): {file_paths} -> {output_dir}"
            )
            self._record_step("parse", True, 0.0)
            parse_output_dir = output_dir
        else:
            try:
                parse_result = self._parse(file_paths, output_dir=output_dir, timeout=parse_timeout)
                self._record_step("parse", True, time.time() - t0)
            except Exception as e:
                self._record_step("parse", False, time.time() - t0, str(e))
                return IngestResult(file_paths=file_paths, steps=self._steps)

            parse_output_dir = parse_result.get("output_dir") or output_dir or (
                self.config.uniparser.get("output_dir", "uniparser_result")
                if backend == "uniparser"
                else self.config.mineru.get("output_dir", "mineru_result")
            )

        # 用 PDF 文件名 (去后缀) 作为 doc_title, 注入为唯一的 title chunk
        doc_title = (
            os.path.splitext(os.path.basename(file_paths[0]))[0]
            if file_paths else None
        )

        # 2) 分块 (按 backend 分发)
        t0 = time.time()
        try:
            chunk_output = (
                os.path.join(output_dir, "knowledge_blocks.json")
                if output_dir else None
            )
            chunk_result = self._chunk(
                parse_output_dir,
                output_path=chunk_output,
                doc_title=doc_title,
                backend=backend,
            )
            self._record_step("chunk", True, time.time() - t0)
        except Exception as e:
            self._record_step("chunk", False, time.time() - t0, str(e))
            return IngestResult(file_paths=file_paths, steps=self._steps, total_chunks=0)

        chunk_output_path = chunk_result.get("output_path", "knowledge_blocks.json")
        total_chunks = chunk_result.get("total_chunks", 0)

        # 3) 向量化
        t0 = time.time()
        try:
            vec_output = os.path.join(output_dir, "knowledge_blocks_vec.json") if output_dir else None
            embed_result = self._embed(chunk_output_path, output_path=vec_output)
            self._record_step("embed", True, time.time() - t0)
        except Exception as e:
            self._record_step("embed", False, time.time() - t0, str(e))
            return IngestResult(file_paths=file_paths, steps=self._steps, total_chunks=total_chunks)

        vec_output_path = embed_result.get("output_path", "knowledge_blocks_vec.json")

        # 推断 doc_id / doc_name: 优先用 output_dir 目录名, 回退到 PDF 文件名
        if output_dir:
            derived_doc_id = os.path.basename(os.path.normpath(output_dir))
        else:
            derived_doc_id = os.path.splitext(os.path.basename(file_paths[0]))[0] if file_paths else None
        derived_doc_name = os.path.basename(file_paths[0]) if file_paths else derived_doc_id

        # 4) 存入 Milvus
        t0 = time.time()
        try:
            store_result = self._store(vec_output_path, doc_id=derived_doc_id, doc_name=derived_doc_name)
            self._record_step("store", True, time.time() - t0)
        except Exception as e:
            self._record_step("store", False, time.time() - t0, str(e))
            return IngestResult(file_paths=file_paths, steps=self._steps, total_chunks=total_chunks)

        doc_id = store_result.get("ingest_result", {}).get("doc_id")

        logger.info("===== Ingest 完成 =====")
        return IngestResult(
            file_paths=file_paths,
            steps=self._steps,
            total_chunks=total_chunks,
            doc_id=doc_id,
        )

    def parse_only(
        self,
        file_paths: List[str],
        output_dir: Optional[str] = None,
        parse_timeout: Optional[int] = None,
        backend: Optional[str] = None,
    ) -> IngestResult:
        """仅运行 parse 步骤 (跳过 chunk/embed/store), 把解析产物落盘并返回。

        主要用法是新增的 ``uniparser`` 支路 — 在 chunker 还没适配新 schema 的
        过渡期, 用本方法跑解析、保存 ``uniparser_result.json``, 供下一回根据
        实际产出设计 chunk 方案.

        Args:
            file_paths: PDF 文件路径列表
            output_dir: 输出目录 (None 则用各 backend 自带的默认 output_dir)
            parse_timeout: 解析整体超时秒数
            backend: 临时覆盖 parsing.backend; None 则沿用 config.
        """
        self._steps = []
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 临时覆盖 backend (恢复时还原, 避免污染同一 IngestFlow 后续调用)
        original_backend = self.config.parsing.get("backend")
        effective_backend = (backend or original_backend or "mineru").strip().lower()
        if backend:
            self.config.parsing["backend"] = backend
        try:
            t0 = time.time()
            if output_dir and _parse_result_exists(output_dir, effective_backend):
                logger.info(
                    f"[parse_only] 跳过解析 (已有结果): {file_paths} -> {output_dir}"
                )
                self._record_step("parse", True, 0.0)
            else:
                try:
                    self._parse(file_paths, output_dir=output_dir, timeout=parse_timeout)
                    self._record_step("parse", True, time.time() - t0)
                except Exception as e:
                    self._record_step("parse", False, time.time() - t0, str(e))
        finally:
            if backend:
                if original_backend is None:
                    self.config.parsing.pop("backend", None)
                else:
                    self.config.parsing["backend"] = original_backend

        return IngestResult(file_paths=file_paths, steps=self._steps)

    def parse_only_from_directory(
        self,
        directory: str,
        pattern: str = "*.pdf",
        per_file_timeout: int = 1800,
        backend: Optional[str] = None,
    ) -> List[IngestResult]:
        """从目录批量扫描 PDF, 仅跑 parse, 逐个落盘。

        每篇 PDF 一个独立子目录 (沿用 backend 自带 output_dir 作为 root):
            <backend_output_dir>/<pdf_stem>/...

        小文件优先解析; 若目标子目录已有解析产物则跳过.
        """
        pdf_files = _sort_pdfs_by_size(
            glob.glob(os.path.join(directory, "**", pattern), recursive=True)
        )
        if not pdf_files:
            logger.warning(f"目录 {directory} 中未找到匹配 {pattern} 的 PDF 文件")
            return []

        # 临时覆盖 backend 以决定 output_dir 默认值
        original_backend = self.config.parsing.get("backend")
        effective_backend = (backend or original_backend or "mineru").strip().lower()
        base_output_dir = (
            self.config.uniparser.get("output_dir", "uniparser_result")
            if effective_backend == "uniparser"
            else self.config.mineru.get("output_dir", "mineru_result")
        )

        logger.info(
            f"[parse_only] 发现 {len(pdf_files)} 个 PDF (小文件优先), "
            f"backend={effective_backend}, base_output_dir={base_output_dir}"
        )
        results: List[IngestResult] = []
        skipped: List[str] = []
        already_parsed: List[str] = []
        for pdf in pdf_files:
            pdf_stem = os.path.splitext(os.path.basename(pdf))[0]
            output_dir = os.path.join(base_output_dir, pdf_stem)
            if _parse_result_exists(output_dir, effective_backend):
                size_kb = os.path.getsize(pdf) / 1024
                logger.info(
                    f"[parse_only] 跳过 (已有解析结果, {size_kb:.1f} KB): {pdf}"
                )
                already_parsed.append(pdf)
                results.append(
                    IngestResult(
                        file_paths=[pdf],
                        steps=[IngestStepSummary(step="parse", success=True, elapsed=0.0)],
                    )
                )
                continue
            logger.info(f"\n{'='*60}\n[parse_only] 处理: {pdf}\n{'='*60}")
            try:
                r = self.parse_only(
                    [pdf],
                    output_dir=output_dir,
                    parse_timeout=per_file_timeout,
                    backend=backend,
                )
                results.append(r)
            except TimeoutError:
                logger.warning(f"跳过 (超时 {per_file_timeout}s): {pdf}")
                skipped.append(pdf)
            except Exception as e:
                logger.warning(f"跳过 (解析失败): {pdf} - {e}")
                skipped.append(pdf)

        if already_parsed:
            print(f"\n{'='*60}")
            print(f"以下 {len(already_parsed)} 个文件已有解析结果, 已跳过:")
            for f in already_parsed:
                print(f"  - {f}")
            print(f"{'='*60}")
        if skipped:
            print(f"\n{'='*60}")
            print(f"以下 {len(skipped)} 个文件未成功解析:")
            for f in skipped:
                print(f"  - {f}")
            print(f"{'='*60}")
        return results

    def run_from_directory(self, directory: str, pattern: str = "*.pdf", per_file_timeout: int = 60) -> List[IngestResult]:
        """从目录批量扫描 PDF 文件并逐个灌入。

        每个 PDF 的中间产物 (解析结果、chunks、vectors) 保存在以 PDF 名称命名的子目录中。
        单个 PDF 解析超过 per_file_timeout 秒则跳过, 继续处理下一个。
        小文件优先处理; 若目标子目录已有解析产物则跳过 parse, 继续 chunk/embed/store.

        目录结构示例:
            mineru_result/
              论文1/
                论文1/                    <- MinerU 解压结果
                  *_content_list_v2.json
                knowledge_blocks.json     <- chunks
                knowledge_blocks_vec.json <- vectors
        """
        pdf_files = _sort_pdfs_by_size(
            glob.glob(os.path.join(directory, "**", pattern), recursive=True)
        )
        if not pdf_files:
            logger.warning(f"目录 {directory} 中未找到匹配 {pattern} 的 PDF 文件")
            return []

        logger.info(
            f"发现 {len(pdf_files)} 个 PDF 文件 (小文件优先): {pdf_files}"
        )
        base_output_dir = self.config.mineru.get("output_dir", "mineru_result")
        results: List[IngestResult] = []
        skipped: List[str] = []
        for pdf in pdf_files:
            pdf_stem = os.path.splitext(os.path.basename(pdf))[0]
            output_dir = os.path.join(base_output_dir, pdf_stem)
            logger.info(f"\n{'='*60}\n处理: {pdf}\n{'='*60}")
            try:
                result = self.run([pdf], output_dir=output_dir, parse_timeout=per_file_timeout)
                results.append(result)
            except TimeoutError:
                logger.warning(f"跳过 (超时 {per_file_timeout}s): {pdf}")
                skipped.append(pdf)
            except Exception as e:
                logger.warning(f"跳过 (解析失败): {pdf} - {e}")
                skipped.append(pdf)

        if skipped:
            print(f"\n{'='*60}")
            print(f"以下 {len(skipped)} 个文件未成功解析:")
            for f in skipped:
                print(f"  - {f}")
            print(f"{'='*60}")

        return results

    def vectorize_from_directory(
        self, directory: str, recreate: Optional[bool] = None,
        skip_existing: bool = True,
        progress_callback: Optional[Any] = None,
    ) -> List[IngestResult]:
        """从 MinerU 解析结果目录批量做 chunk → embed → store (跳过 PDF 解析)。

        扫描目录中所有 *_content_list_v2.json 文件, 每个 json 对应一篇文献,
        执行分块、向量化、存入 Milvus。

        Args:
            directory: MinerU 解析结果根目录 (如 mineru_result/)
            recreate: 是否在第一篇文献入库前清空整个集合.
                - True: rebuild 模式, 强制 drop 集合然后重建, 之前所有文档清空.
                - False: append 模式, 不清空, 直接追加 (会按 doc_id 覆盖同名文档).
                - None: 沿用 config.milvus.recreate.
            skip_existing: append 模式下是否跳过集合中已存在的 doc_id。
                - True (默认): 自动跳过已有数据, 避免重复 chunk/embed/store.
                - False: 不跳过, 同名 doc_id 会被覆盖重灌。
            progress_callback: 可选回调, 每处理完一篇文档调用
                callback(current, total, doc_id, status)

        Returns:
            每篇文献的 IngestResult 列表
        """
        # CLI 显式传入的 recreate 优先级最高; 用实例字段保存, 不污染 config,
        # 避免下游 (如 stats / StoreStep) 看到被改过的配置后误删数据.
        if recreate is not None:
            self._force_recreate = bool(recreate)
        # 同时扫描 MinerU 的 content_list_v2.json 和 UniParser 的 uniparser_result.json,
        # 自动按文件名识别 backend 并分发到对应 chunker.
        content_files: List[tuple] = []  # (path, backend)
        for pattern in ["*_content_list_v2.json", "content_list_v2.json"]:
            for p in glob.glob(os.path.join(directory, "**", pattern), recursive=True):
                content_files.append((p, "mineru"))
        for p in glob.glob(os.path.join(directory, "**", "uniparser_result.json"), recursive=True):
            content_files.append((p, "uniparser"))
        # 去重 + 排序 (path 升序)
        content_files = sorted({(p, b) for p, b in content_files})

        if not content_files:
            logger.warning(
                f"目录 {directory} 中未找到 content_list_v2.json / uniparser_result.json"
            )
            return []

        backends_seen = sorted({b for _, b in content_files})
        logger.info(
            f"发现 {len(content_files)} 个解析结果文件 (backends: {backends_seen})"
        )

        # ── append 模式下查询已有 doc_id, 自动跳过 ────────────────────────
        existing_doc_ids: set = set()
        is_append = not (self._force_recreate if self._force_recreate is not None
                         else self.config.milvus.get("recreate", False))
        if is_append and skip_existing:
            try:
                ingester = self._get_ingester()
                existing_doc_ids = ingester.list_doc_ids()
            except Exception as e:
                logger.warning(
                    f"[append] 查询已有 doc_id 失败, 无法跳过已有数据: {e}"
                )

        results: List[IngestResult] = []
        skipped: List[str] = []
        skipped_existing: List[str] = []
        for content_path, backend in content_files:
            content_dir = os.path.dirname(content_path)
            parent_dir = os.path.dirname(content_dir)
            if backend == "uniparser":
                # UniParser 布局: uniparser_result/<pdf_stem>/uniparser_result.json
                output_dir = content_dir
            else:
                # MinerU 布局: mineru_result/<pdf_stem>/<pdf_stem>/{uuid}_content_list_v2.json
                # 输出到 mineru_result/<pdf_stem>/
                if os.path.basename(parent_dir) == os.path.basename(content_dir):
                    output_dir = parent_dir
                else:
                    output_dir = content_dir

            # 检查 doc_id 是否已存在, 跳过已灌入的文档
            derived_doc_id = os.path.basename(os.path.normpath(output_dir))
            if existing_doc_ids and derived_doc_id in existing_doc_ids:
                logger.info(
                    f"[append] 跳过已存在的 doc_id={derived_doc_id!r}: {content_path}"
                )
                skipped_existing.append(content_path)
                results.append(
                    IngestResult(
                        steps=[IngestStepSummary(step="skip", success=True, elapsed=0.0)],
                        doc_id=derived_doc_id,
                    )
                )
                if progress_callback:
                    try:
                        progress_callback(len(results), len(content_files), derived_doc_id, "skipped")
                    except Exception:
                        pass
                continue

            logger.info(
                f"\n{'='*60}\n处理 [{backend}]: {content_path}\n{'='*60}"
            )

            try:
                result = self._vectorize_single(content_path, output_dir, backend=backend)
                results.append(result)
                if progress_callback:
                    try:
                        progress_callback(len(results), len(content_files), derived_doc_id, "done")
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"跳过 (处理失败): {content_path} - {e}")
                skipped.append(content_path)
                if progress_callback:
                    try:
                        progress_callback(len(results), len(content_files), derived_doc_id, "failed")
                    except Exception:
                        pass

        if skipped_existing:
            print(f"\n{'='*60}")
            print(f"以下 {len(skipped_existing)} 个文档已存在于向量库, 已跳过:")
            for f in skipped_existing:
                print(f"  - {f}")
            print(f"{'='*60}")
        if skipped:
            print(f"\n{'='*60}")
            print(f"以下 {len(skipped)} 个文件未成功处理:")
            for f in skipped:
                print(f"  - {f}")
            print(f"{'='*60}")

        return results

    def _vectorize_single(
        self, content_path: str, output_dir: str, backend: str = "mineru",
    ) -> IngestResult:
        """对单个解析产物 (content_list_v2.json / uniparser_result.json) 执行
        chunk → embed → store, 按 backend 分发."""
        self._steps = []
        os.makedirs(output_dir, exist_ok=True)

        derived_doc_title = os.path.basename(os.path.normpath(output_dir))

        t0 = time.time()
        try:
            chunk_output = os.path.join(output_dir, "knowledge_blocks.json")
            # 传入解析产物所在目录, _chunk_mineru/_chunk_uniparser 内部会 autodiscover
            chunk_result = self._chunk(
                os.path.dirname(content_path),
                output_path=chunk_output,
                doc_title=derived_doc_title,
                backend=backend,
            )
            self._record_step("chunk", True, time.time() - t0)
        except Exception as e:
            self._record_step("chunk", False, time.time() - t0, str(e))
            return IngestResult(steps=self._steps)

        chunk_output_path = chunk_result["output_path"]
        total_chunks = chunk_result["total_chunks"]

        # embed
        t0 = time.time()
        try:
            vec_output = os.path.join(output_dir, "knowledge_blocks_vec.json")
            embed_result = self._embed(chunk_output_path, output_path=vec_output)
            self._record_step("embed", True, time.time() - t0)
        except Exception as e:
            self._record_step("embed", False, time.time() - t0, str(e))
            return IngestResult(steps=self._steps, total_chunks=total_chunks)

        vec_output_path = embed_result["output_path"]

        # store
        derived_doc_id = os.path.basename(os.path.normpath(output_dir))
        derived_doc_name = derived_doc_id
        t0 = time.time()
        try:
            store_result = self._store(vec_output_path, doc_id=derived_doc_id, doc_name=derived_doc_name)
            self._record_step("store", True, time.time() - t0)
        except Exception as e:
            self._record_step("store", False, time.time() - t0, str(e))
            return IngestResult(steps=self._steps, total_chunks=total_chunks)

        doc_id = store_result.get("ingest_result", {}).get("doc_id")
        return IngestResult(steps=self._steps, total_chunks=total_chunks, doc_id=doc_id)

    # ── 内部步骤 ─────────────────────────────────────────────────────────

    def _parsing_backend(self) -> str:
        """读取 parsing.backend, 缺省回退 mineru。"""
        parsing_cfg = self.config.parsing or {}
        backend = (parsing_cfg.get("backend") or "mineru").strip().lower()
        if backend not in ("mineru", "uniparser"):
            raise ValueError(
                f"未知的 parsing.backend={backend!r}, 仅支持 mineru / uniparser"
            )
        return backend

    def _parse(self, file_paths: List[str], output_dir: Optional[str] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        """步骤 1: 调用解析 API (MinerU 或 UniParser, 由 parsing.backend 决定)。"""
        backend = self._parsing_backend()
        if backend == "uniparser":
            return self._parse_uniparser(file_paths, output_dir=output_dir, timeout=timeout)
        return self._parse_mineru(file_paths, output_dir=output_dir, timeout=timeout)

    def _parse_mineru(self, file_paths: List[str], output_dir: Optional[str] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        """支路 A (默认): MinerU API → mineru_result/<pdf_stem>/<pdf_stem>/..."""
        cfg = self.config.mineru
        mineru_output_dir = output_dir or cfg.get("output_dir", "mineru_result")
        client = MinerUClient(
            api_url=cfg.get("api_url", MinerUClient.DEFAULT_API_URL),
            authorization=cfg.get("authorization", ""),
            model_version=cfg.get("model_version", "vlm"),
            output_dir=mineru_output_dir,
            poll_max_retries=cfg.get("poll", {}).get("max_retries", 120),
            poll_interval=cfg.get("poll", {}).get("interval", 5),
        )
        result = client.process(file_paths, timeout=timeout)
        result["backend"] = "mineru"
        return result

    def _parse_uniparser(self, file_paths: List[str], output_dir: Optional[str] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        """支路 B: UniParser API → uniparser_result/<pdf_stem>/uniparser_result.json。

        注: 输出 schema 与 MinerU 完全不同, 后续 chunker 需要单独适配 (待办).
        """
        cfg = self.config.uniparser
        if not cfg.get("api_key"):
            raise RuntimeError(
                "uniparser.api_key 未配置 (建议 export UNIPARSER_API_KEY=...)"
            )
        uniparser_output_dir = output_dir or cfg.get("output_dir", "uniparser_result")
        client = UniParserClient(
            host=cfg.get("host", UniParserClient.DEFAULT_HOST),
            api_key=cfg.get("api_key", ""),
            output_dir=uniparser_output_dir,
            parse_modes=cfg.get("parse_modes"),
            output_flags=cfg.get("output_flags"),
            format_flags=cfg.get("format_flags"),
            sync=bool(cfg.get("sync", True)),
            request_timeout=int(cfg.get("request_timeout", 1800)),
            poll_max_retries=int((cfg.get("poll") or {}).get("max_retries", 120)),
            poll_interval=int((cfg.get("poll") or {}).get("interval", 5)),
        )
        result = client.process(file_paths, timeout=timeout)
        result["backend"] = "uniparser"
        return result

    def _chunk(
        self, parse_output_dir: str, output_path: Optional[str] = None,
        doc_title: Optional[str] = None, backend: Optional[str] = None,
    ) -> Dict[str, Any]:
        """步骤 2 dispatcher: 按 backend 分发到 mineru / uniparser chunker."""
        backend = (backend or self._parsing_backend()).strip().lower()
        if backend == "uniparser":
            return self._chunk_uniparser(
                parse_output_dir, output_path=output_path, doc_title=doc_title,
            )
        return self._chunk_mineru(
            parse_output_dir, output_path=output_path, doc_title=doc_title,
        )

    def _chunk_mineru(
        self, mineru_output_dir: str, output_path: Optional[str] = None,
        doc_title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """步骤 2 (MinerU 支路): 将 MinerU 解析结果转为知识块。"""
        cfg = self.config.chunking
        output_path = output_path or cfg.get("output_path", "knowledge_blocks.json")
        summary_title_count = cfg.get("summary_title_count", 2)
        summary_sim_threshold = cfg.get("summary_sim_threshold", 0.72)
        split_cfg = cfg.get("semantic_split", {}) or {}

        content_list_path = cfg.get("content_list_path") or autodiscover_content_list_v2(mineru_output_dir)
        if not content_list_path or not os.path.exists(content_list_path):
            raise FileNotFoundError("未找到 content_list_v2.json, 请先运行 parse 步骤")

        with open(content_list_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list) or (data and not isinstance(data[0], list)):
            raise ValueError("content_list_v2.json 应是二维 list (按页分组), 实际格式不符")

        emb_cfg = self.config.embedding
        embedder = get_global_registry().get_embedder(
            api_base=emb_cfg.get("api_base", ""),
            model=emb_cfg.get("model", "model"),
            api_key=emb_cfg.get("api_key", ""),
            batch_size=emb_cfg.get("batch_size", 16),
            timeout=emb_cfg.get("timeout", 120),
            max_retries=emb_cfg.get("max_retries", 3),
            normalize=bool(emb_cfg.get("normalize", False)),
        )

        images_root = os.path.dirname(content_list_path)
        summary_kwargs = _summary_kwargs_from_config(self.config)
        # 优先用 summary 子表的阈值; 老字段 summary_sim_threshold 已被并到
        # summary_kwargs 里, 这里不再单独传以避免重复.
        blocks = build_knowledge_blocks(
            data,
            images_root=images_root,
            summary_title_count=int(summary_title_count),
            embedder=embedder,
            llm=_build_summary_llm(self.config),
            doc_title=doc_title,
            split_target_chars=int(split_cfg.get("target_chars", 1200)),
            split_max_chars=int(split_cfg.get("max_chars", 2000)),
            split_min_chars=int(split_cfg.get("min_chars", 300)),
            split_breakpoint_percentile=int(split_cfg.get("breakpoint_percentile", 85)),
            references_batch_size=int(cfg.get("references_batch_size", 5)),
            **summary_kwargs,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(blocks, f, ensure_ascii=False, indent=2)

        type_count: Dict[str, int] = {}
        for b in blocks:
            type_count[b["type"]] = type_count.get(b["type"], 0) + 1

        logger.info(f"共生成 {len(blocks)} 个知识块: {type_count}")
        return {
            "output_path": output_path,
            "total_chunks": len(blocks),
            "type_count": type_count,
        }

    # ----- 向后兼容别名: 老调用方仍可叫 _chunk_from_file (MinerU 单文件版) -----
    _chunk_from_file = None  # type: ignore

    def _chunk_uniparser(
        self, uniparser_output_dir: str, output_path: Optional[str] = None,
        doc_title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """步骤 2 (UniParser 支路): 把 uniparser_result.json 转为知识块。

        与 MinerU 同形输出 + 同样的 ``knowledge_blocks_meta.json`` sidecar,
        下游 vectorize / store 无差异. UniParser 专属配置 (min_conf /
        references_batch_size) 从 ``chunking.uniparser`` 子表读取, 没配置就走默认.
        """
        cfg = self.config.chunking
        output_path = output_path or cfg.get("output_path", "knowledge_blocks.json")
        summary_sim_threshold = cfg.get("summary_sim_threshold", 0.72)
        split_cfg = cfg.get("semantic_split", {}) or {}
        uni_cfg = cfg.get("uniparser", {}) or {}

        result_path = autodiscover_uniparser_result(uniparser_output_dir)
        if not result_path or not os.path.exists(result_path):
            raise FileNotFoundError(
                f"未找到 uniparser_result.json (搜索目录: {uniparser_output_dir})"
            )

        result_json = load_uniparser_result(result_path)

        emb_cfg = self.config.embedding
        embedder = get_global_registry().get_embedder(
            api_base=emb_cfg.get("api_base", ""),
            model=emb_cfg.get("model", "model"),
            api_key=emb_cfg.get("api_key", ""),
            batch_size=emb_cfg.get("batch_size", 16),
            timeout=emb_cfg.get("timeout", 120),
            max_retries=emb_cfg.get("max_retries", 3),
            normalize=bool(emb_cfg.get("normalize", False)),
        )

        summary_kwargs = _summary_kwargs_from_config(self.config)
        blocks = build_knowledge_blocks_uniparser(
            result_json,
            doc_title=doc_title,
            embedder=embedder,
            llm=_build_summary_llm(self.config),
            split_target_chars=int(split_cfg.get("target_chars", 1200)),
            split_max_chars=int(split_cfg.get("max_chars", 2000)),
            split_min_chars=int(split_cfg.get("min_chars", 300)),
            split_breakpoint_percentile=int(split_cfg.get("breakpoint_percentile", 85)),
            references_batch_size=int(uni_cfg.get("references_batch_size", 5)),
            min_conf=float(uni_cfg.get("min_conf", 0.5)),
            source=str(uni_cfg.get("source", "pages_dict")),
            **summary_kwargs,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(blocks, f, ensure_ascii=False, indent=2)

        # 写 sidecar meta (让 MilvusIngester._load_meta_sidecar 拿到 source/year 等元信息);
        # doc_id / doc_name 在 _store 阶段还会被 output_dir 名覆盖, 这里不强填.
        try:
            write_uniparser_meta_sidecar(output_path, result_json)
        except Exception as e:
            logger.warning(f"[chunk-uniparser] 写 meta sidecar 失败: {e}")

        type_count: Dict[str, int] = {}
        for b in blocks:
            type_count[b["type"]] = type_count.get(b["type"], 0) + 1
        logger.info(
            f"[chunk-uniparser] 共生成 {len(blocks)} 个知识块: {type_count}"
        )
        return {
            "output_path": output_path,
            "total_chunks": len(blocks),
            "type_count": type_count,
        }

    def _embed(self, input_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """步骤 3: 对知识块进行向量化。"""
        cfg = self.config.embedding
        output_path = output_path or cfg.get("output_path", "knowledge_blocks_vec.json")

        with open(input_path, "r", encoding="utf-8") as f:
            chunks: List[Dict[str, Any]] = json.load(f)
        logger.info(f"读取 {len(chunks)} 个 chunks: {input_path}")

        embedder = get_global_registry().get_embedder(
            api_base=cfg.get("api_base", ""),
            model=cfg.get("model", "model"),
            api_key=cfg.get("api_key", ""),
            batch_size=cfg.get("batch_size", 16),
            timeout=cfg.get("timeout", 120),
            max_retries=cfg.get("max_retries", 3),
            normalize=bool(cfg.get("normalize", False)),
        )

        vectorized = vectorize_chunks(chunks, embedder, max_chars=cfg.get("max_chars", 8000))

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(vectorized, f, ensure_ascii=False, indent=2)

        dim = vectorized[0]["embedding_dim"] if vectorized else 0
        logger.info(f"已写入 {len(vectorized)} 个带向量 chunks: {output_path}, dim={dim}")

        return {
            "output_path": output_path,
            "total": len(vectorized),
            "embedding_dim": dim,
        }

    def _store(self, input_path: str, doc_id: Optional[str] = None, doc_name: Optional[str] = None) -> Dict[str, Any]:
        """步骤 4: 将向量化的知识块灌入 Milvus。"""
        cfg = self.config.milvus
        doc_id = doc_id or cfg.get("doc_id")
        doc_name = doc_name or cfg.get("doc_name")
        ingester = self._get_ingester()

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

        return {
            "ingest_result": result,
            "stats": stat,
        }
