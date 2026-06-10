"""Pipeline 编排器: 将解析→分块→向量化→存储→检索→生成 串联为可配置流水线。

支持两种使用模式:
1. 编程式: pipeline.ingest(files) → pipeline.query("问题")
2. 单步式: pipeline.run_step("chunk", ...)

使用 pipeline.flows 提供的 IngestFlow 和 QueryFlow。
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from .clients.client_registry import ClientRegistry, set_global_registry
from .config import Config, load_config
from .steps import BaseStep, StepResult, get_step, list_steps
from .flows import IngestFlow, QueryFlow
from .flows.ingest import IngestResult
from .flows.query import ChatSession
from .models import QueryResult

logger = logging.getLogger(__name__)


# 已知"通用"文件名 (本身不包含文献信息, 必须看父目录推断 doc_id).
# 配合 MinerU 默认布局: mineru_result/<paper_title>/knowledge_blocks*.json
_GENERIC_VEC_BASENAMES = {
    "knowledge_blocks",
    "chunks",
    "vectors",
    "data",
    "blocks",
}


def _derive_doc_meta_from_path(path: str):
    """从文件路径推断 (doc_id, doc_name); 不适用时返回 (None, None)。

    规则: 文件名去 _vec/_vectors/_embedded 后缀, 若落到通用文件名 (见
    ``_GENERIC_VEC_BASENAMES``), 则改用父目录名作为 doc_id / doc_name。
    其它情况返回 (None, None), 交给 ``MilvusIngester.ingest_file`` 内置推断。
    """
    import os as _os
    base = _os.path.splitext(_os.path.basename(path))[0]
    for suffix in ("_vec", "_vectors", "_embedded"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    if base.lower() in _GENERIC_VEC_BASENAMES:
        parent = _os.path.basename(
            _os.path.dirname(_os.path.abspath(path))
        )
        # 排除根目录 / 空 / 仅是 mineru_result 顶级的情况
        if parent and parent not in (".", "/", "..", "mineru_result"):
            return parent, parent
    return None, None


class Pipeline:
    """端到端 RAG 流水线。

    用法:
        from pipeline import Pipeline

        pipe = Pipeline()                      # 使用默认配置
        pipe = Pipeline("my_config.yaml")      # 使用自定义配置
        pipe = Pipeline(overrides={"generation": {"temperature": 0.5}})

        # 从已解析的 MinerU 目录灌入 — 两种模式二选一:
        pipe.rebuild("./mineru_result/")   # 清空集合 + 全量重灌
        pipe.append("./mineru_result/")    # 增量追加, 同名 doc_id 会被覆盖

        # 查询
        result = pipe.query("MoS2 的晶格常数是多少?")

        # 单步执行 (例: 单独跑一次 PDF 解析)
        pipe.run_step("parse", files=["论文.pdf"])
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        overrides: Optional[Dict] = None,
    ) -> None:
        self.config = load_config(config_path, overrides)
        # 单进程内共享的客户端连接池 (EmbeddingClient/LLMClient/MilvusIngester);
        # 注册到全局, 让 Steps / Flows 即便不持有 Pipeline 句柄也能命中同一缓存.
        self.clients = ClientRegistry()
        set_global_registry(self.clients)
        # 同时挂到 config 上, 方便 Step.run() 通过 self.config.clients 取用
        # (Config 是普通 dict 容器, 直接 setattr 不会破坏 _data 序列化).
        setattr(self.config, "clients", self.clients)
        self._step_cache: Dict[str, BaseStep] = {}
        self._results: List[StepResult] = []
        # 懒加载: 复用 flow 实例, 避免每次 query/ingest 都重建底层连接
        self._query_flow: Optional[QueryFlow] = None
        self._ingest_flow: Optional[IngestFlow] = None
        # 集合切换追踪: 记录 QueryFlow 当前绑定的集合名, 切换时清空缓存
        self._active_collection: Optional[str] = None
        # 原始默认集合名: collection=None 时一律回退到它。
        # 不能用 config.milvus["collection"], 因为该字段会被 _maybe_switch_collection
        # 覆盖为最近一次显式指定的集合; 否则"默认知识库"会被污染成上次用过的库。
        self._default_collection: str = self.config.milvus.get(
            "collection", "literature_chunks"
        )

    # ── flow 缓存 ─────────────────────────────────────────────────────

    def _maybe_switch_collection(self, collection: Optional[str]) -> None:
        """切换目标集合: 若与当前 QueryFlow 绑定的集合不同, 更新 config 并清空缓存。

        collection=None / 空 时回退到 *原始* 默认集合 (self._default_collection),
        而非 config 里被上一次切换覆盖过的值, 避免"默认知识库"被污染成上次用过的库。
        """
        effective = collection or self._default_collection
        if self._active_collection is not None and self._active_collection != effective:
            self.config.milvus["collection"] = effective
            if self._query_flow is not None:
                self._query_flow.invalidate_caches()
        elif self._active_collection is None:
            self.config.milvus["collection"] = effective
        self._active_collection = effective

    def _get_query_flow(self) -> QueryFlow:
        if self._query_flow is None:
            self._query_flow = QueryFlow(self.config)
        return self._query_flow

    def _get_ingest_flow(self) -> IngestFlow:
        if self._ingest_flow is None:
            self._ingest_flow = IngestFlow(self.config)
        return self._ingest_flow

    # ── 单步执行 ──────────────────────────────────────────────────────

    def _get_step(self, name: str) -> BaseStep:
        if name not in self._step_cache:
            cls = get_step(name)
            self._step_cache[name] = cls(self.config)
        return self._step_cache[name]

    def run_step(self, name: str, **kwargs) -> StepResult:
        step = self._get_step(name)
        result = step._execute(**kwargs)
        self._results.append(result)
        return result

    # ── 解析流程 (parse-only, 支持 mineru / uniparser 两条支路) ────────

    def parse(
        self,
        file_paths: List[str],
        output_dir: Optional[str] = None,
        parse_timeout: Optional[int] = None,
        backend: Optional[str] = None,
    ) -> IngestResult:
        """仅运行 parse 步骤, 落盘后即返回 (不做 chunk/embed/store)。

        主要用于:
        - 新增的 ``uniparser`` 支路: 在新 chunker 还没写之前, 先把解析结果
          落到 ``uniparser_result/<pdf_stem>/uniparser_result.json``,
          供下回根据实际 schema 设计 chunk 方案.
        - MinerU 支路单跑解析也可以走这里 (跳过下游).

        Args:
            file_paths: PDF 文件路径列表
            output_dir: 中间产物输出目录 (None 则用 backend 自带默认)
            parse_timeout: 解析整体超时秒数
            backend: 临时覆盖 parsing.backend, 取值 mineru / uniparser
        """
        return self._get_ingest_flow().parse_only(
            file_paths,
            output_dir=output_dir,
            parse_timeout=parse_timeout,
            backend=backend,
        )

    def parse_directory(
        self,
        directory: str,
        pattern: str = "*.pdf",
        per_file_timeout: int = 1800,
        backend: Optional[str] = None,
    ) -> List[IngestResult]:
        """从目录批量扫描 PDF, 仅跑 parse, 逐个落盘到 backend 默认 output_dir。"""
        return self._get_ingest_flow().parse_only_from_directory(
            directory,
            pattern=pattern,
            per_file_timeout=per_file_timeout,
            backend=backend,
        )

    # ── 灌入流程 (从 MinerU 解析结果目录: chunk → embed → store) ──────
    # PDF 解析单独走 step parse, 不再混在 ingest 入口里.

    def rebuild(self, directory: str) -> List[IngestResult]:
        """rebuild: 清空集合后, 从 MinerU 解析结果目录批量重灌。

        会先 drop 整个 Milvus 集合, 再扫描 directory 下所有
        ``*_content_list_v2.json`` 逐篇灌入. 适合: 切换 schema, 重置数据,
        重新调整 chunk 策略后重灌等场景.

        Args:
            directory: MinerU 解析结果根目录 (如 mineru_result/)
        """
        return self._get_ingest_flow().vectorize_from_directory(directory, recreate=True)

    def append(self, directory: str, skip_existing: bool = True) -> List[IngestResult]:
        """append: 增量追加, 不清空集合。

        扫描 directory 下所有 ``*_content_list_v2.json`` 逐篇灌入. 同名
        doc_id (默认是 PDF 文件名去后缀) 会被覆盖, 其它文献保持不变.
        默认自动跳过集合中已存在的 doc_id, 避免重复 chunk/embed/store.

        Args:
            directory: MinerU 解析结果根目录 (如 mineru_result/)
            skip_existing: 是否跳过集合中已存在的 doc_id (默认 True)。
                设为 False 则强制重灌已有文档 (同名 doc_id 会被覆盖)。
        """
        return self._get_ingest_flow().vectorize_from_directory(
            directory, recreate=False, skip_existing=skip_existing,
        )

    def load_vec(
        self,
        path_or_glob: str,
        recreate: bool = False,
        purge_existing: bool = True,
        skip_existing: bool = False,
    ) -> List[Dict[str, Any]]:
        """直接灌入已向量化的 ``*_vec.json``, 跳过 parse / chunk / embed。

        适用场景: 已经在另一台机器或较早跑过 chunk + embedding, 现在只想把
        这些块批量推到某个 Milvus 实例 (例如 docker-compose 起的
        ``http://localhost:19530``)。配合 ``--milvus-backend server`` 使用。

        Args:
            path_or_glob:
              - 目录: 递归扫描 ``**/*_vec.json``
              - glob 模式 (含 ``*`` / ``?`` / ``[``): 直接 ``glob.glob`` 展开
              - 单个 ``.json`` 文件: 直接灌入
            recreate: ``True`` 则先 drop 整个集合 (慎用, 会清空已有数据).
                默认 ``False`` 走 append 语义.
            purge_existing: ``True`` 时灌入前按 ``doc_id`` 删除集合内同名文档
                (覆盖更新). 默认 ``True``.
            skip_existing: ``True`` 时跳过 Milvus 中已存在的 ``doc_id`` (增量追加).

        Returns:
            每个成功灌入文件的结果 dict 列表 (含 doc_id / count / type_count 等).
        """
        import glob as _glob
        import os as _os

        from .clients.milvus import resolve_milvus_connection, _is_transient_rpc_error

        cfg = self.config.milvus
        index_cfg = cfg.get("index", {}) or {}
        bm25_cfg = cfg.get("bm25", {}) or {}
        uri, token, db_name = resolve_milvus_connection(cfg)
        collection = cfg.get("collection", "literature_chunks")
        dim = int(cfg.get("dim", 1024))

        def _make_ingester(*, use_recreate: bool = False):
            return self.clients.get_milvus_ingester(
                uri=uri,
                token=token,
                db_name=db_name,
                collection=collection,
                dim=dim,
                recreate=use_recreate,
                analyzer_params=bm25_cfg.get("analyzer") or None,
                dense_index_type=str(index_cfg.get("dense_type", "AUTOINDEX")),
                dense_metric=str(index_cfg.get("dense_metric", "IP")),
                dense_index_params=index_cfg.get("dense_params") or None,
            )

        # 复用 ClientRegistry: 同一 (uri, collection, dim) 的 ingester 不重复创建,
        # 避免 _ensure_collection / describe_collection 在批量灌入时被反复触发.
        ingester = _make_ingester(use_recreate=recreate)
        batch_size = int(cfg.get("batch_size", 100))

        existing_doc_ids: set = set()
        if skip_existing and not recreate:
            try:
                existing_doc_ids = ingester.list_doc_ids()
            except Exception as e:
                logger.warning(
                    f"[load-vec] 查询已有 doc_id 失败, 无法跳过: {e}"
                )

        # 路径归一化: dir / glob / single file
        if _os.path.isdir(path_or_glob):
            pattern = _os.path.join(path_or_glob, "**", "*_vec.json")
            paths = sorted(_glob.glob(pattern, recursive=True))
        elif any(c in path_or_glob for c in "*?["):
            paths = sorted(_glob.glob(path_or_glob, recursive=True))
        elif _os.path.isfile(path_or_glob):
            paths = [path_or_glob]
        else:
            paths = []

        if not paths:
            logger.warning(f"[load-vec] 未找到任何 *_vec.json: {path_or_glob}")
            return []

        logger.info(f"[load-vec] 即将灌入 {len(paths)} 个 *_vec.json 文件")
        results: List[Dict[str, Any]] = []
        skipped: List[tuple] = []
        seen_doc_ids: Dict[str, str] = {}
        for i, p in enumerate(paths, 1):
            logger.info(f"\n[{i}/{len(paths)}] >>> {p}")
            # MinerU 默认布局是 `<paper_title>/knowledge_blocks_vec.json`,
            # 这种通用文件名靠 `ingest_file` 内置 (文件名 -> doc_id) 会把
            # 所有文件都映射成同一个 doc_id, 互相 purge 把数据洗光.
            # 这里按父目录名推断 doc_id, 还原 IngestFlow._vectorize_single
            # 的语义.
            doc_id_override, doc_name_override = _derive_doc_meta_from_path(p)
            effective_doc_id = doc_id_override
            if not effective_doc_id:
                # 与 ingest_file 一致: 无 override 时用文件名 stem
                base = _os.path.splitext(_os.path.basename(p))[0]
                for suffix in ("_vec", "_vectors", "_embedded"):
                    if base.endswith(suffix):
                        base = base[: -len(suffix)]
                        break
                effective_doc_id = base
            if (
                skip_existing
                and effective_doc_id
                and effective_doc_id in existing_doc_ids
            ):
                logger.info(
                    f"[load-vec] 跳过已存在 doc_id={effective_doc_id!r}: {p}"
                )
                continue
            if doc_id_override and doc_id_override in seen_doc_ids:
                prev = seen_doc_ids[doc_id_override]
                logger.warning(
                    f"[load-vec] doc_id={doc_id_override!r} 已被 {prev!r} 占用, "
                    f"当前 {p!r} 会覆盖前者; 请确认两者是同一篇文献"
                )
            ingested = False
            last_err: Optional[Exception] = None
            for attempt in range(2):
                try:
                    r = ingester.ingest_file(
                        p,
                        doc_id=doc_id_override,
                        doc_name=doc_name_override,
                        purge_existing=purge_existing,
                        batch_size=batch_size,
                    )
                    results.append(r)
                    if r.get("doc_id"):
                        seen_doc_ids[r["doc_id"]] = p
                    ingested = True
                    break
                except Exception as e:
                    last_err = e
                    if attempt == 0 and _is_transient_rpc_error(e):
                        logger.warning(
                            f"[load-vec] Milvus 连接异常, 淘汰缓存并重连后重试: {e}"
                        )
                        self.clients.evict_milvus_ingester(
                            uri, token, db_name, collection, dim,
                        )
                        ingester = _make_ingester(use_recreate=False)
                        continue
                    break
            if not ingested and last_err is not None:
                logger.warning(f"[load-vec] 跳过 (失败): {p} - {last_err}")
                skipped.append((p, str(last_err)))

        if skipped:
            logger.warning(f"[load-vec] {len(skipped)} 个文件失败:")
            for p, err in skipped:
                logger.warning(f"  - {p}: {err}")
        return results

    # ── 查询流程 (retrieve → generate) ───────────────────────────────

    def query(
        self,
        query: str,
        mode: Optional[str] = None,
        top_k: Optional[int] = None,
        stream: bool = False,
        output_file: Optional[str] = None,
        use_agentic: bool = True,
        professional: bool = False,
        collection: Optional[str] = None,
    ) -> QueryResult:
        """单次查询: 检索 + 生成, 返回 QueryResult。"""
        self._maybe_switch_collection(collection)
        result, _ = self._get_query_flow().run(
            query, mode=mode, top_k=top_k, stream=stream,
            output_file=output_file, use_agentic=use_agentic,
            professional=professional,
        )
        return result

    def chat(
        self,
        query: str,
        session: Optional[ChatSession] = None,
        mode: Optional[str] = None,
        top_k: Optional[int] = None,
        stream: bool = False,
        use_agentic: bool = True,
        professional: bool = False,
        collection: Optional[str] = None,
    ) -> tuple:
        """多轮对话查询, 维护对话历史。

        Args:
            query: 用户问题
            session: 对话会话 (None 则新建)
            mode: 检索模式, 仅非 agentic 模式
            top_k: 返回 top_k 条结果
            stream: 是否流式输出
            use_agentic: 是否使用 Agentic RAG
            professional: 是否使用专业研究模式
            collection: 目标 Milvus 集合名 (None 则用配置默认)

        Returns:
            (QueryResult, ChatSession) 元组
        """
        self._maybe_switch_collection(collection)
        return self._get_query_flow().run(
            query, mode=mode, top_k=top_k, stream=stream,
            use_agentic=use_agentic, session=session,
            professional=professional,
        )

    # ── 灌入到指定集合 ──────────────────────────────────────────────

    def ingest_files(
        self,
        file_paths: List[str],
        collection: str,
        output_dir: Optional[str] = None,
        parse_timeout: Optional[int] = None,
        backend: Optional[str] = None,
    ) -> IngestResult:
        """将 PDF 文件灌入到指定名称的 Milvus 集合 (自动创建集合)。

        Args:
            file_paths: PDF 文件路径列表
            collection: 目标集合名 (建议以 kb_ 开头)
            output_dir: 中间产物输出目录
            parse_timeout: 解析整体超时秒数
            backend: 临时覆盖 parsing.backend
        """
        original_collection = self.config.milvus.get("collection")
        self.config.milvus["collection"] = collection
        # 重置 IngestFlow, 让它以新集合名重新构建 MilvusIngester
        self._ingest_flow = None
        try:
            return self._get_ingest_flow().run(
                file_paths, output_dir=output_dir,
                parse_timeout=parse_timeout,
            )
        finally:
            # 恢复原始集合名
            if original_collection is not None:
                self.config.milvus["collection"] = original_collection
            else:
                self.config.milvus.pop("collection", None)
            self._ingest_flow = None

    def vectorize_directory(
        self,
        directory: str,
        collection: str,
        recreate: bool = False,
        skip_existing: bool = True,
        progress_callback: Optional[Any] = None,
    ) -> List[IngestResult]:
        """从已有解析产物目录批量 chunk→embed→store 到指定集合 (跳过 PDF 解析)。

        用于知识库"重建": 复用 ``uploads/kb_<name>/`` 下已落盘的解析产物,
        recreate=True 时先清空集合再全量重灌, 不重新解析 PDF。

        Args:
            directory: 解析产物根目录 (每篇文档一个子目录)
            collection: 目标集合名
            recreate: True=清空集合后重建; False=增量追加
            skip_existing: append 模式下是否跳过已存在 doc_id
            progress_callback: 进度回调 callback(current, total, doc_id, status)
        """
        original_collection = self.config.milvus.get("collection")
        self.config.milvus["collection"] = collection
        self._ingest_flow = None
        try:
            return self._get_ingest_flow().vectorize_from_directory(
                directory,
                recreate=recreate,
                skip_existing=skip_existing,
                progress_callback=progress_callback,
            )
        finally:
            if original_collection is not None:
                self.config.milvus["collection"] = original_collection
            else:
                self.config.milvus.pop("collection", None)
            self._ingest_flow = None

    def reingest_directory(
        self,
        directory: str,
        collection: str,
        recreate: bool = False,
        skip_existing: bool = True,
        progress_callback: Optional[Any] = None,
    ) -> List[IngestResult]:
        """从已有工作目录复用 ``knowledge_blocks_vec.json`` 直接重灌指定集合。

        与 ``vectorize_directory`` 不同: 不重新 chunk / embed, 而是把已落盘的向量
        逐字节喂回 Milvus, 因此重灌结果与首次入库一致, 且不依赖 embedding 服务在线。
        缺 vec.json 的文档自动回退到完整 chunk→embed→store。
        """
        original_collection = self.config.milvus.get("collection")
        self.config.milvus["collection"] = collection
        self._ingest_flow = None
        try:
            return self._get_ingest_flow().reingest_from_directory(
                directory,
                recreate=recreate,
                skip_existing=skip_existing,
                progress_callback=progress_callback,
            )
        finally:
            if original_collection is not None:
                self.config.milvus["collection"] = original_collection
            else:
                self.config.milvus.pop("collection", None)
            self._ingest_flow = None

    # ── 集合管理 ────────────────────────────────────────────────────

    @contextmanager
    def _admin_milvus_client(self):
        """临时 MilvusClient 上下文: 用完即 close(), 避免 gRPC 连接 fd 泄漏。

        管理类操作 (list/drop/flush/统计) 每次新建一个短连接; 不显式关闭会持续
        占用文件描述符, 高频轮询 (前端列表/健康检查) 下很快耗尽 fd (EMFILE),
        进而导致上传时 multipart 落临时文件失败 (400 parsing the body)。
        """
        from .clients.milvus import resolve_milvus_connection
        from pymilvus import MilvusClient

        cfg = self.config.milvus
        uri, token, db_name = resolve_milvus_connection(cfg)
        kwargs: Dict[str, Any] = {
            "uri": uri,
            "keepalive_time_ms": 300_000,
            "keepalive_timeout_ms": 60_000,
        }
        if token:
            kwargs["token"] = token
        if db_name:
            kwargs["db_name"] = db_name
        client = MilvusClient(**kwargs)
        try:
            yield client
        finally:
            try:
                client.close()
            except Exception:
                pass

    def list_collections(self, prefix: str = "kb_") -> List[Dict[str, Any]]:
        """列出 Milvus 中的集合, 可按前缀过滤。

        Returns:
            每个集合的 {name, row_count} 字典列表
        """
        results: List[Dict[str, Any]] = []
        with self._admin_milvus_client() as client:
            all_collections = client.list_collections()
            for name in all_collections:
                if prefix and not name.startswith(prefix):
                    continue
                row_count = 0
                try:
                    stats = client.get_collection_stats(name)
                    row_count = stats.get("row_count", 0)
                except Exception:
                    pass
                results.append({"name": name, "row_count": row_count})
        return results

    def drop_collection(self, name: str) -> bool:
        """删除一个 Milvus 集合。

        Args:
            name: 集合名 (仅允许 kb_ 前缀的集合)

        Returns:
            True 若成功删除, False 若集合不存在
        """
        from .clients.milvus import resolve_milvus_connection

        cfg = self.config.milvus
        uri, token, db_name = resolve_milvus_connection(cfg)

        with self._admin_milvus_client() as client:
            if not client.has_collection(name):
                return False
            client.drop_collection(name)
        logger.info(f"[pipeline] 已删除集合: {name}")

        # 从 ClientRegistry 缓存中淘汰该集合的 ingester
        dim = cfg.get("dim", 1024)
        self.clients.evict_milvus_ingester(uri, token, db_name, name, dim)

        # 若删除的恰好是当前活跃集合, 清空 QueryFlow 缓存
        if self._active_collection == name:
            if self._query_flow is not None:
                self._query_flow.invalidate_caches()
            self._active_collection = None
        return True

    def flush_collection(self, name: str) -> None:
        """flush 一个集合, 让 row_count 统计立即反映已插入数据。

        Milvus 的 get_collection_stats 只统计已封存 (sealed) 的段,
        灌入后不 flush 时 row_count 会滞后为 0, 故灌入结束后显式 flush。
        """
        try:
            with self._admin_milvus_client() as client:
                if client.has_collection(name):
                    client.flush(name)
        except Exception as e:
            logger.warning(f"[pipeline] flush 集合失败 {name}: {e}")

    def list_doc_ids(self, collection: str) -> set:
        """返回某集合中已入库的全部 doc_id (集合不存在时返回空集)。

        用于"同库再次上传按文件名去重": doc_id == PDF 文件名 stem, 据此跳过
        已入库文献, 只灌未入库的。
        """
        doc_ids: set = set()
        with self._admin_milvus_client() as client:
            if not client.has_collection(collection):
                return doc_ids
            # 优先 query_iterator (无 offset+limit 窗口上限); 老版本回退分页 query
            try:
                it = client.query_iterator(
                    collection_name=collection, filter="",
                    output_fields=["doc_id"], batch_size=5000,
                )
                while True:
                    batch = it.next()
                    if not batch:
                        break
                    for r in batch:
                        d = r.get("doc_id", "")
                        if d:
                            doc_ids.add(d)
                if hasattr(it, "close"):
                    it.close()
            except AttributeError:
                offset, page = 0, 5000
                while True:
                    batch = client.query(
                        collection_name=collection, filter="",
                        output_fields=["doc_id"], limit=page, offset=offset,
                    )
                    if not batch:
                        break
                    for r in batch:
                        d = r.get("doc_id", "")
                        if d:
                            doc_ids.add(d)
                    if len(batch) < page:
                        break
                    offset += page
            except Exception as e:
                logger.warning(f"[pipeline] list_doc_ids 查询失败 {collection}: {e}")
        return doc_ids

    def list_documents(self, collection: str) -> List[Dict[str, Any]]:
        """返回某集合中每篇文献的概览, 供知识库界面展示"库里有哪些文献"。

        聚合 Milvus 中全部 chunk, 按 doc_id 归并, 每篇返回:
        ``{doc_id, doc_name(标题), year, chunks(块数)}``。集合不存在返回空列表。
        """
        docs: Dict[str, Dict[str, Any]] = {}

        def _ingest_row(r: Dict[str, Any]) -> None:
            d = r.get("doc_id") or ""
            if not d:
                return
            info = docs.get(d)
            if info is None:
                info = {"doc_id": d, "doc_name": "", "year": 0, "chunks": 0}
                docs[d] = info
            info["chunks"] += 1
            if not info["doc_name"] and r.get("doc_name"):
                info["doc_name"] = r["doc_name"]
            if not info["year"] and r.get("publication_year"):
                try:
                    info["year"] = int(r["publication_year"])
                except (TypeError, ValueError):
                    pass

        out_fields = ["doc_id", "doc_name", "publication_year"]
        with self._admin_milvus_client() as client:
            if not client.has_collection(collection):
                return []
            try:
                it = client.query_iterator(
                    collection_name=collection, filter="",
                    output_fields=out_fields, batch_size=5000,
                )
                while True:
                    batch = it.next()
                    if not batch:
                        break
                    for r in batch:
                        _ingest_row(r)
                if hasattr(it, "close"):
                    it.close()
            except AttributeError:
                offset, page = 0, 5000
                while True:
                    batch = client.query(
                        collection_name=collection, filter="",
                        output_fields=out_fields, limit=page, offset=offset,
                    )
                    if not batch:
                        break
                    for r in batch:
                        _ingest_row(r)
                    if len(batch) < page:
                        break
                    offset += page
            except Exception as e:
                logger.warning(f"[pipeline] list_documents 查询失败 {collection}: {e}")

        for info in docs.values():
            if not info["doc_name"]:
                info["doc_name"] = info["doc_id"]
        return sorted(docs.values(), key=lambda x: str(x["doc_name"]).lower())

    # ── 便利方法 ──────────────────────────────────────────────────────

    def stats(self, collection: Optional[str] = None) -> Dict[str, Any]:
        """查看 Milvus 集合统计。

        默认统计原始默认库 (_default_collection), 不受检索时集合切换的污染;
        传入 collection 则统计指定集合。
        """
        target = collection or self._default_collection
        prev = self.config.milvus.get("collection")
        self.config.milvus["collection"] = target
        try:
            r = self.run_step("store", stats_only=True)
            return r.data if r.success else {"error": r.error}
        finally:
            if prev is not None:
                self.config.milvus["collection"] = prev
            else:
                self.config.milvus.pop("collection", None)

    def history(self) -> List[Dict]:
        """返回所有已执行步骤的历史记录。"""
        return [
            {
                "step": r.step_name,
                "success": r.success,
                "elapsed": r.elapsed,
                "error": r.error,
            }
            for r in self._results
        ]
