"""基础检索器: 元数据检索、向量检索、混合检索 (RRF 融合)。

数据结构使用 dataclass (Hit, ParsedQuery), 非 LLM 输出无需 Pydantic。
"""

from __future__ import annotations

import logging
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

try:
    from pymilvus import MilvusClient
except ImportError:
    print("缺少 pymilvus, 请先安装: pip install 'pymilvus[milvus_lite]>=2.4.0'", file=sys.stderr)
    sys.exit(1)

from ..clients.embedding import EmbeddingClient
from ..clients.query_format import EMBED_STAGE_PASSAGE

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------

DEFAULT_EMBED_API_BASE = "https://ms-vjqjknwp-100037631824-sw.gw.ap-beijing.ti.tencentcs.com/ms-vjqjknwp/v1"
DEFAULT_EMBED_MODEL = "model"
# 不在代码里硬编码 api_key, 必须通过 config / 环境变量传入
DEFAULT_EMBED_API_KEY = ""
DEFAULT_MILVUS_URI = "./milvus_lite.db"
DEFAULT_MILVUS_TOKEN = ""
DEFAULT_COLLECTION = "literature_chunks"
DEFAULT_TOP_K = 5
RRF_K = 60
# Hybrid 中 dense / BM25 的权重 (RRF 加权)
DEFAULT_DENSE_WEIGHT = 0.6
DEFAULT_BM25_WEIGHT = 0.4
# Dense / BM25 search 时的额外参数 (HNSW: ef; IVF: nprobe; AUTOINDEX: 留空)
DEFAULT_DENSE_METRIC = "IP"
DEFAULT_DENSE_SEARCH_PARAMS: Dict[str, Any] = {}
DEFAULT_BM25_SEARCH_PARAMS: Dict[str, Any] = {"drop_ratio_search": 0.0}
VALID_CHUNK_TYPES = ("summary", "text", "title", "table", "image", "equation", "references")

# Milvus 返回字段
_OUTPUT_FIELDS = [
    "pk", "kb_id", "chunk_id", "doc_id", "doc_name",
    "type", "section", "page_start", "paragraph_index",
    "publication_year",
    "content", "context", "related_assets",
    "bbox", "bboxes", "page_width", "page_height",
]

_REQUIRED_COLLECTION_FIELDS = {
    "pk", "kb_id", "chunk_id", "doc_id", "doc_name", "type",
    "section", "page_start", "paragraph_index", "publication_year",
    "content", "context", "related_assets", "bbox", "bboxes",
    "embedding", "sparse_embedding",  # v3 schema: BM25 稀疏向量
}


# ---------------------------------------------------------------------------
# 关键词提取
# ---------------------------------------------------------------------------

_FIG_REF_RE = re.compile(r"(?:Fig(?:ure|\.)?|图)\s*([0-9IVXivx]+)", re.IGNORECASE)
_TAB_REF_RE = re.compile(r"(?:Table|表)\s*([0-9IVXivx]+)", re.IGNORECASE)
_EN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")
_CN_TOKEN_RE = re.compile(r"[一-鿿]{2,}")

_STOPWORDS = {
    "the", "and", "for", "are", "you", "what", "how", "this", "that", "was", "were",
    "with", "from", "into", "have", "has", "can", "will", "would", "should", "could",
    "where", "which", "when", "why", "all", "any", "some", "tell", "give", "show",
    "about", "describe", "explain",
    "什么", "怎么", "如何", "为什么", "哪个", "哪些", "请问", "告诉", "说说", "介绍",
    "可以", "需要", "数据", "结果",
}


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class Hit:
    """统一的检索命中数据结构 (v4 schema, 增加 paragraph_index)。"""
    pk: str = ""
    kb_id: str = ""
    chunk_id: str = ""
    doc_id: str = ""
    doc_name: str = ""
    type: str = ""
    section: str = ""
    page_start: int = -1
    paragraph_index: int = -1
    publication_year: int = 0
    content: str = ""
    context: str = ""
    related_assets: List[Dict[str, Any]] = field(default_factory=list)
    bbox: Dict[str, Any] = field(default_factory=dict)
    bboxes: List[Dict[str, Any]] = field(default_factory=list)
    page_width: int = 0
    page_height: int = 0
    score: float = 0.0
    rrf_score: float = 0.0
    sources: List[str] = field(default_factory=list)
    matched_keywords: List[str] = field(default_factory=list)
    # reranker 给的相关性分; None = 未参与 rerank (结构化/metadata 命中, 或 reranker 未启用)
    # 与 score (emb_score) 共存, 不互相覆盖, reflect/diagnosis 可同时利用两个信号
    rerank_score: Optional[float] = None
    # 复合查询: 标记 hit 来自哪个子查询及其 rerank 用改写文本
    subquery_id: str = ""
    subquery_rewrite: str = ""
    # 检索阶段标记 (per-route/stage/type 阈值矩阵)
    stage: str = ""


@dataclass
class ParsedQuery:
    fig_labels: List[str]
    tab_labels: List[str]
    keywords: List[str]

    def is_empty(self) -> bool:
        return not (self.fig_labels or self.tab_labels or self.keywords)


def parse_query(query: str) -> ParsedQuery:
    """从 query 中提取图表引用和关键词。"""
    fig_labels = sorted({m.group(1).upper() for m in _FIG_REF_RE.finditer(query)})
    tab_labels = sorted({m.group(1).upper() for m in _TAB_REF_RE.finditer(query)})

    en_tokens = [t.lower() for t in _EN_TOKEN_RE.findall(query)]
    cn_tokens = _CN_TOKEN_RE.findall(query)
    raw = en_tokens + cn_tokens
    keywords: List[str] = []
    seen = set()
    for tok in raw:
        if tok in _STOPWORDS:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        keywords.append(tok)
    return ParsedQuery(fig_labels=fig_labels, tab_labels=tab_labels, keywords=keywords)


# ---------------------------------------------------------------------------
# Milvus 客户端工具
# ---------------------------------------------------------------------------

def _is_milvus_lite_uri(uri: str) -> bool:
    uri = str(uri or "")
    return uri.endswith(".db") or "://" not in uri


# ---------------------------------------------------------------------------
# MilvusClient 连接复用 (单例缓存)
# ---------------------------------------------------------------------------

# 按 (uri, token, db_name) 缓存 MilvusClient 实例, 避免同一 Milvus Lite 进程被多连接频繁 ping
_milvus_client_cache: Dict[Tuple[str, str, str], MilvusClient] = {}

_DUAL_PATH_POOL: Optional[ThreadPoolExecutor] = None
_DUAL_PATH_LOCK = threading.Lock()
_DUAL_PATH_MAX_WORKERS = 8


def _get_dual_path_pool() -> ThreadPoolExecutor:
    global _DUAL_PATH_POOL
    if _DUAL_PATH_POOL is not None:
        return _DUAL_PATH_POOL
    with _DUAL_PATH_LOCK:
        if _DUAL_PATH_POOL is None:
            _DUAL_PATH_POOL = ThreadPoolExecutor(
                max_workers=_DUAL_PATH_MAX_WORKERS,
                thread_name_prefix="dual-retr",
            )
        return _DUAL_PATH_POOL


def run_in_parallel(
    tasks: List[Tuple[str, Callable[[], T]]],
    *,
    on_error: Optional[Callable[[str, Exception], T]] = None,
) -> Dict[str, T]:
    """并行执行 (name, callable) 任务; 0/1 个任务时同步执行。"""
    out: Dict[str, T] = {}
    if not tasks:
        return out
    if len(tasks) == 1:
        name, fn = tasks[0]
        try:
            out[name] = fn()
        except Exception as e:
            if on_error is not None:
                out[name] = on_error(name, e)
            else:
                raise
        return out

    pool = _get_dual_path_pool()
    futures = {name: pool.submit(fn) for name, fn in tasks}
    for name, fut in futures.items():
        try:
            out[name] = fut.result()
        except Exception as e:
            if on_error is not None:
                out[name] = on_error(name, e)
            else:
                raise
    return out


def _create_milvus_client(
    milvus_uri: str,
    milvus_token: str = "",
    keepalive_time_ms: int = 300_000,
    keepalive_timeout_ms: int = 60_000,
    db_name: str = "",
) -> MilvusClient:
    """创建或复用 MilvusClient (按 uri+token+db_name 单例)。

    默认 keepalive 5 分钟, 远大于 Milvus Lite 默认的 20 秒,
    避免 too_many_pings (GOAWAY ENHANCE_YOUR_CALM) 错误。

    ``db_name`` 仅对远程 Milvus server 有效, Lite 模式应传空串。
    """
    cache_key = (str(milvus_uri), str(milvus_token), str(db_name or ""))
    if cache_key in _milvus_client_cache:
        return _milvus_client_cache[cache_key]

    kwargs: Dict[str, Any] = {
        "uri": milvus_uri,
        "keepalive_time_ms": keepalive_time_ms,
        "keepalive_timeout_ms": keepalive_timeout_ms,
    }
    if milvus_token:
        kwargs["token"] = milvus_token
    if db_name:
        kwargs["db_name"] = db_name
    client = MilvusClient(**kwargs)
    _milvus_client_cache[cache_key] = client
    return client


def _ensure_collection_ready(client: MilvusClient, collection: str, milvus_uri: str) -> None:
    if not client.has_collection(collection):
        raise ValueError(
            f"Milvus collection 不存在: {collection}. 请先运行 ingest 流程完成建库和入库。"
        )
    try:
        desc = client.describe_collection(collection)
        field_names = {f.get("name") for f in desc.get("fields", [])}
        missing = _REQUIRED_COLLECTION_FIELDS - field_names
        if missing:
            raise ValueError(
                f"Milvus collection '{collection}' 缺少字段: {sorted(missing)}; "
                "请用最新版流程重建集合。"
            )
    except ValueError:
        raise
    except Exception:
        raise

    if not _is_milvus_lite_uri(milvus_uri):
        client.load_collection(collection)


def _row_to_hit(row: Dict[str, Any], score: float = 0.0) -> Hit:
    related = row.get("related_assets") or []
    if not isinstance(related, list):
        related = []
    bboxes = row.get("bboxes") or []
    if not isinstance(bboxes, list):
        bboxes = []
    bbox = row.get("bbox") or {}
    if not isinstance(bbox, dict):
        bbox = {}
    raw_para = row.get("paragraph_index")
    try:
        paragraph_index = int(raw_para) if raw_para is not None else -1
    except (TypeError, ValueError):
        paragraph_index = -1
    return Hit(
        pk=row.get("pk", ""),
        kb_id=row.get("kb_id", ""),
        chunk_id=row.get("chunk_id", ""),
        doc_id=row.get("doc_id", ""),
        doc_name=row.get("doc_name", ""),
        type=row.get("type", ""),
        section=row.get("section", ""),
        page_start=int(row.get("page_start", -1)),
        paragraph_index=paragraph_index,
        publication_year=int(row.get("publication_year") or 0),
        content=row.get("content", "") or "",
        context=row.get("context", "") or "",
        related_assets=related,
        bbox=bbox,
        bboxes=[b for b in bboxes if isinstance(b, dict)],
        page_width=int(row.get("page_width") or 0),
        page_height=int(row.get("page_height") or 0),
        score=score,
    )


def _escape_like(s: str) -> str:
    """转义 Milvus LIKE 表达式中的通配符: %, _, \\, \"。"""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_").replace('"', '\\"')


def _escape_eq(s: str) -> str:
    """转义 Milvus == 表达式中的特殊字符: 仅 \\ 和 \", 不转义 % 和 _。"""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _build_filter_expr(
    doc_id: Optional[str],
    chunk_type: Optional[str],
    kb_ids: Optional[List[str]] = None,
) -> Optional[str]:
    parts = []
    ids = [x for x in (kb_ids or []) if x]
    if ids:
        escaped = ", ".join(f'"{_escape_eq(x)}"' for x in ids)
        parts.append(f"kb_id in [{escaped}]")
    if doc_id:
        parts.append(f'doc_id == "{_escape_eq(doc_id)}"')
    if chunk_type:
        parts.append(f'type == "{chunk_type}"')
    return " and ".join(parts) if parts else None


# ---------------------------------------------------------------------------
# BM25Retriever (v3: 稀疏向量召回, 替代 LIKE)
# ---------------------------------------------------------------------------

class BM25Retriever:
    """BM25 稀疏向量召回。

    依赖 Milvus schema v3: sparse_embedding 字段由 BM25 Function 自动从
    embedding_text 生成, 检索时直接传 query 文本即可由 Milvus 服务端做分词。

    与原 MetadataRetriever (基于 SQL LIKE) 相比:
    - 真正的倒排索引, 性能从 O(N) 降到 O(log N)
    - 自动分词 (默认 jieba 中文), 支持词频加权
    - 支持中英混排

    search_params 可调:
    - drop_ratio_search: 0.0~1.0, 大值丢弃低权重 token, 提升性能
    """

    def __init__(
        self,
        client: MilvusClient,
        collection: str = DEFAULT_COLLECTION,
        search_params: Optional[Dict[str, Any]] = None,
    ):
        self.client = client
        self.collection = collection
        self.search_params: Dict[str, Any] = (
            dict(search_params) if search_params is not None else dict(DEFAULT_BM25_SEARCH_PARAMS)
        )

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filter_expr: Optional[str] = None,
    ) -> List[Hit]:
        if not query or not query.strip():
            return []

        search_params = {
            "metric_type": "BM25",
            "params": dict(self.search_params),
        }
        try:
            results = self.client.search(
                collection_name=self.collection,
                data=[query],
                anns_field="sparse_embedding",
                limit=top_k,
                search_params=search_params,
                filter=filter_expr or "",
                output_fields=_OUTPUT_FIELDS,
            )
        except Exception as e:
            logger.warning(f"[bm25] sparse search 失败 q={query!r}: {e}")
            return []

        hits: List[Hit] = []
        for entry in results[0]:
            row = entry.get("entity") or {}
            hit = _row_to_hit(row, score=float(entry.get("distance", 0.0)))
            hit.sources = ["bm25"]
            hits.append(hit)
        return hits


# ---------------------------------------------------------------------------
# MetadataRetriever (保留: 仅做 fig/table 编号直查的 LIKE 兜底)
# ---------------------------------------------------------------------------

class MetadataRetriever:
    """图表编号直查 (LIKE) + BM25 关键词召回的薄封装。

    策略:
    1. 解析 query 中的 Fig./Table 编号, 用 LIKE 精确匹配 (编号短、稀有, LIKE 性能可接受)
    2. 关键词部分委托给 BM25Retriever, 不再走 SQL LIKE
    3. 在客户端融合两路结果
    """

    def __init__(
        self,
        client: MilvusClient,
        collection: str = DEFAULT_COLLECTION,
        bm25_retriever: Optional[BM25Retriever] = None,
    ):
        self.client = client
        self.collection = collection
        self.bm25 = bm25_retriever or BM25Retriever(client, collection)

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filter_expr: Optional[str] = None,
        max_candidates: int = 200,
    ) -> List[Hit]:
        parsed = parse_query(query)
        if parsed.is_empty():
            return []

        # 1) 图表编号直查 (LIKE 在小候选集上仍是最直接的方式)
        ref_hits = self._retrieve_by_refs(parsed, filter_expr, max_candidates)

        # 2) 关键词召回 -> BM25
        bm25_hits: List[Hit] = []
        if parsed.keywords:
            kw_query = " ".join(parsed.keywords)
            bm25_hits = self.bm25.retrieve(kw_query, top_k=top_k * 2, filter_expr=filter_expr)
            for h in bm25_hits:
                h.sources = ["metadata-bm25"]
                h.matched_keywords = list(parsed.keywords)

        # 3) 合并: 图表编号全量保留; 有关键词时再补 BM25
        seen: set = set()
        merged: List[Hit] = []
        for h in ref_hits:
            if h.pk in seen:
                continue
            seen.add(h.pk)
            merged.append(h)
        if ref_hits and not parsed.keywords:
            return merged
        for h in bm25_hits:
            if h.pk in seen:
                continue
            seen.add(h.pk)
            merged.append(h)
        if ref_hits:
            return merged
        return merged[:top_k]

    def _retrieve_by_refs(
        self,
        parsed: "ParsedQuery",
        filter_expr: Optional[str],
        max_candidates: int,
    ) -> List[Hit]:
        from .metadata_match import collect_ref_like_clauses, score_fig_table_refs

        ref_clauses = collect_ref_like_clauses(parsed.fig_labels, parsed.tab_labels)
        if not ref_clauses:
            return []
        retrieve_filter = "(" + " or ".join(ref_clauses) + ")"
        if filter_expr:
            retrieve_filter += f" and ({filter_expr})"
        try:
            rows = self.client.query(
                collection_name=self.collection,
                filter=retrieve_filter,
                output_fields=_OUTPUT_FIELDS,
                limit=max_candidates,
            )
        except Exception as e:
            logger.warning(f"[metadata-refs] LIKE 查询失败: {e}")
            return []

        scored: List[Hit] = []
        for row in rows:
            hit = _row_to_hit(row)
            blob = " ".join([hit.content, hit.section, hit.context])
            score, matched = score_fig_table_refs(
                blob, parsed.fig_labels, parsed.tab_labels, hit.type,
            )
            if score == 0:
                continue
            hit.score = score
            hit.matched_keywords = matched
            hit.sources = ["metadata-ref"]
            scored.append(hit)
        scored.sort(key=lambda h: -h.score)
        return scored


# ---------------------------------------------------------------------------
# VectorRetriever
# ---------------------------------------------------------------------------

class VectorRetriever:
    """稠密向量召回。

    search_params 可调 (按索引类型):
    - HNSW: {"ef": 64}        # search-time 候选池大小, 默认 64; 大 -> 召回率↑ / 延迟↑
    - IVF*: {"nprobe": 10}    # 探针 cluster 数, 默认 10; 类似 ef
    - AUTOINDEX/FLAT: 留空    # 内部自管理
    """

    def __init__(
        self,
        client: MilvusClient,
        embedder: EmbeddingClient,
        collection: str = DEFAULT_COLLECTION,
        metric_type: str = DEFAULT_DENSE_METRIC,
        search_params: Optional[Dict[str, Any]] = None,
    ):
        self.client = client
        self.embedder = embedder
        self.collection = collection
        self.metric_type = metric_type.upper()
        self.search_params: Dict[str, Any] = (
            dict(search_params) if search_params is not None else dict(DEFAULT_DENSE_SEARCH_PARAMS)
        )

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filter_expr: Optional[str] = None,
        embed_stage: Optional[str] = EMBED_STAGE_PASSAGE,
    ) -> List[Hit]:
        qvec = self.embedder.embed_for_retrieval(query, embed_stage)
        return self.retrieve_with_vector(
            qvec, top_k=top_k, filter_expr=filter_expr,
        )

    def retrieve_with_vector(
        self,
        qvec: List[float],
        top_k: int = DEFAULT_TOP_K,
        filter_expr: Optional[str] = None,
    ) -> List[Hit]:
        if not qvec:
            return []
        results = self.client.search(
            collection_name=self.collection,
            data=[qvec],
            anns_field="embedding",
            limit=top_k,
            search_params={
                "metric_type": self.metric_type,
                "params": dict(self.search_params),
            },
            filter=filter_expr or "",
            output_fields=_OUTPUT_FIELDS,
        )
        hits: List[Hit] = []
        for entry in results[0]:
            row = entry.get("entity") or {}
            hit = _row_to_hit(row, score=float(entry.get("distance", 0.0)))
            hit.sources = ["vector"]
            hits.append(hit)
        return hits


# ---------------------------------------------------------------------------
# HybridRetriever (Dense + BM25 加权 RRF)
# ---------------------------------------------------------------------------

class HybridRetriever:
    """稠密向量 + BM25 稀疏向量的加权 RRF 融合。

    score(d) = w_dense * 1/(k + rank_dense(d))  +  w_bm25 * 1/(k + rank_bm25(d))

    经验上 dense:bm25 = 0.6:0.4 起步, 中文科研文献场景偏向略高 dense 权重。
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = RRF_K,
        dense_weight: float = DEFAULT_DENSE_WEIGHT,
        bm25_weight: float = DEFAULT_BM25_WEIGHT,
    ):
        self.vec = vector_retriever
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filter_expr: Optional[str] = None,
        per_retriever_k: int = 10,
        dense_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        embed_stage: Optional[str] = EMBED_STAGE_PASSAGE,
    ) -> List[Hit]:
        results = run_in_parallel(
            [
                ("vec", lambda: self.vec.retrieve(
                    query, top_k=per_retriever_k, filter_expr=filter_expr,
                    embed_stage=embed_stage,
                )),
                ("bm25", lambda: self.bm25.retrieve(
                    query, top_k=per_retriever_k, filter_expr=filter_expr,
                )),
            ],
            on_error=lambda name, exc: (
                logger.warning(f"[hybrid] {name} 失败 q={query!r}: {exc}") or []
            ),
        )
        vec_hits = results.get("vec", []) or []
        bm25_hits = results.get("bm25", []) or []

        w_dense = self.dense_weight if dense_weight is None else dense_weight
        w_bm25 = self.bm25_weight if bm25_weight is None else bm25_weight

        merged: Dict[str, Hit] = {}

        def _accumulate(hits: List[Hit], source: str, weight: float) -> None:
            for rank, hit in enumerate(hits):
                rrf = weight * 1.0 / (self.rrf_k + rank + 1)
                if hit.pk in merged:
                    existing = merged[hit.pk]
                    existing.rrf_score += rrf
                    if source not in existing.sources:
                        existing.sources.append(source)
                    if hit.matched_keywords:
                        existing.matched_keywords = list(
                            dict.fromkeys(existing.matched_keywords + hit.matched_keywords)
                        )
                else:
                    hit.rrf_score = rrf
                    hit.sources = [source]
                    merged[hit.pk] = hit

        _accumulate(vec_hits, "vector", w_dense)
        _accumulate(bm25_hits, "bm25", w_bm25)

        fused = sorted(merged.values(), key=lambda h: -h.rrf_score)
        return fused[:top_k]


# ---------------------------------------------------------------------------
# Pipeline 工厂方法
# ---------------------------------------------------------------------------

def build_retrievers(
    milvus_uri: str = DEFAULT_MILVUS_URI,
    milvus_token: str = DEFAULT_MILVUS_TOKEN,
    collection: str = DEFAULT_COLLECTION,
    embed_api_base: str = DEFAULT_EMBED_API_BASE,
    embed_model: str = DEFAULT_EMBED_MODEL,
    embed_api_key: str = DEFAULT_EMBED_API_KEY,
    embed_normalize: bool = False,
    embed_query_instruct_enabled: bool = True,
    embed_query_instructs: Optional[Dict[str, str]] = None,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    dense_metric: str = DEFAULT_DENSE_METRIC,
    dense_search_params: Optional[Dict[str, Any]] = None,
    bm25_search_params: Optional[Dict[str, Any]] = None,
    keepalive_time_ms: int = 300_000,
    keepalive_timeout_ms: int = 60_000,
    db_name: str = "",
) -> Tuple[MetadataRetriever, VectorRetriever, BM25Retriever, HybridRetriever]:
    """构建检索器实例: metadata / vector / bm25 / hybrid (dense + bm25)。

    search_params 透传到底层 retriever, 用户可在 config 里指定 ef / nprobe / drop_ratio_search。
    所有 retriever 共享同一个 MilvusClient 连接, 避免 Milvus Lite too_many_pings。

    ``db_name`` 仅在 Milvus server 后端有效, Lite 模式应传空串。
    """
    client = _create_milvus_client(
        milvus_uri=milvus_uri, milvus_token=milvus_token,
        keepalive_time_ms=keepalive_time_ms,
        keepalive_timeout_ms=keepalive_timeout_ms,
        db_name=db_name,
    )
    _ensure_collection_ready(client, collection=collection, milvus_uri=milvus_uri)
    from ..clients.client_registry import get_global_registry
    embedder = get_global_registry().get_embedder(
        api_base=embed_api_base, model=embed_model, api_key=embed_api_key,
        normalize=embed_normalize,
        query_instruct_enabled=embed_query_instruct_enabled,
        query_instructs=embed_query_instructs,
    )
    bm25 = BM25Retriever(
        client, collection=collection, search_params=bm25_search_params,
    )
    meta = MetadataRetriever(client, collection=collection, bm25_retriever=bm25)
    vec = VectorRetriever(
        client, embedder, collection=collection,
        metric_type=dense_metric, search_params=dense_search_params,
    )
    hybrid = HybridRetriever(
        vec, bm25,
        dense_weight=dense_weight, bm25_weight=bm25_weight,
    )
    return meta, vec, bm25, hybrid
