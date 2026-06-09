"""Milvus 向量数据库客户端: Schema 构建 + 数据灌入 + 查询。

从原始 chunk2milvus.py 搬入, 逻辑完全保留。
"""

from __future__ import annotations

import datetime
import glob
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from pymilvus import DataType, Function, FunctionType, MilvusClient
except ImportError:
    print("缺少 pymilvus, 请先安装: pip install 'pymilvus>=2.5.0'", file=sys.stderr)
    sys.exit(1)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------

DEFAULT_URI = "./milvus_lite.db"
DEFAULT_TOKEN = ""
DEFAULT_DB_NAME = ""
DEFAULT_COLLECTION = "literature_chunks"
DEFAULT_DIM = 1024
DEFAULT_BATCH_SIZE = 100

# 支持的后端 (config: milvus.backend)
#   lite   -> 嵌入式 Milvus Lite, 单文件数据库 (默认)
#   server -> 远程 Milvus 服务 (例如 docker-compose 起的 http://localhost:19530)
VALID_MILVUS_BACKENDS = ("lite", "server")
DEFAULT_MILVUS_BACKEND = "lite"

# BM25 analyzer 默认参数: jieba 分词, 对中英混排有较好覆盖
DEFAULT_ANALYZER_PARAMS: Dict[str, Any] = {"type": "chinese"}

# Dense 向量索引配置
# - AUTOINDEX: Milvus 自动选择 (Lite -> FLAT, 集群 -> HNSW); 写入快、零运维
# - HNSW: 推荐用于 < 1M 量级, 召回 / 延迟均衡, 需要调 M / efConstruction
# - IVF_FLAT: 适合更大数据量, 需要调 nlist
DEFAULT_DENSE_INDEX_TYPE = "AUTOINDEX"
DEFAULT_DENSE_METRIC = "IP"
DEFAULT_DENSE_INDEX_PARAMS: Dict[str, Any] = {}  # AUTOINDEX 不需要额外参数
# BM25 sparse 索引参数 (Milvus 2.5+ 全文检索)
DEFAULT_SPARSE_INDEX_PARAMS: Dict[str, Any] = {
    "inverted_index_algo": "DAAT_MAXSCORE",
}

# stats() / 全表扫描时的分页大小, 避免大集合一次拉取超限
STATS_PAGE_SIZE = 5000

# VARCHAR 字段长度上限
MAX_LEN_PK = 256
MAX_LEN_DOC_ID = 128
MAX_LEN_DOC_NAME = 512
MAX_LEN_CHUNK_ID = 64
MAX_LEN_TYPE = 16
MAX_LEN_SECTION = 1024
MAX_LEN_CONTENT = 32000
MAX_LEN_CONTEXT = 8000
MAX_LEN_EMB_TEXT = 32000

# 合法 type 取值
VALID_TYPES = {"summary", "text", "title", "table", "image", "equation", "references"}
# 注: `type` 字段是 VARCHAR(MAX_LEN_TYPE=16), 没有 Milvus 层的 enum 约束;
# 这里的白名单只在 `_normalize_type` 起作用 (兜底把未知类型改成 "text").
# v5 schema 新增 equation / references 两类: 公式独立成 chunk + 与正文双向 related_assets;
# 参考文献按 batch 聚合为 references chunk, 默认在 progressive/local 路径中被排除.

# ---------------------------------------------------------------------------
# Backend 选择: lite (本地文件) / server (远程 Milvus, 如 docker-compose)
# ---------------------------------------------------------------------------

def resolve_milvus_connection(
    milvus_cfg: Optional[Dict[str, Any]],
) -> Tuple[str, str, str]:
    """根据 ``milvus.backend`` 决定连接哪一套 Milvus。

    支持配置形态:

    .. code-block:: yaml

        milvus:
          backend: lite          # 或 server
          lite:
            uri: "./milvus_lite.db"
            token: ""
          server:
            uri: "http://localhost:19530"
            token: ""            # 服务器若启用认证: "user:password" 或 API key
            db_name: ""          # Milvus 多 db 实例时指定; 留空使用默认 db

    向后兼容: 若没填 ``backend`` / 子节, 仍然读取顶层 ``milvus.uri`` / ``milvus.token``。

    Returns:
        (uri, token, db_name) 三元组. ``db_name`` 在 Lite 后端无意义, 此时返回空串。
    """
    cfg = milvus_cfg or {}
    backend = str(cfg.get("backend") or DEFAULT_MILVUS_BACKEND).strip().lower()
    if backend not in VALID_MILVUS_BACKENDS:
        logger.warning(
            f"[milvus] 未知 backend={backend!r}, 回退到 {DEFAULT_MILVUS_BACKEND!r}; "
            f"合法值: {VALID_MILVUS_BACKENDS}"
        )
        backend = DEFAULT_MILVUS_BACKEND

    sub = cfg.get(backend) if isinstance(cfg.get(backend), dict) else {}
    sub = sub or {}

    # 优先级: backend 子节 > 顶层 uri/token (向后兼容)
    uri = (sub.get("uri") or cfg.get("uri") or DEFAULT_URI).strip()
    token = (sub.get("token") or cfg.get("token") or DEFAULT_TOKEN)
    db_name = (sub.get("db_name") or cfg.get("db_name") or DEFAULT_DB_NAME)

    # Lite 后端忽略 db_name (pymilvus 在 Lite 模式下不识别多 db)
    if backend == "lite":
        db_name = ""

    logger.info(
        f"[milvus] backend={backend} uri={uri}"
        + (f" db={db_name}" if db_name else "")
    )
    return uri, str(token or ""), str(db_name or "")


# 文件名提取年份
# 注意: 不能用 \b 边界, 因为 "_2019_" 里下划线也是 word char, \b 失效
# 改用前后非数字字符 (含字符串首尾) 作为边界
_YEAR_FROM_NAME_RE = re.compile(r"(?:^|[^0-9])(19[0-9]{2}|20[0-9]{2})(?:[^0-9]|$)")

# 从 chunk content 中扫描发表年份的常见模式 (摘要/封面页常见)
_YEAR_FROM_CONTENT_PATTERNS = [
    re.compile(r"©\s*(19[0-9]{2}|20[0-9]{2})"),
    re.compile(r"\bCopyright\s+©?\s*(19[0-9]{2}|20[0-9]{2})", re.IGNORECASE),
    re.compile(r"\b(?:Published|Received|Accepted)[\s:,]*(19[0-9]{2}|20[0-9]{2})", re.IGNORECASE),
    re.compile(r"\b(?:Volume|Vol\.?)\s*\d+.*?\b(19[0-9]{2}|20[0-9]{2})", re.IGNORECASE | re.DOTALL),
    re.compile(r"\b(19[0-9]{2}|20[0-9]{2})\s*[A-Z][a-z]+\s*Society", re.IGNORECASE),
    re.compile(r"\bDOI[:\s].*?\b(19[0-9]{2}|20[0-9]{2})\b", re.IGNORECASE | re.DOTALL),
]


def _extract_year_from_chunks(chunks: List[Dict[str, Any]], scan_top_n: int = 5) -> int:
    """从前 N 个 summary/text chunk 的 content 中扫描发表年份。

    优先扫 summary, 其次 text, 跳过 image/table。返回最早出现的、最频繁的年份。
    """
    if not chunks:
        return 0
    candidates: List[Dict[str, Any]] = []
    for c in chunks:
        if c.get("type") in ("summary", "text", "title"):
            candidates.append(c)
        if len(candidates) >= scan_top_n:
            break
    if not candidates:
        return 0
    year_count: Dict[int, int] = {}
    for c in candidates:
        text = (c.get("content") or "")[:2000]
        for pat in _YEAR_FROM_CONTENT_PATTERNS:
            for m in pat.finditer(text):
                year_str = next((g for g in m.groups() if g), None)
                if year_str:
                    try:
                        y = int(year_str)
                        if 1900 <= y <= datetime.datetime.now().year + 1:
                            year_count[y] = year_count.get(y, 0) + 1
                    except ValueError:
                        continue
    if not year_count:
        return 0
    # 取 (count desc, year desc) 第一个; 多个并列时取较新年份
    top = sorted(year_count.items(), key=lambda kv: (-kv[1], -kv[0]))
    return top[0][0]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def _is_transient_rpc_error(exc: BaseException) -> bool:
    """gRPC 通道断开 / Milvus 短暂不可用 (批量灌入长跑时常见)。"""
    msg = str(exc).lower()
    if (
        "closed channel" in msg
        or "channel closed" in msg
        or "fail connecting to server" in msg
        or "server unavailable" in msg
        or "connection reset" in msg
        or "connection refused" in msg
        or "illegal connection params" in msg
    ):
        return True
    cause = getattr(exc, "__cause__", None)
    if cause is not None and cause is not exc:
        return _is_transient_rpc_error(cause)
    return False


def _is_bm25_function_required_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "bm25 function output" in msg
        or "only bm25 function" in msg
    )


def collection_has_bm25_sparse_function(desc: Dict[str, Any]) -> bool:
    """现有集合的 ``sparse_embedding`` 是否由 BM25 Function 生成 (可建 BM25 索引)。"""
    functions = desc.get("functions") or []
    for fn in functions:
        if not isinstance(fn, dict):
            continue
        raw_type = (
            fn.get("type") or fn.get("function_type") or fn.get("type_name")
        )
        # describe_collection 可能把 type 返回为字符串 ("BM25")、FunctionType 枚举,
        # 或整数枚举值 (FunctionType.BM25 == 1)。三种形式都要识别。
        fn_type = str(raw_type if raw_type is not None else "").upper()
        type_val = getattr(raw_type, "value", raw_type)
        is_bm25 = ("BM25" in fn_type) or (isinstance(type_val, int) and type_val == 1)
        if not is_bm25:
            continue
        outputs = (
            fn.get("output_field_names")
            or fn.get("outputFieldNames")
            or fn.get("output_field_name")
            or []
        )
        if isinstance(outputs, str):
            outputs = [outputs]
        if "sparse_embedding" in outputs:
            return True
    return False


def build_schema(
    client: MilvusClient,
    dim: int,
    analyzer_params: Optional[Dict[str, Any]] = None,
):
    """构建集合 schema (v3: 在 v2 基础上加入 BM25 稀疏向量字段)。

    新增字段:
    - sparse_embedding: SPARSE_FLOAT_VECTOR, 由 BM25 Function 自动从 embedding_text 生成
    - embedding_text 字段开启 analyzer (默认 jieba 中文分词)

    v2 -> v3 不向后兼容, 升级需重建集合。
    """
    if analyzer_params is None:
        analyzer_params = dict(DEFAULT_ANALYZER_PARAMS)

    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
        description="literature chunks (summary/text/title/table/image/equation/references) with dense + BM25 sparse embeddings; v5 (equation + references)",
    )
    schema.add_field("pk", DataType.VARCHAR, is_primary=True, max_length=MAX_LEN_PK)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=MAX_LEN_DOC_ID)
    schema.add_field("doc_name", DataType.VARCHAR, max_length=MAX_LEN_DOC_NAME)
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=MAX_LEN_CHUNK_ID)
    schema.add_field("type", DataType.VARCHAR, max_length=MAX_LEN_TYPE)
    schema.add_field("section", DataType.VARCHAR, max_length=MAX_LEN_SECTION)
    schema.add_field("page_start", DataType.INT32)
    # paragraph_index: 文档内 1-based 段落序号; -1 表示 "非正文段落" (image/table/title);
    # 0 表示 LLM 合成的摘要 (不计入正文段落).
    schema.add_field("paragraph_index", DataType.INT32)
    schema.add_field("publication_year", DataType.INT32)
    schema.add_field("content", DataType.VARCHAR, max_length=MAX_LEN_CONTENT)
    schema.add_field("context", DataType.VARCHAR, max_length=MAX_LEN_CONTEXT)
    schema.add_field("related_assets", DataType.JSON)
    # embedding_text 开启 analyzer, 作为 BM25 Function 的输入
    schema.add_field(
        "embedding_text",
        DataType.VARCHAR,
        max_length=MAX_LEN_EMB_TEXT,
        enable_analyzer=True,
        analyzer_params=analyzer_params,
    )
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)

    # BM25 函数: embedding_text -> sparse_embedding
    schema.add_function(
        Function(
            name="bm25_emb_text",
            function_type=FunctionType.BM25,
            input_field_names=["embedding_text"],
            output_field_names=["sparse_embedding"],
        )
    )
    return schema


# ---------------------------------------------------------------------------
# Sidecar meta + publication_year 解析
# ---------------------------------------------------------------------------

def _meta_sidecar_path(vec_path: str, explicit: Optional[str] = None) -> str:
    """推断 vec.json 对应的 sidecar meta 路径。

    优先级:
    1. 显式指定的路径
    2. 同目录下同名 *_meta.json (剥掉 _vec/_vectors/_embedded 后缀)
    """
    if explicit:
        return explicit
    base = os.path.splitext(vec_path)[0]
    for suffix in ("_vec", "_vectors", "_embedded"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base + "_meta.json"


def _write_meta_sidecar(
    vec_path: str,
    doc_id: str,
    doc_name: str,
    publication_year: int,
    explicit: Optional[str] = None,
) -> None:
    """把本次入库实际使用的 doc_id / doc_name / publication_year 落到 sidecar。

    使 ``(knowledge_blocks_vec.json + knowledge_blocks_meta.json)`` 成为一个
    自包含、可重放的单元: 之后直接 ``ingest_file(vec.json)`` 即可还原出与首次
    入库完全一致的 Milvus 行 (pk / doc_id / doc_name / publication_year 一致),
    且无需重新 embed (向量逐字节复用)。

    采用 merge 写法: 若 sidecar 已存在 (如 UniParser chunker 写过 source/标题等),
    只更新这三个字段, 不覆盖其它信息。best-effort, 失败仅告警不抛。
    """
    path = _meta_sidecar_path(vec_path, explicit)
    meta: Dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                meta = loaded
        except Exception:
            meta = {}
    meta["doc_id"] = doc_id
    meta["doc_name"] = doc_name
    meta["publication_year"] = int(publication_year or 0)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info(f"  [meta] 写入 sidecar: {path}")
    except Exception as e:
        logger.warning(f"  [meta] 写 sidecar 失败 {path}: {e}")


def _load_meta_sidecar(vec_path: str, explicit: Optional[str] = None) -> Dict[str, Any]:
    """根据 vec.json 路径推断 sidecar meta 路径并加载。

    优先级:
    1. 显式指定的路径
    2. 同目录下同名 *_meta.json
    3. 都没有就返回 {}
    """
    path = _meta_sidecar_path(vec_path, explicit)
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        if not isinstance(data, dict):
            logger.warning(f"  [meta] 警告: {path} 内容不是 dict, 忽略")
            return {}
        logger.info(f"  [meta] 加载 sidecar: {path}")
        return data
    except Exception as e:
        logger.warning(f"  [meta] 解析 {path} 失败: {e}")
        return {}


def _resolve_publication_year(
    cli_year: Optional[int],
    meta: Dict[str, Any],
    vec_path: str,
    chunks: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """按 CLI > sidecar > 文件名 > chunk content 扫描 > 0 的顺序解析年份。

    chunks 可选; 若提供则在所有显式来源都失败时, 扫描前 N 个 summary/text chunk
    的 content, 查找 "© 20XX" / "Published 20XX" / DOI 行末年份等常见模式。
    """
    if cli_year and cli_year > 0:
        return int(cli_year)
    yr = meta.get("publication_year")
    if isinstance(yr, (int, float)) and yr > 0:
        return int(yr)
    name = os.path.basename(vec_path)
    m = _YEAR_FROM_NAME_RE.search(name)
    if m:
        return int(m.group(1))
    if chunks:
        y = _extract_year_from_chunks(chunks)
        if y > 0:
            logger.info(f"  [meta] publication_year 从 chunk content 扫描到: {y}")
            return y
    return 0


# ---------------------------------------------------------------------------
# doc_id / doc_name / publication_year 推断 (统一入口)
# ---------------------------------------------------------------------------

def infer_doc_metadata(
    path: str,
    chunks: Optional[List[Dict[str, Any]]] = None,
    explicit_doc_id: Optional[str] = None,
    explicit_doc_name: Optional[str] = None,
    explicit_year: Optional[int] = None,
    meta_sidecar_path: Optional[str] = None,
) -> Tuple[str, str, int, Dict[str, Any]]:
    """统一推断 (doc_id, doc_name, publication_year, raw_meta)。

    优先级:
    - doc_id:   explicit > sidecar.doc_id > 文件名 (去 _vec/_vectors/_embedded 后缀)
    - doc_name: explicit > sidecar.doc_name > 文件名
    - year:     explicit > sidecar.publication_year > 文件名年份 > chunks 内容扫描

    返回值的 raw_meta 是加载到的 sidecar 内容 (空 dict 表示无 sidecar)。
    """
    meta = _load_meta_sidecar(path, meta_sidecar_path)

    if explicit_doc_id:
        doc_id = explicit_doc_id
    else:
        sidecar_id = (meta.get("doc_id") or "").strip() if isinstance(meta, dict) else ""
        if sidecar_id:
            doc_id = sidecar_id
        else:
            base = os.path.splitext(os.path.basename(path))[0]
            for suffix in ("_vec", "_vectors", "_embedded"):
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
                    break
            doc_id = base

    if explicit_doc_name:
        doc_name = explicit_doc_name
    else:
        doc_name = (meta.get("doc_name") or os.path.basename(path)) if isinstance(meta, dict) else os.path.basename(path)

    publication_year = _resolve_publication_year(
        cli_year=explicit_year, meta=meta if isinstance(meta, dict) else {},
        vec_path=path, chunks=chunks,
    )
    return doc_id, doc_name, publication_year, meta if isinstance(meta, dict) else {}


# ---------------------------------------------------------------------------
# 字段裁剪 + chunk -> Milvus row
# ---------------------------------------------------------------------------

def _truncate(s: Optional[str], max_len: int) -> str:
    if not s:
        return ""
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _normalize_type(t: Optional[str]) -> str:
    t = (t or "text").strip().lower()
    return t if t in VALID_TYPES else "text"


def _normalize_related_assets(ra: Any) -> List[Dict[str, Any]]:
    if not isinstance(ra, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in ra:
        if not isinstance(item, dict):
            continue
        out.append({
            "type": str(item.get("type") or ""),
            "label": str(item.get("label") or ""),
            "chunk_id": str(item.get("chunk_id") or ""),
        })
    return out


def chunk_to_row(
    chunk: Dict[str, Any],
    doc_id: str,
    doc_name: str,
    publication_year: int = 0,
) -> Dict[str, Any]:
    """把单条 chunk 转成 Milvus 一行数据 (v4 schema)。"""
    chunk_id = chunk.get("id") or ""
    pages = chunk.get("pages") or []
    page_start = int(pages[0]) if pages else -1
    raw_para = chunk.get("paragraph_index")
    try:
        paragraph_index = int(raw_para) if raw_para is not None else -1
    except (TypeError, ValueError):
        paragraph_index = -1

    return {
        "pk": f"{doc_id}::{chunk_id}",
        "doc_id": _truncate(doc_id, MAX_LEN_DOC_ID),
        "doc_name": _truncate(doc_name, MAX_LEN_DOC_NAME),
        "chunk_id": _truncate(chunk_id, MAX_LEN_CHUNK_ID),
        "type": _normalize_type(chunk.get("type")),
        "section": _truncate(chunk.get("section") or "", MAX_LEN_SECTION),
        "page_start": page_start,
        "paragraph_index": paragraph_index,
        "publication_year": int(publication_year or 0),
        "content": _truncate(chunk.get("content") or "", MAX_LEN_CONTENT),
        "context": _truncate(chunk.get("context") or "", MAX_LEN_CONTEXT),
        "related_assets": _normalize_related_assets(chunk.get("related_assets")),
        "embedding_text": _truncate(chunk.get("embedding_text") or "", MAX_LEN_EMB_TEXT),
        "embedding": chunk["embedding"],
    }


# ---------------------------------------------------------------------------
# Ingester
# ---------------------------------------------------------------------------

class MilvusIngester:
    def __init__(
        self,
        uri: str = DEFAULT_URI,
        token: str = DEFAULT_TOKEN,
        collection: str = DEFAULT_COLLECTION,
        dim: int = DEFAULT_DIM,
        recreate: bool = False,
        analyzer_params: Optional[Dict[str, Any]] = None,
        dense_index_type: str = DEFAULT_DENSE_INDEX_TYPE,
        dense_metric: str = DEFAULT_DENSE_METRIC,
        dense_index_params: Optional[Dict[str, Any]] = None,
        db_name: str = DEFAULT_DB_NAME,
    ) -> None:
        self.collection = collection
        self.dim = dim
        self.analyzer_params = analyzer_params or dict(DEFAULT_ANALYZER_PARAMS)
        self.dense_index_type = dense_index_type.upper()
        self.dense_metric = dense_metric.upper()
        self.dense_index_params = dense_index_params or {}
        kwargs: Dict[str, Any] = {
            "uri": uri,
            "keepalive_time_ms": 300_000,
            "keepalive_timeout_ms": 60_000,
        }
        if token:
            kwargs["token"] = token
        # db_name 只对远程 Milvus server 有意义; Lite 模式下不传, 避免老版本 pymilvus 报错
        if db_name:
            kwargs["db_name"] = db_name
        self._client_kwargs = dict(kwargs)
        self._token = token
        self.client = MilvusClient(**self._client_kwargs)
        self._uri = uri
        self._db_name = db_name

        self._ensure_collection(recreate=recreate)

    def reconnect_client(self) -> None:
        """重建 gRPC 客户端并重新 load collection (长跑 bulk ingest 断线后恢复)。"""
        logger.warning(
            f"[milvus] 重连 {self._uri!r} collection={self.collection!r} ..."
        )
        try:
            close_fn = getattr(self.client, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception:
            pass
        self.client = MilvusClient(**self._client_kwargs)
        is_lite = (str(self._uri).endswith(".db") or "://" not in str(self._uri))
        if not is_lite:
            self.client.load_collection(self.collection)

    def _call_with_reconnect(self, fn, *args, **kwargs):
        """执行一次 RPC; 通道已关则重连后重试一次。"""
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if not _is_transient_rpc_error(e):
                raise
            self.reconnect_client()
            return fn(*args, **kwargs)

    @staticmethod
    def _get_field_dim(fields: List[Dict[str, Any]], field_name: str) -> int:
        """从 describe_collection 返回的 fields 列表里抠出某个向量字段的 dim。

        pymilvus 在不同版本 / 后端下 dim 的存放位置不统一, 这里全部兜底:
          - field["dim"]                       (常见)
          - field["params"]["dim"]             (Milvus Lite / 新版)
          - field["type_params"][i]["value"]   (proto 原始形式, key=="dim")
        找不到时返回 0, 让调用方跳过这次检查而非报错.
        """
        for f in fields:
            if f.get("name") != field_name:
                continue
            for key in ("dim", "Dim"):
                v = f.get(key)
                if isinstance(v, int) and v > 0:
                    return v
            params = f.get("params") or {}
            if isinstance(params, dict):
                v = params.get("dim")
                try:
                    if v is not None:
                        return int(v)
                except (TypeError, ValueError):
                    pass
            type_params = f.get("type_params") or []
            if isinstance(type_params, list):
                for tp in type_params:
                    if isinstance(tp, dict) and tp.get("key") == "dim":
                        try:
                            return int(tp.get("value"))
                        except (TypeError, ValueError):
                            continue
            return 0
        return 0

    def _field_has_index(self, field_name: str) -> bool:
        """检查某字段是否已有索引 (用于修复「集合已建、索引未建」的半成品状态)。"""
        try:
            names = self.client.list_indexes(
                self.collection, field_name=field_name,
            )
            return bool(names)
        except Exception as e:
            logger.debug(f"  [info] list_indexes({field_name!r}) failed: {e}")
            return False

    def _create_dense_index(self) -> None:
        vec_params = self.client.prepare_index_params()
        add_kwargs: Dict[str, Any] = {
            "field_name": "embedding",
            "index_type": self.dense_index_type,
            "metric_type": self.dense_metric,
        }
        if self.dense_index_params:
            add_kwargs["params"] = dict(self.dense_index_params)
        vec_params.add_index(**add_kwargs)
        self.client.create_index(
            collection_name=self.collection, index_params=vec_params,
        )

    @staticmethod
    def _exit_recreate_bm25_collection(reason: str) -> None:
        logger.error(
            f"\n[ERROR] 集合无法使用 BM25 稀疏索引: {reason}"
        )
        logger.error(
            "        现有 collection 的 sparse_embedding 不是 BM25 Function 输出字段 "
            "(常见于 Milvus 升级前创建的集合, 或上次建库在 create_index 前中断)。"
        )
        logger.error(
            "        Milvus 不支持给旧 schema 原地挂上 BM25 Function, 必须 drop 后重建:"
        )
        logger.error(
            "          python -m pipeline.run upload <path> --recreate"
        )
        logger.error(
            "        或在 Milvus 控制台 / Attu 中删除该 collection 后重新 upload。"
        )
        raise SystemExit(1)

    def _create_sparse_index(self) -> None:
        sparse_params = self.client.prepare_index_params()
        sparse_params.add_index(
            field_name="sparse_embedding",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params=dict(DEFAULT_SPARSE_INDEX_PARAMS),
        )
        try:
            self.client.create_index(
                collection_name=self.collection, index_params=sparse_params,
            )
        except Exception as e:
            if _is_bm25_function_required_error(e):
                self._exit_recreate_bm25_collection(str(e))
            raise

    def _create_scalar_indexes(self) -> None:
        for fname in (
            "doc_id", "type", "publication_year",
            "page_start", "paragraph_index",
        ):
            if self._field_has_index(fname):
                continue
            try:
                sp = self.client.prepare_index_params()
                sp.add_index(field_name=fname, index_type="INVERTED")
                self.client.create_index(
                    collection_name=self.collection, index_params=sp,
                )
            except Exception as e:
                logger.debug(f"  [info] scalar index on {fname} skipped: {e}")

    def _create_all_indexes(self) -> None:
        """新建集合后创建全部索引。"""
        logger.info(
            f"[create] indexes on {self.collection}: "
            f"dense={self.dense_index_type}/{self.dense_metric}, sparse=BM25"
        )
        self._create_dense_index()
        self._create_sparse_index()
        self._create_scalar_indexes()

    def _ensure_indexes(self, collection_desc: Optional[Dict[str, Any]] = None) -> None:
        """集合已存在时补建缺失索引 (上次建索引失败会留下半成品集合)。"""
        desc = collection_desc
        if desc is None:
            try:
                desc = self.client.describe_collection(self.collection)
            except Exception as e:
                logger.warning(f"  [warn] describe_collection 失败, 跳过索引检查: {e}")
                desc = {}

        missing: List[str] = []
        if not self._field_has_index("embedding"):
            missing.append("embedding")
        if not self._field_has_index("sparse_embedding"):
            missing.append("sparse_embedding")
        if not missing:
            self._create_scalar_indexes()
            return

        if "sparse_embedding" in missing and not collection_has_bm25_sparse_function(desc):
            self._exit_recreate_bm25_collection(
                "describe_collection 中未找到 BM25 Function → sparse_embedding 映射"
            )

        logger.warning(
            f"[repair] 集合 {self.collection!r} 缺少索引: {missing}, 正在补建..."
        )
        if "embedding" in missing:
            self._create_dense_index()
        if "sparse_embedding" in missing:
            self._create_sparse_index()
        self._create_scalar_indexes()

    def _ensure_collection(self, recreate: bool) -> None:
        exists = self.client.has_collection(self.collection)
        if exists and recreate:
            logger.info(f"[recreate] drop collection: {self.collection}")
            self.client.drop_collection(self.collection)
            exists = False

        collection_desc: Optional[Dict[str, Any]] = None
        if exists:
            try:
                collection_desc = self.client.describe_collection(self.collection)
                fields = collection_desc.get("fields", [])
                field_names = {f["name"] for f in fields}
                # v4 schema 必备字段 (在 v3 上新增 paragraph_index)
                required_fields = {
                    "pk", "doc_id", "publication_year",
                    "embedding", "sparse_embedding",
                    "paragraph_index",
                }
                missing = required_fields - field_names
                if missing:
                    logger.error(
                        f"\n[ERROR] 现有集合 '{self.collection}' schema 缺少字段: {missing}"
                    )
                    logger.error(
                        "        v3 -> v4 schema 不兼容 (新增 paragraph_index 字段)。"
                        " 请用 recreate=True 重建集合并重新灌入数据。"
                    )
                    raise SystemExit(1)

                # dim 一致性检查: 集合里实际的 embedding dim 必须与当前配置一致,
                # 否则后续 ingest / query 都会以 RPC 层 "vector dimension mismatch"
                # 报错, 远不如这里早一步给出清晰提示.
                existing_dim = self._get_field_dim(fields, "embedding")
                if existing_dim and existing_dim != self.dim:
                    logger.error(
                        f"\n[ERROR] 集合 '{self.collection}' 现有 embedding 维度="
                        f"{existing_dim}, 与配置 dim={self.dim} 不一致。"
                    )
                    logger.error(
                        "        通常是换了 embedding 模型 (例如 Qwen3-Embedding-0.6B -> 4B,"
                        " dim 从 1024 变成 2560)。Milvus 不支持原地改维度, 必须重建集合:"
                    )
                    logger.error(
                        "          python -m pipeline.run rebuild <mineru_dir>"
                    )
                    logger.error(
                        "        或在配置里把 milvus.dim 改回与现有集合一致的旧维度。"
                    )
                    raise SystemExit(1)

                if "sparse_embedding" in field_names:
                    if not collection_has_bm25_sparse_function(collection_desc):
                        self._exit_recreate_bm25_collection(
                            "schema 含 sparse_embedding 但无 BM25 Function 定义"
                        )
                    emb_text = next(
                        (f for f in fields if f.get("name") == "embedding_text"),
                        None,
                    )
                    if emb_text is None:
                        self._exit_recreate_bm25_collection(
                            "schema 缺少 embedding_text 字段 (BM25 输入)"
                        )
            except SystemExit:
                raise
            except Exception as e:
                logger.warning(f"  [warn] 无法检查集合 schema: {e}")
            self._ensure_indexes(collection_desc)

        if not exists:
            logger.info(
                f"[create] collection: {self.collection} (dim={self.dim}, "
                f"dense_index={self.dense_index_type}, metric={self.dense_metric})"
            )
            schema = build_schema(self.client, self.dim, analyzer_params=self.analyzer_params)
            self.client.create_collection(
                collection_name=self.collection, schema=schema,
            )
            self._create_all_indexes()

        is_lite = (str(self._uri).endswith(".db") or "://" not in str(self._uri))
        if not is_lite:
            try:
                self.client.load_collection(self.collection)
            except Exception as e:
                err = str(e).lower()
                if "no vector index" in err or "create index firstly" in err:
                    logger.error(
                        f"\n[ERROR] 加载集合 {self.collection!r} 失败: 向量索引不完整。"
                    )
                    logger.error(
                        "        常见原因: 上次灌入在 create_index 阶段中断。"
                    )
                    logger.error(
                        "        修复: python -m pipeline.run upload <path> --recreate"
                    )
                    logger.error(
                        "        或在 Milvus 中 drop 该 collection 后重新 upload。"
                    )
                raise

    def purge_doc(self, doc_id: str) -> int:
        try:
            self._call_with_reconnect(
                self.client.delete,
                collection_name=self.collection,
                filter=f'doc_id == "{doc_id}"',
            )
            time.sleep(0.2)
            return 0
        except Exception as e:
            if _is_transient_rpc_error(e):
                raise
            logger.warning(f"  [warn] purge {doc_id} 失败 (可能是空集合): {e}")
            return 0

    def ingest_file(
        self,
        path: str,
        doc_id: Optional[str] = None,
        doc_name: Optional[str] = None,
        purge_existing: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        meta_json_path: Optional[str] = None,
        cli_publication_year: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        if not isinstance(chunks, list):
            raise ValueError(f"{path} 不是 chunks 列表")

        # 用统一的元数据推断: 优先级 explicit > sidecar > 文件名 > chunk content
        doc_id, doc_name, publication_year, _meta = infer_doc_metadata(
            path,
            chunks=chunks,
            explicit_doc_id=doc_id,
            explicit_doc_name=doc_name,
            explicit_year=cli_publication_year,
            meta_sidecar_path=meta_json_path,
        )

        if not chunks:
            logger.warning(f"  [warn] {path} 没有 chunk, 跳过")
            return {
                "path": path, "doc_id": doc_id, "doc_name": doc_name,
                "publication_year": publication_year, "count": 0, "type_count": {},
            }

        sample_dim = len(chunks[0].get("embedding") or [])
        if sample_dim != self.dim:
            raise ValueError(
                f"{path} 向量维度 {sample_dim} != 集合维度 {self.dim}, 请检查 embedding 模型"
            )

        if publication_year:
            logger.info(f"  [meta] publication_year = {publication_year}")
        else:
            logger.info("  [meta] publication_year 未知 (=0)")

        # 落盘完整 sidecar: 让 (vec.json + meta.json) 成为可重放单元,
        # 之后无需重新 embed 即可还原出完全一致的 Milvus 行。
        _write_meta_sidecar(
            path, doc_id=doc_id, doc_name=doc_name,
            publication_year=publication_year, explicit=meta_json_path,
        )

        if purge_existing:
            self.purge_doc(doc_id)

        rows: List[Dict[str, Any]] = []
        type_count: Dict[str, int] = {}
        for c in chunks:
            row = chunk_to_row(
                c, doc_id=doc_id, doc_name=doc_name,
                publication_year=publication_year,
            )
            rows.append(row)
            type_count[row["type"]] = type_count.get(row["type"], 0) + 1

        ok = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i: i + batch_size]
            self._call_with_reconnect(
                self.client.upsert,
                collection_name=self.collection,
                data=batch,
            )
            ok += len(batch)
            logger.info(f"    [{path}] upsert {ok}/{len(rows)}")

        return {
            "path": path,
            "doc_id": doc_id,
            "doc_name": doc_name,
            "publication_year": publication_year,
            "count": ok,
            "type_count": type_count,
        }

    def ingest_glob(
        self,
        pattern: str,
        purge_existing: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        cli_publication_year: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        paths = sorted(glob.glob(pattern, recursive=True))
        if not paths:
            raise FileNotFoundError(f"glob 没匹配到任何文件: {pattern}")
        results = []
        for p in paths:
            logger.info(f"\n>>> 处理 {p}")
            results.append(self.ingest_file(
                p, purge_existing=purge_existing, batch_size=batch_size,
                cli_publication_year=cli_publication_year,
            ))
        return results

    def list_doc_ids(self, page_size: int = STATS_PAGE_SIZE) -> set:
        """查询集合中所有已存在的 doc_id 集合。

        用于 append 模式下跳过已灌入的文档, 避免重复处理。
        """
        try:
            return self._list_doc_ids_impl(page_size=page_size)
        except Exception as e:
            if _is_transient_rpc_error(e):
                self.reconnect_client()
                return self._list_doc_ids_impl(page_size=page_size)
            raise

    def _list_doc_ids_impl(self, page_size: int = STATS_PAGE_SIZE) -> set:
        doc_ids: set = set()
        out_fields = ["doc_id"]
        rows_iter = None
        try:
            rows_iter = self.client.query_iterator(
                collection_name=self.collection,
                filter="",
                output_fields=out_fields,
                batch_size=page_size,
            )
            while True:
                batch = rows_iter.next()
                if not batch:
                    break
                for r in batch:
                    d = r.get("doc_id", "")
                    if d:
                        doc_ids.add(d)
        except AttributeError:
            offset = 0
            while True:
                batch = self.client.query(
                    collection_name=self.collection,
                    filter="",
                    output_fields=out_fields,
                    limit=page_size,
                    offset=offset,
                )
                if not batch:
                    break
                for r in batch:
                    d = r.get("doc_id", "")
                    if d:
                        doc_ids.add(d)
                if len(batch) < page_size:
                    break
                offset += page_size
        except Exception as e:
            logger.warning(f"[list_doc_ids] 查询失败: {e}")
        finally:
            if rows_iter is not None and hasattr(rows_iter, "close"):
                try:
                    rows_iter.close()
                except Exception:
                    pass
        logger.info(f"[list_doc_ids] 集合中已有 {len(doc_ids)} 个 doc_id")
        return doc_ids

    def stats(self, page_size: int = STATS_PAGE_SIZE) -> Dict[str, Any]:
        try:
            return self._stats_impl(page_size=page_size)
        except Exception as e:
            if _is_transient_rpc_error(e):
                self.reconnect_client()
                return self._stats_impl(page_size=page_size)
            raise

    def _stats_impl(self, page_size: int = STATS_PAGE_SIZE) -> Dict[str, Any]:
        """当前集合统计 (按 doc_id, type 聚合)。

        改用 query_iterator 分页扫描, 不再受 16000 行硬上限影响。
        失败时降级到一次性 query (仅大集合会漏数据)。
        """
        info = self.client.get_collection_stats(self.collection)
        total = info.get("row_count", 0)
        per_doc: Dict[str, Dict[str, Any]] = {}
        out_fields = ["doc_id", "type", "publication_year"]

        scanned = 0
        rows_iter = None
        try:
            rows_iter = self.client.query_iterator(
                collection_name=self.collection,
                filter="",
                output_fields=out_fields,
                batch_size=page_size,
            )
            while True:
                batch = rows_iter.next()
                if not batch:
                    break
                for r in batch:
                    self._stats_accumulate(per_doc, r)
                scanned += len(batch)
        except AttributeError:
            # 老版本 pymilvus 没有 query_iterator, 降级到分页 limit/offset
            logger.debug("[stats] query_iterator 不可用, 降级到 limit/offset")
            offset = 0
            while True:
                batch = self.client.query(
                    collection_name=self.collection,
                    filter="",
                    output_fields=out_fields,
                    limit=page_size,
                    offset=offset,
                )
                if not batch:
                    break
                for r in batch:
                    self._stats_accumulate(per_doc, r)
                scanned += len(batch)
                if len(batch) < page_size:
                    break
                offset += page_size
        except Exception as e:
            logger.warning(
                f"[stats] 分页扫描失败 ({e}), 降级到一次 query, 大集合可能漏数据"
            )
            try:
                batch = self.client.query(
                    collection_name=self.collection,
                    filter="",
                    output_fields=out_fields,
                    limit=page_size,
                )
                for r in batch:
                    self._stats_accumulate(per_doc, r)
                scanned = len(batch)
            except Exception as e2:
                logger.error(f"[stats] fallback 也失败: {e2}")
        finally:
            if rows_iter is not None and hasattr(rows_iter, "close"):
                try:
                    rows_iter.close()
                except Exception:
                    pass

        if total and scanned < total:
            logger.warning(
                f"[stats] 扫描行数 {scanned} < total {total}, 部分 doc 可能未统计"
            )
        return {
            "total": total,
            "scanned": scanned,
            "doc_count": len(per_doc),
            "per_doc": per_doc,
        }

    @staticmethod
    def _stats_accumulate(per_doc: Dict[str, Dict[str, Any]], row: Dict[str, Any]) -> None:
        d = row.get("doc_id", "")
        t = row.get("type", "")
        year = row.get("publication_year") or 0
        entry = per_doc.setdefault(d, {"type": {}, "year": year})
        entry["type"].setdefault(t, 0)
        entry["type"][t] += 1
