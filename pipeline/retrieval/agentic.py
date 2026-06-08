"""Agentic RAG: LLM 路由 + 多路径并行检索 + 模板化上下文 + 生成。

从原始 agentic_rag.py 搬入, 逻辑完全保留。使用 pipeline.models 中的
Pydantic 模型 (RouteDecision, Hit, LocalRetrieveResult) 进行数据验证。
"""

from __future__ import annotations

import datetime
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from ..clients.embedding import EmbeddingClient
from ..clients.llm import LLMClient, DEFAULT_LLM_API_BASE, DEFAULT_LLM_API_KEY, DEFAULT_LLM_MODEL, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS
from ..models import RouteDecision
from ..routing.decision_builder import sanitize_rewrite_keywords
from ..processors.chunker import sanitize_section
from .route_filters import (
    chunk_type_for_route,
    describe_route_chunk_types,
    level1_global_probe_chunk_type,
)
from .progressive_config import (
    DEFAULT_PROGRESSIVE_CONFIG,
    ProgressiveRetrieveConfig,
    chunk_type_skips_summary_l1,
    progressive_config_from_dict,
)
from .hybrid_config import (
    DEFAULT_HYBRID_CONFIG,
    HybridWeightConfig,
    hybrid_config_from_dict,
)
from .hybrid_weights import (
    STAGE_LOCAL_L2,
    STAGE_PROGRESSIVE_L1,
    STAGE_PROGRESSIVE_L1_GLOBAL,
    STAGE_PROGRESSIVE_L2,
    format_weight_log,
    infer_hybrid_weights,
    infer_retrieve_bias_heuristic,
    normalize_retrieve_bias,
)
from .structural_retrieval import (
    STRUCTURAL_CHUNK_TYPES,
    is_structural_chunk_type,
)
from .metadata_match import (
    collect_entity_like_clauses,
    collect_ref_like_clauses,
    score_fig_table_refs,
)
from ..clients.query_format import (
    EMBED_STAGE_PASSAGE,
    EMBED_STAGE_SUMMARY,
    collect_prewarm_embed_texts,
)
from .retrievers import (
    Hit,
    MetadataRetriever,
    VectorRetriever,
    BM25Retriever,
    HybridRetriever,
    _escape_like,
    _escape_eq,
    _row_to_hit,
    _OUTPUT_FIELDS,
    RRF_K,
    DEFAULT_EMBED_API_BASE,
    DEFAULT_EMBED_API_KEY,
    DEFAULT_EMBED_MODEL,
    DEFAULT_MILVUS_TOKEN,
    DEFAULT_MILVUS_URI,
    DEFAULT_COLLECTION,
    DEFAULT_TOP_K,
    DEFAULT_DENSE_WEIGHT,
    DEFAULT_BM25_WEIGHT,
    build_retrievers,
    run_in_parallel,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 路径名 + 常量
# ---------------------------------------------------------------------------

ROUTE_SUMMARY = "summary"
ROUTE_PROGRESSIVE = "progressive"
ROUTE_LOCAL = "local"
ROUTE_NEIGHBOR = "neighbor"   # 邻域扩展回填的伪路由 (related_assets / adjacent / page / similar)
ROUTE_METADATA = "metadata"
VALID_ROUTES = (ROUTE_SUMMARY, ROUTE_PROGRESSIVE, ROUTE_LOCAL, ROUTE_METADATA)
ROUTE_ALIAS = {"global": ROUTE_SUMMARY}

EXTENDED_OUTPUT_FIELDS = list(_OUTPUT_FIELDS)

SUMMARY_TYPE_FILTER = '(type == "summary" or type == "title")'
SUMMARY_CHUNK_FILTER = 'type == "summary"'
TITLE_CHUNK_FILTER = 'type == "title"'
# v6: progressive / local / agentic 默认内容池 = {text, equation, table, image}。
#   - 排除 summary / title: 文献级导航块, L1 定位文献用, 不是事实级内容.
#   - 排除 references: 参考文献单独路由, 对事实/机制类问题是噪声.
#   - 保留 table / image: 大量事实答案 (化学成分/力学性能/规格/图说明) 就在表格与图
#     caption 里; router 无法预知"碳含量是多少"的答案在表格中, 若把 table/image 挡在
#     候选池外, 这些问题永远召不到 gold (见 fact_retrieval 召回低于纯检索的根因).
#     table/image embedding 已做检索增强, 精排交给 reranker, 噪声风险低.
NON_SUMMARY_TYPE_FILTER = (
    '(type != "summary" and type != "title" and type != "references")'
)
# v7: 内容池分离 — 正文 (text/equation) 与图表 (image/table) 分开召回。
# 图表块 caption 短、emb 分常偏高, 与正文混在一个池竞争 top-k 会挤占正文 gold 块;
# 分离后正文有独立配额, 图表只保留少量 (见 ProgressiveRetrieveConfig.structural_content_top_k)。
PROSE_CONTENT_TYPE_FILTER = '(type == "text" or type == "equation")'
STRUCTURAL_CONTENT_TYPE_FILTER = '(type == "image" or type == "table")'
# references/image/table 在候选 doc 内 Milvus query 全量上限
STRUCTURAL_FULL_RECALL_LIMIT = 500
LOCAL_DOC_BM25_MIN_SCORE = 0.12
# v5: chunk_type 路由开放给 router 的可选值 (用于 filters.chunk_type 字段);
# router 主动选 references / equation 即可触发对应专项召回.
VALID_CHUNK_TYPES = (
    "summary", "text", "title", "table", "image", "equation", "references",
)


# ---------------------------------------------------------------------------
# 两级检索结果 (dataclass, 非 LLM 输出)
# ---------------------------------------------------------------------------

@dataclass
class CandidateDoc:
    doc_id: str
    rrf_score: float
    doc_name: str


@dataclass
class LocalRetrieveResult:
    candidate_docs: List[CandidateDoc] = field(default_factory=list)
    chunk_hits: List[Hit] = field(default_factory=list)


# ---------------------------------------------------------------------------
# QueryRouter
# ---------------------------------------------------------------------------

# JSON Schema (Draft-07 兼容): 用于 OpenAI 兼容后端的 response_format 结构化输出
# 约束 router 输出的 JSON 形状, 防止 LLM 自由发挥写出不合法字段。
# strict=False 允许省略可选字段 (与 prompt 里"按需省略空字段"对齐)。
ROUTER_JSON_SCHEMA: Dict[str, Any] = {
    "name": "route_decision",
    "strict": False,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["routes"],
        "properties": {
            "routes": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "string",
                    "enum": ["summary", "progressive", "local", "metadata"],
                },
                "description": "选中的检索路径列表 (可多选)",
            },
            "rewrites": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "description": (
                    "每条路径的检索词数组, key 必须是 routes 中出现过的路径名; "
                    "metadata 路径禁止给 rewrites (输出会被忽略)"
                ),
            },
            "filters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "chunk_type": {
                        "type": "string",
                        "enum": ["image", "table", "equation", "references"],
                        "description": (
                            "类型过滤: image/table 用于图表问题, equation 用于公式问题 "
                            "(\"那个公式\" / \"Hall-Petch 方程\" 等), references 用于"
                            "任何参考文献意图 (\"引用了哪些文献\" / \"看下 references\" / "
                            "\"参考文献里有 X 吗\"); 普通问题省略 — 不填时默认排除 references. "
                            "references 配合 progressive (全库) 或 local (指定/回指文献) 使用, "
                            "系统会全量召回该范围内的 references chunk, 不需要给条目编号."
                        ),
                    },
                    "target_docs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "local 路径的文献名 (整篇标题串)",
                    },
                    "doc_refs": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1},
                        "description": (
                            "已知文献列表的 1-based 编号; 用于 local 路径"
                            "回指会话已检索过的文献, 系统会自动转成 doc_name"
                        ),
                    },
                    "fig_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "图编号, 如 ['1','2']",
                    },
                    "table_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "page_refs": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1},
                        "description": "页码 (1-based)",
                    },
                    "paragraph_refs": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1},
                    },
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "正文子串精确匹配的实体名",
                    },
                    "time": {
                        "type": "string",
                        "description": '时间表达式: "2015-2024" / "2018"',
                    },
                    "retrieve_bias": {
                        "type": "string",
                        "enum": ["semantic", "entity_heavy", "keyword", "balanced"],
                        "description": (
                            "hybrid 检索偏好: semantic/entity_heavy/keyword/balanced; "
                            "仅 progressive/local 生效"
                        ),
                    },
                },
            },
            "retrieve_bias": {
                "type": "string",
                "enum": ["semantic", "entity_heavy", "keyword", "balanced"],
                "description": (
                    "hybrid 检索偏好 (与 filters.retrieve_bias 等价, 顶层优先); "
                    "semantic=概念问句, entity_heavy=专名/化学式/references, "
                    "keyword=短query/编号, balanced=默认"
                ),
            },
            "rerank_mode": {
                "type": "boolean",
                "description": (
                    "可选。仅 true: rerank 用 rewrites 关键词; 省略=用户原话。禁止 false。"
                ),
            },
        },
    },
}


def _router_response_format() -> Dict[str, Any]:
    """构造 OpenAI 兼容的 response_format payload。"""
    return {"type": "json_schema", "json_schema": ROUTER_JSON_SCHEMA}


# 后端拒绝 response_format 时常见的错误关键字 (用于自动降级判断)
_RESPONSE_FORMAT_UNSUPPORTED_HINTS = (
    "response_format",
    "json_schema",
    "unsupported",
    "not support",
    "invalid_request",
    "unrecognized",
)


def _is_response_format_error(err: Exception) -> bool:
    msg = str(err).lower()
    return any(h in msg for h in _RESPONSE_FORMAT_UNSUPPORTED_HINTS)


# ---------------------------------------------------------------------------
# 共享规则块: router 与 reflect 共用同一份"路径/改写/filters"定义,
# 保证两边的输出契约严格一致 (issue #3)
# 内容从 prompts/ 目录加载, 修改 prompt 只需编辑 MD 文件。
# ---------------------------------------------------------------------------

from ..prompts import router_rules as _router_rules_from_file
from ..prompts import render_router_system as _render_router_system_from_file
from ..prompts import generation_system_prompt as _generation_system_prompt_from_file


def ROUTER_RULES_BLOCK(current_year: int) -> str:
    return _router_rules_from_file(current_year)


def ROUTER_SYSTEM_TEMPLATE(current_year: int) -> str:
    return _render_router_system_from_file(current_year)

ROUTER_USER_TEMPLATE = "用户问题: {query}{doc_registry_block}\n\n请输出路由 JSON。"


DEFAULT_REGISTRY_LABEL = "上一轮检索结果中的文献列表"


def _format_doc_registry_block(
    doc_registry: Optional[List[Dict[str, str]]],
    max_items: int = 20,
    label: str = DEFAULT_REGISTRY_LABEL,
) -> str:
    """渲染"已知文献列表"提示块, 供 router/反思器用 doc_refs 回指。

    Args:
        doc_registry: [{doc_id, doc_name}, ...]; 空列表/None 返回空串
            (router 行为回退到无列表的旧路径).
        label: 列表标题. router 默认 "上一轮检索结果中的文献列表"; 反思器
            可传 "本轮已检索到的文献列表" 区分语义 —— 用户口中的"第X篇"
            必须严格锚定到上一轮的最终结果, 不是全会话累计也不是本轮中间
            态。这两类列表的语义区分是该函数 label 参数的全部目的。
    """
    if not doc_registry:
        return ""
    lines = ["", "", f"【{label} (按编号; 1-based)】"]
    pinned_nums: List[str] = []
    for i, entry in enumerate(doc_registry[:max_items], 1):
        name = entry.get("doc_name") or entry.get("doc_id") or "(unknown)"
        is_pinned = bool(entry.get("pinned"))
        marker = " [pinned]" if is_pinned else ""
        if is_pinned:
            pinned_nums.append(str(i))
        lines.append(f"{i}. {name}{marker}")
    if len(doc_registry) > max_items:
        lines.append(f"... (共 {len(doc_registry)} 篇, 已截断到前 {max_items})")
    if pinned_nums:
        lines.append(
            f"(注: [pinned] 标记的是用户之前明确回指过的文献, 当本轮用户用 '上面那篇/它' "
            f"等模糊代词时, 优先指向这些 pinned 项, 编号 {', '.join(pinned_nums)})"
        )
    lines.append("(若用户回指上述某篇, 触发 local 并填 filters.doc_refs: [编号])")
    return "\n".join(lines)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_FIG_REF_LITE_RE = re.compile(r"(?:Fig(?:ure|\.)?|图)\s*([0-9IVXivx]+)", re.IGNORECASE)
_TAB_REF_LITE_RE = re.compile(r"(?:Table|表)\s*([0-9IVXivx]+)", re.IGNORECASE)

# --- 实验性开关辅助 (仅在对应 config flag 开启时被调用) ---
# A3: 钢牌号识别 (耐候钢语料常见: Q450NQR1 / Q345 / 09CuPCrNi / 16Mn / SPA-H)
_STEEL_GRADE_RE = re.compile(
    r"(?:Q\d{3}[A-Za-z0-9]*|\d{2}Cu[A-Za-z0-9]*|\d{2}Mn[A-Za-z0-9]*|SPA[-\s]?H)",
    re.IGNORECASE,
)
# B1: 导航块识别 (Abstract/摘要/Keywords/关键词 开头; 含可选 [Section] 等前缀标记)
_NAV_CHUNK_RE = re.compile(
    r"^\s*(?:\[[^\]]*\]\s*)*(?:abstract|摘\s*要|key\s*words?|关\s*键\s*词)\b",
    re.IGNORECASE,
)
_NAV_SECTION_VALUES = frozenset(
    {"abstract", "摘要", "keywords", "key words", "关键词"}
)


def _extract_steel_grades(query: str) -> List[str]:
    return sorted({m.group(0) for m in _STEEL_GRADE_RE.finditer(query or "")})


def _is_nav_chunk(hit: "Hit") -> bool:
    """判断是否为 summary / Abstract / Keywords 导航块 (非事实答案载体)。"""
    if (getattr(hit, "type", "") or "").strip().lower() == "summary":
        return True
    head = (getattr(hit, "content", "") or "")[:80]
    if _NAV_CHUNK_RE.search(head):
        return True
    sec = (getattr(hit, "section", "") or "").strip().lower()
    return sec in _NAV_SECTION_VALUES
_RECENT_N_YEARS_RE = re.compile(
    r"(?:最近|近)\s*([一二三四五六七八九十百千0-9]+)\s*年|past\s+(\d+)\s*years?|recent\s+(\d+)\s*years?",
    re.IGNORECASE,
)
_SINCE_YEAR_RE = re.compile(r"(?:since|after|自|从)\s*(\d{4})", re.IGNORECASE)
_BEFORE_YEAR_RE = re.compile(r"(?:before|prior to|之前)\s*(\d{4})", re.IGNORECASE)
_RANGE_YEAR_RE = re.compile(r"(\d{4})\s*[-~到至]\s*(\d{4})")
_SINGLE_YEAR_RE = re.compile(r"\b(19[0-9]{2}|20[0-9]{2})\b")
_CN_NUM_MAP = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7,
               "八": 8, "九": 9, "十": 10}
_SECTION_REF_RE = re.compile(
    r"第\s*[0-9一二三四五六七八九十百]+\s*(?:段|节|章|部分|section|paragraph)",
    re.IGNORECASE,
)
_DOC_REF_RE = re.compile(
    r"(?:那篇|这篇|该|此|上面|之前)(?:的|那|这)?(?:文献|论文|文章|paper|article|document)",
    re.IGNORECASE,
)
# references 模式专用的宽松指代识别: 命中"上面那篇的 references"这类无明确名词的表达.
# 比 _DOC_REF_RE 宽松一档 — 允许指代词后只跟"的", 不强求接"文献/论文/article"等名词;
# 仅在 references 意图命中时使用, 避免对普通问句过度匹配.
_LOOSE_DOC_REF_RE = re.compile(
    r"(?:那篇|这篇|该|此|它|这个|那个|上面|之前|上一|刚才|前面|前一)(?:的|那|这)?",
    re.IGNORECASE,
)
# 页码: "第 3 页" / "page 3" / "p.3" / "p 3"
_PAGE_REF_RE = re.compile(
    r"第\s*([0-9一二三四五六七八九十百]+)\s*页|"
    r"\bpage[s]?\s*([0-9]+)\b|"
    r"\bp\.?\s*([0-9]+)\b",
    re.IGNORECASE,
)
# 段落: "第 3 段" / "第 3 自然段" / "paragraph 3" / "para 3"
_PARA_REF_RE = re.compile(
    r"第\s*([0-9一二三四五六七八九十百]+)\s*(?:自然)?段|"
    r"\bparagraph[s]?\s*([0-9]+)\b|"
    r"\bpara\.?\s*([0-9]+)\b",
    re.IGNORECASE,
)
# 实体精确查询: 引号引起的字符串 (中英文引号都接受);
# 通常用户用引号就是想要 "在 content 中精确匹配该术语"
_ENTITY_QUOTE_RE = re.compile(
    r"[\"'\u201c\u2018]([^\"'\u201d\u2019\n]{2,40})[\"'\u201d\u2019]",
)

# v5: 参考文献意图识别 (chunk_type=references)
# 命中即视为"用户想看参考文献", 路由设 chunk_type=references 后由 progressive/local
# 直接全量召回该范围内的 references chunk (不再按编号精确过滤 — 用户场景里精确编号
# 占比很低, 简化为意图级路由).
_REFERENCES_HINT_RE = re.compile(
    r"(参\s*考\s*文\s*献|引\s*用\s*文\s*献|引\s*用\s*的?\s*文\s*献|"
    r"\bbibliograph(?:y|ies)\b|\breferences?\b|\brefs?\b|"
    r"\bcited\s+(?:works?|literature)\b|引\s*文)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# doc_registry 解析 helper (供 router 把 target_docs 字符串归一为规范 doc_id)
# ---------------------------------------------------------------------------

# 模糊匹配时去除的标点/空白 (中英常见分隔符)
_REGISTRY_NORM_RE = re.compile(r"[\s,，;；/\\\.\-_:：()（）\[\]【】]+")


def _normalize_doc_name_for_match(name: str) -> str:
    """归一化 doc_name 用于子串比较: 小写 + 去标点空白."""
    if not name:
        return ""
    return _REGISTRY_NORM_RE.sub("", str(name).lower())


def _match_registry_entry(
    name: str,
    doc_registry: Optional[List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """在 doc_registry 中按 doc_name 做模糊匹配, 返回规范 entry (含 doc_id/doc_name).

    匹配策略 (按优先级):
      1. doc_id 完全相等
      2. doc_name 完全相等 (大小写不敏感, 去标点空白)
      3. 互为子串 (LLM 给的标题往往是 registry 名的子串或超集)

    任一匹配命中即返回; 都不命中返回 None.
    """
    if not name or not doc_registry:
        return None
    target_norm = _normalize_doc_name_for_match(name)
    if not target_norm:
        return None
    # 第 1 轮: doc_id / 完全相等
    for entry in doc_registry:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("doc_id") or "") == name:
            return entry
        if _normalize_doc_name_for_match(str(entry.get("doc_name") or "")) == target_norm:
            return entry
    # 第 2 轮: 互为子串 (长度 >= 4 才认为有意义, 避免短词误匹配)
    if len(target_norm) < 4:
        return None
    for entry in doc_registry:
        if not isinstance(entry, dict):
            continue
        entry_norm = _normalize_doc_name_for_match(str(entry.get("doc_name") or ""))
        if not entry_norm or len(entry_norm) < 4:
            continue
        if target_norm in entry_norm or entry_norm in target_norm:
            return entry
    return None


def _pick_focus_doc(
    doc_registry: Optional[List[Dict[str, Any]]],
) -> Optional[Tuple[Dict[str, Any], str]]:
    """user 用代词 ('这篇/它/上面那篇') + local 但没填 docs/refs 时, 选一篇 focus doc.

    返回 (entry, reason); 无法明确锁定时返回 None.

    优先级 (保守, 宁可放弃也不乱锚定):
      1. registry 中有且仅有 1 个 pinned 项 → 该项 (用户先前已显式回指过)
      2. registry 总共只有 1 个 entry → 该项 (单文献会话, 无歧义)
      其余情况 (多个 pinned / 多个 entry 都未 pin / 空 registry) 返回 None.
    """
    if not doc_registry:
        return None
    pinned = [e for e in doc_registry if isinstance(e, dict) and e.get("pinned")]
    if len(pinned) == 1:
        return pinned[0], "single-pinned"
    valid = [e for e in doc_registry if isinstance(e, dict) and e.get("doc_id")]
    if len(valid) == 1:
        return valid[0], "single-entry"
    return None


def _extract_json_blob(text: str) -> Optional[str]:
    if not text:
        return None
    from ..clients.thinking_utils import strip_think_blocks

    text = strip_think_blocks(text)
    if not text:
        return None
    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    return None


def _parse_cn_num(token: str) -> Optional[int]:
    token = token.strip()
    if token.isdigit():
        return int(token)
    if "十" in token:
        if token == "十":
            return 10
        if token.startswith("十"):
            return 10 + _CN_NUM_MAP.get(token[1:], 0)
        if token.endswith("十"):
            return _CN_NUM_MAP.get(token[:-1], 0) * 10
        a, b = token.split("十", 1)
        return _CN_NUM_MAP.get(a, 0) * 10 + _CN_NUM_MAP.get(b, 0)
    return _CN_NUM_MAP.get(token)


def _heuristic_time_str(query: str, current_year: int) -> str:
    m = _RANGE_YEAR_RE.search(query)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = _SINCE_YEAR_RE.search(query)
    if m:
        return f"{m.group(1)}-{current_year}"
    m = _BEFORE_YEAR_RE.search(query)
    if m:
        return f"1900-{m.group(1)}"
    m = _RECENT_N_YEARS_RE.search(query)
    if m:
        for g in m.groups():
            if g:
                n = _parse_cn_num(g) if not g.isdigit() else int(g)
                if n:
                    return f"{current_year - n}-{current_year}"
    m = _SINGLE_YEAR_RE.search(query)
    if m:
        return m.group(1)
    return ""


def _extract_page_refs(query: str) -> List[int]:
    """启发式提取 query 中的 "第 X 页 / page X" 数字, 返回 1-based 整数列表。"""
    out: List[int] = []
    for m in _PAGE_REF_RE.finditer(query):
        for g in m.groups():
            if not g:
                continue
            if g.isdigit():
                out.append(int(g))
            else:
                v = _parse_cn_num(g)
                if v:
                    out.append(v)
            break
    # 去重保持顺序
    seen: set = set()
    uniq: List[int] = []
    for x in out:
        if x not in seen and x >= 1:
            seen.add(x)
            uniq.append(x)
    return uniq


def _extract_paragraph_refs(query: str) -> List[int]:
    """启发式提取 "第 X 段 / paragraph X" 数字, 返回 1-based 整数列表。"""
    out: List[int] = []
    for m in _PARA_REF_RE.finditer(query):
        for g in m.groups():
            if not g:
                continue
            if g.isdigit():
                out.append(int(g))
            else:
                v = _parse_cn_num(g)
                if v:
                    out.append(v)
            break
    seen: set = set()
    uniq: List[int] = []
    for x in out:
        if x not in seen and x >= 1:
            seen.add(x)
            uniq.append(x)
    return uniq


def _extract_entities(query: str) -> List[str]:
    """启发式: 抓 "查/找/包含 + 引号引起的实体名"。"""
    out: List[str] = []
    seen: set = set()
    for m in _ENTITY_QUOTE_RE.finditer(query):
        s = (m.group(1) or "").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _heuristic_fallback_decision(query: str, current_year: int) -> RouteDecision:
    fig_refs = sorted({m.group(1).upper() for m in _FIG_REF_LITE_RE.finditer(query)})
    tab_refs = sorted({m.group(1).upper() for m in _TAB_REF_LITE_RE.finditer(query)})
    has_section_ref = bool(_SECTION_REF_RE.search(query))
    has_doc_ref = bool(_DOC_REF_RE.search(query))
    page_refs = _extract_page_refs(query)
    paragraph_refs = _extract_paragraph_refs(query)
    entities = _extract_entities(query)
    has_refs_intent = bool(_REFERENCES_HINT_RE.search(query))

    chunk_type: Optional[str] = None
    # references 比 image/table 优先级高 — 用户问参考文献时, [N] 之类的中括号
    # 编号既不是图表也不是页/段, 应统一识为 references chunk 召回意图.
    if has_refs_intent:
        chunk_type = "references"
        fig_refs = []  # 抑制 references 模式下的 fig/tab 误抓
        tab_refs = []
        # references 模式下放宽指代识别 — "上面那篇的 references" / "这篇的引用"
        # 这类常见表达 doc_ref 名词省略, _DOC_REF_RE 不命中, 在这里补一次
        if not has_doc_ref and _LOOSE_DOC_REF_RE.search(query):
            has_doc_ref = True
    elif fig_refs and not tab_refs:
        chunk_type = "image"
    elif tab_refs and not fig_refs:
        chunk_type = "table"

    routes: List[str] = []
    rewrites: Dict[str, str] = {}

    is_summary_q = bool(re.search(r"总结|汇总|概述|综述|对比|主要内容|主要贡献|summarize|overview|main", query, re.IGNORECASE))

    if is_summary_q:
        routes.append(ROUTE_SUMMARY)
        rewrites[ROUTE_SUMMARY] = query
    else:
        if has_doc_ref:
            routes.append(ROUTE_LOCAL)
            rewrites[ROUTE_LOCAL] = query
        else:
            routes.append(ROUTE_PROGRESSIVE)
            rewrites[ROUTE_PROGRESSIVE] = query
        # 摘要类不需要额外走 summary

    # 明确提到图/表/页/段/实体 时才走 metadata
    # references 意图不再走 metadata — chunk_type=references 已经驱动 progressive/local
    # 在 references 池里全量召回, 编号精确匹配是少见场景, 不再单独建路径
    if fig_refs or tab_refs or page_refs or paragraph_refs or entities:
        routes.append(ROUTE_METADATA)
        rewrites[ROUTE_METADATA] = query

    return RouteDecision(
        routes=routes, rewrites=rewrites,
        time=_heuristic_time_str(query, current_year),
        chunk_type=chunk_type,
        fig_refs=fig_refs,
        table_refs=tab_refs,
        page_refs=page_refs,
        paragraph_refs=paragraph_refs,
        entities=entities,
        retrieve_bias=infer_retrieve_bias_heuristic(query, chunk_type=chunk_type),
        reasoning="(fallback)",
    )


@dataclass
class RouterMetrics:
    """路由器调用指标 (累计计数, 进程级)。"""
    total: int = 0                # 总调用次数
    llm_called: int = 0           # 实际发起 LLM 调用 (非纯 fallback) 的次数
    llm_failed: int = 0           # LLM 调用抛异常
    json_missing: int = 0         # 响应里找不到 JSON
    json_invalid: int = 0         # JSON 解析失败
    fallback_used: int = 0        # 走启发式 fallback 的次数
    success: int = 0              # 成功解析出有效 RouteDecision 的次数
    json_schema_disabled: bool = False  # 后端不支持 json_schema, 已自动降级
    by_route: Dict[str, int] = field(default_factory=dict)  # 每条路径被选中次数

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "llm_called": self.llm_called,
            "llm_failed": self.llm_failed,
            "json_missing": self.json_missing,
            "json_invalid": self.json_invalid,
            "fallback_used": self.fallback_used,
            "success": self.success,
            "fallback_ratio": (self.fallback_used / self.total) if self.total else 0.0,
            "json_schema_disabled": self.json_schema_disabled,
            "by_route": dict(self.by_route),
        }


class QueryRouter:
    """LLM 驱动的路由器 + 多路径 query 改写器 + 时间过滤解析。

    metrics 字段累计 fallback / json 失败等计数, 可在外层周期性 dump。
    """

    def __init__(
        self,
        llm: Optional[LLMClient],
        temperature: float = 0.0,
        max_tokens: int = 300,
        current_year: Optional[int] = None,
        use_json_schema: bool = False,
        history_turns: int = 1,
        disable_thinking: bool = True,
    ) -> None:
        """
        Args:
            use_json_schema: 是否下发 response_format=json_schema 约束 LLM 输出.
                后端不支持时会自动一次性降级 (剥掉 response_format 重试),
                后续请求记住状态不再尝试.
            history_turns: 路由器只看最近 N 轮 (1 轮 = 1 user + 1 assistant) 历史用于
                代词指代消解; 太多反而拖慢 router prompt 处理. 0 = 完全不传 history.
            disable_thinking: 是否关闭 LLM 思考模式 (默认 True).
        """
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.current_year = current_year or datetime.datetime.now().year
        self.use_json_schema = use_json_schema
        self.history_turns = max(0, int(history_turns))
        self.disable_thinking = disable_thinking
        self.metrics = RouterMetrics()

    def _truncate_history(
        self, history: Optional[List[Dict[str, str]]],
    ) -> Optional[List[Dict[str, str]]]:
        """只取最近 N 轮 user+assistant, 防止 router prompt 累积越来越长."""
        if not history or self.history_turns <= 0:
            return None
        max_msgs = 2 * self.history_turns
        tail = list(history[-max_msgs:])
        while tail and tail[0].get("role") != "user":
            tail = tail[1:]
        return tail or None

    def _system_prompt(self) -> str:
        return ROUTER_SYSTEM_TEMPLATE(self.current_year)

    def _normalize_routes(self, routes: Any) -> List[str]:
        if not isinstance(routes, list):
            return []
        out: List[str] = []
        for r in routes:
            if not isinstance(r, str):
                continue
            r = ROUTE_ALIAS.get(r, r)
            if r in VALID_ROUTES and r not in out:
                out.append(r)
        return out

    def _validate_decision(
        self, raw_decision: Dict[str, Any], raw_text: str, query: str,
        doc_registry: Optional[List[Dict[str, str]]] = None,
    ) -> RouteDecision:
        routes = self._normalize_routes(raw_decision.get("routes"))
        if not routes:
            routes = [ROUTE_PROGRESSIVE]

        rewrites_in = raw_decision.get("rewrites") or {}
        if not isinstance(rewrites_in, dict):
            rewrites_in = {}
        if "global" in rewrites_in and ROUTE_SUMMARY not in rewrites_in:
            rewrites_in[ROUTE_SUMMARY] = rewrites_in.pop("global")

        clean: Dict[str, str] = {}
        for route in routes:
            # metadata 路径硬约束: 严禁使用 rewrite 关键词, 即使模型输出了也忽略 (issue #2)
            if route == ROUTE_METADATA:
                continue
            val = rewrites_in.get(route)
            if isinstance(val, str) and val.strip():
                clean[route] = val.strip()
            elif isinstance(val, list):
                # prompt 要求 LLM 输出关键词数组, 这里按空格拼接为单字符串供下游检索器消费
                kws = sanitize_rewrite_keywords(
                    [str(v).strip() for v in val if v and str(v).strip()]
                )
                if kws:
                    clean[route] = " ".join(kws)
            elif isinstance(val, dict):
                # 兼容历史格式: {"keywords": [...]} / {"target_docs": [...]} 等
                for v in val.values():
                    if isinstance(v, list) and v:
                        kws = sanitize_rewrite_keywords(
                            [str(x).strip() for x in v if x and str(x).strip()]
                        )
                        if kws:
                            clean[route] = " ".join(kws)
                            break
                    if isinstance(v, str) and v.strip():
                        clean[route] = v.strip()
                        break
            if route not in clean:
                clean[route] = query
        if rewrites_in.get(ROUTE_METADATA):
            logger.info(
                f"[router] 忽略模型为 metadata 路径输出的 rewrites (硬约束): "
                f"{rewrites_in.get(ROUTE_METADATA)!r}"
            )

        # 新格式: filters 嵌套对象; 老格式: 顶层平铺 (兼容).
        # 解析时优先从 filters 取, 缺失再回退到顶层, 再回退到 rewrites.metadata 的 dict 格式 (更老).
        filters_in = raw_decision.get("filters")
        if not isinstance(filters_in, dict):
            filters_in = {}

        def _coerce_int_list(raw: Any) -> List[int]:
            if not isinstance(raw, list):
                return []
            out: List[int] = []
            for x in raw:
                try:
                    v = int(x)
                except (TypeError, ValueError):
                    continue
                if v >= 1 and v not in out:
                    out.append(v)
            return out

        def _coerce_str_list(raw: Any, *, upper: bool = False) -> List[str]:
            if not isinstance(raw, list):
                return []
            out: List[str] = []
            seen: set = set()
            for x in raw:
                s = str(x).strip()
                if not s:
                    continue
                if upper:
                    s = s.upper()
                if s not in seen:
                    seen.add(s)
                    out.append(s)
            return out

        def _pick(name: str) -> Any:
            """优先 filters[name], 其次顶层 raw_decision[name]."""
            if name in filters_in and filters_in[name] not in (None, "", []):
                return filters_in[name]
            return raw_decision.get(name)

        target_docs = _coerce_str_list(_pick("target_docs"))
        if not target_docs:
            lr = rewrites_in.get(ROUTE_LOCAL)
            if isinstance(lr, dict):
                target_docs = _coerce_str_list(lr.get("target_docs"))

        # doc_refs (1-based 编号) → 用 doc_registry 查表追加 doc_name (issue #1)
        # 同步收集规范 doc_id (P0-B): 检索层可直接走 doc_id 短路, 不必再 name 解析.
        target_doc_ids: List[str] = []
        doc_refs = _coerce_int_list(_pick("doc_refs"))
        if doc_refs and doc_registry:
            seen = set(target_docs)
            for ref in doc_refs:
                idx = ref - 1
                if 0 <= idx < len(doc_registry):
                    entry = doc_registry[idx] if isinstance(doc_registry[idx], dict) else {}
                    name = str(entry.get("doc_name") or entry.get("doc_id") or "")
                    did = str(entry.get("doc_id") or "")
                    if name and name not in seen:
                        seen.add(name)
                        target_docs.append(name)
                    if did and did not in target_doc_ids:
                        target_doc_ids.append(did)
                else:
                    logger.warning(
                        f"[router] doc_refs={ref} 越界 (registry 共 {len(doc_registry)} 篇), 已忽略"
                    )
        elif doc_refs and not doc_registry:
            logger.warning(
                f"[router] 模型输出了 doc_refs={doc_refs} 但当前会话无 doc_registry, 已忽略"
            )

        # P0-A/B: 把 LLM 给出的 target_docs 字符串通过 registry 模糊匹配为规范 doc_id.
        # 场景: LLM 把上一轮 answer 里的标题 (可能与 Milvus 实际 doc_name 不一致, 例如
        # 中英翻译或缩写) 抄进 target_docs; 不做归一会让 _locate_docs_by_name 走到
        # BM25 兜底, 容易扩散到无关文献.
        if doc_registry and target_docs:
            canonical: List[str] = []
            seen_names: set = set()
            for raw_name in target_docs:
                match = _match_registry_entry(raw_name, doc_registry)
                if match:
                    canon_name = str(match.get("doc_name") or match.get("doc_id") or raw_name)
                    did = str(match.get("doc_id") or "")
                    if canon_name and canon_name not in seen_names:
                        seen_names.add(canon_name)
                        canonical.append(canon_name)
                    if did and did not in target_doc_ids:
                        target_doc_ids.append(did)
                    if canon_name != raw_name:
                        logger.info(
                            f"[router] target_docs registry 归一: "
                            f"{raw_name!r} → {canon_name!r} (doc_id={did})"
                        )
                else:
                    # 没在 registry 里 — 保留原字符串, 由 _locate_docs_by_name 继续兜底
                    if raw_name and raw_name not in seen_names:
                        seen_names.add(raw_name)
                        canonical.append(raw_name)
            target_docs = canonical

        fig_refs = _coerce_str_list(_pick("fig_refs"), upper=True)
        table_refs = _coerce_str_list(_pick("table_refs"), upper=True)
        page_refs = _coerce_int_list(_pick("page_refs"))
        paragraph_refs = _coerce_int_list(_pick("paragraph_refs"))
        entities = _coerce_str_list(_pick("entities"))

        # rewrites.metadata 是 dict 的老格式兼容
        if not (fig_refs and table_refs and page_refs and paragraph_refs
                and entities):
            mr = rewrites_in.get(ROUTE_METADATA)
            if isinstance(mr, dict):
                if not fig_refs:
                    fig_refs = _coerce_str_list(mr.get("fig_refs"), upper=True)
                if not table_refs:
                    table_refs = _coerce_str_list(mr.get("table_refs"), upper=True)
                if not page_refs:
                    page_refs = _coerce_int_list(mr.get("page_refs"))
                if not paragraph_refs:
                    paragraph_refs = _coerce_int_list(mr.get("paragraph_refs"))
                if not entities:
                    entities = _coerce_str_list(mr.get("entities"))

        # 启发式补足 page/paragraph/entities (LLM 没给但 query 里能直接抽到时)
        if not page_refs:
            page_refs = _extract_page_refs(query)
        if not paragraph_refs:
            paragraph_refs = _extract_paragraph_refs(query)
        if not entities:
            entities = _extract_entities(query)

        # time: filters.time → 顶层 time → 顶层 time_range dict → 启发式
        time_val = ""
        t_raw = _pick("time")
        if isinstance(t_raw, str) and t_raw.strip():
            time_val = t_raw.strip()
        if not time_val:
            tr_in = raw_decision.get("time_range")
            if isinstance(tr_in, dict):
                ys, ye = tr_in.get("year_start"), tr_in.get("year_end")
                if isinstance(ys, (int, float)) and ys > 0:
                    ye = int(ye) if isinstance(ye, (int, float)) and ye > 0 else self.current_year
                    time_val = f"{int(ys)}-{ye}"
        if not time_val:
            time_val = _heuristic_time_str(query, self.current_year)

        # chunk_type: filters.chunk_type → 顶层 chunk_type → 启发式推断
        ct_raw = _pick("chunk_type")
        chunk_type_val: Optional[str] = None
        if isinstance(ct_raw, str) and ct_raw.lower() in VALID_CHUNK_TYPES:
            chunk_type_val = ct_raw.lower()
        # 兜底推断: LLM 没给 chunk_type 但 query 里有 references 关键词时,
        # 默认设 chunk_type=references (确保用户问参考文献时一定能拿到 references chunk)
        if chunk_type_val is None:
            if _REFERENCES_HINT_RE.search(query):
                chunk_type_val = "references"
            elif fig_refs and not table_refs:
                chunk_type_val = "image"
            elif table_refs and not fig_refs:
                chunk_type_val = "table"

        # P0-A: 代词自动锚定 — local / metadata / 结构化 progressive 在未填 docs/refs 时,
        # 从 registry 选唯一 focus doc (single-pinned 或 single-entry), 避免全库 metadata
        # 或 references 结构化召回.
        if (
            not target_doc_ids
            and not target_docs
            and doc_registry
            and (_DOC_REF_RE.search(query) or _LOOSE_DOC_REF_RE.search(query))
        ):
            anchor_route = ""
            if ROUTE_LOCAL in routes:
                anchor_route = "local"
            elif ROUTE_METADATA in routes:
                anchor_route = "metadata"
            elif ROUTE_PROGRESSIVE in routes and (
                chunk_type_val in ("references", "image", "table", "equation")
                or _REFERENCES_HINT_RE.search(query)
            ):
                anchor_route = f"progressive(ctype={chunk_type_val or 'references'})"
            if anchor_route:
                picked = _pick_focus_doc(doc_registry)
                if picked is not None:
                    entry, reason = picked
                    did = str(entry.get("doc_id") or "")
                    name = str(entry.get("doc_name") or did)
                    if did:
                        target_doc_ids.append(did)
                    if name:
                        target_docs.append(name)
                    logger.info(
                        f"[router] 代词回指 + {anchor_route} 路由 (target_docs/refs 均空), "
                        f"自动锚定 registry focus doc: doc_id={did} reason={reason}"
                    )

        # 如果路由器漏选 metadata 但启发式 / LLM 给出了 page/paragraph/entities/refs,
        # 这里补一条 metadata 路径, 保证硬约束被使用
        if (page_refs or paragraph_refs or entities or fig_refs or table_refs) \
                and ROUTE_METADATA not in routes:
            routes.append(ROUTE_METADATA)

        # metadata 没有任何 filter 时强制踢出, 避免空跑 (issue #2)
        if ROUTE_METADATA in routes and not (
            fig_refs or table_refs or page_refs or paragraph_refs or entities
        ):
            logger.info("[router] metadata 路径无任何 filter, 已剔除")
            routes = [r for r in routes if r != ROUTE_METADATA]
            if not routes:
                routes = [ROUTE_PROGRESSIVE]
                if ROUTE_PROGRESSIVE not in clean:
                    clean[ROUTE_PROGRESSIVE] = query

        # metadata 永远不需要 rewrite, 二次清理 (issue #2)
        clean.pop(ROUTE_METADATA, None)

        rb_raw = _pick("retrieve_bias") or raw_decision.get("retrieve_bias")
        retrieve_bias = normalize_retrieve_bias(rb_raw)
        rerank_mode = True if raw_decision.get("rerank_mode") is True else None

        # 邻域扩展模式: 优先顶层, 回退 filters; 仅保留合法值
        expand_raw = raw_decision.get("expand_neighbors")
        if expand_raw is None:
            expand_raw = filters_in.get("expand_neighbors")
        expand_neighbors: List[str] = []
        if isinstance(expand_raw, list):
            for x in expand_raw:
                m = str(x).strip().lower()
                if m in ("assets", "adjacent", "page", "similar") and m not in expand_neighbors:
                    expand_neighbors.append(m)

        return RouteDecision(
            routes=routes, rewrites=clean, time=time_val,
            chunk_type=chunk_type_val,
            target_docs=target_docs,
            target_doc_ids=target_doc_ids,
            fig_refs=fig_refs,
            table_refs=table_refs,
            page_refs=page_refs,
            paragraph_refs=paragraph_refs,
            entities=entities,
            retrieve_bias=retrieve_bias,
            rerank_mode=rerank_mode,
            expand_neighbors=expand_neighbors,
            reasoning=str(raw_decision.get("reasoning", "")),
            raw_response=raw_text,
        )

    def route(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        doc_registry: Optional[List[Dict[str, str]]] = None,
    ) -> RouteDecision:
        """路由决策。

        Args:
            doc_registry: 当前会话已检索/引用过的文献列表
                ([{doc_id, doc_name}, ...]); 提供时会以"按编号列表"渲染进
                user prompt, 让模型可以输出 filters.doc_refs 回指 (issue #1)
        """
        self.metrics.total += 1
        if self.llm is None:
            self.metrics.fallback_used += 1
            decision = _heuristic_fallback_decision(query, self.current_year)
            self._track_routes(decision)
            return decision

        self.metrics.llm_called += 1
        messages: List[Dict[str, str]] = [{"role": "system", "content": self._system_prompt()}]
        truncated_history = self._truncate_history(history)
        if truncated_history:
            messages.extend(truncated_history)
        registry_block = _format_doc_registry_block(doc_registry)
        messages.append({
            "role": "user",
            "content": ROUTER_USER_TEMPLATE.format(
                query=query, doc_registry_block=registry_block,
            ),
        })

        call_kwargs: Dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "disable_thinking": self.disable_thinking,
        }
        if self.use_json_schema and not self.metrics.json_schema_disabled:
            call_kwargs["response_format"] = _router_response_format()

        try:
            result = self.llm.chat_messages(messages, **call_kwargs)
            raw = result.get("answer", "")
        except Exception as e:
            # 后端不支持 response_format=json_schema 时一次性降级, 后续不再尝试
            if "response_format" in call_kwargs and _is_response_format_error(e):
                self.metrics.json_schema_disabled = True
                logger.warning(
                    f"[router] 后端不支持 json_schema response_format, 自动降级并重试. err={e}"
                )
                call_kwargs.pop("response_format", None)
                try:
                    result = self.llm.chat_messages(messages, **call_kwargs)
                    raw = result.get("answer", "")
                except Exception as e2:
                    self.metrics.llm_failed += 1
                    self.metrics.fallback_used += 1
                    logger.warning(f"[router] LLM 调用失败 (降级后仍失败), 走 fallback: {e2}")
                    decision = _heuristic_fallback_decision(query, self.current_year)
                    self._track_routes(decision)
                    return decision
            else:
                self.metrics.llm_failed += 1
                self.metrics.fallback_used += 1
                logger.warning(f"[router] LLM 调用失败, 走 fallback: {e}")
                decision = _heuristic_fallback_decision(query, self.current_year)
                self._track_routes(decision)
                return decision
        blob = _extract_json_blob(raw)
        if not blob:
            self.metrics.json_missing += 1
            self.metrics.fallback_used += 1
            logger.warning(
                f"[router] 未找到 JSON, 走 fallback. raw (前 800 字符): {raw[:800]!r}"
            )
            decision = _heuristic_fallback_decision(query, self.current_year)
            self._track_routes(decision)
            return decision
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError as e:
            self.metrics.json_invalid += 1
            self.metrics.fallback_used += 1
            logger.warning(
                f"[router] JSON 解析失败 ({e}), 走 fallback. blob (前 800 字符): {blob[:800]!r}"
            )
            decision = _heuristic_fallback_decision(query, self.current_year)
            self._track_routes(decision)
            return decision
        decision = self._validate_decision(parsed, raw, query, doc_registry=doc_registry)
        self.metrics.success += 1
        self._track_routes(decision)
        return decision

    def _track_routes(self, decision: RouteDecision) -> None:
        for r in decision.routes or []:
            self.metrics.by_route[r] = self.metrics.by_route.get(r, 0) + 1


# ---------------------------------------------------------------------------
# Filter 工具
# ---------------------------------------------------------------------------

def _and_filter(*parts: Optional[str]) -> Optional[str]:
    pieces = [f"({p})" for p in parts if p]
    return " and ".join(pieces) if pieces else None


# ---------------------------------------------------------------------------
# SummaryRetriever
# ---------------------------------------------------------------------------

class SummaryRetriever:
    """全局俯瞰: summary/title 分池 + vector/bm25 双路各 top-K, 按原始分合并截断。"""

    def __init__(
        self,
        vec_retriever: VectorRetriever,
        rrf_k: int = RRF_K,
        bm25_retriever: Optional[BM25Retriever] = None,
        *,
        per_path_k: Optional[int] = None,
    ) -> None:
        self.vec = vec_retriever
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k  # 保留参数兼容; 已不再用于 summary 路径
        self.per_path_k = per_path_k

    @staticmethod
    def _merge_dual_path_chunks(
        vec_hits: List[Hit],
        bm25_hits: List[Hit],
        *,
        type_tag: str,
    ) -> List[Hit]:
        """同 type 池内 vector/bm25 各一路: 按 chunk pk 取 max(score) 合并。"""
        merged: Dict[str, Hit] = {}
        for source, hits in (("vector", vec_hits), ("bm25", bm25_hits)):
            for h in hits:
                key = h.pk or h.chunk_id
                if not key:
                    continue
                if key in merged:
                    existing = merged[key]
                    if source not in existing.sources:
                        existing.sources.append(source)
                    if h.score > existing.score:
                        existing.score = h.score
                else:
                    h.sources = [source, ROUTE_SUMMARY, type_tag]
                    h.rrf_score = h.score
                    merged[key] = h
        return sorted(merged.values(), key=lambda x: -x.score)

    @staticmethod
    def _merge_type_pools(*pools: List[Hit]) -> List[Hit]:
        """summary 池与 title 池并集, 同 pk 保留更高 score。"""
        by_pk: Dict[str, Hit] = {}
        for hits in pools:
            for h in hits:
                key = h.pk or h.chunk_id
                if not key:
                    continue
                if key in by_pk:
                    prev = by_pk[key]
                    if h.score > prev.score:
                        by_pk[key] = h
                    else:
                        for s in h.sources:
                            if s not in prev.sources:
                                prev.sources.append(s)
                else:
                    by_pk[key] = h
        return sorted(by_pk.values(), key=lambda x: -x.score)

    def _retrieve_type_pool_dual_path(
        self,
        query: str,
        type_filter: str,
        per_path_k: int,
        time_filter: Optional[str],
        *,
        type_tag: str,
    ) -> Tuple[List[Hit], int, int]:
        chunk_filter = _and_filter(type_filter, time_filter)

        # 双路并行: vec + bm25 同时下发到 Milvus, 节省一次 RTT.
        # vec 失败时 vec_hits=[]; bm25 retriever 缺席或失败也回退到 [].
        def _do_vec() -> List[Hit]:
            return self.vec.retrieve(
                query, top_k=per_path_k, filter_expr=chunk_filter,
                embed_stage=EMBED_STAGE_SUMMARY,
            )

        tasks: List[Tuple[str, Any]] = [("vec", _do_vec)]
        if self.bm25 is not None:
            tasks.append((
                "bm25",
                lambda: self.bm25.retrieve(
                    query, top_k=per_path_k, filter_expr=chunk_filter,
                ),
            ))
        results = run_in_parallel(
            tasks,
            on_error=lambda name, exc: (
                logger.warning(
                    f"[summary] {name} 失败 type={type_tag} q={query!r}: {exc}"
                ) or []
            ),
        )
        vec_hits = results.get("vec", []) or []
        bm25_hits = results.get("bm25", []) or []
        merged = self._merge_dual_path_chunks(
            vec_hits, bm25_hits, type_tag=type_tag,
        )
        return merged, len(vec_hits), len(bm25_hits)

    def retrieve(
        self,
        query: str,
        top_k_docs: int = 8,
        per_query_k: int = 8,
        time_filter: Optional[str] = None,
    ) -> List[Hit]:
        if not query:
            return []
        per_path_k = self.per_path_k or per_query_k
        summary_hits, n_vec_s, n_bm25_s = self._retrieve_type_pool_dual_path(
            query, SUMMARY_CHUNK_FILTER, per_path_k, time_filter, type_tag="summary",
        )
        title_hits, n_vec_t, n_bm25_t = self._retrieve_type_pool_dual_path(
            query, TITLE_CHUNK_FILTER, per_path_k, time_filter, type_tag="title",
        )
        ranked = self._merge_type_pools(summary_hits, title_hits)
        logger.info(
            f"[summary] dual-path pools: "
            f"summary(vec={n_vec_s},bm25={n_bm25_s},merged={len(summary_hits)}) "
            f"title(vec={n_vec_t},bm25={n_bm25_t},merged={len(title_hits)}) "
            f"→ top_{top_k_docs} from {len(ranked)}"
        )
        return ranked[:top_k_docs]


# ---------------------------------------------------------------------------
# ProgressiveLocalRetriever
# ---------------------------------------------------------------------------

class ProgressiveLocalRetriever:
    """两级渐进式局部检索 (#9: L1 hybrid + 低置信兜底 + 阈值门控全库探测)。

    bm25_retriever: 可选; L1 BM25 summary 兜底 + retrieve_direct doc_name 定位。
    """

    def __init__(
        self,
        vec_retriever: VectorRetriever,
        hybrid_retriever: HybridRetriever,
        rrf_k: int = RRF_K,
        bm25_retriever: Optional[BM25Retriever] = None,
        config: Optional[ProgressiveRetrieveConfig] = None,
        hybrid_config: Optional[HybridWeightConfig] = None,
    ) -> None:
        self.vec = vec_retriever
        self.hybrid = hybrid_retriever
        self.rrf_k = rrf_k
        self.bm25 = bm25_retriever
        self.config = config or DEFAULT_PROGRESSIVE_CONFIG
        self.hybrid_config = hybrid_config or DEFAULT_HYBRID_CONFIG

    def _hybrid_retrieve(
        self,
        stage: str,
        query: str,
        *,
        top_k: int,
        per_retriever_k: int,
        filter_expr: Optional[str],
        chunk_type: Optional[str] = None,
        retrieve_bias: Optional[str] = None,
    ) -> List[Hit]:
        weights = infer_hybrid_weights(
            stage,
            query,
            retrieve_bias=retrieve_bias,
            chunk_type=chunk_type,
            config=self.hybrid_config,
        )
        logger.info(f"[hybrid] {format_weight_log(weights)}")
        return self.hybrid.retrieve(
            query,
            top_k=top_k,
            per_retriever_k=per_retriever_k,
            filter_expr=filter_expr,
            dense_weight=weights.dense,
            bm25_weight=weights.bm25,
        )

    @staticmethod
    def _top_doc_confidence(
        candidate_docs: List[Tuple[str, float, str]],
    ) -> float:
        if not candidate_docs:
            return 0.0
        return float(candidate_docs[0][1])

    def _aggregate_hits_to_docs_by_score(
        self,
        hits: List[Hit],
        top_k_docs: int,
        *,
        score_scale: float = 1.0,
    ) -> List[Tuple[str, float, str]]:
        """按 doc 聚合: 取该 doc 下 hit 的最高原始分 (非 RRF)。"""
        per_doc: Dict[str, Tuple[float, str]] = {}
        for h in hits:
            if not h.doc_id:
                continue
            s = score_scale * float(h.score)
            if h.doc_id in per_doc:
                old_s, old_n = per_doc[h.doc_id]
                if s > old_s:
                    per_doc[h.doc_id] = (s, h.doc_name or old_n or h.doc_id)
            else:
                per_doc[h.doc_id] = (s, h.doc_name or h.doc_id)
        ranked = sorted(per_doc.items(), key=lambda kv: -kv[1][0])
        return [(d, s, n) for d, (s, n) in ranked[:top_k_docs]]

    def _aggregate_hits_to_docs(
        self,
        hits: List[Hit],
        top_k_docs: int,
        *,
        score_scale: float = 1.0,
    ) -> List[Tuple[str, float, str]]:
        return self._aggregate_hits_to_docs_by_score(
            hits, top_k_docs, score_scale=score_scale,
        )

    @staticmethod
    def _merge_doc_rankings(
        primary: List[Tuple[str, float, str]],
        extra: List[Tuple[str, float, str]],
        *,
        extra_weight: float = 0.85,
    ) -> List[Tuple[str, float, str]]:
        per_doc: Dict[str, Tuple[float, str]] = {
            d: (s, n) for d, s, n in primary
        }
        for d, s, n in extra:
            if not d:
                continue
            bonus = s * extra_weight
            if d in per_doc:
                old_s, old_n = per_doc[d]
                per_doc[d] = (max(old_s, bonus), old_n or n)
            else:
                per_doc[d] = (bonus, n)
        return sorted(
            [(d, s, n) for d, (s, n) in per_doc.items()],
            key=lambda x: -x[1],
        )

    def _level1_dual_path_top_docs(
        self,
        query: str,
        time_filter: Optional[str],
    ) -> Tuple[List[Tuple[str, float, str]], List[Hit], int, int]:
        """L1: summary 池 vector/bm25 各取 top-K doc, 合并去重 (按 doc max score)。"""
        cfg = self.config
        per_path_k = cfg.level1_per_retriever_k
        base_filter = _and_filter(SUMMARY_TYPE_FILTER, time_filter)

        # 双路并行: vec/bm25 同时打 Milvus, 减一次 RTT
        results = run_in_parallel(
            [
                ("vec", lambda: self.vec.retrieve(
                    query, top_k=per_path_k, filter_expr=base_filter,
                    embed_stage=EMBED_STAGE_SUMMARY,
                )),
                ("bm25", lambda: self.hybrid.bm25.retrieve(
                    query, top_k=per_path_k, filter_expr=base_filter,
                )),
            ],
            on_error=lambda name, exc: (
                logger.warning(f"[local-l1] {name} 失败 q={query!r}: {exc}") or []
            ),
        )
        vec_hits = results.get("vec", []) or []
        bm25_hits = results.get("bm25", []) or []
        # P1.1: L1 doc-anchor 命中, 分数本就偏低, 走宽松阈值
        for h in vec_hits:
            h.sources = ["vector"]
            h.stage = "l1"
        for h in bm25_hits:
            h.sources = ["bm25"]
            h.stage = "l1"
        vec_docs = self._aggregate_hits_to_docs_by_score(vec_hits, per_path_k)
        bm25_docs = self._aggregate_hits_to_docs_by_score(bm25_hits, per_path_k)
        merged = self._merge_doc_rankings(
            vec_docs, bm25_docs, extra_weight=1.0,
        )
        # A3 (默认关闭): query 含钢牌号时, 把 summary 命中里真正含该牌号的文档前置,
        # 避免语义相近但不含牌号的文档挤掉正确文献。仅重排, 不改分数/不引入新文档。
        if cfg.experimental_grade_entity_bias:
            grades = _extract_steel_grades(query)
            if grades:
                gl = [g.lower() for g in grades]
                anchor_ids = {
                    h.doc_id for h in (vec_hits + bm25_hits)
                    if h.doc_id and any(g in (h.content or "").lower() for g in gl)
                }
                if anchor_ids:
                    merged = sorted(
                        merged, key=lambda t: (t[0] not in anchor_ids, -t[1]),
                    )
                    logger.info(
                        f"[local-l1] grade-anchor 前置: grades={grades} "
                        f"anchored_docs={len(anchor_ids)}"
                    )
        merged = merged[: cfg.l2_max_candidate_docs]
        return merged, vec_hits + bm25_hits, len(vec_docs), len(bm25_docs)

    def _level1_multi_path_locate_docs(
        self,
        query: str,
        top_k_docs: int,
        time_filter: Optional[str],
        *,
        per_path_k: int,
    ) -> Tuple[List[Tuple[str, float, str]], List[Hit]]:
        _ = (top_k_docs, per_path_k)
        docs, raw, _, _ = self._level1_dual_path_top_docs(query, time_filter)
        return docs, raw

    def _level1_locate_docs(
        self,
        query: str,
        top_k_docs: int,
        per_query_k: int,
        time_filter: Optional[str],
        *,
        per_retriever_k: int,
        retrieve_bias: Optional[str] = None,
    ) -> Tuple[List[Tuple[str, float, str]], List[Hit]]:
        _ = (per_query_k, retrieve_bias, per_retriever_k)
        return self._level1_dual_path_top_docs(query, time_filter)[:2]

    def _level1_bm25_summary_fallback(
        self,
        query: str,
        top_k_docs: int,
        time_filter: Optional[str],
    ) -> List[Tuple[str, float, str]]:
        if self.bm25 is None:
            return []
        base_filter = _and_filter(SUMMARY_TYPE_FILTER, time_filter)
        try:
            hits = self.bm25.retrieve(
                query, top_k=max(top_k_docs * 3, 20), filter_expr=base_filter,
            )
        except Exception as e:
            logger.warning(f"[local-l1] bm25 summary 兜底失败: {e}")
            return []
        return self._aggregate_hits_to_docs(hits, top_k_docs, score_scale=0.9)

    def _level1_global_chunk_probe(
        self,
        query: str,
        top_k_docs: int,
        time_filter: Optional[str],
        chunk_type: Optional[str],
        *,
        retrieve_bias: Optional[str] = None,
        force_chunk_type: bool = False,
    ) -> Tuple[List[Tuple[str, float, str]], List[Hit]]:
        """全库 sub-chunk 向量探测: 取 emb 得分 top-K, 供 L2 短路 + reranker。

        L1 置信低于阈值时触发; 在全量 sub-chunk 池上做 dense 检索 (不用 hybrid),
        按 embedding 相似度降序返回最多 global_fallback_top_k 条。

        P0 #3: force_chunk_type=True → 不剥离 structural 类型 (供 structural-first L1 复用)。

        chunk_type: L2 期望的 type; 默认行为 (force_chunk_type=False) 走 level1_global_probe_chunk_type
        进行收窄, 该函数对 references/image/table/equation 返回 None → 不收窄, 用 NON_SUMMARY 池。
        """
        _ = retrieve_bias  # vector-only 探测, 不使用 hybrid 权重
        cfg = self.config
        if force_chunk_type:
            probe_type = (chunk_type or "").strip().lower() or None
        else:
            probe_type = level1_global_probe_chunk_type(chunk_type)
        type_clause = (
            f'type == "{probe_type}"' if probe_type else NON_SUMMARY_TYPE_FILTER
        )
        chunk_filter = _and_filter(type_clause, time_filter)
        probe_k = cfg.l1_chunk_probe_k
        try:
            hits = self.vec.retrieve(
                query,
                top_k=probe_k,
                filter_expr=chunk_filter,
                embed_stage=EMBED_STAGE_PASSAGE,
            )
        except Exception as e:
            logger.warning(f"[local-l1] global chunk vector 探测失败: {e}")
            return [], []
        hits.sort(key=lambda h: h.score, reverse=True)
        hits = hits[:probe_k]
        docs = self._aggregate_hits_to_docs(
            hits, top_k_docs, score_scale=0.75,
        )
        return docs, hits

    # ---------------------------------------------------------------------------
    # L1 信号质量分层: 用 hit.sources 判断双路/单路/弱信号
    # ---------------------------------------------------------------------------

    _SIGNAL_STRONG = "strong"      # 双路命中 (vector + bm25)
    _SIGNAL_MODERATE = "moderate"  # 单路但多条命中
    _SIGNAL_WEAK = "weak"          # 单路单条命中
    _SIGNAL_EMPTY = "empty"        # 无命中

    @staticmethod
    def _l1_signal_quality(
        candidate_docs: List[Tuple[str, float, str]],
        l1_hits: List[Hit],
    ) -> str:
        """根据 top doc 的命中来源结构判断 L1 信号质量。

        返回:
          strong  — top doc 被 vector 和 bm25 同时命中 (双路信号, 高置信)
          moderate — top doc 仅单路命中但有多条 hit (中置信)
          weak    — top doc 仅单路且只有 1 条 hit (低置信, 易误判)
          empty   — 无候选 doc
        """
        if not candidate_docs:
            return ProgressiveLocalRetriever._SIGNAL_EMPTY

        top_doc_id = candidate_docs[0][0]
        top_hits = [h for h in l1_hits if h.doc_id == top_doc_id]

        if not top_hits:
            if len(candidate_docs) >= 2:
                return ProgressiveLocalRetriever._SIGNAL_MODERATE
            return ProgressiveLocalRetriever._SIGNAL_WEAK

        has_vector = any("vector" in h.sources for h in top_hits)
        has_bm25 = any("bm25" in h.sources for h in top_hits)

        if has_vector and has_bm25:
            return ProgressiveLocalRetriever._SIGNAL_STRONG
        if len(top_hits) >= 2:
            return ProgressiveLocalRetriever._SIGNAL_MODERATE
        return ProgressiveLocalRetriever._SIGNAL_WEAK

    @staticmethod
    def _dual_path_confidence_from_hits(
        vec_hits: List[Hit],
        bm25_hits: List[Hit],
        threshold: float,
    ) -> Tuple[bool, float, float, float]:
        """双路 top-1 原始分 → 平均分配信 (BM25 量纲过大时仅用 vector top-1)。"""
        if not vec_hits or not bm25_hits:
            return False, 0.0, 0.0, 0.0
        top_vec = float(vec_hits[0].score)
        top_bm25 = float(bm25_hits[0].score)
        if top_bm25 > 1.0:
            avg = top_vec
            high = top_vec >= threshold
        else:
            avg = (top_vec + top_bm25) / 2.0
            high = avg >= threshold
        return high, avg, top_vec, top_bm25

    def _build_doc_chunk_filter(
        self,
        doc_ids: List[str],
        chunk_type: Optional[str],
        time_filter: Optional[str],
    ) -> Optional[str]:
        if not doc_ids:
            return None
        doc_clause = " or ".join(
            f'doc_id == "{_escape_eq(d)}"' for d in doc_ids
        )
        type_clause = (
            f'type == "{chunk_type}"' if chunk_type else NON_SUMMARY_TYPE_FILTER
        )
        return _and_filter(f"({doc_clause})", type_clause, time_filter)

    def _build_global_chunk_filter(
        self,
        chunk_type: Optional[str],
        time_filter: Optional[str],
    ) -> Optional[str]:
        probe_type = level1_global_probe_chunk_type(chunk_type)
        type_clause = (
            f'type == "{probe_type}"' if probe_type else NON_SUMMARY_TYPE_FILTER
        )
        return _and_filter(type_clause, time_filter)

    def _evaluate_l2_confidence_in_docs(
        self,
        query: str,
        doc_ids: List[str],
        chunk_type: Optional[str],
        time_filter: Optional[str],
    ) -> Tuple[bool, float, float, float]:
        """在合并 doc 的 chunk 池内双路各 top-K, 用 top-1 平均分判断高置信。"""
        cfg = self.config
        chunk_filter = self._build_doc_chunk_filter(
            doc_ids, chunk_type, time_filter,
        )
        per_k = cfg.l2_per_path_k

        # 双路并行 probe
        results = run_in_parallel(
            [
                ("vec", lambda: self.vec.retrieve(
                    query, top_k=per_k, filter_expr=chunk_filter,
                    embed_stage=EMBED_STAGE_PASSAGE,
                )),
                ("bm25", lambda: self.hybrid.bm25.retrieve(
                    query, top_k=per_k, filter_expr=chunk_filter,
                )),
            ],
            on_error=lambda name, exc: (
                logger.warning(f"[local-l2] probe {name} 失败 q={query!r}: {exc}") or []
            ),
        )
        vec_hits = results.get("vec", []) or []
        bm25_hits = results.get("bm25", []) or []
        return self._dual_path_confidence_from_hits(
            vec_hits, bm25_hits, cfg.l2_drill_min_score,
        )

    def _level1_with_fallbacks(
        self,
        query: str,
        top_k_docs: int,
        per_query_k: int,
        time_filter: Optional[str],
        chunk_type: Optional[str],
        *,
        retrieve_bias: Optional[str] = None,
    ) -> Tuple[List[Tuple[str, float, str]], float, List[str], List[Hit]]:
        """返回 (candidate_docs, top_score, fallback_chain, l1_raw_hits)。"""
        _ = (top_k_docs, per_query_k, retrieve_bias, chunk_type)
        cfg = self.config
        candidate_docs, l1_raw_hits, n_vec, n_bm25 = self._level1_dual_path_top_docs(
            query, time_filter,
        )
        chain: List[str] = [
            "l1_dual_top5",
            f"vec_docs={n_vec}",
            f"bm25_docs={n_bm25}",
            f"merged_docs={len(candidate_docs)}",
        ]
        top_conf = self._top_doc_confidence(candidate_docs)
        return candidate_docs, top_conf, chain, l1_raw_hits

    def _level2_multi_path_chunks(
        self,
        query: str,
        chunk_filter: Optional[str],
        *,
        per_path_k: int,
        route_tag: str = ROUTE_PROGRESSIVE,
        stage: str = "l2",
    ) -> List[Hit]:
        """L2 多路召回: vector/bm25 各取 top-K, 去重合并, 不算 RRF, 交给 reranker 精排。

        ``stage`` (P1.1): 标记 hit 来源, 影响 per-(route, stage, type) 阈值矩阵.
        - "l2"        (默认): doc-scoped drill, 应用严格阈值
        - "l2_global"        : 全库 fallback, 应用最严格阈值
        """
        merged: Dict[str, Hit] = {}
        retrievers = [("vector", self.vec), ("bm25", self.hybrid.bm25)]
        for source, retriever in retrievers:
            try:
                if source == "vector":
                    hits = retriever.retrieve(
                        query, top_k=per_path_k, filter_expr=chunk_filter,
                        embed_stage=EMBED_STAGE_PASSAGE,
                    )
                else:
                    hits = retriever.retrieve(
                        query, top_k=per_path_k, filter_expr=chunk_filter,
                    )
            except Exception as e:
                logger.warning(f"[local-l2] {source} 失败 q={query!r}: {e}")
                continue
            for h in hits:
                if h.pk in merged:
                    existing = merged[h.pk]
                    if source not in existing.sources:
                        existing.sources.append(source)
                    if h.score > existing.score:
                        existing.score = h.score
                else:
                    h.sources = [source]
                    if route_tag not in h.sources:
                        h.sources.append(route_tag)
                    h.rrf_score = h.score
                    h.stage = stage
                    merged[h.pk] = h
        ranked = sorted(merged.values(), key=lambda x: -x.score)
        # B1 (默认关闭): 过滤 Abstract/Keywords 导航块; 过滤后为空则保留原结果。
        if self.config.experimental_l2_drop_nav:
            kept = [h for h in ranked if not _is_nav_chunk(h)]
            if kept:
                if len(kept) != len(ranked):
                    logger.info(
                        f"[local-l2] drop-nav: {len(ranked)} -> {len(kept)} hits"
                    )
                ranked = kept
        return ranked

    def _level2_split_content_chunks(
        self,
        query: str,
        *,
        doc_clause: Optional[str],
        time_filter: Optional[str],
        per_path_k: int,
        route_tag: str,
        stage: str,
        structural_top_k: int,
    ) -> List[Hit]:
        """正文/图表分池召回: {text,equation} 全量 + {image,table} 截断到 top-K, 合并。

        ``doc_clause`` 为 None 时表示全库 (global fallback), 否则为 ``(doc_id == ... or ...)``。
        正文池占主, 图表池仅保留 ``structural_top_k`` 条, 避免图表挤占正文 reranker 名额。
        """
        prose_filter = _and_filter(doc_clause, PROSE_CONTENT_TYPE_FILTER, time_filter)
        struct_filter = _and_filter(doc_clause, STRUCTURAL_CONTENT_TYPE_FILTER, time_filter)
        prose_hits = self._level2_multi_path_chunks(
            query, prose_filter, per_path_k=per_path_k,
            route_tag=route_tag, stage=stage,
        )
        struct_hits: List[Hit] = []
        if structural_top_k > 0:
            struct_hits = self._level2_multi_path_chunks(
                query, struct_filter, per_path_k=per_path_k,
                route_tag=route_tag, stage=stage,
            )[:structural_top_k]
        logger.info(
            f"[local-l2] split-pool: prose={len(prose_hits)} "
            f"structural={len(struct_hits)}(cap={structural_top_k})"
        )
        return prose_hits + struct_hits

    def _level2_drill_chunks(
        self,
        query: str, candidate_doc_ids: List[str], top_k_chunks: int,
        per_query_k: int, per_retriever_k: int, time_filter: Optional[str],
        chunk_type: Optional[str] = None, route_tag: str = ROUTE_PROGRESSIVE,
        retrieve_bias: Optional[str] = None,
    ) -> List[Hit]:
        _ = (top_k_chunks, per_query_k, retrieve_bias)  # L2 截断/RFF 已交给 reranker
        doc_clause = " or ".join(f'doc_id == "{_escape_eq(d)}"' for d in candidate_doc_ids)

        if chunk_type and is_structural_chunk_type(chunk_type):
            return self._level2_structural_chunks(
                candidate_doc_ids, chunk_type, time_filter, route_tag=route_tag,
            )

        # chunk_type 未指定: 正文/图表分池召回 (图表 top-K), 避免图表挤占正文名额
        if not chunk_type and self.config.split_content_pool:
            return self._level2_split_content_chunks(
                query,
                doc_clause=f"({doc_clause})",
                time_filter=time_filter,
                per_path_k=per_retriever_k,
                route_tag=route_tag,
                stage="l2",
                structural_top_k=self.config.structural_content_top_k,
            )

        type_clause = f'type == "{chunk_type}"' if chunk_type else NON_SUMMARY_TYPE_FILTER
        chunk_filter = _and_filter(f"({doc_clause})", type_clause, time_filter)
        return self._level2_multi_path_chunks(
            query,
            chunk_filter,
            per_path_k=per_retriever_k,
            route_tag=route_tag,
        )

    def _level2_structural_chunks(
        self,
        candidate_doc_ids: List[str],
        chunk_type: str,
        time_filter: Optional[str],
        *,
        route_tag: str = ROUTE_PROGRESSIVE,
    ) -> List[Hit]:
        """references/image/table: Milvus 硬过滤全量 query, 不做 hybrid 得分截断。"""
        doc_clause = " or ".join(f'doc_id == "{_escape_eq(d)}"' for d in candidate_doc_ids)
        type_clause = f'type == "{chunk_type}"'
        chunk_filter = _and_filter(f"({doc_clause})", type_clause, time_filter)
        client = self.vec.client
        try:
            rows = client.query(
                collection_name=self.vec.collection,
                filter=chunk_filter or type_clause,
                output_fields=EXTENDED_OUTPUT_FIELDS,
                limit=STRUCTURAL_FULL_RECALL_LIMIT,
            )
        except Exception as e:
            logger.warning(f"[local-l2] structural query 失败 ctype={chunk_type!r}: {e}")
            return []
        hits: List[Hit] = []
        for row in rows:
            hit = _row_to_hit(row)
            hit.score = 1.0
            hit.rrf_score = 1.0
            hit.sources = [route_tag]
            hit.stage = "l2"
            hits.append(hit)
        logger.info(
            f"[local-l2] structural full recall ctype={chunk_type} "
            f"docs={len(candidate_doc_ids)} hits={len(hits)}"
        )
        return hits

    def retrieve(
        self,
        query: str, top_k_docs: int = 6, top_k_chunks: int = 8,
        per_query_k: int = 8, per_retriever_k: int = 10,
        time_filter: Optional[str] = None, chunk_type: Optional[str] = None,
        retrieve_bias: Optional[str] = None,
    ) -> LocalRetrieveResult:
        """两级检索: L1 双路各 top-5 doc 合并; L2 按合并 doc 内双路平均分路由 doc/global。"""
        if not query:
            return LocalRetrieveResult()
        cfg = self.config

        use_structural_first = (
            cfg.structural_skip_summary_l1
            and chunk_type_skips_summary_l1(chunk_type)
        )
        l1_raw_hits: List[Hit] = []
        if use_structural_first:
            candidate_docs, raw_probe = self._level1_global_chunk_probe(
                query, top_k_docs, time_filter, chunk_type,
                retrieve_bias=retrieve_bias,
                force_chunk_type=True,
            )
            chain = [f"l1_structural_first({chunk_type})"]
            top_conf = self._top_doc_confidence(candidate_docs)
            l1_raw_hits = raw_probe
            candidate_docs = candidate_docs[:cfg.l2_max_candidate_docs]
        else:
            candidate_docs, top_conf, chain, l1_raw_hits = self._level1_with_fallbacks(
                query,
                top_k_docs=top_k_docs,
                per_query_k=per_query_k,
                time_filter=time_filter,
                chunk_type=chunk_type,
                retrieve_bias=retrieve_bias,
            )

        logger.info(
            f"[local-l1] docs={len(candidate_docs)} top_score={top_conf:.4f} "
            f"chain={chain}"
        )
        if not candidate_docs:
            logger.warning("[local] 第一级未召回到候选文献, 第二级跳过")
            return LocalRetrieveResult()

        is_structural_l2 = bool(chunk_type) and is_structural_chunk_type(chunk_type)
        if is_structural_l2:
            l2_doc_ids = [d for d, _, _ in candidate_docs]
            chain.append(f"l2_structural(docs={len(l2_doc_ids)})")
            chunk_hits = self._level2_drill_chunks(
                query=query, candidate_doc_ids=l2_doc_ids,
                top_k_chunks=top_k_chunks, per_query_k=per_query_k,
                per_retriever_k=per_retriever_k, time_filter=time_filter,
                chunk_type=chunk_type, retrieve_bias=retrieve_bias,
            )
        else:
            doc_ids = [d for d, _, _ in candidate_docs]
            high_conf, avg, top_vec, top_bm25 = self._evaluate_l2_confidence_in_docs(
                query, doc_ids, chunk_type, time_filter,
            )
            chain.append(
                f"l2_probe_avg={avg:.4f}(vec={top_vec:.4f},bm25={top_bm25:.4f})"
            )
            per_path_k = cfg.l2_per_path_k
            # P1.1: 记录 L2 子 stage 标签, 区分 doc-scoped vs global fallback
            l2_stage = "l2"
            is_global = False
            if high_conf:
                chunk_filter = self._build_doc_chunk_filter(
                    doc_ids, chunk_type, time_filter,
                )
                chain.append("l2_doc_scoped")
            elif cfg.enable_global_chunk_fallback:
                chunk_filter = self._build_global_chunk_filter(
                    chunk_type, time_filter,
                )
                chain.append("l2_global")
                l2_stage = "l2_global"
                is_global = True
            else:
                chunk_filter = self._build_doc_chunk_filter(
                    doc_ids, chunk_type, time_filter,
                )
                chain.append("l2_doc_scoped_fallback")
            # chunk_type 未指定: 正文/图表分池召回 (图表 top-K), 否则单池召回。
            if not chunk_type and cfg.split_content_pool:
                doc_clause = None if is_global else (
                    "(" + " or ".join(
                        f'doc_id == "{_escape_eq(d)}"' for d in doc_ids
                    ) + ")"
                )
                chunk_hits = self._level2_split_content_chunks(
                    query,
                    doc_clause=doc_clause,
                    time_filter=time_filter,
                    per_path_k=per_path_k,
                    route_tag=ROUTE_PROGRESSIVE,
                    stage=l2_stage,
                    structural_top_k=cfg.structural_content_top_k,
                )
            else:
                chunk_hits = self._level2_multi_path_chunks(
                    query,
                    chunk_filter,
                    per_path_k=per_path_k,
                    route_tag=ROUTE_PROGRESSIVE,
                    stage=l2_stage,
                )
            scope = "global" if chain[-1] == "l2_global" else "doc"
            logger.info(
                f"[local-l2] {scope} multi-path: merged_docs={len(doc_ids)} "
                f"per_path_k={per_path_k} hits={len(chunk_hits)} "
                f"high_conf={high_conf} avg={avg:.4f}"
            )

        return LocalRetrieveResult(
            candidate_docs=[CandidateDoc(doc_id=d, rrf_score=s, doc_name=n) for d, s, n in candidate_docs],
            chunk_hits=chunk_hits,
        )

    def _locate_docs_by_name(
        self, target_docs: List[str], time_filter: Optional[str] = None,
        bm25: Optional[BM25Retriever] = None,
        *,
        bm25_max_results: Optional[int] = None,
    ) -> List[Tuple[str, float, str]]:
        """三级降级匹配 doc_name -> doc_id:

        1. exact: doc_name == target  (大小写敏感, 走 INVERTED 索引最快)
        2. like : doc_name LIKE %tok% (按空格切 token, 全部都要命中, 容忍标点差异)
        3. bm25 : 在 summary chunk 上做 BM25 召回, 反查 doc_id
                  (容忍乱序、错别字、近义词; 仅作为兜底)

        任一级有结果就返回, 不再降级。

        Args:
            bm25_max_results: P0-C 安全阀. BM25 兜底召回的最大篇数. None 表示不限.
                单文献意图 (len(target_docs)==1) 时调用方应设 1, 避免一篇定位失败
                后扩散到无关文献 (尤其与 ctype=references 全量召回叠加时会污染 context).
        """
        if not target_docs:
            return []

        results = self._locate_exact(target_docs, time_filter)
        if results:
            logger.debug(f"[local-name] exact 命中 {len(results)} 篇")
            return results

        results = self._locate_like(target_docs, time_filter)
        if results:
            logger.debug(f"[local-name] like 命中 {len(results)} 篇")
            return results

        # 自动启用单篇安全阀: 调用方没显式指定时, len==1 默认上限 1.
        if bm25_max_results is None and len(target_docs) == 1:
            bm25_max_results = 1

        if bm25 is not None:
            results = self._locate_bm25(target_docs, time_filter, bm25)
            if results:
                before = len(results)
                results = [r for r in results if r[1] >= LOCAL_DOC_BM25_MIN_SCORE]
                if not results:
                    logger.warning(
                        f"[local-name] bm25 兜底最高分低于阈值 {LOCAL_DOC_BM25_MIN_SCORE:.2f}, "
                        f"拒绝自动锁定 target_docs={target_docs} (raw_hits={before})"
                    )
                    return []
                if bm25_max_results is not None and len(results) > bm25_max_results:
                    logger.info(
                        f"[local-name] bm25 兜底命中 {len(results)} 篇, "
                        f"按单篇安全阀截断至 {bm25_max_results} 篇 "
                        f"(target_docs={target_docs})"
                    )
                    results = results[:bm25_max_results]
                else:
                    logger.info(f"[local-name] bm25 兜底命中 {len(results)} 篇")
                return results

        logger.warning(f"[local-name] 三级降级均未命中: {target_docs}")
        return []

    def _locate_docs_by_id(
        self,
        target_doc_ids: List[str],
        time_filter: Optional[str] = None,
    ) -> List[Tuple[str, float, str]]:
        """已知规范 doc_id 时, 直接到 Milvus 查 doc_name 返回; 不走 name 解析.

        用于 router 已通过 doc_registry 锁定目标文献的场景 (P0-B). 即使 doc_name
        在 Milvus 与上一轮 registry 略有差异, doc_id 永远一致 — 这是最稳的锚点.
        """
        if not target_doc_ids:
            return []
        seen: set = set()
        unique_ids: List[str] = []
        for d in target_doc_ids:
            d = str(d).strip()
            if d and d not in seen:
                seen.add(d)
                unique_ids.append(d)
        if not unique_ids:
            return []
        eq_clauses = [f'doc_id == "{_escape_eq(d)}"' for d in unique_ids]
        base_filter = _and_filter("(" + " or ".join(eq_clauses) + ")", SUMMARY_TYPE_FILTER, time_filter)
        try:
            rows = self.vec.client.query(
                collection_name=self.vec.collection,
                filter=base_filter,
                output_fields=["doc_id", "doc_name"],
                limit=max(len(unique_ids) * 2, 4),
            )
        except Exception as e:
            logger.warning(f"[local-id] 查 doc_name 失败: {e}; 仅用 doc_id 兜底")
            return [(d, 0.0, d) for d in unique_ids]
        # rows 可能比目标少 (summary 不存在 / time_filter 过滤掉了) — 用 rows 优先,
        # 缺失的 doc_id 用 id 本身当 name 兜底, 不放弃锚点.
        by_id: Dict[str, str] = {}
        for row in rows:
            did = str(row.get("doc_id") or "")
            if did and did not in by_id:
                by_id[did] = str(row.get("doc_name") or did)
        out: List[Tuple[str, float, str]] = []
        for did in unique_ids:
            out.append((did, 0.0, by_id.get(did, did)))
        return out

    def _locate_exact(
        self, target_docs: List[str], time_filter: Optional[str],
    ) -> List[Tuple[str, float, str]]:
        eq_clauses = [f'doc_name == "{_escape_eq(n)}"' for n in target_docs if n]
        if not eq_clauses:
            return []
        base_filter = _and_filter("(" + " or ".join(eq_clauses) + ")", SUMMARY_TYPE_FILTER, time_filter)
        try:
            rows = self.vec.client.query(
                collection_name=self.vec.collection,
                filter=base_filter,
                output_fields=["doc_id", "doc_name"],
                limit=20,
            )
        except Exception as e:
            logger.debug(f"[local-name] exact 查询失败: {e}")
            return []
        return self._dedup_doc_rows(rows)

    def _locate_like(
        self, target_docs: List[str], time_filter: Optional[str],
    ) -> List[Tuple[str, float, str]]:
        # 每个 target 拆 token, 同 target 内 token 必须 AND 命中, 不同 target 之间 OR
        per_target_clauses: List[str] = []
        for name in target_docs:
            tokens = [t for t in re.split(r"[\s,，;；/\\]+", name) if len(t) >= 2]
            if not tokens:
                continue
            and_clauses = [f'doc_name like "%{_escape_like(t)}%"' for t in tokens]
            per_target_clauses.append("(" + " and ".join(and_clauses) + ")")
        if not per_target_clauses:
            return []
        like_block = "(" + " or ".join(per_target_clauses) + ")"
        base_filter = _and_filter(like_block, SUMMARY_TYPE_FILTER, time_filter)
        try:
            rows = self.vec.client.query(
                collection_name=self.vec.collection,
                filter=base_filter,
                output_fields=["doc_id", "doc_name"],
                limit=20,
            )
        except Exception as e:
            logger.debug(f"[local-name] like 查询失败: {e}")
            return []
        return self._dedup_doc_rows(rows)

    def _locate_bm25(
        self, target_docs: List[str], time_filter: Optional[str],
        bm25: BM25Retriever,
    ) -> List[Tuple[str, float, str]]:
        # 用 doc_name 关键词在 summary chunk 上做 BM25, 反查命中 doc
        kw_query = " ".join(t for n in target_docs for t in re.split(r"[\s,，;；/\\]+", n) if t)
        if not kw_query.strip():
            return []
        kw_filter = _and_filter(SUMMARY_TYPE_FILTER, time_filter)
        try:
            hits = bm25.retrieve(kw_query, top_k=20, filter_expr=kw_filter)
        except Exception as e:
            logger.debug(f"[local-name] bm25 查询失败: {e}")
            return []
        seen: set = set()
        out: List[Tuple[str, float, str]] = []
        for h in hits:
            if h.doc_id and h.doc_id not in seen:
                seen.add(h.doc_id)
                out.append((h.doc_id, float(h.score or 0.0), h.doc_name or h.doc_id))
        return out

    @staticmethod
    def _dedup_doc_rows(rows: List[Dict[str, Any]]) -> List[Tuple[str, float, str]]:
        seen: set = set()
        out: List[Tuple[str, float, str]] = []
        for row in rows:
            doc_id = row.get("doc_id", "")
            doc_name = row.get("doc_name", "")
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                out.append((doc_id, 0.0, doc_name))
        return out

    def retrieve_direct(
        self,
        query: str, target_docs: List[str], top_k_chunks: int = 8,
        per_query_k: int = 8, per_retriever_k: int = 10,
        time_filter: Optional[str] = None, chunk_type: Optional[str] = None,
        retrieve_bias: Optional[str] = None,
        *,
        target_doc_ids: Optional[List[str]] = None,
    ) -> LocalRetrieveResult:
        """指定文献内 chunk 检索; chunk_type 仅约束 L2.

        Args:
            target_doc_ids: P0-B 短路. router 通过 doc_registry 解析得到的规范 doc_id;
                非空时优先按 doc_id 定位文档, 跳过 doc_name 的 exact/like/bm25 三级降级.
                这是最稳的锚点 — 避免 doc_name 翻译/缩写差异导致 BM25 兜底扩散到无关文献,
                与 ctype=references 的结构化全量召回叠加时尤其关键.
        """
        if not query:
            return LocalRetrieveResult()
        # P0-B: doc_id 短路优先于 name 解析
        candidate_docs: List[Tuple[str, float, str]] = []
        if target_doc_ids:
            candidate_docs = self._locate_docs_by_id(target_doc_ids, time_filter)
            if candidate_docs:
                logger.info(
                    f"[local-id] 已通过 doc_registry 锁定 {len(candidate_docs)} 篇 "
                    f"(跳过 name 解析): {[d for d, _, _ in candidate_docs]}"
                )
        if not candidate_docs:
            if not target_docs:
                logger.warning("[local] target_docs / target_doc_ids 均为空, 跳过")
                return LocalRetrieveResult()
            # P0-C: 单文献意图时 BM25 兜底上限设 1 (避免一篇定位失败后扩散到多篇)
            bm25_cap: Optional[int] = 1 if len(target_docs) == 1 else None
            candidate_docs = self._locate_docs_by_name(
                target_docs, time_filter, bm25=self.bm25,
                bm25_max_results=bm25_cap,
            )
        if not candidate_docs:
            logger.warning("[local] 未找到匹配文档, 跳过")
            return LocalRetrieveResult()
        chunk_hits = self._level2_drill_chunks(
            query=query, candidate_doc_ids=[d for d, _, _ in candidate_docs],
            top_k_chunks=top_k_chunks, per_query_k=per_query_k,
            per_retriever_k=per_retriever_k, time_filter=time_filter,
            chunk_type=chunk_type, route_tag=ROUTE_LOCAL,
            retrieve_bias=retrieve_bias,
        )
        return LocalRetrieveResult(
            candidate_docs=[CandidateDoc(doc_id=d, rrf_score=s, doc_name=n) for d, s, n in candidate_docs],
            chunk_hits=chunk_hits,
        )


# ---------------------------------------------------------------------------
# EnhancedMetadataRetriever
# ---------------------------------------------------------------------------

class EnhancedMetadataRetriever:
    """元数据检索, 编号直查 (标签 LIKE + 位置选取) + 关键词 BM25 召回两种模式。"""

    def __init__(
        self,
        client: Any,
        collection: str = DEFAULT_COLLECTION,
        bm25_retriever: Optional[BM25Retriever] = None,
    ) -> None:
        self.client = client
        self.collection = collection
        self.bm25 = bm25_retriever or BM25Retriever(client, collection)

    def retrieve_by_position(
        self, fig_refs: List[str], table_refs: List[str],
        doc_id: Optional[str] = None, top_k: int = 10,
        time_filter: Optional[str] = None,
    ) -> List[Hit]:
        """按出现顺序选取第 N 张图/表 (当标签匹配失败时的 fallback)。

        查询指定 type 的所有 chunk, 按 page_start 排序, 取第 N-1 个。
        """
        results: List[Hit] = []

        for label in fig_refs:
            try:
                pos = int(label)
            except (ValueError, TypeError):
                continue
            if pos < 1:
                continue
            type_filter = 'type == "image"'
            doc_filter = f' and doc_id == "{_escape_eq(doc_id)}"' if doc_id else ""
            full_filter = _and_filter(type_filter + doc_filter, time_filter) or type_filter + doc_filter
            try:
                rows = self.client.query(
                    collection_name=self.collection,
                    filter=full_filter,
                    output_fields=EXTENDED_OUTPUT_FIELDS,
                    limit=200,
                )
            except Exception as e:
                logger.warning(f"[metadata-pos] image query 失败: {e}")
                continue
            rows.sort(key=lambda r: int(r.get("page_start", 0)))
            if pos <= len(rows):
                hit = _row_to_hit(rows[pos - 1])
                hit.sources = [ROUTE_METADATA]
                hit.matched_keywords = [f"Fig.{label} by position"]
                results.append(hit)

        for label in table_refs:
            try:
                pos = int(label)
            except (ValueError, TypeError):
                continue
            if pos < 1:
                continue
            type_filter = 'type == "table"'
            doc_filter = f' and doc_id == "{_escape_eq(doc_id)}"' if doc_id else ""
            full_filter = _and_filter(type_filter + doc_filter, time_filter) or type_filter + doc_filter
            try:
                rows = self.client.query(
                    collection_name=self.collection,
                    filter=full_filter,
                    output_fields=EXTENDED_OUTPUT_FIELDS,
                    limit=200,
                )
            except Exception as e:
                logger.warning(f"[metadata-pos] table query 失败: {e}")
                continue
            rows.sort(key=lambda r: int(r.get("page_start", 0)))
            if pos <= len(rows):
                hit = _row_to_hit(rows[pos - 1])
                hit.sources = [ROUTE_METADATA]
                hit.matched_keywords = [f"Table {label} by position"]
                results.append(hit)

        return results

    def retrieve_by_refs(
        self, fig_refs: List[str], table_refs: List[str], top_k: int = 10,
        max_candidates: int = 50, time_filter: Optional[str] = None,
        chunk_type: Optional[str] = None, doc_id: Optional[str] = None,
    ) -> List[Hit]:
        """图/表 metadata 直查: 返回全部命中, 不做 top_k 语义截断。"""
        if not (fig_refs or table_refs):
            return []

        label_hits = self._retrieve_by_label_match(
            fig_refs, table_refs,
            top_k=STRUCTURAL_FULL_RECALL_LIMIT,
            max_candidates=STRUCTURAL_FULL_RECALL_LIMIT,
            time_filter=time_filter,
            chunk_type=chunk_type,
        )
        if label_hits:
            if doc_id:
                return [h for h in label_hits if h.doc_id == doc_id]
            return label_hits

        logger.info("[metadata-refs] 标签匹配无结果, 尝试按位置选取")
        return self.retrieve_by_position(
            fig_refs, table_refs, doc_id=doc_id,
            top_k=STRUCTURAL_FULL_RECALL_LIMIT, time_filter=time_filter,
        )

    def _retrieve_by_label_match(
        self, fig_refs: List[str], table_refs: List[str], top_k: int = 10,
        max_candidates: int = 50, time_filter: Optional[str] = None,
        chunk_type: Optional[str] = None,
    ) -> List[Hit]:

        ref_clauses = collect_ref_like_clauses(fig_refs, table_refs)

        if not ref_clauses:
            return []

        like_block = "(" + " or ".join(ref_clauses) + ")"
        if chunk_type:
            target_types = f'(type == "{chunk_type}" or type == "text")'
        elif fig_refs and table_refs:
            target_types = '(type == "image" or type == "table" or type == "text")'
        elif fig_refs:
            target_types = '(type == "image" or type == "text")'
        else:
            target_types = '(type == "table" or type == "text")'
        retrieve_filter = _and_filter(like_block, target_types, time_filter)

        try:
            rows = self.client.query(
                collection_name=self.collection,
                filter=retrieve_filter,
                output_fields=EXTENDED_OUTPUT_FIELDS,
                limit=max_candidates,
            )
        except Exception as e:
            logger.warning(f"[metadata-refs] query 失败: {e}")
            return []

        scored: List[Hit] = []
        for row in rows:
            hit = _row_to_hit(row)
            blob = " ".join([hit.content, hit.section, hit.context])
            score, matched = score_fig_table_refs(
                blob, fig_refs, table_refs, hit.type,
            )

            if score == 0:
                continue
            hit.score = score
            hit.matched_keywords = matched
            hit.sources = [ROUTE_METADATA]
            scored.append(hit)

        scored.sort(key=lambda h: -h.score)
        return scored

    def retrieve_by_keywords(
        self, keywords: List[str], top_k: int = 10, max_candidates: int = 200,
        time_filter: Optional[str] = None, chunk_type: Optional[str] = None,
    ) -> List[Hit]:
        """关键词召回: 走 BM25 稀疏向量。

        max_candidates 参数保留向后兼容, BM25 模式下不再使用。
        """
        if not keywords:
            return []
        kws = [str(kw).strip() for kw in keywords if kw and len(str(kw).strip()) >= 2]
        if not kws:
            return []

        type_filter = f'type == "{chunk_type}"' if chunk_type else NON_SUMMARY_TYPE_FILTER
        retrieve_filter = _and_filter(type_filter, time_filter)

        kw_query = " ".join(kws)
        hits = self.bm25.retrieve(
            kw_query, top_k=top_k, filter_expr=retrieve_filter,
        )
        for h in hits:
            h.matched_keywords = list(kws)
            h.sources = [ROUTE_METADATA]
        return hits

    def retrieve_by_filters(
        self,
        page_refs: Optional[List[int]] = None,
        paragraph_refs: Optional[List[int]] = None,
        entities: Optional[List[str]] = None,
        doc_id: Optional[str] = None,
        chunk_type: Optional[str] = None,
        time_filter: Optional[str] = None,
        top_k: int = 10,
        max_candidates: int = 200,
    ) -> List[Hit]:
        """按 page_start / paragraph_index / content LIKE entity 做硬过滤召回。

        - page_refs: 1-based, 内部减 1 转 page_start
        - paragraph_refs: 1-based, 直接对 paragraph_index 做 in 比较
        - entities: 在 content 上做 LIKE %entity% (含大小写变体 OR)
        - 三类条件之间是 AND (都满足); 同类条件之间是 OR
        - chunk_type / doc_id / time_filter 作为额外的 AND 过滤项
        """
        page_refs = page_refs or []
        paragraph_refs = paragraph_refs or []
        entities = [e for e in (entities or []) if e and e.strip()]
        if not (page_refs or paragraph_refs or entities):
            return []

        clauses: List[str] = []
        if page_refs:
            zero_pages = sorted({max(0, p - 1) for p in page_refs})
            if len(zero_pages) == 1:
                clauses.append(f"page_start == {zero_pages[0]}")
            else:
                clauses.append(
                    "page_start in [" + ",".join(str(p) for p in zero_pages) + "]"
                )
        if paragraph_refs:
            paras = sorted({int(p) for p in paragraph_refs if int(p) >= 1})
            if len(paras) == 1:
                clauses.append(f"paragraph_index == {paras[0]}")
            else:
                clauses.append(
                    "paragraph_index in [" + ",".join(str(p) for p in paras) + "]"
                )
        if entities:
            ent_likes = collect_entity_like_clauses(entities)
            if ent_likes:
                clauses.append("(" + " or ".join(ent_likes) + ")")
        if doc_id:
            clauses.append(f'doc_id == "{_escape_eq(doc_id)}"')
        if chunk_type:
            clauses.append(f'type == "{chunk_type}"')
        retrieve_filter = _and_filter(" and ".join(f"({c})" for c in clauses), time_filter)

        try:
            rows = self.client.query(
                collection_name=self.collection,
                filter=retrieve_filter,
                output_fields=EXTENDED_OUTPUT_FIELDS,
                limit=max_candidates,
            )
        except Exception as e:
            logger.warning(f"[metadata-filters] query 失败: {e}")
            return []

        hits: List[Hit] = []
        ent_lower = [e.lower() for e in entities]
        for row in rows:
            hit = _row_to_hit(row)
            hit.sources = [ROUTE_METADATA]
            matched: List[str] = []
            if page_refs:
                matched.append(f"page={hit.page_start + 1 if hit.page_start >= 0 else '?'}")
            if paragraph_refs:
                matched.append(f"paragraph={hit.paragraph_index}")
            if entities:
                blob_lower = (hit.content or "").lower()
                for e_lo, e_orig in zip(ent_lower, entities):
                    if e_lo and e_lo in blob_lower:
                        matched.append(f"entity:{e_orig}")
            hit.matched_keywords = matched
            score = 0.0
            if page_refs and hit.page_start in {p - 1 for p in page_refs}:
                score += 1.0
            if paragraph_refs and hit.paragraph_index in set(paragraph_refs):
                score += 1.0
            if entities:
                blob_lower = (hit.content or "").lower()
                score += sum(2.0 for e_lo in ent_lower if e_lo and e_lo in blob_lower)
            hit.score = score
            hits.append(hit)
        hits.sort(key=lambda h: -h.score)
        return hits[:top_k]

    def retrieve(
        self, keywords: List[str], fig_refs: List[str], table_refs: List[str],
        top_k: int = 10, max_candidates: int = 200, time_filter: Optional[str] = None,
        chunk_type: Optional[str] = None, doc_id: Optional[str] = None,
        page_refs: Optional[List[int]] = None,
        paragraph_refs: Optional[List[int]] = None,
        entities: Optional[List[str]] = None,
    ) -> List[Hit]:
        page_refs = page_refs or []
        paragraph_refs = paragraph_refs or []
        entities = entities or []
        has_refs = bool(fig_refs or table_refs)
        has_kws = bool(keywords)
        has_filter = bool(page_refs or paragraph_refs or entities)

        filter_hits: List[Hit] = []
        if has_filter:
            filter_hits = self.retrieve_by_filters(
                page_refs=page_refs,
                paragraph_refs=paragraph_refs,
                entities=entities,
                doc_id=doc_id,
                chunk_type=chunk_type,
                time_filter=time_filter,
                top_k=top_k,
                max_candidates=max_candidates,
            )

        if has_refs and not has_kws and not has_filter:
            return self.retrieve_by_refs(
                fig_refs, table_refs, top_k, max_candidates, time_filter, chunk_type,
                doc_id=doc_id,
            )
        if has_kws and not has_refs and not has_filter:
            return self.retrieve_by_keywords(keywords, top_k, max_candidates, time_filter, chunk_type)
        if not has_refs and not has_kws and not has_filter:
            return []
        if has_filter and not has_refs and not has_kws:
            return filter_hits

        ref_hits: List[Hit] = []
        if has_refs:
            ref_hits = self.retrieve_by_refs(
                fig_refs, table_refs, top_k, 50, time_filter, chunk_type, doc_id=doc_id,
            )
        kw_hits: List[Hit] = []
        if has_kws:
            kw_hits = self.retrieve_by_keywords(
                keywords, top_k, max_candidates, time_filter, chunk_type,
            )
        seen_pks: set = set()
        merged: List[Hit] = []
        # 硬过滤命中优先级最高
        for h in filter_hits + ref_hits + kw_hits:
            if h.pk not in seen_pks:
                seen_pks.add(h.pk)
                merged.append(h)
        merged.sort(key=lambda h: -h.score)
        return merged[:top_k]


# ---------------------------------------------------------------------------
# AgenticContextBuilder
# ---------------------------------------------------------------------------

class AgenticContextBuilder:
    """把多路径 hits 渲染成 LLM 友好的分组模板。

    设计要点 (v3):
    - 跨路径去重: 同一 chunk_id 在多条路径上重复命中时, 只渲染一次, 在头部
      列出 "命中路径=[...]"; 后续路径只引用 "(已在 [route] 路径展示)"
    - 总长度硬上限 max_total_chars: 超过后停止追加新 hit, 在末尾加截断标记
    - 路径预算 per_route_chars: 单路径占用上限, 防止某条路径吃光所有预算
    - context (image/table 周边段落) 截断阈值 max_context_in_prompt 可配
    """

    SEP = "\n\n---\n\n"
    ROUTE_TITLES = {
        ROUTE_SUMMARY:    "全局俯瞰 (type=summary 摘要)",
        ROUTE_PROGRESSIVE: "渐进式检索 (先 summary 找文献, 再文献内精召回)",
        ROUTE_LOCAL:      "指定文献检索 (按 doc_name 定位, 直接在文献内召回)",
        ROUTE_METADATA:   "元数据 (编号直查 or 关键词匹配)",
        ROUTE_NEIGHBOR:   "关联补充 (命中块的关联图/表/公式/上下文, 互为补充)",
    }

    DEFAULT_MAX_CONTEXT_IN_PROMPT = 500
    DEFAULT_MAX_TOTAL_CHARS = 24000     # 整个 context 的字符上限 (≈ 6k tokens)
    DEFAULT_PER_ROUTE_CHARS = 10000     # 每条路径上限

    def __init__(
        self,
        max_context_in_prompt: int = DEFAULT_MAX_CONTEXT_IN_PROMPT,
        max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
        per_route_chars: int = DEFAULT_PER_ROUTE_CHARS,
    ) -> None:
        self.max_context_in_prompt = max_context_in_prompt
        self.max_total_chars = max_total_chars
        self.per_route_chars = per_route_chars

    # ────────────────────────────────────────────────────────────────────
    # 入口
    # ────────────────────────────────────────────────────────────────────

    def build(
        self, query: str, decision: RouteDecision, results: Dict[str, Any],
        subquery_decisions: Optional[List[RouteDecision]] = None,
    ) -> str:
        sections: List[str] = [f"# 用户问题\n{query}"]

        decision_lines = ["# 路由决策"]
        decision_lines.append(f"- 路径: {', '.join(decision.routes) or '(none)'}")
        if decision.chunk_type:
            decision_lines.append(f"- 类型: {decision.chunk_type}")
        if decision.target_docs:
            decision_lines.append("- 文献: " + ", ".join(decision.target_docs))
        if decision.time:
            decision_lines.append(f"- 年份: {decision.time}")
        if decision.fig_refs:
            decision_lines.append("- 图编号: " + ", ".join(decision.fig_refs))
        if decision.table_refs:
            decision_lines.append("- 表编号: " + ", ".join(decision.table_refs))
        if decision.page_refs:
            decision_lines.append("- 页码 (1-based): " + ", ".join(map(str, decision.page_refs)))
        if decision.paragraph_refs:
            decision_lines.append("- 段落号 (1-based): " + ", ".join(map(str, decision.paragraph_refs)))
        if decision.entities:
            decision_lines.append("- 实体: " + ", ".join(decision.entities))
        for route in decision.routes:
            rw = decision.rewrites.get(route, "")
            if rw:
                decision_lines.append(f"- 改写({route}): {rw}")
        sections.append("\n".join(decision_lines))

        # 跨路径去重: chunk_id -> 首次出现的 (route, rank)
        first_seen: Dict[str, Tuple[str, int]] = {}

        # 累计字符数, 用于硬上限截断
        used = sum(len(s) + len(self.SEP) for s in sections)
        truncated = False

        compound = subquery_decisions and len(subquery_decisions) > 1
        if compound:
            for i, sub_dec in enumerate(subquery_decisions):
                if used >= self.max_total_chars:
                    truncated = True
                    break
                sub_id = f"sub{i + 1}"
                sub_section = self._format_subquery_decision_section(sub_id, sub_dec)
                sections.append(sub_section)
                used += len(sub_section) + len(self.SEP)
                for route in sub_dec.routes:
                    if used >= self.max_total_chars:
                        truncated = True
                        break
                    route_key = f"{sub_id}:{route}"
                    title = self.ROUTE_TITLES.get(route, route)
                    budget = min(self.per_route_chars, self.max_total_chars - used)
                    res = results.get(route_key)
                    if route in (ROUTE_PROGRESSIVE, ROUTE_LOCAL) and isinstance(res, LocalRetrieveResult):
                        rendered, route_truncated = self._render_local(
                            route_key, res, title, first_seen, budget,
                        )
                    else:
                        hits = res if isinstance(res, list) else []
                        rendered, route_truncated = self._render_simple(
                            route_key, hits, title, first_seen, budget,
                        )
                    sections.append(rendered)
                    used += len(rendered) + len(self.SEP)
                    if route_truncated:
                        truncated = True
        else:
            for route in decision.routes:
                if used >= self.max_total_chars:
                    truncated = True
                    break
                title = self.ROUTE_TITLES.get(route, route)
                budget = min(self.per_route_chars, self.max_total_chars - used)
                res = results.get(route)
                if route in (ROUTE_PROGRESSIVE, ROUTE_LOCAL) and isinstance(res, LocalRetrieveResult):
                    rendered, route_truncated = self._render_local(
                        route, res, title, first_seen, budget,
                    )
                else:
                    hits = res if isinstance(res, list) else []
                    rendered, route_truncated = self._render_simple(
                        route, hits, title, first_seen, budget,
                    )
                sections.append(rendered)
                used += len(rendered) + len(self.SEP)
                if route_truncated:
                    truncated = True

        # 邻域扩展回填 (related_assets / adjacent / page / similar): 不在 decision.routes 里,
        # 单独渲染一段, 让关联图/表/公式与正文互为补充。已在前面路径渲染过的 chunk 由
        # first_seen 去重跳过, 不会重复。
        neighbor_res = results.get(ROUTE_NEIGHBOR)
        neighbor_hits = [
            h for h in (neighbor_res if isinstance(neighbor_res, list) else [])
            if (h.chunk_id or h.pk) and (h.chunk_id or h.pk) not in first_seen
        ]
        if neighbor_hits and used < self.max_total_chars:
            budget = min(self.per_route_chars, self.max_total_chars - used)
            rendered, route_truncated = self._render_simple(
                ROUTE_NEIGHBOR, neighbor_hits,
                self.ROUTE_TITLES.get(ROUTE_NEIGHBOR, "关联补充"),
                first_seen, budget,
            )
            sections.append(rendered)
            used += len(rendered) + len(self.SEP)
            if route_truncated:
                truncated = True

        if truncated:
            sections.append(
                f"# [系统] context 已截断: 超过总上限 {self.max_total_chars} 字符 "
                f"(或单路径上限 {self.per_route_chars})"
            )

        return self.SEP.join(sections)

    def _format_subquery_decision_section(self, sub_id: str, sub_dec: RouteDecision) -> str:
        lines = [f"# 子查询 [{sub_id}]"]
        lines.append(f"- 路径: {', '.join(sub_dec.routes) or '(none)'}")
        if sub_dec.chunk_type:
            lines.append(f"- 类型: {sub_dec.chunk_type}")
        if sub_dec.target_docs:
            lines.append("- 文献: " + ", ".join(sub_dec.target_docs))
        if sub_dec.fig_refs:
            lines.append("- 图编号: " + ", ".join(sub_dec.fig_refs))
        if sub_dec.table_refs:
            lines.append("- 表编号: " + ", ".join(sub_dec.table_refs))
        for route in sub_dec.routes:
            rw = sub_dec.rewrites.get(route, "")
            if rw:
                lines.append(f"- 改写({route}): {rw}")
        return "\n".join(lines)

    # ────────────────────────────────────────────────────────────────────
    # 路径渲染 (带预算 + 去重)
    # ────────────────────────────────────────────────────────────────────

    def _render_simple(
        self, route: str, hits: List[Hit], title: str,
        first_seen: Dict[str, Tuple[str, int]], budget: int,
    ) -> Tuple[str, bool]:
        header = f"# 来自 [{route}] 的内容 — {title} (共 {len(hits)} 条)"
        if not hits:
            return header + "\n*(本路径未召回到内容)*", False

        lines = [header]
        used = len(header)
        truncated = False
        rendered_count = 0
        skipped_dup = 0
        for i, h in enumerate(hits, 1):
            key = h.chunk_id or h.pk
            if key and key in first_seen:
                prev_route, prev_rank = first_seen[key]
                line = f"[{i}] (已在 [{prev_route}] 路径以 #{prev_rank} 展示, chunk_id={key})"
            else:
                line = self._format_hit(i, h)
                if key:
                    first_seen[key] = (route, i)
            if used + len(line) + 2 > budget:
                truncated = True
                break
            lines.append(line)
            used += len(line) + 2
            rendered_count += 1
            if key and (route, i) != first_seen.get(key, (route, i)):
                skipped_dup += 1

        if truncated:
            lines.append(f"\n*(超过路径预算 {budget} 字符, 截断剩余 {len(hits) - rendered_count} 条)*")
        return "\n\n".join(lines), truncated

    def _render_local(
        self, route: str, res: LocalRetrieveResult, title: str,
        first_seen: Dict[str, Tuple[str, int]], budget: int,
    ) -> Tuple[str, bool]:
        header = f"# 来自 [{route}] 的内容 — {title}"
        lines = [header]
        used = len(header)
        truncated = False

        if res.candidate_docs:
            doc_names = [cd.doc_name for cd in res.candidate_docs]
            doc_line = f"\n## 候选文献 ({len(res.candidate_docs)} 篇): {'; '.join(doc_names)}"
            lines.append(doc_line)
            used += len(doc_line)
        else:
            lines.append("\n## 候选文献\n*(未召回到文献)*")

        if res.chunk_hits:
            count_line = f"\n## 精召回 chunks (共 {len(res.chunk_hits)} 个)"
            lines.append(count_line)
            used += len(count_line)
            rendered_count = 0
            for i, h in enumerate(res.chunk_hits, 1):
                key = h.chunk_id or h.pk
                if key and key in first_seen:
                    prev_route, prev_rank = first_seen[key]
                    line = f"[{i}] (已在 [{prev_route}] 路径以 #{prev_rank} 展示, chunk_id={key})"
                else:
                    line = self._format_hit(i, h)
                    if key:
                        first_seen[key] = (route, i)
                if used + len(line) + 2 > budget:
                    truncated = True
                    break
                lines.append(line)
                used += len(line) + 2
                rendered_count += 1
            if truncated:
                lines.append(
                    f"\n*(超过路径预算 {budget} 字符, 截断剩余 "
                    f"{len(res.chunk_hits) - rendered_count} 条)*"
                )
        else:
            lines.append("\n## 精召回 chunks\n*(未召回到 chunk)*")

        return "\n".join(lines), truncated

    # 兼容旧字段名
    @property
    def MAX_CONTEXT_IN_PROMPT(self) -> int:  # noqa: N802
        return self.max_context_in_prompt

    def _format_hit(self, rank: int, hit: Hit) -> str:
        head_bits = [f"[{rank}] {hit.type.upper()}"]
        if hit.chunk_id:
            head_bits.append(f"chunk_id={hit.chunk_id}")
        if hit.doc_id:
            head_bits.append(f"doc={hit.doc_id}")
        # 兼容历史灌入的乱码 section: 渲染时再过滤一次, 避免污染 LLM context
        clean_section = sanitize_section(hit.section)
        if clean_section:
            head_bits.append(f"section={clean_section}")
        if hit.page_start >= 0:
            head_bits.append(f"page={hit.page_start + 1}")  # 1-based 给 LLM 看, 与用户问 "第x页" 对应
        if hit.paragraph_index >= 1:
            head_bits.append(f"para={hit.paragraph_index}")
        if hit.publication_year:
            head_bits.append(f"year={hit.publication_year}")
        lines = [" | ".join(head_bits)]
        if hit.content:
            lines.append(hit.content)
        if hit.type in ("image", "table") and hit.context:
            ctx = hit.context
            if len(ctx) > self.max_context_in_prompt:
                ctx = ctx[:self.max_context_in_prompt] + "[...truncated]"
            lines.append("[Related Section Context]")
            lines.append(ctx)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# AgenticRAGPipeline
# ---------------------------------------------------------------------------


DEFAULT_AGENTIC_SYSTEM_PROMPT = _generation_system_prompt_from_file()

AGENTIC_USER_TEMPLATE = (
    "{context}\n\n"
    "请基于以上检索到的多路径上下文给出严谨、有引用的回答。"
)


class AgenticRAGPipeline:
    """Agentic RAG 主控: router + 三路径并行 + 模板渲染 + (可选) LLM 生成。"""

    def __init__(
        self,
        router: QueryRouter,
        summary_retriever: SummaryRetriever,
        local_retriever: ProgressiveLocalRetriever,
        metadata_retriever: EnhancedMetadataRetriever,
        context_builder: Optional[AgenticContextBuilder] = None,
        llm: Optional[LLMClient] = None,
        # 4 条检索路径并行: summary / progressive / local / metadata
        # (改造前是 3, 实际只够 3 路同时跑, 第 4 路会被串行排队)
        max_workers: int = 4,
        summary_config: Optional["SummaryRetrieveConfig"] = None,
    ) -> None:
        self.router = router
        self.summary_r = summary_retriever
        self.local_r = local_retriever
        self.metadata_r = metadata_retriever
        self.context_builder = context_builder or AgenticContextBuilder()
        self.llm = llm
        self.max_workers = max_workers
        from .progressive_config import DEFAULT_SUMMARY_CONFIG
        self.summary_config = summary_config or DEFAULT_SUMMARY_CONFIG

    def _dispatch(
        self, query: str, decision: RouteDecision,
        summary_top: int, progressive_top_docs: int, progressive_top_chunks: int,
        local_top_chunks: int, metadata_top: int,
        summary_per_query_k: Optional[int] = None,
        _force_no_time: bool = False,
    ) -> Dict[str, Any]:
        time_filter = None if _force_no_time else decision.to_time_filter()
        route_ct = describe_route_chunk_types(decision)
        if route_ct:
            logger.debug(
                f"[dispatch] chunk_type decision={decision.chunk_type!r} per_route={route_ct}"
            )
        tasks: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            if decision.has(ROUTE_SUMMARY):
                per_qk = (
                    summary_per_query_k
                    if summary_per_query_k is not None
                    else self.summary_config.per_query_k
                )
                tasks[ROUTE_SUMMARY] = ex.submit(
                    self.summary_r.retrieve,
                    decision.get_rewrite(ROUTE_SUMMARY, query),
                    summary_top, per_qk, time_filter,
                )
            if decision.has(ROUTE_PROGRESSIVE):
                tasks[ROUTE_PROGRESSIVE] = ex.submit(
                    self.local_r.retrieve,
                    decision.get_rewrite(ROUTE_PROGRESSIVE, query),
                    progressive_top_docs, progressive_top_chunks, 5, 8, time_filter,
                    chunk_type_for_route(ROUTE_PROGRESSIVE, decision),
                    decision.retrieve_bias,
                )
            if decision.has(ROUTE_LOCAL):
                tasks[ROUTE_LOCAL] = ex.submit(
                    self.local_r.retrieve_direct,
                    decision.get_rewrite(ROUTE_LOCAL, query),
                    decision.target_docs, local_top_chunks, 5, 8, time_filter,
                    chunk_type_for_route(ROUTE_LOCAL, decision),
                    decision.retrieve_bias,
                    target_doc_ids=list(decision.target_doc_ids or []),
                )
            if decision.has(ROUTE_METADATA):
                meta_ct = chunk_type_for_route(ROUTE_METADATA, decision)
                # metadata 严格只用 filters, 不消费任何 rewrite 关键词 (issue #2)
                has_refs = bool(decision.fig_refs or decision.table_refs)
                has_filter = bool(
                    decision.page_refs or decision.paragraph_refs or decision.entities
                )
                # metadata 按 doc_id 过滤; 优先用 registry 解析的 target_doc_ids[0],
                # 否则退到 target_docs[0] (历史行为, 可能误把 doc_name 当 doc_id)
                metadata_doc_id = (
                    decision.target_doc_ids[0]
                    if decision.target_doc_ids
                    else (decision.target_docs[0] if decision.target_docs else None)
                )

                if has_filter:
                    tasks[ROUTE_METADATA] = ex.submit(
                        self.metadata_r.retrieve,
                        [], decision.fig_refs, decision.table_refs,
                        metadata_top, 200, time_filter, meta_ct, metadata_doc_id,
                        decision.page_refs, decision.paragraph_refs, decision.entities,
                    )
                elif has_refs:
                    tasks[ROUTE_METADATA] = ex.submit(
                        self.metadata_r.retrieve_by_refs,
                        decision.fig_refs, decision.table_refs, metadata_top, 50,
                        time_filter, meta_ct, doc_id=metadata_doc_id,
                    )
                else:
                    logger.warning(
                        "[metadata] 无 filters 且无 refs, 跳过 (硬约束: metadata 必须有 filters)"
                    )

            out: Dict[str, Any] = {}
            for route, fut in tasks.items():
                try:
                    out[route] = fut.result()
                except Exception as e:
                    logger.warning(f"[{route}] 路径失败: {e}")
                    if route in (ROUTE_PROGRESSIVE, ROUTE_LOCAL):
                        out[route] = LocalRetrieveResult()
                    else:
                        out[route] = []
        if time_filter:
            total = 0
            for v in out.values():
                if isinstance(v, LocalRetrieveResult):
                    total += len(v.chunk_hits)
                elif isinstance(v, list):
                    total += len(v)
            if total == 0:
                logger.info(
                    f"[dispatch] time={decision.time!r} 过滤后零命中, "
                    f"降级为全量文献检索 (忽略 time 条件)"
                )
                return self._dispatch(
                    query, decision,
                    summary_top, progressive_top_docs, progressive_top_chunks,
                    local_top_chunks, metadata_top,
                    summary_per_query_k=summary_per_query_k,
                    _force_no_time=True,
                )

        # 邻域扩展 (依赖图谱场景): decision 显式要求时才跑, 否则零开销
        if getattr(decision, "expand_neighbors", None):
            from .neighbor_expansion import apply_neighbor_expansion
            out = apply_neighbor_expansion(
                out,
                modes=decision.expand_neighbors,
                expander=self._get_neighbor_expander(),
                time_filter=time_filter,
            )
        return out

    def _get_neighbor_expander(self):
        """懒构造邻域扩展器 (复用 metadata_r 的 Milvus 连接 + local_r 的向量检索器)。"""
        expander = getattr(self, "_neighbor_expander", None)
        if expander is None:
            from .neighbor_expansion import NeighborExpander
            vec = getattr(self.local_r, "vec", None) or getattr(self.summary_r, "vec", None)
            expander = NeighborExpander(
                client=self.metadata_r.client,
                collection=self.metadata_r.collection,
                vector_retriever=vec,
            )
            self._neighbor_expander = expander
        return expander

    def _embedder(self) -> Optional[Any]:
        """获取共享的 EmbeddingClient (跨 summary/progressive/local 三路检索器)."""
        try:
            return self.summary_r.vec.embedder
        except AttributeError:
            return None

    def _prewarm_query_embeddings(
        self,
        query: str,
        decision: "RouteDecision",
    ) -> None:
        """在 dispatch 之前一次性 batch-embed 所有 rewrite, 填入 EmbeddingClient LRU.

        Agentic 多路径会让同一组 rewrite 被 vec.retrieve 多次调用 (summary 池 + title
        池 + progressive L1 + L2 probe + global probe ...). 单条 ``embed()`` 会逐次
        发 HTTP, 而 batch ``embed_batch(texts)`` 一次返回所有向量, 之后所有路径直接
        命中 LRU. 这一步把 N 次 HTTP 压成 1 次.
        """
        embedder = self._embedder()
        if embedder is None:
            return
        # 清空上一轮 query 残留, 避免互相串扰 (LRU 自己也会淘汰, 显式清更稳)
        if hasattr(embedder, "begin_request"):
            embedder.begin_request()

        enabled = bool(getattr(embedder, "query_instruct_enabled", True))
        instructs = getattr(embedder, "query_instructs", None)
        texts = collect_prewarm_embed_texts(
            [decision], query, enabled=enabled, instructs=instructs,
        )
        if not texts:
            return
        try:
            vecs = embedder.embed_batch(texts)
        except Exception as e:
            logger.warning(f"[prewarm] batch embed 失败 (将逐路 fallback): {e}")
            return
        # 把 batch 结果填回 query cache: 后续 vec.retrieve(query) 直接命中
        for txt, vec in zip(texts, vecs):
            if hasattr(embedder, "_cache_put"):
                embedder._cache_put(txt, vec)
        logger.debug(
            f"[prewarm] batch-embed {len(texts)} 条 formatted rewrite -> query cache 预热完成"
        )

    def run(
        self,
        query: str,
        summary_top: Optional[int] = None,
        progressive_top_docs: int = 5,
        progressive_top_chunks: int = 8,
        local_top_chunks: int = 8,
        metadata_top: int = 8,
        summary_per_query_k: Optional[int] = None,
        forced_routes: Optional[List[str]] = None,
        forced_decision: Optional[RouteDecision] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        if summary_top is None:
            summary_top = self.summary_config.top_docs
        if summary_per_query_k is None:
            summary_per_query_k = self.summary_config.per_query_k
        t0 = time.time()
        if forced_decision is not None:
            decision = forced_decision
        elif forced_routes:
            decision = self.router.route(query, history=history)
            normalized = []
            for r in forced_routes:
                r = ROUTE_ALIAS.get(r, r)
                if r in VALID_ROUTES and r not in normalized:
                    normalized.append(r)
            decision.routes = normalized or [ROUTE_PROGRESSIVE]
            for route in decision.routes:
                if route not in decision.rewrites or not decision.rewrites[route]:
                    decision.rewrites[route] = query
        else:
            decision = self.router.route(query, history=history)
        t_route = time.time() - t0

        # 预 batch-embed: 一次 HTTP 把所有 rewrite 喂进 EmbeddingClient LRU.
        # 后续 vec.retrieve(query) 直接命中, 不再各自打 HTTP.
        self._prewarm_query_embeddings(query, decision)

        t1 = time.time()
        results = self._dispatch(
            query, decision,
            summary_top=summary_top, progressive_top_docs=progressive_top_docs,
            progressive_top_chunks=progressive_top_chunks, local_top_chunks=local_top_chunks,
            metadata_top=metadata_top, summary_per_query_k=summary_per_query_k,
        )
        t_retrieve = time.time() - t1

        t2 = time.time()
        context = self.context_builder.build(query, decision, results)
        t_render = time.time() - t2

        # 打印路由决策 JSON: 只展示有值的字段, 把硬过滤项收到 filters 子对象
        decision_json: Dict[str, Any] = {
            "routes": decision.routes,
            "rewrites": decision.rewrites,
        }
        filters_dump: Dict[str, Any] = {}
        if decision.chunk_type:
            filters_dump["chunk_type"] = decision.chunk_type
        if decision.target_docs:
            filters_dump["target_docs"] = decision.target_docs
        if decision.fig_refs:
            filters_dump["fig_refs"] = decision.fig_refs
        if decision.table_refs:
            filters_dump["table_refs"] = decision.table_refs
        if decision.page_refs:
            filters_dump["page_refs"] = decision.page_refs
        if decision.paragraph_refs:
            filters_dump["paragraph_refs"] = decision.paragraph_refs
        if decision.entities:
            filters_dump["entities"] = decision.entities
        if decision.time:
            filters_dump["time"] = decision.time
        if decision.retrieve_bias:
            decision_json["retrieve_bias"] = decision.retrieve_bias
        if filters_dump:
            decision_json["filters"] = filters_dump
        logger.info(f"[路由决策] {json.dumps(decision_json, ensure_ascii=False)}")

        # 打印拼接好的上下文
        logger.info(f"[检索上下文] (长度 {len(context)} 字符)\n{context}")

        # 路由器指标 snapshot (累计计数)
        router_metrics = self.router.metrics.snapshot()
        logger.info(
            f"[router-metrics] total={router_metrics['total']} "
            f"fallback={router_metrics['fallback_used']} "
            f"({router_metrics['fallback_ratio']:.1%}) "
            f"by_route={router_metrics['by_route']}"
        )

        # 阶段耗时汇总 (检索阶段, 不含生成)
        retrieval_total = t_route + t_retrieve + t_render
        logger.info(
            f"[耗时-检索] route={t_route:.2f}s | retrieve={t_retrieve:.2f}s | "
            f"render={t_render:.2f}s | total={retrieval_total:.2f}s"
        )

        return {
            "query": query,
            "decision": decision,
            "results": results,
            "context": context,
            "router_metrics": router_metrics,
            "latency": {
                "route_s": round(t_route, 3),
                "retrieve_s": round(t_retrieve, 3),
                "render_s": round(t_render, 3),
                "total_s": round(retrieval_total, 3),
            },
        }

    def answer(
        self,
        query: str,
        system: str = DEFAULT_AGENTIC_SYSTEM_PROMPT,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        stream: bool = False,
        history: Optional[List[Dict[str, str]]] = None,
        chat_messages: Optional[List[Dict[str, str]]] = None,
        disable_thinking: bool = True,
        **run_kwargs: Any,
    ) -> Dict[str, Any]:
        """生成回答。

        Args:
            query: 用户问题
            system: 系统提示
            temperature: 生成温度
            max_tokens: 最大 token 数
            stream: 是否流式
            history: 路由器用的历史对话文本
            chat_messages: 生成 LLM 用的历史消息列表 (OpenAI 格式, 不含 system 和当前 user)
        """
        if self.llm is None:
            raise RuntimeError("AgenticRAGPipeline 未配置 LLMClient, 无法 answer")
        run_result = self.run(query, history=history, **run_kwargs)
        user_msg = AGENTIC_USER_TEMPLATE.format(context=run_result["context"])

        # 进度提示: 让用户知道接下来在等生成 LLM 而不是卡死
        logger.info(
            f"[generate] 开始调用生成 LLM: model={getattr(self.llm, 'model', '?')} "
            f"stream={stream} max_tokens={max_tokens} prompt_chars={len(user_msg)}"
        )
        t0 = time.time()
        ttft: Optional[float] = None  # time-to-first-token (流式模式)
        if chat_messages:
            # 多轮: 构建 messages 列表
            messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
            messages.extend(chat_messages)
            messages.append({"role": "user", "content": user_msg})

            if stream:
                chunks_list: List[str] = []
                for piece in self.llm.chat_messages_stream(
                    messages, temperature=temperature, max_tokens=max_tokens,
                    disable_thinking=disable_thinking,
                ):
                    if ttft is None:
                        ttft = time.time() - t0
                        # 首包到达时立刻提示, 让用户感知延迟
                        print(f"\n[首包] ttft={ttft:.2f}s", flush=True)
                    chunks_list.append(piece)
                    print(piece, end="", flush=True)
                print()
                answer = "".join(chunks_list)
                usage = None
            else:
                chat_res = self.llm.chat_messages(
                    messages, temperature=temperature, max_tokens=max_tokens,
                    disable_thinking=disable_thinking,
                )
                answer = chat_res["answer"]
                usage = chat_res.get("usage")
        else:
            # 单轮
            if stream:
                chunks_list: List[str] = []
                for piece in self.llm.chat_stream(
                    system=system, user=user_msg,
                    temperature=temperature, max_tokens=max_tokens,
                    disable_thinking=disable_thinking,
                ):
                    if ttft is None:
                        ttft = time.time() - t0
                        print(f"\n[首包] ttft={ttft:.2f}s", flush=True)
                    chunks_list.append(piece)
                    print(piece, end="", flush=True)
                print()
                answer = "".join(chunks_list)
                usage = None
            else:
                chat_res = self.llm.chat(
                    system=system, user=user_msg,
                    temperature=temperature, max_tokens=max_tokens,
                    disable_thinking=disable_thinking,
                )
                answer = chat_res["answer"]
                usage = chat_res.get("usage")
        t_gen = time.time() - t0

        run_result["answer"] = answer
        run_result["usage"] = usage
        run_result["user_message"] = user_msg
        run_result["latency"]["generate_s"] = round(t_gen, 3)
        if ttft is not None:
            run_result["latency"]["ttft_s"] = round(ttft, 3)
        run_result["latency"]["total_s"] = round(run_result["latency"]["total_s"] + t_gen, 3)

        # 端到端耗时汇总: 检索各阶段 + 生成 (流式时 ttft 单列)
        lat = run_result["latency"]
        gen_part = (
            f"generate={lat['generate_s']:.2f}s"
            + (f" (ttft={lat['ttft_s']:.2f}s)" if "ttft_s" in lat else "")
        )
        logger.info(
            f"[耗时-端到端] route={lat['route_s']:.2f}s | "
            f"retrieve={lat['retrieve_s']:.2f}s | "
            f"render={lat['render_s']:.2f}s | "
            f"{gen_part} | "
            f"total={lat['total_s']:.2f}s"
        )
        return run_result


# ---------------------------------------------------------------------------
# Pipeline 工厂
# ---------------------------------------------------------------------------

def build_agentic_pipeline(
    milvus_uri: str = DEFAULT_MILVUS_URI,
    milvus_token: str = DEFAULT_MILVUS_TOKEN,
    collection: str = DEFAULT_COLLECTION,
    embed_api_base: str = DEFAULT_EMBED_API_BASE,
    embed_model: str = DEFAULT_EMBED_MODEL,
    embed_api_key: str = DEFAULT_EMBED_API_KEY,
    # 生成 LLM (主答题模型)
    llm_api_base: str = DEFAULT_LLM_API_BASE,
    llm_model: str = DEFAULT_LLM_MODEL,
    llm_api_key: str = DEFAULT_LLM_API_KEY,
    # 路由 LLM (None 表示与生成 LLM 共用 api_base/model/api_key; 总是独立 LLMClient
    # 实例, 这样可以使用更短的 timeout/max_retries, 失败立即走启发式 fallback,
    # 不会被全局 generation 的 120s timeout/3 retries 拖死)
    router_api_base: Optional[str] = None,
    router_model: Optional[str] = None,
    router_api_key: Optional[str] = None,
    router_temperature: float = 0.0,
    router_max_tokens: int = 300,
    router_timeout: int = 30,
    router_max_retries: int = 1,
    router_history_turns: int = 1,
    router_use_json_schema: bool = True,
    use_router_llm: bool = True,
    enable_generation_llm: bool = False,
    embed_normalize: bool = False,
    embed_query_instruct_enabled: bool = True,
    embed_query_instructs: Optional[Dict[str, str]] = None,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    dense_metric: str = "IP",
    dense_search_params: Optional[Dict[str, Any]] = None,
    bm25_search_params: Optional[Dict[str, Any]] = None,
    context_max_total_chars: Optional[int] = None,
    context_per_route_chars: Optional[int] = None,
    context_max_in_prompt: Optional[int] = None,
    progressive_config: Optional[ProgressiveRetrieveConfig] = None,
    hybrid_config: Optional[HybridWeightConfig] = None,
    summary_config: Optional["SummaryRetrieveConfig"] = None,
    keepalive_time_ms: int = 300_000,
    keepalive_timeout_ms: int = 60_000,
    disable_thinking: bool = True,
    disable_thinking_extra_body: bool = False,
    db_name: str = "",
) -> AgenticRAGPipeline:
    """构建 Agentic RAG 主控管线。

    router_* 为 None 时复用 llm_* 配置 (向后兼容); 显式传 router_* 即使用独立配置,
    建议路由用更小/更快的模型 + 更低温度。

    ``db_name`` 仅在 Milvus server 后端有效, Lite 模式应传空串。
    """
    meta, vec, bm25, hybrid = build_retrievers(
        milvus_uri=milvus_uri, milvus_token=milvus_token, collection=collection,
        embed_api_base=embed_api_base, embed_model=embed_model,
        embed_api_key=embed_api_key,
        embed_normalize=embed_normalize,
        embed_query_instruct_enabled=embed_query_instruct_enabled,
        embed_query_instructs=embed_query_instructs,
        dense_weight=dense_weight, bm25_weight=bm25_weight,
        dense_metric=dense_metric,
        dense_search_params=dense_search_params,
        bm25_search_params=bm25_search_params,
        keepalive_time_ms=keepalive_time_ms,
        keepalive_timeout_ms=keepalive_timeout_ms,
        db_name=db_name,
    )
    client = vec.client

    router_llm: Optional[LLMClient] = None
    gen_llm: Optional[LLMClient] = None

    # 复用 ClientRegistry: 同一进程内 generation LLM 只创建一次
    from ..clients.client_registry import get_global_registry
    _registry = get_global_registry()

    # 生成 LLM
    if enable_generation_llm:
        if not llm_api_key:
            logger.warning("未提供 generation LLM api_key, 生成功能不可用")
        else:
            try:
                gen_llm = _registry.get_llm(
                    api_base=llm_api_base, model=llm_model, api_key=llm_api_key,
                    disable_thinking_extra_body=disable_thinking_extra_body,
                )
            except Exception as e:
                logger.warning(f"generation LLMClient 初始化失败: {e}")

    # 路由 LLM: 独立的 (api_base, model, api_key, extra_body) 组合就自然进不同 cache 槽位,
    # 不会和 generation 串. 同样走 ClientRegistry 复用 requests.Session.
    if use_router_llm:
        r_api_base = router_api_base or llm_api_base
        r_model = router_model or llm_model
        r_api_key = router_api_key or llm_api_key
        if not r_api_key:
            logger.warning("未提供 router LLM api_key, router 将走启发式 fallback")
        else:
            try:
                router_llm = _registry.get_llm(
                    api_base=r_api_base, model=r_model, api_key=r_api_key,
                    timeout=router_timeout, max_retries=router_max_retries,
                    disable_thinking_extra_body=disable_thinking_extra_body,
                )
                shared_with_gen = (r_api_base == llm_api_base and r_model == llm_model)
                logger.info(
                    f"[router-llm] model={r_model} @ {r_api_base} "
                    f"timeout={router_timeout}s max_retries={router_max_retries} "
                    f"extra_body={disable_thinking_extra_body} "
                    f"{'(共用 generation 模型)' if shared_with_gen else '(独立模型)'}"
                )
            except Exception as e:
                logger.warning(f"router LLMClient 初始化失败: {e}")

    router = QueryRouter(
        router_llm, temperature=router_temperature, max_tokens=router_max_tokens,
        use_json_schema=router_use_json_schema,
        history_turns=router_history_turns,
        disable_thinking=disable_thinking,
    )
    summary_r = SummaryRetriever(vec, bm25_retriever=bm25)
    local_r = ProgressiveLocalRetriever(
        vec, hybrid, bm25_retriever=bm25,
        config=progressive_config or DEFAULT_PROGRESSIVE_CONFIG,
        hybrid_config=hybrid_config or DEFAULT_HYBRID_CONFIG,
    )
    metadata_r = EnhancedMetadataRetriever(client, collection=collection, bm25_retriever=bm25)

    cb_kwargs: Dict[str, Any] = {}
    if context_max_in_prompt is not None:
        cb_kwargs["max_context_in_prompt"] = context_max_in_prompt
    if context_max_total_chars is not None:
        cb_kwargs["max_total_chars"] = context_max_total_chars
    if context_per_route_chars is not None:
        cb_kwargs["per_route_chars"] = context_per_route_chars
    context_builder = AgenticContextBuilder(**cb_kwargs) if cb_kwargs else AgenticContextBuilder()

    return AgenticRAGPipeline(
        router=router,
        summary_retriever=summary_r,
        local_retriever=local_r,
        metadata_retriever=metadata_r,
        context_builder=context_builder,
        llm=gen_llm,
        summary_config=summary_config,
    )
