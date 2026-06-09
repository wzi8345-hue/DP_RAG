"""Router retrieve_bias → Hybrid RRF 权重 (#14)。"""

from __future__ import annotations

import re
from typing import Optional

from .hybrid_config import (
    DEFAULT_HYBRID_CONFIG,
    VALID_RETRIEVE_BIASES,
    HybridWeightConfig,
    WeightPair,
)

STAGE_PROGRESSIVE_L1 = "progressive_l1"
STAGE_PROGRESSIVE_L1_GLOBAL = "progressive_l1_global"
STAGE_PROGRESSIVE_L2 = "progressive_l2"
STAGE_LOCAL_L2 = "local_l2"
STAGE_SIMPLE = "simple"

_SEMANTIC_RE = re.compile(
    r"(机理|机制|影响|对比|比较|差异|原理|趋势|关系|如何|为什么|"
    r"mechanism|impact|effect|compare|comparison|difference|principle|trend|relationship|how|why)",
    re.IGNORECASE,
)
_FIG_TAB_PAGE_RE = re.compile(
    r"(图\s*\d|表\s*\d|fig\.?\s*\d|figure\s*\d|table\s*\d|第\s*\d+\s*[页段节])",
    re.IGNORECASE,
)
# 取值型问句 (含量/数值/规格等具体取值): 答案多在表格里, 走 BM25-heavy 更准。
# 放在 semantic 之前判定, 避免 "...影响...含量是多少" 被误判成 dense-heavy。
_VALUE_LOOKUP_RE = re.compile(
    r"(含量|质量分数|占比|百分比|配比|成分|规格|尺寸|速率|强度|硬度|"
    r"上限|下限|最大值|最小值|阈值|是多少|多少|数值|取值|参数)",
)
_QUOTED_ENTITY_RE = re.compile(r'["\'「」『』]([^"\'「」『』]{2,})["\'「」『』]')
_CHEM_FORMULA_RE = re.compile(r"\b[A-Z][a-z]?[0-9]*(?:[A-Z][a-z]?[0-9]*){2,}\b")
_DOI_RE = re.compile(r"\b10\.\d{4,}/\S+", re.IGNORECASE)


class HybridWeightResult:
    __slots__ = ("dense", "bm25", "stage", "retrieve_bias", "source")

    def __init__(
        self,
        dense: float,
        bm25: float,
        stage: str,
        retrieve_bias: Optional[str] = None,
        source: str = "default",
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.stage = stage
        self.retrieve_bias = retrieve_bias
        self.source = source

    def as_tuple(self) -> tuple[float, float]:
        return self.dense, self.bm25


def normalize_retrieve_bias(raw: Optional[str]) -> Optional[str]:
    if not raw or not isinstance(raw, str):
        return None
    key = raw.strip().lower()
    if key in VALID_RETRIEVE_BIASES:
        return key
    return None


def infer_retrieve_bias_heuristic(
    query: str,
    *,
    chunk_type: Optional[str] = None,
) -> str:
    """FC/JSON router 不可用时的 retrieve_bias 兜底 (heuristic fallback)。"""
    q = (query or "").strip()
    ct = (chunk_type or "").strip().lower()

    if ct == "references":
        return "entity_heavy"
    if ct == "equation":
        return "keyword"

    if q:
        if _QUOTED_ENTITY_RE.search(q) or _CHEM_FORMULA_RE.search(q) or _DOI_RE.search(q):
            return "entity_heavy"
        if _FIG_TAB_PAGE_RE.search(q) or len(q.replace(" ", "")) <= 4:
            return "keyword"
        # 取值型优先于 semantic: "影响...含量是多少" 这类答案在表格, 不应判 dense-heavy
        if _VALUE_LOOKUP_RE.search(q):
            return "keyword"
        if _SEMANTIC_RE.search(q):
            return "semantic"

    return "balanced"


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _normalize_pair(dense: float, bm25: float) -> tuple[float, float]:
    total = dense + bm25
    if total <= 0:
        return 0.6, 0.4
    return round(dense / total, 4), round(bm25 / total, 4)


def infer_hybrid_weights(
    stage: str,
    query: str = "",
    *,
    retrieve_bias: Optional[str] = None,
    chunk_type: Optional[str] = None,
    config: Optional[HybridWeightConfig] = None,
) -> HybridWeightResult:
    """按 router 输出的 retrieve_bias (或 heuristic 兜底) 映射 stage 权重。"""
    cfg = config or DEFAULT_HYBRID_CONFIG

    if cfg.mode == "static":
        dense, bm25 = _normalize_pair(cfg.static_dense, cfg.static_bm25)
        return HybridWeightResult(
            dense, bm25, stage, retrieve_bias=None, source="static",
        )

    bias = normalize_retrieve_bias(retrieve_bias)
    source = "router"
    if not bias:
        bias = infer_retrieve_bias_heuristic(query, chunk_type=chunk_type)
        source = "heuristic"

    profile = cfg.bias_profiles.get(bias) or cfg.bias_profiles.get("balanced") or {}
    pair: WeightPair = profile.get(stage) or profile.get(STAGE_SIMPLE) or WeightPair(0.60, 0.40)

    dense = _clamp(pair.dense, cfg.clamp_min, cfg.clamp_max)
    bm25 = _clamp(pair.bm25, cfg.clamp_min, cfg.clamp_max)
    dense, bm25 = _normalize_pair(dense, bm25)

    return HybridWeightResult(
        dense, bm25, stage, retrieve_bias=bias, source=source,
    )


def format_weight_log(result: HybridWeightResult) -> str:
    bias = result.retrieve_bias or "-"
    return (
        f"stage={result.stage} bias={bias} source={result.source} "
        f"dense={result.dense:.2f} bm25={result.bm25:.2f}"
    )
