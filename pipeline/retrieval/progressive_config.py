"""Progressive 两级检索配置 (#9 Phase B)。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ProgressiveRetrieveConfig:
    """Progressive: 双路分路 top-K + 平均分配信路由。

    L1: summary 池 vector/bm25 各 top-`level1_per_retriever_k` doc → 合并。
    L2 探测: 合并 doc 的 chunk 池内各 top-`l2_per_path_k`; 双路 top-1 平均分 ≥ 阈值 → 高置信。
    L2 终检: 高置信 doc-scoped 双路 top-K; 低置信全库 chunk 双路 top-K。
    """

    level1_mode: str = "hybrid"
    level1_min_docs: int = 2  # 已废弃
    doc_confidence_threshold: float = 0.025  # 已废弃
    enable_bm25_summary_fallback: bool = True  # 已废弃
    enable_global_chunk_fallback: bool = True
    global_fallback_top_k: int = 20  # 已废弃
    global_fallback_per_retriever_k: int = 10  # 已废弃
    level1_per_retriever_k: int = 5
    strong_signal_ratio: float = 1.5  # 已废弃
    enable_probe_short_circuit: bool = True  # 已废弃
    structural_skip_summary_l1: bool = True
    weak_signal_threshold_factor: float = 1.5  # 已废弃
    l1_chunk_probe_k: int = 20  # 已废弃
    l2_max_candidate_docs: int = 10
    l2_drill_min_score: float = 0.55
    l2_narrow_top_docs: int = 5  # 已废弃
    l2_per_path_k: int = 10
    # --- L2 内容池分离 (正文 vs 图表) ---
    # split_content_pool=True 时, chunk_type 未指定的 L2 召回把 {text,equation} (正文)
    # 与 {image,table} (图表) 分成两个独立池各自 vector/bm25 双路召回, 图表池截断到
    # structural_content_top_k 条后再与正文合并交 reranker。
    # 动机: 图表块 (caption 短、emb 分常偏高) 与正文块混在一个池竞争 top-k 时, 会挤占
    # 正文 gold 块名额, 导致散文型事实问题召回偏低。分离后正文有独立配额, 图表仅保留少量。
    split_content_pool: bool = True
    structural_content_top_k: int = 2
    # --- 实验性开关 (默认关闭; 关闭时检索行为与历史字节级一致) ---
    # A3: L1 文档选择时, 若 query 含钢牌号 (Q450NQR1/09CuPCrNi/...), 把 summary 命中里
    #     真正含该牌号的文档前置, 缓解"语义相近但无牌号的错误文档"挤占 L1 候选。
    experimental_grade_entity_bias: bool = False
    # B1: L2 召回返回前, 过滤掉 summary/Abstract/摘要/Keywords/关键词 这类导航块
    #     (非事实答案载体); 若过滤后候选为空则保留原结果, 不会清空召回。
    experimental_l2_drop_nav: bool = False

    def __post_init__(self) -> None:
        if self.level1_mode not in ("hybrid", "vector"):
            object.__setattr__(self, "level1_mode", "hybrid")
        if self.level1_min_docs < 1:
            object.__setattr__(self, "level1_min_docs", 1)
        if self.doc_confidence_threshold < 0:
            object.__setattr__(self, "doc_confidence_threshold", 0.0)
        if self.strong_signal_ratio < 1.0:
            object.__setattr__(self, "strong_signal_ratio", 1.0)
        if self.weak_signal_threshold_factor < 1.0:
            object.__setattr__(self, "weak_signal_threshold_factor", 1.0)
        if self.l1_chunk_probe_k < 1:
            object.__setattr__(self, "l1_chunk_probe_k", 1)
        if self.l2_max_candidate_docs < 1:
            object.__setattr__(self, "l2_max_candidate_docs", 1)
        if self.l2_narrow_top_docs < 1:
            object.__setattr__(self, "l2_narrow_top_docs", 1)
        if self.l2_per_path_k < 1:
            object.__setattr__(self, "l2_per_path_k", 1)
        if self.structural_content_top_k < 0:
            object.__setattr__(self, "structural_content_top_k", 0)

    @staticmethod
    def is_strong_signal(
        candidate_docs: List[Tuple[str, float, str]],
        ratio: float,
    ) -> bool:
        """保留兼容; 新路由不再使用 RRF 强信号。"""
        if not candidate_docs:
            return False
        if len(candidate_docs) == 1:
            return True
        top1 = candidate_docs[0][1]
        top2 = candidate_docs[1][1]
        if top2 <= 0:
            return top1 > 0
        return (top1 / top2) >= ratio


SKIP_SUMMARY_L1_CHUNK_TYPES = frozenset({"references", "image", "table", "equation"})


def chunk_type_skips_summary_l1(chunk_type: Optional[str]) -> bool:
    return (chunk_type or "").strip().lower() in SKIP_SUMMARY_L1_CHUNK_TYPES


DEFAULT_PROGRESSIVE_CONFIG = ProgressiveRetrieveConfig()


def progressive_config_from_dict(raw: Optional[Dict[str, Any]]) -> ProgressiveRetrieveConfig:
    raw = raw or {}
    probe_k = raw.get("l1_chunk_probe_k", raw.get("global_fallback_top_k", 20))
    l2_per_k = int(
        raw.get("l2_per_path_k", raw.get("global_fallback_per_retriever_k", 10))
    )
    return ProgressiveRetrieveConfig(
        level1_mode=str(raw.get("level1_mode", "hybrid")),
        level1_min_docs=int(raw.get("level1_min_docs", 2)),
        doc_confidence_threshold=float(raw.get("doc_confidence_threshold", 0.025)),
        enable_bm25_summary_fallback=bool(raw.get("enable_bm25_summary_fallback", True)),
        enable_global_chunk_fallback=bool(raw.get("enable_global_chunk_fallback", True)),
        global_fallback_top_k=int(raw.get("global_fallback_top_k", 20)),
        global_fallback_per_retriever_k=int(raw.get("global_fallback_per_retriever_k", 10)),
        level1_per_retriever_k=int(raw.get("level1_per_retriever_k", 5)),
        strong_signal_ratio=float(raw.get("strong_signal_ratio", 1.5)),
        enable_probe_short_circuit=bool(raw.get("enable_probe_short_circuit", True)),
        structural_skip_summary_l1=bool(raw.get("structural_skip_summary_l1", True)),
        weak_signal_threshold_factor=float(raw.get("weak_signal_threshold_factor", 1.5)),
        l1_chunk_probe_k=int(probe_k),
        l2_max_candidate_docs=int(raw.get("l2_max_candidate_docs", 10)),
        l2_drill_min_score=float(raw.get("l2_drill_min_score", 0.55)),
        l2_narrow_top_docs=int(raw.get("l2_narrow_top_docs", 5)),
        l2_per_path_k=l2_per_k,
        split_content_pool=bool(raw.get("split_content_pool", True)),
        structural_content_top_k=int(raw.get("structural_content_top_k", 2)),
        experimental_grade_entity_bias=bool(
            raw.get("experimental_grade_entity_bias", False)
        ),
        experimental_l2_drop_nav=bool(raw.get("experimental_l2_drop_nav", False)),
    )


@dataclass(frozen=True)
class SummaryRetrieveConfig:
    """Summary 文献发现路径: summary/title 双池召回 + 按 doc 截断。"""

    top_docs: int = 5
    per_query_k: int = 5

    def __post_init__(self) -> None:
        if self.top_docs < 1:
            object.__setattr__(self, "top_docs", 1)
        if self.per_query_k < 1:
            object.__setattr__(self, "per_query_k", 1)


DEFAULT_SUMMARY_CONFIG = SummaryRetrieveConfig()


def summary_config_from_dict(raw: Optional[Dict[str, Any]]) -> SummaryRetrieveConfig:
    raw = raw or {}
    return SummaryRetrieveConfig(
        top_docs=int(raw.get("top_docs", 5)),
        per_query_k=int(raw.get("per_query_k", 5)),
    )
