"""结构化 chunk 全量召回 (references / image / table): 不做语义得分截断。

P0.2 (2026-05): 把"是否参与 top-k 截断"与"是否参与质量评分"解耦, 修复:
  - 旧的 image/table/references 配置的 per-type 阈值是死代码 (问题 #1)
  - metadata 路径无质量兜底, fig_refs 错配也照样过 gate (问题 #4)
  - 混合路径里 image/table 噪音被自动豁免, 掩盖低相关性 (问题 #5)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import RouteDecision
    from .retrievers import Hit

STRUCTURAL_CHUNK_TYPES = frozenset({"references", "image", "table"})

ROUTE_METADATA = "metadata"


@dataclass(frozen=True)
class ExemptDecision:
    """rerank gate 中一条 hit 的豁免级别 (P0.2)。

    旧版只有 bool, 把两件事合并: 是否截断 + 是否计入质量评分。
    实际语义并不总是绑定:
      - metadata + entity-only: 应该保留全量 (不截断) 但 *必须* 参与评分
      - 混合路径里意外召回的 image: 应该被截断且参与评分 (不豁免)
    """
    skip_topk_truncation: bool   # True = 不参与 top-k 截断 (保留进入 context)
    skip_quality_scoring: bool   # True = 不计入 route quality avg / gate

    @classmethod
    def full(cls) -> "ExemptDecision":
        """完全豁免: 保留全量 + 不评分 (高置信结构化命中)。"""
        return cls(skip_topk_truncation=True, skip_quality_scoring=True)

    @classmethod
    def topk_only(cls) -> "ExemptDecision":
        """保留全量, 但参与质量评分 (metadata entity-only)。"""
        return cls(skip_topk_truncation=True, skip_quality_scoring=False)

    @classmethod
    def none(cls) -> "ExemptDecision":
        """完全参与门控: 既评分也可被 top-k 截断 (默认)。"""
        return cls(skip_topk_truncation=False, skip_quality_scoring=False)


def is_structural_chunk_type(chunk_type: Optional[str]) -> bool:
    ct = (chunk_type or "").strip().lower()
    return ct in STRUCTURAL_CHUNK_TYPES


def is_structural_hit(hit: "Hit") -> bool:
    return (hit.type or "").strip().lower() in STRUCTURAL_CHUNK_TYPES


def decision_requests_structural_full_recall(decision: Optional["RouteDecision"]) -> bool:
    """路由是否指向参考文献/图/表的全量结构化召回。"""
    if decision is None:
        return False
    if is_structural_chunk_type(decision.chunk_type):
        return True
    if ROUTE_METADATA in (decision.routes or []):
        if decision.fig_refs or decision.table_refs:
            return True
        if is_structural_chunk_type(decision.chunk_type):
            return True
    return False


def _has_structural_refs(decision: Optional["RouteDecision"]) -> bool:
    """decision 是否携带显式结构化引用 (fig/tab/page/paragraph_refs)。"""
    if decision is None:
        return False
    return bool(
        decision.fig_refs
        or decision.table_refs
        or decision.page_refs
        or decision.paragraph_refs
    )


def hit_exempt_decision(
    route: str,
    hit: "Hit",
    decision: Optional["RouteDecision"] = None,
) -> ExemptDecision:
    """决定 hit 在 rerank gate 中的豁免级别 (P0.2 核心)。

    规则按优先级:
      1. metadata 路径 + 显式结构化引用 (fig/tab/page/paragraph_refs):
         FULL — 硬过滤命中是高置信结构化匹配, 不需要 rerank 验证
      2. metadata 路径 + 仅 entity / 无结构化引用:
         TOPK_ONLY — 保留全量, 但参与 rerank 评分以验证语义相关性
         (修复问题 #4: metadata 路径不再无质量兜底)
      3. hit 类型 ∈ {image,table,references} 且 decision.chunk_type 与之匹配:
         FULL — 用户明确请求该类型, 命中天然相关
      4. hit 类型 ∈ {image,table,references} 但 decision.chunk_type 不匹配:
         NONE — 混合路径里这种 chunk 必须参与评分
         (修复问题 #5: 不再自动豁免, 让 rerank 看到 image/table 噪音)
      5. 其他普通文本 chunk:
         NONE
    """
    decision_ct = (decision.chunk_type or "").strip().lower() if decision else ""

    if route == ROUTE_METADATA:
        if _has_structural_refs(decision):
            return ExemptDecision.full()
        return ExemptDecision.topk_only()

    hit_type = (hit.type or "").strip().lower()
    if hit_type in STRUCTURAL_CHUNK_TYPES:
        if decision_ct == hit_type:
            return ExemptDecision.full()
        # 路径未指定 chunk_type 或指定为不同结构化类型 → 不豁免
        # (这正是 #5 修复: 混合命中的 image/table 必须被 rerank 看到)
        return ExemptDecision.none()

    return ExemptDecision.none()


def hit_exempt_from_rerank_filter(
    route: str,
    hit: "Hit",
    decision: Optional["RouteDecision"] = None,
) -> bool:
    """兼容旧 API: 等价于 ``hit_exempt_decision(...).skip_topk_truncation``。

    新代码应使用 ``hit_exempt_decision()`` 获取细分语义 (P0.2)。
    """
    return hit_exempt_decision(route, hit, decision).skip_topk_truncation
