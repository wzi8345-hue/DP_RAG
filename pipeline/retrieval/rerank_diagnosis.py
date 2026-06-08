"""Reranker 低分诊断: 从分数分布推断 retry 策略 (Phase 1 纯规则, 零 LLM)。

仅在 langgraph reranker 质量门控失败时调用; reranker 未启用或质量达标时不介入。

P1.2 (2026-05): 诊断阈值与门控同源 — 改读 RouteThresholds 矩阵, 取主导 chunk_type
对应的阈值, 避免与 gate 用不同基线产生互相打架的判定 (问题 #6 修复)。
P2.1: confidence 由信号强度连续计算, 不再是硬编码常量 (问题 #8 修复)。
P1.3: 多 cause 并发诊断, 取主因 + 次因合成 patch (问题 #7 修复)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..models import RouteDecision
from .retrievers import Hit, parse_query
from .quality_thresholds import RouteThresholds
from .agentic import (
    ROUTE_SUMMARY,
    ROUTE_PROGRESSIVE,
    ROUTE_LOCAL,
    ROUTE_METADATA,
    _FIG_REF_LITE_RE,
    _TAB_REF_LITE_RE,
    _REFERENCES_HINT_RE,
    _DOC_REF_RE,
    _LOOSE_DOC_REF_RE,
    _extract_page_refs,
    _extract_paragraph_refs,
    _extract_entities,
)

def _retrieve_bias_for_rerank_patch(
    *,
    chunk_type: Optional[str] = None,
    routes: Optional[List[str]] = None,
    query: str = "",
) -> Optional[str]:
    """rerank 诊断 patch 的 retrieve_bias (与 router 枚举一致)。"""
    from .hybrid_weights import infer_retrieve_bias_heuristic

    if chunk_type == "references":
        return "entity_heavy"
    if routes and ROUTE_METADATA in routes:
        return "keyword"
    return infer_retrieve_bias_heuristic(query, chunk_type=chunk_type)


_SUMMARY_Q_RE = re.compile(
    r"总结|汇总|概述|综述|对比|主要内容|主要贡献|summarize|overview|main",
    re.IGNORECASE,
)


@dataclass
class RerankDiagnosisConfig:
    """Phase 2: skip_reflect_confidence + skip_reflect_causes 控制高置信时跳过 reflect。

    confidence 分层 (P1 #9):
      - wrong_type 显式 fig/tab 引用且对应 chunk 缺失/低分  → wrong_type_strong_confidence
      - wrong_type 仅 page/paragraph 引用                  → wrong_type_weak_confidence
      - wrong_type references 意图                         → wrong_type_refs_confidence
      - wrong_route                                        → wrong_route_confidence
    """
    enabled: bool = True
    # 默认从 0.85 提高到 0.90: 只有 "强信号" 的 wrong_type (fig/tab 引用 + 对应 chunk 缺失)
    # 才能跨过门槛跳过 reflect; 仅有页/段引用或者纯文本误召回都会去 reflect 兜底
    skip_reflect_confidence: float = 0.90
    skip_reflect_causes: Tuple[str, ...] = ("wrong_type", "wrong_route")
    type_low_ratio: float = 0.5
    route_dead_score: float = 0.15
    narrow_hit_cap: int = 3
    broad_hit_floor: int = 15
    off_topic_confidence: float = 0.3
    # P1 #9: 拆分 wrong_type 内置 confidence, 避免与 skip_reflect_confidence 贴脸跳过
    wrong_type_strong_confidence: float = 0.92  # 显式 fig/tab 引用 → 强信号
    wrong_type_weak_confidence: float = 0.80    # 仅 page/paragraph 引用 → 弱信号
    wrong_type_refs_confidence: float = 0.86    # references 意图
    wrong_route_confidence: float = 0.86
    too_narrow_confidence: float = 0.72
    too_narrow_relax_confidence: float = 0.70
    too_broad_confidence: float = 0.75
    zero_confidence: float = 0.35
    # P1.3: 多 cause 并发时, 次因合并入主因 patch 的最低置信门槛
    # (与主因差距 ≤ 0.25 且置信 ≥ 此值 才合并; 既避免噪音又允许 too_broad 等次因生效)
    multi_cause_min_secondary_confidence: float = 0.65


@dataclass
class RerankDiagnosis:
    cause: str
    confidence: float
    suggested: RouteDecision
    summary: str
    skip_reflect: bool = False


@dataclass
class _ScoreStats:
    total_hits: int
    global_avg: float
    route_avg: Dict[str, float]
    route_max: Dict[str, float]
    type_avg: Dict[str, float]
    type_counts: Dict[str, int]
    # P1.2: route 级主导 type 映射, 用于诊断层 per-route 阈值查找
    route_dom_type: Dict[str, str] = field(default_factory=dict)
    route_dom_stage: Dict[str, str] = field(default_factory=dict)

    def dominant_type(self) -> str:
        """全局主导 chunk_type (用于 R3/R4/R5 阈值查找)。"""
        if not self.type_counts:
            return ""
        return max(self.type_counts.items(), key=lambda kv: kv[1])[0]

    def route_threshold(
        self, route: str, thresholds: RouteThresholds, fallback: float,
    ) -> float:
        """诊断层取 per-route 阈值的统一入口。"""
        dt = self.route_dom_type.get(route, self.dominant_type())
        ds = self.route_dom_stage.get(route, "")
        val, _ = thresholds.for_(route, ds, dt)
        return val if val is not None else fallback


def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _normalize_scores_per_group(
    score_map: Dict[int, Optional[float]],
    rerank_groups: Optional[Dict[int, str]],
) -> Dict[int, Optional[float]]:
    """P2.2: 跨 rerank group (= 子查询) 做 z-score 归一化, 让统计可比。

    门控 (reranker_node.quality_pool) 用原始分; 诊断统计用归一化分, 避免
    高难度子查询 (0.1-0.3 区间) 与常识子查询 (0.7-0.9 区间) 混在一起拉偏 global_avg。

    没传 rerank_groups (或只有 1 组) 时, 不做归一化, 行为与旧版一致。
    """
    if not rerank_groups or len(set(rerank_groups.values())) <= 1:
        return dict(score_map)

    out: Dict[int, Optional[float]] = dict(score_map)
    grouped: Dict[str, List[Tuple[int, float]]] = {}
    for idx, gid in rerank_groups.items():
        v = score_map.get(idx)
        if v is None:
            continue
        grouped.setdefault(gid, []).append((idx, float(v)))

    for gid, pairs in grouped.items():
        if len(pairs) < 2:
            continue
        vals = [v for _, v in pairs]
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        sd = (var ** 0.5) or 1e-3
        # 把 z-score 映射回 [0, 1] 风格的"等效绝对分", 让阈值依然可用:
        # 归一化分 = mu(global) + z * sd(global), 其中 global 用所有 group 的总均/总 sd
        # 这里简化: 输出 z-score, 让诊断的相对判定 (avg/max 比) 仍然合理
        # 注: 诊断里的阈值比对 (e.g., * 0.8) 对 z-score 也仍是相对比较, 不会失真
        for idx, v in pairs:
            out[idx] = (v - mu) / sd

    return out


def _aggregate_scores(
    all_hits: List[Tuple[str, Hit]],
    score_map: Dict[int, Optional[float]],
    *,
    rerank_groups: Optional[Dict[int, str]] = None,
) -> _ScoreStats:
    """聚合 rerank 分数统计。

    P0 #10: score_map 中 None = 未评分 (reranker 未返回该 idx), 跳过统计;
    显式 0.0 = reranker 评为不相关, 仍计入. 这样未评分不会污染 avg/max。
    type_counts 始终基于全量 hit (反映检索池实际类型分布), 与是否评分无关。
    P1.2: 额外收集 route_dom_type / route_dom_stage, 供诊断层做 per-route 阈值查找。
    P2.2: ``rerank_groups`` (idx → group_id) 提供时, 跨 group 做 z-score 归一化,
          让多子查询场景的统计可比 (问题 #11 修复)。
    """
    norm_map = _normalize_scores_per_group(score_map, rerank_groups)
    route_scores: Dict[str, List[float]] = {}
    type_scores: Dict[str, List[float]] = {}
    type_counts: Dict[str, int] = {}
    route_type_counts: Dict[str, Dict[str, int]] = {}
    route_stage_counts: Dict[str, Dict[str, int]] = {}
    all_scores: List[float] = []

    for i, (route, hit) in enumerate(all_hits):
        raw = norm_map.get(i)
        chunk_type = (hit.type or "unknown").lower()
        stage = (getattr(hit, "stage", "") or "").lower()
        type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        rt_buckets = route_type_counts.setdefault(route, {})
        rt_buckets[chunk_type] = rt_buckets.get(chunk_type, 0) + 1
        if stage:
            rs_buckets = route_stage_counts.setdefault(route, {})
            rs_buckets[stage] = rs_buckets.get(stage, 0) + 1
        if raw is None:
            continue
        score = float(raw)
        all_scores.append(score)
        route_scores.setdefault(route, []).append(score)
        type_scores.setdefault(chunk_type, []).append(score)

    route_avg = {r: _avg(v) for r, v in route_scores.items()}
    route_max = {r: max(v) if v else 0.0 for r, v in route_scores.items()}
    type_avg = {t: _avg(v) for t, v in type_scores.items()}

    route_dom_type = {
        r: (max(b.items(), key=lambda kv: kv[1])[0] if b else "")
        for r, b in route_type_counts.items()
    }
    route_dom_stage = {
        r: (max(b.items(), key=lambda kv: kv[1])[0] if b else "")
        for r, b in route_stage_counts.items()
    }

    return _ScoreStats(
        total_hits=len(all_hits),
        global_avg=_avg(all_scores),
        route_avg=route_avg,
        route_max=route_max,
        type_avg=type_avg,
        type_counts=type_counts,
        route_dom_type=route_dom_type,
        route_dom_stage=route_dom_stage,
    )


def _extract_retry_keywords(query: str) -> str:
    pq = parse_query(query)
    if pq.keywords:
        return " ".join(pq.keywords[:8])
    entities = _extract_entities(query)
    if entities:
        return " ".join(entities[:6])
    stripped = query.strip()
    return stripped if stripped else query


def _extract_query_intent(query: str) -> Dict[str, Any]:
    fig_refs = sorted({m.group(1).upper() for m in _FIG_REF_LITE_RE.finditer(query)})
    tab_refs = sorted({m.group(1).upper() for m in _TAB_REF_LITE_RE.finditer(query)})
    page_refs = _extract_page_refs(query)
    paragraph_refs = _extract_paragraph_refs(query)
    entities = _extract_entities(query)
    has_refs_intent = bool(_REFERENCES_HINT_RE.search(query))
    has_doc_ref = bool(_DOC_REF_RE.search(query)) or bool(_LOOSE_DOC_REF_RE.search(query))
    is_summary_q = bool(_SUMMARY_Q_RE.search(query))
    return {
        "fig_refs": fig_refs,
        "tab_refs": tab_refs,
        "page_refs": page_refs,
        "paragraph_refs": paragraph_refs,
        "entities": entities,
        "has_refs_intent": has_refs_intent,
        "has_doc_ref": has_doc_ref,
        "is_summary_q": is_summary_q,
        "has_structured_intent": bool(
            fig_refs or tab_refs or page_refs or paragraph_refs or entities
        ),
    }


def _merge_decision_for_retry(
    base: Optional[RouteDecision],
    patch: RouteDecision,
    *,
    query: str,
) -> RouteDecision:
    """在 base 决策上增量合并 patch (路径追加 + filter 并集)。"""
    base = base or RouteDecision()
    routes: List[str] = list(base.routes or [])
    for r in patch.routes or []:
        if r not in routes:
            routes.append(r)

    rewrites = dict(base.rewrites or {})
    for route in routes:
        if route in (patch.rewrites or {}):
            rewrites[route] = patch.rewrites[route]
        elif route not in rewrites:
            rewrites[route] = _extract_retry_keywords(query)

    def _union(a: List, b: List) -> List:
        out = list(a or [])
        seen = set(out)
        for x in b or []:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    chunk_type = patch.chunk_type or base.chunk_type
    target_docs = _union(base.target_docs, patch.target_docs)
    if patch.target_docs:
        target_docs = list(patch.target_docs) + [
            d for d in (base.target_docs or []) if d not in patch.target_docs
        ]
    # target_doc_ids: 诊断 patch 通常不会写, 直接继承 base 的规范 doc_id (避免丢失锚点)
    target_doc_ids = _union(base.target_doc_ids, patch.target_doc_ids)

    return RouteDecision(
        routes=routes[:3],
        rewrites=rewrites,
        time=patch.time or base.time or "",
        chunk_type=chunk_type,
        target_docs=target_docs,
        target_doc_ids=target_doc_ids,
        fig_refs=_union(base.fig_refs, patch.fig_refs),
        table_refs=_union(base.table_refs, patch.table_refs),
        page_refs=_union(base.page_refs, patch.page_refs),
        paragraph_refs=_union(base.paragraph_refs, patch.paragraph_refs),
        entities=_union(base.entities, patch.entities),
        retrieve_bias=patch.retrieve_bias or base.retrieve_bias,
        reasoning=patch.reasoning or "(rerank-diagnosis)",
    )


def _format_summary(
    cause: str,
    confidence: float,
    stats: _ScoreStats,
    suggested: RouteDecision,
) -> str:
    route_line = ", ".join(
        f"{r}={stats.route_avg.get(r, 0):.3f}" for r in sorted(stats.route_avg)
    ) or "(none)"
    type_line = ", ".join(
        f"{t}={stats.type_avg.get(t, 0):.3f}" for t in sorted(stats.type_avg)
    ) or "(none)"
    filters: List[str] = []
    if suggested.fig_refs:
        filters.append(f"fig_refs={suggested.fig_refs}")
    if suggested.table_refs:
        filters.append(f"table_refs={suggested.table_refs}")
    if suggested.page_refs:
        filters.append(f"page_refs={suggested.page_refs}")
    if suggested.chunk_type:
        filters.append(f"chunk_type={suggested.chunk_type}")
    filter_str = "; ".join(filters) if filters else "(none)"
    return (
        f"[Reranker 诊断] cause={cause} confidence={confidence:.2f}\n"
        f"  route_avg: {route_line}\n"
        f"  type_avg: {type_line}\n"
        f"  建议路径: {suggested.routes} | filters: {filter_str}\n"
        f"  说明: reflect 可采纳或覆盖上述建议。"
    )


def _scale_confidence(
    base: float,
    signal_01: float,
    *,
    delta: float = 0.06,
    floor: float = 0.50,
    ceiling: float = 0.99,
) -> float:
    """P2.1: 在 base 上下 ±delta 内连续浮动, 由 signal_01 (0~1) 控制。

    signal_01 = 1.0 → base + delta (信号极强, 拉满)
    signal_01 = 0.5 → base       (中等信号)
    signal_01 = 0.0 → base - delta (信号弱, 但仍 ≥ floor)
    """
    s = max(0.0, min(1.0, signal_01))
    val = base + delta * (s - 0.5) * 2  # [-delta, +delta]
    return max(floor, min(ceiling, val))


def _wrong_type_strong_signal(intent: Dict[str, Any], stats: _ScoreStats) -> float:
    """fig/tab 引用 + 对应 chunk 缺失/低分: 三信号融合 → 0~1。"""
    ref_count = len(intent["fig_refs"]) + len(intent["tab_refs"])
    img_cnt = stats.type_counts.get("image", 0)
    tab_cnt = stats.type_counts.get("table", 0)
    img_avg = stats.type_avg.get("image", 0.0)
    tab_avg = stats.type_avg.get("table", 0.0)
    # 信号 1: 引用越多越强 (3+ 满分)
    ref_strength = min(1.0, ref_count / 3.0) if ref_count else 0.0
    # 信号 2: 对应类型完全缺失 (vs 只是低分)
    fully_missing = (
        (intent["fig_refs"] and img_cnt == 0)
        or (intent["tab_refs"] and tab_cnt == 0)
    )
    miss_strength = 1.0 if fully_missing else 0.4
    # 信号 3: 与全局均分的 gap (越大 confidence 越高)
    relevant_avg = max(img_avg, tab_avg)
    if stats.global_avg > 1e-3 and relevant_avg > 0:
        gap_strength = max(0.0, 1.0 - relevant_avg / stats.global_avg)
    else:
        gap_strength = 0.8 if fully_missing else 0.4
    return (ref_strength + miss_strength + gap_strength) / 3.0


@dataclass
class _Candidate:
    cause: str
    confidence: float
    patch: RouteDecision


def _eval_wrong_type_structured(
    *, query: str, intent: Dict[str, Any], stats: _ScoreStats,
    base: RouteDecision, docs: List[Dict[str, str]], cfg: RerankDiagnosisConfig,
) -> Optional[_Candidate]:
    """R1: fig/tab/page/paragraph 引用 + 对应 chunk 缺失/低分 → metadata 补检索。"""
    if not (
        intent["fig_refs"] or intent["tab_refs"]
        or intent["page_refs"] or intent["paragraph_refs"]
    ):
        return None
    if ROUTE_METADATA in (base.routes or []):
        return None

    img_cnt = stats.type_counts.get("image", 0)
    tab_cnt = stats.type_counts.get("table", 0)
    img_avg = stats.type_avg.get("image", 0.0)
    tab_avg = stats.type_avg.get("table", 0.0)
    image_low = img_cnt == 0 or img_avg < stats.global_avg * cfg.type_low_ratio
    table_low = tab_cnt == 0 or tab_avg < stats.global_avg * cfg.type_low_ratio

    need_metadata = False
    strong_signal = False
    if intent["fig_refs"] and image_low:
        need_metadata, strong_signal = True, True
    if intent["tab_refs"] and table_low:
        need_metadata, strong_signal = True, True
    if intent["page_refs"] or intent["paragraph_refs"]:
        need_metadata = True

    if not need_metadata:
        return None

    meta_doc = None
    if base.target_docs:
        meta_doc = base.target_docs[0]
    elif docs:
        meta_doc = docs[0].get("doc_name") or docs[0].get("doc_id")
    patch = RouteDecision(
        routes=[ROUTE_METADATA],
        fig_refs=list(intent["fig_refs"]),
        table_refs=list(intent["tab_refs"]),
        page_refs=list(intent["page_refs"]),
        paragraph_refs=list(intent["paragraph_refs"]),
        entities=list(intent["entities"]),
        target_docs=[meta_doc] if meta_doc else [],
        retrieve_bias=_retrieve_bias_for_rerank_patch(
            routes=[ROUTE_METADATA], query=query,
        ),
        reasoning="(rerank-wrong_type)",
    )
    if strong_signal:
        signal = _wrong_type_strong_signal(intent, stats)
        conf = _scale_confidence(cfg.wrong_type_strong_confidence, signal)
    else:
        # page/paragraph 弱信号: 引用数 → strength
        page_count = len(intent["page_refs"]) + len(intent["paragraph_refs"])
        signal = min(1.0, page_count / 2.0)
        conf = _scale_confidence(cfg.wrong_type_weak_confidence, signal)
    return _Candidate(cause="wrong_type", confidence=conf, patch=patch)


def _eval_wrong_type_refs(
    *, query: str, intent: Dict[str, Any], stats: _ScoreStats,
    base: RouteDecision, docs: List[Dict[str, str]], cfg: RerankDiagnosisConfig,
    references_threshold: float,
) -> Optional[_Candidate]:
    """R2: 参考文献意图 + references chunk 分数低/缺失 → 改 chunk_type=references。"""
    if not intent["has_refs_intent"]:
        return None
    ref_avg = stats.type_avg.get("references", 0.0)
    ref_count = stats.type_counts.get("references", 0)
    ref_missing = ref_count == 0
    if not (ref_missing or ref_avg < references_threshold):
        return None

    if docs or intent["has_doc_ref"]:
        patch = RouteDecision(
            routes=[ROUTE_LOCAL],
            rewrites={ROUTE_LOCAL: _extract_retry_keywords(query)},
            chunk_type="references",
            retrieve_bias="entity_heavy",
            reasoning="(rerank-wrong_type-refs)",
        )
    else:
        patch = RouteDecision(
            routes=[ROUTE_PROGRESSIVE],
            rewrites={ROUTE_PROGRESSIVE: _extract_retry_keywords(query)},
            chunk_type="references",
            retrieve_bias="entity_heavy",
            reasoning="(rerank-wrong_type-refs)",
        )
    # 信号强度: 完全缺失 > 仅低分; 距离阈值越远越强
    if ref_missing:
        signal = 0.85
    else:
        gap = max(0.0, 1.0 - ref_avg / max(references_threshold, 1e-3))
        signal = min(1.0, 0.3 + 0.6 * gap)
    conf = _scale_confidence(cfg.wrong_type_refs_confidence, signal)
    return _Candidate(cause="wrong_type", confidence=conf, patch=patch)


def _eval_wrong_route(
    *, query: str, intent: Dict[str, Any], stats: _ScoreStats,
    base: RouteDecision, docs: List[Dict[str, str]], cfg: RerankDiagnosisConfig,
    thresholds: RouteThresholds, default_threshold: float,
) -> Optional[_Candidate]:
    """R3: 某路径全灭 + 另一路径相对有分 → 追加存活路径 / local。

    P1.2: 改用 per-route 阈值, 不再用全局 quality_threshold * 0.8。
    """
    if not stats.route_avg:
        return None

    dead_routes = [
        r for r, mx in stats.route_max.items()
        if mx < cfg.route_dead_score
    ]
    alive_routes = []
    for r, avg in stats.route_avg.items():
        per_route_th = stats.route_threshold(r, thresholds, default_threshold)
        if avg >= max(per_route_th * 0.8, stats.global_avg * 0.6):
            alive_routes.append(r)

    if not (dead_routes and alive_routes):
        return None

    add_route = None
    if intent["is_summary_q"] and ROUTE_SUMMARY not in (base.routes or []):
        add_route = ROUTE_SUMMARY
    elif docs and ROUTE_LOCAL not in (base.routes or []):
        add_route = ROUTE_LOCAL
    elif (
        ROUTE_PROGRESSIVE not in (base.routes or [])
        and ROUTE_PROGRESSIVE in dead_routes
    ):
        add_route = ROUTE_SUMMARY if intent["is_summary_q"] else ROUTE_PROGRESSIVE

    if not add_route:
        return None

    patch = RouteDecision(
        routes=[add_route],
        rewrites={add_route: _extract_retry_keywords(query)},
        reasoning="(rerank-wrong_route)",
    )
    if add_route == ROUTE_LOCAL and docs and not base.target_docs:
        d = docs[0].get("doc_name") or docs[0].get("doc_id") or ""
        if d:
            patch.target_docs = [d]
    # 信号: dead 比例 + alive max 强度
    dead_ratio = len(dead_routes) / max(1, len(stats.route_avg))
    alive_max = max((stats.route_avg[r] for r in alive_routes), default=0.0)
    alive_strength = min(1.0, alive_max / max(default_threshold, 1e-3))
    signal = (dead_ratio + alive_strength) / 2.0
    conf = _scale_confidence(cfg.wrong_route_confidence, signal)
    return _Candidate(cause="wrong_route", confidence=conf, patch=patch)


def _eval_too_narrow(
    *, query: str, intent: Dict[str, Any], stats: _ScoreStats,
    base: RouteDecision, score_map: Dict[int, Optional[float]],
    cfg: RerankDiagnosisConfig,
    thresholds: RouteThresholds, default_threshold: float,
) -> Optional[_Candidate]:
    """R4: 命中少 + top 分尚可 → 加宽路径 / 放宽过滤。

    P1.2: 用全局主导 type 阈值替代 quality_threshold * 0.75。
    """
    valid_scores = sorted(
        (s for s in score_map.values() if s is not None), reverse=True,
    )
    top3_avg = _avg(valid_scores[:3])
    dom_th = thresholds.for_chunk_type(stats.dominant_type()) or default_threshold
    if not (
        stats.total_hits < cfg.narrow_hit_cap
        and top3_avg >= dom_th * 0.75
    ):
        return None

    add_route = ROUTE_SUMMARY if intent["is_summary_q"] else ROUTE_PROGRESSIVE
    if add_route not in (base.routes or []):
        patch = RouteDecision(
            routes=[add_route],
            rewrites={add_route: _extract_retry_keywords(query)},
            reasoning="(rerank-too_narrow)",
        )
        base_conf = cfg.too_narrow_confidence
    else:
        # 同路径: 去掉 chunk_type/time 等收紧条件
        patch = RouteDecision(
            routes=list(base.routes or [ROUTE_PROGRESSIVE]),
            rewrites={
                r: _extract_retry_keywords(query)
                for r in (base.routes or [ROUTE_PROGRESSIVE])
            },
            chunk_type=None,
            time="",
            reasoning="(rerank-too_narrow-relax)",
        )
        base_conf = cfg.too_narrow_relax_confidence
    # 信号: 命中越少 + top 分越接近阈值 → 越自信"是 too_narrow 而不是 off_topic"
    narrowness = 1.0 - stats.total_hits / max(1.0, cfg.narrow_hit_cap)
    top_strength = min(1.0, top3_avg / max(dom_th, 1e-3))
    signal = (max(0.0, narrowness) + top_strength) / 2.0
    conf = _scale_confidence(base_conf, signal)
    return _Candidate(cause="too_narrow", confidence=conf, patch=patch)


def _eval_too_broad(
    *, query: str, intent: Dict[str, Any], stats: _ScoreStats,
    base: RouteDecision, docs: List[Dict[str, str]], quality_score: float,
    cfg: RerankDiagnosisConfig,
    thresholds: RouteThresholds, default_threshold: float,
) -> Optional[_Candidate]:
    """R5: 命中多 + 整体低分 → 收缩到 local / progressive."""
    dom_th = thresholds.for_chunk_type(stats.dominant_type()) or default_threshold
    if not (stats.total_hits >= cfg.broad_hit_floor and quality_score < dom_th):
        return None

    if docs:
        doc_name = docs[0].get("doc_name") or docs[0].get("doc_id") or ""
        patch = RouteDecision(
            routes=[ROUTE_LOCAL],
            rewrites={ROUTE_LOCAL: _extract_retry_keywords(query)},
            target_docs=[doc_name] if doc_name else [],
            reasoning="(rerank-too_broad)",
        )
    else:
        patch = RouteDecision(
            routes=[ROUTE_PROGRESSIVE],
            rewrites={ROUTE_PROGRESSIVE: _extract_retry_keywords(query)},
            reasoning="(rerank-too_broad)",
        )
    # 信号: 命中越超量 + quality 越低 → 越自信
    over_ratio = min(1.0, stats.total_hits / max(1, cfg.broad_hit_floor * 2))
    quality_gap = max(0.0, 1.0 - quality_score / max(dom_th, 1e-3))
    signal = (over_ratio + quality_gap) / 2.0
    conf = _scale_confidence(cfg.too_broad_confidence, signal)
    return _Candidate(cause="too_broad", confidence=conf, patch=patch)


def _fallback_candidate(
    *, query: str, stats: _ScoreStats, quality_score: float,
    cfg: RerankDiagnosisConfig,
) -> _Candidate:
    """兜底: zero / off_topic."""
    kw = _extract_retry_keywords(query)
    if quality_score <= 0.05 and stats.total_hits > 0:
        patch = RouteDecision(
            routes=[ROUTE_PROGRESSIVE],
            rewrites={ROUTE_PROGRESSIVE: kw},
            reasoning="(rerank-zero-ish)",
        )
        return _Candidate(cause="zero", confidence=cfg.zero_confidence, patch=patch)
    patch = RouteDecision(
        routes=[ROUTE_PROGRESSIVE],
        rewrites={ROUTE_PROGRESSIVE: kw},
        reasoning="(rerank-off_topic)",
    )
    return _Candidate(
        cause="off_topic", confidence=cfg.off_topic_confidence, patch=patch,
    )


def _merge_secondary_patches(primary: RouteDecision, secondaries: List[RouteDecision]) -> RouteDecision:
    """P1.3: 把次因 patch 的 routes / filter 并入主因 patch (主因覆盖冲突字段)。"""
    if not secondaries:
        return primary
    routes = list(primary.routes or [])
    rewrites = dict(primary.rewrites or {})
    target_docs = list(primary.target_docs or [])
    target_doc_ids = list(primary.target_doc_ids or [])

    def _union_list(a, b):
        seen = set(a or [])
        out = list(a or [])
        for x in b or []:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    fig_refs = list(primary.fig_refs or [])
    table_refs = list(primary.table_refs or [])
    page_refs = list(primary.page_refs or [])
    paragraph_refs = list(primary.paragraph_refs or [])
    entities = list(primary.entities or [])

    for sec in secondaries:
        for r in sec.routes or []:
            if r not in routes and len(routes) < 3:
                routes.append(r)
        for r, kw in (sec.rewrites or {}).items():
            rewrites.setdefault(r, kw)
        target_docs = _union_list(target_docs, sec.target_docs)
        target_doc_ids = _union_list(target_doc_ids, sec.target_doc_ids)
        fig_refs = _union_list(fig_refs, sec.fig_refs)
        table_refs = _union_list(table_refs, sec.table_refs)
        page_refs = _union_list(page_refs, sec.page_refs)
        paragraph_refs = _union_list(paragraph_refs, sec.paragraph_refs)
        entities = _union_list(entities, sec.entities)

    return RouteDecision(
        routes=routes,
        rewrites=rewrites,
        time=primary.time or "",
        chunk_type=primary.chunk_type,
        target_docs=target_docs,
        target_doc_ids=target_doc_ids,
        fig_refs=fig_refs,
        table_refs=table_refs,
        page_refs=page_refs,
        paragraph_refs=paragraph_refs,
        entities=entities,
        retrieve_bias=primary.retrieve_bias,
        reasoning=primary.reasoning,
    )


def diagnose_rerank_failure(
    *,
    query: str,
    decision: Optional[RouteDecision],
    all_hits: List[Tuple[str, Hit]],
    score_map: Dict[int, Optional[float]],
    quality_score: float,
    quality_threshold: float,
    this_round_docs: Optional[List[Dict[str, str]]] = None,
    config: Optional[RerankDiagnosisConfig] = None,
    route_thresholds: Optional[RouteThresholds] = None,
    enable_multi_cause: bool = True,
    rerank_groups: Optional[Dict[int, str]] = None,
) -> RerankDiagnosis:
    """根据 reranker 分数分布产出 RouteDecision 改写建议。

    P1.2: ``route_thresholds`` 与 reranker_node 共享同一份 RouteThresholds, 让 R3/R4/R5
          的阈值判定与门控对齐 (问题 #6 修复)。
    P1.3: ``enable_multi_cause=True`` (默认) 时, 同时评估多个 cause, 取主因 +
          merge 次因 patch 的 routes/filter (问题 #7 修复)。
    P2.1: cause confidence 由信号强度连续计算, 不再是硬编码常量 (问题 #8 修复)。
    P2.2: ``rerank_groups`` (idx → group_id) 提供时, 跨子查询做 z-score 归一化,
          让 route_avg/type_avg 在多子查询场景下可比 (问题 #11 修复)。
    """
    cfg = config or RerankDiagnosisConfig()
    stats = _aggregate_scores(all_hits, score_map, rerank_groups=rerank_groups)
    intent = _extract_query_intent(query)
    base = decision or RouteDecision()
    docs = this_round_docs or []
    thresholds = route_thresholds or RouteThresholds(default=quality_threshold)
    # references 自己的阈值 (兜底 0.5 倍 default, 引文短行天然低分)
    ref_th = thresholds.for_chunk_type("references") or (quality_threshold * 0.5)

    # P1.3: 并发评估所有 cause, 收集候选
    candidates: List[_Candidate] = []
    structured = _eval_wrong_type_structured(
        query=query, intent=intent, stats=stats,
        base=base, docs=docs, cfg=cfg,
    )
    if structured:
        candidates.append(structured)

    refs = _eval_wrong_type_refs(
        query=query, intent=intent, stats=stats,
        base=base, docs=docs, cfg=cfg,
        references_threshold=ref_th,
    )
    if refs:
        candidates.append(refs)

    wrong_route = _eval_wrong_route(
        query=query, intent=intent, stats=stats,
        base=base, docs=docs, cfg=cfg,
        thresholds=thresholds, default_threshold=quality_threshold,
    )
    if wrong_route:
        candidates.append(wrong_route)

    narrow = _eval_too_narrow(
        query=query, intent=intent, stats=stats,
        base=base, score_map=score_map, cfg=cfg,
        thresholds=thresholds, default_threshold=quality_threshold,
    )
    if narrow:
        candidates.append(narrow)

    broad = _eval_too_broad(
        query=query, intent=intent, stats=stats,
        base=base, docs=docs, quality_score=quality_score, cfg=cfg,
        thresholds=thresholds, default_threshold=quality_threshold,
    )
    if broad:
        candidates.append(broad)

    if not candidates:
        primary = _fallback_candidate(
            query=query, stats=stats, quality_score=quality_score, cfg=cfg,
        )
        cause = primary.cause
        confidence = primary.confidence
        final_patch = primary.patch
    else:
        # 主因 = 置信度最高
        primary = max(candidates, key=lambda c: c.confidence)
        cause = primary.cause
        confidence = primary.confidence
        if enable_multi_cause:
            # P1.3 (#7 修复): 次因 = 与主因 cause 不同 + 置信度 ≥ 绝对门槛 (默认 0.65)
            # 让主因 patch 吸收次因的 routes/filter, 用一次 retry 覆盖多个失效模式
            secondary_floor = max(
                cfg.multi_cause_min_secondary_confidence,
                primary.confidence - 0.25,
            )
            secondaries = [
                c.patch for c in candidates
                if c is not primary
                and c.confidence >= secondary_floor
                and c.cause != primary.cause
            ]
            final_patch = _merge_secondary_patches(primary.patch, secondaries)
        else:
            final_patch = primary.patch

    suggested = _merge_decision_for_retry(base, final_patch, query=query)
    suggested.reasoning = f"(rerank-diagnosis:{cause})"

    summary = _format_summary(cause, confidence, stats, suggested)
    allowed_causes = set(cfg.skip_reflect_causes or ())
    skip_reflect = (
        confidence >= cfg.skip_reflect_confidence
        and cause in allowed_causes
    )

    return RerankDiagnosis(
        cause=cause,
        confidence=confidence,
        suggested=suggested,
        summary=summary,
        skip_reflect=skip_reflect,
    )


def should_skip_reflect_after_reranker(
    *,
    skip_reflect: bool,
    rewrite_hint: Any,
    subquery_decisions: Optional[List[RouteDecision]],
    retry_count: int,
    max_retries: int,
) -> bool:
    """Phase 2: 高置信诊断 + 有有效 RouteDecision + 未用尽 retry 预算 → 跳过 reflect。"""
    if not skip_reflect:
        return False
    if max_retries <= 0 or retry_count >= max_retries:
        return False
    if subquery_decisions:
        return True
    return isinstance(rewrite_hint, RouteDecision)
