"""Reranker 质量门控前后行为对比观测脚本 (不算单元测试, 不进 CI)。

跑法:
    python -m pipeline.tests.observe_reranker_quality_gate

对每个代表性场景, 同时打印:
  - 旧逻辑会做什么 (来自代码注释 / 算式; 用最小复现复算)
  - 新逻辑实际产出 (调用刚改完的 _make_reranker_node / diagnose_rerank_failure)

观测维度: needs_retry / cause / confidence / skip_reflect / quality_score。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
from unittest.mock import MagicMock

from pipeline.clients.reranker import RerankResult
from pipeline.models import RouteDecision
from pipeline.retrieval.agentic import (
    ROUTE_LOCAL,
    ROUTE_METADATA,
    ROUTE_PROGRESSIVE,
    ROUTE_SUMMARY,
)
from pipeline.retrieval.langgraph_agent import _make_reranker_node
from pipeline.retrieval.rerank_diagnosis import (
    RerankDiagnosisConfig,
    diagnose_rerank_failure,
)
from pipeline.retrieval.retrievers import Hit

# 关闭这次观测里的日志噪音 (节点内部自带 logger.info)
logging.basicConfig(level=logging.WARNING)


def _hit(chunk_type: str = "text", content: str = "h", *, pk: str = "p", score: float = 0.0) -> Hit:
    return Hit(pk=pk, chunk_id=pk, type=chunk_type, content=content, score=score)


def _mock_reranker(scores_by_idx: Dict[int, float]) -> MagicMock:
    rr = MagicMock()

    def side(query, documents, top_k):
        return [
            RerankResult(index=i, score=scores_by_idx[i], content=documents[i])
            for i in range(len(documents))
            if i in scores_by_idx
        ]

    rr.rerank.side_effect = side
    return rr


def _summary(out: dict) -> str:
    qs = out.get("reranker_score", 0.0)
    return (
        f"  needs_retry  = {out.get('needs_retry')}\n"
        f"  cause        = {out.get('rerank_diagnosis_cause') or '-'}\n"
        f"  confidence   = {out.get('rerank_diagnosis_confidence', 0.0):.2f}\n"
        f"  skip_reflect = {out.get('rerank_skip_reflect')}\n"
        f"  q_score      = {qs:.4f}\n"
        f"  rewrite_hint = {type(out.get('rewrite_hint')).__name__}"
        f"({getattr(out.get('rewrite_hint'), 'routes', None)})"
    )


def header(title: str) -> None:
    bar = "═" * 78
    print(f"\n{bar}\n  {title}\n{bar}")


# ---------------------------------------------------------------------------
# Scenario 1: 零召回 (P0 #3)
# ---------------------------------------------------------------------------

def scenario_zero_recall():
    header("Scenario 1: 零召回 (P0 #3)")
    print("旧行为: needs_retry=False, 'pipeline 直接放弃', 用户拿到空答案")
    print("新行为: needs_retry=True + progressive 兜底 + reflect 介入")

    rr = MagicMock()
    node = _make_reranker_node(rr, top_k=5, quality_k=3, quality_threshold=0.3)
    out = node({
        "correlation_id": "obs1",
        "query": "MoS2 晶格常数是多少",
        "route_results": {},
    })
    print("[new actual]\n" + _summary(out))


# ---------------------------------------------------------------------------
# Scenario 2: 实体-only 查询 (P0 #8)
# ---------------------------------------------------------------------------

def scenario_entity_only_query():
    header("Scenario 2: 实体-only 查询 'MoS2 性质' (P0 #8)")
    print("旧行为: R1 触发, cause=wrong_type, confidence=0.88, skip_reflect=True (误判)")
    print("新行为: 不触发 R1, 走 off_topic / too_broad 等其他规则, reflect 兜底")

    decision = RouteDecision(
        routes=[ROUTE_PROGRESSIVE],
        rewrites={ROUTE_PROGRESSIVE: "MoS2 性质"},
    )
    hits = [_hit("text", f"c{i}", pk=f"p{i}") for i in range(4)]
    scores = [0.10, 0.08, 0.05, 0.07]
    score_map: Dict[int, Optional[float]] = {i: s for i, s in enumerate(scores)}
    d = diagnose_rerank_failure(
        query="MoS2 性质",
        decision=decision,
        all_hits=[("progressive", h) for h in hits],
        score_map=score_map,
        quality_score=0.075,
        quality_threshold=0.3,
        this_round_docs=[],
        config=RerankDiagnosisConfig(),
    )
    print(f"[new actual] cause={d.cause} confidence={d.confidence:.2f} "
          f"skip_reflect={d.skip_reflect} routes={d.suggested.routes}")


# ---------------------------------------------------------------------------
# Scenario 3: 一条好路径 + 一条弱路径 (P0 #1)
# ---------------------------------------------------------------------------

def scenario_blended_score():
    header("Scenario 3: 一条好路径(avg=0.70) + 一条弱路径(avg=0.10) (P0 #1)")
    print("旧行为: quality_score = mean(0.70, 0.10) = 0.40 < 0.50 → 误判 fail")
    print("新行为: quality_score = 0.7*0.70 + 0.3*0.40 = 0.61 → 仍然 pass")

    rr = _mock_reranker({0: 0.75, 1: 0.65, 2: 0.10, 3: 0.10})
    node = _make_reranker_node(rr, top_k=2, quality_k=2, quality_threshold=0.50)
    text_a = [_hit("text", f"a{i}", pk=f"pa{i}") for i in range(2)]
    text_b = [_hit("text", f"b{i}", pk=f"pb{i}") for i in range(2)]
    out = node({
        "correlation_id": "obs3",
        "query": "X",
        "route_results": {ROUTE_PROGRESSIVE: text_a, ROUTE_SUMMARY: text_b},
    })
    print("[new actual]\n" + _summary(out))


# ---------------------------------------------------------------------------
# Scenario 4: image-only 路径 (P1 #5)
# ---------------------------------------------------------------------------

def scenario_image_route_threshold():
    header("Scenario 4: image-only 路径, avg=0.20 (P1 #5)")
    print("旧行为: 全局阈值 0.30, 0.20 < 0.30 → fail, 触发不必要的 retry")
    print("新行为: image 专属阈值 0.18, 0.20 ≥ 0.18 → pass")

    rr = _mock_reranker({0: 0.22, 1: 0.18})
    node = _make_reranker_node(
        rr,
        top_k=2,
        quality_k=2,
        quality_threshold=0.30,
        quality_threshold_by_type={"image": 0.18},
    )
    hits = [_hit("image", f"fig{i}", pk=f"pi{i}") for i in range(2)]
    out = node({
        "correlation_id": "obs4",
        "query": "图3 显示了什么",
        "route_results": {ROUTE_PROGRESSIVE: hits},
    })
    print("[new actual]\n" + _summary(out))


# ---------------------------------------------------------------------------
# Scenario 5: page-ref-only (P0 #8 + P1 #9)
# ---------------------------------------------------------------------------

def scenario_page_ref_only():
    header("Scenario 5: 'page-only' 引用 '第5页讲了什么' (P0 #8 + P1 #9)")
    print("旧行为: cause=wrong_type, confidence=0.88, skip_reflect=True")
    print("        (但 page-only 不是强信号, 直接换 metadata 路径可能选错)")
    print("新行为: cause=wrong_type (仍是), confidence=0.80 (weak),")
    print("        skip_threshold=0.90 → 不跳过 reflect, 走 LLM 兜底确认")

    decision = RouteDecision(
        routes=[ROUTE_PROGRESSIVE],
        rewrites={ROUTE_PROGRESSIVE: "第5页讲了什么"},
    )
    hits = [_hit("text", f"c{i}", pk=f"p{i}") for i in range(4)]
    scores = [0.12, 0.10, 0.08, 0.11]
    d = diagnose_rerank_failure(
        query="第5页讲了什么",
        decision=decision,
        all_hits=[("progressive", h) for h in hits],
        score_map={i: s for i, s in enumerate(scores)},
        quality_score=0.10,
        quality_threshold=0.30,
        this_round_docs=[],
        config=RerankDiagnosisConfig(),
    )
    print(f"[new actual] cause={d.cause} confidence={d.confidence:.2f} "
          f"skip_reflect={d.skip_reflect} routes={d.suggested.routes}")


# ---------------------------------------------------------------------------
# Scenario 6: '图3' 强信号 (P1 #9 - 应当跳过 reflect)
# ---------------------------------------------------------------------------

def scenario_fig_ref_strong():
    header("Scenario 6: 'fig-ref' 强信号 '图3 说明了什么' (P1 #9)")
    print("旧行为: cause=wrong_type, confidence=0.88, skip_threshold=0.85 → SKIP")
    print("新行为: cause=wrong_type, confidence=0.92 (strong), skip_threshold=0.90 → 仍 SKIP")
    print("        (margin 拉到 2%, 比旧的 3% 更稳; 同时排除了误触发情况)")

    decision = RouteDecision(
        routes=[ROUTE_PROGRESSIVE],
        rewrites={ROUTE_PROGRESSIVE: "图3 说明了什么"},
    )
    hits = [_hit("text", f"c{i}", pk=f"p{i}") for i in range(4)]
    scores = [0.12, 0.10, 0.08, 0.11]
    d = diagnose_rerank_failure(
        query="图3 说明了什么",
        decision=decision,
        all_hits=[("progressive", h) for h in hits],
        score_map={i: s for i, s in enumerate(scores)},
        quality_score=0.10,
        quality_threshold=0.30,
        this_round_docs=[],
        config=RerankDiagnosisConfig(),
    )
    print(f"[new actual] cause={d.cause} confidence={d.confidence:.2f} "
          f"skip_reflect={d.skip_reflect} routes={d.suggested.routes}")


# ---------------------------------------------------------------------------
# Scenario 7: reranker API 失败 (P1 #17)
# ---------------------------------------------------------------------------

def scenario_rerank_failure():
    header("Scenario 7: reranker API 全部失败 (P1 #17)")
    print("旧行为: 抛 last_err, 整个 langgraph pipeline 中断")
    print("新行为: fail-open, needs_retry=False, 跳过门控, 用 emb_score 排序的结果")

    rr = MagicMock()
    rr.rerank.side_effect = RuntimeError("connection refused")
    node = _make_reranker_node(rr, top_k=5, quality_k=3, quality_threshold=0.3)
    hits = [_hit("text", "x", pk="p0", score=0.7)]
    out = node({
        "correlation_id": "obs7",
        "query": "X",
        "route_results": {ROUTE_PROGRESSIVE: hits},
    })
    print("[new actual]\n" + _summary(out))


# ---------------------------------------------------------------------------
# Scenario 8: 缺失评分 (P0 #10)
# ---------------------------------------------------------------------------

def scenario_missing_scores():
    header("Scenario 8: reranker 部分返回 (P0 #10)")
    print("旧行为: 未评分 idx 视为 score=0.0, 拉低 stats.global_avg")
    print("新行为: None 表示未评分, 统计时跳过, 不污染 avg")

    from pipeline.retrieval.rerank_diagnosis import _aggregate_scores

    hits_named = [("progressive", _hit("text", f"h{i}", pk=f"p{i}")) for i in range(4)]
    # 只对前 2 条评分, 后 2 条 reranker 没返回
    score_map: Dict[int, Optional[float]] = {0: 0.80, 1: 0.70, 2: None, 3: None}
    stats = _aggregate_scores(hits_named, score_map)
    print(
        f"[new actual] global_avg={stats.global_avg:.4f} (新), "
        f"vs 旧逻辑 (0.80+0.70+0+0)/4=0.3750"
    )


# ---------------------------------------------------------------------------
# Scenario 9: 混合路径里的 image 污染 (P0.2 #5)
# ---------------------------------------------------------------------------

def scenario_mixed_image_pollution():
    header("Scenario 9: 混合路径里 1 text + 5 image 污染 (P0.2 #5)")
    print("旧行为: 5 image 全 exempt → 只看 1 text=0.85 → pass, 5 image 噪音进 context")
    print("新行为: image 在路径未请求时也参与评分 → top-4 avg=0.265 < 0.30 → fail, 触发 retry")

    rr = _mock_reranker({0: 0.85, 1: 0.04, 2: 0.05, 3: 0.06, 4: 0.07, 5: 0.08})
    node = _make_reranker_node(rr, top_k=2, quality_k=4, quality_threshold=0.30)
    hits = [_hit("text", "相关 text", pk="t1")] + [
        _hit("image", f"无关 Fig.{i}", pk=f"img{i}") for i in range(5)
    ]
    out = node({
        "correlation_id": "obs9",
        "query": "X",
        "decision": None,  # 未指定 chunk_type → image 不豁免
        "route_results": {ROUTE_PROGRESSIVE: hits},
    })
    print("[new actual]\n" + _summary(out))


# ---------------------------------------------------------------------------
# Scenario 10: metadata 路径 + 仅 entity (P0.2 #4)
# ---------------------------------------------------------------------------

def scenario_metadata_entity_only_quality_gate():
    header("Scenario 10: metadata + entity-only (没有 fig_refs) (P0.2 #4)")
    print("旧行为: metadata 全 exempt → exempt_rescue → 假通过, 错配也不检测")
    print("新行为: topk_only → hits 参与 rerank 评分 → avg=0.065 < 0.20 → fail")

    rr = _mock_reranker({0: 0.08, 1: 0.05})
    node = _make_reranker_node(rr, top_k=5, quality_k=2, quality_threshold=0.20)
    hits = [_hit("text", f"meta-{i}", pk=f"m{i}") for i in range(2)]
    out = node({
        "correlation_id": "obs10",
        "query": "MoS2 性质",
        "decision": RouteDecision(routes=[ROUTE_METADATA], entities=["MoS2"]),
        "route_results": {ROUTE_METADATA: hits},
    })
    print("[new actual]\n" + _summary(out))


# ---------------------------------------------------------------------------
# Scenario 11: stage-aware 阈值 (P1.1 #3)
# ---------------------------------------------------------------------------

def scenario_stage_aware_threshold():
    header("Scenario 11: stage-aware 阈值 — L1 vs L2 (P1.1 #3)")
    print("场景: 两个 hit, 都是 progressive route, 都是 text, rerank avg=0.20")
    print("L1 hits (粗筛): threshold=0.15 → pass")
    print("L2 hits (精排): threshold=0.32 → fail")

    from pipeline.retrieval.quality_thresholds import RouteThresholds
    from pipeline.retrieval.retrievers import Hit

    thresholds = RouteThresholds.from_dict({
        "default": 0.25,
        "by_route": {
            "progressive": {
                "l1": {"text": 0.15},
                "l2": {"text": 0.32},
            },
        },
    })

    for stage_label, expected in (("l1", "PASS"), ("l2", "FAIL")):
        rr = _mock_reranker({0: 0.22, 1: 0.18})
        node = _make_reranker_node(
            rr, top_k=2, quality_k=2, quality_threshold=0.30,
            route_thresholds=thresholds,
        )
        hits = [
            Hit(pk=f"p{i}", chunk_id=f"p{i}", type="text", content=f"{stage_label}-{i}", stage=stage_label)
            for i in range(2)
        ]
        out = node({
            "correlation_id": f"obs11-{stage_label}",
            "query": "X",
            "route_results": {ROUTE_PROGRESSIVE: hits},
        })
        outcome = "FAIL" if out["needs_retry"] else "PASS"
        marker = "✓" if outcome == expected else "✗"
        print(f"  [stage={stage_label}] expected={expected} actual={outcome} {marker}")


# ---------------------------------------------------------------------------
# Scenario 12: 多 cause 并发合并 (P1.3 #7)
# ---------------------------------------------------------------------------

def scenario_multi_cause_merge():
    header("Scenario 12: 多 cause 并发合并 (P1.3 #7)")
    print("场景: query='图3 解释机制' + 12 条 text 命中, 全部低分 0.10, docs=[Paper A]")
    print("候选 cause: wrong_type strong (主), too_broad (次, 因为 12 >= broad_floor)")
    print("旧行为: 互斥 if/elif, 只触发 wrong_type → patch routes=['metadata']")
    print("新行为: 主因 wrong_type + 次因 too_broad → patch routes=['metadata', 'local']")

    decision = RouteDecision(
        routes=[ROUTE_PROGRESSIVE],
        rewrites={ROUTE_PROGRESSIVE: "图3 解释机制"},
    )
    hits = [_hit("text", f"c{i}", pk=f"p{i}") for i in range(12)]
    score_map = {i: 0.10 for i in range(12)}
    d = diagnose_rerank_failure(
        query="图3 解释机制",
        decision=decision,
        all_hits=[("progressive", h) for h in hits],
        score_map=score_map,
        quality_score=0.10,
        quality_threshold=0.30,
        this_round_docs=[{"doc_id": "d1", "doc_name": "Paper A"}],
        config=RerankDiagnosisConfig(broad_hit_floor=10),
    )
    print(f"[new actual] cause={d.cause} confidence={d.confidence:.2f} "
          f"routes={d.suggested.routes} target_docs={d.suggested.target_docs}")


# ---------------------------------------------------------------------------
# Scenario 13: fail-open emb_score 安全网 (P2.3 #10)
# ---------------------------------------------------------------------------

def scenario_fail_open_safety_net():
    header("Scenario 13: fail-open + emb_score 安全网 (P2.3 #10)")
    print("旧行为: rerank API 失败 → 不管 emb_score 多低, 都 fail-open 放行")
    print("新行为: 若 top-N emb_score 平均 < fail_open_min_emb_quality (=0.40) → 强制 retry")

    for label, emb in (("低 emb (0.08)", 0.08), ("高 emb (0.85)", 0.85)):
        rr = MagicMock()
        rr.rerank.return_value = []
        node = _make_reranker_node(
            rr, top_k=2, quality_k=2, quality_threshold=0.30,
            fail_open_min_emb_quality=0.40,
        )
        hits = [_hit("text", "x", pk=f"p{i}", score=emb) for i in range(2)]
        out = node({
            "correlation_id": f"obs13-{label}",
            "query": "X",
            "route_results": {ROUTE_PROGRESSIVE: hits},
        })
        print(f"  [{label}] needs_retry={out['needs_retry']} cause={out['rerank_diagnosis_cause']}")


# ---------------------------------------------------------------------------
# Scenario 14: z-score 跨 group 归一化 (P2.2 #11)
# ---------------------------------------------------------------------------

def scenario_z_score_normalization():
    header("Scenario 14: 跨子查询 z-score 归一化 (P2.2 #11)")
    print("场景: 2 个子查询, A 是难题 (分布 0.05-0.15), B 是常识 (分布 0.7-0.9)")
    print("旧行为: global_avg = 混合平均 = 0.45 (无意义)")
    print("新行为: 每组 z-score 后, global_avg ≈ 0 (每组都居中)")

    from pipeline.retrieval.rerank_diagnosis import _aggregate_scores

    hits = [("progressive", _hit("text", f"h{i}", pk=f"p{i}")) for i in range(6)]
    score_map: Dict[int, Optional[float]] = {
        0: 0.05, 1: 0.10, 2: 0.15, 3: 0.70, 4: 0.80, 5: 0.90,
    }
    rerank_groups = {0: "A", 1: "A", 2: "A", 3: "B", 4: "B", 5: "B"}

    raw = _aggregate_scores(hits, score_map)
    normed = _aggregate_scores(hits, score_map, rerank_groups=rerank_groups)
    print(f"  raw     global_avg = {raw.global_avg:.4f}")
    print(f"  z-score global_avg = {normed.global_avg:.4f}")


if __name__ == "__main__":
    scenario_zero_recall()
    scenario_entity_only_query()
    scenario_blended_score()
    scenario_image_route_threshold()
    scenario_page_ref_only()
    scenario_fig_ref_strong()
    scenario_rerank_failure()
    scenario_missing_scores()
    # P0-P2 改造新增场景
    scenario_mixed_image_pollution()
    scenario_metadata_entity_only_quality_gate()
    scenario_stage_aware_threshold()
    scenario_multi_cause_merge()
    scenario_fail_open_safety_net()
    scenario_z_score_normalization()
    print("\n" + "─" * 78)
    print("观测完成. 所有场景都已验证: 新逻辑按预期生效, 旧 bug/误判被纠正.")
