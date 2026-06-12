"""专业研究模式 (professional) 的 LangGraph 子图与智能体。

与普通 build_langgraph_agent **完全独立**: 本模块新建一张研究专用图, 通过 import 复用
现有的 retrieve_node / reranker_node 等节点工厂, 但不修改它们, 也不接入现有 policy 枢纽图。
因此专业模式对现有链路零影响 (只在上层显式 professional=True 时才会走到这里)。

研究闭环 (检索效率优先):
    plan(规划: 1 次 LLM, 思考可开)
      → retrieve(并行批次) → [reranker] → research_policy(观测+决策: 1 次 LLM/轮)
          ├─ continue : 清空本轮 route_results, 装下一轮批次, 回到 retrieve
          ├─ finish   : 用累计 evidence 渲染研究综述 context, 结束
          └─ clarify  : 语料缺失/目标过宽, 反问用户, 结束

效率要点:
  - 每轮只对"本轮新批次"的结果做 rerank (轮间清空 route_results), 避免对累计大池重复精排;
  - 证据按 chunk 主键去重累积 (evidence_hits), 不重复入册;
  - 轮次预算 + 连续无新增证据熔断 (stall), 防止空耗检索;
  - policy LLM 只看压缩观测 (summarize_for_reflect + 覆盖度), 不灌全文。
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Optional

try:
    from langgraph.graph import StateGraph, END
except ImportError:  # pragma: no cover
    StateGraph = None  # type: ignore[assignment,misc]
    END = None  # type: ignore[assignment]

from ..clients.llm import LLMClient
from ..clients.reranker import RerankerClient
from ..routing.limits import DEFAULT_ROUTING_LIMITS, RoutingLimits
from ..routing.research import (
    PolicyDecision,
    ResearchPlan,
    batches_to_multi_decision,
    compose_plan_system,
    compose_policy_system,
    decide_policy,
    plan_research,
)
from ..routing.research_skills import ResearchSkill, evaluate_guards, select_skill
from .agentic import AgenticRAGPipeline
from .reflect_summary import (
    ReflectSummaryConfig,
    collect_reflect_hits_by_route,
    summarize_for_reflect,
)
from .retrievers import Hit
from . import langgraph_agent as _lg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ResearchState(_lg.AgentState, total=False):
    """研究子图状态: 在 AgentState 基础上加研究专用字段。"""

    research_goal: str
    research_plan_obj: Any              # ResearchPlan
    research_round: int
    research_max_rounds: int
    stall_rounds: int
    evidence_hits: List[Any]           # 累计 Hit (按主键去重)
    evidence_seen_keys: List[str]
    research_gaps: List[str]
    research_covered: List[str]
    research_tried: List[str]
    research_complete: bool
    research_status: str               # complete | insufficient | clarify | error
    research_log: List[Dict[str, Any]]
    research_thinking: List[Dict[str, Any]]  # 规划/每轮评估的可读"思考过程", 供流式展示
    prev_clarify: Optional[Dict[str, str]]   # #9: 上一轮 clarify 的 {question, answer}, 供重规划纠偏
    gap_stall_rounds: int                    # 连续缺口未收敛的轮数
    research_gap_sig: str                     # 上一轮缺口签名 (facet 集合)
    direct_answer: str                        # 规划阶段 reject 的兜底直答 (不检索)
    last_policy_decision: Dict[str, Any]      # 上一轮 policy 决策摘要 (喂给下一轮 observation)
    skill_id: Optional[str]                    # skill_router 选中的 skill id; None=回退通用逻辑
    skill: Any                                 # 选中的 ResearchSkill 对象 (None=通用)
    prev_skill_id: Optional[str]               # 上一轮 skill (跨轮延续/分类器偏置)


# ---------------------------------------------------------------------------
# 证据累积 / 观测 / 综述上下文
# ---------------------------------------------------------------------------

def _harvest_round_hits(route_results: Dict[str, Any]) -> List[Hit]:
    """从本轮 route_results 取可用于综合的正文 hit (排除结构化/metadata)。"""
    by_route = collect_reflect_hits_by_route(route_results or {})
    hits: List[Hit] = []
    for route_hits in by_route.values():
        hits.extend(route_hits)
    return hits


def _accumulate_evidence(
    state: ResearchState, round_hits: List[Hit], *, quality_floor: Optional[float] = None,
) -> "tuple[int, int]":
    """把本轮 hit 去重并入 evidence_hits; 返回 (新增条数, 有质量新增条数)。

    #7: 有质量新增 = 新增且 rerank_score ≥ quality_floor 的条数 (无 rerank 分时按新增计),
    供 stall 判定避免"每轮多 1-2 条低分 hit 却一直 continue"的低效循环。
    """
    evidence: List[Hit] = list(state.get("evidence_hits") or [])
    seen: set = set(state.get("evidence_seen_keys") or [])
    added = 0
    quality_added = 0
    for h in round_hits:
        key = _lg._hit_dedupe_key(h)
        if not key or key in seen:
            continue
        seen.add(key)
        evidence.append(h)
        added += 1
        rs = getattr(h, "rerank_score", None)
        if quality_floor is None or rs is None or rs >= quality_floor:
            quality_added += 1
    state["evidence_hits"] = evidence
    state["evidence_seen_keys"] = list(seen)
    return added, quality_added


def _evidence_doc_groups(evidence: List[Hit]) -> "Dict[str, List[Hit]]":
    groups: Dict[str, List[Hit]] = {}
    for h in evidence:
        name = h.doc_name or h.doc_id or "未知文献"
        groups.setdefault(name, []).append(h)
    return groups


def _hit_sort_score(h: Hit) -> float:
    """单条证据的排序分: 优先 rerank_score, 回退 emb score。"""
    return h.rerank_score if h.rerank_score is not None else (h.score or 0.0)


def _doc_agg_score(hits: List[Hit]) -> float:
    """文献重要性 = 其 top-3 证据分之和 (兼顾峰值与数量)。"""
    s = sorted((_hit_sort_score(h) for h in hits), reverse=True)
    return sum(s[:3])


def _dedupe_doc_hits(hits: List[Hit]) -> List[Hit]:
    """#2: 同一文献内去掉近重复 chunk (按正文规整前缀比对), 保留分数更高者; 按分降序。"""
    ranked = sorted(hits, key=_hit_sort_score, reverse=True)
    out: List[Hit] = []
    seen: set = set()
    for h in ranked:
        body = re.sub(r"\s+", "", (h.content or ""))[:80]
        if body and body in seen:
            continue
        if body:
            seen.add(body)
        out.append(h)
    return out


def _normalize_kw(text: str) -> str:
    """#3: 关键词归一化, 让措辞等价的批次产生相同签名。

    小写化、去标点(保留中英数)、按词排序、去掉英文复数尾 s, 使
    'corrosion rate' / 'corrosion rates' / 'Rate, Corrosion' 归一为同一签名。
    """
    t = (text or "").lower()
    t = re.sub(r"[^\w\u4e00-\u9fff]+", " ", t)
    toks = []
    for tok in t.split():
        if len(tok) > 3 and tok.isascii() and tok.endswith("s"):
            tok = tok[:-1]
        toks.append(tok)
    toks.sort()
    return " ".join(toks)


def _gaps_signature(gaps: List[str]) -> str:
    """把本轮缺口归一为'facet 关键词集合'签名, 用于检测多轮缺口是否收敛。

    缺口多形如 'composition_system: 稀土配比缺失', 取冒号前的 facet 键 (无冒号则取全文),
    归一化后排序去重, 相邻轮签名相同 ⇒ 缺口未收敛 (语料可能根本没有该数据)。
    """
    norms = []
    for g in gaps or []:
        s = str(g)
        key = s.split(":", 1)[0].strip() if ":" in s else s
        n = _normalize_kw(key)
        if n:
            norms.append(n)
    return "|".join(sorted(set(norms)))


def _round_query_signatures(sub_decisions: List[Any]) -> List[str]:
    """从本轮子查询决策里抽出 '路径:归一化关键词' 签名, 用于告诉 policy 已检索过什么。"""
    sigs: List[str] = []
    for dec in sub_decisions or []:
        routes = getattr(dec, "routes", None) or []
        rewrites = getattr(dec, "rewrites", None) or {}
        for route in routes:
            raw = rewrites.get(route)
            if isinstance(raw, (list, tuple)):
                kw = " ".join(str(x) for x in raw).strip()
            else:
                kw = str(raw or "").strip()
            norm = _normalize_kw(kw)
            if norm:
                sigs.append(f"{route}:{norm}")
    return sigs


def _build_observation(
    state: ResearchState,
    *,
    plan: Optional[ResearchPlan],
    round_results: Dict[str, Any],
    new_added: int,
    reflect_cfg: ReflectSummaryConfig,
    summary_max_chars: int = 1800,
) -> str:
    """给 policy LLM 的压缩观测 (不灌全文)。"""
    evidence: List[Hit] = list(state.get("evidence_hits") or [])
    groups = _evidence_doc_groups(evidence)

    parts: List[str] = []

    # #1: 同时给出用户原始问题与规划改写目标, 让 policy 能发现规划是否偏离原意并纠偏
    user_q = (state.get("query") or "").strip()
    goal = (state.get("research_goal") or "").strip()
    if user_q:
        head = f"【用户原始问题】{user_q}"
        if goal and goal != user_q:
            head += (
                f"\n【规划改写目标】{goal}\n"
                "  (注意: 后续检索须服务于'用户原始问题'; 若改写目标遗漏了原问题的关键点"
                "如对比/差异/定量等, 请在 gaps 中指出并据原问题纠偏)"
            )
        parts.append(head)

    if plan and plan.facets:
        facet_lines = [f"  - {f.id}: {f.question}" for f in plan.facets]
        parts.append("【研究维度 facets】\n" + "\n".join(facet_lines))

    # 充分性目标 (plan.sufficiency): 显式告诉 policy 收口标准, 并对照当前进度
    suff = (plan.sufficiency if plan else None) or {}
    if suff:
        suff_lines: List[str] = []
        min_docs = suff.get("min_docs")
        if isinstance(min_docs, (int, float)) and min_docs:
            ok = "✓达标" if len(groups) >= int(min_docs) else "✗未达标"
            suff_lines.append(f"  - 目标文献数 ≥ {int(min_docs)} (当前 {len(groups)} 篇, {ok})")
        must_cover = [str(x) for x in (suff.get("must_cover") or []) if str(x).strip()]
        if must_cover:
            suff_lines.append("  - 必须覆盖维度: " + ", ".join(must_cover))
        if suff.get("need_conflict_check"):
            suff_lines.append("  - 需核对不同文献结论的共识/分歧")
        if suff.get("need_quantitative_data"):
            suff_lines.append("  - 需要定量数据 (速率/含量/电化学参数等)")
        if suff_lines:
            parts.append("【充分性目标 (满足后即可 finish)】\n" + "\n".join(suff_lines))

    doc_lines = [
        f"  - {name} ({len(hits)} 条证据)"
        for name, hits in list(groups.items())[:12]
    ]
    parts.append(
        f"【累计证据】共 {len(groups)} 篇文献 / {len(evidence)} 条证据片段\n"
        + ("\n".join(doc_lines) if doc_lines else "  (无)")
    )

    # 上一轮 policy 决策: 让本轮 policy 知道"刚才为什么这么检索", 避免反复下发同质批次
    prev = state.get("last_policy_decision") or {}
    if prev:
        prev_lines = [f"  - 上轮动作: {prev.get('action', '')}"]
        if prev.get("reason"):
            prev_lines.append(f"  - 上轮理由: {prev['reason']}")
        if prev.get("gaps"):
            prev_lines.append("  - 上轮判定缺口: " + "; ".join(prev["gaps"][:6]))
        if prev.get("next_kw"):
            prev_lines.append("  - 上轮下发关键词: " + ", ".join(prev["next_kw"][:12]))
        parts.append("【上一轮 policy 决策】\n" + "\n".join(prev_lines))

    # 本轮 rerank 命中 (按 rerank 分排序): 直接给出精排分, 让 policy 据质量判断进展
    round_hits = _harvest_round_hits(round_results or {})
    if round_hits:
        ranked = sorted(round_hits, key=_hit_sort_score, reverse=True)[:8]
        rr_lines = []
        for h in ranked:
            sc = h.rerank_score if h.rerank_score is not None else (h.score or 0.0)
            doc = h.doc_name or h.doc_id or "?"
            snip = re.sub(r"\s+", " ", (h.content or "")).strip()[:60]
            rr_lines.append(f"  - [rerank={sc:.3f}] {doc}: {snip}")
        parts.append(
            f"【本轮 rerank 命中 (共 {len(round_hits)} 条, 取前 {len(ranked)} 条按分)】\n"
            + "\n".join(rr_lines)
        )

    round_summary, _ = summarize_for_reflect(round_results or {}, config=reflect_cfg)
    # #8: 截断本轮摘要, 控制 policy prompt 体积 (累计证据/缺口/已检索词已另列)
    if summary_max_chars and len(round_summary) > summary_max_chars:
        round_summary = round_summary[:summary_max_chars].rstrip() + " …(摘要已截断)"
    parts.append(f"【本轮新增 {new_added} 条; 本轮检索摘要】\n{round_summary}")

    covered = state.get("research_covered") or []
    gaps = state.get("research_gaps") or []
    tried = state.get("research_tried") or []
    if covered:
        parts.append("【此前已判定覆盖的维度】" + ", ".join(covered))
    if gaps:
        parts.append("【此前缺口】" + "; ".join(gaps))
    if tried:
        parts.append(
            "【已检索过的 路径:关键词 (不要重复下发, 换角度补缺口)】\n"
            + "\n".join(f"  - {t}" for t in tried[:20])
        )

    return "\n\n".join(parts)


def _build_research_context(
    state: ResearchState,
    *,
    plan: Optional[ResearchPlan],
    snippet_chars: int = 500,
) -> str:
    """把累计证据渲染成综述生成用的 context (按文献分组 + 缺口/局限)。"""
    evidence: List[Hit] = list(state.get("evidence_hits") or [])
    groups = _evidence_doc_groups(evidence)
    user_q = (state.get("query") or "").strip()
    goal = state.get("research_goal") or user_q or ""

    # #1: 综述必须回答用户原始问题; 改写目标仅作辅助, 避免综述偏离原意
    lines: List[str] = [f"# 用户原始问题(综述须正面回答)\n{user_q or goal}"]
    if goal and goal != user_q:
        lines.append(f"# 规划研究目标(辅助)\n{goal}")

    if plan and plan.facets:
        lines.append(
            "# 研究维度\n"
            + "\n".join(f"- {f.id}: {f.question}" for f in plan.facets)
        )

    lines.append(f"# 证据材料 (共 {len(groups)} 篇文献 / {len(evidence)} 条片段, 按重要性排序)")
    # #5: 按文献累计分排序, 重要文献给更多证据条数预算 (8/5/3 分层), 替代统一 top-6
    ordered = sorted(groups.items(), key=lambda kv: _doc_agg_score(kv[1]), reverse=True)
    n_docs = len(ordered)
    top_cut = max(1, n_docs // 3)
    mid_cut = max(top_cut, (2 * n_docs) // 3)
    for i, (name, hits) in enumerate(ordered, 1):
        cap = 8 if i <= top_cut else (5 if i <= mid_cut else 3)
        chosen = _dedupe_doc_hits(hits)[:cap]  # #2: 先去近重复再取前 cap 条
        lines.append(f"\n## [{i}] {name}")
        for h in chosen:
            snippet = _lg_snippet(h.content, snippet_chars)
            score = ""
            if h.rerank_score is not None:
                score = f" (rerank={h.rerank_score:.3f})"
            elif h.score:
                score = f" (emb={h.score:.3f})"
            page = (h.page_start + 1) if isinstance(h.page_start, int) else "?"
            lines.append(f"- [page {page}{score}] {snippet}")

    gaps = state.get("research_gaps") or []
    if gaps:
        lines.append("\n# 已识别的证据缺口 / 局限\n" + "\n".join(f"- {g}" for g in gaps))

    return "\n".join(lines)


def _lg_snippet(text: str, max_chars: int) -> str:
    cleaned = (text or "").replace("\n", " ").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars] + " …"


# ---------------------------------------------------------------------------
# 节点
# ---------------------------------------------------------------------------

def _make_skill_router_node(
    skill_llm: Optional[LLMClient],
    skills: Dict[str, ResearchSkill],
    *,
    mode: str,
    router_max_tokens: int,
    disable_thinking: bool,
    min_confidence: float,
    default_max_rounds: int,
) -> callable:
    """skill_router 节点: 用"思考模型"判断用户发话属于哪类任务; 仅在【强匹配】时启用对应
    skill, 否则 skill=None → 下游 plan/policy/synthesis 回退通用逻辑。

    无论是否命中, 都向用户流式产出一条"任务识别"思考, 展示判定思路。
    """
    def skill_router_node(state: ResearchState) -> ResearchState:
        t0 = time.time()
        cid = state.get("correlation_id", "?")
        query = state["query"]
        sel = select_skill(
            skill_llm, query, skills,
            mode=mode, prev_skill_id=state.get("prev_skill_id"),
            router_max_tokens=router_max_tokens, disable_thinking=disable_thinking,
            min_confidence=min_confidence,
            correlation_id=cid,
        )
        skill = skills.get(sel.skill_id) if sel.skill_id else None
        state["skill_id"] = sel.skill_id
        state["skill"] = skill
        # 轮次预算: 命中且 skill 自带 max_rounds → 用 skill 的; 否则 (未命中/skill 未设) →
        # 回到配置的全局默认, 而非沿用初始 state 里"跨 skill 最大值"(否则通用查询会被
        # 最激进 skill 的轮次预算污染, 多跑无谓轮次)。
        if skill is not None and skill.max_rounds:
            state["research_max_rounds"] = int(skill.max_rounds)
        else:
            state["research_max_rounds"] = int(default_max_rounds)
        think = (sel.thinking or sel.reason or "").strip()
        if skill is not None:
            _append_thinking(
                state, round_idx=0, phase="skill",
                content=(
                    f"🧭 任务识别：{think}\n"
                    f"→ 采用「{skill.name}」技能（匹配置信度 {sel.confidence:.0%}），"
                    "将按该任务方式拆解问题并逐轮引导检索。"
                ),
            )
            logger.info(
                f"[{cid}] [skill_router] → {sel.skill_id} "
                f"(conf={sel.confidence:.2f}, {sel.reason})"
            )
        else:
            _append_thinking(
                state, round_idx=0, phase="skill",
                content=(
                    f"🧭 任务识别：{think}\n"
                    "→ 未强匹配到专门技能，按通用研究方式处理。"
                ),
            )
            logger.info(
                f"[{cid}] [skill_router] 未强匹配 ({sel.reason}), 回退通用逻辑"
            )
        state["agent_phase"] = "skill_router"
        state["next_action"] = "plan"
        state.setdefault("node_timings", {})["skill_router"] = time.time() - t0
        return state

    return skill_router_node


def _make_plan_node(
    planner_llm: LLMClient,
    *,
    validate_fn: Optional[callable],
    limits: RoutingLimits,
    max_batches: int,
    max_tokens: int,
    disable_thinking: Optional[bool],
) -> callable:
    def plan_node(state: ResearchState) -> ResearchState:
        t0 = time.time()
        cid = state.get("correlation_id", "?")
        query = state["query"]
        doc_registry = state.get("last_round_docs") or []

        # #6: 构建跨轮研究状态提示，让规划 LLM 在已有基础上继续而非从零开始
        carryover_hint = ""
        prev_gaps = state.get("research_gaps") or []
        prev_covered = state.get("research_covered") or []
        prev_tried = state.get("research_tried") or []
        if prev_gaps or prev_covered:
            hint_parts: List[str] = []
            if prev_covered:
                hint_parts.append(f"上一轮已覆盖的维度: {', '.join(prev_covered[:12])}")
            if prev_gaps:
                hint_parts.append(f"上一轮的缺口: {', '.join(prev_gaps[:8])}")
            if prev_tried:
                hint_parts.append(f"已检索过的关键词签名 (不要重复): {', '.join(prev_tried[:10])}")
            carryover_hint = (
                "\n\n【上一轮研究状态 (请在此基础上继续, 不要重复已覆盖的维度)】\n"
                + "\n".join(hint_parts)
            )

        # skill 专属规划提示词与默认收口标准 (无 skill 时为 None → plan_research 用通用逻辑)
        skill: Optional[ResearchSkill] = state.get("skill")
        plan_system = (
            compose_plan_system(skill.plan_system, skill.prefer_first_paths) if skill else None
        )
        default_suff = skill.default_sufficiency if skill else None
        eff_max_batches = (
            int(skill.max_batches) if skill and skill.max_batches else max_batches
        )

        outcome = plan_research(
            planner_llm, query,
            doc_registry=doc_registry,
            history=state.get("history"),
            max_batches=eff_max_batches,
            limits=limits,
            max_tokens=max_tokens,
            disable_thinking=disable_thinking,
            correlation_id=cid,
            prev_clarify=state.get("prev_clarify"),  # #9: 注入上一轮反问, 避免重复纠偏方向
            carryover_hint=carryover_hint,             # #6: 注入跨轮研究状态
            system=plan_system,                        # skill 专属规划提示词
            default_sufficiency=default_suff,          # skill 默认收口标准
        )

        # 规划前置过滤: 无关/闲聊 → 兜底直答; 模糊/过宽 → 追问。两者都不检索, 直接结束。
        if outcome is not None and outcome.action == "reject":
            state["research_status"] = "reject"
            state["direct_answer"] = outcome.reply
            state["agent_phase"] = "research_plan"
            state["next_action"] = "clarify"   # 复用 plan→END 边
            # 注: 不重置 research_thinking, 保留 skill_router 已产出的"任务识别"思考
            _append_thinking(
                state, round_idx=0, phase="plan",
                content=f"这条输入不适合进入文献检索（{outcome.reject_kind or '无关/闲聊'}），直接回应。",
            )
            state.setdefault("node_timings", {})["plan"] = time.time() - t0
            logger.info(f"[{cid}] [research.plan] reject ({outcome.reject_kind}), 不检索直答")
            return state
        if outcome is not None and outcome.action == "clarify":
            state["needs_clarify"] = True
            state["research_status"] = "clarify"
            state["clarify_request"] = {"q": outcome.clarify_q, "opts": outcome.clarify_opts}
            state["agent_phase"] = "research_plan"
            state["next_action"] = "clarify"
            # 注: 不重置 research_thinking, 保留 skill_router 已产出的"任务识别"思考
            _append_thinking(
                state, round_idx=0, phase="plan",
                content=f"问题还不够具体，先向用户澄清：{outcome.clarify_q}",
            )
            state.setdefault("node_timings", {})["plan"] = time.time() - t0
            logger.info(f"[{cid}] [research.plan] clarify (规划前置追问)")
            return state

        plan = outcome.plan if outcome is not None else None
        if plan is None:
            # 启发式兜底: 单 progressive 批次, 保证闭环不中断
            logger.info(f"[{cid}] [research.plan] 规划失败, 用启发式单批次兜底")
            plan = ResearchPlan(
                goal=query,
                facets=[],
                initial_batches=[{
                    "id": "b1", "purpose": "fallback",
                    "paths": [{"t": "progressive", "kw": [query]}],
                }],
            )

        multi = batches_to_multi_decision(
            plan.initial_batches,
            query=query,
            doc_registry=doc_registry,
            validate_fn=validate_fn,
            limits=limits,
        )
        if multi is None or not multi.subqueries:
            state["needs_clarify"] = True
            state["research_status"] = "clarify"
            state["clarify_request"] = {
                "q": "我没能把这个研究问题拆成可检索的子问题，能否补充一下具体的研究方向或关注点？",
                "opts": [],
            }
            state["next_action"] = "clarify"
            state.setdefault("node_timings", {})["plan"] = time.time() - t0
            return state

        sub_decs = [s.decision for s in multi.subqueries]
        state["research_goal"] = plan.goal
        state["research_plan_obj"] = plan
        state["research_round"] = 1
        state["multi_decision"] = multi
        state["subquery_decisions"] = sub_decs
        state["decision"] = _lg._merge_route_decisions(sub_decs)
        state["synth_hint"] = ""
        state["route_results"] = {}          # 首轮从空开始
        state["evidence_hits"] = []
        state["evidence_seen_keys"] = []
        state["research_gaps"] = []
        state["research_covered"] = []
        state["stall_rounds"] = 0
        state["gap_stall_rounds"] = 0
        state["research_gap_sig"] = ""
        state["research_log"] = [{
            "round": 0, "action": "plan",
            "facets": plan.facet_ids(), "batches": len(plan.initial_batches),
        }]
        # 注: 不重置 research_thinking, 保留 skill_router 已产出的"任务识别"思考,
        # 否则流式 diff 会因长度不变而漏掉本条 plan 思考
        _append_thinking(
            state, round_idx=0, phase="plan", content=_plan_thinking_text(plan),
        )
        state["agent_phase"] = "research_plan"
        state["next_action"] = "retrieve"
        state.setdefault("node_timings", {})["plan"] = time.time() - t0
        logger.info(
            f"[{cid}] [research.plan] 进入检索: facets={plan.facet_ids()} "
            f"首轮批次={len(sub_decs)}"
        )
        return state

    return plan_node


def _make_research_policy_node(
    policy_llm: LLMClient,
    *,
    validate_fn: Optional[callable],
    limits: RoutingLimits,
    max_batches: int,
    max_rounds: int,
    max_tokens: int,
    disable_thinking: Optional[bool],
    reflect_cfg: ReflectSummaryConfig,
    stall_limit: int = 2,
    snippet_chars: int = 500,
    stall_quality_floor: Optional[float] = None,
    obs_summary_max_chars: int = 1800,
    gap_stall_limit: int = 2,
) -> callable:
    def research_policy_node(state: ResearchState) -> ResearchState:
        t0 = time.time()
        cid = state.get("correlation_id", "?")
        query = state["query"]
        plan: Optional[ResearchPlan] = state.get("research_plan_obj")
        round_idx = int(state.get("research_round") or 1)
        route_results = state.get("route_results") or {}

        # skill 专属策略提示词与调参 (无 skill 时回退闭包默认 / 通用判断段)
        skill: Optional[ResearchSkill] = state.get("skill")
        policy_system = compose_policy_system(skill.policy_system) if skill else None
        eff_max_batches = (
            int(skill.max_batches) if skill and skill.max_batches else max_batches
        )
        eff_max_rounds = int(state.get("research_max_rounds") or max_rounds)
        eff_gap_stall_limit = (
            int(skill.gap_stall_limit)
            if skill and skill.gap_stall_limit is not None else gap_stall_limit
        )
        eff_quality_floor = (
            float(skill.stall_quality_floor)
            if skill and skill.stall_quality_floor is not None else stall_quality_floor
        )

        # 0) 记录本轮实际检索过的关键词 (供 policy 避免重复下发, 提升检索效率)
        state["research_tried"] = _merge_unique(
            state.get("research_tried") or [],
            _round_query_signatures(state.get("subquery_decisions") or []),
        )

        # 1) 累积本轮证据 (#7: stall 按"有质量新增"判定, 低分 hit 不算进步)
        round_hits = _harvest_round_hits(route_results)
        new_added, quality_added = _accumulate_evidence(
            state, round_hits, quality_floor=eff_quality_floor,
        )
        stall = int(state.get("stall_rounds") or 0)
        if quality_added == 0:
            stall += 1
        else:
            stall = 0
        state["stall_rounds"] = stall

        evidence: List[Hit] = list(state.get("evidence_hits") or [])
        doc_count = len(_evidence_doc_groups(evidence))

        # 2) 硬熔断 / 停滞处理: 区分"有证据可收口"与"停滞无进展应反问"
        #    - 连续无有质量新增 且 证据不足 → clarify 让用户补充/缩小范围 (用户诉求);
        #    - 完全无证据 → clarify; 有足够证据但停滞/到预算 → finish。
        forced_finish_reason = ""
        clarify_reason = ""
        if not evidence and round_idx >= 2:
            clarify_reason = "多轮检索仍无任何证据"
        elif stall >= stall_limit:
            if doc_count < 3:
                clarify_reason = f"连续 {stall} 轮无有质量新增证据且证据不足"
            else:
                forced_finish_reason = f"连续 {stall} 轮无有质量新增证据"
        elif round_idx >= eff_max_rounds:
            if doc_count == 0:
                clarify_reason = f"达到最大轮次 {eff_max_rounds} 仍无证据"
            else:
                forced_finish_reason = f"达到最大轮次 {eff_max_rounds}"

        if clarify_reason:
            return _clarify_exit(
                state, plan=plan, cid=cid, t0=t0, round_idx=round_idx,
                reason=clarify_reason,
            )
        if forced_finish_reason:
            return _finish(
                state, plan=plan, cid=cid, t0=t0, round_idx=round_idx,
                reason=forced_finish_reason, new_added=new_added,
                snippet_chars=snippet_chars,
            )

        # 3) 调 policy LLM
        observation = _build_observation(
            state, plan=plan, round_results=route_results,
            new_added=new_added, reflect_cfg=reflect_cfg,
            summary_max_chars=obs_summary_max_chars,
        )
        # skill 守卫: 把"充分性未满足项"作为额外观测注入, 引导 policy 优先补齐再收口
        if skill and skill.guards:
            unmet = evaluate_guards(
                skill, doc_count=doc_count,
                evidence_texts=[(h.content or "") for h in evidence],
                facet_ids=plan.facet_ids() if plan else [],
                covered=state.get("research_covered") or [],
            )
            if unmet:
                observation += (
                    "\n\n【充分性守卫未满足 (建议优先补齐后再收口)】\n"
                    + "\n".join(f"  - {u}" for u in unmet)
                )
        rounds_left = max(0, eff_max_rounds - round_idx)
        decision = decide_policy(
            policy_llm,
            goal=state.get("research_goal") or query,
            observation=observation,
            round_idx=round_idx,
            rounds_left=rounds_left,
            max_batches=eff_max_batches,
            limits=limits,
            max_tokens=max_tokens,
            disable_thinking=disable_thinking,
            correlation_id=cid,
            system=policy_system,
        )

        # 合并覆盖度/缺口 (累积, 去重保序)
        state["research_covered"] = _merge_unique(
            state.get("research_covered") or [], decision.covered,
        )
        if decision.gaps:
            state["research_gaps"] = decision.gaps  # 缺口用最新一轮判断 (反映当前状态)

        log = list(state.get("research_log") or [])
        log.append({
            "round": round_idx, "action": decision.action,
            "new_evidence": new_added, "doc_count": doc_count,
            "gaps": decision.gaps, "reason": decision.reason,
        })
        state["research_log"] = log

        if decision.action == "clarify":
            _append_thinking(
                state, round_idx=round_idx, phase="policy",
                content=_policy_thinking_text(round_idx, decision),
            )
            state["needs_clarify"] = True
            state["research_status"] = "clarify"
            state["clarify_request"] = {
                "q": decision.clarify_q,
                "opts": decision.clarify_opts,
            }
            state["next_action"] = "end"
            state["agent_phase"] = "research_policy"
            state.setdefault("node_timings", {})["research_policy"] = (
                state.get("node_timings", {}).get("research_policy", 0.0) + (time.time() - t0)
            )
            logger.info(f"[{cid}] [research.policy] round={round_idx} → clarify")
            return state

        if decision.action == "continue":
            multi = batches_to_multi_decision(
                decision.next_batches,
                query=query,
                doc_registry=state.get("last_round_docs") or [],
                validate_fn=validate_fn,
                limits=limits,
            )
            if multi is None or not multi.subqueries:
                return _finish(
                    state, plan=plan, cid=cid, t0=t0, round_idx=round_idx,
                    reason="continue 批次无效, 收口", new_added=new_added,
                    snippet_chars=snippet_chars,
                )
            sub_decs = [s.decision for s in multi.subqueries]

            # 规则去重: 若下一轮批次的 路径:关键词 全部已检索过, 再检索也是空耗 → 收口
            next_sigs = _round_query_signatures(sub_decs)
            tried_set = set(state.get("research_tried") or [])
            if next_sigs and all(s in tried_set for s in next_sigs):
                return _finish(
                    state, plan=plan, cid=cid, t0=t0, round_idx=round_idx,
                    reason="下一轮批次与已检索完全重复, 收口避免空耗",
                    new_added=new_added, snippet_chars=snippet_chars,
                )

            # 缺口收敛检测: policy 连续多轮 continue 但缺口(facet)始终不变, 说明它在追
            # 语料里很可能不存在的数据 → 已有证据时尽早收口, 避免烧满轮次预算空转。
            gap_sig = _gaps_signature(decision.gaps)
            prev_gap_sig = state.get("research_gap_sig") or ""
            gap_stall = int(state.get("gap_stall_rounds") or 0)
            if gap_sig and gap_sig == prev_gap_sig:
                gap_stall += 1
            else:
                gap_stall = 0
            state["research_gap_sig"] = gap_sig
            state["gap_stall_rounds"] = gap_stall
            if gap_stall >= eff_gap_stall_limit and doc_count >= 3:
                return _finish(
                    state, plan=plan, cid=cid, t0=t0, round_idx=round_idx,
                    reason=f"连续 {gap_stall + 1} 轮缺口未收敛, 已尽可能检索, 收口",
                    new_added=new_added, snippet_chars=snippet_chars,
                )
            # 记录本轮决策摘要, 供下一轮 observation 回看 (避免反复下发同质批次)
            next_kw: List[str] = []
            for s in sub_decs:
                for kws in (getattr(s, "rewrites", {}) or {}).values():
                    if isinstance(kws, str):
                        next_kw.append(kws)
                    elif isinstance(kws, (list, tuple)):
                        next_kw.extend(str(x) for x in kws)
            state["last_policy_decision"] = {
                "action": "continue",
                "reason": decision.reason,
                "gaps": list(decision.gaps or []),
                "next_kw": next_kw[:12],
            }
            _append_thinking(
                state, round_idx=round_idx, phase="policy",
                content=_policy_thinking_text(round_idx, decision),
            )
            state["multi_decision"] = multi
            state["subquery_decisions"] = sub_decs
            state["decision"] = _lg._merge_route_decisions(sub_decs)
            state["route_results"] = {}              # 轮间清空: 下一轮只 rerank 新批次
            state.pop("route_results_pre_rerank", None)
            state["research_round"] = round_idx + 1
            state["agent_phase"] = "research_policy"
            state["next_action"] = "retrieve"
            state.setdefault("node_timings", {})["research_policy"] = (
                state.get("node_timings", {}).get("research_policy", 0.0) + (time.time() - t0)
            )
            logger.info(
                f"[{cid}] [research.policy] round={round_idx} → continue "
                f"(next batches={len(sub_decs)}, gaps={decision.gaps})"
            )
            return state

        # finish
        return _finish(
            state, plan=plan, cid=cid, t0=t0, round_idx=round_idx,
            reason=decision.reason or "证据充分", new_added=new_added,
            snippet_chars=snippet_chars,
        )

    return research_policy_node


def _finish(
    state: ResearchState, *, plan: Optional[ResearchPlan], cid: str, t0: float,
    round_idx: int, reason: str, new_added: int, snippet_chars: int = 500,
) -> ResearchState:
    context = _build_research_context(state, plan=plan, snippet_chars=snippet_chars)
    state["context"] = context
    state["research_complete"] = True
    evidence = state.get("evidence_hits") or []
    if evidence:
        _append_thinking(
            state, round_idx=round_idx, phase="finish",
            content=f"🔍 共完成 {round_idx} 轮检索，证据已足够，正在综合生成最终结果…",
        )
    else:
        _append_thinking(
            state, round_idx=round_idx, phase="finish",
            content="当前文献库未检索到足够支撑该研究问题的证据。",
        )
    # 语义区分: 有证据=正常完成; 多轮仍无证据=无法完成 (insufficient), 不是"研究成功"
    state["research_status"] = "complete" if evidence else "insufficient"
    state["agent_phase"] = "research_policy"
    state["next_action"] = "end"
    log = list(state.get("research_log") or [])
    log.append({"round": round_idx, "action": "finish", "reason": reason})
    state["research_log"] = log
    state.setdefault("node_timings", {})["research_policy"] = (
        state.get("node_timings", {}).get("research_policy", 0.0) + (time.time() - t0)
    )
    logger.info(
        f"[{cid}] [research.policy] round={round_idx} → finish ({reason}); "
        f"status={state['research_status']} evidence={len(evidence)} 条, "
        f"context_len={len(context)}"
    )
    return state


def _clarify_exit(
    state: ResearchState, *, plan: Optional[ResearchPlan], cid: str, t0: float,
    round_idx: int, reason: str,
) -> ResearchState:
    """检索陷入停滞 (连续多轮无有质量新增) 且证据不足 → 反问用户补充/缩小范围, 而非硬收口。"""
    gaps = state.get("research_gaps") or []
    evidence = state.get("evidence_hits") or []
    doc_count = len(_evidence_doc_groups(evidence))
    gap_txt = "、".join(g.split(":", 1)[0].strip() if ":" in g else g for g in gaps[:4])
    if doc_count > 0:
        q = (
            f"我已检索到 {doc_count} 篇相关文献，但在"
            + (f"「{gap_txt}」等方面" if gap_txt else "关键维度上")
            + "连续多轮没有检索到新的有效证据。你能否补充更具体的方向、关键术语，或指定要关注的文献？"
        )
    else:
        q = (
            "我连续检索了几轮，都没能在当前文献库中找到与这个问题直接相关的证据。"
            "可能是范围太宽或用词与库内文献不一致——你能否换个更具体的说法、补充对象/材料/方法等关键限定？"
        )
    state["needs_clarify"] = True
    state["research_status"] = "clarify"
    state["clarify_request"] = {"q": q, "opts": [str(g) for g in gaps[:4]]}
    state["next_action"] = "end"
    state["agent_phase"] = "research_policy"
    _append_thinking(
        state, round_idx=round_idx, phase="policy",
        content=f"🔍 第 {round_idx} 轮检索后进展停滞，正在向你确认以缩小范围…",
    )
    log = list(state.get("research_log") or [])
    log.append({"round": round_idx, "action": "clarify", "reason": reason})
    state["research_log"] = log
    state.setdefault("node_timings", {})["research_policy"] = (
        state.get("node_timings", {}).get("research_policy", 0.0) + (time.time() - t0)
    )
    logger.info(
        f"[{cid}] [research.policy] round={round_idx} → clarify (停滞反问: {reason})"
    )
    return state


def _merge_unique(base: List[str], extra: List[str]) -> List[str]:
    out = list(base)
    seen = set(base)
    for x in extra or []:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _append_thinking(
    state: ResearchState, *, round_idx: int, phase: str, content: str,
) -> None:
    """累积一条研究"思考过程" (规划/每轮 policy 的可读叙述), 供 run_events 流式产出。

    与 progress (机械检索进度) 不同, 这里给用户看的是"模型在想什么": 怎么拆解目标、
    每轮评估了什么、还缺什么、为什么继续或收口。
    """
    text = (content or "").strip()
    if not text:
        return
    log = list(state.get("research_thinking") or [])
    log.append({"round": int(round_idx), "phase": phase, "content": text})
    state["research_thinking"] = log


def _plan_thinking_text(plan: ResearchPlan) -> str:
    """把研究计划渲染成一段"规划思考"叙述。"""
    lines: List[str] = []
    for f in (plan.facets or [])[:6]:
        q = (f.question or "").strip() or "、".join((f.keywords or [])[:3])
        if q:
            lines.append(f"- {q}")
    body = "\n".join(lines)
    n = len(plan.facets or [])
    return (
        f"🎯 研究目标：{plan.goal}\n"
        f"我先把它拆成 {n} 个相互独立的方向逐一求证：\n"
        f"{body}\n"
        "先并行检索这些方向，拿到初步证据后再判断哪里还不够、要不要继续补检。"
    )


def _policy_thinking_text(round_idx: int, decision: PolicyDecision) -> str:
    """把一轮 policy 决策渲染成一条"检索进度"提示 (不展开内部评估/缺口推理)。"""
    if decision.action == "continue":
        return f"🔍 第 {round_idx} 轮检索完成，正在进行第 {round_idx + 1} 轮检索…"
    if decision.action == "clarify":
        return f"🔍 第 {round_idx} 轮检索完成，需要向你确认问题方向…"
    return f"🔍 共完成 {round_idx} 轮检索，正在综合生成最终结果…"


# ---------------------------------------------------------------------------
# 图构建
# ---------------------------------------------------------------------------

def _after_plan(state: ResearchState) -> str:
    # 规划前置过滤 (clarify 追问 / reject 兜底) 都走 plan→END 边, 不进检索。
    if state.get("needs_clarify") or state.get("next_action") == "clarify":
        return "clarify"
    return "retrieve"


def _after_research_policy(state: ResearchState) -> str:
    return "retrieve" if state.get("next_action") == "retrieve" else "end"


def build_research_graph(
    pipeline: AgenticRAGPipeline,
    *,
    planner_llm: LLMClient,
    policy_llm: LLMClient,
    reranker_client: Optional[RerankerClient] = None,
    limits: Optional[RoutingLimits] = None,
    max_batches: int = 3,
    max_rounds: int = 4,
    stall_limit: int = 2,
    planner_max_tokens: int = 2048,
    policy_max_tokens: int = 2048,
    planner_disable_thinking: Optional[bool] = None,
    policy_disable_thinking: Optional[bool] = None,
    reranker_top_k: int = 5,
    reranker_quality_k: int = 3,
    reranker_quality_threshold: float = 0.5,
    reranker_quality_threshold_by_type: Optional[Dict[str, float]] = None,
    reranker_route_thresholds: Optional[Any] = None,
    reranker_diagnosis_config: Optional[Any] = None,
    fail_open_min_emb_quality: Optional[float] = None,
    max_workers: int = 4,
    summary_top_docs: int = 5,
    summary_per_query_k: int = 5,
    synthesis_snippet_chars: int = 500,
    reflect_summary_config: Optional[ReflectSummaryConfig] = None,
    stall_quality_floor: Optional[float] = None,
    obs_summary_max_chars: int = 1800,
    gap_stall_limit: int = 2,
    skills: Optional[Dict[str, ResearchSkill]] = None,
    skill_router_llm: Optional[LLMClient] = None,
    skill_router_mode: str = "llm",
    skill_router_max_tokens: int = 512,
    skill_router_disable_thinking: bool = False,
    skill_router_min_confidence: float = 0.6,
) -> Any:
    """构建专业研究模式的 LangGraph (CompiledStateGraph)。复用现有节点工厂。

    skills 非空时在 plan 前加 skill_router 节点 (据用户发话选 skill); 选不到 skill
    时下游 plan/policy 自动回退通用提示词, 与未引入 skill 前行为一致。
    """
    if StateGraph is None:
        raise ImportError("langgraph 未安装, 请运行: pip install langgraph")

    lim = limits or DEFAULT_ROUTING_LIMITS
    reflect_cfg = reflect_summary_config or ReflectSummaryConfig()
    validate_fn = getattr(pipeline.router, "_validate_decision", None)
    use_reranker = reranker_client is not None

    # 复用现有 retrieve / reranker 节点工厂 (零修改)
    from .neighbor_expansion import NeighborExpander
    neighbor_expander = NeighborExpander(
        client=pipeline.metadata_r.client,
        collection=pipeline.metadata_r.collection,
        vector_retriever=getattr(pipeline.local_r, "vec", None)
        or getattr(pipeline.summary_r, "vec", None),
    )
    retrieve_node = _lg._make_retrieve_node(
        pipeline.summary_r, pipeline.local_r, pipeline.metadata_r, max_workers,
        neighbor_expander=neighbor_expander,
        summary_top_docs=summary_top_docs,
        summary_per_query_k=summary_per_query_k,
    )
    plan_node = _make_plan_node(
        planner_llm,
        validate_fn=validate_fn, limits=lim, max_batches=max_batches,
        max_tokens=planner_max_tokens, disable_thinking=planner_disable_thinking,
    )
    research_policy_node = _make_research_policy_node(
        policy_llm,
        validate_fn=validate_fn, limits=lim, max_batches=max_batches,
        max_rounds=max_rounds, max_tokens=policy_max_tokens,
        disable_thinking=policy_disable_thinking, reflect_cfg=reflect_cfg,
        stall_limit=stall_limit, snippet_chars=synthesis_snippet_chars,
        stall_quality_floor=(
            stall_quality_floor if stall_quality_floor is not None
            else reranker_quality_threshold
        ),
        obs_summary_max_chars=obs_summary_max_chars,
        gap_stall_limit=gap_stall_limit,
    )

    use_skills = bool(skills)
    graph = StateGraph(ResearchState)
    if use_skills:
        skill_router_node = _make_skill_router_node(
            skill_router_llm, skills,
            mode=skill_router_mode,
            router_max_tokens=skill_router_max_tokens,
            disable_thinking=skill_router_disable_thinking,
            min_confidence=skill_router_min_confidence,
            default_max_rounds=max_rounds,
        )
        graph.add_node("skill_router", skill_router_node)
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("research_policy", research_policy_node)
    if use_reranker:
        reranker_node = _lg._make_reranker_node(
            reranker_client,
            top_k=reranker_top_k,
            quality_k=reranker_quality_k,
            quality_threshold=reranker_quality_threshold,
            quality_threshold_by_type=reranker_quality_threshold_by_type,
            route_thresholds=reranker_route_thresholds,
            diagnosis_config=reranker_diagnosis_config,
            fail_open_min_emb_quality=fail_open_min_emb_quality,
        )
        graph.add_node("reranker", reranker_node)

    if use_skills:
        graph.set_entry_point("skill_router")
        graph.add_edge("skill_router", "plan")
    else:
        graph.set_entry_point("plan")
    graph.add_conditional_edges("plan", _after_plan, {
        "retrieve": "retrieve",
        "clarify": END,
    })
    graph.add_edge("retrieve", "reranker" if use_reranker else "research_policy")
    if use_reranker:
        graph.add_edge("reranker", "research_policy")
    graph.add_conditional_edges("research_policy", _after_research_policy, {
        "retrieve": "retrieve",
        "end": END,
    })
    return graph.compile()


# ---------------------------------------------------------------------------
# 高层智能体
# ---------------------------------------------------------------------------

class ResearchAgent:
    """专业研究模式智能体: run() 完成多轮检索闭环, 产出综述用 context。

    生成 (综述写作) 由上层 flow 负责, 与普通模式保持一致的解耦。
    """

    def __init__(
        self,
        compiled_graph: Any,
        *,
        max_rounds: int = 4,
        recursion_limit: Optional[int] = None,
    ) -> None:
        self.graph = compiled_graph
        self.max_rounds = max_rounds
        # plan(1) + 每轮(retrieve+reranker+policy≈3) + 余量
        self.recursion_limit = recursion_limit or (max_rounds * 4 + 12)

    def _build_initial_state(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]],
        session_meta: Optional[Dict[str, Any]],
    ) -> ResearchState:
        last_round_docs: List[Dict[str, str]] = []
        prev_clarify: Optional[Dict[str, str]] = None
        # #6/#8/#10: 从 carryover 恢复跨轮研究状态
        carryover_gaps: List[str] = []
        carryover_covered: List[str] = []
        carryover_tried: List[str] = []
        carryover_seen_keys: List[str] = []
        prev_skill_id: Optional[str] = None
        if session_meta:
            raw = session_meta.get("doc_registry") or []
            if isinstance(raw, list):
                for entry in raw:
                    if isinstance(entry, dict) and entry.get("doc_id"):
                        last_round_docs.append({
                            "doc_id": str(entry["doc_id"]),
                            "doc_name": str(entry.get("doc_name") or entry["doc_id"]),
                            "pinned": bool(entry.get("pinned", False)),
                        })
            # #9: 上一轮若曾向用户反问 (clarify), 把"反问 + 本次回答"带给 plan, 避免重复纠偏
            cp = session_meta.get("clarify_pending") or session_meta.get("prev_clarify")
            if isinstance(cp, dict):
                cq = (cp.get("question") or cp.get("q") or "").strip()
                if cq:
                    prev_clarify = {"question": cq, "answer": query}

            # #6/#8/#10: 恢复上一轮研究状态
            co = session_meta.get("research_carryover")
            if isinstance(co, dict):
                prev_sid = str(co.get("skill_id") or "").strip()
                if prev_sid:
                    prev_skill_id = prev_sid
                carryover_gaps = [str(g) for g in (co.get("gaps") or []) if str(g).strip()]
                carryover_covered = [str(c) for c in (co.get("covered") or []) if str(c).strip()]
                carryover_tried = [str(t) for t in (co.get("tried") or []) if str(t).strip()]
                carryover_seen_keys = [str(k) for k in (co.get("evidence_seen_keys") or []) if str(k).strip()]
                # #10: carryover 中的 evidence_doc_ids 合并入 last_round_docs (pinned=True)
                existing_ids = {d["doc_id"] for d in last_round_docs if d.get("doc_id")}
                for entry in co.get("evidence_doc_ids") or []:
                    if isinstance(entry, dict) and entry.get("doc_id"):
                        did = str(entry["doc_id"])
                        if did not in existing_ids:
                            existing_ids.add(did)
                            last_round_docs.append({
                                "doc_id": did,
                                "doc_name": str(entry.get("doc_name") or did),
                                "pinned": True,  # 跨轮继承的文献 pin 住，防止被淘汰
                            })
                if carryover_gaps or carryover_covered or carryover_seen_keys:
                    logger.info(
                        f"[research] 恢复 carryover: "
                        f"gaps={len(carryover_gaps)} covered={len(carryover_covered)} "
                        f"tried={len(carryover_tried)} seen_keys={len(carryover_seen_keys)} "
                        f"docs_merged={len(co.get('evidence_doc_ids') or [])}"
                    )

        return {
            "query": query,
            "history": history,
            "prev_clarify": prev_clarify,
            "last_round_docs": last_round_docs,
            "this_round_docs": [],
            "retry_count": 0,
            "max_retries": 0,
            "agent_phase": "start",
            "research_round": 0,
            "research_max_rounds": self.max_rounds,
            "evidence_hits": [],
            "evidence_seen_keys": carryover_seen_keys,  # #8: 跨轮去重
            "research_gaps": carryover_gaps,              # #6: 跨轮缺口继承
            "research_covered": carryover_covered,        # #6: 跨轮覆盖继承
            "research_tried": carryover_tried,            # #8: 跨轮已检索签名继承
            "research_thinking": [],
            "correlation_id": uuid.uuid4().hex[:8],
            "route_results": {},
            "route_errors": {},
            "node_timings": {},
            "prev_skill_id": prev_skill_id,   # 跨轮 skill 延续/分类器偏置
            "skill_id": None,
            "skill": None,
        }

    def _finalize(
        self, query: str, final_state: ResearchState, t0: float,
        last_round_docs: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        cid = final_state.get("correlation_id", "?")
        this_round_docs = final_state.get("this_round_docs", []) or []
        persisted_registry = _lg._persist_doc_registry(
            last_round_docs, this_round_docs,
            final_state.get("decision"),
            subquery_decisions=final_state.get("subquery_decisions") or [],
        )

        needs_clarify = bool(final_state.get("needs_clarify"))
        direct_answer = str(final_state.get("direct_answer") or "").strip()
        evidence = final_state.get("evidence_hits") or []
        doc_count = len(_evidence_doc_groups(evidence)) if evidence else 0
        context = final_state.get("context", "") or ""
        no_answer = (not needs_clarify) and (not direct_answer) and len(evidence) == 0

        # 研究状态: 优先用节点显式写入的, 否则按是否有证据兜底推断
        status = final_state.get("research_status")
        if not status:
            if needs_clarify:
                status = "clarify"
            elif no_answer:
                status = "insufficient"
            else:
                status = "complete"

        # 最终 hits = 跨轮累计去重证据 (非仅最后一轮 route_results), 前端命中更完整
        evidence_hits_data = [
            asdict(h) if isinstance(h, Hit) else h for h in evidence
        ]

        node_timings = final_state.get("node_timings", {}) or {}
        skill_obj: Optional[ResearchSkill] = final_state.get("skill")
        skill_synthesis: Optional[Dict[str, str]] = None
        if skill_obj is not None and (
            skill_obj.synthesis_system or skill_obj.synthesis_user_template
            or skill_obj.synthesis_thinking_system
        ):
            skill_synthesis = {
                "system": skill_obj.synthesis_system,
                "thinking_system": skill_obj.synthesis_thinking_system,
                "user_template": skill_obj.synthesis_user_template,
            }
        result: Dict[str, Any] = {
            "query": query,
            "professional": True,
            "skill_id": final_state.get("skill_id"),
            "skill_name": getattr(final_state.get("skill"), "name", None),
            "skill_synthesis": skill_synthesis,
            "context": context,
            "needs_clarify": needs_clarify,
            "clarify_request": final_state.get("clarify_request"),
            "research_complete": bool(final_state.get("research_complete")),
            "research_status": status,
            "research_rounds": int(final_state.get("research_round") or 0),
            "research_gaps": final_state.get("research_gaps", []) or [],
            "research_covered": final_state.get("research_covered", []) or [],
            "research_log": final_state.get("research_log", []) or [],
            "evidence_doc_count": doc_count,
            "evidence_chunk_count": len(evidence),
            "evidence_hits": evidence_hits_data,
            "doc_registry": persisted_registry,
            "results": final_state.get("route_results", {}),
            "correlation_id": cid,
            "no_answer": no_answer,
            "persist_last_context": _lg._truncate_for_persist(context, 8000)
            if hasattr(_lg, "_truncate_for_persist") else context[:8000],
            # #6/#8/#10: 跨轮研究状态持久化 (精简, 不含全文 hits)
            "research_carryover": {
                "goal": str(final_state.get("research_goal") or ""),
                "skill_id": final_state.get("skill_id"),
                "gaps": list(final_state.get("research_gaps") or []),
                "covered": list(final_state.get("research_covered") or []),
                "tried": list(final_state.get("research_tried") or []),
                "evidence_doc_ids": [
                    {"doc_id": h.doc_id, "doc_name": h.doc_name}
                    for h in evidence
                    if isinstance(h, Hit) and h.doc_id
                ],
                "evidence_seen_keys": list(final_state.get("evidence_seen_keys") or []),
            },
            "latency": {
                "plan_s": round(node_timings.get("plan", 0), 3),
                "retrieve_s": round(node_timings.get("retrieve", 0), 3),
                "reranker_s": round(node_timings.get("reranker", 0), 3),
                "research_policy_s": round(node_timings.get("research_policy", 0), 3),
                "total_s": round(time.time() - t0, 3),
            },
        }
        if needs_clarify:
            result["answer"] = _lg._format_clarify_answer(
                final_state.get("clarify_request") or {}
            )
        elif direct_answer:
            result["answer"] = direct_answer
            result["direct_answer"] = direct_answer
        logger.info(
            f"[{cid}] [research.run] DONE: status={status} "
            f"rounds={result['research_rounds']} evidence_docs={doc_count} "
            f"chunks={len(evidence)} clarify={needs_clarify} gaps={result['research_gaps']}"
        )
        return result

    def run(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        session_meta: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        t0 = time.time()
        initial_state = self._build_initial_state(query, history, session_meta)
        final_state = self.graph.invoke(
            initial_state, config={"recursion_limit": self.recursion_limit},
        )
        return self._finalize(
            query, final_state, t0, initial_state.get("last_round_docs") or [],
        )

    def run_events(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        session_meta: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ):
        """流式驱动: 逐节点 yield ('thinking', {...}); 收尾 yield ('result', result_dict)。

        thinking 事件实时展示研究的"思考过程": 规划如何拆解目标、每轮评估了什么、
        还缺什么、为什么继续或收口 (综述阶段的思考由上层 flow 在生成时另行流式产出)。
        """
        t0 = time.time()
        initial_state = self._build_initial_state(query, history, session_meta)
        last_round_docs = initial_state.get("last_round_docs") or []
        final_state: ResearchState = initial_state
        emitted_thinking = 0
        for state in self.graph.stream(
            initial_state,
            config={"recursion_limit": self.recursion_limit},
            stream_mode="values",
        ):
            final_state = state
            thinking_log = state.get("research_thinking") or []
            if len(thinking_log) > emitted_thinking:
                for entry in thinking_log[emitted_thinking:]:
                    yield ("thinking", {
                        "content": entry.get("content", ""),
                        "round": int(entry.get("round", 0) or 0),
                        "phase": entry.get("phase", ""),
                    })
                emitted_thinking = len(thinking_log)
        yield ("result", self._finalize(query, final_state, t0, last_round_docs))


def build_research_agent_from_pipeline(
    pipeline: AgenticRAGPipeline,
    *,
    planner_llm: LLMClient,
    policy_llm: LLMClient,
    reranker_client: Optional[RerankerClient] = None,
    limits: Optional[RoutingLimits] = None,
    max_batches: int = 3,
    max_rounds: int = 4,
    stall_limit: int = 2,
    planner_max_tokens: int = 2048,
    policy_max_tokens: int = 2048,
    planner_disable_thinking: Optional[bool] = None,
    policy_disable_thinking: Optional[bool] = None,
    reranker_top_k: int = 5,
    reranker_quality_k: int = 3,
    reranker_quality_threshold: float = 0.5,
    reranker_quality_threshold_by_type: Optional[Dict[str, float]] = None,
    reranker_route_thresholds: Optional[Any] = None,
    reranker_diagnosis_config: Optional[Any] = None,
    fail_open_min_emb_quality: Optional[float] = None,
    max_workers: int = 4,
    summary_top_docs: int = 5,
    summary_per_query_k: int = 5,
    synthesis_snippet_chars: int = 500,
    reflect_summary_config: Optional[ReflectSummaryConfig] = None,
    stall_quality_floor: Optional[float] = None,
    obs_summary_max_chars: int = 1800,
    gap_stall_limit: int = 2,
    skills: Optional[Dict[str, ResearchSkill]] = None,
    skill_router_llm: Optional[LLMClient] = None,
    skill_router_mode: str = "llm",
    skill_router_max_tokens: int = 512,
    skill_router_disable_thinking: bool = False,
    skill_router_min_confidence: float = 0.6,
) -> ResearchAgent:
    compiled = build_research_graph(
        pipeline,
        planner_llm=planner_llm,
        policy_llm=policy_llm,
        reranker_client=reranker_client,
        limits=limits,
        max_batches=max_batches,
        max_rounds=max_rounds,
        stall_limit=stall_limit,
        planner_max_tokens=planner_max_tokens,
        policy_max_tokens=policy_max_tokens,
        planner_disable_thinking=planner_disable_thinking,
        policy_disable_thinking=policy_disable_thinking,
        reranker_top_k=reranker_top_k,
        reranker_quality_k=reranker_quality_k,
        reranker_quality_threshold=reranker_quality_threshold,
        reranker_quality_threshold_by_type=reranker_quality_threshold_by_type,
        reranker_route_thresholds=reranker_route_thresholds,
        reranker_diagnosis_config=reranker_diagnosis_config,
        fail_open_min_emb_quality=fail_open_min_emb_quality,
        max_workers=max_workers,
        summary_top_docs=summary_top_docs,
        summary_per_query_k=summary_per_query_k,
        synthesis_snippet_chars=synthesis_snippet_chars,
        reflect_summary_config=reflect_summary_config,
        stall_quality_floor=stall_quality_floor,
        obs_summary_max_chars=obs_summary_max_chars,
        gap_stall_limit=gap_stall_limit,
        skills=skills,
        skill_router_llm=skill_router_llm,
        skill_router_mode=skill_router_mode,
        skill_router_max_tokens=skill_router_max_tokens,
        skill_router_disable_thinking=skill_router_disable_thinking,
        skill_router_min_confidence=skill_router_min_confidence,
    )
    # 某些 skill 可能放大 max_rounds; 递归预算按最大轮次估算 (+ skill_router 节点余量)
    eff_max_rounds = max(
        [max_rounds] + [int(s.max_rounds) for s in (skills or {}).values() if s.max_rounds]
    )
    return ResearchAgent(compiled, max_rounds=eff_max_rounds)


# ---------------------------------------------------------------------------
# 综述生成 prompt (供上层 flow 使用)
# ---------------------------------------------------------------------------

RESEARCH_SYNTHESIS_SYSTEM = (
    "你是严谨的文献研究综述助手。下面给你的是经过多轮检索、按文献分组的证据材料。"
    "全程使用简体中文作答 (材料中的专有名词、化学式、单位可照搬原文)。"
    "请基于这些证据撰写一份**研究综述式回答**, 严格按以下固定结构输出 (用 Markdown 小标题):\n\n"
    "## 核心结论\n"
    "用 3-5 句直接回答'研究目标', 给出最关键的发现。\n\n"
    "## 分论点与文献依据\n"
    "按研究维度/主题分小节展开, 每个关键论点后用文献名标注来源 (如 [文献名 或 编号])。\n\n"
    "## 证据要点表\n"
    "用 Markdown 表格汇总关键定量/定性证据, 列建议为: 维度 | 关键发现 | 来源文献。\n\n"
    "## 共识与分歧\n"
    "明确指出多篇文献达成共识之处, 以及结论不一致/有争议之处 (若证据不足以判断, 直说)。\n\n"
    "## 证据缺口与下一步\n"
    "客观列出当前证据未能覆盖的问题, 以及建议的后续检索/研究方向。\n\n"
    "硬性要求:\n"
    "- 关键论点必须可溯源到给定证据材料, 严禁编造材料中不存在的数据或结论;\n"
    "- **逐一核对'研究维度'**: 只对'证据材料'中确有对应内容的维度展开论述; 对没有任何证据材料支撑的维度, "
    "**严禁编造或脑补**, 必须把它明确列入'## 证据缺口与下一步'并说明'当前检索未覆盖';\n"
    "- 证据不足时如实说明, 不要臆测填充;\n"
    "- 保持术语与单位准确 (公式/化学式照搬材料中的写法);\n"
    "- 公式保留原始 LaTeX, 并用 $...$ 包裹行内公式、$$...$$ 包裹独立公式块。"
    "例如: 行内 $\dot{\\varepsilon}$, 独立 $$t = \\frac{\\varepsilon}{\\dot{\\varepsilon}} \\tag{1}$$。"
    "绝对不要输出没有 $ 定界符的裸 LaTeX, 否则公式无法在前端渲染。"
)

RESEARCH_SYNTHESIS_USER_TEMPLATE = (
    "以下是检索到的研究证据材料:\n\n{context}\n\n"
    "请据此撰写面向研究目标的综述式回答。"
)


# 综述前的"中文分析思路": 用于流式展示思考过程。
# 说明: 本地 Qwen3.5 原生 reasoning 恒为英文且无法用 prompt/前缀纠正, 故改由模型显式
# 用中文产出一段"分析思路"作为可读思考过程, 再单独直出综述正文 (见 flows/query.py)。
RESEARCH_THINKING_SYSTEM = (
    "你是严谨的文献研究综述助手。下面给你的是经过多轮检索、按文献分组的研究证据材料。\n"
    "请用简体中文、以第一人称简要复盘你的分析思路: 梳理了哪些关键文献/研究维度、各维度最重要的"
    "证据是什么、哪里存在共识或分歧、哪里证据仍不足。\n"
    "要求: 4-8 条要点, 像口头复盘一样自然简洁; 全程简体中文, 严禁任何英文句子; "
    "不要写正式综述正文, 不要使用 Markdown 大标题。"
)

RESEARCH_THINKING_USER_TEMPLATE = (
    "以下是检索到的研究证据材料:\n\n{context}\n\n"
    "请用简体中文简要说明你将如何组织这份综述 (只说分析思路, 不要写正文)。"
)
