"""专业研究模式: 规划 (research_plan) 与 policy (continue/finish/clarify) 的 LLM 调用与解析。

设计为 **standalone 函数 + 轻量 dataclass**, 直接复用 LLMClient / fc_parser /
build_from_multi_args, 不触碰 RoutingCore, 因此对现有链路零影响。

约定:
  - 任何 LLM/解析失败都不抛, 返回 None (规划) 或一个安全的 finish 决策 (policy),
    让上层研究图能优雅收口而不是崩溃;
  - batch (research_plan / policy 的检索批次) → MultiRouteDecision 的映射通过把
    batches 转成 multi 工具的 subs 形态后复用 build_from_multi_args, 1:1 对齐普通模式。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..clients.llm import LLMClient
from .decision_builder import (
    MultiRouteDecision,
    SubqueryDecision,
    build_from_multi_args,
    build_from_plan_args,
)
from .fc_parser import parse_tool_calls
from .limits import DEFAULT_ROUTING_LIMITS, RoutingLimits
from .research_schema import (
    RESEARCH_POLICY_TOOL_NAMES,
    TOOL_RESEARCH_CLARIFY,
    TOOL_RESEARCH_CONTINUE,
    TOOL_RESEARCH_FINISH,
    TOOL_RESEARCH_PLAN,
    TOOL_RESEARCH_REJECT,
    research_plan_tools,
    research_policy_tools,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class ResearchFacet:
    id: str
    question: str
    keywords: List[str] = field(default_factory=list)
    priority: str = "medium"
    evidence_needed: List[str] = field(default_factory=list)


@dataclass
class ResearchPlan:
    goal: str
    facets: List[ResearchFacet] = field(default_factory=list)
    initial_batches: List[Dict[str, Any]] = field(default_factory=list)
    task_type: str = ""
    sufficiency: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    def facet_ids(self) -> List[str]:
        return [f.id for f in self.facets if f.id]


@dataclass
class PlanOutcome:
    """规划阶段的产物: 正常计划 / 追问 / 兜底直答。"""
    action: str                       # plan | clarify | reject
    plan: Optional["ResearchPlan"] = None
    clarify_q: str = ""
    clarify_opts: List[str] = field(default_factory=list)
    reject_kind: str = ""             # out_of_scope | chitchat | meaningless
    reply: str = ""                   # reject 的兜底回复
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyDecision:
    action: str                       # continue | finish | clarify
    covered: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    next_batches: List[Dict[str, Any]] = field(default_factory=list)
    reason: str = ""
    clarify_q: str = ""
    clarify_opts: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

# 规划提示词拆成三段, 便于 skill 只替换"拆解段"而保留前置闸门与 FC 约束:
#   _PLAN_GATING_PREAMBLE (reject/clarify/plan 三选一闸门)
#   + 拆解段 (通用 _PLAN_DECOMP_GENERIC, 或 skill 的 plan.md 正文)
#   + _PLAN_FC_SUFFIX (function-calling 约束)
_PLAN_GATING_PREAMBLE = (
    "你是文献研究规划助手。用户处于'专业研究模式', 需要从多篇相互关联的文献中检索、汇总、"
    "递进式求证。\n"
    "第一步先判断该输入是否值得进入检索流程, 三选一调用工具:\n"
    "  - 与文献库主题相关且足够明确 → research_plan: 拆 facets + 首轮批次;\n"
    "  - 属于本库主题但过于模糊/宽泛、缺关键限定(对象/范围/角度)以致无法形成有效检索 → "
    "research_clarify: 用一句话追问, 给 2-4 个候选方向 opts 帮用户缩小范围;\n"
    "  - 与文献库主题无关 / 纯闲聊问候 / 无意义空泛输入 → research_reject: 直接给礼貌兜底回复, 不检索。\n"
    "宁可在模糊时先 clarify, 也不要硬拆成空泛 facets 浪费多轮检索。\n"
)

_PLAN_DECOMP_GENERIC = (
    "选择 research_plan 时: 理解用户真实研究目标 (含潜在需求), 把它拆成若干互相独立、"
    "可分别判定是否覆盖的 facet (子问题/证据维度), 并规划首轮并行检索批次。\n"
    "检索效率原则:\n"
    "  1. 首轮必须以 summary 路径为主做广搜: initial_batches 中至少一个 batch 的 paths 含 summary, "
    "先快速定位相关文献集合; 不要首轮就对全库做昂贵的 progressive(chunk 级)检索 —— "
    "只有当用户明确指向某篇具体文献或某个精确事实时, 才可首轮直接用 progressive;\n"
    "  2. 每个 facet 用独立 batch, 单轮内并行执行;\n"
    "  3. 关键词写'主体+问点'离散词 (含英文同义词), 禁止 文献/研究/有没有/哪些 等元话语。\n"
)

_PLAN_FC_SUFFIX = (
    "只通过 function calling 调用 research_plan / research_clarify / research_reject 之一, 不要输出其他文本。"
)

# policy 提示词同样拆段: 前置(观测说明) + 判断段(通用或 skill) + FC 约束。
_POLICY_PREAMBLE = (
    "你是文献研究的检索策略控制器。每轮检索后, 你会看到: 用户原始问题与研究目标、各 facet 覆盖情况、"
    "上一轮 policy 决策 (动作/理由/缺口/下发关键词)、本轮 rerank 命中 (含精排分)、本轮新证据摘要、"
    "已累计覆盖的文献、历史缺口、剩余轮次预算。\n"
)

_POLICY_JUDGEMENT_GENERIC = (
    "判断依据: 看 rerank 分高的命中是否真正补上了 gaps; 若本轮命中分普遍很低或与上轮高度重复, 说明再检索难有进展。\n"
    "你的任务: 判断当前累计证据是否足以充分回答研究目标, 三选一:\n"
    "  - 不足且仍有预算且能靠再检索补齐 → research_continue, 明确 gaps + 给出与'上一轮下发关键词'显著不同的 next_batches;\n"
    "  - 已足够 (覆盖关键 facet、文献数量够、必要时已核对冲突) → research_finish;\n"
    "  - 连续多轮检索不到新进展 / 语料确实缺关键资料 / 目标过宽无法收敛 → research_clarify 反问, 让用户补充或缩小范围。\n"
    "效率约束: 不要重复上一轮已下发的关键词或已覆盖的 facet; 优先补 gaps 指向的维度; 倾向尽早收口或反问, 避免空耗检索。\n"
)

_POLICY_FC_SUFFIX = (
    "只通过 function calling 调用 research_continue / research_finish / research_clarify 之一, 不要输出其他文本。"
)

# 通用提示词 = 三段拼接 (与拆分前逐字等价, 保证无 skill 时零回归)
_PLAN_SYSTEM = _PLAN_GATING_PREAMBLE + _PLAN_DECOMP_GENERIC + _PLAN_FC_SUFFIX
_POLICY_SYSTEM = _POLICY_PREAMBLE + _POLICY_JUDGEMENT_GENERIC + _POLICY_FC_SUFFIX


_PREFER_PATH_LABELS = {
    "summary": "summary(综述/广搜)",
    "progressive": "progressive(chunk 级正文深挖)",
    "local": "local(指定文献内精读)",
    "metadata": "metadata(结构化元数据)",
}


def _prefer_paths_hint(prefer_first_paths: Optional[List[str]]) -> str:
    """把 skill 的 prefer_first_paths 渲染成首轮检索路径偏好提示 (仅引导, 不强制)。"""
    paths = [p for p in (prefer_first_paths or []) if p in _PREFER_PATH_LABELS]
    if not paths:
        return ""
    labels = "、".join(_PREFER_PATH_LABELS[p] for p in paths)
    return (
        f"首轮检索路径偏好: 优先在 initial_batches 中使用 {labels} 路径"
        "(除非用户明确指向某篇具体文献或某个精确事实)。\n"
    )


def compose_plan_system(
    skill_decomp: Optional[str],
    prefer_first_paths: Optional[List[str]] = None,
) -> str:
    """用 skill 的拆解段组合规划 system; skill_decomp 为空则用通用拆解段。

    prefer_first_paths 非空时追加首轮检索路径偏好提示 (来自 skill frontmatter)。
    """
    body = (skill_decomp or "").strip()
    body = (body + "\n") if body else _PLAN_DECOMP_GENERIC
    return _PLAN_GATING_PREAMBLE + body + _prefer_paths_hint(prefer_first_paths) + _PLAN_FC_SUFFIX


def compose_policy_system(skill_judgement: Optional[str]) -> str:
    """用 skill 的判断段组合 policy system; 为空则用通用判断段。"""
    body = (skill_judgement or "").strip()
    body = (body + "\n") if body else _POLICY_JUDGEMENT_GENERIC
    return _POLICY_PREAMBLE + body + _POLICY_FC_SUFFIX


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------

def _parse_facets(raw: Any) -> List[ResearchFacet]:
    out: List[ResearchFacet] = []
    if not isinstance(raw, list):
        return out
    for i, f in enumerate(raw):
        if not isinstance(f, dict):
            continue
        fid = str(f.get("id") or f"facet{i + 1}").strip()
        question = str(f.get("question") or "").strip()
        kws = [str(k).strip() for k in (f.get("keywords") or []) if str(k).strip()]
        if not question and not kws:
            continue
        out.append(ResearchFacet(
            id=fid,
            question=question,
            keywords=kws,
            priority=str(f.get("priority") or "medium"),
            evidence_needed=[
                str(e).strip() for e in (f.get("evidence_needed") or []) if str(e).strip()
            ],
        ))
    return out


def _clean_batches(raw: Any) -> List[Dict[str, Any]]:
    """规整 batches: 只保留有合法 paths 的批次。"""
    out: List[Dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for i, b in enumerate(raw):
        if not isinstance(b, dict):
            continue
        paths = b.get("paths")
        if not isinstance(paths, list) or not paths:
            continue
        out.append({
            "id": str(b.get("id") or f"b{i + 1}"),
            "facet_id": str(b.get("facet_id") or ""),
            "purpose": str(b.get("purpose") or ""),
            "paths": paths,
        })
    return out


def _think_kwargs(disable_thinking: Optional[bool]) -> Dict[str, Any]:
    return {} if disable_thinking is None else {"disable_thinking": disable_thinking}


# ---------------------------------------------------------------------------
# 规划调用
# ---------------------------------------------------------------------------

def plan_research(
    llm: LLMClient,
    query: str,
    *,
    doc_registry: Optional[List[Dict[str, str]]] = None,
    history: Optional[List[Dict[str, str]]] = None,
    max_batches: int = 3,
    limits: Optional[RoutingLimits] = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    disable_thinking: Optional[bool] = None,
    correlation_id: str = "-",
    prev_clarify: Optional[Dict[str, str]] = None,
    carryover_hint: str = "",
    system: Optional[str] = None,
    default_sufficiency: Optional[Dict[str, Any]] = None,
) -> Optional[PlanOutcome]:
    """调用规划 LLM。返回 PlanOutcome(plan/clarify/reject); 失败返回 None (上层走启发式兜底)。

    system: 规划 system 提示词 (skill 专属); None 时用通用 _PLAN_SYSTEM。
    default_sufficiency: skill 默认收口标准, 当 LLM 未给出 sufficiency 时注入。
    """
    lim = limits or DEFAULT_ROUTING_LIMITS
    tools = research_plan_tools(max_batches=max_batches, limits=lim)

    registry_hint = ""
    if doc_registry:
        names = [
            str(e.get("doc_name") or e.get("doc_id") or "")
            for e in doc_registry[:15]
            if isinstance(e, dict)
        ]
        names = [n for n in names if n]
        if names:
            registry_hint = "\n\n【会话中已出现的文献 (可在 docs/refs 中回指)】\n" + "\n".join(
                f"{i + 1}. {n}" for i, n in enumerate(names)
            )

    clarify_hint = ""
    if prev_clarify and prev_clarify.get("question"):
        clarify_hint = (
            "\n\n【上一轮你曾向用户反问】\n"
            f"反问: {prev_clarify['question']}\n"
            f"用户本次输入即为对该反问的回应。请据此精准规划: 聚焦用户澄清后的真实意图, "
            f"不要再重复之前已被纠正的方向, 也不要再问同样的问题。"
        )

    # #6: carryover_hint 由 plan_node 从跨轮状态构建，包含上一轮的 gaps/covered/tried
    user_msg = (
        f"用户研究型问题: {query}{registry_hint}{clarify_hint}{carryover_hint}\n\n"
        f"请通过 function calling 调用 research_plan 产出研究计划。"
    )
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system or _PLAN_SYSTEM}
    ]
    if history:
        messages.extend(history[-2:])
    messages.append({"role": "user", "content": user_msg})

    t0 = time.time()
    try:
        resp = llm.chat_with_tools(
            messages=messages,
            tools=tools,
            tool_choice="required",
            temperature=temperature,
            max_tokens=max_tokens,
            parallel_tool_calls=False,
            **_think_kwargs(disable_thinking),
        )
    except Exception as e:
        logger.warning(f"[{correlation_id}] [research.plan] LLM 调用失败: {e}")
        return None

    calls, source = parse_tool_calls(resp)
    ms = int((time.time() - t0) * 1000)
    if not calls:
        logger.warning(
            f"[{correlation_id}] [research.plan] 未解析到 tool_call (source={source}, {ms}ms)"
        )
        return None

    call = calls[0]
    args = call.arguments or {}

    # 兜底直答: 无关 / 闲聊 / 无意义
    if call.name == TOOL_RESEARCH_REJECT:
        reply = str(args.get("reply") or "").strip()
        kind = str(args.get("kind") or "out_of_scope").strip()
        if not reply:
            reply = "这个问题似乎和当前文献库的主题不太相关，欢迎提出与库内文献相关的研究问题。"
        logger.info(f"[{correlation_id}] [research.plan] DONE ({ms}ms): reject kind={kind!r}")
        return PlanOutcome(action="reject", reject_kind=kind, reply=reply, raw=args)

    # 追问: 问题模糊/过宽
    if call.name == TOOL_RESEARCH_CLARIFY:
        q = str(args.get("q") or "").strip()
        opts = [str(o) for o in (args.get("opts") or []) if str(o).strip()]
        if not q:
            logger.warning(f"[{correlation_id}] [research.plan] clarify 内容为空, 退化为兜底")
            return PlanOutcome(
                action="reject", reject_kind="meaningless",
                reply="能否补充一下你想研究的具体对象或方向？这样我才能帮你检索相关文献。",
                raw=args,
            )
        logger.info(f"[{correlation_id}] [research.plan] DONE ({ms}ms): clarify q={q[:50]!r}")
        return PlanOutcome(action="clarify", clarify_q=q, clarify_opts=opts, raw=args)

    if call.name != TOOL_RESEARCH_PLAN:
        logger.warning(f"[{correlation_id}] [research.plan] 非预期工具 {call.name!r}")
        return None

    facets = _parse_facets(args.get("facets"))
    batches = _clean_batches(args.get("initial_batches"))
    goal = str(args.get("goal") or query).strip()

    if not batches:
        logger.warning(f"[{correlation_id}] [research.plan] initial_batches 为空, 规划无效")
        return None

    sufficiency = args.get("sufficiency") if isinstance(args.get("sufficiency"), dict) else {}
    # skill 默认收口标准: 仅在 LLM 未给出对应键时补齐, 不覆盖 LLM 的显式判断
    if default_sufficiency:
        merged = dict(default_sufficiency)
        merged.update(sufficiency)
        sufficiency = merged
    plan = ResearchPlan(
        goal=goal,
        facets=facets,
        initial_batches=batches,
        task_type=str(args.get("task_type") or ""),
        sufficiency=sufficiency,
        raw=args,
    )
    logger.info(
        f"[{correlation_id}] [research.plan] DONE ({ms}ms): goal={goal[:60]!r} "
        f"facets={plan.facet_ids()} batches={len(batches)} task_type={plan.task_type!r}"
    )
    return PlanOutcome(action="plan", plan=plan, raw=args)


# ---------------------------------------------------------------------------
# Policy 调用
# ---------------------------------------------------------------------------

def decide_policy(
    llm: LLMClient,
    *,
    goal: str,
    observation: str,
    round_idx: int,
    rounds_left: int,
    max_batches: int = 3,
    limits: Optional[RoutingLimits] = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    disable_thinking: Optional[bool] = None,
    correlation_id: str = "-",
    system: Optional[str] = None,
) -> PolicyDecision:
    """调用 policy LLM 决定下一步。失败/解析不到时安全收口 (finish)。

    system: policy system 提示词 (skill 专属); None 时用通用 _POLICY_SYSTEM。
    """
    lim = limits or DEFAULT_ROUTING_LIMITS
    tools = research_policy_tools(max_batches=max_batches, limits=lim)

    user_msg = (
        f"研究目标: {goal}\n\n"
        f"当前是第 {round_idx} 轮检索后的评估; 剩余检索轮次预算: {rounds_left}。\n\n"
        f"{observation}\n\n"
        f"请通过 function calling 调用 research_continue / research_finish / research_clarify 之一。"
        + ("\n注意: 剩余预算为 0, 除非语料缺关键资料需 clarify, 否则应 finish。" if rounds_left <= 0 else "")
    )
    messages = [
        {"role": "system", "content": system or _POLICY_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    def _call(dt: Optional[bool], mt: int):
        return llm.chat_with_tools(
            messages=messages,
            tools=tools,
            tool_choice="required",
            temperature=temperature,
            max_tokens=mt,
            parallel_tool_calls=False,
            **_think_kwargs(dt),
        )

    t0 = time.time()
    try:
        resp = _call(disable_thinking, max_tokens)
    except Exception as e:
        logger.warning(
            f"[{correlation_id}] [research.policy] LLM 调用失败, 安全收口 finish: {e}"
        )
        return PolicyDecision(action="finish", reason="policy LLM 调用失败, 安全收口", raw={})

    calls, source = parse_tool_calls(resp)
    # 防御: 开思考时超长 CoT 可能把 tool_call 顶出 max_tokens 导致解析为空。
    # 此时关思考重试一次 (直出 FC), 避免静默提前收口。
    if (not calls or calls[0].name not in RESEARCH_POLICY_TOOL_NAMES) and disable_thinking is not True:
        logger.warning(
            f"[{correlation_id}] [research.policy] tool_call 解析为空 (source={source}), "
            f"关思考重试一次"
        )
        try:
            resp = _call(True, max_tokens)
            calls, source = parse_tool_calls(resp)
        except Exception as e:
            logger.warning(f"[{correlation_id}] [research.policy] 重试失败: {e}")

    ms = int((time.time() - t0) * 1000)
    if not calls or calls[0].name not in RESEARCH_POLICY_TOOL_NAMES:
        logger.warning(
            f"[{correlation_id}] [research.policy] 未解析到合法 tool_call "
            f"(source={source}, {ms}ms), 安全收口 finish"
        )
        return PolicyDecision(action="finish", reason="policy 解析失败, 安全收口", raw={})

    call = calls[0]
    args = call.arguments or {}

    if call.name == TOOL_RESEARCH_FINISH:
        dec = PolicyDecision(
            action="finish",
            covered=[str(c) for c in (args.get("covered") or [])],
            gaps=[str(g) for g in (args.get("residual_gaps") or [])],
            reason=str(args.get("reason") or ""),
            raw=args,
        )
    elif call.name == TOOL_RESEARCH_CLARIFY:
        dec = PolicyDecision(
            action="clarify",
            clarify_q=str(args.get("q") or "").strip(),
            clarify_opts=[str(o) for o in (args.get("opts") or []) if str(o).strip()],
            reason=str(args.get("reason") or ""),
            raw=args,
        )
        if not dec.clarify_q:
            # 反问内容缺失则退化为 finish, 避免空反问
            dec = PolicyDecision(action="finish", reason="clarify 内容为空, 收口", raw=args)
    else:  # research_continue
        batches = _clean_batches(args.get("next_batches"))
        if not batches:
            dec = PolicyDecision(
                action="finish",
                covered=[str(c) for c in (args.get("covered") or [])],
                gaps=[str(g) for g in (args.get("gaps") or [])],
                reason="continue 未给出有效 next_batches, 收口",
                raw=args,
            )
        else:
            dec = PolicyDecision(
                action="continue",
                covered=[str(c) for c in (args.get("covered") or [])],
                gaps=[str(g) for g in (args.get("gaps") or [])],
                next_batches=batches,
                reason=str(args.get("reason") or ""),
                raw=args,
            )

    logger.info(
        f"[{correlation_id}] [research.policy] DONE ({ms}ms): action={dec.action} "
        f"covered={dec.covered} gaps={dec.gaps} next_batches={len(dec.next_batches)} "
        f"reason={dec.reason[:60]!r}"
    )
    return dec


# ---------------------------------------------------------------------------
# batches → MultiRouteDecision
# ---------------------------------------------------------------------------

def batches_to_multi_decision(
    batches: List[Dict[str, Any]],
    *,
    query: str,
    doc_registry: Optional[List[Dict[str, str]]] = None,
    validate_fn: Optional[callable] = None,
    limits: Optional[RoutingLimits] = None,
) -> Optional[MultiRouteDecision]:
    """把研究批次列表转成 MultiRouteDecision, 直接喂给现有 retrieve_node。

    每个 batch 映射成 multi 工具的一个 sub (paths + id), 复用 build_from_multi_args。
    返回 None 表示 batches 无效 (上层应跳过本轮检索)。
    """
    if not batches:
        return None
    lim = limits or DEFAULT_ROUTING_LIMITS
    subs: List[Dict[str, Any]] = []
    for b in batches:
        paths = b.get("paths")
        if not isinstance(paths, list) or not paths:
            continue
        subs.append({
            "id": str(b.get("id") or f"b{len(subs) + 1}"),
            "paths": paths,
        })
    if not subs:
        return None

    # 单批次: 直接构造 1-sub MultiRouteDecision, 避免 build_from_multi_args 的
    # "subs 不足 2 个" 警告 (研究模式单批次轮次很常见)。
    if len(subs) == 1:
        sub_dec = build_from_plan_args(
            subs[0], query=query, doc_registry=doc_registry,
            validate_fn=validate_fn, reasoning_tag="(research-batch)",
            limits=lim,
        )
        return MultiRouteDecision(
            subqueries=[SubqueryDecision(id=subs[0]["id"], decision=sub_dec, raw=subs[0])],
            raw={"subs": subs},
        )

    return build_from_multi_args(
        {"subs": subs},
        query=query,
        doc_registry=doc_registry,
        validate_fn=validate_fn,
        limits=lim,
    )
