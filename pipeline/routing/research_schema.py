"""专业研究模式 (professional) 的 Function Calling 工具 schema。

与普通模式的 fc_schema.py **完全隔离**: 本文件只新增工具, 不修改、不暴露给现有
router_tools()/reflect_tools(), 因此对现有链路零影响。

两类工具:
  1. research_plan       —— 研究规划 (router 层第一步): 把用户的研究型问题拆成
     facets (子问题/证据维度) + initial_batches (首轮并行检索批次)。
  2. policy 三件套       —— 每轮检索后的观测决策:
       - research_continue : 证据不足, 给出缺口 + 下一轮检索批次
       - research_finish   : 证据足够, 进入综述生成
       - research_clarify  : 语料缺失 / 目标过宽, 反问用户

检索效率设计:
  - initial_batches / next_batches 的 paths 直接复用普通模式的 4 路径 schema
    (summary/progressive/local/metadata), 保证 decision_builder 1:1 映射, 不需要新解析;
  - batch 是并行执行单元 (映射成 MultiRouteDecision 的一个 sub), 一轮内多 batch 并行;
  - 鼓励 "先 summary 广搜定位文献集合, 再 progressive/local 抽证据" 的廉价优先级联,
    把昂贵的 chunk 级精排留到高相关文献上。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .fc_schema import _PATH_ONEOF
from .limits import DEFAULT_ROUTING_LIMITS, RoutingLimits


# ---------------------------------------------------------------------------
# 公共子 schema
# ---------------------------------------------------------------------------

def _batch_paths_schema(*, max_paths: int) -> Dict[str, Any]:
    cap = max(1, int(max_paths))
    return {
        "type": "array",
        "minItems": 1,
        "maxItems": cap,
        "items": _PATH_ONEOF,
        "description": (
            f"该检索批次的 1-{cap} 条路径 (与普通模式同构: summary/progressive/local/metadata)。"
            "效率建议: 文献发现用 summary; 证据抽取用 progressive/local; "
            "跨文献关联用 progressive + expand:['similar']。"
        ),
    }


def _batches_schema(*, max_batches: int, max_paths: int) -> Dict[str, Any]:
    cap = max(1, int(max_batches))
    return {
        "type": "array",
        "minItems": 1,
        "maxItems": cap,
        "description": (
            f"本轮并行检索批次数组 (1-{cap})。每个 batch 独立并行执行, 互不合并 filter。"
            "把不同 facet / 不同检索目的拆成不同 batch, 单轮内并行跑完。"
        ),
        "items": {
            "type": "object",
            "required": ["purpose", "paths"],
            "additionalProperties": False,
            "properties": {
                "id": {
                    "type": "string",
                    "description": "可选批次 id (如 facet 的 id); 不填则系统按顺序编号。",
                },
                "facet_id": {
                    "type": "string",
                    "description": "可选: 本批次服务于哪个 facet 的 id (用于覆盖度统计)。",
                },
                "purpose": {
                    "type": "string",
                    "description": "≤30 字: 这个批次想拿到什么证据 (如 '定位耐蚀钢相关文献'/'抽腐蚀速率定量指标')。",
                },
                "paths": _batch_paths_schema(max_paths=max_paths),
            },
        },
    }


def _facets_schema() -> Dict[str, Any]:
    return {
        "type": "array",
        "minItems": 1,
        "maxItems": 8,
        "description": (
            "研究子问题 / 证据维度数组。把用户的研究目标 (含潜在需求) 分解成可独立检索、"
            "可独立判定是否覆盖的若干 facet。"
        ),
        "items": {
            "type": "object",
            "required": ["id", "question", "keywords"],
            "additionalProperties": False,
            "properties": {
                "id": {
                    "type": "string",
                    "description": "facet 短 id (如 'corrosion_mechanism'), 用于覆盖度跟踪与 batch 关联。",
                },
                "question": {
                    "type": "string",
                    "description": "该维度要回答的具体子问题。",
                },
                "keywords": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string"},
                    "description": "检索关键词 (主体+问点, 含英文同义词); 禁止 文献/研究/有没有 等元话语。",
                },
                "priority": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "可选: facet 优先级, 高优先级先检索。",
                },
                "evidence_needed": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "可选: 该维度需要的证据类型 (如 '定量腐蚀速率'/'实验条件'/'机理图')。",
                },
            },
        },
    }


def _sufficiency_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "description": "回答研究目标所需的最低证据标准 (供 policy 判定何时收口)。",
        "properties": {
            "min_docs": {
                "type": "integer",
                "minimum": 1,
                "description": "至少需要覆盖多少篇相关文献。",
            },
            "must_cover": {
                "type": "array",
                "items": {"type": "string"},
                "description": "必须覆盖的 facet id 列表 (缺一不可收口)。",
            },
            "need_conflict_check": {
                "type": "boolean",
                "description": "是否需要核对文献间结论是否冲突。",
            },
            "need_quantitative_data": {
                "type": "boolean",
                "description": "是否必须拿到定量数据/指标。",
            },
        },
    }


# ---------------------------------------------------------------------------
# Tool 1: research_plan (研究规划)
# ---------------------------------------------------------------------------

def build_research_plan_tool(
    *,
    max_batches: int = 3,
    limits: Optional[RoutingLimits] = None,
) -> Dict[str, Any]:
    lim = limits or DEFAULT_ROUTING_LIMITS
    return {
        "type": "function",
        "function": {
            "name": "research_plan",
            "description": (
                "【专业研究模式专用】把用户的研究型问题拆成研究计划: 明确真实研究目标 (含潜在需求), "
                "分解 facets (子问题/证据维度), 给出首轮并行检索批次 initial_batches。\n"
                "首轮策略 (效率优先): 通常先用 summary 路径广搜定位相关文献集合, 再在后续轮次用 "
                "progressive/local 在高相关文献内抽具体证据。不要一上来就对全库做昂贵的 chunk 级检索。\n"
                "facets 要互相独立、可分别判定是否覆盖; initial_batches 的每个 batch 并行执行。"
            ),
            "parameters": {
                "type": "object",
                "required": ["goal", "facets", "initial_batches"],
                "additionalProperties": False,
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "用户真实研究目标 (补全潜在需求后的一句话)。",
                    },
                    "task_type": {
                        "type": "string",
                        "enum": [
                            "literature_review",
                            "evidence_synthesis",
                            "comparison",
                            "mechanism_analysis",
                            "gap_analysis",
                            "method_survey",
                        ],
                        "description": "可选: 研究任务类型。",
                    },
                    "facets": _facets_schema(),
                    "initial_batches": _batches_schema(
                        max_batches=max_batches,
                        max_paths=lim.max_paths_per_sub,
                    ),
                    "sufficiency": _sufficiency_schema(),
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Tool 2/3/4: policy 三件套 (每轮检索后的观测决策)
# ---------------------------------------------------------------------------

def build_research_continue_tool(
    *,
    max_batches: int = 3,
    limits: Optional[RoutingLimits] = None,
) -> Dict[str, Any]:
    lim = limits or DEFAULT_ROUTING_LIMITS
    return {
        "type": "function",
        "function": {
            "name": "research_continue",
            "description": (
                "评估结论: 当前累计证据仍不足以充分回答研究目标, 需要再检索一轮。\n"
                "必须明确指出缺口 (gaps), 并给出**与已检索显著不同**的下一轮检索批次 next_batches "
                "(换 facet / 换关键词 / 收窄到具体文献 / 补定量数据)。\n"
                "效率约束: 不要重复已经检索过且已覆盖的 facet; 优先补 gaps 指向的维度。"
            ),
            "parameters": {
                "type": "object",
                "required": ["gaps", "next_batches"],
                "additionalProperties": False,
                "properties": {
                    "covered": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "已被充分覆盖的 facet id 列表。",
                    },
                    "gaps": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string"},
                        "description": "当前缺什么证据 (具体到维度/数据类型)。",
                    },
                    "next_batches": _batches_schema(
                        max_batches=max_batches,
                        max_paths=lim.max_paths_per_sub,
                    ),
                    "reason": {
                        "type": "string",
                        "description": "≤60 字: 为什么继续、下一轮想补什么。",
                    },
                },
            },
        },
    }


RESEARCH_FINISH_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "research_finish",
        "description": (
            "评估结论: 累计证据已能充分回答研究目标 (覆盖关键 facet、文献数量足够、必要时已核对冲突), "
            "进入综述式综合生成。\n"
            "倾向规则: 当达到 sufficiency 标准、或连续轮次无新增有效证据、或已达轮次预算时, 选 finish。"
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "covered": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "已覆盖的 facet id 列表。",
                },
                "residual_gaps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "可选: 仍未完全覆盖、但不阻塞收口的残余缺口 (会写进综述的'局限性')。",
                },
                "reason": {
                    "type": "string",
                    "description": "≤60 字: 为什么判定可以收口。",
                },
            },
        },
    },
}


RESEARCH_CLARIFY_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "research_clarify",
        "description": (
            "评估结论: 多轮检索后语料库仍明显缺少关键资料, 或用户研究目标过宽/过于模糊, "
            "继续检索也无法收敛 —— 反问用户以缩小范围或确认方向。\n"
            "仅在确实无法靠再检索解决时使用, 不要轻易反问。"
        ),
        "parameters": {
            "type": "object",
            "required": ["q"],
            "additionalProperties": False,
            "properties": {
                "q": {
                    "type": "string",
                    "description": "反问内容, 明确指出缺失的资料维度或需要用户缩小的范围。",
                },
                "opts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "可选: 2-4 个候选方向, 帮用户快速确认。",
                },
            },
        },
    },
}


RESEARCH_REJECT_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "research_reject",
        "description": (
            "【规划前置过滤】当用户输入不适合进入文献研究流程时调用, 直接给出兜底回复, 不做任何检索:\n"
            "  - 与本文献知识库主题无关 / 超出范围 (out_of_scope);\n"
            "  - 纯闲聊、问候、对系统能力的提问等非研究型输入 (chitchat);\n"
            "  - 明显无意义/空泛到无法形成任何检索方向的输入。\n"
            "注意: 只是'问题宽泛但仍属于本库主题'应改用 research_clarify 追问, 而不是 reject。"
        ),
        "parameters": {
            "type": "object",
            "required": ["reply"],
            "additionalProperties": False,
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["out_of_scope", "chitchat", "meaningless"],
                    "description": "兜底类型。",
                },
                "reply": {
                    "type": "string",
                    "description": "直接回复用户的话 (礼貌、简短; 必要时引导其提出与文献库相关的研究问题)。",
                },
            },
        },
    },
}


def research_plan_tools(
    *,
    max_batches: int = 3,
    limits: Optional[RoutingLimits] = None,
) -> List[Dict[str, Any]]:
    """router 层规划工具。

    暴露三件套, 让规划 LLM 先做"是否值得检索"的判断:
      - research_plan    : 正常研究 → 拆 facets + 首轮批次;
      - research_clarify : 问题模糊/过宽但属于本库主题 → 追问;
      - research_reject  : 无关/闲聊/无意义 → 兜底直答, 不检索。
    """
    return [
        build_research_plan_tool(max_batches=max_batches, limits=limits),
        RESEARCH_CLARIFY_TOOL,
        RESEARCH_REJECT_TOOL,
    ]


def research_policy_tools(
    *,
    max_batches: int = 3,
    limits: Optional[RoutingLimits] = None,
) -> List[Dict[str, Any]]:
    """policy 层三件套 (continue/finish/clarify, tool_choice=required)。"""
    return [
        build_research_continue_tool(max_batches=max_batches, limits=limits),
        RESEARCH_FINISH_TOOL,
        RESEARCH_CLARIFY_TOOL,
    ]


# 工具名常量
TOOL_RESEARCH_PLAN = "research_plan"
TOOL_RESEARCH_CONTINUE = "research_continue"
TOOL_RESEARCH_FINISH = "research_finish"
TOOL_RESEARCH_CLARIFY = "research_clarify"
TOOL_RESEARCH_REJECT = "research_reject"

RESEARCH_POLICY_TOOL_NAMES = (
    TOOL_RESEARCH_CONTINUE,
    TOOL_RESEARCH_FINISH,
    TOOL_RESEARCH_CLARIFY,
)
