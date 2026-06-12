"""文件定义式 skill 的加载、选择, 及其接入研究图后的提示词生效 / 通用回退测试。

复用 test_research_agent 的 mock 框架, 不依赖真实 LLM / Milvus / reranker。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import pytest

pytest.importorskip("langgraph")

from pipeline.retrieval import research_agent as ra
from pipeline.routing import research as rsearch
from pipeline.routing import research_skills as rs
from pipeline.routing.research import compose_plan_system

from pipeline.tests.test_research_agent import (
    ScriptLLM,
    _make_retrieve_stub,
    _hit,
    _plan,
    _tool_resp,
    _StubPipeline,
    _B1,
)

_SKILLS_DIR = os.path.join(os.path.dirname(__file__), "..", "skills")


def _load():
    return rs.load_skills([_SKILLS_DIR])


class RouterLLM:
    """脚本化的 skill 分类模型: 依次吐出 {skill, confidence, reason} JSON。

    select_skill 现在纯靠思考模型判断 (无触发词), 测试用它精确控制路由结果。
    """

    def __init__(self, decisions: List[Dict[str, Any]]):
        self.decisions = list(decisions)
        self.calls: List[Dict[str, Any]] = []

    @property
    def model(self) -> str:
        return "router-mock"

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        d = self.decisions.pop(0) if self.decisions else {"skill": "none", "confidence": 0.0}
        return {"answer": json.dumps(d, ensure_ascii=False)}


def _route(decision: Optional[Dict[str, Any]], query: str = "随便问问", **kw):
    llm = RouterLLM([decision] if decision is not None else [])
    return rs.select_skill(llm, query, _load(), mode="llm", **kw)


# ---------------------------------------------------------------------------
# 加载 / 选择
# ---------------------------------------------------------------------------

def test_load_skills():
    skills = _load()
    assert {"literature_review", "comparison", "mechanism_analysis"} <= set(skills)
    comp = skills["comparison"]
    assert comp.name and comp.plan_system and comp.policy_system
    assert comp.when_to_use          # 自描述段非空
    assert comp.examples and comp.anti_examples
    assert comp.default_sufficiency.get("min_docs") == 2
    assert comp.synthesis_system  # synthesis.md 的 ## System 段被解析


@pytest.mark.parametrize("decision,query,expected", [
    ({"skill": "comparison", "confidence": 0.9}, "锌铝镁和纯锌镀层耐蚀性对比", "comparison"),
    ({"skill": "mechanism_analysis", "confidence": 0.85}, "耐候钢的耐蚀机理是什么", "mechanism_analysis"),
    ({"skill": "none", "confidence": 0.0}, "钢的性能怎么样", None),
])
def test_llm_selection(decision, query, expected):
    sel = _route(decision, query)
    assert sel.skill_id == expected


def test_low_confidence_falls_back_to_none():
    # 模型给出 skill 但置信度低于阈值 → 回退通用
    sel = _route({"skill": "comparison", "confidence": 0.4}, min_confidence=0.6)
    assert sel.skill_id is None


def test_no_llm_returns_none():
    sel = rs.select_skill(None, "对比一下", _load(), mode="llm")
    assert sel.skill_id is None


def test_prefer_first_paths_injected_into_plan_system():
    # skill 设了 prefer_first_paths → 组合出的 plan system 含路径偏好提示
    s = compose_plan_system("自定义拆解段。", ["progressive"])
    assert "首轮检索路径偏好" in s and "progressive" in s
    # 未设 / 非法路径 → 不注入
    assert "首轮检索路径偏好" not in compose_plan_system("自定义拆解段。", [])
    assert "首轮检索路径偏好" not in compose_plan_system("自定义拆解段。", ["bogus"])


def test_select_off_mode_returns_none():
    sel = _route({"skill": "comparison", "confidence": 0.99})
    assert sel.skill_id == "comparison"   # sanity: llm 模式能命中
    sel_off = rs.select_skill(RouterLLM([]), "对比", _load(), mode="off")
    assert sel_off.skill_id is None


def test_guards_min_docs():
    skills = _load()
    comp = skills["comparison"]
    # 构造一个带 min_docs 守卫的 skill 验证原语
    comp.guards = ["min_docs"]
    unmet = rs.evaluate_guards(comp, doc_count=1, evidence_texts=[])
    assert unmet and "文献数不足" in unmet[0]


def test_guards_per_object_evidence():
    skills = _load()
    comp = skills["comparison"]   # guards 含 per_object_evidence
    # 规划两个对称维度, 仅覆盖其一 → 提示另一未覆盖
    unmet = rs.evaluate_guards(
        comp, doc_count=2, evidence_texts=["x"],
        facet_ids=["corrosion_A", "corrosion_B"], covered=["corrosion_A"],
    )
    assert unmet and "corrosion_B" in unmet[0]
    # 两个维度都覆盖 → 无未满足
    assert rs.evaluate_guards(
        comp, doc_count=2, evidence_texts=["x"],
        facet_ids=["corrosion_A", "corrosion_B"], covered=["corrosion_A", "corrosion_B"],
    ) == []


def test_guards_causal_chain_evidence():
    skills = _load()
    mech = skills["mechanism_analysis"]   # guards 含 causal_chain_evidence
    # 纯现象描述, 无因果连接词 → 提示缺因果链
    unmet = rs.evaluate_guards(mech, doc_count=2, evidence_texts=["锈层很致密, 表面均匀"])
    assert unmet and "因果" in unmet[0]
    # 含因果语句 → 满足
    assert rs.evaluate_guards(
        mech, doc_count=2, evidence_texts=["致密锈层之所以耐蚀, 是因为它阻碍了氧扩散"],
    ) == []


# ---------------------------------------------------------------------------
# 接入研究图: skill 提示词生效 / 通用回退
# ---------------------------------------------------------------------------

def _build_with_skills(monkeypatch, *, planner, policy, retrieve, router=None, mode="llm"):
    monkeypatch.setattr(ra._lg, "_make_retrieve_node", lambda *a, **k: retrieve)
    monkeypatch.setattr(ra, "NeighborExpander", lambda *a, **k: object(), raising=False)
    import pipeline.retrieval.neighbor_expansion as ne
    monkeypatch.setattr(ne, "NeighborExpander", lambda *a, **k: object())
    return ra.build_research_agent_from_pipeline(
        _StubPipeline(),
        planner_llm=planner,
        policy_llm=policy,
        reranker_client=None,
        max_rounds=4,
        skills=_load(),
        skill_router_llm=router if router is not None else RouterLLM([]),
        skill_router_mode=mode,
    )


def test_comparison_skill_applied(monkeypatch):
    """命中 comparison: plan/policy 用 skill 专属提示词, 结果带 skill_id。"""
    planner = ScriptLLM([_plan(_B1)])
    policy = ScriptLLM([_tool_resp("research_finish", {"reason": "够了"})])
    retrieve = _make_retrieve_stub([[_hit(0, 0), _hit(1, 1)]])
    router = RouterLLM([{"skill": "comparison", "confidence": 0.9, "reason": "明确对比"}])
    agent = _build_with_skills(
        monkeypatch, planner=planner, policy=policy, retrieve=retrieve, router=router,
    )

    out = agent.run("锌铝镁和纯锌镀层耐蚀性对比")
    assert out["skill_id"] == "comparison"
    assert out["research_complete"] is True
    # 规划 system 含 comparison/plan.md 的专属措辞
    plan_system = planner.calls[0]["messages"][0]["content"]
    assert "对比分析任务" in plan_system or "被比较的对象" in plan_system
    # 仍保留前置闸门与 FC 约束 (通用框架未被破坏)
    assert "research_reject" in plan_system and "research_clarify" in plan_system
    # policy system 含 comparison/policy.md 的专属措辞
    policy_system = policy.calls[0]["messages"][0]["content"]
    assert "对称" in policy_system


def test_no_skill_falls_back_to_generic(monkeypatch):
    """未命中 skill: plan/policy 逐字使用通用提示词 (零回归)。"""
    planner = ScriptLLM([_plan(_B1)])
    policy = ScriptLLM([_tool_resp("research_finish", {"reason": "ok"})])
    retrieve = _make_retrieve_stub([[_hit(0, 0), _hit(1, 1)]])
    router = RouterLLM([{"skill": "none", "confidence": 0.0, "reason": "泛泛而问"}])
    agent = _build_with_skills(
        monkeypatch, planner=planner, policy=policy, retrieve=retrieve, router=router,
    )

    out = agent.run("钢的性能怎么样")   # 模型判 none
    assert out["skill_id"] is None
    plan_system = planner.calls[0]["messages"][0]["content"]
    assert plan_system == rsearch._PLAN_SYSTEM
    policy_system = policy.calls[0]["messages"][0]["content"]
    assert policy_system == rsearch._POLICY_SYSTEM


def test_skill_carryover_roundtrip(monkeypatch):
    """skill_id 写入 carryover, 下一轮可作为偏置恢复。"""
    planner = ScriptLLM([_plan(_B1)])
    policy = ScriptLLM([_tool_resp("research_finish", {"reason": "够了"})])
    retrieve = _make_retrieve_stub([[_hit(0, 0)]])
    router = RouterLLM([{"skill": "comparison", "confidence": 0.9, "reason": "明确对比"}])
    agent = _build_with_skills(
        monkeypatch, planner=planner, policy=policy, retrieve=retrieve, router=router,
    )
    out = agent.run("锌铝镁和纯锌镀层耐蚀性对比")
    assert out["research_carryover"]["skill_id"] == "comparison"
