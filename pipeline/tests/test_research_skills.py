"""文件定义式 skill 的加载、选择, 及其接入研究图后的提示词生效 / 通用回退测试。

复用 test_research_agent 的 mock 框架, 不依赖真实 LLM / Milvus / reranker。
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("langgraph")

from pipeline.retrieval import research_agent as ra
from pipeline.routing import research as rsearch
from pipeline.routing import research_skills as rs

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


# ---------------------------------------------------------------------------
# 加载 / 选择
# ---------------------------------------------------------------------------

def test_load_skills():
    skills = _load()
    assert {"literature_review", "comparison", "mechanism_analysis"} <= set(skills)
    comp = skills["comparison"]
    assert comp.name and comp.plan_system and comp.policy_system
    assert comp.triggers          # 触发词非空
    assert comp.default_sufficiency.get("min_docs") == 2
    assert comp.synthesis_system  # synthesis.md 的 ## System 段被解析


@pytest.mark.parametrize("query,expected", [
    ("锌铝镁和纯锌镀层耐蚀性对比", "comparison"),
    ("耐候钢的耐蚀机理是什么", "mechanism_analysis"),
    ("综述一下锌铝镁的研究现状", "literature_review"),
    ("钢的性能怎么样", None),       # 无触发词 → 回退通用
])
def test_heuristic_selection(query, expected):
    sid, _ = rs.select_skill(None, query, _load(), mode="heuristic")
    assert sid == expected


def test_select_off_mode_returns_none():
    sid, _ = rs.select_skill(None, "对比", _load(), mode="off")
    assert sid is None


def test_guards_min_docs():
    skills = _load()
    comp = skills["comparison"]
    # comparison 的 guards 不含 min_docs (用 per_object_evidence), 故无未满足项
    assert rs.evaluate_guards(comp, doc_count=0, evidence_texts=[]) == []
    # 构造一个带 min_docs 守卫的 skill 验证原语
    comp.guards = ["min_docs"]
    unmet = rs.evaluate_guards(comp, doc_count=1, evidence_texts=[])
    assert unmet and "文献数不足" in unmet[0]


# ---------------------------------------------------------------------------
# 接入研究图: skill 提示词生效 / 通用回退
# ---------------------------------------------------------------------------

def _build_with_skills(monkeypatch, *, planner, policy, retrieve, mode="heuristic"):
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
        skill_router_llm=planner,       # heuristic 模式下不会真的调用 chat
        skill_router_mode=mode,
    )


def test_comparison_skill_applied(monkeypatch):
    """命中 comparison: plan/policy 用 skill 专属提示词, 结果带 skill_id。"""
    planner = ScriptLLM([_plan(_B1)])
    policy = ScriptLLM([_tool_resp("research_finish", {"reason": "够了"})])
    retrieve = _make_retrieve_stub([[_hit(0, 0), _hit(1, 1)]])
    agent = _build_with_skills(monkeypatch, planner=planner, policy=policy, retrieve=retrieve)

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
    agent = _build_with_skills(monkeypatch, planner=planner, policy=policy, retrieve=retrieve)

    out = agent.run("钢的性能怎么样")   # 无触发词
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
    agent = _build_with_skills(monkeypatch, planner=planner, policy=policy, retrieve=retrieve)
    out = agent.run("锌铝镁和纯锌镀层耐蚀性对比")
    assert out["research_carryover"]["skill_id"] == "comparison"
