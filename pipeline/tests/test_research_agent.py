"""专业研究模式 (ResearchAgent) 端到端测试。

用 mock LLM (脚本化 tool call) + mock 检索节点驱动真实编译图, 验证:
  - 规划 → 多轮检索 → 综述 的完整闭环;
  - continue/finish/clarify 三种 policy 分支;
  - 证据跨轮去重累积、轮间 route_results 清空;
  - 轮次预算 / stall 熔断;
  - 规划失败的启发式兜底。

不依赖真实 LLM / Milvus / reranker。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

pytest.importorskip("langgraph")

from pipeline.retrieval import research_agent as ra
from pipeline.retrieval.retrievers import Hit


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

def _tool_resp(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """模拟 LLMClient.chat_with_tools 的返回 (OpenAI tool_calls 形态)。"""
    return {
        "tool_calls": [{
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        }],
        "raw": {}, "usage": {},
    }


class ScriptLLM:
    """按脚本依次返回 tool call 的 mock LLM。"""

    def __init__(self, script: List[Dict[str, Any]]):
        self.script = list(script)
        self.calls: List[Dict[str, Any]] = []

    @property
    def model(self) -> str:
        return "mock"

    def chat_with_tools(self, **kwargs):
        self.calls.append(kwargs)
        if not self.script:
            # 脚本耗尽: 安全收口
            return _tool_resp("research_finish", {"reason": "script exhausted"})
        return self.script.pop(0)


def _hit(i: int, doc: int) -> Hit:
    return Hit(
        pk=f"pk{i}", chunk_id=f"c{i}", doc_id=f"d{doc}", doc_name=f"Doc{doc}",
        type="text", page_start=1, content=f"证据片段{i} 关于腐蚀机理与速率",
        score=0.8, rerank_score=0.7,
    )


def _make_retrieve_stub(hits_per_round: List[List[Hit]]):
    """模拟 retrieve_node: 每次调用按顺序吐一批 hits 写进 route_results。"""
    state_counter = {"i": 0}

    def retrieve_node(state):
        idx = state_counter["i"]
        state_counter["i"] += 1
        hits = hits_per_round[idx] if idx < len(hits_per_round) else []
        # 模拟真实 retrieve: 合并进 (本测试里上游已清空) route_results
        rr = dict(state.get("route_results") or {})
        rr["progressive"] = list(rr.get("progressive", [])) + list(hits)
        state["route_results"] = rr
        # 累积 this_round_docs (模拟真实节点行为)
        docs = list(state.get("this_round_docs") or [])
        seen = {d["doc_id"] for d in docs}
        for h in hits:
            if h.doc_id not in seen:
                seen.add(h.doc_id)
                docs.append({"doc_id": h.doc_id, "doc_name": h.doc_name})
        state["this_round_docs"] = docs
        state["agent_phase"] = "retrieve"
        return state

    return retrieve_node


class _StubRouter:
    # 不提供 _validate_decision → build_research_graph 拿到 validate_fn=None,
    # batches→decision 走 _minimal_route_decision (无需真实 router)。
    pass


class _StubVec:
    pass


class _StubRetr:
    vec = _StubVec()


class _StubMeta:
    client = object()
    collection = object()


class _StubPipeline:
    router = _StubRouter()
    summary_r = _StubRetr()
    local_r = _StubRetr()
    metadata_r = _StubMeta()


def _build(monkeypatch, *, planner_llm, policy_llm, retrieve_stub, max_rounds=4):
    # mock retrieve_node 工厂
    monkeypatch.setattr(ra._lg, "_make_retrieve_node", lambda *a, **k: retrieve_stub)
    # NeighborExpander 在 build 里会被实例化, 用一个无害 stub 替换
    monkeypatch.setattr(
        ra, "NeighborExpander", lambda *a, **k: object(), raising=False,
    )
    # 避免真正 import neighbor_expansion 里的实现: build_research_graph 内部
    # `from .neighbor_expansion import NeighborExpander` 会重新绑定, 故 patch 该符号
    import pipeline.retrieval.neighbor_expansion as ne
    monkeypatch.setattr(ne, "NeighborExpander", lambda *a, **k: object())

    return ra.build_research_agent_from_pipeline(
        _StubPipeline(),
        planner_llm=planner_llm,
        policy_llm=policy_llm,
        reranker_client=None,
        max_rounds=max_rounds,
    )


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

def _plan(batches, facets=None):
    return _tool_resp("research_plan", {
        "goal": "耐蚀钢在海洋环境的研究综述",
        "facets": facets or [{"id": "mech", "question": "机理", "keywords": ["腐蚀机理"]}],
        "initial_batches": batches,
    })


_B1 = [{"id": "b1", "purpose": "定位文献", "paths": [{"t": "summary", "kw": ["耐蚀钢 海洋腐蚀"]}]}]
_B2 = [{"id": "b2", "purpose": "补定量", "paths": [{"t": "progressive", "kw": ["腐蚀速率 EIS"]}]}]


def test_plan_then_finish(monkeypatch):
    """规划 → 1 轮检索 → policy 直接 finish。"""
    planner = ScriptLLM([_plan(_B1)])
    policy = ScriptLLM([_tool_resp("research_finish", {"covered": ["mech"], "reason": "够了"})])
    retrieve = _make_retrieve_stub([[_hit(0, 0), _hit(1, 0), _hit(2, 1)]])
    agent = _build(monkeypatch, planner_llm=planner, policy_llm=policy, retrieve_stub=retrieve)

    out = agent.run("耐蚀钢综述")
    assert out["research_complete"] is True
    assert out["research_status"] == "complete"
    assert out["needs_clarify"] is False
    assert out["evidence_chunk_count"] == 3
    assert out["evidence_doc_count"] == 2
    # 最终 hits = 跨轮累计去重证据, 而非最后一轮 route_results
    assert len(out["evidence_hits"]) == 3
    assert "研究目标" in out["context"]
    assert "Doc0" in out["context"] and "Doc1" in out["context"]
    assert len(policy.calls) == 1


def test_continue_then_finish(monkeypatch):
    """规划 → 检索 → continue → 再检索 → finish; 证据跨轮累积去重。"""
    planner = ScriptLLM([_plan(_B1)])
    policy = ScriptLLM([
        _tool_resp("research_continue", {"gaps": ["缺定量"], "next_batches": _B2}),
        _tool_resp("research_finish", {"covered": ["mech"], "reason": "齐了"}),
    ])
    # 第 1 轮: pk0..2 (Doc0/Doc1); 第 2 轮: pk2(重复) + pk3 (Doc1) → 只新增 pk3
    retrieve = _make_retrieve_stub([
        [_hit(0, 0), _hit(1, 0), _hit(2, 1)],
        [_hit(2, 1), _hit(3, 1)],
    ])
    agent = _build(monkeypatch, planner_llm=planner, policy_llm=policy, retrieve_stub=retrieve)

    out = agent.run("耐蚀钢综述")
    assert out["research_rounds"] == 2
    assert out["evidence_chunk_count"] == 4   # pk0,1,2,3 去重后
    assert out["research_complete"] is True
    # policy 第二轮应能看到"已检索过"的签名
    second_obs = policy.calls[1]["messages"][1]["content"]
    assert "已检索过" in second_obs


def test_max_rounds_forced_finish(monkeypatch):
    """policy 一直 continue, 应在 max_rounds 处强制收口, 不无限循环。"""
    planner = ScriptLLM([_plan(_B1)])
    # 永远 continue
    policy = ScriptLLM([
        _tool_resp("research_continue", {"gaps": ["g"], "next_batches": _B2})
        for _ in range(10)
    ])
    retrieve = _make_retrieve_stub([[_hit(i, i)] for i in range(10)])
    agent = _build(monkeypatch, planner_llm=planner, policy_llm=policy, retrieve_stub=retrieve, max_rounds=3)

    out = agent.run("耐蚀钢综述")
    assert out["research_complete"] is True
    assert out["research_rounds"] <= 3
    # policy 最多被调用 max_rounds-1 次 (最后一轮强制 finish 不调 LLM)
    assert len(policy.calls) <= 2


def test_stall_finish_when_evidence_enough(monkeypatch):
    """连续无新增证据但已有足够文献(≥3篇) → stall 收口 finish。"""
    planner = ScriptLLM([_plan(_B1)])
    policy = ScriptLLM([
        _tool_resp("research_continue", {"gaps": ["g"], "next_batches": _B2})
        for _ in range(10)
    ])
    # 首轮拿到 3 篇文献, 之后每轮重复 pk0 → 0 新增
    rounds = [[_hit(0, 0), _hit(1, 1), _hit(2, 2)]] + [[_hit(0, 0)] for _ in range(9)]
    retrieve = _make_retrieve_stub(rounds)
    agent = _build(monkeypatch, planner_llm=planner, policy_llm=policy, retrieve_stub=retrieve, max_rounds=9)

    out = agent.run("耐蚀钢综述")
    # 证据充足(3篇) + stall → finish (而非反问)
    assert out["research_complete"] is True
    assert out["research_status"] == "complete"
    assert out["research_rounds"] <= 4


def test_stall_clarify_when_evidence_thin(monkeypatch):
    """连续无新增证据且证据不足(<3篇) → 触发 clarify 让用户补充。"""
    planner = ScriptLLM([_plan(_B1)])
    # 每轮换不同关键词 (避免触发"重复批次收口"), 但检索始终命中同一 dupe
    distinct = [
        [{"id": f"b{n}", "purpose": "补证据",
          "paths": [{"t": "progressive", "kw": [f"角度{n} 关键词"]}]}]
        for n in range(10)
    ]
    policy = ScriptLLM([
        _tool_resp("research_continue", {"gaps": ["腐蚀速率定量"], "next_batches": b})
        for b in distinct
    ])
    # 始终只有 1 篇文献且无新增 → 停滞 + 证据薄
    retrieve = _make_retrieve_stub([[_hit(0, 0)] for _ in range(10)])
    agent = _build(monkeypatch, planner_llm=planner, policy_llm=policy, retrieve_stub=retrieve, max_rounds=9)

    out = agent.run("钢的研究")
    assert out["needs_clarify"] is True
    assert out["research_status"] == "clarify"
    assert out["answer"]


def test_clarify(monkeypatch):
    """policy 判定需澄清时, needs_clarify=True 且带反问文本。"""
    planner = ScriptLLM([_plan(_B1)])
    policy = ScriptLLM([
        _tool_resp("research_clarify", {"q": "请缩小到具体钢种或环境?", "opts": ["海洋", "酸性"]}),
    ])
    retrieve = _make_retrieve_stub([[_hit(0, 0)]])
    agent = _build(monkeypatch, planner_llm=planner, policy_llm=policy, retrieve_stub=retrieve)

    out = agent.run("钢的研究")
    assert out["needs_clarify"] is True
    assert out["research_status"] == "clarify"
    assert "缩小" in out["answer"]
    assert out["clarify_request"]["opts"] == ["海洋", "酸性"]


def test_plan_fallback(monkeypatch):
    """规划 LLM 不产 tool call 时, 应用启发式单批次兜底而非崩溃。"""
    planner = ScriptLLM([{"tool_calls": [], "raw": {}, "usage": {}}])
    policy = ScriptLLM([_tool_resp("research_finish", {"reason": "ok"})])
    retrieve = _make_retrieve_stub([[_hit(0, 0), _hit(1, 1)]])
    agent = _build(monkeypatch, planner_llm=planner, policy_llm=policy, retrieve_stub=retrieve)

    out = agent.run("随便问问")
    assert out["research_complete"] is True
    assert out["evidence_chunk_count"] == 2


def test_no_evidence_triggers_clarify(monkeypatch):
    """多轮检索都查不到证据时, 转为反问让用户补充 (而非静默 no_answer)。"""
    planner = ScriptLLM([_plan(_B1)])
    policy = ScriptLLM([
        _tool_resp("research_continue", {"gaps": ["g"], "next_batches": _B2}),
        _tool_resp("research_continue", {"gaps": ["g"], "next_batches": _B1}),
    ])
    retrieve = _make_retrieve_stub([[], []])   # 始终零命中
    agent = _build(monkeypatch, planner_llm=planner, policy_llm=policy, retrieve_stub=retrieve, max_rounds=4)

    out = agent.run("不存在的主题")
    assert out["evidence_chunk_count"] == 0
    # 第 2 轮仍零证据 → clarify 反问 (而非 insufficient 静默)
    assert out["needs_clarify"] is True
    assert out["research_status"] == "clarify"


def test_plan_reject_direct_answer(monkeypatch):
    """规划阶段判定无关/闲聊 → 兜底直答, 不检索。"""
    planner = ScriptLLM([_tool_resp("research_reject", {
        "kind": "out_of_scope", "reply": "这个问题超出了本文献库的范围。",
    })])
    policy = ScriptLLM([])
    retrieve = _make_retrieve_stub([[_hit(0, 0)]])  # 不应被调用
    agent = _build(monkeypatch, planner_llm=planner, policy_llm=policy, retrieve_stub=retrieve)

    out = agent.run("今天天气怎么样")
    assert out["research_status"] == "reject"
    assert out["needs_clarify"] is False
    assert "范围" in out["answer"]
    assert out["evidence_chunk_count"] == 0


def test_plan_clarify_when_vague(monkeypatch):
    """规划阶段判定问题模糊 → 前置追问, 不检索。"""
    planner = ScriptLLM([_tool_resp("research_clarify", {
        "q": "你想研究哪种材料的哪方面性能?", "opts": ["耐蚀性", "力学性能"],
    })])
    policy = ScriptLLM([])
    retrieve = _make_retrieve_stub([[_hit(0, 0)]])  # 不应被调用
    agent = _build(monkeypatch, planner_llm=planner, policy_llm=policy, retrieve_stub=retrieve)

    out = agent.run("材料")
    assert out["needs_clarify"] is True
    assert out["research_status"] == "clarify"
    assert out["clarify_request"]["opts"] == ["耐蚀性", "力学性能"]


def test_continue_all_repeated_forces_finish(monkeypatch):
    """policy 给出的 next_batches 与已检索签名完全重复 → 强制收口避免空耗。"""
    planner = ScriptLLM([_plan(_B1)])
    # 第二轮 continue 复用与首轮完全相同的批次 (_B1), 触发规则去重收口
    policy = ScriptLLM([
        _tool_resp("research_continue", {"gaps": ["g"], "next_batches": _B1}),
        # 不应再被调用 (上面 continue 后规则去重直接 finish)
        _tool_resp("research_finish", {"reason": "should-not-reach"}),
    ])
    retrieve = _make_retrieve_stub([[_hit(0, 0)], [_hit(1, 1)]])
    agent = _build(monkeypatch, planner_llm=planner, policy_llm=policy, retrieve_stub=retrieve)

    out = agent.run("耐蚀钢综述")
    assert out["research_complete"] is True
    assert out["research_status"] == "complete"
    # 只调用了一次 policy LLM, 其 continue 批次与首轮重复 → 当轮即收口
    assert len(policy.calls) == 1
    assert out["research_rounds"] == 1


def test_run_events_thinking(monkeypatch):
    """run_events 先产出 thinking 思考过程事件, 最后产出 result。"""
    planner = ScriptLLM([_plan(_B1)])
    policy = ScriptLLM([
        _tool_resp("research_continue", {"gaps": ["缺定量"], "next_batches": _B2}),
        _tool_resp("research_finish", {"reason": "齐了"}),
    ])
    retrieve = _make_retrieve_stub([[_hit(0, 0)], [_hit(1, 1)]])
    agent = _build(monkeypatch, planner_llm=planner, policy_llm=policy, retrieve_stub=retrieve)

    kinds = []
    phases = []
    result = None
    for kind, payload in agent.run_events("耐蚀钢综述"):
        kinds.append(kind)
        if kind == "thinking":
            assert "content" in payload and payload["content"]
            assert "round" in payload and "phase" in payload
            phases.append(payload["phase"])
        elif kind == "result":
            result = payload
    assert kinds[-1] == "result"
    # 至少: 规划思考 (plan) + 一轮 continue 评估 + finish 收口
    assert kinds.count("thinking") >= 3
    assert "plan" in phases and "policy" in phases and "finish" in phases
    assert result is not None
    assert result["research_complete"] is True
    assert result["evidence_chunk_count"] == 2
