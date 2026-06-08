"""专家模式 (professional) 性能观测用例 — 基于已灌入的真实数据 chunk。

这是一个**可运行脚本**(非 pytest), 打到运行中的 API /chat/stream, 端到端跑专家模式,
采集性能指标并对照"研究模式应有表现"的软阈值给出 PASS/FAIL 报告。

为什么用这个 case:
  语料里 "锌铝镁镀层 (Zn-Al-Mg)" 是覆盖最广的主题簇 (~147 篇), 且一个完整回答天然
  横跨多个维度 (成分体系 / 耐蚀机理 / 显微组织 / 与热镀锌对比 / 应用 / 工艺)。
  单轮检索无法覆盖全部维度 → 强制触发多轮递进检索 + 缺口补检 + 综述综合, 正好压测专家模式。

用法:
    .venv-api/bin/python -m pipeline.tests.observe_research_perf
    .venv-api/bin/python -m pipeline.tests.observe_research_perf "你的研究型问题"
环境变量:
    API_BASE (默认 http://localhost:8080)、API_KEY (可选)、CASE (main|ooc, 默认 main)
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

API_BASE = os.environ.get("API_BASE", "http://localhost:8080").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")

# ── 用例库 ───────────────────────────────────────────────────────────────
# main: 主性能用例 (多维度, 应多轮 + 多文献 + complete + 结构化综述)
# ooc : 越界鲁棒性用例 (语料外, 应 insufficient/no_answer, 不编造)
CASES = {
    "main": (
        "系统综述锌铝镁(Zn-Al-Mg)镀层的成分体系(Al/Mg及稀土配比)、耐蚀机理与腐蚀产物、"
        "显微组织特征，以及相比传统热镀锌的性能优势和典型应用领域。"
    ),
    "ooc": "请综述石墨烯量子点在钙钛矿太阳能电池中提升光致发光量子产率的最新机理。",
}

STRUCT_HEADERS = ["核心结论", "文献依据", "证据", "共识与分歧", "缺口"]


def _post_stream(query: str):
    """POST /chat/stream, 逐事件 yield (t_rel, event_dict)。"""
    url = f"{API_BASE}/api/v1/chat/stream"
    body = json.dumps({"query": query, "professional": True, "stream": True}).encode()
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        buf = ""
        for raw in resp:
            buf += raw.decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                try:
                    ev = json.loads(line[5:].strip())
                except Exception:
                    continue
                yield (time.time() - t0, ev)


def run_case(name: str, query: str) -> int:
    print("=" * 78)
    print(f"CASE [{name}]  professional=True")
    print(f"Q: {query}\n")

    progress, text_chunks, answer_parts = [], 0, []
    thinking_chunks, thinking_chars = 0, 0
    done = None
    err = None
    for t_rel, ev in _post_stream(query):
        et = ev.get("type")
        if et in ("progress", "thinking"):
            thinking_chunks += 1
            thinking_chars += len(ev.get("content") or ev.get("message") or "")
            # 规划/评估/收口阶段的思考会整段下发; 综述阶段是 token 流, 只采样首帧打印
            msg = ev.get("content") or ev.get("message") or ""
            if ev.get("phase") != "synthesis" or thinking_chunks % 40 == 1:
                progress.append((t_rel, msg))
                print(f"  [{t_rel:6.1f}s] 🧠 {msg[:90].replace(chr(10), ' ')}")
        elif et == "text":
            text_chunks += 1
            answer_parts.append(ev.get("content", ""))
        elif et == "done":
            done = ev
        elif et == "error":
            err = ev.get("message")
            print(f"  ‼ ERROR: {err}")

    if err or done is None:
        print(f"\nRESULT: FAIL (error={err}, done={'yes' if done else 'no'})")
        return 1

    answer = done.get("answer") or "".join(answer_parts)
    research = done.get("research") or {}
    hits = done.get("hits") or []
    distinct_docs = {h.get("doc_id") or h.get("doc_name") for h in hits if isinstance(h, dict)}
    # 研究模式按"文献名/编号"引用 (prompt 要求), 形如 [DDTL201705004] / [1020657872.nh] / [4];
    # 统计方括号内非空引用 token (排除 markdown 链接/纯空)。
    import re
    cited = {
        m.strip()
        for m in re.findall(r"\[([^\]\n]{1,60})\]", answer)
        if m.strip() and not m.strip().startswith("^")
    }
    headers_hit = [h for h in STRUCT_HEADERS if h in answer]
    total_s = done.get("latency_s", progress[-1][0] if progress else 0)

    # ── 指标 ──
    print("\n----- METRICS -----")
    print(f"  status          : {research.get('status')}")
    print(f"  rounds          : {research.get('rounds')}")
    print(f"  evidence_docs   : {research.get('evidence_docs')}")
    print(f"  evidence_chunks : {research.get('evidence_chunks')}")
    print(f"  final hits      : {len(hits)}  (distinct docs: {len(distinct_docs)})")
    print(f"  gaps            : {len(research.get('gaps', []))}  {research.get('gaps', [])}")
    print(f"  thinking chunks : {thinking_chunks}  ({thinking_chars} chars)")
    print(f"  answer chars    : {len(answer)}  (citation markers: {len(cited)})")
    print(f"  struct headers  : {headers_hit}")
    print(f"  total latency   : {total_s:.1f}s")

    # ── 软阈值检查 (研究模式应有表现) ──
    checks = []
    if name == "ooc":
        checks.append(("越界应判 insufficient/no_answer",
                       research.get("status") == "insufficient"))
    else:
        checks.append(("研究状态=complete", research.get("status") == "complete"))
        checks.append(("多轮检索 rounds>=2", (research.get("rounds") or 0) >= 2))
        checks.append(("覆盖文献 evidence_docs>=8", (research.get("evidence_docs") or 0) >= 8))
        checks.append(("证据片段 evidence_chunks>=12", (research.get("evidence_chunks") or 0) >= 12))
        checks.append(("最终命中=累计证据 hits>=12", len(hits) >= 12))
        checks.append(("综述含核心结论小节", "核心结论" in answer))
        checks.append(("固定结构>=3 小节", len(headers_hit) >= 3))
        checks.append(("答案带文献引用 (文献名/编号)", len(cited) >= 3))
        checks.append(("流式思考过程事件>=3", thinking_chunks >= 3))

    print("\n----- CHECKLIST -----")
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok = ok and passed

    print("\n----- ANSWER (head 1200) -----\n")
    print(answer[:1200])
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}\n")
    return 0 if ok else 1


def main() -> int:
    case = os.environ.get("CASE", "main")
    if len(sys.argv) > 1:
        return run_case("custom", sys.argv[1])
    if case == "all":
        rc = 0
        for n in ("main", "ooc"):
            rc |= run_case(n, CASES[n])
        return rc
    return run_case(case, CASES.get(case, CASES["main"]))


if __name__ == "__main__":
    sys.exit(main())
