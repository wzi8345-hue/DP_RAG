"""技能路由评测: 用各技能目录下的 eval_cases.yaml 跑一遍 select_skill, 输出命中率与混淆矩阵。

依赖真实分类模型 (复用 generation 配置), 因此不进 CI 单测, 作为人工/回归校准工具:

    python -m pipeline.skills.eval_router               # 用默认配置
    python -m pipeline.skills.eval_router --config local_api_config.yaml

判定口径与线上一致: mode=llm, 仅置信度 ≥ min_confidence 才算命中, 否则 none。
should_hit 期望命中本技能; should_miss 期望【不】命中本技能 (none 或别的技能均可)。
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import Counter
from typing import Dict, List

import yaml

from ..clients.client_registry import get_global_registry
from ..config import load_config
from ..routing.research_skills import (
    DEFAULT_MIN_CONFIDENCE,
    load_skills,
    resolve_skills_config,
    select_skill,
)

_SKILLS_DIR = os.path.dirname(__file__)


def _load_eval_cases(skills_dir: str) -> List[dict]:
    cases = []
    for path in sorted(glob.glob(os.path.join(skills_dir, "*", "eval_cases.yaml"))):
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if data.get("skill"):
            cases.append(data)
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description="技能路由评测")
    ap.add_argument("--config", default=None, help="配置文件路径 (默认走项目默认配置)")
    ap.add_argument("--skills-dir", default=_SKILLS_DIR)
    args = ap.parse_args()

    cfg = load_config(args.config)
    gen = cfg.generation
    prof = (cfg.retrieval.get("langgraph", {}) or {}).get("professional", {}) or {}
    sk = resolve_skills_config(prof.get("skills", {}) or {})
    min_conf = sk.get("router_min_confidence", DEFAULT_MIN_CONFIDENCE)

    skills = load_skills([args.skills_dir])
    if not skills:
        print("未加载到任何技能, 退出。")
        return 1
    print(f"已加载技能: {', '.join(skills)}  (min_confidence={min_conf})\n")

    llm = get_global_registry().get_llm(
        api_base=gen.get("api_base", ""),
        model=gen.get("model", ""),
        api_key=gen.get("api_key", ""),
        timeout=gen.get("timeout", 120),
        max_retries=gen.get("max_retries", 2),
        disable_thinking_extra_body=bool(gen.get("disable_thinking_extra_body", False)),
    )

    eval_cases = _load_eval_cases(args.skills_dir)
    # 混淆矩阵: 行=期望(skill 或 NONE), 列=实际(skill 或 NONE)
    confusion: Counter = Counter()
    n_pass = n_total = 0
    failures: List[str] = []

    def _run(query: str) -> str:
        sel = select_skill(
            llm, query, skills, mode="llm",
            router_max_tokens=sk.get("router_max_tokens", 512),
            disable_thinking=sk.get("router_disable_thinking", False),
            min_confidence=min_conf,
        )
        return sel.skill_id or "NONE"

    for data in eval_cases:
        target = data["skill"]
        for q in data.get("should_hit", []):
            pred = _run(q)
            confusion[(target, pred)] += 1
            ok = pred == target
            n_total += 1
            n_pass += int(ok)
            print(f"[{'✓' if ok else '✗'}] hit  期望={target:<20} 实际={pred:<20} | {q}")
            if not ok:
                failures.append(f"  should_hit {target}: «{q}» → {pred}")
        for q in data.get("should_miss", []):
            pred = _run(q)
            confusion[("NONE→" + target, pred)] += 1
            ok = pred != target
            n_total += 1
            n_pass += int(ok)
            print(f"[{'✓' if ok else '✗'}] miss 期望≠{target:<19} 实际={pred:<20} | {q}")
            if not ok:
                failures.append(f"  should_miss {target}: «{q}» → 误命中 {pred}")

    print(f"\n通过 {n_pass}/{n_total}  ({(n_pass / n_total * 100) if n_total else 0:.0f}%)")
    if failures:
        print("\n失败用例:")
        print("\n".join(failures))
    return 0 if n_pass == n_total else 2


if __name__ == "__main__":
    raise SystemExit(main())
