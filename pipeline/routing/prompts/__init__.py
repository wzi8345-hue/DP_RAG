"""FC 版 prompt 加载器: 复用 pipeline.prompts.router_rules, 拼接到 FC system prompt。"""

from __future__ import annotations

import datetime
from pathlib import Path

from ...prompts import router_rules as _router_rules_from_file

_PROMPTS_DIR = Path(__file__).resolve().parent


def _load(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


def render_router_system_fc(current_year: int | None = None) -> str:
    """加载 router_system_fc.md, 把 __ROUTER_RULES__ 占位符替换为现有 router_rules.md 内容。"""
    year = current_year or datetime.datetime.now().year
    raw = _load("router_system_fc.md")
    rules = _router_rules_from_file(year)
    return raw.replace("__ROUTER_RULES__", rules)


def render_reflect_system_fc(current_year: int | None = None) -> str:
    """加载 reflect_system_fc.md, 同样嵌入 router_rules。"""
    year = current_year or datetime.datetime.now().year
    raw = _load("reflect_system_fc.md")
    rules = _router_rules_from_file(year)
    return raw.replace("__ROUTER_RULES__", rules)
