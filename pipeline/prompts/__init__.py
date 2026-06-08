"""Prompt 管理: 从 prompts/ 目录加载 MD 文件, 支持占位符替换。

占位符约定:
  __CURRENT_YEAR__  → 当前年份 (int)
  __ROUTER_RULES__  → router_rules.md 内容 (已替换 __CURRENT_YEAR__)

用法:
    from pipeline.prompts import load_prompt, render_router_system, render_reflect_system
"""

from __future__ import annotations

import datetime
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent


def load_prompt(filename: str) -> str:
    """加载 prompts/ 目录下的 MD 文件, 返回原始内容。"""
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


def _replace_placeholders(text: str, current_year: int) -> str:
    """替换所有已知占位符。"""
    return text.replace("__CURRENT_YEAR__", str(current_year))


def router_rules(current_year: int) -> str:
    """加载并渲染 router_rules.md。"""
    raw = load_prompt("router_rules.md")
    return _replace_placeholders(raw, current_year)


def render_router_system(current_year: int) -> str:
    """加载并渲染 router_system.md (含 router_rules 嵌入)。"""
    raw = load_prompt("router_system.md")
    rules = router_rules(current_year)
    text = raw.replace("__ROUTER_RULES__", rules)
    return _replace_placeholders(text, current_year)


def render_reflect_system(current_year: int) -> str:
    """加载并渲染 reflect_system.md (含 router_rules 嵌入)。"""
    raw = load_prompt("reflect_system.md")
    rules = router_rules(current_year)
    text = raw.replace("__ROUTER_RULES__", rules)
    return _replace_placeholders(text, current_year)


def generation_system_prompt() -> str:
    """加载 generation system prompt (无占位符)。"""
    return load_prompt("system_prompt.md")
